#!/usr/bin/env python3

# Copyright 2020 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hjson
import pathlib
import sys
import re
import logging
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from subprocess import run
import csv

from jsonref import JsonRef

from mako.template import Template

import occamy
# from occamy import check_occamy_cfg, get_cluster_generators, generate_wrappers, generate_memories, get_cluster_cfg_list, generate_snitch

sys.path.append(str(pathlib.Path(__file__).parent / '../'))
from solder import solder, device_tree, util  # noqa: E402

# Compile a regex to trim trailing whitespaces on lines.
re_trailws = re.compile(r'[ \t\r]+$', re.MULTILINE)

# Default name for all generated sources
DEFAULT_NAME = "occamy"


def write_template(tpl_path, outdir, fname=None, **kwargs):
    if tpl_path:
        tpl_path = pathlib.Path(tpl_path).absolute()
        if tpl_path.exists():
            tpl = Template(filename=str(tpl_path))
            fname = tpl_path.with_suffix("").name.replace("occamy", kwargs["name"]) \
                if not fname else fname
            with open(outdir / fname, "w") as file:
                code = tpl.render_unicode(**kwargs)
                code = re_trailws.sub("", code)
                file.write(code)
        else:
            print(f'Could not find file {tpl_path}')
            raise FileNotFoundError
    else:
        print("No template file provided, skipping template generation.")

def read_json_file(file):
    try:
        srcfull = file.read()
        obj = hjson.loads(srcfull, use_decimal=True)
        obj = JsonRef.replace_refs(obj)
    except ValueError:
        raise SystemExit(sys.exc_info()[1])
    return obj

def main():
    """Generate the Occamy system and all corresponding configuration files."""
    parser = argparse.ArgumentParser(prog="clustergen")
    parser.add_argument("--cfg",
                        "-c",
                        metavar="file",
                        type=argparse.FileType('r'),
                        required=True,
                        help="A cluster configuration file")
    parser.add_argument("--outdir",
                        "-o",
                        type=pathlib.Path,
                        required=True,
                        help="Target directory.")
    # Parse arguments.
    parser.add_argument("--top-sv",
                        metavar="TOP_SV",
                        help="Name of top-level file (output).")
    parser.add_argument("--soc-sv",
                        metavar="TOP_SYNC_SV",
                        help="Name of synchronous SoC file (output).")
    parser.add_argument("--pkg-sv",
                        metavar="PKG_SV",
                        help="Name of top-level package file (output)")
    parser.add_argument("--quad",
                        metavar="QUAD",
                        help="Name of quadrant template file (output)")
    parser.add_argument("--quad-ctrl",
                        metavar="QUAD_CTRL",
                        help="Name of quadrant controller template file (output)")
    parser.add_argument("--quad-noc",
                        metavar="QUAD_NOC",
                        help="Name of quadrant NoC template file (output)")
    parser.add_argument("--xilinx-sv",
                        metavar="XILINX_SV",
                        help="Name of the Xilinx wrapper file (output).")
    parser.add_argument("--testharness-sv",
                        metavar="TESTHARNESS_SV",
                        help="Name of the testharness wrapper file (output).")
    parser.add_argument("--mem-macro-testharness-sv",
                        metavar="MEM_MACRO_TESTHARNESS_SV",
                        help="Name of the memory macro testharness wrapper file (output).")
    parser.add_argument("--netlist-testharness-sv",
                        metavar="NETLIST_TESTHARNESS_SV",
                        help="Name of the netlist testharness wrapper file (output).")

    parser.add_argument("--cva6-sv",
                        metavar="CVA6_SV",
                        help="Name of the CVA6 wrapper file (output).")
    parser.add_argument("--snitch",
                        metavar="SNITCH",
                        help="Define this to generate Snitch Cluster.")
    parser.add_argument("--bootdata",
                        metavar="BOOTDATA",
                        help="Name of the bootdata file (output)")
    parser.add_argument("--cheader",
                        metavar="CHEADER",
                        help="Name of the cheader file (output)")
    parser.add_argument("--csv",
                        metavar="CSV",
                        help="Name of the csv file (output)")
    parser.add_argument("--xdma",
                        action="store_true",
                        help="Enable XDMA inside HeMAiA Mem System")
    parser.add_argument("--chip",
                        metavar="CHIP_TOP",
                        help="(Optional) Chip Top-level")
    parser.add_argument("--io",
                        metavar="CHIP_IO",
                        help="(Optional) Chip I/O Cell Wrapper")
    parser.add_argument("--ctrl",
                        metavar="SoC and Quad Ctrl",
                        help="Name of SoC or Quadrant template file (output)")
    parser.add_argument("--graph", "-g", metavar="DOT")
    parser.add_argument("--memories", "-m", action="store_true")
    parser.add_argument("--wrapper", "-w", action="store_true")
    parser.add_argument("--am-cheader", "-D", metavar="ADDRMAP_CHEADER")
    parser.add_argument("--am-csv", "-aml", metavar="ADDRMAP_CSV")
    parser.add_argument("--name", metavar="NAME",
                        default=DEFAULT_NAME, help="System's name.")

    parser.add_argument("-v",
                        "--verbose",
                        help="increase output verbosity",
                        action="store_true")
    parser.add_argument("--gen-host-ld",
                        action="store_true",
                        help="Generate host.ld file for host build")

    args = parser.parse_args()
    occamy_root = pathlib.Path(__file__).parent / "../../"

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Read HJSON description of System.
    with args.cfg as file:
        occamy_cfg = read_json_file(file)

    # If name argument provided, change config
    if args.name != DEFAULT_NAME:
        occamy_cfg["cluster"]["name"] = args.name+"_cluster"
        # occamy_cfg["clster"]["name"] = args.name

    # Check the outdir
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    if not args.outdir.is_dir():
        exit("Out directory is not a valid path.")

    # We first check the cfg by schema
    occamy.check_occamy_cfg(occamy_cfg)
    # Now we have a valid cfg

    # In occamy cfg we specify
    # clusters:[
    #     "snax_streamer_gemm",
    #     "snax_wide_gemm_data_reshuffler",
    # ],
    #
    # The name in the ["clusters"] corresponds to the file names in occamy/target/rtl/cfg/cluster_cfg
    # And each cluster is stores in cluster generator

    cluster_cfg_dir = occamy_root / "deps/snitch_cluster/target/snitch_cluster/cfg"
    cluster_generators = occamy.get_cluster_generators(occamy_cfg, cluster_cfg_dir)

    # Check and fix the xbar id/width
    occamy.check_and_fix_occamy_xbar_id_width(occamy_cfg, cluster_generators)
    # Each cluster will be generated seperately
    # The generated file's name is specified in the ["name"] field of each cluster's cfg file
    # e.g
    # cluster: {
    #     name: "snax_streamer_gemm"

    # As all the source is able to be generated inside snax cluster, Occamy does not need to handle the wrapper gen any more.
    cluster_cfg_list = occamy.get_cluster_cfg_list(occamy_cfg, cluster_cfg_dir)
    if args.snitch:
        print(cluster_cfg_list)
        occamy.generate_snitch(cluster_cfg_list, args.snitch)

    if args.wrapper:
        occamy.generate_wrappers(cluster_generators, outdir)

    if args.memories:
        occamy.generate_memories(cluster_generators, outdir)

    # Arguments.
    nr_s1_quadrants = occamy_cfg["nr_s1_quadrant"]
    nr_s1_clusters = len(occamy_cfg["clusters"])

    core_per_cluster_list = [cluster_generator.cfg["nr_cores"]
                             for cluster_generator in cluster_generators]
    #####################################
    # Now we create all the am for xbars
    #####################################
    # Each XBAR is a node
    # Create the address map
    am = solder.AddrMap()

    # Toplevel crossbar address map
    am_soc_narrow_xbar = am.new_node("soc_narrow_xbar")
    am_soc_wide_xbar = am.new_node("soc_wide_xbar")

    # Toplevel Peripheral crossbar address map
    am_soc_axi_lite_periph_xbar = am.new_node("soc_axi_lite_periph_xbar")
    am_soc_axi_lite_narrow_periph_xbar = am.new_node("soc_axi_lite_narrow_periph_xbar")

    # Quadrant crossbar address map
    am_quad_wide_xbar = am.new_node("quad_wide_xbar")
    am_quad_narrow_xbar = am.new_node("quad_narrow_xbar")
    
    # We do not have multiple quadrants anymore
    # am_wide_xbar_quadrant_s1 = list()
    # am_narrow_xbar_quadrant_s1 = list()
    # for i in range(nr_s1_quadrants):
    #     am_wide_xbar_quadrant_s1.append(
    #         am.new_node("wide_xbar_quadrant_s1_{}".format(i)))
    #     am_narrow_xbar_quadrant_s1.append(
    #         am.new_node("narrow_xbar_quadrant_s1_{}".format(i)))

    # Quad Ctrl crossbar address map
    am_soc_to_quad_axi_xbar = am.new_node("soc_to_quad_axi_xbar")
    am_quad_to_soc_axi_xbar = am.new_node("quad_to_soc_axi_xbar")

    # Quad Ctrl peripheral crossbar address map
    am_quad_axi_lite_xbar = am.new_node("quad_axi_lite_xbar")
    am_quad_axi_lite_narrow_xbar = am.new_node("quad_axi_lite_narrow_xbar")
    ##################################################################
    # After that, we need to create the leaves that connect to xbars
    ##################################################################
    # The leaves are referred to the endpoints in the system
    # Like peripherals, memories, etc
    # Hence we are creating the master ports that connect to the xbars
    # 64bit SoC AXI Lite Peripherals
    am_soc_axi_lite_peripherals = occamy.am_connect_soc_lite_periph_xbar(
        am, am_soc_axi_lite_periph_xbar, occamy_cfg)
    # 32bit SoC AXI Lite Narrow Peripherals
    am_soc_axi_lite_narrow_peripherals, am_bootrom, am_clint = occamy.am_connect_soc_lite_narrow_periph_xbar(
        am, am_soc_axi_lite_narrow_periph_xbar, occamy_cfg)

    # 64bit Quad AXI Lite Peripherals
    am_quad_axi_lite_peripherals, addrs_quad_axi_lite_peripherals = occamy.am_connect_quad_axi_lite_xbar(am, am_quad_axi_lite_xbar, occamy_cfg)
    # 32bit Quad AXI Lite Narrow Peripherals
    am_quad_axi_lite_narrow_peripheral, am_quad_axi_lite_narrow_h2c_mailboxs, addrs_quad_axi_lite_narrow_peripherals = occamy.am_connect_quad_axi_lite_narrow_xbar(am, am_quad_axi_lite_narrow_xbar, occamy_cfg)

    # SoC Narrow Xbar
    am_spm_narrow, am_sys_idma_cfg, am_narrow_hemaia_xdma_ctrl_io, am_narrow_hemaia_xdma_cfg_io = occamy.am_connect_soc_narrow_xbar_mem(
        am, am_soc_narrow_xbar, occamy_cfg)
    # SoC Wide Xbar
    am_wide_hemaia_mem, am_wide_hemaia_xdma_data_io, am_wide_zero_mem = occamy.am_connect_soc_wide_xbar_mem(
        am, am_soc_wide_xbar, occamy_cfg)
    # Quad Wide and Narrow Xbar
    am_clusters, am_clusters_periph, am_clusters_leftover_spaces, addrs_clusters, addrs_clusters_periph, addrs_clusters_leftover_spaces = occamy.am_connect_quad_wide_and_narrow_xbar(
        am, am_quad_wide_xbar, am_quad_narrow_xbar, cluster_generators)
    ####################################################################
    # Then we connect between xbars
    ####################################################################
    # Attach means a master port from one to antoher
    # e.g
    # xbar1.attach(xbar2)
    # means xbar1 has a master port connected to xbar2's slave port
    # This step is to create the slave ports in each xbar
    ####################
    # SoC narrow Xbar  #
    ####################
    am_soc_narrow_xbar.attach(am_soc_to_quad_axi_xbar)
    am_soc_narrow_xbar.attach(am_soc_axi_lite_periph_xbar)
    am_soc_narrow_xbar.attach(am_soc_axi_lite_narrow_periph_xbar)
    am_soc_narrow_xbar.attach(am_soc_wide_xbar)
    
    #####################
    # SoC wide Xbar     #
    #####################
    am_soc_wide_xbar.attach(am_quad_wide_xbar)
    am_soc_wide_xbar.attach(am_soc_narrow_xbar)
    
    ###################################
    # 64 bit SoC AXI Lite Periph Xbar #
    ###################################
    # Mainly for spi and jtag
    am_soc_axi_lite_periph_xbar.attach(am_soc_narrow_xbar)
    
    ########################
    # SoC to Quad AXI Xbar #
    ########################
    am_soc_to_quad_axi_xbar.attach(am_quad_narrow_xbar)
    am_soc_to_quad_axi_xbar.attach(am_quad_axi_lite_xbar)
    am_soc_to_quad_axi_xbar.attach(am_quad_axi_lite_narrow_xbar)
    
    ########################
    # Quad to SoC AXI Xbar #
    ########################    
    am_quad_to_soc_axi_xbar.attach(am_soc_narrow_xbar)
    am_quad_to_soc_axi_xbar.attach(am_quad_axi_lite_xbar)
    am_quad_to_soc_axi_xbar.attach(am_quad_axi_lite_narrow_xbar)
    
    #############################
    # 64 bit Quad AXI Lite Xbar #
    #############################
    # For the Bingo HW Schedulers
    am_quad_axi_lite_xbar.attach(am_soc_to_quad_axi_xbar)
    
    ########################
    # Quad Narrow AXI XBar #
    ########################
    am_quad_narrow_xbar.attach(am_quad_to_soc_axi_xbar)
    ######################
    # Quad Wide AXI XBar #
    ######################
    am_quad_wide_xbar.attach(am_soc_wide_xbar)
    ####################################################################
    # Next we create the xbars according to am 
    ####################################################################
    # The inputs are Xbar Slave ports
    # The outputs are Xbar Master ports
    ###################################
    # 64 bit SoC AXI Lite Periph Xbar #
    ###################################
    soc_axi_lite_periph_xbar = solder.AxiLiteXbar(
        48,
        64,
        chipidw=occamy_cfg["hemaia_multichip"]["chip_id_width"],
        name="soc_axi_lite_periph_xbar",
        clk="clk_periph_i",
        rst="rst_periph_ni",
        context="top_axi_lite_periph",
        node=am_soc_axi_lite_periph_xbar)

    soc_axi_lite_periph_xbar.add_input("soc")
    soc_axi_lite_periph_xbar.add_output_entry("soc", am_soc_narrow_xbar)

    # connect AXI lite peripherals
    nr_axi_lite_peripherals = len(
        occamy_cfg["peripherals"]["axi_lite_peripherals"])
    for p in range(nr_axi_lite_peripherals):
        soc_axi_lite_periph_xbar.add_input(
            occamy_cfg["peripherals"]["axi_lite_peripherals"][p]["name"]
        )
        if "address" in occamy_cfg["peripherals"]["axi_lite_peripherals"][p]:
            soc_axi_lite_periph_xbar.add_output_entry(
            occamy_cfg["peripherals"]["axi_lite_peripherals"][p]["name"],
            am_soc_axi_lite_peripherals[p]
            )
    ###################################
    # 32 bit SoC AXI Lite Periph Xbar #
    ###################################
    soc_axi_lite_narrow_periph_xbar = solder.AxiLiteXbar(
        48,
        32,
        chipidw=occamy_cfg["hemaia_multichip"]["chip_id_width"],
        name="soc_axi_lite_narrow_periph_xbar",
        clk="clk_periph_i",
        rst="rst_periph_ni",
        fall_through=False,
        context="top_axi_lite_periph",
        node=am_soc_axi_lite_narrow_periph_xbar)

    soc_axi_lite_narrow_periph_xbar.add_input("soc")

    # connect Regbus peripherals
    nr_axi_lite_narrow_peripherals = len(
        occamy_cfg["peripherals"]["axi_lite_narrow_peripherals"])
    for p in range(nr_axi_lite_narrow_peripherals):
        soc_axi_lite_narrow_periph_xbar.add_output_entry(
            occamy_cfg["peripherals"]["axi_lite_narrow_peripherals"][p]["name"],
            am_soc_axi_lite_narrow_peripherals[p]
        )
    # add bootrom and clint seperately
    soc_axi_lite_narrow_periph_xbar.add_output_entry("bootrom", am_bootrom)
    soc_axi_lite_narrow_periph_xbar.add_output_entry("clint", am_clint)

    #################
    # SoC Wide Xbar #
    #################
    soc_wide_xbar = solder.AxiXbar(
        48,
        512,
        # This is the cleanest solution minimizing ID width conversions
        iw=occamy_cfg["wide_xbar_slv_id_width"],
        uw=occamy_cfg["wide_xbar_slv_user_width"],
        chipidw=occamy_cfg["hemaia_multichip"]["chip_id_width"],
        name="soc_wide_xbar",
        clk="clk_i",
        rst="rst_ni",
        max_slv_trans=occamy_cfg["wide_xbar"]["max_slv_trans"],
        max_mst_trans=occamy_cfg["wide_xbar"]["max_mst_trans"],
        fall_through=occamy_cfg["wide_xbar"]["fall_through"],
        no_loopback=True,
        atop_support=False,
        context="soc",
        node=am_soc_wide_xbar)

    if occamy_cfg["hemaia_multichip"]["single_chip"] is False:
        # The chiplet output port does not have the rule; it is the default port
        soc_wide_xbar.outputs.append("hemaia_multichip")
        soc_wide_xbar.add_input("hemaia_multichip")
    soc_wide_xbar.add_output_entry("soc_narrow", am_soc_narrow_xbar)
    soc_wide_xbar.add_input("soc_narrow")
    soc_wide_xbar.add_input("sys_idma_mst")
    soc_wide_xbar.add_output_multi_entries("hemaia_mem", [am_wide_hemaia_mem, am_wide_hemaia_xdma_data_io])
    soc_wide_xbar.add_input("hemaia_mem")

    soc_wide_xbar.add_output_entry("wide_zero_mem", am_wide_zero_mem)
    # --> mask this route, forcing it through default wide xbar
    soc_wide_xbar.add_output_entry("quad",
                                    am_quad_wide_xbar)
    soc_wide_xbar.add_input("quad")

    ###################
    # SoC Narrow Xbar #
    ###################
    soc_narrow_xbar = solder.AxiXbar(
        48,
        64,
        iw=occamy_cfg["narrow_xbar_slv_id_width"],
        uw=occamy_cfg["narrow_xbar_slv_user_width"],
        chipidw=occamy_cfg["hemaia_multichip"]["chip_id_width"],
        name="soc_narrow_xbar",
        clk="clk_i",
        rst="rst_ni",
        max_slv_trans=occamy_cfg["narrow_xbar"]["max_slv_trans"],
        max_mst_trans=occamy_cfg["narrow_xbar"]["max_mst_trans"],
        fall_through=occamy_cfg["narrow_xbar"]["fall_through"],
        no_loopback=True,
        context="soc",
        node=am_soc_narrow_xbar)
    
    # Default port: wide xbar (Should stay on the first position)
    soc_narrow_xbar.add_output_entry("soc_wide", am_soc_wide_xbar)
    soc_narrow_xbar.add_input("soc_wide")

    # Quadrant
    # soc_narrow_xbar.add_output_symbolic_multi("quad",
    #                                         [("ClusterBaseOffset",
    #                                         "QuadrantAddressSpace"),
    #                                         ("QuadrantCfgBaseOffset",
    #                                         "QuadrantCfgAddressSpace")])
    soc_narrow_xbar.add_output_entry("quad", am_soc_to_quad_axi_xbar)
    soc_narrow_xbar.add_input("quad")

    soc_narrow_xbar.add_input("cva6")
    soc_narrow_xbar.add_input("periph")

    soc_narrow_xbar.add_output_entry("periph", am_soc_axi_lite_periph_xbar)
    soc_narrow_xbar.add_output_entry("spm_narrow", am_spm_narrow)
    soc_narrow_xbar.add_output_entry("sys_idma_cfg", am_sys_idma_cfg)
    soc_narrow_xbar.add_output_entry("axi_lite_narrow_periph",
                                     am_soc_axi_lite_narrow_periph_xbar)

    # hemaia mem system
    soc_narrow_xbar.add_input("hemaia_mem")
    soc_narrow_xbar.add_output_multi_entries("hemaia_mem", [am_narrow_hemaia_xdma_ctrl_io, am_narrow_hemaia_xdma_cfg_io])
    #############
    # Quad Ctrl #
    #############
    quad_ctrl_soc_to_quad_xbar = solder.AxiXbar(
        48,
        64,
        soc_narrow_xbar.iw_out(),
        soc_narrow_xbar.uw,
        chipidw=occamy_cfg["hemaia_multichip"]["chip_id_width"], 
        name="quad_ctrl_soc_to_quad_xbar",  
        clk="clk_i",
        rst="rst_ni",
        max_slv_trans=occamy_cfg["narrow_xbar"]["max_slv_trans"],
        max_mst_trans=occamy_cfg["narrow_xbar"]["max_mst_trans"],
        fall_through=occamy_cfg["narrow_xbar"]["fall_through"],
        latency_mode="axi_pkg::CUT_SLV_PORTS",
        context="quad_ctrl",
        node=am_soc_to_quad_axi_xbar
    )
    quad_ctrl_soc_to_quad_xbar.add_input("soc_narrow")
    quad_ctrl_soc_to_quad_xbar.add_output("clusters",
                                          [*addrs_clusters, *addrs_clusters_periph])
    quad_ctrl_soc_to_quad_xbar.add_output("quad_axi_lite", addrs_quad_axi_lite_peripherals)
    quad_ctrl_soc_to_quad_xbar.add_output("quad_axi_lite_narrow", addrs_quad_axi_lite_narrow_peripherals)
        

    quad_ctrl_quad_to_soc_xbar = solder.AxiXbar(
        48,
        64,
        soc_narrow_xbar.iw-1, # Very Important!!!!!!!!!! -1 because the quad to soc has two slave ports TODO: Make a clear documentation about this
        soc_narrow_xbar.uw,
        chipidw=occamy_cfg["hemaia_multichip"]["chip_id_width"], 
        name="quad_ctrl_quad_to_soc_xbar",  
        clk="clk_i",
        rst="rst_ni",
        max_slv_trans=occamy_cfg["narrow_xbar"]["max_slv_trans"],
        max_mst_trans=occamy_cfg["narrow_xbar"]["max_mst_trans"],
        fall_through=occamy_cfg["narrow_xbar"]["fall_through"],
        latency_mode="axi_pkg::CUT_MST_PORTS",
        context="quad_ctrl",
        node=am_quad_to_soc_axi_xbar)
    quad_ctrl_quad_to_soc_xbar.add_input("clusters")
    quad_ctrl_quad_to_soc_xbar.add_input("quad_axi_lite")
    # Leave it be empty as the default port
    quad_ctrl_quad_to_soc_xbar.add_output("soc_narrow", [])
    quad_ctrl_quad_to_soc_xbar.add_output("quad_axi_lite", addrs_quad_axi_lite_peripherals)
    quad_ctrl_quad_to_soc_xbar.add_output("quad_axi_lite_narrow", addrs_quad_axi_lite_narrow_peripherals)
    ################################
    # Quad Ctrl 32bit AXI Lite mux #
    ################################
    # Here we hook the hw mailboxes
    # The number of mailboxes equals to the number of clusters
    quad_ctrl_axi_lite_narrow_mux = solder.AxiLiteXbar(
        48,
        32,
        chipidw=occamy_cfg["hemaia_multichip"]["chip_id_width"],
        name="quad_axi_lite_narrow_mux",
        clk="clk_i",
        rst="rst_ni",
        max_slv_trans=occamy_cfg["narrow_xbar"]["max_slv_trans"],
        max_mst_trans=occamy_cfg["narrow_xbar"]["max_mst_trans"],
        fall_through=False,
        context="quad_ctrl",
        node=am_quad_axi_lite_narrow_xbar)

    quad_ctrl_axi_lite_narrow_mux.add_input("soc_to_quad")
    quad_ctrl_axi_lite_narrow_mux.add_input("quad_to_soc")
    quad_ctrl_axi_lite_narrow_mux.add_output_entry("quad_ctrl_perpheral", am_quad_axi_lite_narrow_peripheral)
    for i in range(nr_s1_clusters):
        quad_ctrl_axi_lite_narrow_mux.add_output_entry(
            "h2c_mailbox_{}".format(i), am_quad_axi_lite_narrow_h2c_mailboxs[i]
        )

    ################################
    # Quad Ctrl 64bit AXI Lite Xbar#
    ################################
    quad_ctrl_axi_lite_xbar = solder.AxiLiteXbar(
        48,
        64,
        chipidw=occamy_cfg["hemaia_multichip"]["chip_id_width"],
        name="quad_axi_lite_xbar",
        clk="clk_i",
        rst="rst_ni",
        max_slv_trans=occamy_cfg["narrow_xbar"]["max_slv_trans"],
        max_mst_trans=occamy_cfg["narrow_xbar"]["max_mst_trans"],
        fall_through=False,
        context="quad_ctrl",
        node=am_quad_axi_lite_xbar)
    quad_ctrl_axi_lite_xbar.add_input("soc_to_quad")
    quad_ctrl_axi_lite_xbar.add_input("quad_to_soc")
    quad_ctrl_axi_lite_xbar.add_input("bingo_hw_scheduler_read_local_task")
    quad_ctrl_axi_lite_xbar.add_input("bingo_hw_scheduler_write_remote_done")
    quad_ctrl_axi_lite_xbar.add_input("bingo_hw_scheduler_write_pm")
    quad_ctrl_axi_lite_xbar.add_output_entry("quad_to_soc", am_quad_to_soc_axi_xbar)
    quad_ctrl_axi_lite_xbar.add_output_entry("bingo_hw_scheduler_chiplet_done_queue", am_quad_axi_lite_peripherals[0])
    quad_ctrl_axi_lite_xbar.add_output_entry("bingo_hw_scheduler_host_ready_done_intf", am_quad_axi_lite_peripherals[1])
    
    ##############
    # Quad XBars #
    ##############
    quad_wide_xbar = solder.AxiXbar(
        48,
        512,
        iw=occamy_cfg["s1_quadrant"]["wide_xbar_slv_id_width"],
        uw=occamy_cfg["s1_quadrant"]["wide_xbar_slv_user_width"],
        chipidw=occamy_cfg["hemaia_multichip"]["chip_id_width"],
        name="quad_wide_xbar",
        clk="clk_i",
        rst="rst_ni",
        max_slv_trans=occamy_cfg["s1_quadrant"]["wide_xbar"]["max_slv_trans"],
        max_mst_trans=occamy_cfg["s1_quadrant"]["wide_xbar"]["max_mst_trans"],
        fall_through=occamy_cfg["s1_quadrant"]["wide_xbar"]["fall_through"],
        no_loopback=True,
        atop_support=False,
        context="quad",
        node=am_quad_wide_xbar)
    quad_wide_xbar.add_input("soc_wide")
    quad_wide_xbar.add_output_entry("soc_wide", am_soc_wide_xbar)
    for i in range(nr_s1_clusters):
        quad_wide_xbar.add_output_multi_entries("cluster_{}".format(i),
                                        [am_clusters[i], am_clusters_leftover_spaces[i]])
        quad_wide_xbar.add_input("cluster_{}".format(i))

    quad_narrow_xbar = solder.AxiXbar(
        48,
        64,
        iw=occamy_cfg["s1_quadrant"]["narrow_xbar_slv_id_width"],
        uw=occamy_cfg["s1_quadrant"]["narrow_xbar_slv_user_width"],
        chipidw=occamy_cfg["hemaia_multichip"]["chip_id_width"],
        name="quad_narrow_xbar",
        clk="clk_i",
        rst="rst_ni",
        max_slv_trans=occamy_cfg["s1_quadrant"]["narrow_xbar"]
        ["max_slv_trans"],
        max_mst_trans=occamy_cfg["s1_quadrant"]["narrow_xbar"]
        ["max_mst_trans"],
        fall_through=occamy_cfg["s1_quadrant"]["narrow_xbar"]["fall_through"],
        no_loopback=True,
        context="quad",
        node=am_quad_narrow_xbar)
    quad_narrow_xbar.add_input("soc_to_quad")
    quad_narrow_xbar.add_output_entry("quad_to_soc", am_quad_to_soc_axi_xbar)
    for i in range(nr_s1_clusters):
        quad_narrow_xbar.add_output_multi_entries("cluster_{}".format(i), [am_clusters[i], am_clusters_periph[i], am_clusters_leftover_spaces[i]])
        quad_narrow_xbar.add_input("cluster_{}".format(i))
    
    # wide_xbar_quadrant_s1.add_output("top", [])
    # wide_xbar_quadrant_s1.add_input("top")

    # narrow_xbar_quadrant_s1.add_output("top", [])
    # narrow_xbar_quadrant_s1.add_input("top")

    # for i in range(nr_s1_clusters):
    #     wide_xbar_quadrant_s1.add_output_symbolic("cluster_{}".format(i),
    #                                               "cluster_base_addr",
    #                                               "ClusterAddressSpace")

    #     wide_xbar_quadrant_s1.add_input("cluster_{}".format(i))
    #     narrow_xbar_quadrant_s1.add_output_symbolic("cluster_{}".format(i),
    #                                                 "cluster_base_addr",
    #                                                 "ClusterAddressSpace")
    #     narrow_xbar_quadrant_s1.add_input("cluster_{}".format(i))

    # Generate the Verilog code for occamy_pkg.sv (Only include the definition related to xbars)
    solder.render()

    ##############################################
    # Die2Die AXI Bus For Module I/O Declaration #
    ##############################################
    # As the Die2Die communication is irrelevant to XBars inside one chip, it is declared in the standalone way, so it should be placed below solder.render()
    soc2router_bus = solder.AxiBus(
        clk=soc_wide_xbar.clk,
        rst=soc_wide_xbar.rst,
        aw=soc_wide_xbar.aw,
        dw=soc_wide_xbar.dw,
        iw=occamy_cfg["hemaia_multichip"]["soc_to_router_iw"],
        uw=soc_wide_xbar.uw,
        name="soc2router_bus",
        # declared=True
    )

    router2soc_bus = solder.AxiBus(
        clk=soc_wide_xbar.clk,
        rst=soc_wide_xbar.rst,
        aw=soc_wide_xbar.aw,
        dw=soc_wide_xbar.dw,
        iw=occamy_cfg["hemaia_multichip"]["router_to_soc_iw"],
        uw=soc_wide_xbar.uw,
        name="router2soc_bus",
        # declared=True
    )

    # Emit the code.
    #############
    # Top-Level #
    #############
    if args.top_sv:
        top_kwargs = occamy.get_top_kwargs(occamy_cfg, cluster_generators,
                                    soc_axi_lite_narrow_periph_xbar, soc_wide_xbar, soc_narrow_xbar, soc2router_bus, router2soc_bus, util, args.name)
        write_template(args.top_sv,
                       outdir,
                       fname="{}_top.sv".format(args.name),
                       module=solder.code_module['top_axi_lite_periph'],
                       soc_periph_xbar=soc_axi_lite_periph_xbar,
                       **top_kwargs)
    ###########################
    # SoC (fully synchronous) #
    ###########################
    if args.soc_sv:
        soc_kwargs = occamy.get_soc_kwargs(
            occamy_cfg, cluster_generators, soc_narrow_xbar, soc_wide_xbar, soc2router_bus, router2soc_bus, util, args.name)
        write_template(args.soc_sv,
                       outdir,
                       module=solder.code_module['soc'],
                       soc_periph_xbar=soc_axi_lite_periph_xbar,
                       **soc_kwargs)
    #############
    # Quad CTRL #
    #############
    # The "module" will appear at the template file
    # The "module" is the "context" foe each xbar
    # It groups the xbars together 
    if args.quad_ctrl:
        quad_ctrl_kwargs = occamy.get_quad_ctrl_kwargs(
            occamy_cfg, soc_wide_xbar, soc_narrow_xbar, quad_ctrl_soc_to_quad_xbar, quad_ctrl_quad_to_soc_xbar, quad_ctrl_axi_lite_narrow_mux, quad_ctrl_axi_lite_xbar, cluster_generators,args.name)
        write_template(args.quad_ctrl,
                       outdir,
                       module=solder.code_module['quad_ctrl'],
                       **quad_ctrl_kwargs)
    ########
    # Quad #
    ########
    if args.quad:
        if occamy_cfg["s1_quadrant"].get("noc_cfg", None):
            quadrant_s1_noc_kwargs = occamy.get_quadrant_noc_kwargs(occamy_cfg, cluster_generators)
            write_template(args.quad_noc,
                        outdir,
                        fname="{}_quad_noc.yml".format(args.name),
                        **quadrant_s1_noc_kwargs)
            occamy.generate_floonoc(f"{outdir}/{args.name}_quad_noc.yml", outdir)
        quadrant_kwargs = occamy.get_quadrant_kwargs(occamy_cfg, cluster_generators, soc_wide_xbar,
                                              soc_narrow_xbar, quad_wide_xbar, quad_narrow_xbar, quad_ctrl_quad_to_soc_xbar, args.name)
        if nr_s1_quadrants > 0:
            write_template(args.quad,
                           outdir,
                           fname="{}_quad.sv".format(args.name),
                           module=solder.code_module['quad'],
                           **quadrant_kwargs)
        else:
            tpl_path = args.quad
            if tpl_path:
                tpl_path = pathlib.Path(tpl_path).absolute()
                if tpl_path.exists():
                    print(outdir, args.name)
                    with open("{}/{}_quad.sv".format(outdir, args.name), 'w') as f:
                        f.write("// no quad in this design")
    ##################
    # Xilinx Wrapper #
    ##################
    if args.xilinx_sv:
        xilinx_kwargs = occamy.get_xilinx_kwargs(
            occamy_cfg, soc_wide_xbar, soc_axi_lite_narrow_periph_xbar, args.name)
        write_template(args.xilinx_sv,
                       outdir,
                       fname="{}_xilinx.sv".format(args.name),
                       module="",
                       **xilinx_kwargs)

    ###########
    # Package #
    ###########
    if args.pkg_sv:
        pkg_kwargs = occamy.get_pkg_kwargs(occamy_cfg, cluster_generators, util, args.name)
        write_template(args.pkg_sv, outdir, **pkg_kwargs,
                       package=solder.code_package)

    ################
    # CVA6 Wrapper #
    ################
    if args.cva6_sv:
        cva6_kwargs = occamy.get_cva6_kwargs(occamy_cfg, cluster_generators, soc_narrow_xbar, util, args.name)
        write_template(args.cva6_sv, outdir, **cva6_kwargs)

    ###################
    # Generic CHEADER #
    ###################
    if args.cheader:
        cheader_kwargs = occamy.get_cheader_kwargs(
            occamy_cfg, cluster_generators, args.name)
        write_template(args.cheader, outdir, **cheader_kwargs)

    ###################
    # ADDRMAP CHEADER #
    ###################
    if args.am_cheader:
        with open(args.am_cheader, "w") as file:
            file.write(am.print_cheader())

    ###############
    # ADDRMAP CSV #
    ###############
    if args.am_csv:
        with open(args.am_csv, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            am.print_csv(csv_writer)

    ###############
    # Testharness #
    ###############
    if args.testharness_sv:
        testharness_kwargs = occamy.get_testharness_kwargs(
            occamy_cfg, soc2router_bus, router2soc_bus, args.name)
        write_template(args.testharness_sv, outdir, **testharness_kwargs)

    if args.mem_macro_testharness_sv:
        mem_macro_testharness_kwargs = occamy.get_mem_macro_testharness_kwargs(
            occamy_cfg, soc2router_bus, router2soc_bus, args.name)
        write_template(args.mem_macro_testharness_sv, outdir, **mem_macro_testharness_kwargs)

    if args.netlist_testharness_sv:
        netlist_testharness_kwargs = occamy.get_netlist_testharness_kwargs(
            occamy_cfg, soc2router_bus, router2soc_bus, args.name)
        write_template(args.netlist_testharness_sv, outdir, **netlist_testharness_kwargs)

    ############
    # BOOTDATA #
    ############
    if args.bootdata:
        bootdata_kwargs = occamy.get_bootdata_kwargs(
            occamy_cfg, cluster_generators, args.name)
        bootdata_fname = "bootdata.cc"
        write_template(args.bootdata, outdir,
                       bootdata_fname, **bootdata_kwargs)
    ########
    # XDMA #
    ########
    if args.xdma is True and occamy_cfg["hemaia_xdma_cfg"] is not None:
        print("------------------------------------------------")
        print("    Generate HeMAiA Compute Chip XDMA")
        print("------------------------------------------------")
        import importlib.util
        script_dir = pathlib.Path(__file__).parent.resolve()
        snaxgen_path = script_dir / ".." / ".." / "deps" / "snitch_cluster" /  "util" / "snaxgen" / "snaxgen.py"
        spec = importlib.util.spec_from_file_location("snaxgen", snaxgen_path)
        snaxgen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(snaxgen)
        gen_file = snaxgen.gen_file
        gen_chisel_file = snaxgen.gen_chisel_file
        get_template = snaxgen.get_template

        tpl_rtl_wrapper_file = script_dir / ".." / ".." / "deps" / "snitch_cluster" / "hw" / "templates" / "snax_xdma_wrapper.sv.tpl"

        tpl_rtl_wrapper = get_template(tpl_rtl_wrapper_file)

        gen_file(
            cfg={
                "name": "hemaia",
                "xdma_cfg_io_width": occamy_cfg["hemaia_xdma_cfg"]["cfg_io_width"],
                "dma_data_width": 512,
                "data_width": occamy_cfg["data_width"],
                "addr_width": occamy_cfg["addr_width"],
                "tcdm": {
                    "size": int(occamy_cfg["spm_wide"]["length"]/1024),
                }
            },
            tpl=tpl_rtl_wrapper,
            target_path=str(script_dir / ".." / ".." / "hw" / "hemaia" / "hemaia_mem_system") + "/",
            file_name= "hemaia_xdma_wrapper.sv",
        )

        gen_chisel_file(
            chisel_path=str(script_dir / ".." / ".." / "deps" / "snitch_cluster" / "hw" / "chisel"),
            chisel_param="snax.xdma.xdmaTop.XDMATopGen",
            gen_path=" --clusterName "
            + "hemaia"
            + " --tcdmDataWidth "
            + str(occamy_cfg["data_width"])
            + " --axiDataWidth "
            + str(512)
            + " --axiAddrWidth "
            + str(occamy_cfg["addr_width"])
            + " --tcdmSize "
            + str(int(occamy_cfg["spm_wide"]["length"]/1024))
            + " --xdmaCfg "
            + hjson.dumpsJSON(obj=occamy_cfg["hemaia_xdma_cfg"], separators=(",", ":")).replace(" ", "")
            + " --hw-target-dir "
            + str(script_dir / ".." / ".." / "hw" / "hemaia" / "hemaia_mem_system") + "/"
            + " --sw-target-dir "
            + str(script_dir / ".." / ".." / "target" / "sw" / "shared" / "vendor" / "xdma" / "hemaia-xdma-addr.h")
        )
        print("HeMAiA Chip XDMA generation finished")

    if args.xdma is True and any(occamy_cfg["hemaia_multichip"]["testbench_cfg"]["hemaia_mem_chip"]) and occamy_cfg["hemaia_xdma_cfg"] is not None:
        print("------------------------------------------------")
        print("    Generate HeMAiA Mem Chip XDMA")
        print("------------------------------------------------")
        import importlib.util
        script_dir = pathlib.Path(__file__).parent.resolve()
        snaxgen_path = script_dir / ".." / ".." / "deps" / "snitch_cluster" /  "util" / "snaxgen" / "snaxgen.py"
        spec = importlib.util.spec_from_file_location("snaxgen", snaxgen_path)
        snaxgen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(snaxgen)
        gen_file = snaxgen.gen_file
        gen_chisel_file = snaxgen.gen_chisel_file
        get_template = snaxgen.get_template

        tpl_rtl_wrapper_file = script_dir / ".." / ".." / "deps" / "snitch_cluster" / "hw" / "templates" / "snax_xdma_wrapper.sv.tpl"

        tpl_rtl_wrapper = get_template(tpl_rtl_wrapper_file)

        gen_file(
            cfg={
                "name": "hemaia_mem_chip",
                "xdma_cfg_io_width": occamy_cfg["hemaia_xdma_cfg"]["cfg_io_width"],
                "dma_data_width": 512,
                "data_width": occamy_cfg["data_width"],
                "addr_width": occamy_cfg["addr_width"],
                "tcdm": {
                    "size": int(occamy_cfg["hemaia_multichip"]["testbench_cfg"]["hemaia_mem_chip"][0]["mem_size"]/1024),
                }
            },
            tpl=tpl_rtl_wrapper,
            target_path=str(script_dir / ".." / ".." / "hw" / "hemaia" / "hemaia_mem_system") + "/",
            file_name= "hemaia_mem_chip_xdma_wrapper.sv",
        )

        gen_chisel_file(
            chisel_path=str(script_dir / ".." / ".." / "deps" / "snitch_cluster" / "hw" / "chisel"),
            chisel_param="snax.xdma.xdmaTop.XDMATopGen",
            gen_path=" --clusterName "
            + "hemaia_mem_chip"
            + " --tcdmDataWidth "
            + str(occamy_cfg["data_width"])
            + " --axiDataWidth "
            + str(512)
            + " --axiAddrWidth "
            + str(occamy_cfg["addr_width"])
            + " --tcdmSize "
            + str(int(occamy_cfg["hemaia_multichip"]["testbench_cfg"]["hemaia_mem_chip"][0]["mem_size"]/1024))
            + " --xdmaCfg "
            + hjson.dumpsJSON(obj=occamy_cfg["hemaia_xdma_cfg"], separators=(",", ":")).replace(" ", "")
            + " --hw-target-dir "
            + str(script_dir / ".." / ".." / "hw" / "hemaia" / "hemaia_mem_system") + "/"
            + " --sw-target-dir "
            + str(script_dir / ".." / ".." / "target" / "sw" / "shared" / "vendor" / "xdma" / "hemaia-mem-chip-xdma-addr.h")
        )
        print("HeMAiA Mem Chip XDMA generation finished")


    # # generate the loader script
    if args.gen_host_ld:
        print("------------------------------------------------")
        print("    Generate host.ld")
        print("------------------------------------------------")
        import importlib.util
        script_dir = pathlib.Path(__file__).parent.resolve()
        host_ld_tpl_file = script_dir / ".." / ".." / "target" / "sw" / "host" / "runtime" / "host.ld.tpl"
        cfg={
            "name": args.name,
            "spm_narrow": occamy_cfg["spm_narrow"],
            "spm_wide": occamy_cfg["spm_wide"],
        }
        write_template(
            host_ld_tpl_file,
            outdir,
            **cfg
        )

    ########
    # CHIP #
    ########
    if args.chip:
        chip_kwargs = occamy.get_chip_kwargs(
            soc_wide_xbar, soc_narrow_xbar, soc_axi_lite_narrow_periph_xbar, soc2router_bus, router2soc_bus, occamy_cfg, cluster_generators, util, args.name)
        write_template(args.chip, outdir, **chip_kwargs)

    ########
    # IO #
    ########
    if args.io:
        io_kwargs = occamy.get_io_kwargs(
            occamy_cfg, util, args.name)
        write_template(args.io, outdir, **io_kwargs)

    ########
    # CTRL #
    ########
    if args.ctrl:
        ctrl_kwargs = occamy.get_ctrl_kwargs(occamy_cfg, cluster_generators, args.name)
        write_template(args.ctrl, outdir, **ctrl_kwargs)

    # Emit the address map as a dot file if requested.
    # If want to see the graph, apt-get install graphviz and run:
    # dot -Tpng addrmap.dot -o addrmap.png
    if args.graph:
        with open(args.graph, "w") as file:
            file.write(am.render_graphviz())


if __name__ == "__main__":
    main()
