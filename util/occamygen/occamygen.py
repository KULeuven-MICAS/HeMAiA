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
    parser.add_argument("--quadrant-s1",
                        metavar="QUADRANT_S1",
                        help="Name of S1 quadrant template file (output)")
    parser.add_argument("--quadrant-s1-ctrl",
                        metavar="QUADRANT_S1_CTL",
                        help="Name of S1 quadrant controller template file (output)")
    parser.add_argument("--xilinx-sv",
                        metavar="XILINX_SV",
                        help="Name of the Xilinx wrapper file (output).")
    parser.add_argument("--testharness-sv",
                        metavar="TESTHARNESS_SV",
                        help="Name of the testharness wrapper file (output).")
    parser.add_argument("--multichip-testharness-sv",
                        metavar="MULTICHIP_TESTHARNESS_SV",
                        help="Name of the multichip testharness wrapper file (output).")
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
    parser.add_argument("--chip",
                        metavar="CHIP_TOP",
                        help="(Optional) Chip Top-level")
    parser.add_argument("--ctrl",
                        metavar="SoC and Quad Ctrl",
                        help="Name of SoC or Quadrant template file (output)")
    parser.add_argument("--graph", "-g", metavar="DOT")
    parser.add_argument("--memories", "-m", action="store_true")
    parser.add_argument("--wrapper", "-w", action="store_true")
    parser.add_argument("--am-cheader", "-D", metavar="ADDRMAP_CHEADER")
    parser.add_argument("--am-csv", "-aml", metavar="ADDRMAP_CSV")
    parser.add_argument("--dts", metavar="DTS", help="System's device tree.")
    parser.add_argument("--name", metavar="NAME",
                        default=DEFAULT_NAME, help="System's name.")

    parser.add_argument("-v",
                        "--verbose",
                        help="increase output verbosity",
                        action="store_true")

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

    # Now we create all the am for xbars
    # Each XBAR is a node
    # Create the address map
    am = solder.AddrMap()
    # Create a device tree object.
    dts = device_tree.DeviceTree()

    # Toplevel crossbar address map
    am_soc_narrow_xbar = am.new_node("soc_narrow_xbar")
    am_soc_wide_xbar = am.new_node("soc_wide_xbar")

    # Quadrant inter crossbar address map:
    # am_quadrant_inter_xbar = am.new_node("am_quadrant_inter_xbar")

    # Quadrant crossbar address map
    am_wide_xbar_quadrant_s1 = list()
    am_narrow_xbar_quadrant_s1 = list()
    for i in range(nr_s1_quadrants):
        am_wide_xbar_quadrant_s1.append(
            am.new_node("wide_xbar_quadrant_s1_{}".format(i)))
        am_narrow_xbar_quadrant_s1.append(
            am.new_node("narrow_xbar_quadrant_s1_{}".format(i)))

    # Peripheral crossbar address map
    am_soc_axi_lite_periph_xbar = am.new_node("soc_axi_lite_periph_xbar")
    am_soc_axi_lite_narrow_periph_xbar = am.new_node(
        "soc_axi_lite_narrow_periph_xbar")

    # After that, we need to create the leaves that connect to xbars
    am_axi_lite_peripherals = occamy.am_connect_soc_lite_periph_xbar(
        am, am_soc_axi_lite_periph_xbar, occamy_cfg)

    am_axi_lite_narrow_peripherals, am_bootrom, am_clint = occamy.am_connect_soc_lite_narrow_periph_xbar(
        am, am_soc_axi_lite_narrow_periph_xbar, occamy_cfg)

    am_spm_narrow, am_sys_idma_cfg = occamy.am_connect_soc_narrow_xbar(
        am, am_soc_narrow_xbar, occamy_cfg)
    am_spm_wide, am_wide_zero_mem = occamy.am_connect_soc_wide_xbar_mem(
        am, am_soc_wide_xbar, occamy_cfg)

    am_clusters = occamy.am_connect_soc_wide_xbar_quad(
        am, am_soc_narrow_xbar, am_wide_xbar_quadrant_s1, am_narrow_xbar_quadrant_s1, occamy_cfg, cluster_generators)
    # Then we connect between xbars

    # Connect quadrants AXI xbar
    for i in range(nr_s1_quadrants):
        am_narrow_xbar_quadrant_s1[i].attach(am_wide_xbar_quadrant_s1[i])
        am_soc_narrow_xbar.attach(am_narrow_xbar_quadrant_s1[i])
        am_soc_wide_xbar.attach(am_wide_xbar_quadrant_s1[i])

    # Connect quadrant inter xbar
    # am_soc_wide_xbar.attach(am_quadrant_inter_xbar)
    # am_quadrant_inter_xbar.attach(am_soc_wide_xbar)

    # Connect narrow xbar
    am_soc_narrow_xbar.attach(am_soc_axi_lite_periph_xbar)
    am_soc_narrow_xbar.attach(am_soc_axi_lite_narrow_periph_xbar)
    am_soc_narrow_xbar.attach(am_soc_wide_xbar)

    am_soc_axi_lite_periph_xbar.attach(am_soc_narrow_xbar)

    # Connect wide xbar
    am_soc_wide_xbar.attach(am_soc_narrow_xbar)

    # Next we create the xbars according to am
    #######################
    # SoC Peripheral Xbar #
    #######################
    # AXI-Lite
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
            am_axi_lite_peripherals[p]
            )
    ##################
    # AxiLite Narrow #
    ##################
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
            am_axi_lite_narrow_peripherals[p]
        )

    # add bootrom and clint seperately
    soc_axi_lite_narrow_periph_xbar.add_output_entry("bootrom", am_bootrom)
    soc_axi_lite_narrow_periph_xbar.add_output_entry("clint", am_clint)

    soc_wide_xbar = solder.AxiXbar(
        48,
        512,
        # This is the cleanest solution minimizing ID width conversions
        occamy_cfg["quadrant_inter_xbar_slv_id_width_no_rocache"] + (
            1 if occamy_cfg["s1_quadrant"].get("ro_cache_cfg") else 0),
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

    soc_wide_xbar.add_output_entry("soc_narrow", am_soc_narrow_xbar)
    soc_wide_xbar.add_input("soc_narrow")
    soc_wide_xbar.add_input("sys_idma_mst")
    soc_wide_xbar.add_output_entry("spm_wide", am_spm_wide)
    soc_wide_xbar.add_output_entry("wide_zero_mem", am_wide_zero_mem)
    for i in range(nr_s1_quadrants):
        # Default route passes HBI through quadrant 0
        # --> mask this route, forcing it through default wide xbar
        soc_wide_xbar.add_output_entry("quadrant_{}".format(i),
                                       am_wide_xbar_quadrant_s1[i])
        soc_wide_xbar.add_input("quadrant_{}".format(i))

    ###################
    # SoC Narrow Xbar #
    ###################
    soc_narrow_xbar = solder.AxiXbar(
        48,
        64,
        occamy_cfg["narrow_xbar_slv_id_width"],
        occamy_cfg["narrow_xbar_user_width"],
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

    for i in range(nr_s1_quadrants):
        # soc_narrow_xbar.add_output_symbolic_multi("s1_quadrant_{}".format(i),
        #                                           [("s1_quadrant_base_addr",
        #                                             "S1QuadrantAddressSpace"),
        #                                            ("s1_quadrant_cfg_base_addr",
        #                                             "S1QuadrantCfgAddressSpace")])
        soc_narrow_xbar.add_output_entry(
            "s1_quadrant_{}".format(i), am_narrow_xbar_quadrant_s1[i])
        soc_narrow_xbar.add_input("s1_quadrant_{}".format(i))

    soc_narrow_xbar.add_input("cva6")
    soc_narrow_xbar.add_input("soc_wide")
    soc_narrow_xbar.add_input("periph")

    # Default port: wide xbar
    soc_narrow_xbar.add_output_entry("soc_wide", am_soc_wide_xbar)
    soc_narrow_xbar.add_output_entry("periph", am_soc_axi_lite_periph_xbar)
    soc_narrow_xbar.add_output_entry("spm_narrow", am_spm_narrow)
    soc_narrow_xbar.add_output_entry("sys_idma_cfg", am_sys_idma_cfg)
    soc_narrow_xbar.add_output_entry("axi_lite_narrow_periph",
                                     am_soc_axi_lite_narrow_periph_xbar)

    ##########################
    # S1 Quadrant controller #
    ##########################

    # We need 3 "crossbars", which are really simple muxes and demuxes
    quadrant_s1_ctrl_xbars = dict()
    for name, (iw, lm) in {
        'soc_to_quad': (soc_narrow_xbar.iw_out(), "axi_pkg::CUT_SLV_PORTS"),
        'quad_to_soc': (soc_narrow_xbar.iw, "axi_pkg::CUT_MST_PORTS"),
    }.items():
        # Reuse (preserve) narrow Xbar IDs and max transactions
        quadrant_s1_ctrl_xbars[name] = solder.AxiXbar(
            48,
            64,
            iw,
            soc_narrow_xbar.uw,
            chipidw=occamy_cfg["hemaia_multichip"]["chip_id_width"],
            name="quadrant_s1_ctrl_{}_xbar".format(name),
            clk="clk_i",
            rst="rst_ni",
            max_slv_trans=occamy_cfg["narrow_xbar"]["max_slv_trans"],
            max_mst_trans=occamy_cfg["narrow_xbar"]["max_mst_trans"],
            fall_through=occamy_cfg["narrow_xbar"]["fall_through"],
            latency_mode=lm,
            context="quadrant_s1_ctrl")

    for name in ['soc_to_quad', 'quad_to_soc']:
        quadrant_s1_ctrl_xbars[name].add_output("out", [])
        quadrant_s1_ctrl_xbars[name].add_input("in")
        quadrant_s1_ctrl_xbars[name].add_output_symbolic("internal",
                                                         "internal_xbar_base_addr",
                                                         "S1QuadrantCfgAddressSpace")

    # AXI Lite mux to combine register requests
    quadrant_s1_ctrl_mux = solder.AxiLiteXbar(
        48,
        32,
        chipidw=0,      # quadrant_ctrl_mux is a pure mux, all transactions just send out from out
        name="quadrant_s1_ctrl_mux",
        clk="clk_i",
        rst="rst_ni",
        max_slv_trans=occamy_cfg["narrow_xbar"]["max_slv_trans"],
        max_mst_trans=occamy_cfg["narrow_xbar"]["max_mst_trans"],
        fall_through=False,
        context="quadrant_s1_ctrl")

    quadrant_s1_ctrl_mux.add_output("out", [(0, (1 << 48) - 1)])
    quadrant_s1_ctrl_mux.add_input("soc")
    quadrant_s1_ctrl_mux.add_input("quad")

    ################
    # S1 Quadrants #
    ################
    # Dummy entries to generate associated types.
    wide_xbar_quadrant_s1 = solder.AxiXbar(
        48,
        512,
        occamy_cfg["s1_quadrant"]["wide_xbar_slv_id_width"],
        chipidw=occamy_cfg["hemaia_multichip"]["chip_id_width"],
        name="wide_xbar_quadrant_s1",
        clk="clk_quadrant_uncore",
        rst="rst_quadrant_n",
        max_slv_trans=occamy_cfg["s1_quadrant"]["wide_xbar"]["max_slv_trans"],
        max_mst_trans=occamy_cfg["s1_quadrant"]["wide_xbar"]["max_mst_trans"],
        fall_through=occamy_cfg["s1_quadrant"]["wide_xbar"]["fall_through"],
        no_loopback=True,
        atop_support=False,
        context="quadrant_s1",
        node=am_wide_xbar_quadrant_s1[0])

    narrow_xbar_quadrant_s1 = solder.AxiXbar(
        48,
        64,
        occamy_cfg["s1_quadrant"]["narrow_xbar_slv_id_width"],
        occamy_cfg["s1_quadrant"]["narrow_xbar_user_width"],
        chipidw=occamy_cfg["hemaia_multichip"]["chip_id_width"],
        name="narrow_xbar_quadrant_s1",
        clk="clk_quadrant_uncore",
        rst="rst_quadrant_n",
        max_slv_trans=occamy_cfg["s1_quadrant"]["narrow_xbar"]
        ["max_slv_trans"],
        max_mst_trans=occamy_cfg["s1_quadrant"]["narrow_xbar"]
        ["max_mst_trans"],
        fall_through=occamy_cfg["s1_quadrant"]["narrow_xbar"]["fall_through"],
        no_loopback=True,
        context="quadrant_s1")

    wide_xbar_quadrant_s1.add_output("top", [])
    wide_xbar_quadrant_s1.add_input("top")

    narrow_xbar_quadrant_s1.add_output("top", [])
    narrow_xbar_quadrant_s1.add_input("top")

    for i in range(nr_s1_clusters):
        wide_xbar_quadrant_s1.add_output_symbolic("cluster_{}".format(i),
                                                  "cluster_base_addr",
                                                  "ClusterAddressSpace")

        wide_xbar_quadrant_s1.add_input("cluster_{}".format(i))
        narrow_xbar_quadrant_s1.add_output_symbolic("cluster_{}".format(i),
                                                    "cluster_base_addr",
                                                    "ClusterAddressSpace")
        narrow_xbar_quadrant_s1.add_input("cluster_{}".format(i))

    # Generate the Verilog code.
    solder.render()
    cluster_cfgs = list()
    nr_clusters = len(occamy_cfg["clusters"])
    for i in range(nr_clusters):
        cluster_cfgs.append(cluster_generators[i].cfg)
    # Emit the code.
    #############
    # Top-Level #
    #############
    if args.top_sv:
        top_kwargs = occamy.get_top_kwargs(occamy_cfg, cluster_generators,
                                    soc_axi_lite_narrow_periph_xbar, soc_wide_xbar, soc_narrow_xbar, args.name)
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
            occamy_cfg, cluster_generators, soc_narrow_xbar, soc_wide_xbar, util, args.name)
        write_template(args.soc_sv,
                       outdir,
                       module=solder.code_module['soc'],
                       soc_periph_xbar=soc_axi_lite_periph_xbar,
                       **soc_kwargs)
    ##########################
    # S1 Quadrant controller #
    ##########################
    if args.quadrant_s1_ctrl:
        quadrant_ctrl_kwargs = occamy.get_quadrant_ctrl_kwargs(
            occamy_cfg, soc_wide_xbar, soc_narrow_xbar, quadrant_s1_ctrl_xbars, quadrant_s1_ctrl_mux, args.name)
        write_template(args.quadrant_s1_ctrl,
                       outdir,
                       module=solder.code_module['quadrant_s1_ctrl'],
                       **quadrant_ctrl_kwargs)
    ###############
    # S1 Quadrant #
    ###############
    if args.quadrant_s1:
        quadrant_kwargs = occamy.get_quadrant_kwargs(occamy_cfg, cluster_generators, soc_wide_xbar,
                                              soc_narrow_xbar, wide_xbar_quadrant_s1, narrow_xbar_quadrant_s1, args.name)
        if nr_s1_quadrants > 0:
            write_template(args.quadrant_s1,
                           outdir,
                           fname="{}_quadrant_s1.sv".format(args.name),
                           module=solder.code_module['quadrant_s1'],
                           **quadrant_kwargs)
        else:
            tpl_path = args.quadrant_s1
            if tpl_path:
                tpl_path = pathlib.Path(tpl_path).absolute()
                if tpl_path.exists():
                    print(outdir, args.name)
                    with open("{}/{}_quadrant_s1.sv".format(outdir, args.name), 'w') as f:
                        f.write("// no quadrants in this design")

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
        cva6_kwargs = occamy.get_cva6_kwargs(occamy_cfg, soc_narrow_xbar, args.name)
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
            soc_wide_xbar, soc_axi_lite_narrow_periph_xbar, occamy_cfg["hemaia_multichip"]["single_chip_id"], solder, name)
        write_template(args.testharness_sv, outdir, **testharness_kwargs)

    if args.multichip_testharness_sv:
        multichip_testharness_kwargs = occamy.get_multichip_testharness_kwargs(occamy_cfg, name)
        write_template(args.multichip_testharness_sv, outdir, **multichip_testharness_kwargs)

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
    # CHIP #
    ########
    if args.chip:
        chip_kwargs = occamy.get_chip_kwargs(
            soc_wide_xbar, soc_axi_lite_narrow_periph_xbar, occamy_cfg, cluster_generators, util, args.name)
        write_template(args.chip, outdir, **chip_kwargs)

    ########
    # CTRL #
    ########
    if args.ctrl:
        ctrl_kwargs = occamy.get_ctrl_kwargs(occamy_cfg, args.name)
        write_template(args.ctrl, outdir, **ctrl_kwargs)
    #######
    # DTS #
    #######
    # TODO(niwis, zarubaf): We probably need to think about genrating a couple
    # of different systems here. I can at least think about two in that context:
    # 1. RTL sim
    # 2. FPGA
    # 3. (ASIC) in the private wrapper repo
    # I think we have all the necessary ingredients for this. What is missing is:
    # - Create a second(/third) configuration file.
    # - Generate the RTL into dedicated directories
    # - (Manually) adapt the `Bender.yml` to include the appropriate files.

    if args.dts:
        dts = occamy.get_dts(occamy_cfg, am_clint, am_axi_lite_peripherals,
                      am_axi_lite_narrow_peripherals)
        # TODO(zarubaf): Figure out whether there are any requirements on the
        # model and compatability.
        dts_str = dts.emit("eth,occamy-dev", "eth,occamy")
        with open(args.dts, "w") as file:
            file.write(dts_str)
        # Compile to DTB and save to a file with `.dtb` extension.
        with open(pathlib.Path(args.dts).with_suffix(".dtb"), "wb") as file:
            run(["dtc", args.dts],
                input=dts_str,
                stdout=file,
                shell=True,
                universal_newlines=True)

    # Emit the address map as a dot file if requested.
    if args.graph:
        with open(args.graph, "w") as file:
            file.write(am.render_graphviz())


if __name__ == "__main__":
    main()
