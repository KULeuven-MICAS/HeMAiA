# Copyright 2020 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import os
import sys
import math
import importlib.util
from pathlib import Path
import hjson
from jsonref import JsonRef

sys.path.append(str(Path(__file__).parent /
                '../../deps/snitch_cluster/util/clustergen'))
from cluster import Generator, PMA, PMACfg, SnitchCluster, clog2  # noqa: E402


def read_json_file(file):
    try:
        srcfull = file.read()
        obj = hjson.loads(srcfull, use_decimal=True)
        obj = JsonRef.replace_refs(obj)
    except ValueError:
        raise SystemExit(sys.exc_info()[1])
    return obj


def generate_pma_cfg(occamy_cfg):
    pma_cfg = PMACfg()
    addr_width = occamy_cfg["addr_width"]
    # Make Wide SPM cacheable
    pma_cfg.add_region_length(PMA.CACHED,
                              occamy_cfg["spm_wide"]["address"],
                              occamy_cfg["spm_wide"]["length"],
                              addr_width)
    # Make the SPM cacheable
    pma_cfg.add_region_length(PMA.CACHED,
                              occamy_cfg["spm_narrow"]["address"],
                              occamy_cfg["spm_narrow"]["length"],
                              addr_width)
    # Make the boot ROM cacheable
    pma_cfg.add_region_length(PMA.CACHED,
                              occamy_cfg["peripherals"]["rom"]["address"],
                              occamy_cfg["peripherals"]["rom"]["length"],
                              addr_width)
    return pma_cfg


def check_occamy_cfg(occamy_cfg):
    occamy_root = (Path(__file__).parent / "../../").resolve()
    snitch_root = (Path(__file__).parent /
                   "../../deps/snitch_cluster").resolve()
    schema = occamy_root / "docs/schema/occamy.schema.json"
    remote_schemas = [occamy_root / "docs/schema/axi_xbar.schema.json",
                      occamy_root / "docs/schema/axi_tlb.schema.json",
                      occamy_root / "docs/schema/address_range.schema.json",
                      occamy_root / "docs/schema/peripherals.schema.json",
                      snitch_root / "docs/schema/snitch_cluster.schema.json"]
    generator_obj = Generator(schema, remote_schemas)
    generator_obj.validate(occamy_cfg)




def get_cluster_generators(occamy_cfg, cluster_cfg_dir):
    cluster_generators = list()
    pma_cfg = generate_pma_cfg(occamy_cfg)
    cluster_name_list = occamy_cfg["clusters"]
    for cluster_name in cluster_name_list:
        cluster_cfg_path = cluster_cfg_dir / f"{cluster_name}.hjson"
        with open(cluster_cfg_path, 'r') as file:
            cluster_cfg = read_json_file(file)
        # Now cluster_cfg has three field
        # cluster, dram, clint
        # We only need the cluster
        cluster_cfg = cluster_cfg["cluster"]
        # Add some field
        cluster_processing(cluster_cfg, occamy_cfg)
        cluster_obj = SnitchCluster(cluster_cfg, pma_cfg)
        cluster_add_mem(cluster_obj, occamy_cfg)
        cluster_generators.append(cluster_obj)
    return cluster_generators


def check_and_fix_occamy_xbar_id_width(occamy_cfg, cluster_generators):
    # We take the cluster axi parameters from the first cluster
    cluster_cfg = cluster_generators[0].cfg
    # This is from line 78 of snitch_cluster/hw/snitch_cluster/src/snitch_cluster_wrapper.sv.tpl
    # localparam int unsigned NrMasters = 3;
    cluster_narrow_num_masters = 3
    # This is from line 81 of snitch_cluster/hw/snitch_cluster/src/snitch_cluster_wrapper.sv.tpl
    # localparam int unsigned NrDmaMasters = 3 + ${cfg['nr_hives']};
    cluster_wide_num_masters = 3 + cluster_cfg["nr_hives"]

    # cluster id/width
    cluster_narrow_in_id_width = cluster_cfg["id_width_in"]
    cluster_narrow_out_id_width = cluster_narrow_in_id_width + \
        clog2(cluster_narrow_num_masters)
    cluster_narrow_user_width = cluster_cfg["user_width"]
    cluster_wide_in_id_width = cluster_cfg["dma_id_width_in"]
    cluster_wide_out_id_width = cluster_wide_in_id_width + \
        clog2(cluster_wide_num_masters)
    cluster_wide_user_width = cluster_cfg["dma_user_width"]

    # quadrant id/width
    # Derive the quadrant_narrow_xbar_id from the cluster cfg
    occamy_cfg["s1_quadrant"]["narrow_xbar_slv_id_width"] = cluster_narrow_out_id_width
    # Derive the quadrant_narrow_xbar_user from the cluster cfg
    occamy_cfg["s1_quadrant"]["narrow_xbar_slv_user_width"] = cluster_narrow_user_width
    # Derive the quadrant_wide_xbar_id from the cluster cfg
    occamy_cfg["s1_quadrant"]["wide_xbar_slv_id_width"] = cluster_wide_out_id_width
    # Derive the quadrant_wide_xbar_user from the cluster cfg
    occamy_cfg["s1_quadrant"]["wide_xbar_slv_user_width"] = cluster_wide_user_width
    # Compute the quad xbar out id
    num_clusters = len(occamy_cfg["clusters"])
    quad_narrow_xbar_out_id = occamy_cfg["s1_quadrant"]["narrow_xbar_slv_id_width"] + clog2(num_clusters+1)
    quad_wide_xbar_out_id = occamy_cfg["s1_quadrant"]["wide_xbar_slv_id_width"] + clog2(num_clusters+1)

    # soc id/width
    # occamy_cfg["narrow_xbar_slv_id_width"] = quad_narrow_xbar_out_id
    occamy_cfg["narrow_xbar_slv_user_width"] = occamy_cfg["s1_quadrant"]["narrow_xbar_slv_user_width"]
    # occamy_cfg["wide_xbar_slv_id_width"] = quad_wide_xbar_out_id + (
    #         1 if occamy_cfg["s1_quadrant"].get("ro_cache_cfg") else 0)
    occamy_cfg["wide_xbar_slv_user_width"] = occamy_cfg["s1_quadrant"]["wide_xbar_slv_user_width"]


def get_cluster_cfg_list(occamy_cfg, cluster_cfg_dir):
    cluster_name_list = occamy_cfg["clusters"]
    get_cluster_cfg_list = list()
    for cluster_name in cluster_name_list:
        get_cluster_cfg_list.append(cluster_cfg_dir / f"{cluster_name}.hjson")
    return get_cluster_cfg_list


def generate_snitch(cluster_cfg_dir, snitch_path, xmx="8G", xms="8G"):
    cluster_cfg_dir = set(cluster_cfg_dir)
    env = os.environ.copy()
    jvm_args = f"-Xms{xms} -Xmx{xmx}"

    # Ensure JVM memory settings reach whatever launcher is used (sbt/coursier/mill/java)
    for var in (
        "JAVA_OPTS",
        "SBT_OPTS",
        "JVM_OPTS",
        "MILL_OPTS",
        "COURSIER_JVM_ARGS",
        "JAVA_TOOL_OPTIONS",
        "_JAVA_OPTIONS",
        "FIRRTL_JAVA_ARGS",
    ):
        env[var] = f"{jvm_args} {env.get(var, '')}".strip()

    for cfg in cluster_cfg_dir:
        try:
            subprocess.check_call(
                [
                    "make",
                    "-C",
                    f"{snitch_path}/target/snitch_cluster",
                    f"CFG_OVERRIDE={cfg}",
                    "DISABLE_HEADER_GEN=true",
                    "rtl-gen",
                ],
                env=env,
            )
        except subprocess.CalledProcessError:
            print("Error! SNAX gen fails. Check the log.")
            raise


def generate_wrappers(cluster_generators, out_dir):
    for cluster_generator in cluster_generators:
        cluster_name = cluster_generator.cfg["name"]
        with open(out_dir / f"{cluster_name}_wrapper.sv", "w") as f:
            f.write(cluster_generator.render_wrapper())


def generate_memories(cluster_generators, out_dir):
    for cluster_generator in cluster_generators:
        cluster_name = cluster_generator.cfg["name"]
        with open(out_dir / f"{cluster_name}_memories.json", "w") as f:
            f.write(cluster_generator.memory_cfg())

def generate_floonoc(floonoc_cfg, out_dir):
    """
    Generate the Floonoc sv from the cfg
    :param floonoc_cfg: The floonoc cfg (generated from the template)
    :param out_dir: The output directory
    """
    python_exec = sys.executable
    try:
        # Step0: Using bender to get the floonoc project dir
        result = subprocess.run(
            ["bender", "path", "floo_noc"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        path = result.stdout.strip()
        if not path:
            raise RuntimeError("bender returns empty dir!")
        floo_path = Path(path)
        if not floo_path.exists():
            raise FileNotFoundError(f"bender returns non-exist path: {path}")
        # Step1: Using pip to install the floogen
        spec = importlib.util.find_spec("floogen")
        if spec is None:
            print("Floogen is not installed!")
            install_cmd = [python_exec, "-m", "pip", "install", "."]
            print(f"Installing floogen: {' '.join(install_cmd)}")
            result = subprocess.run(
                install_cmd,
                cwd=floo_path,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("Install Success!")
            print(result.stdout)
        else:
            print("Floogen has installed!")
            print("Generating SV...")
        # Step2: Using the floogen to generate the sv
        cmd = [python_exec, "-m", "floogen.cli"]
        cmd += ["-c", floonoc_cfg, "-o", out_dir]
        cmd += ["--no-format"]
        subprocess.run(cmd, check=True)
        print("Floogen is done.")
    except subprocess.CalledProcessError as e:
        print(f"Install Failed! Erro Code: {e.returncode}")
        print("Stdout:")
        print(e.stdout)
        print("Stderror:")
        print(e.stderr)
        raise
    except Exception as e:
        print(f"Execution Failed: {str(e)}")
        raise


def cluster_processing(cluster_cfg, occamy_cfg):
    # in snith_cluster_wrapper.sv.tpl
    # % if not cfg['tie_ports']:
    #   //-----------------------------
    #   // Cluster base addressing
    #   //-----------------------------
    #   input  logic [9:0]                             hart_base_id_i,
    #   input  logic [${cfg['addr_width']-1}:0]        cluster_base_addr_i,
    # % endif
    # This is to make sure the hartid is exposed to wrapper
    cluster_cfg["tie_ports"] = False

    # Overwrite boot address with base of bootrom
    cluster_cfg["boot_addr"] = occamy_cfg["peripherals"]["rom"]["address"]
    cluster_cfg["cluster_base_expose"] = True
    # # Set the cluster base addr as 0x1000_0000
    # cluster_cfg["cluster_base_addr"] = 268435456
    # # Set the cluster base offset as 0x0004_000
    # cluster_cfg["cluster_base_offset"] = 262144
    # Set the cluster base id as 1
    cluster_cfg["cluster_base_hartid"] = 1
    # Set the enable_debug to false, since we do not need snitch debugging in occamy
    cluster_cfg["enable_debug"] = False
    # Set the vm_support to false, since do not need snitch (core-internal) virtual memory support
    cluster_cfg["vm_support"] = False


def cluster_add_mem(cluster_obj, occamy_cfg):

    # Add Cache to cluster mem
    if "ro_cache_cfg" in occamy_cfg["s1_quadrant"]:
        ro_cache = occamy_cfg["s1_quadrant"]["ro_cache_cfg"]
        ro_tag_width = occamy_cfg["addr_width"] - clog2(
            ro_cache['width'] // 8) - clog2(ro_cache['count']) + 3
        cluster_obj.add_mem(ro_cache["count"],
                            ro_cache["width"],
                            desc="ro cache data",
                            byte_enable=False,
                            speed_optimized=True,
                            density_optimized=True)
        cluster_obj.add_mem(ro_cache["count"],
                            ro_tag_width,
                            desc="ro cache tag",
                            byte_enable=False,
                            speed_optimized=True,
                            density_optimized=True)
    cluster_obj.add_mem(occamy_cfg["spm_narrow"]["length"] // 8,
                        64,
                        desc="SPM Narrow",
                        speed_optimized=False,
                        density_optimized=True)

    cluster_obj.add_mem(occamy_cfg["spm_wide"]["length"] // 64,
                        512,
                        desc="SPM Wide",
                        speed_optimized=False,
                        density_optimized=True)

    # CVA6
    cluster_obj.add_mem(256,
                        128,
                        desc="cva6 data cache array",
                        byte_enable=True,
                        speed_optimized=True,
                        density_optimized=False)
    cluster_obj.add_mem(256,
                        128,
                        desc="cva6 instruction cache array",
                        byte_enable=True,
                        speed_optimized=True,
                        density_optimized=False)
    cluster_obj.add_mem(256,
                        44,
                        desc="cva6 data cache tag",
                        byte_enable=True,
                        speed_optimized=True,
                        density_optimized=False)
    cluster_obj.add_mem(256,
                        45,
                        desc="cva6 instruction cache tag",
                        byte_enable=True,
                        speed_optimized=True,
                        density_optimized=False)
    cluster_obj.add_mem(256,
                        64,
                        desc="cva6 data cache valid and dirty",
                        byte_enable=True,
                        speed_optimized=True,
                        density_optimized=False)


def am_connect_soc_lite_periph_xbar(am, am_soc_axi_lite_periph_xbar, occamy_cfg):
    ########################
    # Periph AXI Lite XBar #
    ########################

    # AM
    nr_axi_lite_peripherals = len(
        occamy_cfg["peripherals"]["axi_lite_peripherals"])
    am_axi_lite_peripherals = []

    for p in range(nr_axi_lite_peripherals):
        if "address" in occamy_cfg["peripherals"]["axi_lite_peripherals"][p]:
            am_axi_lite_peripherals.append(
                am.new_leaf(
                    occamy_cfg["peripherals"]["axi_lite_peripherals"][p]["name"],
                    occamy_cfg["peripherals"]["axi_lite_peripherals"][p]["length"],
                    occamy_cfg["peripherals"]["axi_lite_peripherals"][p]["address"]
                ).attach_to(am_soc_axi_lite_periph_xbar)
            )
    return am_axi_lite_peripherals


def am_connect_soc_lite_narrow_periph_xbar(am, am_soc_axi_lite_narrow_periph_xbar, occamy_cfg):
    ##########################
    # AM: Periph Regbus XBar #
    ##########################
    nr_axi_lite_narrow_peripherals = len(
        occamy_cfg["peripherals"]["axi_lite_narrow_peripherals"])
    am_axi_lite_narrow_peripherals = []

    for p in range(nr_axi_lite_narrow_peripherals):
        am_axi_lite_narrow_peripherals.append(
            am.new_leaf(
                occamy_cfg["peripherals"]["axi_lite_narrow_peripherals"][p]["name"],
                occamy_cfg["peripherals"]["axi_lite_narrow_peripherals"][p]["length"],
                occamy_cfg["peripherals"]["axi_lite_narrow_peripherals"][p]["address"]
            ).attach_to(am_soc_axi_lite_narrow_periph_xbar)
        )
    # add bootrom seperately
    am_bootrom = am.new_leaf(
        "bootrom",
        occamy_cfg["peripherals"]["rom"]["length"],
        occamy_cfg["peripherals"]["rom"]["address"]).attach_to(am_soc_axi_lite_narrow_periph_xbar)
    # add clint seperately
    am_clint = am.new_leaf(
        "clint",
        occamy_cfg["peripherals"]["clint"]["length"],
        occamy_cfg["peripherals"]["clint"]["address"]).attach_to(am_soc_axi_lite_narrow_periph_xbar)
    return am_axi_lite_narrow_peripherals, am_bootrom, am_clint


def am_connect_soc_narrow_xbar_mem(am, am_soc_narrow_xbar, occamy_cfg):
    # Connect narrow SPM to Narrow AXI
    am_spm_narrow = am.new_leaf(
        "spm_narrow",
        occamy_cfg["spm_narrow"]["length"],
        occamy_cfg["spm_narrow"]["address"]).attach_to(am_soc_narrow_xbar)

    ############
    # AM: IDMA #
    ############
    am_sys_idma_cfg = am.new_leaf(
        "sys_idma_cfg",
        occamy_cfg["sys_idma_cfg"]["length"],
        occamy_cfg["sys_idma_cfg"]["address"]).attach_to(am_soc_narrow_xbar)
    
    ##################
    # AM: HeMAiA Mem #
    ##################
    # The MMIO of the ctrl is at
    # (end_addr - 12kB) - end_addr
    # size 12KB
    # cfg grant finish
    am_narrow_hemaia_xdma_ctrl_io = am.new_leaf(
        "hemaia_xdma_ctrl_io",
        4096*3,
        0x100000000 - 4096*3).attach_to(am_soc_narrow_xbar)
    am_narrow_hemaia_xdma_cfg_io = am.new_leaf(
        "hemaia_xdma_cfg_mmio",
        4096,
        0x100000000 - 4096*5).attach_to(am_soc_narrow_xbar)

    return am_spm_narrow, am_sys_idma_cfg, am_narrow_hemaia_xdma_ctrl_io, am_narrow_hemaia_xdma_cfg_io


def am_connect_soc_wide_xbar_mem(am, am_soc_wide_xbar, occamy_cfg):
    # Connect wide SPM to Wide AXI
    am_wide_hemaia_mem = am.new_leaf(
        "spm_wide",
        occamy_cfg["spm_wide"]["length"],
        occamy_cfg["spm_wide"]["address"]).attach_to(am_soc_wide_xbar)
    am_wide_hemaia_xdma_data_io = am.new_leaf(
        "hemaia_xdma_data_io",
        4096,
        0x100000000 - 4096*4).attach_to(am_soc_wide_xbar)
    # Connect wide Zero Memory to Wide AXI
    am_wide_zero_mem = am.new_leaf(
        "wide_zero_mem",
        occamy_cfg["wide_zero_mem"]["length"],
        occamy_cfg["wide_zero_mem"]["address"]).attach_to(am_soc_wide_xbar)
    return am_wide_hemaia_mem, am_wide_hemaia_xdma_data_io, am_wide_zero_mem


def am_connect_soc_wide_xbar_quad(am, am_soc_narrow_xbar, am_wide_xbar_quadrant_s1, am_narrow_xbar_quadrant_s1, occamy_cfg, cluster_generators):
    ##############################
    # AM: Quadrants and Clusters #
    ##############################
    clusters_base_offset = [0]
    clusters_tcdm_size = [0]
    clusters_periph_size = [0]
    clusters_zero_mem_size = [0]

    cluster_base_addr = cluster_generators[0].cfg["cluster_base_addr"]
    nr_s1_clusters = len(cluster_generators)
    nr_s1_quadrants = occamy_cfg["nr_s1_quadrant"]
    for i in range(nr_s1_clusters):
        clusters_base_offset.append(
            cluster_generators[i].cfg["cluster_base_offset"])
        clusters_tcdm_size.append(
            # config is in KiB
            cluster_generators[i].cfg["tcdm"]["size"] * 1024)
        clusters_periph_size.append(
            cluster_generators[i].cfg["cluster_periph_size"] * 1024)
        clusters_zero_mem_size.append(
            cluster_generators[i].cfg["zero_mem_size"] * 1024)

        # assert memory region allocation
        error_str = "ERROR: cluster peripherals, zero memory and tcdm \
                    do not fit into the allocated memory region!!!"
        assert (clusters_tcdm_size[i+1] + clusters_periph_size[i+1] + clusters_zero_mem_size[i+1]) <= \
            clusters_base_offset[i+1], error_str

    quadrant_size = sum(clusters_base_offset)
    for i in range(nr_s1_quadrants):
        cluster_i_start_addr = cluster_base_addr + i * quadrant_size
        am_clusters = list()
        for j in range(nr_s1_clusters):
            bases_cluster = list()
            bases_cluster.append(cluster_i_start_addr +
                                 sum(clusters_base_offset[0:j+1]) + 0)

            # TCDM is accessible from both narrow and wide xbar
            am_clusters.append(
                am.new_leaf(
                    "quadrant_{}_cluster_{}_tcdm".format(i, j),
                    clusters_tcdm_size[j+1],
                    *bases_cluster
                ).attach_to(
                    am_narrow_xbar_quadrant_s1[i]
                ).attach_to(
                    am_wide_xbar_quadrant_s1[i]
                )
            )

            # Cluster peripherals are only accessible from narrow xbar
            bases_cluster = list()
            bases_cluster.append(cluster_i_start_addr + sum(clusters_base_offset[0:j+1])
                                 + clusters_tcdm_size[j+1])
            am_clusters.append(
                am.new_leaf(
                    "quadrant_{}_cluster_{}_periph".format(i, j),
                    clusters_periph_size[j+1],
                    *bases_cluster
                ).attach_to(
                    am_narrow_xbar_quadrant_s1[i]
                )
            )

            # Cluster zero memory are only accessible from wide xbar
            bases_cluster = list()
            bases_cluster.append(cluster_i_start_addr + sum(clusters_base_offset[0:j+1]) +
                                 clusters_tcdm_size[j+1] + clusters_periph_size[j+1])
            am_clusters.append(
                am.new_leaf(
                    "quadrant_{}_cluster_{}_zero_mem".format(i, j),
                    clusters_zero_mem_size[j+1],
                    *bases_cluster
                ).attach_to(
                    am_wide_xbar_quadrant_s1[i]
                )
            )

            # The remaining addresses are reserved for XDMA, and are only accessible from wide xbar
            bases_cluster = list()
            bases_cluster.append(cluster_i_start_addr + sum(clusters_base_offset[0:j+1]) +
                                 clusters_tcdm_size[j+1] + clusters_periph_size[j+1] + clusters_zero_mem_size[j+1])
            am_clusters.append(
                am.new_leaf(
                    "quadrant_{}_cluster_{}_space_after_zero_mem".format(i, j),
                    clusters_base_offset[j+1] - clusters_tcdm_size[j+1] -
                    clusters_periph_size[j+1] - clusters_zero_mem_size[j+1],
                    *bases_cluster
                ).attach_to(
                    am_wide_xbar_quadrant_s1[i]
                )
            )

        am.new_leaf(
            "quad_{}_cfg".format(i),
            occamy_cfg["s1_quadrant"]["cfg_base_offset"],
            occamy_cfg["s1_quadrant"]["cfg_base_addr"] +
            i * occamy_cfg["s1_quadrant"]["cfg_base_offset"]
        ).attach_to(
            am_narrow_xbar_quadrant_s1[i]
        ).attach_to(
            am_soc_narrow_xbar
        )
    return am_clusters


def get_dts(occamy_cfg, am_clint, am_axi_lite_peripherals, am_axi_lite_narrow_peripherals):
    dts = device_tree.DeviceTree()

    #########################
    #  Periph AXI Lite XBar #
    #########################

    nr_axi_lite_peripherals = len(
        occamy_cfg["peripherals"]["axi_lite_peripherals"])
    for p in range(nr_axi_lite_peripherals):
        # add debug module to devicetree
        if occamy_cfg["peripherals"]["axi_lite_peripherals"][p]["name"] == "debug":
            dts.add_device("debug", "riscv,debug-013", am_axi_lite_peripherals[p], [
                "interrupts-extended = <&CPU0_intc 65535>", "reg-names = \"control\""
            ])

    ######################
    # Periph Regbus XBar #
    ######################
    nr_axi_lite_narrow_peripherals = len(
        occamy_cfg["peripherals"]["axi_lite_narrow_peripherals"])
    for p in range(nr_axi_lite_narrow_peripherals):
        # add uart to devicetree
        if occamy_cfg["peripherals"]["axi_lite_narrow_peripherals"][p]["name"] == "uart":
            dts.add_device("serial", "lowrisc,serial", am_axi_lite_narrow_peripherals[p], [
                "clock-frequency = <50000000>", "current-speed = <115200>",
                "interrupt-parent = <&PLIC0>", "interrupts = <1>"
            ])
        # add plic to devicetree
        elif occamy_cfg["peripherals"]["axi_lite_narrow_peripherals"][p]["name"] == "plic":
            dts.add_plic([0], am_axi_lite_narrow_peripherals[p])

    dts.add_clint([0], am_clint)
    dts.add_cpu("eth,ariane")
    htif = dts.add_node("htif", "ucb,htif0")
    dts.add_chosen("stdout-path = \"{}\";".format(htif))

    return dts


def get_top_kwargs(occamy_cfg, cluster_generators, soc_axi_lite_narrow_periph_xbar, soc_wide_xbar, soc_narrow_xbar, soc2router_bus, router2soc_bus, util, name):
    core_per_cluster_list = [cluster_generator.cfg["nr_cores"]
                             for cluster_generator in cluster_generators]
    nr_cores_quadrant = sum(core_per_cluster_list)
    nr_s1_quadrants = occamy_cfg["nr_s1_quadrant"]

    for p in range(len(occamy_cfg["peripherals"]["axi_lite_peripherals"])):
        if occamy_cfg["peripherals"]["axi_lite_peripherals"][p]["name"] == "h2h_mailbox":
            h2h_mailbox_base_addr = occamy_cfg["peripherals"]["axi_lite_peripherals"][p]["address"]
            
    for p in range(len(occamy_cfg["peripherals"]["axi_lite_narrow_peripherals"])):
        if occamy_cfg["peripherals"]["axi_lite_narrow_peripherals"][p]["name"] == "c2h_mailbox":
            c2h_mailbox_base_addr = occamy_cfg["peripherals"]["axi_lite_narrow_peripherals"][p]["address"]
    top_kwargs = {
        "name": name,
        "occamy_cfg": occamy_cfg,
        "soc_axi_lite_narrow_periph_xbar": soc_axi_lite_narrow_periph_xbar,
        "soc_wide_xbar": soc_wide_xbar,
        "soc_narrow_xbar": soc_narrow_xbar,
        "soc2router_bus": soc2router_bus,
        "router2soc_bus": router2soc_bus,
        "cores": nr_s1_quadrants * nr_cores_quadrant + 1,
        "h2h_mailbox_base_addr": util.to_sv_hex(h2h_mailbox_base_addr),
        "c2h_mailbox_base_addr": util.to_sv_hex(c2h_mailbox_base_addr)
    }
    return top_kwargs


def get_soc_kwargs(occamy_cfg, cluster_generators, soc_narrow_xbar, soc_wide_xbar, soc2router_bus, router2soc_bus, util, name):
    core_per_cluster_list = [cluster_generator.cfg["nr_cores"]
                             for cluster_generator in cluster_generators]
    nr_cores_quadrant = sum(core_per_cluster_list)
    nr_s1_quadrants = occamy_cfg["nr_s1_quadrant"]
    soc_kwargs = {
        "name": name,
        "util": util,
        "occamy_cfg": occamy_cfg,
        "soc_narrow_xbar": soc_narrow_xbar,
        "soc_wide_xbar": soc_wide_xbar,
        "soc2router_bus": soc2router_bus,
        "router2soc_bus": router2soc_bus,
        "cores": nr_s1_quadrants * nr_cores_quadrant + 1,
        "nr_s1_quadrants": nr_s1_quadrants,
        "nr_cores_quadrant": nr_cores_quadrant
    }
    return soc_kwargs


def get_quadrant_ctrl_kwargs(occamy_cfg, soc_wide_xbar, soc_narrow_xbar, quadrant_s1_ctrl_xbars, quadrant_s1_ctrl_mux, name):
    num_clusters = len(occamy_cfg["clusters"])
    ro_cache_cfg = occamy_cfg["s1_quadrant"].get("ro_cache_cfg", {})
    ro_cache_regions = ro_cache_cfg.get("address_regions", 1)
    narrow_tlb_cfg = occamy_cfg["s1_quadrant"].get("narrow_tlb_cfg", {})
    narrow_tlb_entries = narrow_tlb_cfg.get("l1_num_entries", 1)
    wide_tlb_cfg = occamy_cfg["s1_quadrant"].get("wide_tlb_cfg", {})
    wide_tlb_entries = wide_tlb_cfg.get("l1_num_entries", 1)

    quadrant_ctrl_kwargs = {
        "name": name,
        "occamy_cfg": occamy_cfg,
        "num_clusters": num_clusters,
        "ro_cache_cfg": ro_cache_cfg,
        "ro_cache_regions": ro_cache_regions,
        "narrow_tlb_cfg": narrow_tlb_cfg,
        "narrow_tlb_entries": narrow_tlb_entries,
        "wide_tlb_cfg": wide_tlb_cfg,
        "wide_tlb_entries": wide_tlb_entries,
        "soc_wide_xbar": soc_wide_xbar,
        "soc_narrow_xbar": soc_narrow_xbar,
        "quadrant_s1_ctrl_xbars": quadrant_s1_ctrl_xbars,
        "quadrant_s1_ctrl_mux": quadrant_s1_ctrl_mux
    }
    return quadrant_ctrl_kwargs


def get_quadrant_kwargs(occamy_cfg, cluster_generators, soc_wide_xbar, soc_narrow_xbar, wide_xbar_quadrant_s1, narrow_xbar_quadrant_s1, name):
    cluster_cfgs = list()
    nr_clusters = len(occamy_cfg["clusters"])
    for i in range(nr_clusters):
        cluster_cfgs.append(cluster_generators[i].cfg)
    quadrant_kwargs = {
        "name": name,
        "occamy_cfg": occamy_cfg,
        "cluster_cfgs": cluster_cfgs,
        "soc_wide_xbar": soc_wide_xbar,
        "soc_narrow_xbar": soc_narrow_xbar,
        "wide_xbar_quadrant_s1": wide_xbar_quadrant_s1,
        "narrow_xbar_quadrant_s1": narrow_xbar_quadrant_s1
    }
    return quadrant_kwargs


def get_quadrant_noc_kwargs(occamy_cfg, cluster_generators):

    # noc parameters
    noc_name = occamy_cfg["s1_quadrant"]["noc_cfg"]["noc_name"]
    routing_algo = occamy_cfg["s1_quadrant"]["noc_cfg"]["routing_algo"]
    noc_array = occamy_cfg["s1_quadrant"]["noc_cfg"]["noc_array"]

    # Check if the noc array size matches with the num of clusters
    num_clusters = len(occamy_cfg["clusters"])
    if num_clusters != noc_array[0]*noc_array[1]:
        raise Exception(
            "The noc array size must match with the num of clusters!")
    # Add one col of the noc_array (since we need the default port)
    x_num = noc_array[0]
    y_num = noc_array[1]
    x_num += 1
    new_noc_array = [x_num, y_num]

    # We take the cluster axi parameters from the first cluster
    cluster_cfg = cluster_generators[0].cfg
    # This is from line 78 of snitch_cluster/hw/snitch_cluster/src/snitch_cluster_wrapper.sv.tpl
    # localparam int unsigned NrMasters = 3;
    cluster_narrow_num_masters = 3
    # This is from line 81 of snitch_cluster/hw/snitch_cluster/src/snitch_cluster_wrapper.sv.tpl
    # localparam int unsigned NrDmaMasters = 3 + ${cfg['nr_hives']};
    cluster_wide_num_masters = 3 + cluster_cfg["nr_hives"]
    # Narrow link
    narrow_data_width = cluster_cfg["data_width"]
    narrow_addr_width = cluster_cfg["addr_width"]
    narrow_in_id_width = cluster_cfg["id_width_in"]
    narrow_out_id_width = narrow_in_id_width + \
        clog2(cluster_narrow_num_masters)
    narrow_user_width = cluster_cfg["user_width"]
    # wide link
    wide_data_width = cluster_cfg["dma_data_width"]
    wide_addr_width = cluster_cfg["addr_width"]
    wide_in_id_width = cluster_cfg["dma_id_width_in"]
    wide_out_id_width = wide_in_id_width + \
        clog2(cluster_wide_num_masters)
    wide_user_width = cluster_cfg["dma_user_width"]
    # cluster
    cluster_base_addr = cluster_cfg["cluster_base_addr"]
    cluster_size = cluster_cfg["cluster_base_offset"]

    quadrant_noc_kwargs = {
        "noc_name": noc_name,
        "noc_array": new_noc_array,
        "routing_algo": routing_algo,
        "narrow_data_width": narrow_data_width,
        "narrow_addr_width": narrow_addr_width,
        "narrow_in_id_width": narrow_in_id_width,
        "narrow_out_id_width": narrow_out_id_width,
        "narrow_user_width": narrow_user_width,
        "wide_data_width": wide_data_width,
        "wide_addr_width": wide_addr_width,
        "wide_in_id_width": wide_in_id_width,
        "wide_out_id_width": wide_out_id_width,
        "wide_user_width": wide_user_width,
        "cluster_base_addr": cluster_base_addr,
        "cluster_size": cluster_size
    }
    return quadrant_noc_kwargs


def get_xilinx_kwargs(occamy_cfg, soc_wide_xbar, soc_axi_lite_narrow_periph_xbar, name):
    xilinx_kwargs = {
        "name": name,
        "occamy_cfg": occamy_cfg,
        "soc_wide_xbar": soc_wide_xbar,
        "soc_axi_lite_narrow_periph_xbar": soc_axi_lite_narrow_periph_xbar
    }
    return xilinx_kwargs


def get_cores_cluster_offset(core_per_cluster):
    # We need the offset for each cores
    # e.g we have three clusters with different cores
    # core_per_cluster = [2, 3, 4]
    # Now we like to assign the hart_base_id for each clusters
    # Ideally it would be
    # hart_base_id_0 = 0 (0,1)->first cluster
    # hart_base_id_1 = 2 (2,3,4) -> second cluster
    # hart_base_id_2 = 5 (5,6,7,8) -> third cluster
    # hence we could first get the running sum of core_per_cluster
    # until end-1 and add extra 0 in the beginning
    running_sum = []
    current_sum = 0
    for core in core_per_cluster:
        current_sum += core
        running_sum.append(current_sum)
    nr_cores_cluster_offset = [0] + running_sum[:-1]
    return nr_cores_cluster_offset


def get_pkg_kwargs(occamy_cfg, cluster_generators, util, name):
    core_per_cluster_list = [cluster_generator.cfg["nr_cores"]
                             for cluster_generator in cluster_generators]
    cluster_cfg = cluster_generators[0].cfg
    nr_cores_cluster_offset = get_cores_cluster_offset(core_per_cluster_list)
    nr_cores_quadrant = sum(core_per_cluster_list)

    core_per_cluster = "{" + ",".join(map(str, core_per_cluster_list)) + "}"
    nr_cores_cluster_offset = "{" + \
        ",".join(map(str, nr_cores_cluster_offset)) + "}"

    pkg_kwargs = {
        "name": name,
        "util": util,
        "addr_width": occamy_cfg["addr_width"],
        "narrow_user_width": cluster_cfg["user_width"],
        "wide_user_width": cluster_cfg["dma_user_width"],
        "nr_clusters_s1_quadrant": len(occamy_cfg["clusters"]),
        "core_per_cluster": core_per_cluster,
        "nr_cores_cluster_offset": nr_cores_cluster_offset,
        "nr_cores_quadrant": nr_cores_quadrant,
        "sram_cfg_fields": cluster_cfg["sram_cfg_fields"],
        "cluster_base_addr": util.to_sv_hex(cluster_cfg["cluster_base_addr"]),
        "cluster_base_offset": util.to_sv_hex(cluster_cfg["cluster_base_offset"]),
        "quad_cfg_base_addr": util.to_sv_hex(occamy_cfg["s1_quadrant"]["cfg_base_addr"]),
        "quad_cfg_base_offset": util.to_sv_hex(occamy_cfg["s1_quadrant"]["cfg_base_offset"]),
        "hemaia_multichip": occamy_cfg["hemaia_multichip"]
    }
    return pkg_kwargs


def get_cva6_kwargs(occamy_cfg, soc_narrow_xbar, name):
    cva6_kwargs = {
        "name": name,
        "occamy_cfg": occamy_cfg,
        "soc_narrow_xbar": soc_narrow_xbar
    }
    return cva6_kwargs


def get_cheader_kwargs(occamy_cfg, cluster_generators, name):
    core_per_cluster_list = [cluster_generator.cfg["nr_cores"]
                             for cluster_generator in cluster_generators]
    if occamy_cfg['hemaia_multichip']['single_chip']:
        nr_chiplets = 1
    else:
        nr_chiplets = occamy_cfg['hemaia_multichip']['nr_chiplets']
        upper_left_coordinate = occamy_cfg['hemaia_multichip']['testbench_cfg']['upper_left_coordinate']
        lower_right_coordinate = occamy_cfg['hemaia_multichip']['testbench_cfg']['lower_right_coordinate']
        delta_x = lower_right_coordinate[0] - upper_left_coordinate[0]
        delta_y = lower_right_coordinate[1] - upper_left_coordinate[1]
        expected_nr_chiplets = (delta_x + 1) * (delta_y + 1)
        if not (nr_chiplets==expected_nr_chiplets):
            raise ValueError(f"The number of chiplets ({nr_chiplets}) does not match the coordinates provided in the testbench_cfg ({upper_left_coordinate} to {lower_right_coordinate}, expected {expected_nr_chiplets} chiplets). Please check your configuration.")
    nr_quads = occamy_cfg['nr_s1_quadrant']
    nr_clusters = len(occamy_cfg["clusters"])
    nr_cores = sum(core_per_cluster_list)
    nr_cores_per_cluster = "{" + \
        ",".join(map(str, core_per_cluster_list)) + "}"
    cluster_offset_list = [cluster_generator.cfg["cluster_base_offset"]
                           for cluster_generator in cluster_generators]
    if not all(cluster_offset == cluster_offset_list[0] for cluster_offset in cluster_offset_list):
        raise ValueError(
            "Not all cluster base offset in the cluster cfg are equal.")
    cluster_offset = cluster_offset_list[0]
    cluster_addr_width = int(math.log2(cluster_offset))
    cluster_base_addr = cluster_generators[0].cfg["cluster_base_addr"]
    cluster_tcdm_size =  cluster_generators[0].cfg["tcdm"]["size"]*1024
    wide_spm_size = occamy_cfg["spm_wide"]["length"]
    narrow_spm_size = occamy_cfg["spm_narrow"]["length"]
    mailbox_size = occamy_cfg["spm_narrow_mailbox_size"]
    cheader_kwargs = {
        "name": name,
        "nr_chiplets": nr_chiplets,
        "nr_quads": nr_quads,
        "nr_clusters": nr_clusters,
        "nr_cores_per_cluster": nr_cores_per_cluster,
        "nr_cores": nr_cores,
        "wide_spm_size": hex(wide_spm_size),
        "narrow_spm_size": hex(narrow_spm_size),
        "mailbox_size": hex(mailbox_size),
        "cluster_tcdm_size": hex(cluster_tcdm_size),
        "cluster_offset": hex(cluster_offset),
        "cluster_addr_width": cluster_addr_width,
        "cluster_base_addr": hex(cluster_base_addr)
    }
    return cheader_kwargs


def get_bootdata_kwargs(occamy_cfg, cluster_generators, name):
    # We use the 1st cluster cfg as the template to assign
    #   tcdm start
    #   tcdm size
    #   tcdm offset
    # We only have 1 quadrant right now so we just sum up the nr_cores within 1 quad
    core_per_cluster_list = [cluster_generator.cfg["nr_cores"]
                             for cluster_generator in cluster_generators]
    nr_cores_quadrant = sum(core_per_cluster_list)

    cluster_cfg = cluster_generators[0].cfg

    bootdata_kwargs = {
        "name": name,
        "boot_addr": occamy_cfg["peripherals"]["rom"]["address"],
        "core_count": nr_cores_quadrant,
        "hart_id_base": 1,
        "addr_width": occamy_cfg["addr_width"],
        "chip_id_width": occamy_cfg["hemaia_multichip"]["chip_id_width"],
        "single_chip": occamy_cfg["hemaia_multichip"]["single_chip"],
        "single_chip_id": occamy_cfg["hemaia_multichip"]["single_chip_id"],
        "tcdm_start": cluster_cfg["cluster_base_addr"],
        "tcdm_size": cluster_cfg["tcdm"]["size"]*1024,
        "tcdm_offset": cluster_cfg["cluster_base_offset"],
        "global_mem_start": occamy_cfg["spm_wide"]["address"],
        "global_mem_end": occamy_cfg["spm_wide"]["address"] + occamy_cfg["spm_wide"]["length"],
        "cluster_count": len(occamy_cfg["clusters"]),
        "s1_quadrant_count": occamy_cfg["nr_s1_quadrant"],
        "clint_base": occamy_cfg["peripherals"]["clint"]["address"]
    }
    return bootdata_kwargs


def get_testharness_kwargs(soc_wide_xbar, soc_axi_lite_narrow_periph_xbar, chip_id, solder, name):
    testharness_kwargs = {
        "name": name,
        "solder": solder,
        "chip_id": chip_id,
        "soc_wide_xbar": soc_wide_xbar,
        "soc_axi_lite_narrow_periph_xbar": soc_axi_lite_narrow_periph_xbar
    }
    return testharness_kwargs


def get_multichip_testharness_kwargs(occamy_cfg, soc2router_bus, router2soc_bus, name):
    testharness_kwargs = {
        "name": name,
        "mem_macro": False,
        "netlist": False,
        "multichip_cfg": occamy_cfg["hemaia_multichip"],
        "soc2router_bus": soc2router_bus,
        "router2soc_bus": router2soc_bus,
        "mem_size": occamy_cfg["spm_wide"]["length"],
        "mem_bank": occamy_cfg["spm_wide"]["banks"],
    }
    return testharness_kwargs

def get_mem_macro_testharness_kwargs(occamy_cfg, soc2router_bus, router2soc_bus, name):
    testharness_kwargs = {
        "name": name,
        "mem_macro": True,
        "netlist": False,
        "multichip_cfg": occamy_cfg["hemaia_multichip"],
        "soc2router_bus": soc2router_bus,
        "router2soc_bus": router2soc_bus,
        "mem_size": occamy_cfg["spm_wide"]["length"],
        "mem_bank": occamy_cfg["spm_wide"]["banks"],
    }
    return testharness_kwargs

def get_netlist_testharness_kwargs(occamy_cfg, soc2router_bus, router2soc_bus, name):
    testharness_kwargs = {
        "name": name,
        "mem_macro": False,
        "netlist": True,
        "multichip_cfg": occamy_cfg["hemaia_multichip"],
        "soc2router_bus": soc2router_bus,
        "router2soc_bus": router2soc_bus,
        "mem_size": occamy_cfg["spm_wide"]["length"],
        "mem_bank": occamy_cfg["spm_wide"]["banks"],
    }
    return testharness_kwargs

def get_chip_kwargs(soc_wide_xbar, soc_narrow_xbar, soc_axi_lite_narrow_periph_xbar, soc2router_bus, router2soc_bus, occamy_cfg, cluster_generators, util, name):
    core_per_cluster_list = [cluster_generator.cfg["nr_cores"]
                             for cluster_generator in cluster_generators]
    nr_cores_quadrant = sum(core_per_cluster_list)
    nr_s1_quadrants = occamy_cfg["nr_s1_quadrant"]
    chip_kwargs = {
        "name": name,
        "util": util,
        "occamy_cfg": occamy_cfg,
        "cluster_address_space": cluster_generators[0].cfg["cluster_base_offset"],
        "soc_wide_xbar": soc_wide_xbar,
        "soc_narrow_xbar": soc_narrow_xbar,
        "soc_axi_lite_narrow_periph_xbar": soc_axi_lite_narrow_periph_xbar,
        "soc2router_bus": soc2router_bus,
        "router2soc_bus": router2soc_bus,
        "cores": nr_s1_quadrants * nr_cores_quadrant + 1
    }
    return chip_kwargs

def get_io_kwargs(occamy_cfg, util, name):
    io_kwargs = {
        "name": name,
        "util": util,
        "occamy_cfg": occamy_cfg,
        "multichip_cfg": occamy_cfg["hemaia_multichip"]
    }
    return io_kwargs

def get_ctrl_kwargs(occamy_cfg, cluster_generators, name):
    default_boot_addr = occamy_cfg["peripherals"]["rom"]["address"]
    backup_boot_addr = occamy_cfg["backup_boot_addr"]
    addr_width = occamy_cfg["addr_width"]
    # core_per_cluster_list = [cluster_generator.cfg["nr_cores"]
    #                          for cluster_generator in cluster_generators]
    # nr_cores_quadrant = sum(core_per_cluster_list)
    hex_default_boot_addr = hex(default_boot_addr)
    hex_backup_boot_addr = hex(backup_boot_addr)
    # Remove the prefix 0x
    hex_default_boot_addr = hex_default_boot_addr[2:]
    hex_backup_boot_addr = hex_backup_boot_addr[2:]
    ctrl_kwargs = {
        "name": name,
        "nr_s1_quadrants": occamy_cfg["nr_s1_quadrant"],
        "nr_clusters": len(occamy_cfg["clusters"]),
        # "nr_cores": nr_cores_quadrant,
        "default_boot_addr": hex_default_boot_addr,
        "backup_boot_addr": hex_backup_boot_addr,
        "occamy_cfg": occamy_cfg,
        "addr_width": addr_width
    }
    return ctrl_kwargs
