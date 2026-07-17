#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

# BEGIN WORKLOAD DESCRIPTION AND TASK GRAPH
# DMA-only version of the M-split GEMM data movement. It loads A tiles on each
# chiplet, loads B on chip00, broadcasts B, and checks the local A/B buffers.
#
# Task dependency graph:
#
# Chip00:
#   Load_A1_Chip00 -> Load_B_Chip00
#   Load_B_Chip00 -> Check_A1_Chip00 -> Check_B_Chip00
#   Load_B_Chip00 -> Broadcast_B_Chip00
#
# Remote chiplet A lanes:
#   chip01: Load_A2_Chip01 -> Check_A2_Chip01 -> Check_B_Chip01
#   chip10: Load_A3_Chip10 -> Check_A3_Chip10 -> Check_B_Chip10
#   chip11: Load_A4_Chip11 -> Check_A4_Chip11 -> Check_B_Chip11
#
# Broadcast fanout:
#   Broadcast_B_Chip00 -> Check_B_Chip01
#   Broadcast_B_Chip00 -> Check_B_Chip10
#   Broadcast_B_Chip00 -> Check_B_Chip11
# END WORKLOAD DESCRIPTION AND TASK GRAPH

import os
import sys
import argparse
import pathlib
import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)

print(f"ROOT_DIR: {ROOT_DIR}")
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")

# Import emit_header_file from gemm_multi_chiplet_datagen to emit the A/B data header.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gemm_multi_chiplet_datagen import emit_header_file  # noqa E402

from bingo_dfg import BingoDFG
from bingo_platform import (
    parse_platform_cfg,
    guard_cluster_count,
    guard_chiplet_count,
)
from bingo_node import BingoNode
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol, BingoMemFixedAddr
from bingo_kernel_args import (
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelIdmaBroadcastArgs,
    SnaxBingoKernelXdma1dCopyArgs,
    HostBingoKernelCheckResultArgs,
)
from bingo_helpers import chiplet_addr_transform_loc

def get_args():
    parser = argparse.ArgumentParser(description="Bingo HW Manager")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--output_offload_file_name",
        type=str,
        default="offload_bingo_hw.h",
        help="Output filename for the offload header file",
    )
    parser.add_argument(
        "-c",
        "--cfg",
        type=pathlib.Path,
        required=True,
        help="Select param config file (params.hjson)",
    )
    parser.add_argument(
        "--hwcfg",
        type=pathlib.Path,
        required=True,
        help="Select hardware config file",
    )
    parser.add_argument(
        "--platformcfg",
        type=pathlib.Path,
        required=True,
        help="Path to generated occamy.h with HW platform defines",
    )
    parser.add_argument(
        "--data_h",
        type=pathlib.Path,
        default=None,
        help="Output path for the generated data header (e.g. dma_for_gemm_data.h). If omitted, data header is not written.",
    )
    return parser.parse_args()

def define_workload_params(cfg_path, hwcfg_path):
    """Load workload params and derive DMA buffer sizes from the HW config."""
    with open(cfg_path) as f:
        param = hjson.loads(f.read())
    with open(hwcfg_path) as f:
        hw = hjson.loads(f.read())
    merged = {**param, **hw}

    # Derive meshRow/tileSize/meshCol from the hw config
    data_type = 0  # int8
    array_shape = merged["array_shape"]
    snax_acc_cfg = merged["snax_versacore_core_template"]["snax_acc_cfg"][0]
    unrolling = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]
    meshRow  = unrolling[0]
    tileSize = unrolling[1]
    meshCol  = unrolling[2]

    M = merged["M"]
    K = merged["K"]
    N = merged["N"]

    params = {
        "M": M,
        "K": K,
        "N": N,
        "meshRow":  meshRow,
        "tileSize": tileSize,
        "meshCol":  meshCol,
        "arrayShapeIdx": array_shape,
    }
    params["app_name"] = "DMA for GEMM M-split"
    # Derived sizes
    params["Atile_size"] = M * K * meshRow * tileSize * 1   # int8
    params["B_size"]     = K * N * tileSize * meshCol * 1   # int8

    # The hardcoded MemPool address for A1, A2, A3, A4, B
    main_mem_base_addr = 0x8000_0000
    mem_pool_chip_loc = {"x": 2, "y": 0}  # Chiplet at (2,0)
    mem_pool_base_addr = chiplet_addr_transform_loc(
        mem_pool_chip_loc["x"], mem_pool_chip_loc["y"], main_mem_base_addr
    )
    params["A1_mp_base_addr"] = mem_pool_base_addr + 0 * params["Atile_size"]
    params["A2_mp_base_addr"] = mem_pool_base_addr + 1 * params["Atile_size"]
    params["A3_mp_base_addr"] = mem_pool_base_addr + 2 * params["Atile_size"]
    params["A4_mp_base_addr"] = mem_pool_base_addr + 3 * params["Atile_size"]
    params["B_mp_base_addr"]  = mem_pool_base_addr + 4 * params["Atile_size"]
    return params, merged


def define_memory_handles(params):
    """Defines memory symbols and handles."""
    mem_handles = {}
    # 1. Define Memory Fixed Addresses
    # The MemFixedAddr are the hardcoded MemPool addresses for A1, A2, A3, A4, B
    
    mem_handles["A1_mp"] = BingoMemFixedAddr(params["A1_mp_base_addr"])
    mem_handles["A2_mp"] = BingoMemFixedAddr(params["A2_mp_base_addr"])
    mem_handles["A3_mp"] = BingoMemFixedAddr(params["A3_mp_base_addr"])
    mem_handles["A4_mp"] = BingoMemFixedAddr(params["A4_mp_base_addr"])
    mem_handles["B_mp"] = BingoMemFixedAddr(params["B_mp_base_addr"])
    
    # 2. Define Memory Symbols
    # The MemSymbol are the variables defined in the data.h file which the memory location is already known at compile time
    # every chip will have its own golden A tile and golden B in L3 for checking results
    mem_handles["A1_golden_l3"] = BingoMemSymbol("A_data_L3", offset=0)
    mem_handles["A2_golden_l3"] = BingoMemSymbol("A_data_L3", offset=1*params["Atile_size"])
    mem_handles["A3_golden_l3"] = BingoMemSymbol("A_data_L3", offset=2*params["Atile_size"])
    mem_handles["A4_golden_l3"] = BingoMemSymbol("A_data_L3", offset=3*params["Atile_size"])
    mem_handles["B_golden_l3"] = BingoMemSymbol("B_data_L3")

    # 3. Define Memory Handles (Dynamic Allocations)
    # Prepare the per-chiplet A and B L1 buffers.

    ###########################
    # for chip00
    ###########################
    mem_handles["A_l1_chip00"] = BingoMemAlloc(
        "A_l1_chip00",
        size=params["Atile_size"],
        mem_level="L1",
        chip_id=0x00,
        cluster_id=0,
    )
    mem_handles["B_l1_chip00"] = BingoMemAlloc(
        "B_l1_chip00", size=params["B_size"], mem_level="L1", chip_id=0x00, cluster_id=0
    )

    ###########################
    # for chip01
    ###########################
    mem_handles["A_l1_chip01"] = BingoMemAlloc(
        "A_l1_chip01",
        size=params["Atile_size"],
        mem_level="L1",
        chip_id=0x01,
        cluster_id=0,
    )
    mem_handles["B_l1_chip01"] = BingoMemAlloc(
        "B_l1_chip01", size=params["B_size"], mem_level="L1", chip_id=0x01, cluster_id=0
    )

    ###########################
    # for chip10
    ###########################
    mem_handles["A_l1_chip10"] = BingoMemAlloc(
        "A_l1_chip10",
        size=params["Atile_size"],
        mem_level="L1",
        chip_id=0x10,
        cluster_id=0,
    )
    mem_handles["B_l1_chip10"] = BingoMemAlloc(
        "B_l1_chip10", size=params["B_size"], mem_level="L1", chip_id=0x10, cluster_id=0
    )

    ###########################
    # for chip11
    ###########################
    mem_handles["A_l1_chip11"] = BingoMemAlloc(
        "A_l1_chip11",
        size=params["Atile_size"],
        mem_level="L1",
        chip_id=0x11,
        cluster_id=0,
    )
    mem_handles["B_l1_chip11"] = BingoMemAlloc(
        "B_l1_chip11", size=params["B_size"], mem_level="L1", chip_id=0x11, cluster_id=0
    )

    return mem_handles

def create_dfg(params, mem_handles, platform, eval_case):
    """Creates the Bingo Data Flow Graph with nodes and dependencies."""
    bingo_dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )
    dma_core_id = 1  # Core 1 for Load
    host_core_id = 2  # Core 2 for host-side checks

    chiplets = platform["chiplet_ids"]
    a_indices = {chiplet: i + 1 for i, chiplet in enumerate(chiplets)}

    #######################################################
    ########################baseline dma###################
    #####no broadcast, no dual-dma, no half-duplex#########
    #######################################################

    if eval_case == 0:
        for chiplet in chiplets:
            a_idx = a_indices[chiplet]
            chiplet_hex = f"{chiplet:02x}"

            load_a = BingoNode(
                assigned_chiplet_id=chiplet,
                assigned_cluster_id=0,
                assigned_core_id=dma_core_id,
                node_name=f"Load_A{a_idx}_Chip{chiplet_hex}",
                kernel_name="__snax_bingo_kernel_idma_1d_copy",
                kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                    src_addr=mem_handles[f"A{a_idx}_mp"],
                    dst_addr=mem_handles[f"A_l1_chip{chiplet_hex}"],
                    size=params["Atile_size"],
                ),
            )
            load_b = BingoNode(
                assigned_chiplet_id=chiplet,
                assigned_cluster_id=0,
                assigned_core_id=dma_core_id,
                node_name=f"Load_B_Chip{chiplet_hex}",
                kernel_name="__snax_bingo_kernel_idma_1d_copy",
                kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                    src_addr=mem_handles["B_mp"],
                    dst_addr=mem_handles[f"B_l1_chip{chiplet_hex}"],
                    size=params["B_size"],
                ),
            )
            check_a = BingoNode(
                assigned_chiplet_id=chiplet,
                assigned_cluster_id=0,
                assigned_core_id=host_core_id,
                node_name=f"Check_A{a_idx}_Chip{chiplet_hex}",
                kernel_name="__host_bingo_kernel_check_result",
                kernel_args=HostBingoKernelCheckResultArgs(
                    golden_data_addr=mem_handles[f"A{a_idx}_golden_l3"],
                    output_data_addr=mem_handles[f"A_l1_chip{chiplet_hex}"],
                    data_size=params["Atile_size"],
                ),
            )
            check_b = BingoNode(
                assigned_chiplet_id=chiplet,
                assigned_cluster_id=0,
                assigned_core_id=host_core_id,
                node_name=f"Check_B_Chip{chiplet_hex}",
                kernel_name="__host_bingo_kernel_check_result",
                kernel_args=HostBingoKernelCheckResultArgs(
                    golden_data_addr=mem_handles["B_golden_l3"],
                    output_data_addr=mem_handles[f"B_l1_chip{chiplet_hex}"],
                    data_size=params["B_size"],
                ),
            )

            for node in [load_a, check_a, load_b, check_b]:
                bingo_dfg.bingo_add_node(node)

            # sequentially load A then load B, and check A then check B.
            # four chiplet work in parrellel, but the dma request is never optimized
            bingo_dfg.add_edge(load_a, load_b)
            bingo_dfg.add_edge(load_b, check_a)
            bingo_dfg.add_edge(check_a, check_b)

    elif eval_case == 1:
        #######################################################
        ########################baseline dma###################
        #####with broadcast, no dual-dma, no half-duplex#######
        #######################################################

        #######################################################
        # chip 00
        #######################################################
        # Load A1
        node_chiplet_00_load_A1 = BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=dma_core_id,
            node_name="Load_A1_Chip00",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_handles["A1_mp"],
                dst_addr=mem_handles["A_l1_chip00"],
                size=params["Atile_size"],
            ),
        )
        # Load B
        node_chiplet_00_load_B = BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=dma_core_id,
            node_name="Load_B_Chip00",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_handles["B_mp"],
                dst_addr=mem_handles["B_l1_chip00"],
                size=params["B_size"],
            ),
        )

        # Broadcast B
        # use idma to do broadcast for each chiplet
        node_chiplet_00_broadcast_B = BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=dma_core_id,
            node_name="Broadcast_B_Chip00",
            kernel_name="__snax_bingo_kernel_idma_broadcast",
            kernel_args=SnaxBingoKernelIdmaBroadcastArgs(
                src_addr=mem_handles["B_l1_chip00"],
                dst_addr=mem_handles["B_l1_chip00"],  # Broadcast to all chiplets at this address
                size=params["B_size"],
            ),
        )
        # Check A1
        node_chiplet_00_check_A1 = BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Check_A1_Chip00",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles["A1_golden_l3"],
                output_data_addr=mem_handles["A_l1_chip00"],
                data_size=64,
            ),
        )
        # Check B
        node_chiplet_00_check_B = BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Check_B_Chip00",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles["B_golden_l3"],
                output_data_addr=mem_handles["B_l1_chip00"],
                data_size=64,
            ),
        )

        #######################################################
        # chip 01
        #######################################################
        # Load A2
        node_chiplet_01_load_A2 = BingoNode(
            assigned_chiplet_id=0x01,
            assigned_cluster_id=0,
            assigned_core_id=dma_core_id,
            node_name="Load_A2_Chip01",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_handles["A2_mp"],
                dst_addr=mem_handles["A_l1_chip01"],
                size=params["Atile_size"],
            ),
        )
        # Check A2
        node_chiplet_01_check_A2 = BingoNode(
            assigned_chiplet_id=0x01,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Check_A2_Chip01",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles["A2_golden_l3"],
                output_data_addr=mem_handles["A_l1_chip01"],
                data_size=64,
            ),
        )
        # Check B (after broadcast)
        node_chiplet_01_check_B = BingoNode(
            assigned_chiplet_id=0x01,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Check_B_Chip01",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles["B_golden_l3"],
                output_data_addr=mem_handles["B_l1_chip01"],
                data_size=64,
            ),
        )

        #######################################################
        # chip 10
        #######################################################
        # Load A3
        node_chiplet_10_load_A3 = BingoNode(
            assigned_chiplet_id=0x10,
            assigned_cluster_id=0,
            assigned_core_id=dma_core_id,
            node_name="Load_A3_Chip10",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_handles["A3_mp"],
                dst_addr=mem_handles["A_l1_chip10"],
                size=params["Atile_size"],
            ),
        )
        # Check A3
        node_chiplet_10_check_A3 = BingoNode(
            assigned_chiplet_id=0x10,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Check_A3_Chip10",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles["A3_golden_l3"],
                output_data_addr=mem_handles["A_l1_chip10"],
                data_size=64,
            ),
        )
        # Check B (after broadcast)
        node_chiplet_10_check_B = BingoNode(
            assigned_chiplet_id=0x10,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Check_B_Chip10",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles["B_golden_l3"],
                output_data_addr=mem_handles["B_l1_chip10"],
                data_size=64,
            ),
        )

        #######################################################
        # chip 11
        #######################################################
        # Load A4
        node_chiplet_11_load_A4 = BingoNode(
            assigned_chiplet_id=0x11,
            assigned_cluster_id=0,
            assigned_core_id=dma_core_id,
            node_name="Load_A4_Chip11",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_handles["A4_mp"],
                dst_addr=mem_handles["A_l1_chip11"],
                size=params["Atile_size"],
            ),
        )
        # Check A4
        node_chiplet_11_check_A4 = BingoNode(
            assigned_chiplet_id=0x11,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Check_A4_Chip11",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles["A4_golden_l3"],
                output_data_addr=mem_handles["A_l1_chip11"],
                data_size=64,
            ),
        )
        # Check B (after broadcast)
        node_chiplet_11_check_B = BingoNode(
            assigned_chiplet_id=0x11,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Check_B_Chip11",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles["B_golden_l3"],
                output_data_addr=mem_handles["B_l1_chip11"],
                data_size=64,
            ),
        )

        for node in [
            node_chiplet_00_load_A1,
            node_chiplet_00_check_A1,
            node_chiplet_00_load_B,
            node_chiplet_00_check_B,
            node_chiplet_00_broadcast_B,
            node_chiplet_01_load_A2,
            node_chiplet_01_check_A2,
            node_chiplet_01_check_B,
            node_chiplet_10_load_A3,
            node_chiplet_10_check_A3,
            node_chiplet_10_check_B,
            node_chiplet_11_load_A4,
            node_chiplet_11_check_A4,
            node_chiplet_11_check_B,
        ]:
            bingo_dfg.bingo_add_node(node)

        # firstly load a then load b.
        bingo_dfg.add_edge(node_chiplet_00_load_A1, node_chiplet_00_load_B)
        # then broadcast b.
        bingo_dfg.add_edge(node_chiplet_00_load_B, node_chiplet_00_broadcast_B)
        # check result
        bingo_dfg.add_edge(node_chiplet_00_load_B, node_chiplet_00_check_A1)
        bingo_dfg.add_edge(node_chiplet_00_check_A1, node_chiplet_00_check_B)

        # chip 01
        bingo_dfg.add_edge(node_chiplet_01_load_A2, node_chiplet_01_check_A2)
        bingo_dfg.add_edge(node_chiplet_00_broadcast_B, node_chiplet_01_check_B)
        bingo_dfg.add_edge(node_chiplet_01_check_A2, node_chiplet_01_check_B)

        # chip 10
        bingo_dfg.add_edge(node_chiplet_10_load_A3, node_chiplet_10_check_A3)
        bingo_dfg.add_edge(node_chiplet_00_broadcast_B, node_chiplet_10_check_B)
        bingo_dfg.add_edge(node_chiplet_10_check_A3, node_chiplet_10_check_B)

        # chip 11
        bingo_dfg.add_edge(node_chiplet_11_load_A4, node_chiplet_11_check_A4)
        bingo_dfg.add_edge(node_chiplet_00_broadcast_B, node_chiplet_11_check_B)
        bingo_dfg.add_edge(node_chiplet_11_check_A4, node_chiplet_11_check_B)

    elif eval_case == 2:
        #######################################################
        ########################baseline dma###################
        #####with broadcast, with dual-dma, no half-duplex#####
        #######################################################
        #######################################################
        # chip 00
        #######################################################
        # Load B using idma
        node_chiplet_00_load_B = BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=dma_core_id,
            node_name="Load_B_Chip00",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_handles["B_mp"],
                dst_addr=mem_handles["B_l1_chip00"],
                size=params["B_size"],
            ),
        )

        # Broadcast B using idma
        # use idma to do broadcast for each chiplet
        node_chiplet_00_broadcast_B = BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=dma_core_id,
            node_name="Broadcast_B_Chip00",
            kernel_name="__snax_bingo_kernel_idma_broadcast",
            kernel_args=SnaxBingoKernelIdmaBroadcastArgs(
                src_addr=mem_handles["B_l1_chip00"],
                dst_addr=mem_handles["B_l1_chip00"],  # Broadcast to all chiplets at this address
                size=params["B_size"],
            ),
        )

        # Load A1 using xdma
        node_chiplet_00_load_A1 = BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=dma_core_id,
            node_name="Load_A1_Chip00",
            kernel_name="__snax_bingo_kernel_xdma_1d_copy",
            kernel_args=SnaxBingoKernelXdma1dCopyArgs(
                src_addr=mem_handles["A1_mp"],
                dst_addr=mem_handles["A_l1_chip00"],
                size=params["Atile_size"],
            ),
        )

        # Check A1
        node_chiplet_00_check_A1 = BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Check_A1_Chip00",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles["A1_golden_l3"],
                output_data_addr=mem_handles["A_l1_chip00"],
                data_size=64,
            ),
        )
        # Check B
        node_chiplet_00_check_B = BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Check_B_Chip00",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles["B_golden_l3"],
                output_data_addr=mem_handles["B_l1_chip00"],
                data_size=64,
            ),
        )

        #######################################################
        # chip 01
        #######################################################
        # Load A2
        node_chiplet_01_load_A2 = BingoNode(
            assigned_chiplet_id=0x01,
            assigned_cluster_id=0,
            assigned_core_id=dma_core_id,
            node_name="Load_A2_Chip01",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_handles["A2_mp"],
                dst_addr=mem_handles["A_l1_chip01"],
                size=params["Atile_size"],
            ),
        )
        # Check A2
        node_chiplet_01_check_A2 = BingoNode(
            assigned_chiplet_id=0x01,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Check_A2_Chip01",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles["A2_golden_l3"],
                output_data_addr=mem_handles["A_l1_chip01"],
                data_size=64,
            ),
        )
        # Check B (after broadcast)
        node_chiplet_01_check_B = BingoNode(
            assigned_chiplet_id=0x01,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Check_B_Chip01",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles["B_golden_l3"],
                output_data_addr=mem_handles["B_l1_chip01"],
                data_size=64,
            ),
        )

        #######################################################
        # chip 10
        #######################################################
        # Load A3
        node_chiplet_10_load_A3 = BingoNode(
            assigned_chiplet_id=0x10,
            assigned_cluster_id=0,
            assigned_core_id=dma_core_id,
            node_name="Load_A3_Chip10",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_handles["A3_mp"],
                dst_addr=mem_handles["A_l1_chip10"],
                size=params["Atile_size"],
            ),
        )
        # Check A3
        node_chiplet_10_check_A3 = BingoNode(
            assigned_chiplet_id=0x10,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Check_A3_Chip10",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles["A3_golden_l3"],
                output_data_addr=mem_handles["A_l1_chip10"],
                data_size=64,
            ),
        )
        # Check B (after broadcast)
        node_chiplet_10_check_B = BingoNode(
            assigned_chiplet_id=0x10,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Check_B_Chip10",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles["B_golden_l3"],
                output_data_addr=mem_handles["B_l1_chip10"],
                data_size=64,
            ),
        )

        #######################################################
        # chip 11
        #######################################################
        # Load A4
        node_chiplet_11_load_A4 = BingoNode(
            assigned_chiplet_id=0x11,
            assigned_cluster_id=0,
            assigned_core_id=dma_core_id,
            node_name="Load_A4_Chip11",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_handles["A4_mp"],
                dst_addr=mem_handles["A_l1_chip11"],
                size=params["Atile_size"],
            ),
        )
        # Check A4
        node_chiplet_11_check_A4 = BingoNode(
            assigned_chiplet_id=0x11,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Check_A4_Chip11",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles["A4_golden_l3"],
                output_data_addr=mem_handles["A_l1_chip11"],
                data_size=64,
            ),
        )
        # Check B (after broadcast)
        node_chiplet_11_check_B = BingoNode(
            assigned_chiplet_id=0x11,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Check_B_Chip11",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles["B_golden_l3"],
                output_data_addr=mem_handles["B_l1_chip11"],
                data_size=64,
            ),
        )

        for node in [
            node_chiplet_00_load_A1,
            node_chiplet_00_check_A1,
            node_chiplet_00_load_B,
            node_chiplet_00_check_B,
            node_chiplet_00_broadcast_B,
            node_chiplet_01_load_A2,
            node_chiplet_01_check_A2,
            node_chiplet_01_check_B,
            node_chiplet_10_load_A3,
            node_chiplet_10_check_A3,
            node_chiplet_10_check_B,
            node_chiplet_11_load_A4,
            node_chiplet_11_check_A4,
            node_chiplet_11_check_B,
        ]:
            bingo_dfg.bingo_add_node(node)

        # firstly load b then load a.
        bingo_dfg.add_edge(node_chiplet_00_load_B, node_chiplet_00_load_A1)
        # then broadcast b after load a. to check if the idma and xdma can use the full axi bw (r/w channel)
        bingo_dfg.add_edge(node_chiplet_00_load_B, node_chiplet_00_broadcast_B)
        # check result
        bingo_dfg.add_edge(node_chiplet_00_load_A1, node_chiplet_00_check_A1)
        bingo_dfg.add_edge(node_chiplet_00_check_A1, node_chiplet_00_check_B)

        # chip 01
        bingo_dfg.add_edge(node_chiplet_01_load_A2, node_chiplet_01_check_A2)
        bingo_dfg.add_edge(node_chiplet_00_broadcast_B, node_chiplet_01_check_B)
        bingo_dfg.add_edge(node_chiplet_01_check_A2, node_chiplet_01_check_B)

        # chip 10
        bingo_dfg.add_edge(node_chiplet_10_load_A3, node_chiplet_10_check_A3)
        bingo_dfg.add_edge(node_chiplet_00_broadcast_B, node_chiplet_10_check_B)
        bingo_dfg.add_edge(node_chiplet_10_check_A3, node_chiplet_10_check_B)

        # chip 11
        bingo_dfg.add_edge(node_chiplet_11_load_A4, node_chiplet_11_check_A4)
        bingo_dfg.add_edge(node_chiplet_00_broadcast_B, node_chiplet_11_check_B)
        bingo_dfg.add_edge(node_chiplet_11_check_A4, node_chiplet_11_check_B)


    elif eval_case == 3:
        #######################################################
        ########################baseline dma###################
        #####with broadcast, with dual-dma, with half-duplex###
        #######################################################


        pass

    else:
        raise ValueError(f"Unsupported eval_case: {eval_case}")

    return bingo_dfg

def main():
    args = get_args()
    output_dir = args.output_dir
    output_file_name = args.output_offload_file_name
    print(f"Output DIR: {output_dir}")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Execute Pipeline
    params, merged_config = define_workload_params(args.cfg, args.hwcfg)

    # Emit the A/B data header and mempool.bin.
    # out_dir is passed so datagen can also write mempool.bin there
    if args.data_h is not None:
        build_dir = os.path.join(output_dir, "build")
        data_h_content = emit_header_file(**merged_config, out_dir=build_dir)
        with open(args.data_h, "w") as f:
            f.write(data_h_content)
        print(f"Written data header: {args.data_h}")

    mem_handles = define_memory_handles(params)
    platform = parse_platform_cfg(args.platformcfg)
    if not guard_chiplet_count(merged_config, platform, args.output_dir, args.output_offload_file_name):
        return
    if not guard_cluster_count(merged_config, platform, args.output_dir, args.output_offload_file_name):
        return
    dfg = create_dfg(params, mem_handles, platform, eval_case=1)
    data_header_name = os.path.basename(args.data_h) if args.data_h is not None else "dma_for_gemm_data.h"
    dfg.bingo_compile_dfg(params["app_name"], output_dir, output_file_name, extra_include_header_list=[data_header_name])


if __name__ == "__main__":
    main()
