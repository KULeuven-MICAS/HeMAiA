#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

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

# Import emit_header_file from gemm_multi_chiplet_datagen to emit gemm_data.h directly
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
    SnaxBingoKernelGemmFullArgs,
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
        help="Output path for the generated data header (e.g. gemm_data.h). If omitted, data header is not written.",
    )
    return parser.parse_args()

def define_workload_params(cfg_path, hwcfg_path):
    """Load workload params from hjson config files.
    meshRow/tileSize/meshCol are derived from the hw config.
    Returns (params dict, merged_config dict).
    """
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
        "transposeA": merged.get("transposed_A", 0),
        "transposeB": merged.get("transposed_B", 0),
        "accumPrevC": merged.get("accumPrevC",  0),
        # DVFS stress: number of sequential gemm super-stages per chiplet.
        "producer_stages": merged.get("producer_stages", 6),
        "consumer_stages": merged.get("consumer_stages", 4),
    }
    params["app_name"] = "Multi-Chip DVFS Cross-Chiplet Trigger"
    # Derived sizes
    params["Atile_size"] = M * K * meshRow * tileSize * 1   # int8
    params["B_size"]     = K * N * tileSize * meshCol * 1   # int8
    params["C_size"]     = M * N * meshRow * meshCol * 4    # int32
    params["D_size"]     = M * N * meshRow * meshCol * 4    # int32

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
    # every chip will have its own golden A tile and golden B in L3, and golden D in L3 for checking results
    mem_handles["A1_golden_l3"] = BingoMemSymbol("A_data_L3", offset=0)
    mem_handles["A2_golden_l3"] = BingoMemSymbol("A_data_L3", offset=1*params["Atile_size"])
    mem_handles["A3_golden_l3"] = BingoMemSymbol("A_data_L3", offset=2*params["Atile_size"])
    mem_handles["A4_golden_l3"] = BingoMemSymbol("A_data_L3", offset=3*params["Atile_size"])
    mem_handles["B_golden_l3"] = BingoMemSymbol("B_data_L3")

    mem_handles["D1_golden_l3"] = BingoMemSymbol("D_data_L3", offset=0)
    mem_handles["D2_golden_l3"] = BingoMemSymbol("D_data_L3", offset=1*params["D_size"])
    mem_handles["D3_golden_l3"] = BingoMemSymbol("D_data_L3", offset=2*params["D_size"])
    mem_handles["D4_golden_l3"] = BingoMemSymbol("D_data_L3", offset=3*params["D_size"])

    # 3. Define Memory Handles (Dynamic Allocations)
    # Prepare all the A, B, D L1 buffers and final D L3 buffer for all chpiplets

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
    mem_handles["D_l1_chip00"] = BingoMemAlloc(
        "D_l1_chip00", size=params["D_size"], mem_level="L1", chip_id=0x00, cluster_id=0
    )
    mem_handles["D_l3_chip00"] = BingoMemAlloc(
        "D_l3_chip00", size=params["D_size"], mem_level="L3", chip_id=0x00
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
    mem_handles["D_l1_chip01"] = BingoMemAlloc(
        "D_l1_chip01", size=params["D_size"], mem_level="L1", chip_id=0x01, cluster_id=0
    )
    mem_handles["D_l3_chip01"] = BingoMemAlloc(
        "D_l3_chip01", size=params["D_size"], mem_level="L3", chip_id=0x01
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
    mem_handles["D_l1_chip10"] = BingoMemAlloc(
        "D_l1_chip10", size=params["D_size"], mem_level="L1", chip_id=0x10, cluster_id=0
    )
    mem_handles["D_l3_chip10"] = BingoMemAlloc(
        "D_l3_chip10", size=params["D_size"], mem_level="L3", chip_id=0x10
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
    mem_handles["D_l1_chip11"] = BingoMemAlloc(
        "D_l1_chip11", size=params["D_size"], mem_level="L1", chip_id=0x11, cluster_id=0
    )
    mem_handles["D_l3_chip11"] = BingoMemAlloc(
        "D_l3_chip11", size=params["D_size"], mem_level="L3", chip_id=0x11
    )

    return mem_handles

def create_dfg(params, mem_handles, platform):
    """DVFS cross-chiplet-trigger DAG.

    chip00 is the PRODUCER: it loads A+B and runs ``producer_stages`` gemm stages
    (busy -> high-power) while chip01/10/11 have NO ready task and sit IDLE, so the PM
    drops each of them to the low-power (idle) level. When chip00 finishes and
    BROADCASTS B to the others, their gated first task (Load_A) is released -> they go
    busy -> the PM raises them to the high-power (normal) level. So chip01/10/11
    demonstrate exactly: idle / low-power at the start of the run, then TRIGGERED by
    another chiplet (chip00) into the high-power state after some progress.

    Correctness: each consumer computes A_i x B_broadcast = D_i, checked against the same
    golden D_i the datagen emits (B_broadcast == golden B, A_i == golden A_i); chip00
    computes A1 x B = D1.
    """
    bingo_dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    gemm_core_id = 0   # versacore compute core
    dma_core_id = 1    # cluster dma core
    host_core_id = 2   # host core (excluded from the PM power domain)
    producer_stages = params["producer_stages"]

    # ---------------------------------------------------------------------
    # chip00 : producer. Loads A+B, runs producer_stages gemm stages (stays busy /
    # high-power), then broadcasts B to the other chiplets to trigger them.
    # ---------------------------------------------------------------------
    c0_load_A = BingoNode(
        assigned_chiplet_id=0x00, assigned_cluster_id=0, assigned_core_id=dma_core_id,
        node_name="Load_A_chip00", kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_handles["A1_mp"], dst_addr=mem_handles["A_l1_chip00"], size=params["Atile_size"]))
    c0_load_B = BingoNode(
        assigned_chiplet_id=0x00, assigned_cluster_id=0, assigned_core_id=dma_core_id,
        node_name="Load_B_chip00", kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_handles["B_mp"], dst_addr=mem_handles["B_l1_chip00"], size=params["B_size"]))
    bingo_dfg.bingo_add_node(c0_load_A)
    bingo_dfg.bingo_add_node(c0_load_B)
    bingo_dfg.add_edge(c0_load_A, c0_load_B)

    prev = c0_load_B
    for s in range(producer_stages):
        gemm = BingoNode(
            assigned_chiplet_id=0x00, assigned_cluster_id=0, assigned_core_id=gemm_core_id,
            node_name=f"Gemm_s{s}_chip00", kernel_name="__snax_bingo_kernel_gemm_full",
            kernel_args=SnaxBingoKernelGemmFullArgs(
                input_A_addr=mem_handles["A_l1_chip00"], input_B_addr=mem_handles["B_l1_chip00"],
                input_C_addr=0, output_D_addr=mem_handles["D_l1_chip00"],
                M=params["M"], K=params["K"], N=params["N"], array_shape_idx=params["arrayShapeIdx"],
                transpose_A=params["transposeA"], transpose_B=params["transposeB"], accumPrevC=params["accumPrevC"]))
        bingo_dfg.bingo_add_node(gemm)
        bingo_dfg.add_edge(prev, gemm)
        prev = gemm

    c0_store = BingoNode(
        assigned_chiplet_id=0x00, assigned_cluster_id=0, assigned_core_id=dma_core_id,
        node_name="Store_chip00", kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_handles["D_l1_chip00"], dst_addr=mem_handles["D_l3_chip00"], size=params["D_size"]))
    bingo_dfg.bingo_add_node(c0_store)
    bingo_dfg.add_edge(prev, c0_store)
    c0_check = BingoNode(
        assigned_chiplet_id=0x00, assigned_cluster_id=0, assigned_core_id=host_core_id,
        node_name="Check_chip00", kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            name="D_chip00", golden_data_addr=mem_handles["D1_golden_l3"],
            output_data_addr=mem_handles["D_l3_chip00"], data_size=params["D_size"]))
    bingo_dfg.bingo_add_node(c0_check)
    bingo_dfg.add_edge(c0_store, c0_check)

    # The TRIGGER: after chip00's compute, broadcast B to the same L1 address on every
    # chiplet. This releases the consumers' gated Load_A below.
    c0_broadcast = BingoNode(
        assigned_chiplet_id=0x00, assigned_cluster_id=0, assigned_core_id=dma_core_id,
        node_name="Broadcast_B_chip00", kernel_name="__snax_bingo_kernel_idma_broadcast",
        kernel_args=SnaxBingoKernelIdmaBroadcastArgs(
            src_addr=mem_handles["B_l1_chip00"], dst_addr=mem_handles["B_l1_chip00"], size=params["B_size"]))
    bingo_dfg.bingo_add_node(c0_broadcast)
    bingo_dfg.add_edge(prev, c0_broadcast)   # broadcast after the last producer gemm

    # ---------------------------------------------------------------------
    # chip01/10/11 : consumers. Their first task (Load_A) is gated on chip00's broadcast,
    # so they stay IDLE / low-power until chip00 triggers them, then go busy / high-power.
    # ---------------------------------------------------------------------
    consumers = [
        (0x01, "chip01", "A2_mp", "A_l1_chip01", "B_l1_chip01", "D_l1_chip01", "D_l3_chip01", "D2_golden_l3"),
        (0x10, "chip10", "A3_mp", "A_l1_chip10", "B_l1_chip10", "D_l1_chip10", "D_l3_chip10", "D3_golden_l3"),
        (0x11, "chip11", "A4_mp", "A_l1_chip11", "B_l1_chip11", "D_l1_chip11", "D_l3_chip11", "D4_golden_l3"),
    ]
    for chip, tag, a_mp, a_l1, b_l1, d_l1, d_l3, d_gold in consumers:
        c_load_A = BingoNode(
            assigned_chiplet_id=chip, assigned_cluster_id=0, assigned_core_id=dma_core_id,
            node_name=f"Load_A_{tag}", kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_handles[a_mp], dst_addr=mem_handles[a_l1], size=params["Atile_size"]))
        bingo_dfg.bingo_add_node(c_load_A)
        # CROSS-CHIPLET TRIGGER: this consumer does nothing until chip00 broadcasts.
        bingo_dfg.add_edge(c0_broadcast, c_load_A)
        # Sustained high-power phase: several gemm stages so the triggered RAISE fires
        # clearly during a real busy window (not a single tiny gemm).
        prev_c = c_load_A
        for cs in range(params["consumer_stages"]):
            c_gemm = BingoNode(
                assigned_chiplet_id=chip, assigned_cluster_id=0, assigned_core_id=gemm_core_id,
                node_name=f"Gemm_s{cs}_{tag}", kernel_name="__snax_bingo_kernel_gemm_full",
                kernel_args=SnaxBingoKernelGemmFullArgs(
                    input_A_addr=mem_handles[a_l1], input_B_addr=mem_handles[b_l1],
                    input_C_addr=0, output_D_addr=mem_handles[d_l1],
                    M=params["M"], K=params["K"], N=params["N"], array_shape_idx=params["arrayShapeIdx"],
                    transpose_A=params["transposeA"], transpose_B=params["transposeB"], accumPrevC=params["accumPrevC"]))
            bingo_dfg.bingo_add_node(c_gemm)
            bingo_dfg.add_edge(prev_c, c_gemm)
            prev_c = c_gemm
        c_store = BingoNode(
            assigned_chiplet_id=chip, assigned_cluster_id=0, assigned_core_id=dma_core_id,
            node_name=f"Store_{tag}", kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_handles[d_l1], dst_addr=mem_handles[d_l3], size=params["D_size"]))
        bingo_dfg.bingo_add_node(c_store)
        bingo_dfg.add_edge(prev_c, c_store)
        c_check = BingoNode(
            assigned_chiplet_id=chip, assigned_cluster_id=0, assigned_core_id=host_core_id,
            node_name=f"Check_{tag}", kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                name=f"D_{tag}", golden_data_addr=mem_handles[d_gold],
                output_data_addr=mem_handles[d_l3], data_size=params["D_size"]))
        bingo_dfg.bingo_add_node(c_check)
        bingo_dfg.add_edge(c_store, c_check)

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

    # Emit gemm_data.h (same as running gemm_multi_chiplet_datagen.py separately)
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
    dfg = create_dfg(params, mem_handles, platform)
    dfg.bingo_compile_dfg(params["app_name"], output_dir, output_file_name, extra_include_header_list=["gemm_data.h"])


if __name__ == "__main__":
    main()
