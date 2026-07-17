#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

# BEGIN WORKLOAD DESCRIPTION AND TASK GRAPH
# Four-chiplet int32 add using host kernels. Each chiplet loads one input array
# to L3. Chiplet 00 receives remote arrays and performs running sums with checks.
#
# Task dependency graph:
#
# Initial loads:
#   Load_A1_MemPool_to_Chip00_L3
#   Load_A2_MemPool_to_Chip01_L3
#   Load_A3_MemPool_to_Chip10_L3
#   Load_A4_MemPool_to_Chip11_L3
#
# Running host reduction on chip00:
#   Load_A1 + Load_A2 -> Copy_A2_Chip01_to_Chip00_L3
#   Load_A1 + Copy_A2_Chip01_to_Chip00_L3 -> Add_A1_to_A2
#   Add_A1_to_A2 -> Load_Golden_A1_to_A2 -> Check_A1_to_A2
#
#   Check_A1_to_A2 + Load_A3 -> Copy_A3_Chip10_to_Chip00_L3
#   Check_A1_to_A2 + Copy_A3_Chip10_to_Chip00_L3 -> Add_A1_to_A3
#   Add_A1_to_A3 -> Load_Golden_A1_to_A3 -> Check_A1_to_A3
#
#   Check_A1_to_A3 + Load_A4 -> Copy_A4_Chip11_to_Chip00_L3
#   Check_A1_to_A3 + Copy_A4_Chip11_to_Chip00_L3 -> Add_A1_to_A4
#   Add_A1_to_A4 -> Load_Golden_A1_to_A4 -> Check_A1_to_A4
# END WORKLOAD DESCRIPTION AND TASK GRAPH

"""
INT32 add reduction across 4 chiplets with one cluster per chiplet.

Data layout in the MemPool chip:
  A1, A2, A3, A4, golden(A1+A2), golden(A1+A2+A3),
  golden(A1+A2+A3+A4).

The DFG loads A1..A4 into chiplets 00, 01, 10, and 11 respectively. Remote
A tiles are copied into a shared chiplet-0 L3 receive slot one at a time;
chiplet 0 performs the in-place running int32_add and checks each sum against
the MemPool golden data.
"""

import argparse
import os
import pathlib
import sys

import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)

sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(current_dir)

from bingo_dfg import BingoDFG  # noqa E402
from bingo_helpers import chiplet_addr_transform_loc  # noqa E402
from bingo_kernel_args import (  # noqa E402
    HostBingoKernelCheckResultArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelAraAddI32Args,
)
from bingo_mem_handle import BingoMemAlloc, BingoMemFixedAddr, BingoMemSymbol  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_platform import guard_chiplet_count, guard_cluster_count, parse_platform_cfg  # noqa E402
from int32_add_multi_chiplet_datagen import emit_header_file  # noqa E402


APP_NAME = "host_int32_add_4chiplet_1cluster"
REQUIRED_CHIPLETS = [0x00, 0x01, 0x10, 0x11]
REDUCTION_CHIPLET = 0x00
HOST_CORE = 2
LOW_40_BIT_ADDR_MASK = "0x000000ffffffffffULL"


def get_args():
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def chip_hex(chiplet_id):
    return f"{chiplet_id:02x}"


def chiplet_full_addr_expr(chiplet_id, symbol_name):
    local_addr_expr = f"((uint64_t)((uintptr_t){symbol_name}) & {LOW_40_BIT_ADDR_MASK})"
    return f"(chiplet_addr_transform_full(0x{chiplet_id:02x}, {local_addr_expr}))"


def get_input_bytes(param):
    input_bytes = int(param.get("input_bytes", param.get("bytes_per_input", 1024)))
    if input_bytes <= 0:
        raise ValueError(f"input_bytes must be positive, got {input_bytes}")
    if input_bytes % 4 != 0:
        raise ValueError(f"input_bytes ({input_bytes}) must be divisible by sizeof(int32_t)")
    if input_bytes % 64 != 0:
        raise ValueError(f"input_bytes ({input_bytes}) must be 64-byte aligned for host DMA")
    return input_bytes


def main():
    args = get_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        hw = hjson.loads(f.read())
    merged = {**param, **hw}

    if args.data_h is not None:
        content = emit_header_file(**merged, out_dir=os.path.join(args.output_dir, "build"))
        with args.data_h.open("w") as f:
            f.write(content)
        print(f"Written data header: {args.data_h}")

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_chiplet_count(param, platform, args.output_dir, args.output_offload_file_name):
        return
    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return

    missing_chiplets = [chiplet for chiplet in REQUIRED_CHIPLETS if chiplet not in platform["chiplet_ids"]]
    if missing_chiplets:
        missing = ", ".join(chip_hex(chiplet) for chiplet in missing_chiplets)
        available = ", ".join(chip_hex(chiplet) for chiplet in platform["chiplet_ids"])
        raise ValueError(f"{APP_NAME} requires chiplets {missing}; platform has [{available}]")

    input_bytes = get_input_bytes(param)
    num_elements = input_bytes // 4

    # The MemPool layout is A1, A2, A3, A4, golden(A1+A2), golden(A1+A2+A3), golden(A1+A2+A3+A4).
    # Mmepool address space
    mempool_base = chiplet_addr_transform_loc(2, 0, 0x8000_0000)
    A_mp_base = mempool_base
    golden_mp_base = A_mp_base + len(REQUIRED_CHIPLETS) * input_bytes

    mem_A = {
        i: BingoMemFixedAddr(A_mp_base + i * input_bytes)
        for i in range(len(REQUIRED_CHIPLETS))
    }
    mem_golden_sum = {
        i: BingoMemFixedAddr(golden_mp_base + (i - 1) * input_bytes)
        for i in range(1, len(REQUIRED_CHIPLETS))
    }

    # local L3 addresses on each chiplet for storing the A tiles loaded from the MemPool; 
    # these are the source addresses for the remote copies to chiplet 0's L3 reduction slot.
    # 
    l3_A_local = {}
    for i, chiplet in enumerate(REQUIRED_CHIPLETS):
        h = chip_hex(chiplet)
        l3_A_local[i] = BingoMemAlloc(
            f"A{i + 1}_chip{h}_l3",
            size=input_bytes,
            mem_level="L3",
            chip_id=chiplet,
        )

    l3_golden_sum = {}
    for i in range(1, len(REQUIRED_CHIPLETS)):
        l3_golden_sum[i] = BingoMemAlloc(
            f"golden_sum_A1_to_A{i + 1}_chip00_l3",
            size=input_bytes,
            mem_level="L3",
            chip_id=REDUCTION_CHIPLET,
        )
    remote_reduce_slot = BingoMemSymbol("A_remote_chip00_l3")

    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    load_A = {}
    for i, chiplet in enumerate(REQUIRED_CHIPLETS):
        h = chip_hex(chiplet)
        load_A[i] = BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=0,
            assigned_core_id=HOST_CORE,
            node_name=f"Load_A{i + 1}_MemPool_to_Chip{h}_L3",
            kernel_name="__host_bingo_kernel_idma",
            kernel_args=HostBingoKernelIdmaArgs(
                src_addr=mem_A[i],
                dst_addr=l3_A_local[i],
                size=input_bytes,
            ),
        )
        dfg.bingo_add_node(load_A[i])

    prev_sum = l3_A_local[0]
    prev_ready = load_A[0]
    final_check = None

    for i in range(1, len(REQUIRED_CHIPLETS)):
        source_chiplet = REQUIRED_CHIPLETS[i]
        h = chip_hex(source_chiplet)
        copy_remote = BingoNode(
            assigned_chiplet_id=source_chiplet,
            assigned_cluster_id=0,
            assigned_core_id=HOST_CORE,
            node_name=f"Copy_A{i + 1}_Chip{h}_to_Chip00_L3",
            kernel_name="__host_bingo_kernel_idma",
            kernel_args=HostBingoKernelIdmaArgs(
                src_addr=l3_A_local[i],
                dst_addr=chiplet_full_addr_expr(REDUCTION_CHIPLET, "A_remote_chip00_l3"),
                size=input_bytes,
            ),
        )
        add = BingoNode(
            assigned_chiplet_id=REDUCTION_CHIPLET,
            assigned_cluster_id=0,
            assigned_core_id=HOST_CORE,
            node_name=f"Add_A1_to_A{i + 1}",
            kernel_name="__host_bingo_kernel_add_i32",
            kernel_args=HostBingoKernelAraAddI32Args(
                input_a_addr=prev_sum,
                input_b_addr=remote_reduce_slot,
                output_addr=prev_sum,
                num_elements=num_elements,
            ),
        )
        load_golden = BingoNode(
            assigned_chiplet_id=REDUCTION_CHIPLET,
            assigned_cluster_id=0,
            assigned_core_id=HOST_CORE,
            node_name=f"Load_Golden_A1_to_A{i + 1}",
            kernel_name="__host_bingo_kernel_idma",
            kernel_args=HostBingoKernelIdmaArgs(
                src_addr=mem_golden_sum[i],
                dst_addr=l3_golden_sum[i],
                size=input_bytes,
            ),
        )
        final_check = BingoNode(
            assigned_chiplet_id=REDUCTION_CHIPLET,
            assigned_cluster_id=0,
            assigned_core_id=HOST_CORE,
            node_name=f"Check_A1_to_A{i + 1}",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=l3_golden_sum[i],
                output_data_addr=prev_sum,
                data_size=input_bytes,
                name=f"A1_to_A{i + 1}",
            ),
        )

        for node in (copy_remote, add, load_golden, final_check):
            dfg.bingo_add_node(node)

        dfg.bingo_add_edge(load_A[i], copy_remote)
        dfg.bingo_add_edge(prev_ready, copy_remote)
        dfg.bingo_add_edge(prev_ready, add)
        dfg.bingo_add_edge(copy_remote, add)
        dfg.bingo_add_edge(add, load_golden)
        dfg.bingo_add_edge(load_golden, final_check)

        prev_ready = final_check

    print(f"Built DFG: 16 nodes (4 A loads, 3 remote copies, 3 adds, 3 golden loads, 3 checks)")
    print(f"  chiplets={[chip_hex(chiplet) for chiplet in REQUIRED_CHIPLETS]}")
    print(f"  input_bytes={input_bytes}, num_elements={num_elements}")

    dfg.bingo_compile_dfg(
        APP_NAME,
        output_dir,
        args.output_offload_file_name,
        extra_include_header_list=["int32_add_data.h"],
    )

    return final_check


if __name__ == "__main__":
    main()
