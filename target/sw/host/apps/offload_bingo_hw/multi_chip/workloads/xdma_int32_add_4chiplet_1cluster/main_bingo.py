#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

"""
INT32 add reduction across 4 chiplets with one cluster per chiplet.

Data layout in the MemPool chip:
  A1, A2, A3, A4, golden(A1+A2), golden(A1+A2+A3),
  golden(A1+A2+A3+A4).

The DFG loads A1..A4 into cluster 0's TCDM of chiplets 00, 01, 10,
and 11, respectively. Remote A tiles are copied into cluster 0's TCDM
of chiplet 00, one received slot at a time: A2, then A3, then A4.
Cluster 0 of chiplet 00 performs gradual two-operand xDMA int32 adds
and checks each running sum against the MemPool golden data.
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
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdmaElementwiseAddAbArgs,
)
from bingo_mem_handle import BingoMemAlloc, BingoMemFixedAddr  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_platform import guard_chiplet_count, guard_cluster_count, parse_platform_cfg  # noqa E402
from int32_add_multi_chiplet_datagen import emit_header_file  # noqa E402


APP_NAME = "xdma_int32_add_4chiplet_1cluster"
REQUIRED_CHIPLETS = [0x00, 0x01, 0x10, 0x11]
REDUCTION_CHIPLET = 0x00
DMA_CORE = 1
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


def chiplet_full_addr_from_c_expr(chiplet_id, c_expr):
    local_addr_expr = f"((uint64_t)({c_expr}) & {LOW_40_BIT_ADDR_MASK})"
    return f"(chiplet_addr_transform_full(0x{chiplet_id:02x}, {local_addr_expr}))"


def get_input_bytes(param):
    input_bytes = int(param.get("input_bytes", param.get("bytes_per_input", 1024)))
    if input_bytes <= 0:
        raise ValueError(f"input_bytes must be positive, got {input_bytes}")
    if input_bytes % 4 != 0:
        raise ValueError(f"input_bytes ({input_bytes}) must be divisible by sizeof(int32_t)")
    if input_bytes % 64 != 0:
        raise ValueError(f"input_bytes ({input_bytes}) must be 64-byte aligned for xDMA")
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
    # MemPool address space.
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

    # Local TCDM buffers. All chiplets use the same A_local_l1 name so their
    # first L1 allocation has the same cluster-local offset. Chiplet 00 uses
    # that offset to pull remote chiplet A tiles into Z_remote_l1.
    l1_A_local = {}
    for i, chiplet in enumerate(REQUIRED_CHIPLETS):
        l1_A_local[i] = BingoMemAlloc(
            "A_local_l1",
            size=input_bytes,
            mem_level="L1",
            chip_id=chiplet,
            cluster_id=0,
        )

    l1_sum_ping = BingoMemAlloc(
        "B_sum_ping_l1",
        size=input_bytes,
        mem_level="L1",
        chip_id=REDUCTION_CHIPLET,
        cluster_id=0,
    )
    l1_sum_pong = BingoMemAlloc(
        "C_sum_pong_l1",
        size=input_bytes,
        mem_level="L1",
        chip_id=REDUCTION_CHIPLET,
        cluster_id=0,
    )
    l1_remote_slot = BingoMemAlloc(
        "Z_remote_l1",
        size=input_bytes,
        mem_level="L1",
        chip_id=REDUCTION_CHIPLET,
        cluster_id=0,
    )

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
            assigned_core_id=DMA_CORE,
            node_name=f"Load_A{i + 1}_MemPool_to_Chip{h}_L1",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_A[i],
                dst_addr=l1_A_local[i],
                size=input_bytes,
            ),
        )
        dfg.bingo_add_node(load_A[i])

    prev_sum = l1_A_local[0]
    prev_ready = load_A[0]
    final_check = None
    sum_outputs = [l1_sum_ping, l1_sum_pong]
    local_a_c_var = l1_A_local[0].get_c_var_name()

    for i in range(1, len(REQUIRED_CHIPLETS)):
        source_chiplet = REQUIRED_CHIPLETS[i]
        h = chip_hex(source_chiplet)
        remote_src = chiplet_full_addr_from_c_expr(source_chiplet, local_a_c_var)
        dst_sum = sum_outputs[(i - 1) % len(sum_outputs)]
        copy_remote = BingoNode(
            assigned_chiplet_id=REDUCTION_CHIPLET,
            assigned_cluster_id=0,
            assigned_core_id=DMA_CORE,
            node_name=f"Copy_A{i + 1}_Chip{h}_to_Chip00_L1",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=remote_src,
                dst_addr=l1_remote_slot,
                size=input_bytes,
            ),
        )
        add = BingoNode(
            assigned_chiplet_id=REDUCTION_CHIPLET,
            assigned_cluster_id=0,
            assigned_core_id=DMA_CORE,
            node_name=f"XDMA_Add_A1_to_A{i + 1}",
            kernel_name="__snax_bingo_kernel_xdma_elementwise_add_ab",
            kernel_args=SnaxBingoKernelXdmaElementwiseAddAbArgs(
                src_a_addr=prev_sum,
                src_b_addr=l1_remote_slot,
                dst_addr=dst_sum,
                num_int32_elements=num_elements,
            ),
        )
        final_check = BingoNode(
            assigned_chiplet_id=REDUCTION_CHIPLET,
            assigned_cluster_id=0,
            assigned_core_id=HOST_CORE,
            node_name=f"Check_A1_to_A{i + 1}",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_golden_sum[i],
                output_data_addr=dst_sum,
                data_size=input_bytes,
                name=f"A1_to_A{i + 1}",
            ),
        )

        for node in (copy_remote, add, final_check):
            dfg.bingo_add_node(node)

        dfg.bingo_add_edge(load_A[i], copy_remote)
        dfg.bingo_add_edge(prev_ready, copy_remote)
        dfg.bingo_add_edge(prev_ready, add)
        dfg.bingo_add_edge(copy_remote, add)
        dfg.bingo_add_edge(add, final_check)

        prev_sum = dst_sum
        prev_ready = final_check

    print(f"Built DFG: 13 nodes (4 A loads, 3 remote copies, 3 xDMA adds, 3 checks)")
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
