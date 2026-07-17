#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

# BEGIN WORKLOAD DESCRIPTION AND TASK GRAPH
# Cross-chiplet DMA read test. Every chiplet first loads and checks a local L3
# chunk. Chiplet 00 then pulls chunks from chiplets 01, 10, and 11 and checks the
# received buffers.
#
# Task dependency graph:
#
# Local load/check phase:
#   Load_A1_MemChip_to_Chip00_L3 -> Check_A1_Chip00_Local_L3
#   Check_A1_Chip00_Local_L3 -> Check_A2_Chip01_Local_L3
#   Load_A2_MemChip_to_Chip01_L3 -> Check_A2_Chip01_Local_L3
#   Check_A2_Chip01_Local_L3 -> Check_A3_Chip10_Local_L3
#   Load_A3_MemChip_to_Chip10_L3 -> Check_A3_Chip10_Local_L3
#   Check_A3_Chip10_Local_L3 -> Check_A4_Chip11_Local_L3
#   Load_A4_MemChip_to_Chip11_L3 -> Check_A4_Chip11_Local_L3
#
# Chip00 read/pull phase:
#   Check_A4_Chip11_Local_L3 -> Pull_A2_Chip01_L3_to_Chip00
#   Pull_A2_Chip01_L3_to_Chip00 -> Check_A2_Received_from_Chip01
#   Check_A2_Received_from_Chip01 -> Pull_A3_Chip10_L3_to_Chip00
#   Pull_A3_Chip10_L3_to_Chip00 -> Check_A3_Received_from_Chip10
#   Check_A3_Received_from_Chip10 -> Pull_A4_Chip11_L3_to_Chip00
#   Pull_A4_Chip11_L3_to_Chip00 -> Check_A4_Received_from_Chip11
# END WORKLOAD DESCRIPTION AND TASK GRAPH

"""
Cross-chiplet DMA sub-test extracted from the 4-chiplet K-split GEMM flow.

Flow:
  1. Load A1..A4 from the memory chip into the local L3 of chiplets
     00, 01, 10, and 11.
  2. Each chiplet checks its loaded local L3 buffer against its local golden.
  3. Chip00 pulls the local L3 data from chiplets 01, 10, and 11 into chip00 L3.
  4. Chip00 checks the received chunks one by one.
"""

import argparse
import os
import pathlib
import sys

import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))

sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(current_dir)

from bingo_dfg import BingoDFG  # noqa E402
from bingo_helpers import chiplet_addr_transform_loc  # noqa E402
from bingo_kernel_args import (  # noqa E402
    HostBingoKernelCheckResultArgs,
    HostBingoKernelXdma1dCopyArgs,
)
from bingo_mem_handle import BingoMemFixedAddr, BingoMemSymbol  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_platform import guard_chiplet_count, guard_cluster_count, parse_platform_cfg  # noqa E402
from dma_cross_datagen import emit_header_file  # noqa E402


WORKLOAD_NAME = "dma_read_from_chip00_4chiplet_1cluster"
EXPECTED_CHIPLETS = [0x00, 0x01, 0x10, 0x11]

HOST_CORE = 2


def get_args():
    parser = argparse.ArgumentParser(description=WORKLOAD_NAME)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def chip_hex(chiplet_id):
    return f"{chiplet_id:02x}"


def chiplet_symbol_expr(chiplet_id, symbol_name, byte_offset=0):
    offset = f" + {byte_offset}" if byte_offset else ""
    return (
        f"(chiplet_addr_transform_full(0x{chiplet_id:02x}, "
        f"(uint64_t)((uintptr_t){symbol_name}{offset})))"
    )


def validate_platform(platform):
    chiplets = set(platform["chiplet_ids"])
    missing = [chip for chip in EXPECTED_CHIPLETS if chip not in chiplets]
    if missing:
        raise ValueError(
            f"{WORKLOAD_NAME} expects chiplets "
            f"{[chip_hex(c) for c in EXPECTED_CHIPLETS]}, "
            f"but platform is missing {[chip_hex(c) for c in missing]}"
        )


def define_memory_handles(data_bytes, mempool_base):
    mem = {
        "mempool_A": {},
        "golden_A": {},
        "local_A_l3_full": {},
        "recv_A_chip00": {},
    }
    mem["local_A_l3"] = BingoMemSymbol("A_local_l3")

    for idx, chiplet in enumerate(EXPECTED_CHIPLETS):
        mem["mempool_A"][idx] = BingoMemFixedAddr(mempool_base + idx * data_bytes)
        mem["golden_A"][idx] = BingoMemSymbol("A_golden_l3", offset=idx * data_bytes)
        mem["local_A_l3_full"][idx] = chiplet_symbol_expr(chiplet, "A_local_l3")

    for idx in range(1, len(EXPECTED_CHIPLETS)):
        mem["recv_A_chip00"][idx] = BingoMemSymbol(
            "A_recv_chip00_l3",
            offset=(idx - 1) * data_bytes,
        )

    return mem


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
        content = emit_header_file(**{**merged, "out_dir": os.path.join(output_dir, "build")})
        with args.data_h.open("w") as f:
            f.write(content)
        print(f"Written data header: {args.data_h}")

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_chiplet_count(param, platform, output_dir, args.output_offload_file_name):
        return
    if not guard_cluster_count(param, platform, output_dir, args.output_offload_file_name):
        return
    validate_platform(platform)

    num_data = int(param.get("num_data", 4))
    data_bytes = int(param.get("data_bytes", 1024))
    if num_data != 4:
        raise ValueError(f"{WORKLOAD_NAME} expects num_data=4, got {num_data}")
    if data_bytes <= 0:
        raise ValueError(f"data_bytes must be positive, got {data_bytes}")

    mempool_loc_x = 2
    mempool_loc_y = 0
    mempool_local_base = 0x8000_0000
    mempool_base = chiplet_addr_transform_loc(
        mempool_loc_x, mempool_loc_y, mempool_local_base
    )
    mem = define_memory_handles(data_bytes, mempool_base)

    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    load_checks = {}
    prev_load_check = None

    for idx, chiplet in enumerate(EXPECTED_CHIPLETS):
        h = chip_hex(chiplet)

        load = BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=0,
            assigned_core_id=HOST_CORE,
            node_name=f"Load_A{idx + 1}_MemChip_to_Chip{h}_L3",
            kernel_name="__host_bingo_kernel_xdma_1d_copy",
            kernel_args=HostBingoKernelXdma1dCopyArgs(
                src_addr=mem["mempool_A"][idx],
                dst_addr=mem["local_A_l3"],
                size=data_bytes,
            ),
        )
        check = BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=0,
            assigned_core_id=HOST_CORE,
            node_name=f"Check_A{idx + 1}_Chip{h}_Local_L3",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem["golden_A"][idx],
                output_data_addr=mem["local_A_l3"],
                data_size=data_bytes,
                name=f"A{idx + 1}_loaded_chip{h}",
            ),
        )

        dfg.bingo_add_node(load)
        dfg.bingo_add_node(check)
        dfg.bingo_add_edge(load, check)
        if prev_load_check is not None:
            dfg.bingo_add_edge(prev_load_check, check)
        prev_load_check = check
        load_checks[idx] = check

    prev_remote_check = load_checks[num_data - 1]
    for idx, chiplet in enumerate(EXPECTED_CHIPLETS[1:], start=1):
        h = chip_hex(chiplet)

        copy_to_chip00 = BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=HOST_CORE,
            node_name=f"Pull_A{idx + 1}_Chip{h}_L3_to_Chip00",
            kernel_name="__host_bingo_kernel_xdma_1d_copy",
            kernel_args=HostBingoKernelXdma1dCopyArgs(
                src_addr=mem["local_A_l3_full"][idx],
                dst_addr=mem["recv_A_chip00"][idx],
                size=data_bytes,
            ),
        )
        check_recv = BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=HOST_CORE,
            node_name=f"Check_A{idx + 1}_Received_from_Chip{h}",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem["golden_A"][idx],
                output_data_addr=mem["recv_A_chip00"][idx],
                data_size=data_bytes,
                name=f"A{idx + 1}_recv_from_chip{h}",
            ),
        )

        dfg.bingo_add_node(copy_to_chip00)
        dfg.bingo_add_node(check_recv)
        dfg.bingo_add_edge(prev_remote_check, copy_to_chip00)
        dfg.bingo_add_edge(copy_to_chip00, check_recv)
        prev_remote_check = check_recv

    print("Built DFG: load/check A1..A4, then pull/check A2..A4 back to chip00")
    print(f"  active_chiplets={[chip_hex(c) for c in EXPECTED_CHIPLETS]}")
    print(f"  data_bytes={data_bytes}, total_mempool_bytes={num_data * data_bytes}")
    print(f"  mempool_base=0x{mempool_base:x}")

    dfg.bingo_compile_dfg(
        WORKLOAD_NAME,
        output_dir,
        args.output_offload_file_name,
        extra_include_header_list=["dma_cross_data.h"],
    )

    return prev_remote_check


if __name__ == "__main__":
    main()
