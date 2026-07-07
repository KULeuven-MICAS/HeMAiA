#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

"""
Cross-chiplet DMA sub-test for four chiplets with two active clusters each.

Flow:
  1. Load A1..A8 from the memory chip into local TCDM:
       chiplet 00 cluster0/1 -> A1/A2
       chiplet 01 cluster0/1 -> A3/A4
       chiplet 10 cluster0/1 -> A5/A6
       chiplet 11 cluster0/1 -> A7/A8
  2. Each cluster stores its loaded TCDM chunk to the chiplet-local L3 staging
     buffer.
  3. Each chiplet checks the L3 staging chunks against its local golden data.
  4. Chip00 pulls the per-cluster L3 staging chunks from chiplets 01, 10, and
     11 into chip00 L3.
  5. Chip00 checks the received chunks one by one.

Each chiplet has the full A1..A8 golden image in local L3.
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
    SnaxBingoKernelIdma1dCopyArgs,
)
from bingo_mem_handle import BingoMemAlloc, BingoMemFixedAddr, BingoMemSymbol  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_platform import guard_chiplet_count, guard_cluster_count, parse_platform_cfg  # noqa E402
from dma_cross_datagen import emit_header_file  # noqa E402


WORKLOAD_NAME = "dma_read_from_chip00_4chiplet_2cluster"
EXPECTED_CHIPLETS = [0x00, 0x01, 0x10, 0x11]
CLUSTER_IDS = [0, 1]

DMA_CORE = 1
HOST_CORE = 2
LOW_40_BIT_ADDR_MASK = "0x000000ffffffffffULL"


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


def data_index(chiplet_pos, cluster_id):
    return chiplet_pos * len(CLUSTER_IDS) + cluster_id


def chiplet_symbol_expr(chiplet_id, symbol_name, byte_offset=0):
    local_addr_expr = f"((uint64_t)((uintptr_t){symbol_name}) & {LOW_40_BIT_ADDR_MASK})"
    if byte_offset:
        local_addr_expr = f"({local_addr_expr} + {byte_offset})"
    return f"(chiplet_addr_transform_full(0x{chiplet_id:02x}, {local_addr_expr}))"


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
        "local_A_l1": {},
        "local_A_l3": {},
        "local_A_l3_full": {},
        "recv_A_chip00": {},
    }

    for chiplet_pos, chiplet in enumerate(EXPECTED_CHIPLETS):
        h = chip_hex(chiplet)
        for cluster_id in CLUSTER_IDS:
            idx = data_index(chiplet_pos, cluster_id)
            l3_offset = cluster_id * data_bytes

            mem["mempool_A"][idx] = BingoMemFixedAddr(mempool_base + idx * data_bytes)
            mem["golden_A"][idx] = BingoMemSymbol("A_golden_l3", offset=idx * data_bytes)
            mem["local_A_l1"][(chiplet, cluster_id)] = BingoMemAlloc(
                f"A_l1_chip{h}_c{cluster_id}",
                size=data_bytes,
                mem_level="L1",
                chip_id=chiplet,
                cluster_id=cluster_id,
            )
            mem["local_A_l3"][(chiplet, cluster_id)] = BingoMemSymbol(
                "A_local_l3",
                offset=l3_offset,
            )
            mem["local_A_l3_full"][(chiplet, cluster_id)] = chiplet_symbol_expr(
                chiplet,
                "A_local_l3",
                l3_offset,
            )

    local_chip_chunks = len(CLUSTER_IDS)
    for idx in range(local_chip_chunks, len(EXPECTED_CHIPLETS) * len(CLUSTER_IDS)):
        mem["recv_A_chip00"][idx] = BingoMemSymbol(
            "A_recv_chip00_l3",
            offset=(idx - local_chip_chunks) * data_bytes,
        )

    return mem


def add_local_cluster_flow(dfg, mem, chiplet, cluster_id, idx, data_bytes):
    h = chip_hex(chiplet)
    a_name = f"A{idx + 1}"

    load = BingoNode(
        assigned_chiplet_id=chiplet,
        assigned_cluster_id=cluster_id,
        assigned_core_id=DMA_CORE,
        node_name=f"Load_{a_name}_MemChip_to_Chip{h}_C{cluster_id}_TCDM",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem["mempool_A"][idx],
            dst_addr=mem["local_A_l1"][(chiplet, cluster_id)],
            size=data_bytes,
        ),
    )
    store = BingoNode(
        assigned_chiplet_id=chiplet,
        assigned_cluster_id=cluster_id,
        assigned_core_id=DMA_CORE,
        node_name=f"Store_{a_name}_Chip{h}_C{cluster_id}_TCDM_to_L3",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem["local_A_l1"][(chiplet, cluster_id)],
            dst_addr=mem["local_A_l3"][(chiplet, cluster_id)],
            size=data_bytes,
        ),
    )
    check = BingoNode(
        assigned_chiplet_id=chiplet,
        assigned_cluster_id=0,
        assigned_core_id=HOST_CORE,
        node_name=f"Check_{a_name}_Chip{h}_C{cluster_id}_Local_L3",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=mem["golden_A"][idx],
            output_data_addr=mem["local_A_l3"][(chiplet, cluster_id)],
            data_size=data_bytes,
            name=f"{a_name}_loaded_chip{h}_c{cluster_id}",
        ),
    )

    dfg.bingo_add_node(load)
    dfg.bingo_add_node(store)
    dfg.bingo_add_node(check)
    dfg.bingo_add_edge(load, store)
    dfg.bingo_add_edge(store, check)

    return load, check


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

    expected_num_data = len(EXPECTED_CHIPLETS) * len(CLUSTER_IDS)
    num_data = int(param.get("num_data", expected_num_data))
    data_bytes = int(param.get("data_bytes", 1024))
    if num_data != expected_num_data:
        raise ValueError(f"{WORKLOAD_NAME} expects num_data={expected_num_data}, got {num_data}")
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

    prev_local_check = None
    for chiplet_pos, chiplet in enumerate(EXPECTED_CHIPLETS):
        prev_cluster_load = None
        for cluster_id in CLUSTER_IDS:
            idx = data_index(chiplet_pos, cluster_id)
            load, check = add_local_cluster_flow(dfg, mem, chiplet, cluster_id, idx, data_bytes)
            if prev_cluster_load is not None:
                dfg.bingo_add_edge(prev_cluster_load, load)
            prev_cluster_load = load
            if prev_local_check is not None:
                dfg.bingo_add_edge(prev_local_check, check)
            prev_local_check = check

    prev_remote_check = prev_local_check
    for chiplet_pos, chiplet in enumerate(EXPECTED_CHIPLETS[1:], start=1):
        h = chip_hex(chiplet)
        for cluster_id in CLUSTER_IDS:
            idx = data_index(chiplet_pos, cluster_id)
            a_name = f"A{idx + 1}"

            copy_to_chip00 = BingoNode(
                assigned_chiplet_id=0x00,
                assigned_cluster_id=0,
                assigned_core_id=HOST_CORE,
                node_name=f"Pull_{a_name}_Chip{h}_C{cluster_id}_L3_to_Chip00",
                kernel_name="__host_bingo_kernel_xdma_1d_copy",
                kernel_args=HostBingoKernelXdma1dCopyArgs(
                    src_addr=mem["local_A_l3_full"][(chiplet, cluster_id)],
                    dst_addr=mem["recv_A_chip00"][idx],
                    size=data_bytes,
                ),
            )
            check_recv = BingoNode(
                assigned_chiplet_id=0x00,
                assigned_cluster_id=0,
                assigned_core_id=HOST_CORE,
                node_name=f"Check_{a_name}_Received_from_Chip{h}_C{cluster_id}",
                kernel_name="__host_bingo_kernel_check_result",
                kernel_args=HostBingoKernelCheckResultArgs(
                    golden_data_addr=mem["golden_A"][idx],
                    output_data_addr=mem["recv_A_chip00"][idx],
                    data_size=data_bytes,
                    name=f"{a_name}_recv_from_chip{h}_c{cluster_id}",
                ),
            )

            dfg.bingo_add_node(copy_to_chip00)
            dfg.bingo_add_node(check_recv)
            dfg.bingo_add_edge(prev_remote_check, copy_to_chip00)
            dfg.bingo_add_edge(copy_to_chip00, check_recv)
            prev_remote_check = check_recv

    print("Built DFG: load/store/check A1..A8, then pull/check A3..A8 on chip00")
    print(f"  active_chiplets={[chip_hex(c) for c in EXPECTED_CHIPLETS]}")
    print(f"  active_clusters={CLUSTER_IDS}")
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
