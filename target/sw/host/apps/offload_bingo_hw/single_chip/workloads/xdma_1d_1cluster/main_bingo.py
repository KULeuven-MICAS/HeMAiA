# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# xdma_1d: a single xDMA 1D-copy functional test (no sweep).
#   Load_input (iDMA L3 -> L1) -> XDMA_copy (xdma_1d_copy L1 -> L1)
#   -> Store (host iDMA L1 -> L3) -> Host check against golden.
# The xDMA 1d_copy is the identity, so golden == input.

# BEGIN WORKLOAD DESCRIPTION AND TASK GRAPH
# Minimal single xDMA 1D copy functional test.
#
# Task dependency graph:
#
# Load_input -> XDMA_copy -> Store_copy -> Check_copy
# END WORKLOAD DESCRIPTION AND TASK GRAPH

import os
import sys
import argparse
import pathlib
import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
WORKLOADS_DIR = os.path.dirname(current_dir)
sys.path.append(WORKLOADS_DIR)
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(current_dir)

from xdma_1d_datagen import emit_header_file  # noqa E402
from bingo_dfg import BingoDFG  # noqa E402
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_kernel_args import (  # noqa E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdma1dCopyArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)

DMA_CORE = 1
HOST_CORE = 2


def get_args():
    parser = argparse.ArgumentParser(description="xdma_1d — single xDMA 1D copy")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True,
                        help="Path to generated occamy.h with HW platform defines")
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.cfg) as f:
        param = hjson.loads(f.read())
    size = int(param["size"])

    if args.data_h is not None:
        with open(args.data_h, "w") as f:
            f.write(emit_header_file(size))
        print(f"Written data header: {args.data_h}")

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return
    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"])

    l1_src = BingoMemAlloc("l1_src", size=size, mem_level="L1", chip_id=0, cluster_id=0)
    l1_dst = BingoMemAlloc("l1_dst", size=size, mem_level="L1", chip_id=0, cluster_id=0)
    l3_out = BingoMemAlloc("l3_out", size=size, mem_level="L3")

    load = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="Load_input",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol("input_data"), l1_src, size))
    xdma_copy = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="XDMA_copy",
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(l1_src, l1_dst, size))
    store = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Store_copy",
        kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(l1_dst, l3_out, size))
    check = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Check_copy",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            BingoMemSymbol("golden_copy"), l3_out, size, name="XDMA_1d_copy"))

    for node in (load, xdma_copy, store, check):
        dfg.bingo_add_node(node)
    dfg.bingo_add_edge(load, xdma_copy)
    dfg.bingo_add_edge(xdma_copy, store)
    dfg.bingo_add_edge(store, check)

    dfg.bingo_compile_dfg("Single-Chip xDMA 1D Copy", args.output_dir,
                          args.output_offload_file_name,
                          extra_include_header_list=["xdma_1d_data.h"])
    print("Generated xdma_1d copy DFG")


if __name__ == "__main__":
    main()
