#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# F3 S1 -- P=2 cross-cluster IN-FABRIC moment-merge collective (the flag's first
# measured distributed result). Cluster 0 ("assembler") assembles a 2-lane beat
# from two (m,l) constants, then pushes it to cluster 1 ("merger") with the
# writer-side StreamMomentMergeRt extension armed. The fold happens on cluster 1's
# writer AS the beat lands (writerExtCfg carries the armed extension across the
# link) -- this is the in-fabric mechanism, contrasted against the
# gather-then-local baseline in the sibling xdma_moment_merge_p2_ablation workload.

# BEGIN WORKLOAD DESCRIPTION AND TASK GRAPH
# Task dependency graph:
#   Init_lane0_m -> MergePush
#   Init_lane0_l -> MergePush
#   Init_lane1_m -> MergePush
#   Init_lane1_l -> MergePush
#   MergePush -> Store_result -> Check_result
# (Init_lane* are independent loads of shared compile-time constants into
# cluster 0's staging beat; MergePush is the cross-cluster fold-triggering push.)
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

from momentmerge_p2_datagen import emit_header_file, CHECK_TOLERANCE  # noqa: E402
from bingo_dfg import BingoDFG  # noqa: E402
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa: E402
from bingo_node import BingoNode  # noqa: E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa: E402
from bingo_kernel_args import (  # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdmaMomentMergePushArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
    BINGO_CHECK_TYPE_FP32_TOL,
)

DMA_CORE = 1
HOST_CORE = 2
BEAT_BYTES = 64
LANE_M0, LANE_M1 = 0, 4       # byte offsets of m_0, m_1 (lane k = 4*k)
LANE_L0, LANE_L1 = 32, 36     # byte offsets of l_0, l_1 (lane 8+k = 32 + 4*k)


def get_args():
    parser = argparse.ArgumentParser(description="xdma_moment_merge_p2_infabric")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.cfg) as f:
        param = hjson.loads(f.read())

    if args.data_h is not None:
        with open(args.data_h, "w") as f:
            f.write(emit_header_file(**param))
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

    l1_stage_a = BingoMemAlloc("mm_p2_stage_a", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=0)
    l1_dest_b = BingoMemAlloc("mm_p2_dest_b", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=1)
    l3_result = BingoMemAlloc("mm_p2_result_l3", size=8, mem_level="L3")

    init_lane0_m = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="Init_lane0_m",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=BingoMemSymbol("mm_p2_m_vals", offset=0), dst_addr=l1_stage_a.view(LANE_M0), size=4))
    init_lane0_l = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="Init_lane0_l",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=BingoMemSymbol("mm_p2_l_vals", offset=0), dst_addr=l1_stage_a.view(LANE_L0), size=4))
    init_lane1_m = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="Init_lane1_m",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=BingoMemSymbol("mm_p2_m_vals", offset=4), dst_addr=l1_stage_a.view(LANE_M1), size=4))
    init_lane1_l = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="Init_lane1_l",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=BingoMemSymbol("mm_p2_l_vals", offset=4), dst_addr=l1_stage_a.view(LANE_L1), size=4))

    # The cross-cluster in-fabric fold: cluster 0 -> cluster 1, writer-ext armed (nvalid=2).
    merge_push = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="MergePush",
        kernel_name="__snax_bingo_kernel_xdma_moment_merge_push",
        kernel_args=SnaxBingoKernelXdmaMomentMergePushArgs(
            src_addr=l1_stage_a, dst_addr=l1_dest_b, nvalid=2))

    store_result = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Store_result",
        kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(src_addr=l1_dest_b, dst_addr=l3_result, size=8))

    check_result = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Check_result",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=BingoMemSymbol("mm_p2_golden_pair"),
            output_data_addr=l3_result,
            check_type=BINGO_CHECK_TYPE_FP32_TOL,
            tolerance=CHECK_TOLERANCE,
            num_elements=2,
            name="mm_p2_infabric"))

    for n in (init_lane0_m, init_lane0_l, init_lane1_m, init_lane1_l, merge_push, store_result, check_result):
        dfg.bingo_add_node(n)

    for src in (init_lane0_m, init_lane0_l, init_lane1_m, init_lane1_l):
        dfg.bingo_add_edge(src, merge_push)
    dfg.bingo_add_edge(merge_push, store_result)
    dfg.bingo_add_edge(store_result, check_result)

    print("Built DFG: 7 nodes (4 parallel lane inits + cross-cluster merge push + store + check)")

    dfg.bingo_compile_dfg(
        "F3 S1 P2 in-fabric moment-merge", args.output_dir, args.output_offload_file_name,
        extra_include_header_list=["momentmerge_p2_data.h"])
    print("Generated xdma_moment_merge_p2_infabric DFG")


if __name__ == "__main__":
    main()
