#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# F3 S2 -- P=2 gather-then-local ablation baseline. Same inputs/golden as the
# sibling xdma_moment_merge_p2_infabric workload; the ONLY difference is where
# the fold happens: here cluster 0 pushes the assembled beat to cluster 1 RAW
# (extension OFF, __snax_bingo_kernel_xdma_1d_copy explicitly disables all
# extensions), then a SEPARATE, purely-local pass on cluster 1 does the actual
# StreamMomentMergeRt fold. Comparing this workload's total latency/D2D-bytes
# against xdma_moment_merge_p2_infabric's is the paper's systems claim (the
# in-fabric-vs-gather-then-local delta) -- the arithmetic is identical in both.

# BEGIN WORKLOAD DESCRIPTION AND TASK GRAPH
# Task dependency graph:
#   Init_lane0_m -> RawPush
#   Init_lane0_l -> RawPush
#   Init_lane1_m -> RawPush
#   Init_lane1_l -> RawPush
#   RawPush -> LocalMergePush -> Store_result -> Check_result
# (RawPush crosses cluster 0 -> cluster 1 with the extension OFF -- a plain
# copy; LocalMergePush is a purely LOCAL cluster-1 pass with the extension ON.)
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
    SnaxBingoKernelXdma1dCopyArgs,
    SnaxBingoKernelXdmaMomentMergePushArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
    BINGO_CHECK_TYPE_FP32_TOL,
)

DMA_CORE = 1
HOST_CORE = 2
BEAT_BYTES = 64
LANE_M0, LANE_M1 = 0, 4
LANE_L0, LANE_L1 = 32, 36


def get_args():
    parser = argparse.ArgumentParser(description="xdma_moment_merge_p2_ablation")
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

    l1_stage_a = BingoMemAlloc("mm_p2a_stage_a", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=0)
    l1_raw_b = BingoMemAlloc("mm_p2a_raw_b", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=1)
    l1_result_b = BingoMemAlloc("mm_p2a_result_b", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=1)
    l3_result = BingoMemAlloc("mm_p2a_result_l3", size=8, mem_level="L3")

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

    # Cross-cluster hop, extension OFF (xdma_1d_copy disables all extensions) -- the "gather" step.
    raw_push = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="RawPush",
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(src_addr=l1_stage_a, dst_addr=l1_raw_b, size=BEAT_BYTES))

    # Purely LOCAL fold on cluster 1 -- the "then-local-merge" step.
    local_merge = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=1, assigned_core_id=DMA_CORE,
        node_name="LocalMergePush",
        kernel_name="__snax_bingo_kernel_xdma_moment_merge_push",
        kernel_args=SnaxBingoKernelXdmaMomentMergePushArgs(
            src_addr=l1_raw_b, dst_addr=l1_result_b, nvalid=2))

    store_result = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Store_result",
        kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(src_addr=l1_result_b, dst_addr=l3_result, size=8))

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
            name="mm_p2_ablation"))

    for n in (init_lane0_m, init_lane0_l, init_lane1_m, init_lane1_l,
              raw_push, local_merge, store_result, check_result):
        dfg.bingo_add_node(n)

    for src in (init_lane0_m, init_lane0_l, init_lane1_m, init_lane1_l):
        dfg.bingo_add_edge(src, raw_push)
    dfg.bingo_add_edge(raw_push, local_merge)
    dfg.bingo_add_edge(local_merge, store_result)
    dfg.bingo_add_edge(store_result, check_result)

    print("Built DFG: 8 nodes (4 parallel lane inits + raw cross-cluster push "
          "+ local merge + store + check)")

    dfg.bingo_compile_dfg(
        "F3 S2 P2 gather-then-local ablation", args.output_dir, args.output_offload_file_name,
        extra_include_header_list=["momentmerge_p2_data.h"])
    print("Generated xdma_moment_merge_p2_ablation DFG")


if __name__ == "__main__":
    main()
