#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# F3 S3 -- P=4 cross-cluster in-fabric moment-merge, the scaling point (latency
# vs P, alongside xdma_moment_merge_p2_infabric). All 4 clusters genuinely
# involved: cluster 0 is the assembler (holds its own (m_0,l_0) locally, then
# REMOTE-reads (m_1,l_1)/(m_2,l_2)/(m_3,l_3) from clusters 1/2/3's own L1 into
# its staging beat), then pushes the assembled 4-lane beat to cluster 1 with
# the writer-side StreamMomentMergeRt extension armed (nvalid=4) -- the
# cross-cluster in-fabric fold, same mechanism as P=2, just more live lanes.

# BEGIN WORKLOAD DESCRIPTION AND TASK GRAPH
# Task dependency graph:
#   Init_c1_own -> Gather_lane1_m, Gather_lane1_l
#   Init_c2_own -> Gather_lane2_m, Gather_lane2_l
#   Init_c3_own -> Gather_lane3_m, Gather_lane3_l
#   Init_lane0_m, Init_lane0_l  (cluster 0's own value, no gather needed)
#   {Init_lane0_m, Init_lane0_l, Gather_lane1_m, Gather_lane1_l,
#    Gather_lane2_m, Gather_lane2_l, Gather_lane3_m, Gather_lane3_l} -> MergePush
#   MergePush -> Store_result -> Check_result
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

from momentmerge_p4_datagen import emit_header_file, CHECK_TOLERANCE  # noqa: E402
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
OWN_BYTES = 8  # a producer's own (m,l), held contiguously before gathering


def lane_m(k):
    return 4 * k


def lane_l(k):
    return 32 + 4 * k


def get_args():
    parser = argparse.ArgumentParser(description="xdma_moment_merge_p4_infabric")
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

    l1_stage_a = BingoMemAlloc("mm_p4_stage_a", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=0)
    l1_own = {i: BingoMemAlloc(f"mm_p4_own_c{i}", size=OWN_BYTES, mem_level="L1", chip_id=0, cluster_id=i)
              for i in (1, 2, 3)}
    l1_dest = BingoMemAlloc("mm_p4_dest_c1", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=1)
    l3_result = BingoMemAlloc("mm_p4_result_l3", size=8, mem_level="L3")

    nodes = []
    edges = []

    # Producer clusters 1,2,3: place their own (m,l) in their own L1. m_vals/l_vals are
    # SEPARATE arrays, so this is two explicit 4B copies per producer (m_i, then l_i).
    init_own_m = {}
    init_own_l = {}
    for i in (1, 2, 3):
        init_own_m[i] = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=f"Init_c{i}_own_m",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=BingoMemSymbol("mm_p4_m_vals", offset=4 * i), dst_addr=l1_own[i].view(0), size=4))
        init_own_l[i] = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=f"Init_c{i}_own_l",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=BingoMemSymbol("mm_p4_l_vals", offset=4 * i), dst_addr=l1_own[i].view(4), size=4))
        nodes += [init_own_m[i], init_own_l[i]]

    # Cluster 0's own value goes straight into its staging beat, lane 0.
    init_lane0_m = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="Init_lane0_m",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=BingoMemSymbol("mm_p4_m_vals", offset=0), dst_addr=l1_stage_a.view(lane_m(0)), size=4))
    init_lane0_l = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="Init_lane0_l",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=BingoMemSymbol("mm_p4_l_vals", offset=0), dst_addr=l1_stage_a.view(lane_l(0)), size=4))
    nodes += [init_lane0_m, init_lane0_l]

    # Cluster 0 REMOTE-reads clusters 1,2,3's own (m,l) into its staging beat, lanes 1,2,3.
    gather_m = {}
    gather_l = {}
    for i in (1, 2, 3):
        gather_m[i] = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
            node_name=f"Gather_lane{i}_m",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=l1_own[i].view(0), dst_addr=l1_stage_a.view(lane_m(i)), size=4))
        gather_l[i] = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
            node_name=f"Gather_lane{i}_l",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=l1_own[i].view(4), dst_addr=l1_stage_a.view(lane_l(i)), size=4))
        nodes += [gather_m[i], gather_l[i]]
        edges += [(init_own_m[i], gather_m[i]), (init_own_l[i], gather_l[i])]

    # The cross-cluster in-fabric fold: cluster 0 -> cluster 1, writer-ext armed (nvalid=4).
    merge_push = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="MergePush",
        kernel_name="__snax_bingo_kernel_xdma_moment_merge_push",
        kernel_args=SnaxBingoKernelXdmaMomentMergePushArgs(src_addr=l1_stage_a, dst_addr=l1_dest, nvalid=4))
    nodes.append(merge_push)
    for src in [init_lane0_m, init_lane0_l] + [gather_m[i] for i in (1, 2, 3)] + [gather_l[i] for i in (1, 2, 3)]:
        edges.append((src, merge_push))

    store_result = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Store_result",
        kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(src_addr=l1_dest, dst_addr=l3_result, size=8))
    check_result = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Check_result",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=BingoMemSymbol("mm_p4_golden_pair"),
            output_data_addr=l3_result,
            check_type=BINGO_CHECK_TYPE_FP32_TOL,
            tolerance=CHECK_TOLERANCE,
            num_elements=2,
            name="mm_p4_infabric"))
    nodes += [store_result, check_result]
    edges += [(merge_push, store_result), (store_result, check_result)]

    for n in nodes:
        dfg.bingo_add_node(n)
    for u, v in edges:
        dfg.bingo_add_edge(u, v)

    print(f"Built DFG: {len(nodes)} nodes, {len(edges)} edges "
          f"(3 producer inits + 1 local init + 6 gather + merge push + store + check)")

    dfg.bingo_compile_dfg(
        "F3 S3 P4 in-fabric moment-merge", args.output_dir, args.output_offload_file_name,
        extra_include_header_list=["momentmerge_p4_data.h"])
    print("Generated xdma_moment_merge_p4_infabric DFG")


if __name__ == "__main__":
    main()
