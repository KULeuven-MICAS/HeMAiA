#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# F3 S3' -- P=4 cross-cluster in-fabric moment-merge with a PARALLEL (push) assemble.
#
# Same fold, same arithmetic, same golden as xdma_moment_merge_p4_infabric; the ONLY
# difference is the shape of the assemble phase, which is what that workload measures
# badly. There, cluster 0 REMOTE-READS each producer in turn, so the assemble is O(P)
# serialised work pinned to one DM core and it buries the flat fold. Here each producer
# writes its own (m_i,l_i) directly into lane i of cluster 0's staging beat, from its own
# DM core, so the P transfers are independent and concurrent.
#
# Why an assemble phase exists at all (it is NOT an artefact): StreamMomentMergeRt is
# stateless per beat and folds only the lanes of the single beat being written, has no
# TCDM read port, and splats its result across the full 512-bit beat -- so P independent
# armed pushes at one destination clobber rather than accumulate. Co-residency of the P
# partials in ONE beat is intrinsic to the mechanism. See the block comment in main().
#
# Run this against xdma_moment_merge_p4_infabric to separate "the fold is O(1) in P"
# from "the assemble was written serially".

# BEGIN WORKLOAD DESCRIPTION AND TASK GRAPH
# Task dependency graph:
#   Push_c0_lane0_m, Push_c0_lane0_l   (cluster 0, local write into its own staging beat)
#   Push_c1_lane1_m, Push_c1_lane1_l   (cluster 1 -> cluster 0, cross-cluster device write)
#   Push_c2_lane2_m, Push_c2_lane2_l   (cluster 2 -> cluster 0)
#   Push_c3_lane3_m, Push_c3_lane3_l   (cluster 3 -> cluster 0)
#   {all 8 pushes} -> MergePush -> Store_result -> Check_result
# All 8 pushes are mutually independent: no edges between them, each on its own cluster.
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


def lane_m(k):
    return 4 * k


def lane_l(k):
    return 32 + 4 * k


def get_args():
    parser = argparse.ArgumentParser(description="xdma_moment_merge_p4_pushgather")
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
    l1_dest = BingoMemAlloc("mm_p4_dest_c1", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=1)
    l3_result = BingoMemAlloc("mm_p4_result_l3", size=8, mem_level="L3")

    nodes = []
    edges = []

    # ---------------------------------------------------------------------------------
    # PUSH gather (this workload) vs PULL gather (xdma_moment_merge_p4_infabric).
    #
    # The fold itself is O(1) in P and cannot be otherwise: StreamMomentMergeRt is
    # STATELESS PER BEAT (StreamMomentMergeRt.scala:82) -- its fold tree reads only the
    # lanes of the ONE beat being written (:64-80), it has no TCDM read port at all
    # (XDMADataPath.scala gives tcdmWriter a `req` but no `rsp`), and its output is the
    # merged pair SPLATTED across the full 512-bit beat (:106-107). So P independent
    # armed pushes at one destination CLOBBER each other rather than accumulate, and
    # co-residency of the P partials in a single beat is intrinsic to the mechanism.
    # An assemble phase therefore always exists; the only question is its SHAPE.
    #
    # p4_infabric assembles by having cluster 0 REMOTE-READ each producer in turn --
    # 6 nodes all pinned to cluster 0's DM core, i.e. O(P) serialised work on one core,
    # which dominates the measurement and buries the flat fold.
    #
    # Here each producer instead WRITES ITS OWN (m,l) straight into its assigned lane of
    # cluster 0's staging beat, from its OWN DM core, so the P transfers are independent
    # and concurrent. No RTL change, no new kernel -- only the DAG shape differs. This
    # also drops the per-producer L1 staging buffers and the 6 pull nodes entirely
    # (8 concurrent 4B writes replace 2 local writes + 6 serialised remote reads).
    #
    # Ordering is safe without a hand-rolled barrier: a Bingo node's `done` fires only
    # after the transfer has LANDED (bingo.h:352 writes done after the kernel returns;
    # the kernel blocks in xdma_wait_task; a remote task retires on XDMA_FINISH_REMOTE,
    # driven by xdma_finish_o from the DESTINATION). So MergePush cannot start until all
    # P lane writes are actually resident in the staging beat.
    # ---------------------------------------------------------------------------------
    push_m = {}
    push_l = {}
    for i in range(4):
        # Cluster i writes lane i of cluster 0's staging beat. i == 0 is a local write;
        # i in 1..3 is a genuine cross-cluster device write.
        push_m[i] = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=f"Push_c{i}_lane{i}_m",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=BingoMemSymbol("mm_p4_m_vals", offset=4 * i),
                dst_addr=l1_stage_a.view(lane_m(i)), size=4))
        push_l[i] = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=f"Push_c{i}_lane{i}_l",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=BingoMemSymbol("mm_p4_l_vals", offset=4 * i),
                dst_addr=l1_stage_a.view(lane_l(i)), size=4))
        nodes += [push_m[i], push_l[i]]

    # The cross-cluster in-fabric fold: cluster 0 -> cluster 1, writer-ext armed (nvalid=4).
    merge_push = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="MergePush",
        kernel_name="__snax_bingo_kernel_xdma_moment_merge_push",
        kernel_args=SnaxBingoKernelXdmaMomentMergePushArgs(src_addr=l1_stage_a, dst_addr=l1_dest, nvalid=4))
    nodes.append(merge_push)
    for i in range(4):
        edges += [(push_m[i], merge_push), (push_l[i], merge_push)]

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
            name="mm_p4_pushgather"))
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
