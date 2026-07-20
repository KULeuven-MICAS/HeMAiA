#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# F3 S3'' -- P=4 cross-cluster in-fabric moment-merge with NO ASSEMBLE PHASE.
#
# This is the third point of the assemble-shape ablation, and the only one that is a true
# "each producer pushes straight at the merger" collective:
#
#   xdma_moment_merge_p4_infabric    PULL gather  -- cluster 0 remote-READS each producer in
#                                    turn; O(P) serialised work pinned to one DM core.
#   xdma_moment_merge_p4_pushgather  PUSH gather  -- each producer WRITES its (m,l) into its
#                                    lane of cluster 0's staging beat; P concurrent 4B writes,
#                                    then ONE nvalid=4 fold.
#   xdma_moment_merge_p4_accum       NO gather    -- (this) each producer pushes its own (m,l)
#                                    DIRECTLY at the merger with accumulate-on-arrival armed;
#                                    the receiver folds each arrival into a persistent slot.
#
# The first two are forced by the base RTL: the fold is stateless per beat, so the P partials
# must be co-resident in ONE beat and an assemble phase is unavoidable (and P <= 8 lanes).
# accEn (snax_cluster@5e17bd6f) adds persistent (m,l) accumulators to StreamMomentMergeRt, so
# a push folds INTO the running value instead of clobbering it. That removes the assemble
# entirely AND the P<=8 wall: P is now bounded only by how many producers exist.
#
# Arming: accInit=1 OVERWRITES the slot, so exactly one producer arms it and that push must be
# ORDERED BEFORE the folding pushes. Cluster 0 arms; clusters 1..3 fold and are mutually
# concurrent (no edges among them). The monoid is associative and commutative, so the order in
# which the three folding pushes land does not matter.

# BEGIN WORKLOAD DESCRIPTION AND TASK GRAPH
# Task dependency graph:
#   Arm_c0      (cluster 0 -> merger, acc_en=1 acc_init=1 acc_slot=0)   nvalid=1
#     -> Fold_c1 (cluster 1 -> merger, acc_en=1 acc_init=0 acc_slot=0)  nvalid=1
#     -> Fold_c2 (cluster 2 -> merger, acc_en=1 acc_init=0 acc_slot=0)  nvalid=1
#     -> Fold_c3 (cluster 3 -> merger, acc_en=1 acc_init=0 acc_slot=0)  nvalid=1
#   {Fold_c1, Fold_c2, Fold_c3} -> Store_result -> Check_result
# Fold_c1/c2/c3 are mutually independent; only the arming edge is required.
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
ACC_SLOT = 0
NUM_PRODUCERS = 4


def lane_m(k):
    return 4 * k


def lane_l(k):
    return 32 + 4 * k


def get_args():
    parser = argparse.ArgumentParser(description="xdma_moment_merge_p4_accum")
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

    # Each producer needs its own single-lane source beat: the push sends ONE 64B beat whose lane 0
    # carries (m_i) and lane 8 carries (l_i); nvalid=1 masks every other lane to the monoid identity.
    l1_src = {i: BingoMemAlloc(f"mm_p4_src_c{i}", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=i)
              for i in range(NUM_PRODUCERS)}
    # The merger's accumulator lands here (cluster 1, matching the other two P=4 arms).
    l1_dest = BingoMemAlloc("mm_p4_dest_c1", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=1)
    l3_result = BingoMemAlloc("mm_p4_result_l3", size=8, mem_level="L3")

    nodes = []
    edges = []

    # Stage each producer's own (m_i, l_i) into lane 0 / lane 8 of its LOCAL source beat.
    init_m = {}
    init_l = {}
    for i in range(NUM_PRODUCERS):
        init_m[i] = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=f"Init_c{i}_m",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=BingoMemSymbol("mm_p4_m_vals", offset=4 * i),
                dst_addr=l1_src[i].view(lane_m(0)), size=4))
        init_l[i] = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=f"Init_c{i}_l",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=BingoMemSymbol("mm_p4_l_vals", offset=4 * i),
                dst_addr=l1_src[i].view(lane_l(0)), size=4))
        nodes += [init_m[i], init_l[i]]

    # The collective itself: P direct pushes at the merger, folded on arrival. No assemble node exists.
    push = {}
    for i in range(NUM_PRODUCERS):
        is_arm = (i == 0)
        push[i] = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=("Arm_c0" if is_arm else f"Fold_c{i}"),
            kernel_name="__snax_bingo_kernel_xdma_moment_merge_push",
            kernel_args=SnaxBingoKernelXdmaMomentMergePushArgs(
                src_addr=l1_src[i], dst_addr=l1_dest, nvalid=1,
                acc_en=1, acc_init=1 if is_arm else 0, acc_slot=ACC_SLOT))
        nodes.append(push[i])
        edges += [(init_m[i], push[i]), (init_l[i], push[i])]

    # Only the arming edge is required; the three folding pushes are mutually concurrent.
    for i in range(1, NUM_PRODUCERS):
        edges.append((push[0], push[i]))

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
            name="mm_p4_accum"))
    nodes += [store_result, check_result]
    # The result is complete only once the LAST folding push has landed.
    for i in range(1, NUM_PRODUCERS):
        edges.append((push[i], store_result))
    edges.append((store_result, check_result))

    for n in nodes:
        dfg.bingo_add_node(n)
    for u, v in edges:
        dfg.bingo_add_edge(u, v)

    print(f"Built DFG: {len(nodes)} nodes, {len(edges)} edges "
          f"({NUM_PRODUCERS} producer stages x2 + {NUM_PRODUCERS} direct pushes + store + check; "
          f"NO assemble phase)")

    dfg.bingo_compile_dfg(
        "F3 S3 P4 in-fabric moment-merge, accumulate-on-arrival (no assemble)",
        args.output_dir, args.output_offload_file_name,
        extra_include_header_list=["momentmerge_p4_data.h"])
    print("Generated xdma_moment_merge_p4_accum DFG")


if __name__ == "__main__":
    main()
