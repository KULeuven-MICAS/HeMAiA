#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# F3 -- P=12 cross-cluster moment-merge on a 2-CLUSTER platform, proving accEn removes the
# P<=8 lane wall. The stateless fold packs at most maxPairs=8 (m,l) lanes into one beat, so a
# stateless collective caps at 8 shards. accEn folds each ARRIVING beat into a persistent slot,
# so P is bounded only by the number of pushes. Here 12 shards (2 clusters x 6 pushes) > 8 are
# accumulated into ONE slot -- a collective the stateless path cannot express in one pass.
#
# Each shard's (m_i, l_i) is a single-lane beat; each push has nvalid=1, acc_en=1, acc_slot=0.
# Shard 0 arms the slot (acc_init=1); shards 1..11 fold (acc_init=0). All 12 pushes are
# SERIALIZED in a chain (arm -> fold1 -> ... -> fold11) so the slot's read-modify-write never
# races -- the online-softmax monoid is associative AND commutative, so the serial order does
# not change the result, it just removes any hazard.

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

from momentmerge_p12_datagen import emit_header_file, CHECK_TOLERANCE, NUM_SHARDS  # noqa: E402
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


def lane_m(k):
    return 4 * k


def lane_l(k):
    return 32 + 4 * k


def get_args():
    p = argparse.ArgumentParser(description="xdma_moment_merge_p12_accum")
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    p.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    p.add_argument("--platformcfg", type=pathlib.Path, required=True)
    p.add_argument("--data_h", type=pathlib.Path, default=None)
    return p.parse_args()


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

    P = platform["num_clusters_per_chiplet"]      # 2
    # shard -> cluster: split the 12 shards evenly across the P clusters.
    per = (NUM_SHARDS + P - 1) // P
    shard_cluster = [min(i // per, P - 1) for i in range(NUM_SHARDS)]

    # the merger's accumulator lands on cluster 1 (matches the other F3 arms).
    l1_dest = BingoMemAlloc("mm_p12_dest_c1", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=1)
    l3_result = BingoMemAlloc("mm_p12_result_l3", size=8, mem_level="L3")
    l1_src = {i: BingoMemAlloc(f"mm_p12_src_s{i}", size=BEAT_BYTES, mem_level="L1",
                               chip_id=0, cluster_id=shard_cluster[i]) for i in range(NUM_SHARDS)}

    nodes = []
    edges = []
    pushes = []

    for i in range(NUM_SHARDS):
        c = shard_cluster[i]
        init_m = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=c, assigned_core_id=DMA_CORE,
            node_name=f"Init_s{i}_m", kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=BingoMemSymbol("mm_p12_m_vals", offset=4 * i),
                dst_addr=l1_src[i].view(lane_m(0)), size=4))
        init_l = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=c, assigned_core_id=DMA_CORE,
            node_name=f"Init_s{i}_l", kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=BingoMemSymbol("mm_p12_l_vals", offset=4 * i),
                dst_addr=l1_src[i].view(lane_l(0)), size=4))
        push = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=c, assigned_core_id=DMA_CORE,
            node_name=("Arm_s0" if i == 0 else f"Fold_s{i}"),
            kernel_name="__snax_bingo_kernel_xdma_moment_merge_push",
            kernel_args=SnaxBingoKernelXdmaMomentMergePushArgs(
                src_addr=l1_src[i], dst_addr=l1_dest, nvalid=1,
                acc_en=1, acc_init=1 if i == 0 else 0, acc_slot=ACC_SLOT))
        nodes += [init_m, init_l, push]
        edges += [(init_m, push), (init_l, push)]
        if i > 0:
            edges.append((pushes[-1], push))   # chain: previous push must land before this one
        pushes.append(push)

    store = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Store_result", kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(src_addr=l1_dest, dst_addr=l3_result, size=8))
    check = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Check_result", kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=BingoMemSymbol("mm_p12_golden_pair"), output_data_addr=l3_result,
            check_type=BINGO_CHECK_TYPE_FP32_TOL, tolerance=CHECK_TOLERANCE,
            num_elements=2, name="mm_p12_accum"))
    nodes += [store, check]
    edges += [(pushes[-1], store), (store, check)]

    for n in nodes:
        dfg.bingo_add_node(n)
    for u, v in edges:
        dfg.bingo_add_edge(u, v)

    print(f"Built DFG: {len(nodes)} nodes, {len(edges)} edges "
          f"({NUM_SHARDS} shards on {P} clusters, chained accEn -- proves P>8 with no lane wall)")
    dfg.bingo_compile_dfg(
        "F3 P=12 accumulate (no P<=8 wall)", args.output_dir, args.output_offload_file_name,
        extra_include_header_list=["momentmerge_p12_data.h"])
    print("Generated xdma_moment_merge_p12_accum DFG")


if __name__ == "__main__":
    main()
