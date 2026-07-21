#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# SPDX-License-Identifier: Apache-2.0
#
# The 3rd monoid of the in-fabric collective algebra: distributed MoE top-k expert routing
# (StreamTopKMergeRt). Two shards each push their local (logit, expert-id) candidate list at the merger
# with accumulate-on-arrival armed; the writer extension SORTS+MERGES them (a compare-exchange network,
# NOT a linear reduce -- genuinely beyond-SHARP) into a running global top-8 routing table. The host
# reads the sorted table and byte-exact-checks it against the numpy global top-8. This is the SAME accEn
# substrate the softmax moment-merge and the norm-stat merge use -- a third transformer distributed
# reduction, measured on real cross-cluster RTL.
#
# Beat layout matches the extension: value_k = lane k (byte 4k), index_k = lane (8+k) (byte 32+4k).
# Shard 0 arms the slot (acc_init=1); shard 1 folds (acc_init=0). nvalid = EXPERTS_PER_SHARD.

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

from topk_datagen import emit_header_file, NUM_SHARDS, EXPERTS_PER_SHARD  # noqa: E402
from bingo_dfg import BingoDFG  # noqa: E402
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa: E402
from bingo_node import BingoNode  # noqa: E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa: E402
from bingo_kernel_args import (  # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdmaTopKPushArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
    BINGO_CHECK_TYPE_BYTE_EXACT,
)

DMA_CORE = 1
HOST_CORE = 2
BEAT_BYTES = 64
TOPK_KERNEL = "__snax_bingo_kernel_xdma_topk_push"


def get_args():
    p = argparse.ArgumentParser(description="xdma_topk_p2")
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

    # merger = cluster 1; each shard's local candidate beat lives on its own cluster's L1
    l1_dest = BingoMemAlloc("topk_dest_c1", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=1)
    l3_result = BingoMemAlloc("topk_result_l3", size=BEAT_BYTES, mem_level="L3")
    l1_src = {i: BingoMemAlloc(f"topk_src_s{i}", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=i)
              for i in range(NUM_SHARDS)}

    nodes = []
    edges = []
    pushes = []
    for i in range(NUM_SHARDS):
        # stage each shard's pre-packed 64 B candidate beat (8 logits + 8 expert-ids) into its L1
        init = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=f"Init_s{i}_beat", kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=BingoMemSymbol("topk_shard_beats", offset=BEAT_BYTES * i),
                dst_addr=l1_src[i], size=BEAT_BYTES))
        # push the shard's candidate list at the merger; accEn folds it into the running global top-8
        push = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=("Arm_s0" if i == 0 else f"Fold_s{i}"),
            kernel_name=TOPK_KERNEL,
            kernel_args=SnaxBingoKernelXdmaTopKPushArgs(
                src_addr=l1_src[i], dst_addr=l1_dest, nvalid=EXPERTS_PER_SHARD,
                acc_en=1, acc_init=1 if i == 0 else 0, acc_slot=0))
        nodes += [init, push]
        edges += [(init, push)]
        if i > 0:
            edges.append((pushes[-1], push))   # arm-before-fold ordering (accInit=1 must precede folds)
        pushes.append(push)

    # copy the merged global top-8 routing table (64 B) to L3 and byte-exact check the whole table
    store = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Store_result", kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(src_addr=l1_dest, dst_addr=l3_result, size=BEAT_BYTES))
    check = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Check_result", kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=BingoMemSymbol("topk_golden_beat"), output_data_addr=l3_result,
            check_type=BINGO_CHECK_TYPE_BYTE_EXACT, tolerance=0,
            num_elements=BEAT_BYTES, name="topk_p2"))
    nodes += [store, check]
    edges += [(pushes[-1], store), (store, check)]

    for n in nodes:
        dfg.bingo_add_node(n)
    for u, v in edges:
        dfg.bingo_add_edge(u, v)

    print(f"Built DFG: {len(nodes)} nodes, {len(edges)} edges (top-k routing monoid, {NUM_SHARDS} shards)")
    dfg.bingo_compile_dfg(
        "collective algebra monoid 3 (distributed MoE top-k routing)", args.output_dir,
        args.output_offload_file_name, extra_include_header_list=["topk_data.h"])
    print("Generated xdma_topk_p2 DFG")


if __name__ == "__main__":
    main()
