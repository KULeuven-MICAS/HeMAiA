#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# SPDX-License-Identifier: Apache-2.0
#
# The IN-FABRIC SAMPLER (tier-jump, `10-new-capabilities.md`): distributed token sampling as a monoid fold.
# Reuses StreamTopKMergeRt / __snax_bingo_kernel_xdma_topk_push UNCHANGED -- NO new RTL. Each shard pushes its
# local (perturbed_logit, token_id) list (Gumbel-perturbed) with accEn; the merger's slot folds them into the
# global sorted top-8 table, whose TOP-1 is the sampled token (Gumbel-max = argmax(logit+g) ~ softmax(logit)).
# Byte-exact check on the whole 64 B table verifies the sampled token (and the rest of the sorted table) is
# computed correctly in-fabric. Same DFG shape as xdma_topk_p2; only the data (perturbed logits) differs.

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

from sampler_datagen import emit_header_file, NUM_SHARDS, VOCAB_PER_SHARD  # noqa: E402
from bingo_dfg import BingoDFG  # noqa: E402
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa: E402
from bingo_node import BingoNode  # noqa: E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa: E402
from bingo_kernel_args import (  # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdmaTopKPushArgs,   # the sampler reuses the top-k monoid push
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
    BINGO_CHECK_TYPE_BYTE_EXACT,
)

DMA_CORE = 1
HOST_CORE = 2
BEAT_BYTES = 64
TOPK_KERNEL = "__snax_bingo_kernel_xdma_topk_push"


def get_args():
    p = argparse.ArgumentParser(description="xdma_sampler_p2")
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

    l1_dest = BingoMemAlloc("sampler_dest_c1", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=1)
    l3_result = BingoMemAlloc("sampler_result_l3", size=BEAT_BYTES, mem_level="L3")
    l1_src = {i: BingoMemAlloc(f"sampler_src_s{i}", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=i)
              for i in range(NUM_SHARDS)}

    nodes, edges, pushes = [], [], []
    for i in range(NUM_SHARDS):
        init = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=f"Init_s{i}_beat", kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=BingoMemSymbol("sampler_shard_beats", offset=BEAT_BYTES * i),
                dst_addr=l1_src[i], size=BEAT_BYTES))
        push = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=("Arm_s0" if i == 0 else f"Fold_s{i}"),
            kernel_name=TOPK_KERNEL,
            kernel_args=SnaxBingoKernelXdmaTopKPushArgs(
                src_addr=l1_src[i], dst_addr=l1_dest, nvalid=VOCAB_PER_SHARD,
                acc_en=1, acc_init=1 if i == 0 else 0, acc_slot=0))
        nodes += [init, push]
        edges += [(init, push)]
        if i > 0:
            edges.append((pushes[-1], push))
        pushes.append(push)

    store = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Store_result", kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(src_addr=l1_dest, dst_addr=l3_result, size=BEAT_BYTES))
    check = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Check_result", kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=BingoMemSymbol("sampler_golden_beat"), output_data_addr=l3_result,
            check_type=BINGO_CHECK_TYPE_BYTE_EXACT, tolerance=0,
            num_elements=BEAT_BYTES, name="sampler_p2"))
    nodes += [store, check]
    edges += [(pushes[-1], store), (store, check)]

    for n in nodes:
        dfg.bingo_add_node(n)
    for u, v in edges:
        dfg.bingo_add_edge(u, v)

    print(f"Built DFG: {len(nodes)} nodes, {len(edges)} edges (in-fabric sampler, {NUM_SHARDS} shards)")
    dfg.bingo_compile_dfg(
        "in-fabric sampler (distributed token sampling = a monoid fold)", args.output_dir,
        args.output_offload_file_name, extra_include_header_list=["sampler_data.h"])
    print("Generated xdma_sampler_p2 DFG")


if __name__ == "__main__":
    main()
