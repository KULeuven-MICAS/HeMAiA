#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# SPDX-License-Identifier: Apache-2.0
#
# The COMPLETE distributed flash-attention collective -- the (m, ell, O) triple (StreamAttnMergeRt). Two
# shards each own a KV block and push their local partial (m_k, ell_k, O_k) at the merger with accEn; the
# writer folds them in-transit into the global (m*, ell*, O*) -- the whole distributed attention (not just
# the softmax normalizer) computed in the fabric. The host checks (m*, ell*, O*) within LUT-exp tolerance
# and would do the final O*/ell* divide. Same DFG shape as the other monoids; the kernel is the attn push.

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

from attn_datagen import emit_header_file, NUM_SHARDS, D_HEAD, CHECK_TOLERANCE  # noqa: E402
from bingo_dfg import BingoDFG  # noqa: E402
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa: E402
from bingo_node import BingoNode  # noqa: E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa: E402
from bingo_kernel_args import (  # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdmaAttnPushArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
    BINGO_CHECK_TYPE_FP32_TOL,
)

DMA_CORE = 1
HOST_CORE = 2
BEAT_BYTES = 64
GOLDEN_ELEMS = 2 + D_HEAD          # (m*, ell*, O*[dHead]) fp32 values
ATTN_KERNEL = "__snax_bingo_kernel_xdma_attn_push"


def get_args():
    p = argparse.ArgumentParser(description="xdma_attn_p2")
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

    l1_dest = BingoMemAlloc("attn_dest_c1", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=1)
    l3_result = BingoMemAlloc("attn_result_l3", size=BEAT_BYTES, mem_level="L3")
    l1_src = {i: BingoMemAlloc(f"attn_src_s{i}", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=i)
              for i in range(NUM_SHARDS)}

    nodes, edges, pushes = [], [], []
    for i in range(NUM_SHARDS):
        init = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=f"Init_s{i}_partial", kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=BingoMemSymbol("attn_shard_beats", offset=BEAT_BYTES * i),
                dst_addr=l1_src[i], size=BEAT_BYTES))
        push = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=("Arm_s0" if i == 0 else f"Fold_s{i}"),
            kernel_name=ATTN_KERNEL,
            kernel_args=SnaxBingoKernelXdmaAttnPushArgs(
                src_addr=l1_src[i], dst_addr=l1_dest, nvalid=1,
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
            golden_data_addr=BingoMemSymbol("attn_golden"), output_data_addr=l3_result,
            check_type=BINGO_CHECK_TYPE_FP32_TOL, tolerance=CHECK_TOLERANCE,
            num_elements=GOLDEN_ELEMS, name="attn_p2"))
    nodes += [store, check]
    edges += [(pushes[-1], store), (store, check)]

    for n in nodes:
        dfg.bingo_add_node(n)
    for u, v in edges:
        dfg.bingo_add_edge(u, v)

    print(f"Built DFG: {len(nodes)} nodes, {len(edges)} edges (flash-attn (m,ell,O) triple, {NUM_SHARDS} shards)")
    dfg.bingo_compile_dfg(
        "complete distributed flash-attention (m,ell,O) triple", args.output_dir,
        args.output_offload_file_name, extra_include_header_list=["attn_data.h"])
    print("Generated xdma_attn_p2 DFG")


if __name__ == "__main__":
    main()
