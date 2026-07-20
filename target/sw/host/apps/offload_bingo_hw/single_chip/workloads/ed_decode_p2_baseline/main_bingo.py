#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# E-d BASELINE arm -- KV-sharded decode softmax epilogue with the cross-shard combine
# done by GATHER-THEN-LOCAL (the cores-do-the-merge baseline), NOT in-fabric.
#
# Identical per-shard prologue to ed_decode_p2_infabric: P clusters each compute their
# own (m_p, l_p) on the SIMD datapath and push both lanes into cluster 0's stage beat.
# The ONLY difference is the final combine: instead of one extension-armed push that
# folds in-transit, the assembled beat is
#   RawPush        -- a PLAIN cross-cluster copy (extension OFF) landing raw at cluster 1,
#   LocalMergePush -- a separate, purely LOCAL cluster-1 pass with the extension ON.
# So the merge runs on cluster 1's datapath as a distinct scheduled step. The delta
# between this arm and the in-fabric arm is exactly that extra local fold -- the offload
# win. Mirrors xdma_moment_merge_p2_ablation's structure.
#
# Per-shard chain (all on cluster p; xDMA on the DM core). B = S_p/32 beats.
#   Load   scores_p (BingoMemSymbol) -> cluster p L1                       [DM iDMA]
#   A  m_f16  = StreamReduce(scores_p, MAX)                 FP16 splat beat
#   A2 m_f32  = map_reduce(m_f16, LINEAR a=1, MAX, out_fp32)  FP32  <- collective m-lane
#   B  emn    = StreamMap(m_f16, EXP, a=-1.0)               exp(-m_p) FP16 splat
#   C  sum    = map_reduce(scores_p, EXP, ADD, out_fp32)    FP32 sum exp(s), UNSHIFTED
#   D  l_f32  = map_reduce(emn, LINEAR a=a_addr(sum), MAX, out_fp32)  FP32  <- collective l-lane
#            l_p = sum*exp(-m_p) = sum exp(s - m_p): the shift is applied via the FP32
#            a_addr operand, so exp(s) itself never leaves fp16 range.
#   Push_m  m_f32[0:4] -> stage.view(4*p)       (cluster p -> cluster 0)
#   Push_l  l_f32[0:4] -> stage.view(32+4*p)    (cluster p -> cluster 0)
# Then the existing collective: MergePush(stage, nvalid=P) -> dest (cluster 1) -> store -> check.

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

from ed_decode_datagen import emit_header_file, CHECK_TOLERANCE  # noqa: E402
from bingo_dfg import BingoDFG  # noqa: E402
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa: E402
from bingo_node import BingoNode  # noqa: E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa: E402
from bingo_kernel_args import (  # noqa: E402
    SnaxBingoKernelXdmaStreamReduceArgs,
    SnaxBingoKernelXdmaStreamMapArgs,
    SnaxBingoKernelXdmaStreamMapReduceArgs,
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

FUNC_LINEAR = 0
FUNC_EXP = 1
REDUCE_MAX = 0
REDUCE_ADD = 1

NEG_ONE_F32 = 0xBF800000  # -1.0f bit pattern (StreamMap 'a')


def lane_m(k):
    return 4 * k


def lane_l(k):
    return 32 + 4 * k


def get_args():
    p = argparse.ArgumentParser(description="ed_decode_p2_baseline")
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    p.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    p.add_argument("--platformcfg", type=pathlib.Path, required=True)
    p.add_argument("--data_h", type=pathlib.Path, default=None)
    return p.parse_args()


def build_shard_prologue(nodes, edges, p, B, scores_sym):
    """Emit shard p's per-shard softmax statistics on cluster p; return (m_f32, l_f32)."""
    scores = BingoMemAlloc(f"ed_scores_c{p}", size=B * BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=p)
    m_f16 = BingoMemAlloc(f"ed_m16_c{p}", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=p)
    m_f32 = BingoMemAlloc(f"ed_m32_c{p}", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=p)
    emn = BingoMemAlloc(f"ed_emn_c{p}", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=p)
    sum_f32 = BingoMemAlloc(f"ed_sum_c{p}", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=p)
    l_f32 = BingoMemAlloc(f"ed_l32_c{p}", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=p)

    load = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=p, assigned_core_id=DMA_CORE,
        node_name=f"Load_c{p}", kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=BingoMemSymbol(scores_sym), dst_addr=scores, size=B * BEAT_BYTES))
    a_m16 = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=p, assigned_core_id=DMA_CORE,
        node_name=f"A_m16_c{p}", kernel_name="__snax_bingo_kernel_xdma_stream_reduce",
        kernel_args=SnaxBingoKernelXdmaStreamReduceArgs(
            src_addr=scores, dst_addr=m_f16, beats=B, op=REDUCE_MAX, out_fp32=False))
    a2_m32 = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=p, assigned_core_id=DMA_CORE,
        node_name=f"A2_m32_c{p}", kernel_name="__snax_bingo_kernel_xdma_stream_map_reduce",
        kernel_args=SnaxBingoKernelXdmaStreamMapReduceArgs(
            src_addr=m_f16, dst_addr=m_f32, beats=1, func=FUNC_LINEAR, reduce_op=REDUCE_MAX,
            tap=False, out_fp32=True))
    b_emn = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=p, assigned_core_id=DMA_CORE,
        node_name=f"B_emn_c{p}", kernel_name="__snax_bingo_kernel_xdma_stream_map",
        kernel_args=SnaxBingoKernelXdmaStreamMapArgs(
            src_addr=m_f16, dst_addr=emn, beats=1, func=FUNC_EXP, a_f32bits=NEG_ONE_F32))
    c_sum = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=p, assigned_core_id=DMA_CORE,
        node_name=f"C_sum_c{p}", kernel_name="__snax_bingo_kernel_xdma_stream_map_reduce",
        kernel_args=SnaxBingoKernelXdmaStreamMapReduceArgs(
            src_addr=scores, dst_addr=sum_f32, beats=B, func=FUNC_EXP, reduce_op=REDUCE_ADD,
            tap=False, out_fp32=True))
    d_l32 = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=p, assigned_core_id=DMA_CORE,
        node_name=f"D_l32_c{p}", kernel_name="__snax_bingo_kernel_xdma_stream_map_reduce",
        kernel_args=SnaxBingoKernelXdmaStreamMapReduceArgs(
            src_addr=emn, dst_addr=l_f32, beats=1, func=FUNC_LINEAR, reduce_op=REDUCE_MAX,
            tap=False, out_fp32=True, a_addr=sum_f32))

    nodes += [load, a_m16, a2_m32, b_emn, c_sum, d_l32]
    edges += [(load, a_m16), (a_m16, a2_m32), (a_m16, b_emn), (load, c_sum),
              (b_emn, d_l32), (c_sum, d_l32)]
    return m_f32, l_f32, a2_m32, d_l32


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

    P = int(param["num_clusters"])
    S_p = int(param.get("seq_len", 2048)) // P
    B = S_p // 32

    stage = BingoMemAlloc("ed_stage", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=0)
    raw_b = BingoMemAlloc("ed_raw_c1", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=1)
    dest = BingoMemAlloc("ed_dest_c1", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=1)
    l3_result = BingoMemAlloc("ed_result_l3", size=8, mem_level="L3")

    nodes = []
    edges = []

    # Baseline combine: a PLAIN cross-cluster copy of the assembled beat (extension OFF)
    # then a separate LOCAL fold on cluster 1 (extension ON, local loopback). This is what
    # "the cores do the merge" costs -- one extra scheduled datapath pass on the receiver.
    raw_push = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="RawPush", kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(src_addr=stage, dst_addr=raw_b, size=BEAT_BYTES))
    local_merge = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=1, assigned_core_id=DMA_CORE,
        node_name="LocalMergePush", kernel_name="__snax_bingo_kernel_xdma_moment_merge_push",
        kernel_args=SnaxBingoKernelXdmaMomentMergePushArgs(src_addr=raw_b, dst_addr=dest, nvalid=P))

    for p in range(P):
        m_f32, l_f32, a2_node, d_node = build_shard_prologue(
            nodes, edges, p, B, f"ed_scores_shard{p}")
        push_m = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=p, assigned_core_id=DMA_CORE,
            node_name=f"Push_m_c{p}", kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=m_f32, dst_addr=stage.view(lane_m(p)), size=4))
        push_l = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=p, assigned_core_id=DMA_CORE,
            node_name=f"Push_l_c{p}", kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=l_f32, dst_addr=stage.view(lane_l(p)), size=4))
        nodes += [push_m, push_l]
        edges += [(a2_node, push_m), (d_node, push_l), (push_m, raw_push), (push_l, raw_push)]

    nodes += [raw_push, local_merge]
    edges += [(raw_push, local_merge)]
    merge_push = local_merge  # the node whose completion feeds Store_result

    store = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Store_result", kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(src_addr=dest, dst_addr=l3_result, size=8))
    check = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Check_result", kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=BingoMemSymbol("ed_golden_pair"), output_data_addr=l3_result,
            check_type=BINGO_CHECK_TYPE_FP32_TOL, tolerance=CHECK_TOLERANCE,
            num_elements=2, name="ed_decode_p2_baseline"))
    nodes += [store, check]
    edges += [(merge_push, store), (store, check)]

    for n in nodes:
        dfg.bingo_add_node(n)
    for u, v in edges:
        dfg.bingo_add_edge(u, v)

    print(f"Built DFG: {len(nodes)} nodes, {len(edges)} edges "
          f"(P={P}, S_p={S_p}, B={B}; in-fabric epilogue)")
    dfg.bingo_compile_dfg(
        "E-d KV-sharded decode softmax epilogue (baseline gather-then-local)", args.output_dir,
        args.output_offload_file_name, extra_include_header_list=["ed_decode_data.h"])
    print("Generated ed_decode_p2_baseline DFG")


if __name__ == "__main__":
    main()
