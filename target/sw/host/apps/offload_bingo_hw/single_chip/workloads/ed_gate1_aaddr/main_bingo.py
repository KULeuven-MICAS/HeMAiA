#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# E-d Gate 1 -- isolate the two untested mechanisms before the full decode epilogue:
#   (a) an out_fp32 MapReduce writing an FP32 scalar into the beat, and
#   (b) the runtime a_addr path (multiply operand read from an L1 word a prior node wrote).
#
#   SumExp : sum_f32 = map_reduce(EXP, ADD, out_fp32) over g1_scores   (G1_BEATS beats)
#   Cast   : l_f32   = map_reduce(LINEAR a=a_addr(sum_f32), MAX, out_fp32) over g1_ones (1 beat)
#   Store/Check : l_f32 == golden sum_exp   (LINEAR: 1.0*sum; MAX over identical lanes = sum)
#
# All device (xDMA) nodes on cluster 0's DM core; store/check on the host core.

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

from ed_gate1_datagen import emit_header_file, CHECK_TOLERANCE  # noqa: E402
from bingo_dfg import BingoDFG  # noqa: E402
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa: E402
from bingo_node import BingoNode  # noqa: E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa: E402
from bingo_kernel_args import (  # noqa: E402
    SnaxBingoKernelXdmaStreamMapReduceArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
    BINGO_CHECK_TYPE_FP32_TOL,
)

DMA_CORE = 1
HOST_CORE = 2
BEAT_BYTES = 64

# StreamMap func / StreamReduce op encodings.
FUNC_LINEAR = 0
FUNC_EXP = 1
REDUCE_MAX = 0
REDUCE_ADD = 1


def get_args():
    p = argparse.ArgumentParser(description="ed_gate1_aaddr")
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

    G1_BEATS = int(param.get("g1_beats", 4))  # N/32; default matches N=128 datagen

    scores = BingoMemAlloc("g1_scores_l1", size=G1_BEATS * BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=0)
    ones = BingoMemAlloc("g1_ones_l1", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=0)
    sum_f32 = BingoMemAlloc("g1_sum_l1", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=0)
    l_f32 = BingoMemAlloc("g1_l_l1", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=0)
    l3_result = BingoMemAlloc("g1_result_l3", size=4, mem_level="L3")

    nodes = []
    edges = []

    # Stage the two inputs into L1 (host -> cluster-0 L1).
    ld_scores = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Load_scores", kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(
            src_addr=BingoMemSymbol("g1_scores"), dst_addr=scores, size=G1_BEATS * BEAT_BYTES))
    ld_ones = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Load_ones", kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(
            src_addr=BingoMemSymbol("g1_ones"), dst_addr=ones, size=BEAT_BYTES))
    nodes += [ld_scores, ld_ones]

    # (a) SumExp: exp then add-reduce, FP32 scalar out.
    sumexp = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="SumExp", kernel_name="__snax_bingo_kernel_xdma_stream_map_reduce",
        kernel_args=SnaxBingoKernelXdmaStreamMapReduceArgs(
            src_addr=scores, dst_addr=sum_f32, beats=G1_BEATS,
            func=FUNC_EXP, reduce_op=REDUCE_ADD, tap=False, out_fp32=True))
    # (b) Cast/multiply: LINEAR with a read from a_addr=sum_f32; MAX over the ones beat.
    cast = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="Cast_aaddr", kernel_name="__snax_bingo_kernel_xdma_stream_map_reduce",
        kernel_args=SnaxBingoKernelXdmaStreamMapReduceArgs(
            src_addr=ones, dst_addr=l_f32, beats=1,
            func=FUNC_LINEAR, reduce_op=REDUCE_MAX, tap=False, out_fp32=True, a_addr=sum_f32))
    nodes += [sumexp, cast]
    edges += [(ld_scores, sumexp), (ld_ones, cast), (sumexp, cast)]  # sumexp before cast: a_addr dep

    store = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Store_result", kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(src_addr=l_f32, dst_addr=l3_result, size=4))
    check = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Check_result", kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=BingoMemSymbol("g1_golden"), output_data_addr=l3_result,
            check_type=BINGO_CHECK_TYPE_FP32_TOL, tolerance=CHECK_TOLERANCE,
            num_elements=1, name="ed_gate1"))
    nodes += [store, check]
    edges += [(cast, store), (store, check)]

    for n in nodes:
        dfg.bingo_add_node(n)
    for u, v in edges:
        dfg.bingo_add_edge(u, v)

    print(f"Built DFG: {len(nodes)} nodes, {len(edges)} edges (Gate 1: a_addr + out_fp32 cast idiom)")
    dfg.bingo_compile_dfg(
        "E-d Gate 1 (a_addr + FP32 cast)", args.output_dir, args.output_offload_file_name,
        extra_include_header_list=["ed_gate1_data.h"])
    print("Generated ed_gate1_aaddr DFG")


if __name__ == "__main__":
    main()
