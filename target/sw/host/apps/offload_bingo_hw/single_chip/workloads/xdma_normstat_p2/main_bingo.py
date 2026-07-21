#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# SPDX-License-Identifier: Apache-2.0
#
# The 2nd monoid of the in-fabric collective algebra: distributed LayerNorm/RMSNorm statistics
# (StreamNormStatMergeRt). Two shards each push their local (Σx, Σx²) at the merger with
# accumulate-on-arrival armed; the writer extension folds them (a linear pair-add) into a running
# slot -- the SAME accEn substrate the softmax moment-merge uses, a different monoid. Proves the
# substrate generalizes to a second transformer distributed reduction, measured on real RTL.
#
# Beat layout matches the extension: Σx at lane 0 (byte 0), Σx² at lane 8 (byte 32); nvalid=1.
# Shard 0 arms the slot (acc_init=1); shard 1 folds (acc_init=0). Check the merged (Σx, Σx²).

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

from normstat_datagen import emit_header_file, CHECK_TOLERANCE, NUM_SHARDS  # noqa: E402
from bingo_dfg import BingoDFG  # noqa: E402
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa: E402
from bingo_node import BingoNode  # noqa: E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa: E402
from bingo_kernel_args import (  # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdmaMomentMergePushArgs,   # reused: identical arg struct
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
    BINGO_CHECK_TYPE_FP32_TOL,
)

DMA_CORE = 1
HOST_CORE = 2
BEAT_BYTES = 64
NORMSTAT_KERNEL = "__snax_bingo_kernel_xdma_normstat_push"


def lane_sx(k):   # Σx at lane k
    return 4 * k


def lane_sxx(k):  # Σx² at lane 8+k
    return 32 + 4 * k


def get_args():
    p = argparse.ArgumentParser(description="xdma_normstat_p2")
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

    l1_dest = BingoMemAlloc("normstat_dest_c1", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=1)
    l3_result = BingoMemAlloc("normstat_result_l3", size=8, mem_level="L3")
    l1_src = {i: BingoMemAlloc(f"normstat_src_s{i}", size=BEAT_BYTES, mem_level="L1", chip_id=0, cluster_id=i)
              for i in range(NUM_SHARDS)}

    nodes = []
    edges = []
    pushes = []
    for i in range(NUM_SHARDS):
        init_sx = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=f"Init_s{i}_sx", kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=BingoMemSymbol("normstat_sx_vals", offset=4 * i),
                dst_addr=l1_src[i].view(lane_sx(0)), size=4))
        init_sxx = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=f"Init_s{i}_sxx", kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=BingoMemSymbol("normstat_sxx_vals", offset=4 * i),
                dst_addr=l1_src[i].view(lane_sxx(0)), size=4))
        push = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=("Arm_s0" if i == 0 else f"Fold_s{i}"),
            kernel_name=NORMSTAT_KERNEL,
            kernel_args=SnaxBingoKernelXdmaMomentMergePushArgs(
                src_addr=l1_src[i], dst_addr=l1_dest, nvalid=1,
                acc_en=1, acc_init=1 if i == 0 else 0, acc_slot=0))
        nodes += [init_sx, init_sxx, push]
        edges += [(init_sx, push), (init_sxx, push)]
        if i > 0:
            edges.append((pushes[-1], push))
        pushes.append(push)

    store = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Store_result", kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(src_addr=l1_dest, dst_addr=l3_result, size=8))
    check = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Check_result", kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=BingoMemSymbol("normstat_golden_pair"), output_data_addr=l3_result,
            check_type=BINGO_CHECK_TYPE_FP32_TOL, tolerance=CHECK_TOLERANCE,
            num_elements=2, name="normstat_p2"))
    nodes += [store, check]
    edges += [(pushes[-1], store), (store, check)]

    for n in nodes:
        dfg.bingo_add_node(n)
    for u, v in edges:
        dfg.bingo_add_edge(u, v)

    print(f"Built DFG: {len(nodes)} nodes, {len(edges)} edges (norm-stat monoid, {NUM_SHARDS} shards)")
    dfg.bingo_compile_dfg(
        "collective algebra monoid 2 (distributed norm stats)", args.output_dir,
        args.output_offload_file_name, extra_include_header_list=["normstat_data.h"])
    print("Generated xdma_normstat_p2 DFG")


if __name__ == "__main__":
    main()
