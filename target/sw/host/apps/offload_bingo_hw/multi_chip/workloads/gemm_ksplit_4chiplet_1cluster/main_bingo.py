#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

"""
K-split GEMM test across 4 chiplets with one cluster per chiplet.

Each chiplet i computes one K chunk:
  Load A_i, load B_i, GEMM, store partial D_i to local L3,
  load local golden D_i, and check the local L3 result.

After local checks, remote partial D_i buffers are copied directly into a
shared chiplet 0 L3 receive slot.
Chiplet 0 performs the running int32 reductions, checks each running sum,
then dequantizes and checks the FP32 output.
"""

import argparse
import os
import pathlib
import sys

import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)

sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(current_dir)

from bingo_dfg import BingoDFG  # noqa E402
from bingo_helpers import chiplet_addr_transform_loc  # noqa E402
from bingo_kernel_args import (  # noqa E402
    HostBingoKernelCheckResultArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelInt32AddArgs,
    HostBingoKernelInt32DequantizeArgs,
    SnaxBingoKernelGemmFullArgs,
    SnaxBingoKernelIdma1dCopyArgs,
)
from bingo_mem_handle import BingoMemAlloc, BingoMemFixedAddr, BingoMemSymbol  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_platform import guard_chiplet_count, guard_cluster_count, parse_platform_cfg  # noqa E402
from ksplit_gemm_multi_chiplet_datagen import emit_header_file  # noqa E402


def get_args():
    parser = argparse.ArgumentParser(description="gemm_ksplit_4chiplet_1cluster")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def chip_hex(chiplet_id):
    return f"{chiplet_id:02x}"


def chiplet_full_addr_expr(chiplet_id, symbol_name):
    return f"(chiplet_addr_transform_full(0x{chiplet_id:02x}, (uint64_t){symbol_name}))"


def main():
    args = get_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(args.cfg) as f:
        param = hjson.loads(f.read())
    with open(args.hwcfg) as f:
        hw = hjson.loads(f.read())
    merged = {**param, **hw}

    if args.data_h is not None:
        content = emit_header_file(**merged, out_dir=os.path.join(args.output_dir, "build"))
        with open(args.data_h, "w") as f:
            f.write(content)
        print(f"Written data header: {args.data_h}")

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_chiplet_count(param, platform, args.output_dir, args.output_offload_file_name):
        return
    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return

    chiplets = platform["chiplet_ids"]
    reduction_chiplet = 0
    if reduction_chiplet not in chiplets:
        raise ValueError(f"chiplet 0 is required for reduction, platform chiplets={chiplets}")

    array_shape = merged["array_shape"]
    M = merged["M"]
    K = merged["K"]
    N = merged["N"]
    k_split = merged["k_split"]
    if k_split != 4:
        raise ValueError(f"gemm_ksplit_4chiplet_1cluster expects k_split=4, got {k_split}")
    if K % k_split != 0:
        raise ValueError(f"K ({K}) must be divisible by k_split ({k_split})")
    if len(chiplets) < k_split:
        raise ValueError(f"Need at least {k_split} chiplets, got {len(chiplets)}")

    transpose_A = merged.get("transposed_A", 0)
    transpose_B = merged.get("transposed_B", 0)
    accumPrevC = merged.get("accumPrevC", 0)

    data_type = merged.get("data_type", 0)
    snax_acc_cfg = merged["snax_versacore_core_template"]["snax_acc_cfg"][0]
    meshRow, tileSize, meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]

    K_tile = K // k_split
    A_tile_bytes = M * K_tile * meshRow * tileSize
    B_tile_bytes = K_tile * N * tileSize * meshCol
    D_num_elements = M * N * meshRow * meshCol
    D_bytes = D_num_elements * 4
    fp32_D_bytes = D_bytes
    if D_bytes % 64 != 0:
        raise ValueError(f"D_bytes ({D_bytes}) must be 64-byte aligned for partial transfers")

    mempool_base = chiplet_addr_transform_loc(2, 0, 0x8000_0000)
    A_mp_base = mempool_base
    B_mp_base = A_mp_base + k_split * A_tile_bytes
    golden_D_mp_base = B_mp_base + k_split * B_tile_bytes
    golden_sum_mp_base = golden_D_mp_base + k_split * D_bytes
    golden_fp32_mp_base = golden_sum_mp_base + (k_split - 1) * D_bytes
    combined_scale_mp_base = golden_fp32_mp_base + fp32_D_bytes

    GEMM_CORE = 0
    DMA_CORE = 1
    HOST_CORE = 2

    active_chiplets = chiplets[:k_split]

    mem_A = {}
    mem_B = {}
    mem_golden_D = {}
    for i in range(k_split):
        mem_A[i] = BingoMemFixedAddr(A_mp_base + i * A_tile_bytes)
        mem_B[i] = BingoMemFixedAddr(B_mp_base + i * B_tile_bytes)
        mem_golden_D[i] = BingoMemFixedAddr(golden_D_mp_base + i * D_bytes)

    mem_golden_sum = {
        i: BingoMemFixedAddr(golden_sum_mp_base + (i - 1) * D_bytes)
        for i in range(1, k_split)
    }
    mem_golden_fp32 = BingoMemFixedAddr(golden_fp32_mp_base)
    mem_combined_scale = BingoMemFixedAddr(combined_scale_mp_base)

    l1_A = {}
    l1_B = {}
    l1_D = {}
    l3_D_local = {}
    l3_check_scratch = {}
    l3_D_reduce = {}
    for i, chiplet in enumerate(active_chiplets):
        h = chip_hex(chiplet)
        l1_A[i] = BingoMemAlloc(f"l1_A_k{i}_chip{h}", size=A_tile_bytes, mem_level="L1",
                                chip_id=chiplet, cluster_id=0)
        l1_B[i] = BingoMemAlloc(f"l1_B_k{i}_chip{h}", size=B_tile_bytes, mem_level="L1",
                                chip_id=chiplet, cluster_id=0)
        l1_D[i] = BingoMemAlloc(f"l1_D_k{i}_chip{h}", size=D_bytes, mem_level="L1",
                                chip_id=chiplet, cluster_id=0)
        l3_D_local[i] = BingoMemAlloc(f"D_partial_k{i}_chip{h}_l3", size=D_bytes,
                                      mem_level="L3", chip_id=chiplet)
        l3_check_scratch[i] = BingoMemAlloc(f"check_scratch_k{i}_chip{h}_l3",
                                            size=D_bytes, mem_level="L3",
                                            chip_id=chiplet)
        if chiplet == reduction_chiplet:
            l3_D_reduce[i] = l3_D_local[i]
        else:
            l3_D_reduce[i] = BingoMemSymbol("D_partial_remote_chip00_l3")

    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    node_load_A = {}
    node_load_B = {}
    node_gemm = {}
    node_store_D = {}
    node_load_golden_D = {}
    node_check_D = {}
    node_copy_D_to_chip0 = {}
    reduce_ready = {}

    for i, chiplet in enumerate(active_chiplets):
        h = chip_hex(chiplet)
        node_load_A[i] = BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=0,
            assigned_core_id=DMA_CORE,
            node_name=f"Load_A_k{i}_Chip{h}",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_A[i],
                dst_addr=l1_A[i],
                size=A_tile_bytes,
            ),
        )
        node_load_B[i] = BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=0,
            assigned_core_id=DMA_CORE,
            node_name=f"Load_B_k{i}_Chip{h}",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_B[i],
                dst_addr=l1_B[i],
                size=B_tile_bytes,
            ),
        )
        node_gemm[i] = BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=0,
            assigned_core_id=GEMM_CORE,
            node_name=f"Gemm_k{i}_Chip{h}",
            kernel_name="__snax_bingo_kernel_gemm_full",
            kernel_args=SnaxBingoKernelGemmFullArgs(
                input_A_addr=l1_A[i],
                input_B_addr=l1_B[i],
                input_C_addr=0,
                output_D_addr=l1_D[i],
                M=M,
                K=K_tile,
                N=N,
                array_shape_idx=array_shape,
                transpose_A=transpose_A,
                transpose_B=transpose_B,
                accumPrevC=accumPrevC,
            ),
        )
        node_store_D[i] = BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=0,
            assigned_core_id=DMA_CORE,
            node_name=f"Store_D_k{i}_Chip{h}",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=l1_D[i],
                dst_addr=l3_D_local[i],
                size=D_bytes,
            ),
        )
        node_load_golden_D[i] = BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=0,
            assigned_core_id=HOST_CORE,
            node_name=f"Load_Golden_D_partial_k{i}_chip{h}",
            kernel_name="__host_bingo_kernel_idma",
            kernel_args=HostBingoKernelIdmaArgs(
                src_addr=mem_golden_D[i],
                dst_addr=l3_check_scratch[i],
                size=D_bytes,
            ),
        )
        node_check_D[i] = BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=0,
            assigned_core_id=HOST_CORE,
            node_name=f"Check_D_partial_k{i}_chip{h}",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=l3_check_scratch[i],
                output_data_addr=l3_D_local[i],
                data_size=D_bytes,
                name=f"D_partial_k{i}_chip{h}",
            ),
        )

        for node in [
            node_load_A[i],
            node_load_B[i],
            node_gemm[i],
            node_store_D[i],
            node_load_golden_D[i],
            node_check_D[i],
        ]:
            dfg.bingo_add_node(node)

        dfg.bingo_add_edge(node_load_A[i], node_load_B[i])
        dfg.bingo_add_edge(node_load_B[i], node_gemm[i])
        dfg.bingo_add_edge(node_gemm[i], node_store_D[i])
        dfg.bingo_add_edge(node_store_D[i], node_load_golden_D[i])
        dfg.bingo_add_edge(node_load_golden_D[i], node_check_D[i])

        if chiplet == reduction_chiplet:
            reduce_ready[i] = node_check_D[i]
        else:
            node_copy_D_to_chip0[i] = BingoNode(
                assigned_chiplet_id=chiplet,
                assigned_cluster_id=0,
                assigned_core_id=DMA_CORE,
                node_name=f"Copy_D_partial_k{i}_Chip{h}_to_Chip00_L3",
                kernel_name="__snax_bingo_kernel_idma_1d_copy",
                kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                    src_addr=l3_D_local[i],
                    dst_addr=chiplet_full_addr_expr(reduction_chiplet,
                                                   "D_partial_remote_chip00_l3"),
                    size=D_bytes,
                ),
            )
            dfg.bingo_add_node(node_copy_D_to_chip0[i])
            dfg.bingo_add_edge(node_check_D[i], node_copy_D_to_chip0[i])
            reduce_ready[i] = node_copy_D_to_chip0[i]

    # sequence the reduction on the reduction chiplet
    prev_sum = l3_D_reduce[0]
    prev_ready = reduce_ready[0]
    final_check = None
    for i in range(1, k_split):
        add = BingoNode(
            assigned_chiplet_id=reduction_chiplet,
            assigned_cluster_id=0,
            assigned_core_id=HOST_CORE,
            node_name=f"Add_k0_to_k{i}",
            kernel_name="__host_bingo_kernel_int32_add",
            kernel_args=HostBingoKernelInt32AddArgs(
                input_a_addr=prev_sum,
                input_b_addr=l3_D_reduce[i],
                output_addr=prev_sum,
                num_elements=D_num_elements,
            ),
        )
        node_load_golden_sum = BingoNode(
            assigned_chiplet_id=reduction_chiplet,
            assigned_cluster_id=0,
            assigned_core_id=HOST_CORE,
            node_name=f"Load_Golden_sum_k0_to_k{i}",
            kernel_name="__host_bingo_kernel_idma",
            kernel_args=HostBingoKernelIdmaArgs(
                src_addr=mem_golden_sum[i],
                dst_addr=l3_check_scratch[0],
                size=D_bytes,
            ),
        )
        final_check = BingoNode(
            assigned_chiplet_id=reduction_chiplet,
            assigned_cluster_id=0,
            assigned_core_id=HOST_CORE,
            node_name=f"Check_sum_k0_to_k{i}",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=l3_check_scratch[0],
                output_data_addr=prev_sum,
                data_size=D_bytes,
                name=f"sum_k0_to_k{i}",
            ),
        )
        dfg.bingo_add_node(add)
        dfg.bingo_add_node(node_load_golden_sum)
        dfg.bingo_add_node(final_check)
        if i in node_copy_D_to_chip0:
            dfg.bingo_add_edge(prev_ready, node_copy_D_to_chip0[i])
        dfg.bingo_add_edge(prev_ready, add)
        dfg.bingo_add_edge(reduce_ready[i], add)
        dfg.bingo_add_edge(add, node_load_golden_sum)
        dfg.bingo_add_edge(node_load_golden_sum, final_check)
        prev_ready = final_check

    node_dequant = BingoNode(
        assigned_chiplet_id=reduction_chiplet,
        assigned_cluster_id=0,
        assigned_core_id=HOST_CORE,
        node_name="Dequant_Chip00",
        kernel_name="__host_bingo_kernel_int32_dequantize",
        kernel_args=HostBingoKernelInt32DequantizeArgs(
            input_addr=prev_sum,
            output_addr=prev_sum,
            scale_addr=mem_combined_scale,
            num_elements=D_num_elements,
        ),
    )
    dfg.bingo_add_node(node_dequant)

    dfg.bingo_add_edge(final_check, node_dequant)

    node_load_golden_fp32 = BingoNode(
        assigned_chiplet_id=reduction_chiplet,
        assigned_cluster_id=0,
        assigned_core_id=HOST_CORE,
        node_name="Load_Golden_fp32_D",
        kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(
            src_addr=mem_golden_fp32,
            dst_addr=l3_check_scratch[0],
            size=fp32_D_bytes,
        ),
    )
    node_check_fp32 = BingoNode(
        assigned_chiplet_id=reduction_chiplet,
        assigned_cluster_id=0,
        assigned_core_id=HOST_CORE,
        node_name="Check_fp32_D",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=l3_check_scratch[0],
            output_data_addr=prev_sum,
            data_size=fp32_D_bytes,
            name="fp32_D",
        ),
    )
    dfg.bingo_add_node(node_load_golden_fp32)
    dfg.bingo_add_node(node_check_fp32)

    dfg.bingo_add_edge(node_dequant, node_load_golden_fp32)
    dfg.bingo_add_edge(node_load_golden_fp32, node_check_fp32)

    remote_chiplets = sum(1 for c in active_chiplets if c != reduction_chiplet)
    total_nodes = k_split * 6 + remote_chiplets + (k_split - 1) * 3 + 3
    print(f"Built DFG: {total_nodes} nodes ({k_split} chiplets, {remote_chiplets} direct remote partial copies to chiplet 0)")
    print(f"  active_chiplets={[chip_hex(c) for c in active_chiplets]}")
    print(f"  M={M}, K_tile={K_tile}, N={N}, k_split={k_split}")
    print(f"  A_tile_bytes={A_tile_bytes}, B_tile_bytes={B_tile_bytes}, D_bytes={D_bytes}")

    dfg.bingo_compile_dfg(
        "gemm_ksplit_4chiplet_1cluster",
        output_dir,
        args.output_offload_file_name,
        extra_include_header_list=["ksplit_gemm_data.h"],
    )

    return node_check_fp32


if __name__ == "__main__":
    main()
