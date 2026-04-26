# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Test ALL xDMA data layout transformation operators in one DFG:
#   1. COPY          - 1D data copy
#   2. XDMA_6D       - generic AGU copy
#   3. TRANSPOSE     - [M,N] -> [N,M]
#   4. SUBMATRIX     - extract sub-region with start/end
#   5. EXPAND        - broadcast [1,N] -> [M,N]
#   6. CONCAT        - combine two halves -> full matrix
#   7. PAD           - zero-pad with borders
#   8. GATHER        - select every Nth row
#   9. A/B/D layout  - row-major <-> VersaCore blocked layouts
#
# Each test: IDMA load -> xDMA transform -> Host store -> Host check
# All tests run independently (no inter-test dependency).

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

from xdma_datagen import emit_header_file  # noqa E402
from bingo_dfg import BingoDFG  # noqa E402
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_kernel_args import (  # noqa E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdma1dCopyArgs,
    SnaxBingoKernelXdma6dArgs,
    SnaxBingoKernelXdmaTranspose2dArgs,
    SnaxBingoKernelXdmaSubmatrix2dArgs,
    SnaxBingoKernelXdmaExpand2dArgs,
    SnaxBingoKernelXdmaConcat2dArgs,
    SnaxBingoKernelXdmaPad2dArgs,
    SnaxBingoKernelXdmaGather2dArgs,
    SnaxBingoKernelXdmaDToRowMajorArgs,
    SnaxBingoKernelXdmaRowMajorToDArgs,
    SnaxBingoKernelXdmaRowMajorToAArgs,
    SnaxBingoKernelXdmaAToRowMajorArgs,
    SnaxBingoKernelXdmaRowMajorToBArgs,
    SnaxBingoKernelXdmaBToRowMajorArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)

DMA_CORE = 1
HOST_CORE = 2


def parse_layout_shape(hwcfg, array_shape):
    unrolling = hwcfg["snax_versacore_core_template"]["snax_acc_cfg"][0]\
                    ["snax_versacore_spatial_unrolling"][0][array_shape]
    return int(unrolling[0]), int(unrolling[1]), int(unrolling[2])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True,
                        help="Path to snax_versacore_to_cluster.hjson")
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True,
                        help="Path to generated occamy.h with HW platform defines")
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def add_xdma_test(dfg, name, l1_dst, xdma_node, golden_sym, out_size,
                  host_core_id):
    """Helper: add xdma_op -> store -> check chain."""

    # Store result to L3
    l3_out = BingoMemAlloc(f"out_{name}", size=out_size, mem_level="L3")
    store = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=host_core_id,
        node_name=f"Store_{name}",
        kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(l1_dst, l3_out, out_size))
    dfg.bingo_add_node(store)

    # Check golden
    check = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=host_core_id,
        node_name=f"Check_{name}",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(golden_sym, l3_out, out_size))
    dfg.bingo_add_node(check)

    # Wire: xdma -> store -> check
    dfg.bingo_add_edge(xdma_node, store)
    dfg.bingo_add_edge(store, check)

    return check


def add_l1_layout_test(dfg, name, src_sym, src_size, kernel_name, kernel_args_cls,
                       kernel_arg_kwargs_no_addrs, dst_size, golden_sym, prev_chk):
    l1_src = BingoMemAlloc(f"l1_{name}_src", size=src_size, mem_level="L1",
                           chip_id=0, cluster_id=0)
    l1_dst = BingoMemAlloc(f"l1_{name}_dst", size=dst_size, mem_level="L1",
                           chip_id=0, cluster_id=0)
    l3_dst = BingoMemAlloc(f"l3_{name}_dst", size=dst_size, mem_level="L3")

    load = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name=f"Load_{name}",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(src_sym, l1_src, src_size))
    convert = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name=f"Conv_{name}",
        kernel_name=kernel_name,
        kernel_args=kernel_args_cls(src_addr=l1_src, dst_addr=l1_dst,
                                    **kernel_arg_kwargs_no_addrs))
    store = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name=f"Store_{name}",
        kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(l1_dst, l3_dst, dst_size))
    check = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name=f"Check_{name}",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(golden_sym, l3_dst, dst_size,
                                                   name=f"L1_{name}"))

    for node in (load, convert, store, check):
        dfg.bingo_add_node(node)
    dfg.bingo_add_edge(prev_chk, load)
    dfg.bingo_add_edge(load, convert)
    dfg.bingo_add_edge(convert, store)
    dfg.bingo_add_edge(store, check)
    return check


def add_l3_direct_layout_test(dfg, name, src_sym, kernel_name, kernel_args_cls,
                              kernel_arg_kwargs_no_addrs, dst_size, golden_sym,
                              prev_chk):
    l3_dst = BingoMemAlloc(f"l3_direct_{name}_dst", size=dst_size, mem_level="L3")
    convert = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name=f"ConvL3_{name}",
        kernel_name=kernel_name,
        kernel_args=kernel_args_cls(src_addr=src_sym, dst_addr=l3_dst,
                                    **kernel_arg_kwargs_no_addrs))
    check = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name=f"CheckL3_{name}",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(golden_sym, l3_dst, dst_size,
                                                   name=f"L3_{name}"))

    for node in (convert, check):
        dfg.bingo_add_node(node)
    dfg.bingo_add_edge(prev_chk, convert)
    dfg.bingo_add_edge(convert, check)
    return check


def layout_conversions(M_T, K_T, N_T, meshRow, tileSize, meshCol, elem_bytes):
    A_kw = dict(M_T=M_T, K_T=K_T, meshRow=meshRow, tileSize=tileSize,
                elem_bytes=elem_bytes)
    B_kw = dict(K_T=K_T, N_T=N_T, tileSize=tileSize, meshCol=meshCol,
                elem_bytes=elem_bytes)
    D_kw = dict(M_T=M_T, N_T=N_T, meshRow=meshRow, meshCol=meshCol,
                elem_bytes=elem_bytes)
    A_bytes = M_T * meshRow * K_T * tileSize * elem_bytes
    B_bytes = K_T * tileSize * N_T * meshCol * elem_bytes
    D_bytes = M_T * meshRow * N_T * meshCol * elem_bytes
    return [
        ("row_to_A", "src_A_rm", A_bytes, "__snax_bingo_kernel_xdma_row_major_to_a",
         SnaxBingoKernelXdmaRowMajorToAArgs, A_kw, A_bytes, "golden_A_layout"),
        ("A_to_row", "src_A_layout", A_bytes, "__snax_bingo_kernel_xdma_a_to_row_major",
         SnaxBingoKernelXdmaAToRowMajorArgs, A_kw, A_bytes, "golden_A_rm"),
        ("row_to_B", "src_B_rm", B_bytes, "__snax_bingo_kernel_xdma_row_major_to_b",
         SnaxBingoKernelXdmaRowMajorToBArgs, B_kw, B_bytes, "golden_B_layout"),
        ("B_to_row", "src_B_layout", B_bytes, "__snax_bingo_kernel_xdma_b_to_row_major",
         SnaxBingoKernelXdmaBToRowMajorArgs, B_kw, B_bytes, "golden_B_rm"),
        ("row_to_D", "src_D_rm", D_bytes, "__snax_bingo_kernel_xdma_row_major_to_d",
         SnaxBingoKernelXdmaRowMajorToDArgs, D_kw, D_bytes, "golden_D_layout"),
        ("D_to_row", "src_D_layout", D_bytes, "__snax_bingo_kernel_xdma_d_to_row_major",
         SnaxBingoKernelXdmaDToRowMajorArgs, D_kw, D_bytes, "golden_D_rm"),
    ]


def main():
    args = get_args()
    with open(args.cfg) as f:
        param = hjson.loads(f.read())
    with open(args.hwcfg) as f:
        hwcfg = hjson.loads(f.read())

    array_shape = int(param["array_shape"])
    meshRow, tileSize, meshCol = parse_layout_shape(hwcfg, array_shape)
    merged_param = {**param, "meshRow": meshRow, "tileSize": tileSize, "meshCol": meshCol}

    rows = int(param["rows"])
    cols = int(param["cols"])
    elem = int(param["elem_bytes"])
    data_size = rows * cols * elem
    row_bytes = cols * elem

    # Emit data header
    if args.data_h is not None:
        with open(args.data_h, "w") as f:
            f.write(emit_header_file(**merged_param))
        print(f"Written data header: {args.data_h}")

    # DFG setup using HW params derived from occamy.h + RTL config
    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return
    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"])

    # Shared source: load input_data from L3 -> L1
    input_sym = BingoMemSymbol("input_data")
    l1_input = BingoMemAlloc("l1_input", size=data_size, mem_level="L1", chip_id=0, cluster_id=0)

    load_input = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="Load_input",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(input_sym, l1_input, data_size))
    dfg.bingo_add_node(load_input)

    # Serial test chain: each op runs after the previous one's check completes.
    # prev_chk tracks the last check node to enforce serial ordering.
    prev_chk = load_input

    # ── 1. COPY ──────────────────────────────────────────────────
    l1_copy_dst = BingoMemAlloc("l1_copy_dst", size=data_size, mem_level="L1", chip_id=0, cluster_id=0)
    xdma_copy = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="XDMA_copy",
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(l1_input, l1_copy_dst, data_size))
    dfg.bingo_add_node(xdma_copy)
    dfg.bingo_add_edge(prev_chk, xdma_copy)
    prev_chk = add_xdma_test(dfg, "copy", l1_copy_dst, xdma_copy, BingoMemSymbol("golden_copy"),
                             data_size, HOST_CORE)

    # ── 2. GENERIC 6D ────────────────────────────────────────────
    l1_6d_dst = BingoMemAlloc("l1_6d_dst", size=data_size, mem_level="L1", chip_id=0, cluster_id=0)
    xdma_6d = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="XDMA_6d",
        kernel_name="__snax_bingo_kernel_xdma_6d",
        kernel_args=SnaxBingoKernelXdma6dArgs(
            l1_input, l1_6d_dst,
            row_bytes, row_bytes,
            [8, row_bytes * 8], [row_bytes // 8, rows // 8],
            [8, row_bytes * 8], [row_bytes // 8, rows // 8]))
    dfg.bingo_add_node(xdma_6d)
    dfg.bingo_add_edge(prev_chk, xdma_6d)
    prev_chk = add_xdma_test(dfg, "xdma_6d", l1_6d_dst, xdma_6d,
                             BingoMemSymbol("golden_xdma_6d"), data_size, HOST_CORE)

    # ── 3. TRANSPOSE ─────────────────────────────────────────────
    # Software transpose (CPU-based): xDMA AGU can't reorder bytes within 8-byte channels.
    l1_trans_dst = BingoMemAlloc("l1_trans_dst", size=data_size, mem_level="L1", chip_id=0, cluster_id=0)
    xdma_transpose = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="XDMA_transpose",
        kernel_name="__snax_bingo_kernel_xdma_transpose_2d",
        kernel_args=SnaxBingoKernelXdmaTranspose2dArgs(l1_input, l1_trans_dst, rows, cols, elem))
    dfg.bingo_add_node(xdma_transpose)
    dfg.bingo_add_edge(prev_chk, xdma_transpose)
    prev_chk = add_xdma_test(dfg, "transpose", l1_trans_dst, xdma_transpose, BingoMemSymbol("golden_transpose"),
                             data_size, HOST_CORE)

    # ── 4. SUBMATRIX ─────────────────────────────────────────────
    # All slice boundaries must be multiples of 8.
    sub_rs, sub_re = 0, 8
    sub_cs, sub_ce = 8, cols
    sub_rows = sub_re - sub_rs
    sub_cols = sub_ce - sub_cs
    sub_out_size = sub_rows * sub_cols * elem
    l1_sub_dst = BingoMemAlloc("l1_sub_dst", size=sub_out_size, mem_level="L1", chip_id=0, cluster_id=0)
    xdma_submatrix = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="XDMA_submatrix",
        kernel_name="__snax_bingo_kernel_xdma_submatrix_2d",
        kernel_args=SnaxBingoKernelXdmaSubmatrix2dArgs(
            l1_input, l1_sub_dst, rows, cols, sub_rs, sub_re, sub_cs, sub_ce, elem))
    dfg.bingo_add_node(xdma_submatrix)
    dfg.bingo_add_edge(prev_chk, xdma_submatrix)
    prev_chk = add_xdma_test(dfg, "submatrix", l1_sub_dst, xdma_submatrix, BingoMemSymbol("golden_submatrix"),
                             sub_out_size, HOST_CORE)

    # ── 5. EXPAND ────────────────────────────────────────────────
    # Broadcast first row [1, cols] -> [rows, cols]
    # Expand kernel reads directly from l1_input (first row), no separate copy needed.
    l1_expand_dst = BingoMemAlloc("l1_expand_dst", size=data_size, mem_level="L1", chip_id=0, cluster_id=0)
    xdma_expand = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="XDMA_expand",
        kernel_name="__snax_bingo_kernel_xdma_expand_2d",
        kernel_args=SnaxBingoKernelXdmaExpand2dArgs(l1_input, l1_expand_dst, rows, cols, elem))
    dfg.bingo_add_node(xdma_expand)
    dfg.bingo_add_edge(prev_chk, xdma_expand)
    prev_chk = add_xdma_test(dfg, "expand", l1_expand_dst, xdma_expand, BingoMemSymbol("golden_expand"),
                             data_size, HOST_CORE)

    # ── 6. CONCAT ────────────────────────────────────────────────
    half = rows // 2
    half_size = half * cols * elem
    l1_concat_dst = BingoMemAlloc("l1_concat_dst", size=data_size, mem_level="L1", chip_id=0, cluster_id=0)

    # Concat top half (offset=0)
    xdma_concat_top = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="XDMA_concat_top",
        kernel_name="__snax_bingo_kernel_xdma_concat_2d",
        kernel_args=SnaxBingoKernelXdmaConcat2dArgs(
            l1_input, l1_concat_dst, half, cols, rows, cols, 0, 0, elem))
    dfg.bingo_add_node(xdma_concat_top)
    dfg.bingo_add_edge(prev_chk, xdma_concat_top)

    # Concat bottom half: load bottom half of input to separate L1 buffer
    l1_concat_src_bottom = BingoMemAlloc("l1_concat_src_bottom", size=half_size, mem_level="L1",
                                          chip_id=0, cluster_id=0)
    load_bottom = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="Load_input_bottom",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            BingoMemSymbol("input_data", offset=half_size), l1_concat_src_bottom, half_size))
    dfg.bingo_add_node(load_bottom)
    dfg.bingo_add_edge(xdma_concat_top, load_bottom)

    xdma_concat_bottom = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="XDMA_concat_bottom",
        kernel_name="__snax_bingo_kernel_xdma_concat_2d",
        kernel_args=SnaxBingoKernelXdmaConcat2dArgs(
            l1_concat_src_bottom, l1_concat_dst, half, cols, rows, cols, 0, half, elem))
    dfg.bingo_add_node(xdma_concat_bottom)
    dfg.bingo_add_edge(load_bottom, xdma_concat_bottom)
    prev_chk = add_xdma_test(dfg, "concat", l1_concat_dst, xdma_concat_bottom, BingoMemSymbol("golden_concat"),
                             data_size, HOST_CORE)

    # ── 7. PAD ───────────────────────────────────────────────────
    # All pad amounts must be multiples of 8.
    pt, pb, pl, pr = 8, 0, 0, 0
    padded_rows = rows + pt + pb
    padded_cols = cols + pl + pr
    padded_size = padded_rows * padded_cols * elem
    l1_pad_dst = BingoMemAlloc("l1_pad_dst", size=padded_size, mem_level="L1", chip_id=0, cluster_id=0)
    xdma_pad = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="XDMA_pad",
        kernel_name="__snax_bingo_kernel_xdma_pad_2d",
        kernel_args=SnaxBingoKernelXdmaPad2dArgs(l1_input, l1_pad_dst, rows, cols, pt, pb, pl, pr, elem))
    dfg.bingo_add_node(xdma_pad)
    dfg.bingo_add_edge(prev_chk, xdma_pad)
    prev_chk = add_xdma_test(dfg, "pad", l1_pad_dst, xdma_pad, BingoMemSymbol("golden_pad"),
                             padded_size, HOST_CORE)

    # ── 8. GATHER ────────────────────────────────────────────────
    g_start, g_stride = 0, 2
    g_count = rows // g_stride
    g_out_size = g_count * cols * elem
    l1_gather_dst = BingoMemAlloc("l1_gather_dst", size=g_out_size, mem_level="L1", chip_id=0, cluster_id=0)
    xdma_gather = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="XDMA_gather",
        kernel_name="__snax_bingo_kernel_xdma_gather_2d",
        kernel_args=SnaxBingoKernelXdmaGather2dArgs(
            l1_input, l1_gather_dst, rows, cols, g_count, g_start, g_stride, elem))
    dfg.bingo_add_node(xdma_gather)
    dfg.bingo_add_edge(prev_chk, xdma_gather)
    prev_chk = add_xdma_test(dfg, "gather", l1_gather_dst, xdma_gather, BingoMemSymbol("golden_gather"),
                             g_out_size, HOST_CORE)

    # ── 9. VERSACORE LAYOUT CONVERSIONS ─────────────────────────
    M_T = int(param["M_T"])
    K_T = int(param["K_T"])
    N_T = int(param["N_T"])
    convs = layout_conversions(M_T, K_T, N_T, meshRow, tileSize, meshCol, elem)

    for name, src, src_size, kernel_name, kernel_args_cls, kwargs, dst_size, golden in convs:
        prev_chk = add_l1_layout_test(
            dfg, name, BingoMemSymbol(src), src_size, kernel_name, kernel_args_cls,
            kwargs, dst_size, BingoMemSymbol(golden), prev_chk)

    for name, src, _src_size, kernel_name, kernel_args_cls, kwargs, dst_size, golden in convs:
        if name in ("row_to_A", "row_to_B", "row_to_D"):
            prev_chk = add_l3_direct_layout_test(
                dfg, name, BingoMemSymbol(src), kernel_name, kernel_args_cls,
                kwargs, dst_size, BingoMemSymbol(golden), prev_chk)

    # ── Compile DFG ──────────────────────────────────────────────
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    dfg.bingo_compile_dfg("xDMA All Operators Test", output_dir, args.output_offload_file_name,
                          extra_include_header_list=["xdma_data.h"])


if __name__ == "__main__":
    main()
