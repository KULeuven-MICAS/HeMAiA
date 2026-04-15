# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Test ALL xDMA data layout transformation operators in one DFG:
#   1. COPY          - 1D data copy
#   2. TRANSPOSE     - [M,N] -> [N,M]
#   3. SUBMATRIX     - extract sub-region with start/end
#   4. EXPAND        - broadcast [1,N] -> [M,N]
#   5. CONCAT        - combine two halves -> full matrix
#   6. PAD           - zero-pad with borders
#   7. GATHER        - select every Nth row
#
# Each test: IDMA load -> xDMA transform -> Host store -> Host check
# All tests run independently (no inter-test dependency).

import os
import sys
import re
import argparse
import pathlib
import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(current_dir)

from xdma_datagen import emit_header_file  # noqa E402
from bingo_dfg import BingoDFG  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_kernel_args import (  # noqa E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdma1dCopyArgs,
    SnaxBingoKernelXdmaTranspose2dArgs,
    SnaxBingoKernelXdmaSubmatrix2dArgs,
    SnaxBingoKernelXdmaExpand2dArgs,
    SnaxBingoKernelXdmaConcat2dArgs,
    SnaxBingoKernelXdmaPad2dArgs,
    SnaxBingoKernelXdmaGather2dArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)


def parse_platform_cfg(occamy_h_path, rtlcfg_path):
    """Parse HW platform params from generated occamy.h and RTL config hjson.
    Chiplet IDs are coordinate-encoded: (x, y) -> (x << 4) | y.
    """
    defines = {}
    with open(occamy_h_path) as f:
        for line in f:
            m = re.match(r'#define\s+(\w+)\s+(\d+)', line)
            if m:
                defines[m.group(1)] = int(m.group(2))
    with open(rtlcfg_path) as f:
        rtlcfg = hjson.loads(f.read())
    multichip = rtlcfg["hemaia_multichip"]
    if multichip["single_chip"]:
        chiplet_ids = [0x00]
    else:
        chiplet_ids = [(c["coordinate"][0] << 4) | c["coordinate"][1]
                       for c in multichip["testbench_cfg"]["hemaia_compute_chip"]]
    return {
        "num_chiplets": defines["N_CHIPLETS"],
        "num_clusters_per_chiplet": defines["N_CLUSTERS_PER_CHIPLET"],
        "num_cores_per_cluster": defines["N_CORES_PER_CLUSTER"],
        "chiplet_ids": chiplet_ids,
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True,
                        help="Path to generated occamy.h with HW platform defines")
    parser.add_argument("--rtlcfg", type=pathlib.Path, required=True,
                        help="Path to hemaia RTL config hjson")
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def add_xdma_test(dfg, name, l1_dst, xdma_node, golden_sym, out_size,
                  data_size, dma_core_id, host_core_id, prev_nodes):
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


def main():
    args = get_args()
    with open(args.cfg) as f:
        param = hjson.loads(f.read())

    rows = int(param["rows"])
    cols = int(param["cols"])
    elem = int(param["elem_bytes"])
    data_size = rows * cols * elem

    # Emit data header
    if args.data_h is not None:
        with open(args.data_h, "w") as f:
            f.write(emit_header_file(**param))
        print(f"Written data header: {args.data_h}")

    # DFG setup using HW params derived from occamy.h + RTL config
    platform = parse_platform_cfg(args.platformcfg, args.rtlcfg)
    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"])
    dma_core = 1
    host_core = 2

    # Shared source: load input_data from L3 -> L1
    input_sym = BingoMemSymbol("input_data")
    l1_input = BingoMemAlloc("l1_input", size=data_size, mem_level="L1", chip_id=0, cluster_id=0)

    load_input = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
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
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
        node_name="XDMA_copy",
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(l1_input, l1_copy_dst, data_size))
    dfg.bingo_add_node(xdma_copy)
    dfg.bingo_add_edge(prev_chk, xdma_copy)
    prev_chk = add_xdma_test(dfg, "copy", l1_copy_dst, xdma_copy, BingoMemSymbol("golden_copy"),
                             data_size, data_size, dma_core, host_core, [load_input])

    # ── 2. TRANSPOSE ─────────────────────────────────────────────
    # Software transpose (CPU-based): xDMA AGU can't reorder bytes within 8-byte channels.
    l1_trans_dst = BingoMemAlloc("l1_trans_dst", size=data_size, mem_level="L1", chip_id=0, cluster_id=0)
    xdma_transpose = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
        node_name="XDMA_transpose",
        kernel_name="__snax_bingo_kernel_xdma_transpose_2d",
        kernel_args=SnaxBingoKernelXdmaTranspose2dArgs(l1_input, l1_trans_dst, rows, cols, elem))
    dfg.bingo_add_node(xdma_transpose)
    dfg.bingo_add_edge(prev_chk, xdma_transpose)
    prev_chk = add_xdma_test(dfg, "transpose", l1_trans_dst, xdma_transpose, BingoMemSymbol("golden_transpose"),
                             data_size, data_size, dma_core, host_core, [load_input])

    # ── 3. SUBMATRIX ─────────────────────────────────────────────
    # All slice boundaries must be multiples of 8.
    sub_rs, sub_re = 0, 8
    sub_cs, sub_ce = 8, cols
    sub_rows = sub_re - sub_rs
    sub_cols = sub_ce - sub_cs
    sub_out_size = sub_rows * sub_cols * elem
    l1_sub_dst = BingoMemAlloc("l1_sub_dst", size=sub_out_size, mem_level="L1", chip_id=0, cluster_id=0)
    xdma_submatrix = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
        node_name="XDMA_submatrix",
        kernel_name="__snax_bingo_kernel_xdma_submatrix_2d",
        kernel_args=SnaxBingoKernelXdmaSubmatrix2dArgs(
            l1_input, l1_sub_dst, rows, cols, sub_rs, sub_re, sub_cs, sub_ce, elem))
    dfg.bingo_add_node(xdma_submatrix)
    dfg.bingo_add_edge(prev_chk, xdma_submatrix)
    prev_chk = add_xdma_test(dfg, "submatrix", l1_sub_dst, xdma_submatrix, BingoMemSymbol("golden_submatrix"),
                             sub_out_size, data_size, dma_core, host_core, [load_input])

    # ── 4. EXPAND ────────────────────────────────────────────────
    # Broadcast first row [1, cols] -> [rows, cols]
    # Expand kernel reads directly from l1_input (first row), no separate copy needed.
    l1_expand_dst = BingoMemAlloc("l1_expand_dst", size=data_size, mem_level="L1", chip_id=0, cluster_id=0)
    xdma_expand = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
        node_name="XDMA_expand",
        kernel_name="__snax_bingo_kernel_xdma_expand_2d",
        kernel_args=SnaxBingoKernelXdmaExpand2dArgs(l1_input, l1_expand_dst, rows, cols, elem))
    dfg.bingo_add_node(xdma_expand)
    dfg.bingo_add_edge(prev_chk, xdma_expand)
    prev_chk = add_xdma_test(dfg, "expand", l1_expand_dst, xdma_expand, BingoMemSymbol("golden_expand"),
                             data_size, data_size, dma_core, host_core, [load_input])

    # ── 5. CONCAT ────────────────────────────────────────────────
    half = rows // 2
    half_size = half * cols * elem
    l1_concat_dst = BingoMemAlloc("l1_concat_dst", size=data_size, mem_level="L1", chip_id=0, cluster_id=0)

    # Concat top half (offset=0)
    xdma_concat_top = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
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
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
        node_name="Load_input_bottom",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            BingoMemSymbol("input_data", offset=half_size), l1_concat_src_bottom, half_size))
    dfg.bingo_add_node(load_bottom)
    dfg.bingo_add_edge(xdma_concat_top, load_bottom)

    xdma_concat_bottom = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
        node_name="XDMA_concat_bottom",
        kernel_name="__snax_bingo_kernel_xdma_concat_2d",
        kernel_args=SnaxBingoKernelXdmaConcat2dArgs(
            l1_concat_src_bottom, l1_concat_dst, half, cols, rows, cols, 0, half, elem))
    dfg.bingo_add_node(xdma_concat_bottom)
    dfg.bingo_add_edge(load_bottom, xdma_concat_bottom)
    prev_chk = add_xdma_test(dfg, "concat", l1_concat_dst, xdma_concat_bottom, BingoMemSymbol("golden_concat"),
                             data_size, data_size, dma_core, host_core, [load_input])

    # ── 6. PAD ───────────────────────────────────────────────────
    # All pad amounts must be multiples of 8.
    pt, pb, pl, pr = 8, 0, 0, 0
    padded_rows = rows + pt + pb
    padded_cols = cols + pl + pr
    padded_size = padded_rows * padded_cols * elem
    l1_pad_dst = BingoMemAlloc("l1_pad_dst", size=padded_size, mem_level="L1", chip_id=0, cluster_id=0)
    xdma_pad = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
        node_name="XDMA_pad",
        kernel_name="__snax_bingo_kernel_xdma_pad_2d",
        kernel_args=SnaxBingoKernelXdmaPad2dArgs(l1_input, l1_pad_dst, rows, cols, pt, pb, pl, pr, elem))
    dfg.bingo_add_node(xdma_pad)
    dfg.bingo_add_edge(prev_chk, xdma_pad)
    prev_chk = add_xdma_test(dfg, "pad", l1_pad_dst, xdma_pad, BingoMemSymbol("golden_pad"),
                             padded_size, data_size, dma_core, host_core, [load_input])

    # ── 7. GATHER ────────────────────────────────────────────────
    g_start, g_stride = 0, 2
    g_count = rows // g_stride
    g_out_size = g_count * cols * elem
    l1_gather_dst = BingoMemAlloc("l1_gather_dst", size=g_out_size, mem_level="L1", chip_id=0, cluster_id=0)
    xdma_gather = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
        node_name="XDMA_gather",
        kernel_name="__snax_bingo_kernel_xdma_gather_2d",
        kernel_args=SnaxBingoKernelXdmaGather2dArgs(
            l1_input, l1_gather_dst, rows, cols, g_count, g_start, g_stride, elem))
    dfg.bingo_add_node(xdma_gather)
    dfg.bingo_add_edge(prev_chk, xdma_gather)
    prev_chk = add_xdma_test(dfg, "gather", l1_gather_dst, xdma_gather, BingoMemSymbol("golden_gather"),
                             g_out_size, data_size, dma_core, host_core, [load_input])

    # ── Compile DFG ──────────────────────────────────────────────
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    dfg.bingo_compile_dfg("xDMA All Operators Test", output_dir, args.output_offload_file_name,
                          extra_include_header_list=["xdma_data.h"])


if __name__ == "__main__":
    main()
