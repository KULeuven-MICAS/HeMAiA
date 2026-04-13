# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end test for the 6 xDMA layout transform kernels.
#
# Configuration sources (mirrors attn_reference/Makefile):
#   - params.hjson  : workload tile counts (M_T, K_T, N_T) + array_shape + elem_bytes
#   - --hwcfg       : snax_versacore_to_cluster.hjson — provides the spatial
#                     unrolling (meshRow, tileSize, meshCol) per array_shape
#   - --platformcfg : generated occamy.h — gives N_CHIPLETS / N_CLUSTERS_PER_CHIPLET
#                     / N_CORES_PER_CLUSTER
#   - --rtlcfg      : RTL config hjson (lru.hjson) — multichip → chiplet IDs
#
# Two paths per conversion:
#   1. L1 path: IDMA load src from L3 → local L1, run the kernel with
#      L1 src + L1 dst, IDMA store result back to L3, host check vs golden.
#   2. L3-direct path: pass the L3 BingoMemSymbol directly as the kernel's
#      src and an L3 BingoMemAlloc as dst — exercises the in-kernel IDMA
#      stage-in fallback (xdma_layout_stage_in in snax_xdma_lib.h).
#
# When meshRow == 8 and (tileSize | meshCol) * elem % 8 == 0 the A↔R / D↔R
# kernels use the AGU-only xDMA path; B↔R uses the HW Transposer extension
# when tileSize == meshCol == 8. Otherwise the kernels transparently fall
# back to their CPU loops.

import os
import re
import sys
import argparse
import pathlib
import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(current_dir)

from layout_convert_datagen import emit_header_file  # noqa E402
from bingo_dfg import BingoDFG  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_kernel_args import (  # noqa E402
    SnaxBingoKernelIdma1dCopyArgs,
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


def get_args():
    p = argparse.ArgumentParser(description="xdma_layout_transform")
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    p.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    p.add_argument("--hwcfg", type=pathlib.Path, required=True)
    p.add_argument("--platformcfg", type=pathlib.Path, required=True)
    p.add_argument("--rtlcfg", type=pathlib.Path, required=True)
    p.add_argument("--data_h", type=pathlib.Path, default=None)
    return p.parse_args()


def parse_platform_cfg(occamy_h_path, rtlcfg_path):
    """Same parser as attn_reference: pull topology from occamy.h #defines and
    chiplet IDs from the RTL multichip section."""
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


def add_l1_test(dfg, name, src_sym, src_size, kernel_name, kernel_args_cls,
                kernel_arg_kwargs_no_addrs, dst_size, golden_sym, prev_chk):
    """L1 path: IDMA load src to L1 -> kernel (L1 src+dst) -> IDMA store to L3 -> check."""
    l1_src = BingoMemAlloc(f"l1_{name}_src", size=src_size, mem_level="L1",
                           chip_id=0, cluster_id=0)
    l1_dst = BingoMemAlloc(f"l1_{name}_dst", size=dst_size, mem_level="L1",
                           chip_id=0, cluster_id=0)
    l3_dst = BingoMemAlloc(f"l3_{name}_dst", size=dst_size, mem_level="L3")

    load = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name=f"Load_{name}", kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(src_sym, l1_src, src_size))
    convert = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name=f"Conv_{name}", kernel_name=kernel_name,
        kernel_args=kernel_args_cls(src_addr=l1_src, dst_addr=l1_dst,
                                    **kernel_arg_kwargs_no_addrs))
    store = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name=f"Store_{name}", kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(l1_dst, l3_dst, dst_size))
    chk = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name=f"Check_{name}", kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(golden_sym, l3_dst, dst_size,
                                                   name=f"L1_{name}"))

    for n in (load, convert, store, chk):
        dfg.bingo_add_node(n)
    if prev_chk is not None:
        dfg.bingo_add_edge(prev_chk, load)
    dfg.bingo_add_edge(load, convert)
    dfg.bingo_add_edge(convert, store)
    dfg.bingo_add_edge(store, chk)
    return chk


def add_l3_direct_test(dfg, name, src_sym, kernel_name, kernel_args_cls,
                       kernel_arg_kwargs_no_addrs, dst_size, golden_sym, prev_chk):
    """L3-direct path: pass L3 src/dst straight to kernel — exercises the
    in-kernel xdma_layout_stage_in IDMA fallback. No load/store nodes."""
    l3_dst = BingoMemAlloc(f"l3_direct_{name}_dst", size=dst_size, mem_level="L3")

    convert = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name=f"ConvL3_{name}", kernel_name=kernel_name,
        kernel_args=kernel_args_cls(src_addr=src_sym, dst_addr=l3_dst,
                                    **kernel_arg_kwargs_no_addrs))
    chk = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name=f"CheckL3_{name}", kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(golden_sym, l3_dst, dst_size,
                                                   name=f"L3_{name}"))

    for n in (convert, chk):
        dfg.bingo_add_node(n)
    if prev_chk is not None:
        dfg.bingo_add_edge(prev_chk, convert)
    dfg.bingo_add_edge(convert, chk)
    return chk


def _conversions(M_T, K_T, N_T, meshRow, tileSize, meshCol, elem_bytes):
    A_kw = dict(M_T=M_T, K_T=K_T, meshRow=meshRow, tileSize=tileSize, elem_bytes=elem_bytes)
    B_kw = dict(K_T=K_T, N_T=N_T, tileSize=tileSize, meshCol=meshCol, elem_bytes=elem_bytes)
    D_kw = dict(M_T=M_T, N_T=N_T, meshRow=meshRow, meshCol=meshCol, elem_bytes=elem_bytes)
    A_bytes = M_T * meshRow * K_T * tileSize * elem_bytes
    B_bytes = K_T * tileSize * N_T * meshCol * elem_bytes
    D_bytes = M_T * meshRow * N_T * meshCol * elem_bytes
    return [
        ("row_to_A", "src_A_rm",      A_bytes, "__snax_bingo_kernel_xdma_row_major_to_a",
         SnaxBingoKernelXdmaRowMajorToAArgs, A_kw, A_bytes, "golden_A_layout"),
        ("A_to_row", "src_A_layout",  A_bytes, "__snax_bingo_kernel_xdma_a_to_row_major",
         SnaxBingoKernelXdmaAToRowMajorArgs, A_kw, A_bytes, "golden_A_rm"),
        ("row_to_B", "src_B_rm",      B_bytes, "__snax_bingo_kernel_xdma_row_major_to_b",
         SnaxBingoKernelXdmaRowMajorToBArgs, B_kw, B_bytes, "golden_B_layout"),
        ("B_to_row", "src_B_layout",  B_bytes, "__snax_bingo_kernel_xdma_b_to_row_major",
         SnaxBingoKernelXdmaBToRowMajorArgs, B_kw, B_bytes, "golden_B_rm"),
        ("row_to_D", "src_D_rm",      D_bytes, "__snax_bingo_kernel_xdma_row_major_to_d",
         SnaxBingoKernelXdmaRowMajorToDArgs, D_kw, D_bytes, "golden_D_layout"),
        ("D_to_row", "src_D_layout",  D_bytes, "__snax_bingo_kernel_xdma_d_to_row_major",
         SnaxBingoKernelXdmaDToRowMajorArgs, D_kw, D_bytes, "golden_D_rm"),
    ]


def main():
    args = get_args()

    with open(args.cfg) as f:
        param = hjson.loads(f.read())
    with open(args.hwcfg) as f:
        hwcfg = hjson.loads(f.read())
    merged = {**param, **hwcfg}

    # Spatial unrolling (meshRow, tileSize, meshCol) for the chosen array_shape.
    array_shape = merged["array_shape"]
    unrolling = merged["snax_versacore_core_template"]["snax_acc_cfg"][0]\
                       ["snax_versacore_spatial_unrolling"][0][array_shape]
    meshRow, tileSize, meshCol = int(unrolling[0]), int(unrolling[1]), int(unrolling[2])

    M_T = int(merged["M_T"]); K_T = int(merged["K_T"]); N_T = int(merged["N_T"])
    elem_bytes = int(merged["elem_bytes"])

    # Stash derived shape into the merged dict so the datagen can use it.
    merged.update(dict(meshRow=meshRow, tileSize=tileSize, meshCol=meshCol))

    if args.data_h is not None:
        with open(args.data_h, "w") as f:
            f.write(emit_header_file(**merged))
        print(f"Written data header: {args.data_h}")

    platform = parse_platform_cfg(args.platformcfg, args.rtlcfg)
    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    convs = _conversions(M_T, K_T, N_T, meshRow, tileSize, meshCol, elem_bytes)
    prev = None

    # ── L1 path: all 6 conversions (forward + inverse for A, B, D) ───────
    for name, src, src_sz, kname, kcls, kw, dst_sz, golden in convs:
        prev = add_l1_test(dfg, name, BingoMemSymbol(src), src_sz,
                           kname, kcls, kw, dst_sz, BingoMemSymbol(golden), prev)

    # ── L3-direct path: one forward conversion per layout family ─────────
    # Smoke-tests the in-kernel xdma_layout_stage_in fallback for A, B, D.
    for name, src, src_sz, kname, kcls, kw, dst_sz, golden in convs:
        if name in ("row_to_A", "row_to_B", "row_to_D"):
            prev = add_l3_direct_test(dfg, name, BingoMemSymbol(src),
                                      kname, kcls, kw, dst_sz,
                                      BingoMemSymbol(golden), prev)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    dfg.bingo_compile_dfg("xDMA Layout Transform", output_dir,
                          args.output_offload_file_name,
                          extra_include_header_list=["layout_convert_data.h"])


if __name__ == "__main__":
    main()
