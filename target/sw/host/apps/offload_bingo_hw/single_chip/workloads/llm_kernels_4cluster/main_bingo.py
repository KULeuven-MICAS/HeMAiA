#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# ======================================================================================
# UNIT TESTS FOR EVERY KERNEL THE LLM LAYER DISPATCHES
# ======================================================================================
#
# One test per kernel, at the layer's own toy shape, each Load -> Kernel -> Store -> Check
# against a golden that describes THAT kernel and nothing else. When the layer fails, this
# app says whether a kernel is wrong or the composition is.
#
# THE TESTS ARE CHAINED, not parallel. There is one of each engine, so they serialise
# anyway; saying so keeps the ready set small and -- more usefully -- makes the UART
# output a sequence, so the first failure names the first broken kernel instead of
# arriving interleaved with three others.
#
# WHERE THE TOLERANCES COME FROM, and why two of them are zero:
#
#   quantise, reshape   BYTE-EXACT. Both are permutations or roundings with exactly one
#                       right answer. A tolerance here would accept a conversion that
#                       dropped the low bits of every element, which is the failure these
#                       tests exist to catch.
#   add                 byte-exact too: the operands are integers whose sum is exact in
#                       fp16, chosen so the compare tests the kernel and not fp16.
#   rmsnorm, rope       fp16 tolerance. The device has no FPU on the engine cores, so
#                       rmsnorm does an integer sqrt and reciprocal and rope reads a
#                       precomputed table; neither reproduces the fp32 reference bit for
#                       bit and neither is meant to.
#   gemm                fp16 tolerance on a D-layout tile. The accumulation is exact in
#                       int32; the narrowing to fp16 on the D port is what costs.
#
#
# WHAT THIS RUNG DOES NOT COVER, deliberately. FlashAttention's own kernels -- gemm_fa_qk,
# gemm_fa_pv, simd_fa_softmax, pack_fa_partial, xdma_chain_gather, xdma_memset -- are
# exercised by fa_decode_4cluster, which is their stage test and validated on RTL. Testing
# them again here would mean rebuilding FA's buffer geometry outside FA, where a passing
# test would prove the copy right rather than the block. Everything the LAYER'S OWN blocks
# dispatch (RMSNorm, Quantize, Reshape, Linear, Residual) is covered below.
#
# ======================================================================================

import argparse
import os
import sys

import hjson
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(f"{ROOT_DIR}/util/sim/common")   # bingo_data_staging, goldens
sys.path.append(f"{ROOT_DIR}/util/sim/llm")      # the layer builder and its datagen
sys.path.append(current_dir)

from kernels_datagen import generate, stage                            # noqa: E402
import _bingo_paths  # noqa: F401,E402
from bingo_dfg import BingoDFG                                         # noqa: E402
from bingo_data_staging import DataStaging                             # noqa: E402
from bingo_node import BingoNode                                       # noqa: E402
from bingo_mem_handle import BingoMemAlloc, BingoMemAllocView                             # noqa: E402
from bingo_platform import (core_roles, guard_cluster_count,           # noqa: E402
                            parse_platform_cfg)
from bingo_kernel_args import (                                        # noqa: E402
    HostBingoKernelCheckResultBytesArgs,
    HostBingoKernelCheckResultF16Args,
    HostBingoKernelIdmaArgs,
    SnaxBingoKernelGemmFullArgs,
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelSimdAddF16Args,
    SnaxBingoKernelSimdFp16ToInt8Args,
    SnaxBingoKernelSimdRmsnormArgs,
    SnaxBingoKernelSimdRopeArgs,
    SnaxBingoKernelIdmaPairwiseSwapArgs,
)
from libs.block.flash_attention import mesh_from_hwcfg                 # noqa: E402
from libs.comm.nest import convert_args                                # noqa: E402

CHIPLET_ID = 0x00
CLUSTER = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_h", default=None)
    ap.add_argument("--output_offload_file_name", default="offload_bingo_hw.h")
    ap.add_argument("-c", "--cfg", required=True)
    ap.add_argument("--hwcfg", required=True)
    ap.add_argument("--platformcfg", required=True)
    args = ap.parse_args()

    with open(args.cfg) as f:
        p = dict(hjson.load(f))
    mesh = mesh_from_hwcfg(args.hwcfg, int(p.get("array_shape", 0)))
    mr, ts, mc = mesh
    p["meshRow"], p["tileSize"], p["meshCol"] = mesh
    plat = parse_platform_cfg(args.platformcfg)
    guard_cluster_count(p, plat, args.output_dir, args.output_offload_file_name)

    T, d = p["tokens"], p["d_model"]
    g = generate(p)
    st = DataStaging(plat)
    hs = stage(st, g, p)

    dfg = BingoDFG(num_chiplets=1,
                   num_clusters_per_chiplet=plat["num_clusters_per_chiplet"],
                   num_cores_per_cluster=plat["num_cores_per_cluster"],
                   is_host_as_acc=True, chiplet_ids=[CHIPLET_ID],
                   dep_tag_width=plat["dep_tag_width"])
    dfg.l1_capacity_bytes = 514816
    R = core_roles()

    def l1(name, nbytes):
        return BingoMemAlloc(f"k_{name}_cl{CLUSTER}", size=nbytes, mem_level="L1",
                             chip_id=CHIPLET_ID, cluster_id=CLUSTER)

    def node(name, core, kname, kargs, after=()):
        n = BingoNode(assigned_chiplet_id=CHIPLET_ID, assigned_cluster_id=CLUSTER,
                      assigned_core_id=core, node_name=name,
                      kernel_name=kname, kernel_args=kargs)
        dfg.bingo_add_node(n)
        for q in (after if isinstance(after, (list, tuple)) else [after]):
            if q is not None:
                dfg.bingo_add_edge(q, n)
        return n

    def host(name, kname, kargs, after=()):
        n = BingoNode(assigned_chiplet_id=CHIPLET_ID, assigned_cluster_id=0,
                      assigned_core_id=R["host"], node_name=name,
                      kernel_name=kname, kernel_args=kargs)
        dfg.bingo_add_node(n)
        for q in (after if isinstance(after, (list, tuple)) else [after]):
            if q is not None:
                dfg.bingo_add_edge(q, n)
        return n

    def load(tag, src, dst, nbytes, after):
        return node(f"Ld_{tag}", R["dm"], "__snax_bingo_kernel_idma_1d_copy",
                    SnaxBingoKernelIdma1dCopyArgs(src, dst, nbytes), after)

    def store_check(tag, buf, nbytes, golden, after, *, exact, elems=None, tol=0.0):
        """Read the result back to L3 and compare. Returns the check node."""
        l3 = BingoMemAlloc(f"k_out_{tag}", size=nbytes, mem_level="L3",
                           chip_id=CHIPLET_ID)
        stn = host(f"St_{tag}", "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(buf, l3, nbytes), after)
        a = (HostBingoKernelCheckResultBytesArgs(golden, l3, nbytes, name=tag) if exact
             else HostBingoKernelCheckResultF16Args(golden, l3, elems, tol=tol, name=tag))
        return host(f"Ck_{tag}", "__host_bingo_kernel_check_result", a, stn)

    n_f16 = T * d * 2
    n_i8 = T * d
    prev = None

    # ---- 1. rmsnorm -------------------------------------------------------------------
    b_in, b_out = l1("rms_in", n_f16), l1("rms_out", n_f16)
    ld = load("rms", hs["rms_x"], b_in, n_f16, prev)
    k = node("Rmsnorm", R["simd"], "__snax_bingo_kernel_simd_rmsnorm",
             SnaxBingoKernelSimdRmsnormArgs(b_in, b_out, rows=T, cols=d), ld)
    prev = store_check("rmsnorm", b_out, n_f16, hs["rms_golden"], k,
                       exact=False, elems=T * d, tol=0.05)

    # ---- 2. rope ----------------------------------------------------------------------
    # THE FOUR OPERANDS ARE ONE BLOCK: [ x | cos_full | xswap | sin_signed ], because the
    # reader adds a single stride per axis and the fused kernel derives that stride from
    # the shape. The order IS the pairing -- EW0 multiplies slot 0 by 1 and 2 by 3, EW1
    # adds the two -- so permuting it computes x*xswap + cos*sin, not a rotation.
    r_ops = l1("rope_ops", 4 * n_f16)
    r_o = l1("rope_out", n_f16)
    slot = lambda k: r_ops if k == 0 else BingoMemAllocView(r_ops, k * n_f16)
    l1n = load("rope_x", hs["rope_x"], slot(0), n_f16, prev)
    l2n = load("rope_cos", hs["rope_cos"], slot(1), n_f16, l1n)
    l3n = load("rope_sin", hs["rope_sin"], slot(3), n_f16, l2n)
    # The adjacent fp16-pair swap, on the DM core. It is a 2-byte reorder inside an 8-byte
    # TCDM word, which is below the granularity the SIMD reader's AGU can address, so it
    # has to be a real byte-addressed DMA rather than a stride.
    sw = node("Rope_swap", R["dm"], "__snax_bingo_kernel_idma_pairwise_swap",
              SnaxBingoKernelIdmaPairwiseSwapArgs(slot(0), slot(2), T * d, 2), l3n)
    k = node("Rope", R["simd"], "__snax_bingo_kernel_simd_rope",
             SnaxBingoKernelSimdRopeArgs(r_ops, r_o, cols=d, rows=T), sw)
    prev = store_check("rope", r_o, n_f16, hs["rope_golden"], k,
                       exact=False, elems=T * d, tol=0.05)

    # ---- 3. quantise ------------------------------------------------------------------
    q_in, q_out = l1("quant_in", n_f16), l1("quant_out", n_i8)
    ld = load("quant", hs["quant_x"], q_in, n_f16, prev)
    k = node("Quant", R["simd"], "__snax_bingo_kernel_simd_fp16_to_int8",
             SnaxBingoKernelSimdFp16ToInt8Args(
                 q_in, q_out, beats=n_f16 // 64, rows=1,
                 inv_scale_f32bits=g["quant_scale_bits"]), ld)
    prev = store_check("quant", q_out, n_i8, hs["quant_golden"], k, exact=True)

    # ---- 4. elementwise add (the residual) ---------------------------------------------
    # a and b may be placed in EITHER order. The reader AGU strides forward only, so the
    # kernel bases the 2-operand interleave at the lower address and swaps if it has to
    # (snax_simd_ew2_base) -- sound because ADD commutes. Nothing here has to pre-order
    # them, and no PLACEMENT_ORDER constraint is declared.
    a_buf, b_buf, s_buf = l1("add_a", n_f16), l1("add_b", n_f16), l1("add_out", n_f16)
    l1n = load("add_a", hs["add_a"], a_buf, n_f16, prev)
    l2n = load("add_b", hs["add_b"], b_buf, n_f16, l1n)
    k = node("Add", R["simd"], "__snax_bingo_kernel_simd_stream_elementwise",
             SnaxBingoKernelSimdAddF16Args(a_buf, b_buf, s_buf, rows=T, cols=d), l2n)
    prev = store_check("add", s_buf, n_f16, hs["add_golden"], k, exact=True)

    # ---- 5. gemm ----------------------------------------------------------------------
    ga, gb, gd = l1("gemm_a", n_i8), l1("gemm_b", d * d), l1("gemm_d", n_f16)
    l1n = load("gemm_a", hs["gemm_a"], ga, n_i8, prev)
    l2n = load("gemm_b", hs["gemm_b"], gb, d * d, l1n)
    # input_C_addr=0: a non-zero C is READ and added, and gd is uninitialised here.
    k = node("Gemm", R["gemm"], "__snax_bingo_kernel_gemm_full",
             SnaxBingoKernelGemmFullArgs(
                 input_A_addr=ga, input_B_addr=gb, input_C_addr=0, output_D_addr=gd,
                 M=T // mr, K=d // ts, N=d // mc,
                 array_shape_idx=int(p.get("array_shape", 0)),
                 transpose_A=0, transpose_B=0, accumPrevC=0, int32tofp16_enable=1), l2n)
    prev = store_check("gemm", gd, n_f16, hs["gemm_golden"], k,
                       exact=False, elems=T * d, tol=0.5)

    # ---- 6. the two reshapes -----------------------------------------------------------
    for tag, src_lay, dst_lay, src_h, gold_h in (
            ("rs_d2p", "D", "row_major", "rs_d_src", "rs_d2p_golden"),
            ("rs_p2a", "row_major", "A", "rs_p_src", "rs_p2a_golden")):
        s_b, d_b = l1(f"{tag}_in", n_f16), l1(f"{tag}_out", n_f16)
        ld = load(tag, hs[src_h], s_b, n_f16, prev)
        k = node(f"Reshape_{tag}", R["xdma"], "__snax_bingo_kernel_xdma_6d",
                 convert_args(src_lay, dst_lay, T, d, mesh, 2, s_b, d_b), ld)
        prev = store_check(tag, d_b, n_f16, hs[gold_h], k, exact=True)

    if args.data_h:
        st.emit(args.data_h, args.output_dir)
    extra = [os.path.basename(str(args.data_h))] if args.data_h else None
    dfg.bingo_compile_dfg(
        app_name=f"LLM kernel unit tests (T={T} d={d})",
        output_dir=args.output_dir,
        output_file_name=args.output_offload_file_name,
        extra_include_header_list=extra)
    print(f"Generated: {os.path.join(args.output_dir, args.output_offload_file_name)}")


if __name__ == "__main__":
    main()
