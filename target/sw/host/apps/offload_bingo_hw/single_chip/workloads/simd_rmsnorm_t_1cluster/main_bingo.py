# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# TRANSPOSED SIMD FP16 RMSNorm — the A/B that decides whether the cheap path is real.
#
# Both kernels compute out[t,:] = x[t,:] / sqrt(mean_j x[t,j]^2) over the SAME [32, 128]
# tile against the SAME golden. They differ only in which way the tile is stored while the
# SIMD block reduces it, and that is worth ~3x:
#
#   ROW-MAJOR  reduce(SUMSQ) -> bcast carrying StreamMap(1/D, RSQRT) -> ew2(MUL)
#              A beat is 32 consecutive FEATURES of one token, so lane k collects every
#              32nd feature and a row's terms scatter over all 32 lanes. Collapsing them
#              is a serialised log-depth fold that holds the reader's input port low for
#              ~35 cc PER ROW, and the single scalar then has to be replicated into a
#              whole [T, D] plane before a 2-operand elementwise can read it.
#
#   TRANSPOSED reduce(SUMSQ|LANEWISE) -> map(1/D, RSQRT) on ONE beat -> ew(MUL|STICKY_B)
#              A beat is one FEATURE across all 32 tokens, and lane t is token t in EVERY
#              beat, so acc[t] collects token t's whole row and the answer is already in
#              the accumulators when the stream ends. No fold, no splat, and the scale
#              rides back in as a sticky operand rather than a replicated plane.
#
# Reference measurement on snax_split_cluster (target/snitch_cluster/sw/apps/
# snax-simd-rmsnorm): 3,135 cc row-major vs 1,073 cc transposed on the SIMD core, plus
# 244 + 385 cc of xDMA for the two block transposes. THIS workload is what says whether
# HeMAiA reproduces that.
#
# ======================================================================================
# WHAT EACH CHECK ISOLATES — the reason there are four and not one
# ======================================================================================
#
# The transposed path is three new things at once (a new kernel entry point, a new
# seed-adjacency contract, and two xDMA transposer nodes), so a single end-to-end compare
# would say "wrong" without saying which. Each stage is therefore stored and checked:
#
#   xpose_in     x -> x^T      BIT-EXACT. A transpose moves bytes; any tolerance here
#                              would hide the thing most likely to be wrong, which is
#                              whether the transposer was armed on the side this cfg
#                              actually built it on. If the extension is missing on BOTH
#                              sides the kernel falls back to a DM-core element loop --
#                              still correct, so this check PASSES and only the cycle
#                              count betrays it. Read the trace, not just the verdict.
#   norm_t       y^T           against the transposed golden. Isolates the kernel itself:
#                              the LANEWISE reduce, the one-beat RSQRT map, and the
#                              sticky-B multiply.
#   norm_t_e2e   y             the whole transposed chain, in the layer's own layout. This
#                              is also the OUTPUT TRANSPOSE's check and there is no
#                              separate one: y^T has already been scored by norm_t, so the
#                              only thing left that can move y is y^T -> y. It cannot be
#                              bit-exact against a golden the way xpose_in is, because the
#                              value it carries came through the kernel.
#   norm_rm      y             the row-major kernel over the same x. The reference arm:
#                              if this fails too, the fault is in RSQRT or the data, not
#                              in the transposition.
#
# ======================================================================================
# THE ONE-BEAT HEADROOM
# ======================================================================================
#
# The sticky elementwise reads ONE FLAT SWEEP of 1 + D beats whose first beat is the
# scale, so the seed must physically precede the tile -- there is no argument that
# separates them. `l1_xt` is therefore (1 + cols) beats: the kernel is handed its base as
# `seed_addr` and base + 64 as `input_addr`, the input transpose writes to base + 64, and
# the kernel CHECKS the adjacency rather than trusting it. Get it wrong and the seed
# lands on feature row 0 and the tile is read one beat out of phase, with nothing
# reported -- which is exactly why the kernel refuses instead.
#
# SHAPE IS NOT A SWEEP HERE. rows must be 32, the FP16 lanes in one beat: at fewer a beat
# holds several features and the per-lane accumulators mix tokens, at more a feature spans
# several beats and they mix the other way. Neither faults. cols is a power of two because
# the 1/D the RSQRT map carries is an exponent-only immediate.

import os
import sys
import json
import argparse
import pathlib
import hjson
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(_THIS, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(f"{ROOT_DIR}/util/sim")
for _p in [p for p in list(sys.path) if str(p).rstrip('/').endswith('util/sim')]:
    for _s in ('common', 'gemm', 'xdma', 'ara'):
        _sub = os.path.join(_p, _s)
        if _sub not in sys.path:
            sys.path.append(_sub)

import _bingo_paths  # noqa: F401,E402  (puts mini_compiler's grouped subdirs on sys.path)
from bingo_dfg import BingoDFG                                   # noqa: E402
from bingo_platform import core_roles, guard_cluster_count, parse_platform_cfg  # noqa: E402
from bingo_node import BingoNode                                 # noqa: E402
from bingo_mem_handle import BingoMemAlloc, BingoMemAllocView    # noqa: E402
from bingo_data_staging import DataStaging                       # noqa: E402
from bingo_kernel_args import (                                  # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelSimdRmsnormF16F16Args,
    SnaxBingoKernelSimdRmsnormTF16F16Args,
    SnaxBingoKernelXdmaTranspose2dArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)

_ROLES = core_roles()
SIMD_CORE = _ROLES["simd"]
XDMA_CORE = _ROLES["xdma"]
DMA_CORE = _ROLES["dm"]
HOST_CORE = _ROLES["host"]

CHECK_FP16_TOL = 2
BEAT = 64

# ROWS is not a knob: one FP16 lane per token is what makes the LANEWISE reduce a per-token
# reduce, and 32 lanes is SIMD_WIDTH / elemWidth on this cluster.
ROWS = 32
COLS = 128

# BINGO_CHECK_TYPE_FP16_TOL is an ABSOLUTE tolerance, not a relative one. RMSNorm's output
# has unit RMS and this input reaches about |1.7|, so 0.03 is roughly 2% of full scale --
# a wide margin against the hardware rsqrt's ~1 FP16 ULP, and still far tighter than any
# layout error, which scrambles values across the whole range. Deliberately the SAME
# tolerance both arms are scored at, because the point is that they agree.
TOL = 0.03


def _rmsnorm_ref(rows, cols, seed=0x12345):
    """x, and the four goldens the checks score against.

    The reduce accumulates in FP32 and narrows the scalar to FP16 before the rsqrt sees
    it, so the reference narrows too -- that step is in the datapath and is not an
    approximation the model gets to skip. What follows it is a hardware table, so the
    reference is the TRUE 1/sqrt rather than a model of the core's old integer
    sqrt+reciprocal.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-4.0, 4.0, size=(rows, cols)).astype(np.float32).astype(np.float16)
    xf = x.astype(np.float32)
    ssq = np.array([np.float16((xf[r] ** 2).sum(dtype=np.float32)) for r in range(rows)],
                   dtype=np.float16)
    mean = ssq.astype(np.float32) / np.float32(cols)          # cols is 2^k, so exact
    inv = np.float16(np.float32(1.0) / np.sqrt(mean)).astype(np.float32)
    y = (xf * inv[:, None]).astype(np.float16)
    # `.copy()` is load-bearing: a transposed numpy view is not contiguous, and .view()
    # on it would reinterpret the ORIGINAL buffer's bytes rather than the transposed ones.
    return x, x.T.copy(), y, y.T.copy()


class G:
    def __init__(self, dfg):
        self.dfg = dfg

    def l1(self, name, size):
        return BingoMemAlloc(name, size=size, mem_level="L1", chip_id=0, cluster_id=0)

    def node(self, name, core, kname, kargs, after):
        nd = BingoNode(assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=core,
                       node_name=name, kernel_name=kname, kernel_args=kargs)
        self.dfg.bingo_add_node(nd)
        if after is not None:
            self.dfg.bingo_add_edge(after, nd)
        return nd


def build_mempool(st):
    x, xt, y, yt = _rmsnorm_ref(ROWS, COLS)
    return {
        "x":  st.put("rmsnorm_t_in", "uint16_t", x.reshape(-1).view(np.uint16)),
        "xt": st.put("rmsnorm_t_xt_golden", "uint16_t", xt.reshape(-1).view(np.uint16)),
        "y":  st.put("rmsnorm_t_y_golden", "uint16_t", y.reshape(-1).view(np.uint16)),
        "yt": st.put("rmsnorm_t_yt_golden", "uint16_t", yt.reshape(-1).view(np.uint16)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    p.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    p.add_argument("--hwcfg", type=pathlib.Path, required=True)
    p.add_argument("--platformcfg", type=pathlib.Path, required=True)
    p.add_argument("--data_h", type=pathlib.Path, default=None)
    p.add_argument("--configs_out", type=pathlib.Path, default=None)
    args = p.parse_args()
    with open(args.cfg) as f:
        param = hjson.loads(f.read())

    platform = parse_platform_cfg(args.platformcfg)
    st = DataStaging(platform)
    off = build_mempool(st)
    if args.data_h is not None:
        n = st.emit(args.data_h, args.output_dir)
        print(f"Staged {n} B of inputs and goldens "
              f"{'on the memory chiplet' if st.on_memchip else 'in the host image'}")
    if args.configs_out is not None:
        with open(args.configs_out, "w") as f:
            json.dump({"op": "rmsnorm_t",
                       "configs": [{"rows": ROWS, "cols": COLS}]}, f, indent=2)
    if not guard_cluster_count(param, platform, args.output_dir,
                               args.output_offload_file_name):
        return

    dfg = BingoDFG(num_chiplets=1,
                   num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
                   num_cores_per_cluster=platform["num_cores_per_cluster"],
                   is_host_as_acc=True, chiplet_ids=[0x00])
    g = G(dfg)

    n = ROWS * COLS
    tot_b = n * 2

    l1_x = g.l1("rmst_x", tot_b)            # row-major input, both arms read it
    l1_xt = g.l1("rmst_xt", BEAT + tot_b)   # [ seed beat | x^T ] -- ONE allocation
    l1_yt = g.l1("rmst_yt", tot_b)          # y^T, the transposed kernel's output
    l1_y = g.l1("rmst_y", tot_b)            # y^T transposed back to row-major
    l1_yrm = g.l1("rmst_yrm", tot_b)        # the row-major arm's output
    xt_tile = BingoMemAllocView(l1_xt, BEAT)

    l3_xt = BingoMemAlloc("out_rmst_xt", size=tot_b, mem_level="L3")
    l3_yt = BingoMemAlloc("out_rmst_yt", size=tot_b, mem_level="L3")
    l3_y = BingoMemAlloc("out_rmst_y", size=tot_b, mem_level="L3")
    l3_yrm = BingoMemAlloc("out_rmst_yrm", size=tot_b, mem_level="L3")

    def check(name, golden, got, after, tol):
        return g.node(f"Check_{name}", HOST_CORE, "__host_bingo_kernel_check_result",
                      HostBingoKernelCheckResultArgs(
                          golden, got, name=name, check_type=CHECK_FP16_TOL,
                          num_elements=n, tolerance=tol), after)

    # ---- load -------------------------------------------------------------------------
    load = g.node("Load_x", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                  SnaxBingoKernelIdma1dCopyArgs(off["x"], l1_x, tot_b), None)

    # ---- the transposed arm -----------------------------------------------------------
    # x [32,128] -> x^T [128,32], landing ONE BEAT IN so the seed slot sits below it.
    xin = g.node("Xpose_in", XDMA_CORE, "__snax_bingo_kernel_xdma_transpose_2d",
                 SnaxBingoKernelXdmaTranspose2dArgs(l1_x, xt_tile, ROWS, COLS, 2), load)
    st_xt = g.node("Store_xt", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(xt_tile, l3_xt, tot_b), xin)
    ck_xt = check("rmsnorm_t_xpose_in", off["xt"], l3_xt, st_xt, 0.0)

    # The kernel writes the seed into l1_xt[0] and reads 1 + cols beats from there.
    rn_t = g.node("RMSNormT", SIMD_CORE, "__snax_bingo_kernel_simd_rmsnorm_t_f16_f16",
                  SnaxBingoKernelSimdRmsnormTF16F16Args(l1_xt, xt_tile, l1_yt, ROWS, COLS),
                  ck_xt)
    st_yt = g.node("Store_yt", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(l1_yt, l3_yt, tot_b), rn_t)
    ck_yt = check("rmsnorm_t_norm", off["yt"], l3_yt, st_yt, TOL)

    # y^T [128,32] -> y [32,128]: the other way round, because y^T's shape is swapped.
    xout = g.node("Xpose_out", XDMA_CORE, "__snax_bingo_kernel_xdma_transpose_2d",
                  SnaxBingoKernelXdmaTranspose2dArgs(l1_yt, l1_y, COLS, ROWS, 2), ck_yt)
    st_y = g.node("Store_y", HOST_CORE, "__host_bingo_kernel_idma",
                  HostBingoKernelIdmaArgs(l1_y, l3_y, tot_b), xout)
    ck_y = check("rmsnorm_t_e2e", off["y"], l3_y, st_y, TOL)

    # ---- the row-major reference arm, over the SAME x ---------------------------------
    rn_rm = g.node("RMSNormRM", SIMD_CORE, "__snax_bingo_kernel_simd_rmsnorm_f16_f16",
                   SnaxBingoKernelSimdRmsnormF16F16Args(l1_x, l1_yrm, ROWS, COLS), ck_y)
    st_rm = g.node("Store_yrm", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(l1_yrm, l3_yrm, tot_b), rn_rm)
    check("rmsnorm_rowmajor", off["y"], l3_yrm, st_rm, TOL)

    os.makedirs(args.output_dir, exist_ok=True)
    dfg.bingo_compile_dfg("SIMD rmsnorm, transposed vs row-major", args.output_dir,
                          args.output_offload_file_name,
                          extra_include_header_list=["rmsnorm_t_data.h"])
    print(f"Generated rmsnorm_t: [{ROWS}, {COLS}], both arms, 4 checks")


if __name__ == "__main__":
    main()
