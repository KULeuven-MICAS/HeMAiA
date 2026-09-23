# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# TRANSPOSED SIMD FP16 softmax — the A/B that decides whether the cheap path is real.
#
# Both kernels compute out[t,:] = exp(x[t,:] - max) / SUM exp(x[t,:] - max) over the SAME
# [32, 128] tile against the SAME golden. They differ only in which way the tile is stored
# while the SIMD block reduces it.
#
# SOFTMAX HAS TWO PER-ROW SCALARS WHERE RMSNORM HAS ONE, and both sit in the middle of the
# dependence chain: nothing can be exponentiated until the max is known, nothing divided
# until every exponential has been summed. So everything the transposed rmsnorm wins, this
# wins twice.
#
#   ROW-MAJOR  reduce(MAX) -> bcast(a=-1) -> EW0(ADD)|Map(EXP)|Reduce(ADD,TAP)
#              -> EW0(MUL) squaring -> bcast(RSQRT) -> ew2(MUL)
#              A beat is 32 consecutive SCORES of one token, so a row's terms scatter over
#              all 32 lanes and collapsing them is a serialised log-depth fold that holds
#              the reader's input port low for ~35 cc PER ROW -- paid TWICE, once for the
#              max and once for the sum. Each scalar then has to be replicated into a whole
#              [T, D] plane before a 2-operand elementwise can read it.
#
#   TRANSPOSED reduce(MAX|LANEWISE) -> map(a=-1) -> EW0(ADD|STICKY)|Map(EXP)|
#              Reduce(ADD|TAP|LANEWISE) -> EW0(MUL|STICKY)|Map(RSQRT) -> ew(MUL|STICKY)
#              A beat is one SCORE position across all 32 tokens, and lane t is token t in
#              EVERY beat, so acc[t] collects token t's whole row. Both reduces come out of
#              the accumulators with no fold, and both scalars ride back in as sticky seed
#              beats rather than replicated planes.
#
# Reference measurement on snax_split_cluster (target/snitch_cluster/sw/apps/
# snax-simd-softmax): 5,665 cc row-major vs 1,748 cc transposed on the SIMD core, plus
# 236 + 385 cc of xDMA for the two block transposes. THIS workload is what says whether
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
#   xpose_in       x -> x^T    BIT-EXACT. A transpose moves bytes; any tolerance here would
#                              hide the thing most likely to be wrong, which is whether the
#                              transposer was armed on the side this cfg actually built it
#                              on. If the extension is missing on BOTH sides the kernel
#                              falls back to a DM-core element loop -- still correct, so
#                              this check PASSES and only the cycle count betrays it. Read
#                              the trace, not just the verdict.
#   softmax_t      y^T         against the transposed golden. Isolates the kernel itself:
#                              two LANEWISE reduces, the fused subtract+exp+sum, the
#                              one-beat rsqrt(s*s) reciprocal, and two sticky multiplies.
#   softmax_t_e2e  y           the whole transposed chain, in the layer's own layout. This
#                              is also the OUTPUT TRANSPOSE's check and there is no separate
#                              one: y^T has already been scored, so the only thing left that
#                              can move y is y^T -> y. It cannot be bit-exact against a
#                              golden the way xpose_in is, because the value it carries came
#                              through the kernel.
#   softmax_rm     y           the row-major kernel over the same x. The reference arm: if
#                              this fails too, the fault is in the data or in a mechanism
#                              both arms share, not in the transposition.
#
# THE TWO ARMS SHARE THEIR RECIPROCAL, which is what makes this a clean layout A/B. At
# rows = 32 and cols = 128 the row-major kernel's `wide` is false, so it uses the same
# rsqrt(Sexp*Sexp) = 1/Sexp route the transposed one does -- the core's integer divide is
# out of the picture on both sides and the only variable left is orientation.
#
# ======================================================================================
# THE ONE-BEAT HEADROOM
# ======================================================================================
#
# The sticky elementwise reads ONE FLAT SWEEP of 1 + D beats whose first beat is the
# scalar, so the seed must physically precede the tile -- there is no argument that
# separates them. `l1_xt` is therefore (1 + cols) beats: the kernel is handed its base as
# `seed_addr` and base + 64 as `input_addr`, the input transpose writes to base + 64, and
# the kernel CHECKS the adjacency rather than trusting it. Get it wrong and the negated
# maxima land on score row 0 and the tile is read one beat out of phase, with nothing
# reported -- which is exactly why the kernel refuses instead.
#
# SOFTMAX NEEDS A SECOND SUCH BUFFER, for 1/Sexp below the exp tile, and that one is the
# KERNEL's own scratch: it allocates the beat of headroom itself and nothing here has to
# know. Only the input side is a contract on the caller.
#
# SHAPE IS NOT A SWEEP HERE. rows must be 32, the FP16 lanes in one beat: at fewer a beat
# holds several score positions and the per-lane accumulators mix tokens, at more a
# position spans several beats and they mix the other way. Neither faults. And cols <= 255,
# because the reciprocal squares Sexp in FP16 and Sexp <= cols.

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
    SnaxBingoKernelSimdSoftmaxF16F16Args,
    SnaxBingoKernelSimdSoftmaxTF16F16Args,
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

# ROWS is not a knob: one FP16 lane per token is what makes the LANEWISE reduces per-token,
# and 32 lanes is SIMD_WIDTH / elemWidth on this cluster. COLS <= 255 is the reciprocal's
# bound -- the square of Sexp is FP16 and Sexp <= COLS.
ROWS = 32
COLS = 128

# BINGO_CHECK_TYPE_FP16_TOL IS AN ABSOLUTE TOLERANCE, and for softmax that is worth being
# precise about. This tile is deliberately peaked (three spikes per row), so its outputs
# span roughly [1e-3, 0.46]: an absolute 0.02 pins the peaks to ~4% and leaves the tail
# essentially unconstrained.
#
# That is adequate for what this workload is FOR and not for more. A wrong layout, a
# mis-armed transposer, an unseeded sticky latch or a broken reduce all MOVE A PEAK --
# 0.46 landing where 0.001 belongs is off by 23x the tolerance -- so every failure this
# A/B exists to catch is caught. What it cannot see is a few-ULP precision drift in the
# tail, because no absolute tolerance can. The reference app scores that properly, in ULP:
# target/snitch_cluster/sw/apps/snax-simd-softmax prints worst-ULP per path.
#
# 0.02 also matches simd_softmax_1cluster, so the two workloads' verdicts mean the same
# thing. Deliberately the SAME tolerance for both arms here, because the point is that
# they agree.
TOL = 0.02


def _softmax_ref(rows, cols, seed=0x12345):
    """x, and the four goldens the checks score against.

    MODELLED AT THE DEVICE'S ROUNDING, not in FP32 throughout. Every transport between two
    operators in the chain is FP16, so the max, the subtract, the exponential and the row
    sum each narrow -- and a reference that stayed in FP32 would drift from a correct
    kernel by more than the mechanisms under test do. The reciprocal is the exception
    worth naming: it is the TRUE 1/Sexp here, because the device computes rsqrt(s*s)
    rather than an approximation of a reciprocal, and scoring against the true value is
    what makes the ~1.5 ULP of that route visible rather than baked in.

    The tile carries three deliberate spikes per row so the max subtraction has something
    to do; without them every row is nearly flat and a kernel that skipped the subtract
    entirely would still pass.
    """
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    x_rows, y_rows, sums = [], [], []
    for r in range(rows):
        xr = rng.uniform(-2.0, 2.0, size=cols).astype(np.float32)
        for pos, val in ((0, 6.0), (cols // 3, 5.0), (2 * cols // 3, 4.5)):
            xr[pos] = val
        xr = xr.astype(np.float16)
        xf = xr.astype(np.float32)
        m = np.float16(xf.max())                                    # reduce(MAX) -> fp16
        xs = (xf - m.astype(np.float32)).astype(np.float16)         # EW0(ADD), fp16
        e16 = np.exp(xs.astype(np.float32)).astype(np.float16)      # Map(EXP), fp16
        s = np.float16(np.float32(e16.astype(np.float32).sum()))    # Reduce(ADD) -> fp16
        inv16 = np.float16(np.float32(1.0) / s.astype(np.float32))  # rsqrt(s*s) = 1/s
        y_rows.append((e16.astype(np.float32) * inv16.astype(np.float32)).astype(np.float16))
        x_rows.append(xr)
        sums.append(np.float32(s))
    x, y = np.stack(x_rows), np.stack(y_rows)
    # THE SQUARE MUST NOT OVERFLOW, and the bound is on s*s rather than s: every term of
    # Sexp is exp(x - max) <= 1, so s <= cols, but what the datapath carries between EW0
    # and the RSQRT map is s*s in FP16. Refuse here rather than emit data that makes the
    # device return zeros.
    worst = float(max(sums))
    if worst * worst >= 65504.0:
        raise ValueError(
            f"Sexp*Sexp = {worst * worst:.0f} overflows FP16 (65504). The device squares "
            f"Sexp to invert it via rsqrt, so this tile cannot be run at cols={cols}.")
    # `.copy()` is load-bearing: a transposed numpy view is not contiguous, and .view() on
    # it would reinterpret the ORIGINAL buffer's bytes rather than the transposed ones.
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
    x, xt, y, yt = _softmax_ref(ROWS, COLS)
    return {
        "x":  st.put("softmax_t_in", "uint16_t", x.reshape(-1).view(np.uint16)),
        "xt": st.put("softmax_t_xt_golden", "uint16_t", xt.reshape(-1).view(np.uint16)),
        "y":  st.put("softmax_t_y_golden", "uint16_t", y.reshape(-1).view(np.uint16)),
        "yt": st.put("softmax_t_yt_golden", "uint16_t", yt.reshape(-1).view(np.uint16)),
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
            json.dump({"op": "softmax_t",
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

    l1_x = g.l1("smt_x", tot_b)             # row-major input, both arms read it
    l1_xt = g.l1("smt_xt", BEAT + tot_b)    # [ seed beat | x^T ] -- ONE allocation
    l1_yt = g.l1("smt_yt", tot_b)           # y^T, the transposed kernel's output
    l1_y = g.l1("smt_y", tot_b)             # y^T transposed back to row-major
    l1_yrm = g.l1("smt_yrm", tot_b)         # the row-major arm's output
    xt_tile = BingoMemAllocView(l1_xt, BEAT)

    l3_xt = BingoMemAlloc("out_smt_xt", size=tot_b, mem_level="L3")
    l3_yt = BingoMemAlloc("out_smt_yt", size=tot_b, mem_level="L3")
    l3_y = BingoMemAlloc("out_smt_y", size=tot_b, mem_level="L3")
    l3_yrm = BingoMemAlloc("out_smt_yrm", size=tot_b, mem_level="L3")

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
    ck_xt = check("softmax_t_xpose_in", off["xt"], l3_xt, st_xt, 0.0)

    # The kernel writes the negated maxima into l1_xt[0] and reads 1 + cols beats from
    # there; the second headroom buffer (1/Sexp below the exp tile) is its own scratch.
    sm_t = g.node("SoftmaxT", SIMD_CORE, "__snax_bingo_kernel_simd_softmax_t_f16_f16",
                  SnaxBingoKernelSimdSoftmaxTF16F16Args(l1_xt, xt_tile, l1_yt, ROWS, COLS),
                  ck_xt)
    st_yt = g.node("Store_yt", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(l1_yt, l3_yt, tot_b), sm_t)
    ck_yt = check("softmax_t_norm", off["yt"], l3_yt, st_yt, TOL)

    # y^T [128,32] -> y [32,128]: the other way round, because y^T's shape is swapped.
    xout = g.node("Xpose_out", XDMA_CORE, "__snax_bingo_kernel_xdma_transpose_2d",
                  SnaxBingoKernelXdmaTranspose2dArgs(l1_yt, l1_y, COLS, ROWS, 2), ck_yt)
    st_y = g.node("Store_y", HOST_CORE, "__host_bingo_kernel_idma",
                  HostBingoKernelIdmaArgs(l1_y, l3_y, tot_b), xout)
    ck_y = check("softmax_t_e2e", off["y"], l3_y, st_y, TOL)

    # ---- the row-major reference arm, over the SAME x ---------------------------------
    sm_rm = g.node("SoftmaxRM", SIMD_CORE, "__snax_bingo_kernel_simd_softmax_f16_f16",
                   SnaxBingoKernelSimdSoftmaxF16F16Args(l1_x, l1_yrm, ROWS, COLS), ck_y)
    st_rm = g.node("Store_yrm", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(l1_yrm, l3_yrm, tot_b), sm_rm)
    check("softmax_rowmajor", off["y"], l3_yrm, st_rm, TOL)

    os.makedirs(args.output_dir, exist_ok=True)
    dfg.bingo_compile_dfg("SIMD softmax, transposed vs row-major", args.output_dir,
                          args.output_offload_file_name,
                          extra_include_header_list=["softmax_t_data.h"])
    print(f"Generated softmax_t: [{ROWS}, {COLS}], both arms, 4 checks")


if __name__ == "__main__":
    main()
