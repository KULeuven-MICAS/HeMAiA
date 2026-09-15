# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# FLASHATTENTION on the four-engine cluster, as a BINGO task graph.
#
# This is the HeMAiA port of target/snitch_cluster/sw/apps/snax-flashattn, which runs the
# same algorithm as one hand-written multi-core program. Everything about the arithmetic
# is identical -- the same transposed formulation, the same eleven SIMD tasks, the same
# two matmuls, the same descriptors. What changes is WHO SEQUENCES IT.
#
# THE ENCAPSULATION PROBLEM. BINGO schedules at CORE granularity: a node is one kernel on
# one hart, and the hardware manager fires it when its incoming edges are satisfied.
# FlashAttention is a two-engine pipeline whose halves must overlap -- the GEMM computes
# tile j+1 while the SIMD block does the softmax of tile j -- and the SNAX accelerator
# CSRs are reachable only from the hart the engine hangs off, so no single kernel can
# drive both. The reference resolves that with two hand-written loops and a pair of
# monotonic counters in TCDM.
#
# The port keeps the core granularity and puts the overlap in the GRAPH:
#
#     Load K,Q,V,zeros ─┬─> QK(0) ──> SM(0) ──> PV(0) ──> PV(1) ──> PV(2) ──> PV(3)
#                       │      \\        \\   ____/           /        /
#                       │       \\        \\ /    ___________/        /
#                       └─> QK(1) ──> SM(1) ──> ...                 /
#                                  ...
#
# per KV tile j:
#     QK(j)  gemm core   S^T = K.Q^T                  -> S16[j & 1]   (fp16)
#     SM(j)  simd core   the whole online softmax     -> P8[j & 1]
#     PV(j)  gemm core   O^T += V^T.P^T, in place     -> oacc32
#
# THREE NODES PER TILE, not one per accelerator task. One node per SIMD task would be 11
# nodes a tile; at a measured ~82 cycles per BINGO edge that is more scheduling than
# softmax. The eleven tasks are one kernel (__snax_bingo_kernel_simd_fa_softmax) that
# fires them back to back into the block's own task queue and drains once at the end,
# which is exactly what the reference's inner loop does.
#
# THE SKEWED EDGES ARE THE WHOLE POINT. S and P8 are double-buffered, so the pipeline
# depth comes from WAR edges rather than from the data flow:
#
#     SM(j) ──> QK(j+2)      S16[j&1] is free once the softmax that read it is done
#     PV(j) ──> SM(j+2)      P8[j&1]  is free once the matmul that read it is done
#
# Those two are what let QK(j+1) run concurrently with SM(j). Making QK(j+1) depend on
# SM(j) instead -- the naive "the buffer is free when the consumer finished" edge -- would
# serialise the two engines and give up every cycle of overlap the design exists for.
#
# WHAT IS MEASURED AND WHAT IS CHECKED. The cycle numbers come from the BINGO trace
# (GEMM_FULL_RUN and SIMD_RUN spans), so they are directly comparable with the reference's
# own per-engine counters. Correctness is pinned by the two statistics the softmax carries
# across tiles, read back out of the arena and compared against a float model of the same
# tile: m, the running row maximum over all Bc keys, and rowsum, the sum of exp(S - m)
# over every key. Between them they cover the reduce, the subtract, the exponential, the
# tap and the l recurrence. O is NOT checked -- as in the reference, the GEMM accumulates
# it in INT32 while the SIMD rescales an FP16 copy, and the two are never joined because
# the datapath has Int32ToFp16 but no Fp16ToInt32. Treat O as a cycle measurement.

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

from bingo_dfg import BingoDFG                            # noqa: E402
from bingo_platform import core_roles, guard_cluster_count, parse_platform_cfg  # noqa: E402
from bingo_node import BingoNode                          # noqa: E402
from bingo_mem_handle import BingoMemAlloc                     # noqa: E402
from bingo_data_staging import DataStaging                     # noqa: E402
from sim_golden_models import block_gemm_golden_model     # noqa: E402
from bingo_kernel_args import (                           # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelSimdFaSoftmaxArgs,
    SnaxBingoKernelGemmFaQkArgs,
    SnaxBingoKernelGemmFaPvArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)

# Core placement, from the generated map (snax_core_roles_defs.h).
_ROLES = core_roles()
GEMM_CORE = _ROLES["gemm"]
SIMD_CORE = _ROLES["simd"]
DMA_CORE = _ROLES["dm"]
HOST_CORE = _ROLES["host"]
CHECK_FP16_TOL = 2
BEAT_F16 = 32   # fp16 elements per 64-B beat

# ---- the tile ------------------------------------------------------------------------
# VersaCore's single spatial unrolling on this cluster; the same numbers the device reads
# out of gemm_shapes.h. Shapes below are in ARRAY BLOCKS, as the streamer states them.
MESH_ROW, TILE_SIZE, MESH_COL = 16, 4, 16

# Shape 1, S^T = K.Q^T. From params.hjson, whose defaults match the reference's own
# data/params.hjson exactly, so the cycle counts are comparable point for point. Shrink M
# there to get a fast debug loop -- same graph, same tasks, a fraction of the sim time.
M = K = N = NKV = None   # filled by _load_params() before anything derived is computed

BC = BR = DHEAD = S2_M = S2_K = S2_N = QSHIFT = None


def _load_params(param):
    """Derive the whole geometry from params.hjson and the array, in ONE place.

    Everything below -- the tile, the second matmul's shape, the operand bound, the
    golden -- is a function of (M, K, N) and the mesh. Deriving it here is what stops the
    kernel's idea of the tile and the descriptors' idea of it from drifting apart.
    """
    global M, K, N, NKV, BC, BR, DHEAD, S2_M, S2_K, S2_N, QSHIFT
    M, K, N = int(param["M"]), int(param["K"]), int(param["N"])
    NKV = int(param["NKV"])
    BC = M * MESH_ROW          # key columns per tile -- the tiling knob
    BR = N * MESH_COL          # query rows
    DHEAD = K * TILE_SIZE      # head dimension -- a model property, not a knob
    # Shape 2, O^T = V^T.P^T. Bc and d are independent, so the two matmuls differ in M, K.
    S2_M = DHEAD // MESH_ROW
    S2_K = BC // TILE_SIZE
    S2_N = N                   # unchanged: N is Br/meshCol either way
    QSHIFT = qshift(DHEAD)
    # Bc IS the beat count of the score tile -- the transposed layout puts one key, all
    # Br queries, in each 64-B beat -- and the quantiser packs 2:1, so it must be even.
    if BC % 2:
        raise ValueError(f"Bc={BC} must be even: the quantiser packs two beats into one")


def qshift(d):
    """How far to bound the INT8 operands so a score cannot leave FP16.

    A score is a sum of d products of two shifted INT8s, so |S| <= (128>>q)^2 * d.
    Int32ToFp16 SATURATES past 65504 and exp(inf - inf) is NaN, so an overflowing row
    fails the softmax rather than degrading. Real attention divides by sqrt(d) for the
    same reason; bounding the operands is cheaper here and needs no extra pass.
    """
    for q in range(8):
        if (128 >> q) ** 2 * d <= 65504:
            return q
    raise ValueError(f"no INT8 shift keeps d={d} scores inside FP16")



def build_data():
    """Operands plus the float model of the softmax over the same tile.

    Follows the hardware's own sequence at the precision each stage works in:

        S       exact INT32 out of the mesh, converted to FP16 on the D32 port
        m       max over KEYS, per query row
        P       exp(S16 - m), FP32 internally, stored FP16
        rowsum  summed over KEYS in the FP32 accumulator, narrowed once to FP16

    Every KV tile is fed the SAME K, so the running maximum stops moving after tile 0 and
    the final m is this tile's m; rowsum likewise stands for the last tile.
    """
    rng = np.random.RandomState(42)
    # Block operand layouts, as block_gemm_golden_model reads them and as the streamer
    # descriptors walk them: A is [M][K][meshRow][tileSize], B is [N][K][meshCol][tileSize].
    a = rng.randint(-128, 127, size=M * K * MESH_ROW * TILE_SIZE).astype(np.int8) >> QSHIFT
    b = rng.randint(-128, 127, size=N * K * MESH_COL * TILE_SIZE).astype(np.int8) >> QSHIFT
    # V for the second matmul: [S2_M][S2_K][meshRow][tileSize]. Its value never reaches a
    # check (O is a cycle measurement), but it must be a real operand so the matmul does
    # the work -- a zero V would still take the same cycles, but a denormal-free operand
    # keeps the array off any slow path.
    v = rng.randint(-128, 127,
                    size=S2_M * S2_K * MESH_ROW * TILE_SIZE).astype(np.int8) >> QSHIFT

    d32 = block_gemm_golden_model(
        M, K, N, MESH_ROW, TILE_SIZE, MESH_COL, a, b, 0, 0,
        np.zeros(M * N * MESH_ROW * MESH_COL, dtype=np.int64))
    # Block order [M][N][meshRow][meshCol] -> [key][query], which is how the D port lays
    # the tile down (its [4, 8] channel grouping interleaves the two N blocks inside one
    # key row) and how the SIMD core reads it back.
    s = np.asarray(d32).reshape(M, N, MESH_ROW, MESH_COL)
    s = s.transpose(0, 2, 1, 3).reshape(BC, BR)

    s16 = s.astype(np.float16)
    m = s16.max(axis=0)
    p = np.exp(s16.astype(np.float32) - m.astype(np.float32)).astype(np.float16)
    rowsum = p.astype(np.float32).sum(axis=0).astype(np.float16)
    return a, b, v, m, rowsum


def stage(st, a, b, v, m, rowsum):
    """Hand every array to the staging helper and return the handles.

    WHERE these land is the platform's business, not this workload's: a config with a
    memory chiplet gets a mempool.bin, one without gets C arrays in the host image. See
    util/sim/common/bingo_data_staging.py -- addressing a memory chiplet that a config
    does not have reads unmapped memory rather than faulting, which surfaces as an
    arithmetic bug a long way from the cause.
    """
    return {
        "a": st.put("fa_k8", "int8_t", a.astype(np.int8)),
        "b": st.put("fa_q8", "int8_t", b.astype(np.int8)),
        "v": st.put("fa_v8", "int8_t", v.astype(np.int8)),
        # The score matmul's C is a zero BIAS and the O accumulator starts at zero. The
        # larger of the two is C, so one region serves both -- and on the host path it
        # costs no image bytes at all, because an uninitialised array lands in .bss.
        "zero": st.put_zeros("fa_zero", "int32_t", M * N * MESH_ROW * MESH_COL),
        "m": st.put("fa_m_golden", "uint16_t",
                    m.astype(np.float16).view(np.uint16)),
        "rowsum": st.put("fa_rowsum_golden", "uint16_t",
                         rowsum.astype(np.float16).view(np.uint16)),
    }


class G:
    def __init__(self, dfg):
        self.dfg = dfg

    def l1(self, name, size):
        return BingoMemAlloc(name, size=size, mem_level="L1", chip_id=0, cluster_id=0)

    def node(self, name, core, kname, kargs, after=()):
        nd = BingoNode(assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=core,
                       node_name=name, kernel_name=kname, kernel_args=kargs)
        self.dfg.bingo_add_node(nd)
        for pred in (after if isinstance(after, (list, tuple)) else [after]):
            if pred is not None:
                self.dfg.bingo_add_edge(pred, nd)
        return nd


def build(dfg, h, m, rowsum):
    g = G(dfg)

    # ---- L1 ---------------------------------------------------------------------------
    k8 = g.l1("fa_k8", M * K * MESH_ROW * TILE_SIZE)            # A of the score matmul
    q8 = g.l1("fa_q8", N * K * MESH_COL * TILE_SIZE)            # B of the score matmul
    v8 = g.l1("fa_v8", S2_M * S2_K * MESH_ROW * TILE_SIZE)      # A of the O matmul
    cz = g.l1("fa_cz", M * N * MESH_ROW * MESH_COL * 4)         # zero bias for the scores
    s16 = [g.l1(f"fa_s16_{i}", BC * BR * 2) for i in range(2)]  # score tile, fp16
    p8 = [g.l1(f"fa_p8_{i}", BC * BR) for i in range(2)]        # quantised P
    oacc = g.l1("fa_oacc32", BR * DHEAD * 4)                    # O, INT32, accumulated
    arena = g.l1("fa_arena", SnaxBingoKernelSimdFaSoftmaxArgs.arena_bytes(BC, DHEAD))

    def load(tag, key, dst, nbytes, after=None):
        return g.node(f"Load_{tag}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                      SnaxBingoKernelIdma1dCopyArgs(h[key], dst, nbytes), after)

    # The loads are chained: one iDMA engine, and serialising them here keeps the graph's
    # ready set small rather than handing the manager four nodes that must queue anyway.
    ld_k = load("K", "a", k8, M * K * MESH_ROW * TILE_SIZE)
    ld_q = load("Q", "b", q8, N * K * MESH_COL * TILE_SIZE, ld_k)
    ld_v = load("V", "v", v8, S2_M * S2_K * MESH_ROW * TILE_SIZE, ld_q)
    ld_cz = load("Czero", "zero", cz, M * N * MESH_ROW * MESH_COL * 4, ld_v)
    # O starts at zero: the O matmul sets take_in_new_c, so every output block starts from
    # C, and C is oacc itself. A prefix of the same zero region does it.
    ld_o = load("Ozero", "zero", oacc, BR * DHEAD * 4, ld_cz)

    qk, sm, pv = [], [], []
    for j in range(NKV):
        # --- QK(j): the score tile, emitted as fp16 for the SIMD block ------------------
        deps = [ld_cz] if j == 0 else []
        if j == 1:
            deps.append(ld_cz)
        if j >= 2:
            # S16[j&1] last held tile j-2, which is free once the softmax that read it is
            # done. THIS is the edge that keeps the two engines overlapped: QK(j) runs
            # while SM(j-1) is still going.
            deps.append(sm[j - 2])
        qk.append(g.node(f"QK_{j}", GEMM_CORE, "__snax_bingo_kernel_gemm_fa_qk",
                         SnaxBingoKernelGemmFaQkArgs(k8, q8, cz, s16[j & 1], M, K, N),
                         deps))

        # --- SM(j): the whole online softmax, one kernel, eleven SIMD tasks -------------
        deps = [qk[j]]
        if j >= 1:
            # The online softmax is STRICTLY SEQUENTIAL in j: this tile reads the m and l
            # that the previous tile's commit task wrote, and rescales the O that it
            # rescaled. Both SIMD nodes land on the same hart, so they cannot overlap
            # whatever the graph says -- but the ORDER would then be the manager's
            # readiness order rather than the recurrence's, and nothing in the data flow
            # forces it, because QK(j) deliberately does NOT depend on SM(j-1). Making it
            # explicit costs no parallelism: it serialises an engine against itself.
            deps.append(sm[j - 1])
        if j >= 2:
            # P8[j&1] is free once the O matmul that read it is done.
            deps.append(pv[j - 2])
        sm.append(g.node(f"SM_{j}", SIMD_CORE, "__snax_bingo_kernel_simd_fa_softmax",
                         SnaxBingoKernelSimdFaSoftmaxArgs(
                             s16[j & 1], p8[j & 1], arena, bc=BC, dhead=DHEAD,
                             tile_idx=j),
                         deps))

        # --- PV(j): O += P.V, accumulated in place -------------------------------------
        # C and D are the SAME buffer, so the accumulation across KV tiles is the matmul's
        # own C input and costs nothing extra -- which also means the dispatches must not
        # overlap each other, hence the chain through pv[j-1].
        deps = [sm[j], ld_o] if j == 0 else [sm[j], pv[j - 1]]
        pv.append(g.node(f"PV_{j}", GEMM_CORE, "__snax_bingo_kernel_gemm_fa_pv",
                         SnaxBingoKernelGemmFaPvArgs(v8, p8[j & 1], oacc, oacc,
                                                     S2_M, S2_K, S2_N),
                         deps))

    # ---- check the two statistics the recurrence carries -------------------------------
    # Read straight out of the arena: layout() mirrors the device's own simd_fa_layout(),
    # so there is no second copy of the offsets here.
    lay = SnaxBingoKernelSimdFaSoftmaxArgs.layout(BC, DHEAD)
    last = sm[NKV - 1]

    def check(tag, field, golden_key, golden, tol):
        l3 = BingoMemAlloc(f"out_fa_{tag}", size=64, mem_level="L3")
        st = g.node(f"Store_{tag}", HOST_CORE, "__host_bingo_kernel_idma",
                    HostBingoKernelIdmaArgs(arena.view(lay[field]), l3, 64), last)
        return g.node(f"Check_{tag}", HOST_CORE, "__host_bingo_kernel_check_result",
                      HostBingoKernelCheckResultArgs(
                          h[golden_key], l3,
                          name=f"fa_{tag}", check_type=CHECK_FP16_TOL,
                          num_elements=BR, tolerance=tol), st)

    # The tolerances are derived from the goldens themselves rather than fixed, because an
    # absolute tolerance means nothing without a magnitude: one FP16 step at a score of
    # 1000 is 1.0, and at 0.5 it is 0.0005. The reference compares in ULP for the same
    # reason; it can, because it runs on the core. m is a max of converted integers and is
    # expected exact, so two steps is already slack; rowsum accumulates 512 terms in FP32
    # and narrows once, so it gets four.
    tol_m = float(2 * np.max(np.spacing(m.astype(np.float16))))
    tol_s = float(4 * np.max(np.spacing(rowsum.astype(np.float16))))
    check("m", "mrun", "m", m, tol_m)
    check("rowsum", "rsum", "rowsum", rowsum, tol_s)


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
    _load_params(param)

    # The kernel's idea of the array and the descriptors' idea of it must be the same one,
    # so read the mesh out of the cluster cfg rather than trusting the constants above.
    with open(args.hwcfg) as f:
        hw = hjson.loads(f.read())
    unroll = (hw["snax_versacore_core_template"]["snax_acc_cfg"][0]
                ["snax_versacore_spatial_unrolling"][0][0])
    if tuple(int(x) for x in unroll) != (MESH_ROW, TILE_SIZE, MESH_COL):
        raise ValueError(
            f"{args.hwcfg} declares (Mu, Ku, Nu) = {tuple(unroll)} but this workload is "
            f"written for {(MESH_ROW, TILE_SIZE, MESH_COL)}. Bc, Br and d all derive from "
            f"the array, so the descriptors and the golden would both be wrong.")

    # The platform decides where the arrays go, so it has to be parsed before staging.
    platform = parse_platform_cfg(args.platformcfg)
    a, b, v, m, rowsum = build_data()
    st = DataStaging(platform)
    h = stage(st, a, b, v, m, rowsum)
    if args.data_h is not None:
        n = st.emit(args.data_h, args.output_dir)
        print(f"Staged {n} B of operands and goldens "
              f"{'on the memory chiplet' if st.on_memchip else 'in the host image'}")

    if args.configs_out is not None:
        with open(args.configs_out, "w") as f:
            json.dump({"op": "flashattn",
                       "configs": [{"Br": BR, "Bc": BC, "d": DHEAD, "nkv": NKV,
                                    "M": M, "K": K, "N": N,
                                    "S2_M": S2_M, "S2_K": S2_K, "S2_N": S2_N,
                                    "qshift": QSHIFT,
                                    "mesh": [MESH_ROW, TILE_SIZE, MESH_COL]}]},
                      f, indent=2)

    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return

    dfg = BingoDFG(num_chiplets=1,
                   num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
                   num_cores_per_cluster=platform["num_cores_per_cluster"],
                   is_host_as_acc=True, chiplet_ids=[0x00])
    build(dfg, h, m, rowsum)

    os.makedirs(args.output_dir, exist_ok=True)
    dfg.bingo_compile_dfg("FlashAttention (GEMM + SIMD, one query tile x NKV KV tiles)",
                          args.output_dir, args.output_offload_file_name,
                          extra_include_header_list=["fa_data.h"])
    l1 = (M * K * MESH_ROW * TILE_SIZE + N * K * MESH_COL * TILE_SIZE
          + S2_M * S2_K * MESH_ROW * TILE_SIZE + M * N * MESH_ROW * MESH_COL * 4
          + 2 * BC * BR * 2 + 2 * BC * BR + BR * DHEAD * 4
          + SnaxBingoKernelSimdFaSoftmaxArgs.arena_bytes(BC, DHEAD))
    print(f"Generated FlashAttention: Br={BR} Bc={BC} d={DHEAD} NKV={NKV}, "
          f"qshift={QSHIFT}, L1 buffers {l1} B of {512 * 1024} B")


if __name__ == "__main__":
    main()
