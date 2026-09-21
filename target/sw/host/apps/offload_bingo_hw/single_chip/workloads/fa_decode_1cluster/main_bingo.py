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
# nodes a tile; at roughly 80 cycles per BINGO edge that is more scheduling than
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
# WHAT IS MEASURED AND WHAT IS CHECKED. The cycle numbers come from the BINGO trace, so
# they are directly comparable with the reference's own per-engine counters. Correctness is
# pinned by three quantities compared against a float model of the same tile:
#
#   m        the running row maximum over all Bc keys
#   rowsum   the sum of exp(S - m) over every key
#   O        the INT32 PV accumulator, under CHECK_O
#
# m and rowsum cover the reduce, the subtract, the exponential, the tap and the l
# recurrence, but both are per-tile values that are identical on every tile here, so
# neither says anything about accumulation ACROSS tiles or across query tiles. O is the
# only one that reaches the second matmul. It is off by default because staging its golden
# perturbs the timing; turn it on for correctness runs.

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
    SnaxBingoKernelXdma1dCopyArgs,
    SnaxBingoKernelXdmaMemsetArgs,
    SnaxBingoKernelSimdFaSoftmaxArgs,
    SnaxBingoKernelGemmFaQkArgs,
    SnaxBingoKernelGemmFaPvArgs,
    SnaxBingoKernelGemmPerfReportArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)

# Core placement, from the generated map (snax_core_roles_defs.h).
_ROLES = core_roles()
GEMM_CORE = _ROLES["gemm"]
SIMD_CORE = _ROLES["simd"]
DMA_CORE = _ROLES["dm"]
XDMA_CORE = _ROLES["xdma"]
HOST_CORE = _ROLES["host"]
CHECK_FP16_TOL = 2
# Signed-int32, relative tolerance -- for an ACCUMULATOR (see host_kernel_args.h).
CHECK_INT32_RELTOL = 5

# ---- the tile ------------------------------------------------------------------------
# VersaCore's single spatial unrolling on this cluster; the same numbers the device reads
# out of gemm_shapes.h. Shapes below are in ARRAY BLOCKS, as the streamer states them.
MESH_ROW, TILE_SIZE, MESH_COL = 16, 4, 16

# Read VersaCore's own busy/stall counters and print them. Must stay in step with the
# 4-cluster workload's flag of the same name, or the two rungs of the scaling ladder stop
# being the same measurement.
MEASURE_ARRAY = True

# Shape 1, S^T = K.Q^T. From params.hjson, whose defaults match the reference's own
# data/params.hjson exactly, so the cycle counts are comparable point for point. Shrink M
# there to get a fast debug loop -- same graph, same tasks, a fraction of the sim time.
M = K = N = NKV = None   # filled by _load_params() before anything derived is computed
# K and V both stream on the iDMA. Moving V to the xDMA hart wedges the run mid-pipeline,
# with the GEMM core short of its task count and the SIMD core stuck in its drain.
CHECK_O = 0              # params.hjson: validate the O accumulator (perturbs timing)
NQ = 1                   # params.hjson: query tiles sharing one K/V pass

BC = BR = DHEAD = S2_M = S2_K = S2_N = QSHIFT = None


def _load_params(param):
    """Derive the whole geometry from params.hjson and the array, in ONE place.

    Everything below -- the tile, the second matmul's shape, the operand bound, the
    golden -- is a function of (M, K, N) and the mesh. Deriving it here is what stops the
    kernel's idea of the tile and the descriptors' idea of it from drifting apart.
    """
    global M, K, N, NKV, CHECK_O, NQ, BC, BR, DHEAD, S2_M, S2_K, S2_N, QSHIFT
    M, K, N = int(param["M"]), int(param["K"]), int(param["N"])
    NKV = int(param["NKV"])
    # Optional, so an older params.hjson still loads.
    CHECK_O = int(param.get("CHECK_O", 0))
    NQ = int(param.get("NQ", 1))
    BC = M * MESH_ROW          # key columns per tile -- the tiling knob
    BR = N * MESH_COL          # query rows
    DHEAD = K * TILE_SIZE      # head dimension -- a model property, not a knob
    # Shape 2, O^T = V^T.P^T. Bc and d are independent, so the two matmuls differ in M, K.
    # Br IS NOT A FREE KNOB. Every per-query-row vector in the softmax arena -- the running
    # max, the running sum, every correction factor -- is ONE SIMD beat, and a score row is
    # one beat too (simd_fa_layout() in offload_hw_kernels/simd.h walks `bc * B`, not
    # `bc * ceil(Br/lanes) * B`). A beat is SIMD_WIDTH = 512 b from the cluster hjson, so it
    # holds exactly 512/16 = 32 fp16 lanes and Br must be 32.
    #
    # A larger Br does not fault: it writes the first 32 rows scrambled and leaves the rest
    # zero, which only a host check catches -- hence the guard below. Raising Br means
    # giving every one of those vectors a second temporal dimension in all eleven SIMD task
    # shapes AND in the arena layout on both sides -- a kernel change, not a parameter. Bc
    # is the knob that is actually free; it is L1 that bounds it.
    if BR != 32:
        raise SystemExit(
            f"fa_decode: Br={BR} (N={N}) but the softmax kernel packs one query-row "
            f"vector per {MESH_COL * 2}-byte SIMD beat, so Br must be 32. "
            f"Grow Bc (M) instead, or rework simd_fa_layout() and the SIMD task shapes.")

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

    # O, the accumulated PV output. This is the ONLY quantity that exercises the GEMM's
    # accumulation: m and rowsum are per-tile values that are IDENTICAL on every tile here,
    # so a run that drops tiles still passes both of them.
    #
    # P leaves the softmax quantised to INT8 -- exp(S-m) is in [0,1], so the baked scale is
    # 127.0 (BINGO_SIMD_I8_SCALE_UNIT) -- and PV contracts over KEYS: O^T = V^T . P^T.
    # P goes in with NO permutation, and that is the whole subtlety.
    #
    # `s` above is ALREADY the score buffer's memory order. The QK GEMM's D port writes with
    # a spatial stride of one key row (Dsl = {bw/8, key_row}), which interleaves the two N
    # blocks as it writes -- "a beat is exactly one key, all Br queries" in gemm_fa.h. So
    # memory holds [key][query], and the transpose in build_data() is how the golden model's
    # [M][N][meshRow][meshCol] output is brought INTO that order, not out of it.
    #
    # PV then reads that same flat buffer as B = [N][K][meshCol][tileSize]
    # (Btb = {K, N, M}, Bts = {b_tile, K*b_tile, 0} -- broadcast over M, b_tile = 64 B).
    # block_gemm_golden_model takes B as a flat array in exactly that convention, so the
    # quantised [key][query] array IS the operand. Permuting it first yields the right
    # VALUES in the wrong PLACES.
    p_b = np.clip(np.rint(p.astype(np.float32) * 127.0),
                  -128, 127).astype(np.int8).reshape(-1)
    o_tile = block_gemm_golden_model(
        S2_M, S2_K, S2_N, MESH_ROW, TILE_SIZE, MESH_COL, v, p_b, 0, 0,
        np.zeros(S2_M * S2_N * MESH_ROW * MESH_COL, dtype=np.int64))
    # Every KV tile is fed the same bytes, so m never moves and every correction factor is
    # exactly 1 -- the accumulator is NKV copies of one tile's contribution.
    o = np.asarray(o_tile, dtype=np.int64) * NKV

    # ---- and now put O where the D32 port ACTUALLY writes it ---------------------------
    #
    # This block is the difference between an O check that passes and one that reports
    # 3338 of 4096 elements wrong while the hardware is perfectly correct.
    #
    # The C/D spatial map is chosen so the FP16 score tile lands row-major -- one key per
    # 64 B beat, which is what the LANEWISE reduce needs. C and D share those strides and
    # the INT32 side is twice as wide, so at INT32 the SAME strides do NOT come out
    # row-major: O lands PERMUTED. That is deliberate and harmless to the arithmetic (C and
    # D permute identically, so O += P.V still accumulates against itself -- the cluster
    # cfg says exactly this), but it means a golden in canonical block order is not
    # comparable with what the host reads back.
    #
    # The map is DERIVED from the descriptors, not assumed: the array serialises a
    # meshRow x meshCol block into chunks of serial_c_d_width, each chunk into channels of
    # bankWidth, and the AGU places channel i at sl0*(i%4) + sl1*((i/4)%4). Ported from the
    # reference's data/datagen.py, which is the model that makes the cluster app's own
    # exact-equality O check pass.
    BANK_W, OUT_W, SERIAL_CD = 64, 32, 1024   # bits; snax_versacore_serial_c_d_width
    SBOUNDS = [4, 4]                          # data_reader_writer_params.spatial_bounds[0]
    slstride = [BANK_W // 8, S2_N * MESH_COL * 16 // 8]
    ts0, ts1 = SERIAL_CD * S2_N // 8, MESH_COL * 16 // 8

    def chan_off(ch):
        off, rem = 0, ch
        for dim, bnd in enumerate(SBOUNDS):
            off += slstride[dim] * (rem % bnd)
            rem //= bnd
        return off

    def scatter(width, blocks, ts2, elem_bytes):
        """Byte offset of every (block, n, row, col) element, in the port's order."""
        per_chunk = SERIAL_CD // width        # elements in one serialised chunk
        per_chan = BANK_W // width            # elements in one channel
        out = {}
        for mm in range(blocks):
            for nn in range(S2_N):
                for r in range(MESH_ROW):
                    for c in range(MESH_COL):
                        e = r * MESH_COL + c
                        out[(mm, nn, r, c)] = (
                            mm * ts2 + nn * ts1 + (e // per_chunk) * ts0
                            + chan_off((e % per_chunk) // per_chan)
                            + (e % per_chan) * elem_bytes)
        return out

    # SELF-TEST, and it is the load-bearing part: run the same model at FP16 and it must
    # reproduce the plain [key][query] row-major layout that `s` above already assumes. If
    # the serialisation model were wrong this assert fires instead of the golden silently
    # disagreeing with the hardware.
    for (mm, nn, r, c), byte in scatter(16, 1, S2_N * 16 * MESH_ROW * MESH_COL // 8, 2).items():
        want = ((mm * MESH_ROW + r) * (S2_N * MESH_COL) + nn * MESH_COL + c) * 2
        assert byte == want, "D32 address model disagrees with the FP16 row-major layout"

    ts2 = S2_N * OUT_W * MESH_ROW * MESH_COL // 8
    o_can = o.reshape(S2_M, S2_N, MESH_ROW, MESH_COL)
    o_mem = np.zeros(o.size, dtype=np.int64)
    seen = np.zeros(o.size, dtype=bool)
    for (mm, nn, r, c), byte in scatter(OUT_W, S2_M, ts2, 4).items():
        w = byte // 4
        assert byte % 4 == 0 and not seen[w], "D32 INT32 address map is not a bijection"
        seen[w] = True
        o_mem[w] = o_can[mm, nn, r, c]
    assert seen.all(), "D32 INT32 address map does not cover the output"
    return a, b, v, m, rowsum, o_mem.astype(np.int32)


def stage(st, a, b, v, m, rowsum, o):
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
        "o": st.put("fa_o_golden", "int32_t", o.astype(np.int32)),
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
    # Allocated FIRST, before the score buffers, and that order is load-bearing: the negate
    # task writes -m_new to rmax (here) and to the one-beat prefix of the live score buffer
    # (below) as a single two-beat shape, and a shape stride is unsigned. Arena first keeps
    # it positive.
    # ONE ARENA PER QUERY TILE. Each carries its own independent (m, l, O) recurrence,
    # which is exactly why NQ costs nothing in correctness: the query tiles never interact.
    arena = [g.l1(f"fa_arena_{q}",
                  SnaxBingoKernelSimdFaSoftmaxArgs.arena_bytes(BC, DHEAD))
             for q in range(NQ)]

    # K and V are DOUBLE BUFFERED because they STREAM: one pair per KV tile, refilled from
    # main memory while the previous tile computes. Every tile is fed the same bytes -- the
    # golden relies on that, and m stops moving after tile 0 -- but they are RE-FETCHED per
    # tile, so the memory traffic is what a real KV stream would cost. Loading once and
    # replaying leaves the iDMA idle and makes any utilisation figure exclude memory.
    # TWO K buffers. A third would let the loads run one tile further ahead, which is the
    # right diagnosis -- every QK is gated by its own K load -- but it also raises the
    # number of simultaneously-ready DMA nodes past what the manager's ready set tolerates,
    # and the run wedges in teardown. Revisit only alongside that limit.
    k8 = [g.l1(f"fa_k8_{i}", M * K * MESH_ROW * TILE_SIZE) for i in range(2)]
    q8 = [g.l1(f"fa_q8_{q}", N * K * MESH_COL * TILE_SIZE)      # B of the score matmul
          for q in range(NQ)]
    v8 = [g.l1(f"fa_v8_{i}", S2_M * S2_K * MESH_ROW * TILE_SIZE) for i in range(2)]
    # The score matmul masks every C channel off (see gemm_fa.h), so nothing is ever
    # READ through this pointer -- the AGU walks addresses that are never dereferenced.
    # It exists only because the descriptor needs a base pointer. One block, not M*N.
    cz = g.l1("fa_cz", MESH_ROW * MESH_COL * 4)                 # C base ptr, never read
    # TWO score buffers. A third lets QK run three deep and does remove the WAR edge it
    # targets, but it loses overall: the extra QK streams A, B and D through TCDM alongside
    # the softmax, and the softmax is the critical path. Slack bought for the GEMM is paid
    # for in SIMD bandwidth.
    # ONE BEAT OF HEADROOM in front of each score buffer. The softmax's fused exp pass reads
    # [-m_new][the tile] as a single contiguous stream. Reserving the beat here lets that
    # pass read the GEMM's own output buffer directly, instead of copying the whole tile
    # into the arena beside a -m_new slot -- a copy comparable in size to the K/V stream
    # itself. The GEMM writes its D at +64; the beat below it is the prefix.
    s16 = [g.l1(f"fa_s16_{i}", 64 + BC * BR * 2) for i in range(2)]  # score tile, fp16
    # ONE TRAILING BEAT past the quantised P. The fused exp pass emits bc/2 INT8 beats and
    # then the tapped row sum, unnarrowed, as ONE contiguous stream -- and no shape can span
    # two allocations, so the row sum lives here rather than in the arena.
    p8 = [g.l1(f"fa_p8_{i}", BC * BR + 64) for i in range(2)]   # quantised P + row sum
    # One arena, one Q buffer and one O accumulator PER QUERY TILE -- that is the
    # entire cost of the reuse. K, V, the score tile and P stay single sets: they are
    # exactly what we are trying not to re-read.
    oacc = [g.l1(f"fa_oacc32_{q}", BR * DHEAD * 4) for q in range(NQ)]


    def load(tag, key, dst, nbytes, after=None):
        return g.node(f"Load_{tag}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                      SnaxBingoKernelIdma1dCopyArgs(h[key], dst, nbytes), after)

    def xload(tag, key, dst, nbytes, after=None):
        """Same transfer on the xDMA hart instead of the DM core's iDMA."""
        return g.node(f"Load_{tag}", XDMA_CORE, "__snax_bingo_kernel_xdma_1d_copy",
                      SnaxBingoKernelXdma1dCopyArgs(h[key], dst, nbytes), after)


    # The loads are chained: one iDMA engine, and serialising them here keeps the graph's
    # ready set small rather than handing the manager four nodes that must queue anyway.
    # ORDER MATTERS, and it is not the order the buffers are declared in. There is one
    # iDMA engine so these serialise regardless; what the chain decides is WHICH of them
    # QK(0) has to wait behind. QK reads K, Q and the zero C bias. V and the O zero are
    # PV's, and PV cannot run until SM(0) is done anyway -- so they belong AFTER the
    # loads QK needs, where they stream underneath QK(0) and SM(0) instead of delaying
    # the first matmul by their own duration.
    # ---- warm every core's config path BEFORE the first byte moves ---------------------
    # A core's FIRST dispatch of a given kernel costs far more than its next, and the
    # difference is instruction-cache line refills. A refill is tens of cycles on an idle
    # fabric and well over a thousand while the iDMA is streaming a tile through the same
    # path, so what the first dispatch costs is decided by WHEN it runs, not by what it
    # does. Left to itself every core takes its first dispatch with the load chain already
    # saturating the fabric, and all three pay the contended price at once.
    #
    # These three nodes take that hit while nothing else is running. They are tiny, they
    # write only their own scratch, and the whole load chain is anchored behind them, so
    # the refills happen on an idle fabric and every real dispatch afterwards hits in the
    # icache. The cost is that the first load starts a little later; the gain is that the
    # first QK, the first softmax and the first arena fill all configure at warm speed.
    # The array's own hardware counters -- the same instrument the 4-cluster workload
    # carries, so the two rungs of the scaling ladder are the SAME measurement. See the
    # perf_addr comment in device_kernel_args.h.
    perf = g.l1("gemm_perf", 64) if MEASURE_ARRAY else 0

    warm_buf = g.l1("fa_warm_buf", 1024)
    warm_a = g.l1("fa_warm_a", MESH_ROW * TILE_SIZE)
    warm_b = g.l1("fa_warm_b", MESH_COL * TILE_SIZE)
    # Zeroed first, on the otherwise idle xDMA core. Not decoration: the warm GEMM reads
    # these, and TCDM that nothing has written reads back X -- harmless in the array's
    # datapath, but there is no reason to feed it X when the same node also warms the
    # xdma_memset path that the arena fills use.
    warm_z = g.node("WarmZero", XDMA_CORE, "__snax_bingo_kernel_xdma_memset",
                    SnaxBingoKernelXdmaMemsetArgs(
                        warm_buf, 1024, SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO))
    warm_ab = g.node("WarmZeroAB", XDMA_CORE, "__snax_bingo_kernel_xdma_memset",
                     SnaxBingoKernelXdmaMemsetArgs(
                         warm_a, MESH_ROW * TILE_SIZE,
                         SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), warm_z)
    # One array block: the smallest dispatch gemm_fa_qk accepts, purely to fetch its
    # config path. C is masked off for the score matmul, so warm_b doubles as its base.
    # Only the xDMA's MEMSET path is warmed, not its 1d_copy path, even though the V
    # stream uses the latter and pays a cold config for it. A warm-up copy hangs: the
    # engine is armed but its finish counter never advances, and xdma_wait_task has no
    # bound, so the xDMA core spins forever. The size is a legal multiple of the datapath
    # width, so it is not the alignment guard in bingo_helpers -- the descriptor is being
    # rejected for some other reason. Diagnose that before adding one back.
    warm_gemm = g.node("WarmGemm", GEMM_CORE, "__snax_bingo_kernel_gemm_fa_qk",
                       SnaxBingoKernelGemmFaQkArgs(warm_a, warm_b, warm_b,
                                                   warm_buf.view(64), 1, 1, 1), warm_ab)
    # Only the CONFIG paths are warmed, not the SIMD task-firing path. Warming that too --
    # a full tiny softmax here -- does remove the first tile's cold run, but it re-times the
    # pipeline so that later softmax runs land under the K/V stream instead, and their
    # contended cost is larger than the cold run it saved. Net zero, so it is not done.
    # ---- build the SIMD task geometry off the critical path ----------------------------
    # The per-tile SIMD kernel opens by writing its task-shape descriptors into the head of
    # the arena and memoises them, so only the first tile of each query tile pays. That
    # first call is expensive for two separate reasons:
    #
    # 1. THE BUILD, a few hundred scalar stores. The cost is not the stores but where they
    #    land: with the iDMA streaming a K/V tile through the same TCDM ports, each store
    #    costs several times its uncontended price, scaling with traffic in flight.
    #
    # 2. THE DISPATCH TAIL, the code after the build, whose instruction-cache lines are
    #    cold and whose refills pay the same contended price.
    #
    # A PROLOGUE node runs both before the load chain has spun up, on the real arena with
    # the real geometry, against buffers tile 0 overwrites anyway. Every real tile then
    # passes PRIMED and takes the memo path. It moves no data and queues no accelerator
    # task, so it does not disturb m, l or O.
    #
    # Making the prologue run the FULL kernel instead -- firing the tasks as well -- is a
    # regression: its own run and drain then sit on the critical path ahead of the first
    # real tile on the same in-order core, which costs more than the cold tail it saves.
    # One per query tile: the descriptors hold absolute pointers into their own arena.
    warm = [g.node(f"Geom{q}", SIMD_CORE, "__snax_bingo_kernel_simd_fa_softmax",
                   SnaxBingoKernelSimdFaSoftmaxArgs(
                       s16[0].view(64), p8[0], arena[q],
                       bc=BC, dhead=DHEAD, tile_idx=0, seed_state=0,
                       geom_mode=SnaxBingoKernelSimdFaSoftmaxArgs.GEOM_PROLOGUE))
            for q in range(NQ)]

    # Q is the query tile: fixed for the whole run, loaded once.
    ld_q = [load(f"Q{q}", "b", q8[q], N * K * MESH_COL * TILE_SIZE, [warm_gemm])
            for q in range(NQ)]
    # No Czero load: with the C channels masked the score matmul never reads this buffer.
    # That removes the single largest load from the head of the chain -- 64 KiB of zeros
    # that QK(0) used to wait behind.
    # O starts at zero: the O matmul sets take_in_new_c, so every output block starts from
    # C, and C is oacc itself. A prefix of the same zero region does it.

    # The softmax's OWN running O lives in the arena and is a different buffer from the
    # INT32 oacc32 above. simd_fa_init_state used to zero it with the SIMD core's stores:
    # dhead beats, ~8 KiB, and the single largest serial bubble in the run. The DM core is
    # idle here, so hand it the same zero region. The device side then only has to seed
    # mrun/lrun, which is two beats.
    _lay = SnaxBingoKernelSimdFaSoftmaxArgs.layout(BC, DHEAD)
    # ZERO THE O ACCUMULATOR ON THE xDMA, not the iDMA.
    #
    # This is a CONSTANT, so fetching it from main memory is wasted traffic on the one
    # engine that is on the critical head: at NQ query tiles it is NQ separate transfers
    # standing in front of K(0), which is the largest single gap in the fill. The xDMA core
    # is otherwise idle -- it owns nothing but its exit node -- so it generates the zeros in
    # place, with the reader channels disabled so no TCDM read is issued either.
    #
    # It no longer waits on ld_q: generating a constant needs no operand, so this runs from
    # cycle zero and overlaps the Q load rather than queueing behind it.
    # THE WHOLE RECURRENCE STATE IS GENERATED, NOT LOADED.
    #
    # m = -inf, l = 0 and O = 0 are constants. The SIMD kernel used to seed m and l itself
    # on tile 0 and the iDMA carried O's zeros; both sat on the critical head, and at NQ
    # query tiles that is NQ transfers standing in front of K(0) plus NQ scalar fills in
    # front of the first softmax. None of it needs an operand, so the idle xDMA core
    # generates all three in place -- reader channels disabled, so not even a TCDM read.
    #
    # -65504 is FP16's most negative finite value, which is what the kernel used; a true
    # -inf would make the first exp(S - m) NaN rather than 0. The pattern is 32 bits
    # because no single BYTE repeats into 0xFBFF.

    NEG_INF16 = 0xFBFFFBFF
    ld_az = []
    for q in range(NQ):
        # ANCHORED BEHIND Q, deliberately.
        #
        # These need no operand, so leaving them dependency-free looks right -- but
        # "eligible" is not "runs first". Nothing runs until the staging barrier lifts,
        # and when it does every ready task competes for the manager at once; unanchored,
        # the fills win that race and push the SIMD prologue and K(0) behind them.
        #
        # Anchoring them behind Q costs nothing real -- Q is short and on a different
        # engine -- and puts them behind the critical chain in the dispatch order rather
        # than ahead of it. The fills are cheap and carry no main-memory traffic, which is
        # the whole reason to keep them on the xDMA.
        z = g.node(f"ArenaOzero{q}", XDMA_CORE, "__snax_bingo_kernel_xdma_memset",
                   SnaxBingoKernelXdmaMemsetArgs(
                       arena[q].view(_lay["oacc"]), DHEAD * 64,
                       SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), ld_q[q])
        # m and l are one beat each and NOT adjacent (rmax, mrun, mnew, delta, corrL,
        # lrun), so they are two fills rather than one. Chained after the O fill: they
        # share the one xDMA engine, and stating the order costs nothing at run time while
        # leaving the dep-tag allocator one chain instead of three concurrent ones.
        mfill = g.node(f"SeedM{q}", XDMA_CORE, "__snax_bingo_kernel_xdma_memset",
                       SnaxBingoKernelXdmaMemsetArgs(
                           arena[q].view(_lay["mrun"]), 64, NEG_INF16), z)
        lfill = g.node(f"SeedL{q}", XDMA_CORE, "__snax_bingo_kernel_xdma_memset",
                       SnaxBingoKernelXdmaMemsetArgs(
                           arena[q].view(_lay["lrun"]), 64,
                           SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), mfill)
        ld_az.append(lfill)

    KBYTES = M * K * MESH_ROW * TILE_SIZE
    VBYTES = S2_M * S2_K * MESH_ROW * TILE_SIZE

    # SOFTWARE-PIPELINED EMISSION: QK(i+1) is created BEFORE PV(i).
    #
    # The manager dispatches a core's tasks in CREATION ORDER, so a ready task cannot
    # overtake a blocked one -- head-of-line blocking on the GEMM's own queue. Emitted
    # naively as QK(0), PV(0), QK(1)..., PV(0) waits on SM(0) and QK(1) sits behind it even
    # though its K landed long before, and the array idles for the whole softmax.
    #
    # Interleaving gives the GEMM queue QK0, QK1, PV0, QK2, PV1, ... so while SM(i) runs the
    # array always has QK(i+1) in front of it. This is what the cluster-side kernel does by
    # construction: it has both QK(i+1) and PV(i-1) available and between them they cover
    # the softmax's latency.
    #
    # Dependencies are unchanged -- they are by reference, not by order -- and every one
    # still names a node created earlier: SM(i) reaches back to pv[i-2] (created at step
    # i-1) and PV(i-1) to sm[i-1] (created at step i-1).
    ld_k, ld_v = [], []
    qk, sm, pv = [], [], []
    TOTAL = NKV * NQ
    for i in range(TOTAL + 1):
        if i < TOTAL:
            j, q = divmod(i, NQ)
            if q == 0:
                # --- stream this KV tile's K and V --------------------------------------
                # Into the buffer every query tile of step j-2 has finished with, so it
                # waits for the LAST of them. Before tile 2 there is nothing but Q.
                k_after = ld_q if j < 2 else [qk[(j - 1) * NQ - 1]]
                ld_k.append(load(f"K{j}", "a", k8[j % 2], KBYTES, k_after))
                # Behind the arena fills, not just behind Q. Issuing V(0) BEFORE the
                # fills instead does clear the first softmax of V traffic -- its config and
                # run drop to their warm values -- but it pushes the fills far enough back
                # that the head grows by more than the contention cost. The fills go first.
                # V and the fills now share
                # the xDMA engine, and the fills are needed by the FIRST softmax while V
                # is not needed until the first PV. Left to the tie-break the V stream
                # gets in front of them and the first softmax waits out two whole V
                # transfers. The fills are a few tens of cycles, so stating the order
                # costs nothing.
                v_after = list(ld_az) if j < 2 else [pv[(j - 1) * NQ - 1]]
                # V streams on the xDMA, K on the iDMA. The two engines then move their
                # tiles CONCURRENTLY instead of queueing on one, which takes the K/V pair
                # off the critical path whenever the array is the longer of the two.
                ld_v.append(xload(f"V{j}", "v", v8[j % 2], VBYTES, v_after))

            # SAME-CORE EDGES ARE NOT REDUNDANT -- do not "optimise" them away.
            #
            # It is tempting to drop a dependency whose producer sits on the SAME core as
            # its consumer, on the grounds that the manager's per-core waiting queue is
            # FIFO and the core runs one task at a time, so the order is guaranteed anyway.
            # It is not: that edge is what PINS the order. The queue order is the
            # compiler's topological order, and with the edge gone the sort is free to
            # interleave the core's tasks differently, which costs far more than the dummy
            # nodes the edge creates.
            #
            # The cost they carry is real but must be paid another way: every predecessor
            # beyond the first per core becomes a dummy_check node
            # (bingo_transform_dfg_add_dummy_check_nodes), which occupies a slot in the
            # CONSUMER's waiting queue. The fix for that is stream ORDER, not edge removal
            # -- see bingo_dfg.bingo_stream_order().
            deps = [ld_k[j]] if q == 0 else [qk[i - 1]]
            if i >= 2:
                deps.append(sm[i - 2])
            qk.append(g.node(f"QK_{j}_{q}", GEMM_CORE,
                             "__snax_bingo_kernel_gemm_fa_qk",
                             SnaxBingoKernelGemmFaQkArgs(k8[j % 2], q8[q], cz,
                                                         s16[i & 1].view(64), M, K, N,
                                                         perf_addr=perf),
                             deps))

            deps = [qk[i]]
            if j == 0:
                deps.append(ld_az[q])
                deps.append(warm[q])
            if i >= 1:
                deps.append(sm[i - 1])
            if i >= 2:
                deps.append(pv[i - 2])
            sm.append(g.node(f"SM_{j}_{q}", SIMD_CORE,
                             "__snax_bingo_kernel_simd_fa_softmax",
                             SnaxBingoKernelSimdFaSoftmaxArgs(
                                 s16[i & 1].view(64), p8[i & 1], arena[q],
                                 bc=BC, dhead=DHEAD, tile_idx=j,
                                 seed_state=0,
                                 geom_mode=SnaxBingoKernelSimdFaSoftmaxArgs.GEOM_PRIMED),
                             deps))

        if i >= 1:
            # PV for the PREVIOUS step, emitted after this step's QK.
            p = i - 1
            pj, pq = divmod(p, NQ)
            deps = [sm[p]] if p == 0 else [sm[p], pv[p - 1]]
            if pq == 0:
                deps.append(ld_v[pj])
            pv.append(g.node(f"PV_{pj}_{pq}", GEMM_CORE,
                             "__snax_bingo_kernel_gemm_fa_pv",
                             SnaxBingoKernelGemmFaPvArgs(v8[pj % 2], p8[p & 1],
                                                         oacc[pq] if pj else 0, oacc[pq],
                                                         S2_M, S2_K, S2_N,
                                                         perf_addr=perf),
                             deps))

    # The array counters, printed once. Anchored on the last PV so every matmul has retired.
    if MEASURE_ARRAY:
        ideal_cc = (2 * (BC * BR * DHEAD)) // (MESH_ROW * MESH_COL * TILE_SIZE) * (NKV * NQ)
        g.node("ArrayPerf", GEMM_CORE, "__snax_bingo_kernel_gemm_perf_report",
               SnaxBingoKernelGemmPerfReportArgs(perf, ideal_cc), pv[-1])

    # ---- check the two statistics the recurrence carries -------------------------------
    # Read straight out of the arena: layout() mirrors the device's own simd_fa_layout(),
    # so there is no second copy of the offsets here.
    lay = SnaxBingoKernelSimdFaSoftmaxArgs.layout(BC, DHEAD)
    last = sm[NKV * NQ - 1]

    def check(tag, field, golden_key, golden, tol):
        l3 = BingoMemAlloc(f"out_fa_{tag}", size=64, mem_level="L3")
        st = g.node(f"Store_{tag}", HOST_CORE, "__host_bingo_kernel_idma",
                    HostBingoKernelIdmaArgs(arena[NQ - 1].view(lay[field]), l3, 64), last)
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
    # rsum moved out of the arena into the LAST tile's p8 buffer, straight after its P
    # beats, because that is where the fused pass's contiguous output stream puts it.
    l3_rs = BingoMemAlloc("out_fa_rowsum", size=64, mem_level="L3")
    st_rs = g.node("Store_rowsum", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(p8[(NKV * NQ - 1) & 1].view((BC // 2) * 64),
                                           l3_rs, 64), last)
    ck_rs = g.node("Check_rowsum", HOST_CORE, "__host_bingo_kernel_check_result",
                   HostBingoKernelCheckResultArgs(h["rowsum"], l3_rs, name="fa_rowsum",
                                                  check_type=CHECK_FP16_TOL,
                                                  num_elements=BR, tolerance=tol_s), st_rs)

    # O -- the accumulator, and the only check that covers the GEMM's contribution to the
    # recurrence. It waits on the LAST PV rather than on the last softmax, because those
    # are different cores and it is precisely a missing PV this is here to catch.
    #
    # The tolerance is RELATIVE and deliberately loose. The golden re-derives P's INT8
    # quantisation from a numpy FP16 exp, and where the hardware rounds the other way a P
    # element moves one LSB, which the matmul then sums over Bc terms. 2% absorbs that; a
    # dropped tile is 1/NKV = 12.5%, so the failure mode this exists for is still caught by
    # a wide margin. Chained after the rowsum check: unordered peers on the one host core
    # are free to the scheduler and expensive to the dep-tag allocator.
    #
    # OFF BY DEFAULT, because the check is not free: staging a 16 KiB INT32 golden into the
    # image and adding two host nodes MEASURABLY slows the cluster's scalar config paths --
    # GEMM_FA_QK_CFG median 81 -> 1,299 cc, and the compute window 66,212 -> 74,720 cc
    # (+12.8%). QK's config runs before the quantiser and writes only descriptor addresses,
    # so this is not the operand values; it is more L3 traffic stalling scalar code, the
    # same mechanism as the memo statics. Enable it for correctness runs, leave it off for
    # anything whose cycles you intend to quote.
    # EVERY query tile is checked, not just the last one. All NQ of them load the SAME Q
    # (ld_q[q] reads h["b"] for every q), so all NQ accumulators must equal the same golden
    # -- which makes "check them all" free of extra golden data and the only way to catch a
    # per-arena failure. Checking oacc[NQ-1] alone left query tiles 0..NQ-2 completely
    # unvalidated, and those are exactly the ones whose arenas differ: arena 0 sits at the
    # bottom of the L1 heap and every later one does not.
    #
    # One L3 landing buffer PER query tile rather than one shared one: a shared buffer would
    # need Store_o(q) ordered behind Check_o(q-1), and an ordering that the graph does not
    # state is an ordering the scheduler is free to violate.
    o_bytes = BR * DHEAD * 4
    l3_o = [BingoMemAlloc(f"out_fa_o_{q}", size=o_bytes, mem_level="L3")
            for q in range(NQ)]
    if CHECK_O:
        prev = ck_rs
        for q in range(NQ):
            st_o = g.node(f"Store_o{q}", HOST_CORE, "__host_bingo_kernel_idma",
                          HostBingoKernelIdmaArgs(oacc[q], l3_o[q], o_bytes),
                          [pv[NKV * NQ - 1], prev])
            prev = g.node(f"Check_o{q}", HOST_CORE, "__host_bingo_kernel_check_result",
                          HostBingoKernelCheckResultArgs(h["o"], l3_o[q], name=f"fa_o{q}",
                                                         check_type=CHECK_INT32_RELTOL,
                                                         num_elements=BR * DHEAD,
                                                         tolerance=0.02), st_o)


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
    a, b, v, m, rowsum, o = build_data()
    st = DataStaging(platform)
    h = stage(st, a, b, v, m, rowsum, o)
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
                   is_host_as_acc=True, chiplet_ids=[0x00],
                   dep_tag_width=platform["dep_tag_width"])
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
