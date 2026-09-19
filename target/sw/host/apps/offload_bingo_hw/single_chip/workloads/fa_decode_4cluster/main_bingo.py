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
    HostBingoKernelIdmaMultiArgs,
    HostBingoKernelCheckResultArgs,
    SnaxBingoKernelPackFaPartialArgs,
    SnaxBingoKernelXdmaChainGatherArgs,
    xdma_monoid_csr0,
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
CHECK_FP32_TOL = 1
BEAT_F16 = 32   # fp16 elements per 64-B beat

# ---- the tile ------------------------------------------------------------------------
# VersaCore's single spatial unrolling on this cluster; the same numbers the device reads
# out of gemm_shapes.h. Shapes below are in ARRAY BLOCKS, as the streamer states them.
MESH_ROW, TILE_SIZE, MESH_COL = 16, 4, 16

# Shape 1, S^T = K.Q^T. From params.hjson, whose defaults match the reference's own
# data/params.hjson exactly, so the cycle counts are comparable point for point. Shrink M
# there to get a fast debug loop -- same graph, same tasks, a fraction of the sim time.
M = K = N = NKV = None   # filled by _load_params() before anything derived is computed
# K and V both stream on the iDMA. Moving V to the xDMA hart wedges the run mid-pipeline,
# with the GEMM core short of its task count and the SIMD core stuck in its drain.
CHECK_O = 0              # params.hjson: validate the O accumulator (perturbs timing)
NQ = 1                   # params.hjson: query tiles sharing one K/V pass

BC = BR = DHEAD = S2_M = S2_K = S2_N = QSHIFT = None

# Clusters the KV axis is split over. FA's softmax recurrence runs ALONG KV, so splitting
# KV gives each cluster an independent recurrence and leaves exactly ONE cross-cluster
# fold at the end -- which is the whole point of this workload. Splitting the GEMM
# contraction instead would make every cluster need every KV tile.
# ACTIVE_CLUSTERS in params.hjson overrides it, which is what makes a scaling sweep
# possible on ONE binary: the shards are independent, so running 1 or 2 of them measures
# the same per-cluster pipeline with less of the fabric and the task manager contended
# for. The default is the full count, so an unchanged params.hjson behaves exactly as before.
NCL = 4
NKV_PER = None           # KV tiles per cluster; set by _load_params()

# Slots per beat in the monoid's lane geometry. F = 2 fields (m, l) and a 512-bit beat
# holds 16 FP32 lanes, so 8 is the largest legal value and packs 8 query rows per beat.
# It is NOT free to change: it must equal the `sigma` the gather's CSR(0) was built with,
# and Br must be a multiple of it.
MONOID_SLOTS = 8

# Score/probability buffers in the QK -> softmax -> PV rotation. Two is the minimum that
# lets QK(i+1) run while the softmax reads QK(i)'s tile; a third breaks the WAR edge that
# otherwise stalls the array for the whole of the first softmax.
#
# MEASURED AT ONE CLUSTER: three LOST. The extra QK streams A, B and D through TCDM
# alongside the softmax, and there the softmax was the critical path, so slack bought for
# the array was paid for in SIMD bandwidth.
#
# That balance does not hold at four clusters. The array idles ~8,300 cc per shard waiting
# out the first softmax and is busy only ~51% of its pipeline, while the SIMD is busy 48% --
# both engines idle in lockstep, so neither is the critical path and the TCDM argument has
# nothing to trade against. Costs one s16 (32,832 B) and one p8 (16,448 B) per cluster.
NSCORE = 2

# Read VersaCore's own busy/stall counters and print them per cluster. The paper metric is
# array utilisation, and a timed trace span is not the same quantity as the array's busy
# counter -- a span also contains the START writes and the retire poll. See the perf_addr
# comment in device_kernel_args.h for the full argument.
MEASURE_ARRAY = False

# STREAM V ON THE SYSTEM iDMA (a PUSH) INSTEAD OF THE CLUSTER xDMA (a PULL).
#
# WHY. Measured on the 4-cluster decode: the fabric delivers 62.7 B/cc AGGREGATE against a
# 512-bit = 64 B/cc port -- 98% of its width -- and 57% of the array's idle time is spent
# waiting on the xDMA carrying V. That is not an xDMA defect: with 2.96 engines overlapping
# they jointly saturate the port, so each engine's apparent rate is only its SHARE of a full
# pipe. Making the xDMA faster cannot help; the port is the wall.
#
# The wall is specific and it is not the only pipe. The quadrant has THREE independent
# 512-bit wide paths, and this workload puts all 2,048 KiB of K+V on one of them:
#
#   occamy_quad.sv:32-35   quadrant_wide_out_req_o  (d512) -- clusters PULL   <- all traffic
#                          quadrant_wide_in_req_i   (d512) -- anything PUSHES <- ~unused
#   occamy_soc.sv:106      Connectivity 16'b0111110111111110 decodes to
#                            IN_QUAD         -> HEMAIA_MEM      (the pull, used today)
#                            IN_SYS_IDMA_MST -> QUAD            (a push, unused for bulk)
#                            IN_HEMAIA_MEM   -> QUAD            (a push, unused)
#   occamy_quad.sv:275     Connectivity 25'b0111110111110111110111110 decodes to
#                            IN_SOC_WIDE -> CLUSTER_0..3        (a push reaches every L1)
#
# So a system-iDMA push into cluster L1 travels a DISJOINT set of wires from a
# cluster-initiated pull, end to end. Splitting K (pull) from V (push) uses two 512-bit
# pipes concurrently instead of queueing both on one.
#
# ARITHMETIC. FA decode's intensity is 4.19 M MAC / 128 KiB = 32 MAC/byte. One array against
# one 64 B/cc port balances at 16 MAC/byte, so a single cluster is compute-bound with 2x
# margin -- which is why 1-cluster reaches 86.1%. FOUR arrays against the SAME port balance
# at 64 MAC/byte, so the 4-cluster case is memory-bound by exactly 2x, and that factor of two
# IS the drop to 65%. Two pipes put the balance back at 32 = the workload's own intensity.
#
# COST: all host kernels run on cluster 0's host core, so the 16 V pushes serialise there.
# That is the correct shape anyway -- there is one system iDMA -- and 1,024 KiB at 64 B/cc is
# ~16,400 cc against a ~29,000 cc pipeline, so it fits alongside the K pulls rather than
# extending them.
# WHICH V TILES RIDE THE PUSH. Measured, both extremes:
#
#   all 16 on the cluster xDMAs (pull):  4 engines concurrent, 18.9 B/cc EACH but
#                                        3.03x overlap -> 47.7 B/cc effective, 21,983 cc
#   all 16 on the host (push):           56.4 B/cc PER TRANSFER -- 3x better, the two-pipe
#                                        model was right about bandwidth -- but ONE issuer,
#                                        so 16 x (1,161 + 433 cc manager round trip)
#                                        -> 41.1 B/cc effective, 35,721 cc. WORSE, and
#                                        busy/pipeline fell 65.0% -> 50.5%.
#
# One serial issuer loses to four concurrent ones even on a faster pipe; 16 x 433 = 6,928 cc
# is pure BINGO dispatch overhead. The answer is not to pick a pipe but to USE BOTH: split so
# each finishes at the same time. Balancing 2,048 KiB across 62.7 B/cc (pull, aggregate) and
# 41.1 B/cc (push, effective) puts ~1,237 KiB on the pull and ~811 KiB on the push, and both
# land at ~20,200 cc against the baseline's 33,447 -- 40% less fabric time.
#
# V(0) deliberately stays on the PULL: it is the tile PV(0) needs first, and the pull path is
# the concurrent one, so it arrives soonest there. Tiles 1..3 go on the push.
# MEASURED, three arms, and the trend is monotonic -- every tile moved to the push COSTS:
#
#   V on push   busy/pipeline   pipeline    array busy
#        0         65.0%         29,291       19,029     <- BEST: all V on the cluster xDMAs
#       12         55.2%         34,412       19,008
#       16         50.5%         37,219       18,785
#
# Array busy never moved (19,029 / 19,008 / 18,785). The entire effect is pipeline stretch,
# i.e. the array waiting longer for V. The push pipe is REAL -- 56.4 B/cc per transfer against
# a contended pull share of 18.9 -- but there is exactly ONE issuer, and each BINGO task on it
# costs a 433 cc manager round trip (MGR_WRITE_DONE -> MGR_GET_READY -> MGR_PREP). At 16 tasks
# that is 6,928 cc of pure dispatch, dragging the effective rate to 41.1 B/cc, below the four
# cluster xDMAs' 47.7 B/cc aggregate.
#
# The lesson is not "the second pipe does not exist" -- it does, and the RTL analysis of it was
# right. It is that ONE SERIAL ISSUER LOSES TO FOUR CONCURRENT ONES, even on a faster pipe.
# Making the push pay off needs the per-task dispatch overhead removed, not more tiles moved.
#
# Empty = all V on the cluster xDMAs, which is the best measured configuration.
V_PUSH_TILES = set()

# HOW MANY V BUFFERS. This is the prefetch depth, and the prefetch depth is what decides how
# much of the K/V stream lands in the FILL rather than inside the pipeline.
#
# The milestone is array busy / pipeline with pipeline = first full-size QK to last PV end,
# so a tile fetched before the first QK does not count against it -- and it is not merely a
# bookkeeping trick, it is the same overlap the snax reference gets by staging its next
# dispatch while the array runs. Measured: total load-engine busy is 37,644 cc against a
# 29,291 cc pipeline, so 8,353 cc of loading ALREADY hides in the fill. Each extra buffer
# moves one more tile per cluster out.
#
# V(j) reuses v8[j % NVBUF], which PV(j - NVBUF) was the last to read, so the WAR edge walks
# back with the depth. L1 is the cap: 392,320 B of 514,816 B at NVBUF=2, and a V tile is
# 65,536 B, so exactly one more buffer fits (89%). A second would need 131,072 B and overflow.
# MEASURED: 3 is WORSE than 2 (63.2% vs 65.0%). The extra buffer did exactly what it was
# supposed to -- the array's wait on V fell 5,801 -> 4,204 cc -- but the pipeline did not
# shrink, because the port is already saturated at 98% of its width. Prefetching earlier
# does not create bandwidth; it only moves which engine is waiting. SIMD and dispatch
# absorbed the whole gain (2,281 -> 3,540 and 1,629 -> 2,570).
#
# This is the same wall the V-push arms hit. When the bottleneck is aggregate bandwidth,
# RESCHEDULING CANNOT HELP -- only less traffic or more bandwidth can.
NVBUF = 2

# PUSH V IN PAIRS ON THE SYSTEM iDMA, one BINGO task per PAIR.
#
# Two measured facts make this the shape to try:
#
#  1. A host iDMA transfer moves 64 KiB at 56.4 B/cc -- three times a contended pull share,
#     because it rides quadrant_wide_in, a pipe disjoint from the clusters' pull. But each
#     BINGO task on the host costs ~433 cc of dispatch, so ONE tile per task yields 41.1
#     B/cc effective and the push loses. Two tiles per task gives 47.6, four gives 51.6.
#
#  2. The all-push arm was worse than the model predicted (35,721 cc of V delivery against
#     a predicted 25,504) because V(j) waits on PV(j-NVBUF): the V chain serialises THROUGH
#     THE COMPUTE, not just through the issuer. Pairing attacks that too -- {V2,V3} issue
#     together after PV(1) instead of V(3) waiting behind PV(2).
#
# PAIRS, NOT QUADS, DELIBERATELY. V(j) and V(j+1) live in DIFFERENT buffers (NVBUF=2), so a
# pair is expressible inside one cluster. Batching four would have to cross clusters, which
# couples their dependencies -- every cluster's batch would wait on the slowest -- and that
# is exactly the coupling that made the 12-push split arm imbalanced (cluster 0 at 60.2%,
# cluster 3 at 49.2%). Worth 47.6 over 51.6 B/cc to keep the shards independent.
# MEASURED 54.6% -- better than one tile per task (50.5%) and still far below leaving V on
# the cluster xDMAs (65.0%). Batching did exactly what it was designed to do:
#
#                    delivered   while running   ISSUER IDLE
#   1 tile/task      29.4 B/cc     56.4 B/cc        39%
#   2 tiles/task     37.0 B/cc     96.9 B/cc        54%      <- 96.9 > the 64 B/cc port
#                                                               because the engine pipelines
#                                                               the second transfer under the
#                                                               first; the pair is genuinely
#                                                               overlapped, not serialised
#
# and it was not enough, because the push tasks are NOT issue-bound. The issuer sits idle for
# MORE than half the window, waiting on the WAR edge: V(j) cannot start until PV(j-NVBUF)
# has freed its buffer. Deeper buffering is the only thing that relaxes that, and L1 caps
# NVBUF at 3 -- which measured WORSE on its own (63.2%).
#
# FIVE ARMS, one conclusion: 4 concurrent cluster xDMAs beat one host issuer every time.
# Leave V on the pull.
V_PUSH_PAIRS = False


def _load_params(param):
    """Derive the whole geometry from params.hjson and the array, in ONE place.

    Everything below -- the tile, the second matmul's shape, the operand bound, the
    golden -- is a function of (M, K, N) and the mesh. Deriving it here is what stops the
    kernel's idea of the tile and the descriptors' idea of it from drifting apart.
    """
    global M, K, N, NKV, CHECK_O, NQ, BC, BR, DHEAD, S2_M, S2_K, S2_N, QSHIFT, NKV_PER, NCL
    M, K, N = int(param["M"]), int(param["K"]), int(param["N"])
    NKV = int(param["NKV"])
    # Optional, so an older params.hjson still loads.
    CHECK_O = int(param.get("CHECK_O", 0))
    # Extra right-shift on the INT8 operands, BEYOND the minimum that keeps a score inside
    # FP16. It exists for the cross-cluster fold, not for the arithmetic.
    #
    # At the minimum shift the scores span thousands, so exp(S - m) underflows for every
    # key but the argmax and every shard's l is exactly 1.0. The monoid folds a key field
    # (m, a max) and an exp-twisted value field (l); with l identically 1 the second field
    # is folding 1 + 0 + 0 + 0 on every lane, so a junction that dropped the value
    # coordinate entirely would still produce the right answer. Shrinking the operands
    # narrows the score spread until the softmax is genuinely soft and l carries a real
    # distribution, which is what makes the merged l* evidence of anything.
    NQ = int(param.get("NQ", 1))
    # Shards actually built. Fewer than num_clusters leaves the rest idle, which is the
    # control arm for any contention measurement; below 2 there is nothing to fold and the
    # gather is skipped. NKV must still divide by it, so a 1-cluster arm wants NKV = NKV_PER.
    NCL = int(param.get("ACTIVE_CLUSTERS", NCL))
    if not 1 <= NCL <= int(param.get("num_clusters", NCL)):
        raise ValueError(f"ACTIVE_CLUSTERS={NCL} must be between 1 and num_clusters")
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
    QSHIFT = qshift(DHEAD) + int(param.get("SCORE_SHIFT_EXTRA", 0))
    # Bc IS the beat count of the score tile -- the transposed layout puts one key, all
    # Br queries, in each 64-B beat -- and the quantiser packs 2:1, so it must be even.
    if BC % 2:
        raise ValueError(f"Bc={BC} must be even: the quantiser packs two beats into one")
    if NKV % NCL:
        raise ValueError(
            f"NKV={NKV} must divide across {NCL} clusters: each cluster owns a disjoint "
            f"run of KV tiles and the merge assumes every shard covers the same count.")
    NKV_PER = NKV // NCL
    if BR % MONOID_SLOTS:
        raise ValueError(
            f"Br={BR} must be a multiple of MONOID_SLOTS={MONOID_SLOTS}: the monoid packs "
            f"that many query rows per beat and a partial beat would fold lanes the pack "
            f"never wrote.")


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



def build_data(a=None, b=None, v=None, seed=42):
    """Operands plus the float model of the softmax over ONE shard's tile.

    Follows the hardware's own sequence at the precision each stage works in:

        S       exact INT32 out of the mesh, converted to FP16 on the D32 port
        m       max over KEYS, per query row
        P       exp(S16 - m), FP32 internally, stored FP16
        rowsum  summed over KEYS in the FP32 accumulator, narrowed once to FP16

    Every KV tile inside a shard is fed the SAME K, so the running maximum stops moving
    after tile 0 and the shard's final m is this tile's m; rowsum likewise.

    The operands are arguments rather than locals so build_shards() can give every cluster
    a DIFFERENT K while they share one Q and one V -- see the note there on why identical
    shards would make the cross-cluster fold untestable.
    """
    rng = np.random.RandomState(seed)
    # Block operand layouts, as block_gemm_golden_model reads them and as the streamer
    # descriptors walk them: A is [M][K][meshRow][tileSize], B is [N][K][meshCol][tileSize].
    if a is None:
        a = rng.randint(-128, 127, size=M * K * MESH_ROW * TILE_SIZE).astype(np.int8) >> QSHIFT
    if b is None:
        b = rng.randint(-128, 127, size=N * K * MESH_COL * TILE_SIZE).astype(np.int8) >> QSHIFT
    # V for the second matmul: [S2_M][S2_K][meshRow][tileSize]. Its value never reaches a
    # check (O is a cycle measurement), but it must be a real operand so the matmul does
    # the work -- a zero V would still take the same cycles, but a denormal-free operand
    # keeps the array off any slow path.
    if v is None:
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
    o = np.asarray(o_tile, dtype=np.int64) * NKV_PER

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


def build_shards():
    """NCL shards with DIFFERENT K, plus the global merge the fabric is supposed to compute.

    Every shard gets its own K and therefore its own (m_c, l_c). That is not incidental:
    with one shared K -- which is what the single-cluster workload uses, since there the
    recurrence just repeats one tile -- all four partials are identical, and a fold that
    silently dropped three of them, or returned the collector's own operand untouched,
    would produce exactly the right answer. Distinct shards are what make the gather's
    result evidence that the gather happened.

    Q and V are shared, as they are in the real decomposition: splitting KV leaves the
    query tile common to every cluster.

    The merge itself is the online-softmax combine the monoid junction implements:

        m* = max_c m_c              l* = sum_c exp(m_c - m*) * l_c

    computed here in FP32 on FP16 inputs, mirroring the device: the arena holds m and l in
    FP16, pack_fa_partial widens the bit pattern to FP32, and the junction folds in FP32.
    """
    rng = np.random.RandomState(1234)
    b = rng.randint(-128, 127, size=N * K * MESH_COL * TILE_SIZE).astype(np.int8) >> QSHIFT
    v = rng.randint(-128, 127,
                    size=S2_M * S2_K * MESH_ROW * TILE_SIZE).astype(np.int8) >> QSHIFT

    shards = []
    for c in range(NCL):
        a_c = (rng.randint(-128, 127, size=M * K * MESH_ROW * TILE_SIZE)
                  .astype(np.int8) >> QSHIFT)
        shards.append(build_data(a=a_c, b=b, v=v))

    m_c = np.stack([np.asarray(sh[3], dtype=np.float16) for sh in shards])     # [NCL, BR]
    l_c = np.stack([np.asarray(sh[4], dtype=np.float16) for sh in shards])

    m_star = m_c.astype(np.float32).max(axis=0)
    l_star = (np.exp(m_c.astype(np.float32) - m_star) *
              l_c.astype(np.float32)).sum(axis=0)

    # Pack (m*, l*) into the junction's lane geometry so the check compares what the
    # collector's buffer actually holds: lane = field*S + slot, field 0 = m, field 1 = l,
    # S = MONOID_SLOTS rows per beat, 16 FP32 lanes per 512-bit beat.
    beats = BR // MONOID_SLOTS
    merged = np.zeros(beats * 16, dtype=np.float32)
    for beat in range(beats):
        for slot in range(MONOID_SLOTS):
            row = beat * MONOID_SLOTS + slot
            merged[beat * 16 + 0 * MONOID_SLOTS + slot] = m_star[row]
            merged[beat * 16 + 1 * MONOID_SLOTS + slot] = l_star[row]

    a_list = [sh[0] for sh in shards]
    return a_list, b, v, m_c, l_c, merged, shards[0][5]


def stage(st, a, b, v, m, rowsum, o):
    """Hand every array to the staging helper and return the handles.

    WHERE these land is the platform's business, not this workload's: a config with a
    memory chiplet gets a mempool.bin, one without gets C arrays in the host image. See
    util/sim/common/bingo_data_staging.py -- addressing a memory chiplet that a config
    does not have reads unmapped memory rather than faulting, which surfaces as an
    arithmetic bug a long way from the cause.
    """
    return {
        # One K per shard. The handles are per-cluster only in WHICH cluster loads them;
        # they are staged once, in L3 or the memory chip as the platform dictates.
        "a": [st.put(f"fa_k8_c{c}", "int8_t", np.asarray(a[c]).astype(np.int8))
              for c in range(NCL)],
        "b": st.put("fa_q8", "int8_t", b.astype(np.int8)),
        "v": st.put("fa_v8", "int8_t", v.astype(np.int8)),
        # The score matmul's C is a zero BIAS and the O accumulator starts at zero. The
        # larger of the two is C, so one region serves both -- and on the host path it
        # costs no image bytes at all, because an uninitialised array lands in .bss.
        "zero": st.put_zeros("fa_zero", "int32_t", M * N * MESH_ROW * MESH_COL),
        # Per-shard goldens, one per cluster. Keeping them is what separates "the fold is
        # wrong" from "a shard is wrong": a cluster whose recurrence quietly did nothing
        # still produces a well-formed partial, and the merged result alone cannot say
        # which of the four it came from.
        "m": [st.put(f"fa_m_golden_c{c}", "uint16_t",
                     np.asarray(m[c]).astype(np.float16).view(np.uint16))
              for c in range(NCL)],
        "rowsum": [st.put(f"fa_rowsum_golden_c{c}", "uint16_t",
                          np.asarray(rowsum[c]).astype(np.float16).view(np.uint16))
                   for c in range(NCL)],
        "o": st.put("fa_o_golden", "int32_t", o.astype(np.int32)),
    }


def stage_merged(st, merged):
    """The gathered (m*, l*), in the junction's own lane order, as FP32."""
    return st.put("fa_ml_merged_golden", "float", np.asarray(merged, dtype=np.float32))


class G:
    """Node and L1-handle factory, bound to one cluster.

    Every handle a cluster allocates carries its cluster id, so a handle from another
    cluster already resolves to a full (chip | cluster | offset) address. That is what
    lets the gather name its remote operands directly -- nothing in this workload has to
    know the cluster map.
    """

    def __init__(self, dfg, cluster=0):
        self.dfg = dfg
        self.cluster = cluster

    def at(self, cluster):
        return G(self.dfg, cluster)

    def l1(self, name, size, cluster=None):
        # The cluster id is part of the NAME as well as the handle: allocation is keyed by
        # name, and four clusters each want their own `fa_arena_0`. The suffix is uniform,
        # so the alphabetical order the allocator lays a cluster's heap out in is the same
        # on every cluster -- which is what keeps any cross-cluster offset reasoning valid.
        cl = self.cluster if cluster is None else cluster
        return BingoMemAlloc(f"{name}_cl{cl}", size=size, mem_level="L1", chip_id=0,
                             cluster_id=cl)

    def node(self, name, core, kname, kargs, after=(), cluster=None):
        # Names carry the cluster for the same reason handles do: the per-cluster pipeline
        # is instantiated four times and every node in it would otherwise collide with its
        # three twins.
        cl = self.cluster if cluster is None else cluster
        nd = BingoNode(assigned_chiplet_id=0, assigned_cluster_id=cl, assigned_core_id=core,
                       node_name=f"{name}_cl{cl}", kernel_name=kname, kernel_args=kargs)
        self.dfg.bingo_add_node(nd)
        for pred in (after if isinstance(after, (list, tuple)) else [after]):
            if pred is not None:
                self.dfg.bingo_add_edge(pred, nd)
        return nd


def writer_junction_index(hw, name):
    """The value WRITER_JCT_<name> has for this cluster, derived from the cluster cfg.

    NOT taken from the cluster's generated snax-xdma-addr.h, where that macro actually
    lives: that header cannot go on the host include path, because all 42 of its XDMA_*
    address macros collide with the chip-level hemaia-xdma-addr.h the host already
    includes. And not a literal either -- the id is the junction's POSITION in the cfg's
    writer_junctions list, so it shifts the moment a cluster gains or loses one.

    Reading it out of the same hjson the RTL was elaborated from is what keeps the number
    and the hardware in step. Searched by key rather than by path so a reorganisation of
    the cfg tree does not silently return the wrong index.
    """
    def find(node):
        if isinstance(node, dict):
            if "writer_junctions" in node:
                return node["writer_junctions"]
            for v in node.values():
                r = find(v)
                if r is not None:
                    return r
        elif isinstance(node, list):
            for v in node:
                r = find(v)
                if r is not None:
                    return r
        return None

    jcts = find(hw)
    if jcts is None:
        raise ValueError("cluster cfg declares no writer_junctions: this workload needs "
                         "the monoid junction to fold the shards in the fabric")
    names = list(jcts.keys())
    if name not in names:
        raise ValueError(f"cluster cfg has writer junctions {names} but this workload "
                         f"needs {name}")
    return names.index(name)


def _build_cluster(dfg, c, h_all, m_all, rowsum_all):
    """The whole single-cluster FA pipeline, placed on cluster `c` over its own KV shard.

    Byte-for-byte the tuned one-cluster graph -- same warm-ups, same double buffering, same
    load-chain order -- with three differences and no others: it runs NKV_PER tiles instead
    of NKV, it reads this shard's K, and it ends at the partial rather than at a check.
    """
    g = G(dfg, c)
    # Resolve the per-shard staged arrays so the body below can keep saying h["a"].
    h = dict(h_all)
    for key in ("a", "m", "rowsum"):
        h[key] = h_all[key][c]
    m, rowsum = m_all[c], rowsum_all[c]

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
    v8 = [g.l1(f"fa_v8_{i}", S2_M * S2_K * MESH_ROW * TILE_SIZE) for i in range(NVBUF)]
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
    s16 = [g.l1(f"fa_s16_{i}", 64 + BC * BR * 2) for i in range(NSCORE)]  # score tile, fp16
    # ONE TRAILING BEAT past the quantised P. The fused exp pass emits bc/2 INT8 beats and
    # then the tapped row sum, unnarrowed, as ONE contiguous stream -- and no shape can span
    # two allocations, so the row sum lives here rather than in the arena.
    p8 = [g.l1(f"fa_p8_{i}", BC * BR + 64) for i in range(NSCORE)]   # quantised P + row sum
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
    # ---- the array's own hardware counters ---------------------------------------------
    # Five words the two FA matmuls accumulate into (busy, stall_a, stall_b, stall_d,
    # dispatches), read straight out of VersaCore's read-only CSRs. This is what makes the
    # utilisation figure comparable to the snax reference's `GEMM core busy %`, which is
    # also a counter read and not a timed span -- see device_kernel_args.h.
    #
    # It is a knob because it is not free: five RO CSR reads per dispatch, and one more L1
    # allocation per cluster. Both are small (~25 cc against a ~4,000 cc dispatch) but the
    # point of the measurement is the dispatch cost, so it should be possible to take the
    # instrument out and confirm the number did not move.
    perf = g.l1(f"gemm_perf_c{c}", 64) if MEASURE_ARRAY else 0

    warm_buf = g.l1("fa_warm_buf", 1024)
    warm_a = g.l1("fa_warm_a", MESH_ROW * TILE_SIZE)
    warm_b = g.l1("fa_warm_b", MESH_COL * TILE_SIZE)
    # Zeroed first, on the otherwise idle xDMA core. Not decoration: the warm GEMM reads
    # these, and TCDM that nothing has written reads back X -- harmless in the array's
    # datapath, but there is no reason to feed it X when the same node also warms the
    # xdma_memset path that the arena fills use.
    # ZERO THE COUNTER ACCUMULATOR BEFORE ANY MATMUL TOUCHES IT.
    #
    # The matmuls do `pf[i] += csrr(...)`, a read-modify-write. TCDM is tc_sram with
    # SimInit="none", so a word this run has not written reads back X -- and X in an
    # accumulator reaches printf, where it trips the RegWriteKnown assertion and the hart
    # dies. That is not a hang: the sim runs to its wall-clock budget with the UART frozen
    # mid-line ("[Cluster 0] GEMM-ARRAY busy="), which reads exactly like a fabric deadlock.
    # Cost is one 64-B fill on the otherwise idle xDMA core, ahead of everything.
    perf_z = ([g.node(f"PerfZero_c{c}", XDMA_CORE, "__snax_bingo_kernel_xdma_memset",
                      SnaxBingoKernelXdmaMemsetArgs(
                          perf, 64, SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO))]
              if MEASURE_ARRAY else [])

    warm_z = g.node("WarmZero", XDMA_CORE, "__snax_bingo_kernel_xdma_memset",
                    SnaxBingoKernelXdmaMemsetArgs(
                        warm_buf, 1024, SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO),
                    perf_z)
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
    TOTAL = NKV_PER * NQ          # this cluster's shard, not the whole KV axis
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
                v_after = list(ld_az) if j < NVBUF else [pv[(j - NVBUF + 1) * NQ - 1]]
                # V streams on the xDMA, K on the iDMA. The two engines then move their
                # tiles CONCURRENTLY instead of queueing on one, which takes the K/V pair
                # off the critical path whenever the array is the longer of the two.
                if V_PUSH_PAIRS:
                    # PUSH, PAIRED. The system iDMA (SOC_WIDE_XBAR_IN_SYS_IDMA_MST) writes
                    # into this cluster's L1 over quadrant_wide_in while the cluster's own
                    # iDMA pulls K over quadrant_wide_out -- two disjoint 512-bit pipes.
                    # One task carries V(j) and V(j+1); the ODD tile reuses the node the
                    # even one created, so the pair costs one dispatch instead of two.
                    # Pinned to cluster 0 because every host kernel is; the DESTINATIONS are
                    # still this cluster's L1, reached by absolute address.
                    if j % 2 == 0:
                        nxt = j + 1
                        pairs = [(h["v"], v8[j % NVBUF])]
                        if nxt < NKV_PER:
                            pairs.append((h["v"], v8[nxt % NVBUF]))
                        ld_v.append(g.node(f"LoadV{j}_{nxt}_c{c}", HOST_CORE,
                                           "__host_bingo_kernel_idma_multi",
                                           HostBingoKernelIdmaMultiArgs(pairs, VBYTES),
                                           v_after, cluster=0))
                    else:
                        # the pair's partner: same node, already issued
                        ld_v.append(ld_v[-1])
                elif j in V_PUSH_TILES:
                    ld_v.append(g.node(f"LoadV{j}_c{c}", HOST_CORE,
                                       "__host_bingo_kernel_idma",
                                       HostBingoKernelIdmaArgs(h["v"], v8[j % NVBUF], VBYTES),
                                       v_after, cluster=0))
                else:
                    ld_v.append(xload(f"V{j}", "v", v8[j % NVBUF], VBYTES, v_after))

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
            # WAR on the score buffer: QK(i) overwrites the tile SM(i-NSCORE) read.
            if i >= NSCORE:
                deps.append(sm[i - NSCORE])
            qk.append(g.node(f"QK_{j}_{q}", GEMM_CORE,
                             "__snax_bingo_kernel_gemm_fa_qk",
                             SnaxBingoKernelGemmFaQkArgs(k8[j % 2], q8[q], cz,
                                                         s16[i % NSCORE].view(64), M, K, N,
                                                         perf_addr=perf),
                             deps))

            deps = [qk[i]]
            if j == 0:
                deps.append(ld_az[q])
                deps.append(warm[q])
            if i >= 1:
                deps.append(sm[i - 1])
            # WAR on the probability buffer: SM(i) overwrites what PV(i-NSCORE) read.
            if i >= NSCORE:
                deps.append(pv[i - NSCORE])
            sm.append(g.node(f"SM_{j}_{q}", SIMD_CORE,
                             "__snax_bingo_kernel_simd_fa_softmax",
                             SnaxBingoKernelSimdFaSoftmaxArgs(
                                 s16[i % NSCORE].view(64), p8[i % NSCORE], arena[q],
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
                             SnaxBingoKernelGemmFaPvArgs(v8[pj % NVBUF], p8[p % NSCORE],
                                                         oacc[pq] if pj else 0, oacc[pq],
                                                         S2_M, S2_K, S2_N,
                                                         perf_addr=perf),
                             deps))

    # The array counters, printed once per cluster. Anchored on the last PV so it cannot
    # run until every matmul this cluster owns has retired, and placed on the GEMM core
    # because the counters are that core's accelerator CSRs -- no other core can read them.
    if MEASURE_ARRAY:
        # One KV tile is QK (Bc x Br x d MAC) + PV (Br x d x Bc MAC) = 4.19 M MAC, which is
        # 4096 cycles at meshRow*meshCol*tileSize = 1024 MAC/cc. The reference states the
        # same figure (snax-flashattn-decode.c: "4.19 M MAC and 4096 GEMM cycles").
        ideal_cc = (2 * (BC * BR * DHEAD)) // (MESH_ROW * MESH_COL * TILE_SIZE) \
                   * (NKV_PER * NQ)
        g.node(f"ArrayPerf_c{c}", GEMM_CORE, "__snax_bingo_kernel_gemm_perf_report",
               SnaxBingoKernelGemmPerfReportArgs(perf, ideal_cc), pv[-1])

    # ---- this shard's own statistics ---------------------------------------------------
    # Checked per shard, not only after the merge. A cluster whose recurrence quietly did
    # nothing still hands the gather a well-formed partial, and the merged result alone
    # cannot say which of the four it came from. Read straight out of the arena: layout()
    # mirrors the device's own simd_fa_layout(), so there is no second copy of the offsets.
    lay = SnaxBingoKernelSimdFaSoftmaxArgs.layout(BC, DHEAD)
    last = sm[NKV_PER * NQ - 1]

    # Host kernels live on cluster 0 core HOST_CORE and may not be placed anywhere else, so
    # these are pinned there explicitly even though the data they read is on cluster c.
    def host(name, kname, kargs, after):
        return g.node(name, HOST_CORE, kname, kargs, after, cluster=0)

    # The tolerances are derived from the goldens themselves rather than fixed, because an
    # absolute tolerance means nothing without a magnitude: one FP16 step at a score of
    # 1000 is 1.0, and at 0.5 it is 0.0005. m is a max of converted integers and is
    # expected exact, so two steps is already slack; rowsum accumulates Bc terms in FP32
    # and narrows once, so it gets four.
    tol_m = float(2 * np.max(np.spacing(m.astype(np.float16))))
    tol_s = float(4 * np.max(np.spacing(rowsum.astype(np.float16))))

    l3_m = BingoMemAlloc(f"out_fa_m_c{c}", size=64, mem_level="L3")
    st_m = host(f"Store_m_c{c}", "__host_bingo_kernel_idma",
                HostBingoKernelIdmaArgs(arena[NQ - 1].view(lay["mrun"]), l3_m, 64), last)
    ck_m = host(f"Check_m_c{c}", "__host_bingo_kernel_check_result",
                HostBingoKernelCheckResultArgs(h["m"], l3_m, name=f"fa_m_c{c}",
                                               check_type=CHECK_FP16_TOL,
                                               num_elements=BR, tolerance=tol_m), st_m)

    # rsum lives in the LAST tile's p8 buffer, straight after its P beats, because that is
    # where the fused pass's contiguous output stream puts it -- not in the arena.
    l3_rs = BingoMemAlloc(f"out_fa_rowsum_c{c}", size=64, mem_level="L3")
    st_rs = host(f"Store_rowsum_c{c}", "__host_bingo_kernel_idma",
                 HostBingoKernelIdmaArgs(p8[(NKV_PER * NQ - 1) % NSCORE].view((BC // 2) * 64),
                                         l3_rs, 64), [last, ck_m])
    ck_rs = host(f"Check_rowsum_c{c}", "__host_bingo_kernel_check_result",
                 HostBingoKernelCheckResultArgs(h["rowsum"], l3_rs,
                                                name=f"fa_rowsum_c{c}",
                                                check_type=CHECK_FP16_TOL,
                                                num_elements=BR, tolerance=tol_s), st_rs)

    # The shard checks are chained rather than left as unordered peers. There is one host
    # core, so they run serially regardless; saying so costs nothing at run time and keeps
    # the per-edge dep tags affordable -- four shards of unordered store->check pairs all
    # land in the same (cluster 0, host, host) cell and each would otherwise need its own.
    return {
        "last_sm": last,
        "arena": arena[NQ - 1],
        "p8_last": p8[(NKV_PER * NQ - 1) % NSCORE],
        "checks": ck_rs,
    }


def build(dfg, h, m_all, rowsum_all, merged_h, jct_monoid):
    """Four KV shards, then ONE cross-cluster fold done in the fabric.

    The shards are independent: each is the tuned single-cluster pipeline over its own KV
    tiles, producing its own (m_c, l_c). What makes this workload different from four
    copies of the one-cluster run is the epilogue -- the online-softmax merge

        m* = max_c m_c        l* = sum_c exp(m_c - m*) * l_c

    is not gathered to one cluster and reduced there. It is computed BY THE FABRIC: each
    cluster packs its partial into the monoid junction's lane geometry, and one
    ChainGather walks the four of them, folding at each hop, so the collector's buffer
    receives the answer rather than the operands.
    """
    shards = [_build_cluster(dfg, c, h, m_all, rowsum_all) for c in range(NCL)]
    g = G(dfg, 0)

    # One shard is the control arm, not a degenerate merge: with nothing to fold, m* = m_0
    # and l* = l_0, so a gather would only copy. The per-shard checks still run, and the
    # pipeline it measures is the same one each shard runs in the four-cluster case.
    if NCL < 2:
        return

    # ---- pack each shard's (m, l) into the junction's lanes -----------------------------
    # On each cluster's own xDMA core, so the gather that consumes it is the very next
    # thing that core does. The pack is scalar FP16->FP32 bit work; that core has no FPU,
    # which is the whole reason it is a kernel and not two lines in the caller.
    part_bytes = SnaxBingoKernelPackFaPartialArgs.packed_bytes(BR, MONOID_SLOTS)
    parts, packs = [], []
    for c, sh in enumerate(shards):
        gc = g.at(c)
        part = gc.l1("fa_ml_partial", part_bytes)
        parts.append(part)
        packs.append(gc.node(
            f"PackPartial_c{c}", XDMA_CORE, "__snax_bingo_kernel_pack_fa_partial",
            SnaxBingoKernelPackFaPartialArgs(
                src_m=sh["arena"].view(SnaxBingoKernelSimdFaSoftmaxArgs.layout(BC, DHEAD)["mrun"]),
                src_l=sh["p8_last"].view((BC // 2) * 64),
                dst=part, n_rows=BR, slots=MONOID_SLOTS),
            sh["last_sm"]))

    # ---- the fold itself ----------------------------------------------------------------
    # chain is the path in DATA order, ENDING at the collector's own destination, and
    # local_src is the collector's own operand. Cluster 0 collects.
    #
    # The gather may not start until EVERY shard has packed: it reads the other three
    # clusters' buffers directly, and nothing in the fabric would tell it that a partial is
    # still being written. Those three edges are the entire synchronisation.
    merged = g.l1("fa_ml_merged", part_bytes)
    gather = g.node(
        "GatherML", XDMA_CORE, "__snax_bingo_kernel_xdma_chain_gather",
        SnaxBingoKernelXdmaChainGatherArgs(
            local_src=parts[0], chain=parts[1:] + [merged],
            size=part_bytes,
            # The identifier WRITER_JCT_MONOIDJUNCTION is device-side only; see
            # writer_junction_index() for why the host emits the derived number instead.
            junction=f"{jct_monoid} /* WRITER_JCT_MONOIDJUNCTION */",
            # nValid is the number of ROWS folded per beat, not a count of operands: the
            # geometry packs MONOID_SLOTS query rows into one beat, and at nValid=1 seven
            # of every eight rows would silently keep the collector's own value.
            jct_csr0=xdma_monoid_csr0(n_valid=MONOID_SLOTS, n=1, n_exp=1, n_add=0,
                                      sigma=3)),
        packs)

    # ---- check the merged result --------------------------------------------------------
    # Compared in the junction's own lane order as FP32, which is what the collector's
    # buffer holds -- m in lanes 0..S-1 of each beat, l in lanes S..2S-1.
    l3_ml = BingoMemAlloc("out_fa_ml_merged", size=part_bytes, mem_level="L3")
    st_ml = g.node("Store_ml", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(merged, l3_ml, part_bytes),
                   [gather] + [sh["checks"] for sh in shards], cluster=0)
    g.node("Check_ml", HOST_CORE, "__host_bingo_kernel_check_result",
           HostBingoKernelCheckResultArgs(merged_h, l3_ml, name="fa_ml_merged",
                                          check_type=CHECK_FP32_TOL,
                                          num_elements=part_bytes // 4,
                                          tolerance=0.02), st_ml, cluster=0)


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
    a, b, v, m, rowsum, merged, o = build_shards()
    st = DataStaging(platform)
    h = stage(st, a, b, v, m, rowsum, o)
    merged_h = stage_merged(st, merged)
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
    build(dfg, h, m, rowsum, merged_h,
          writer_junction_index(hw, "HasMonoidJunction"))

    os.makedirs(args.output_dir, exist_ok=True)
    dfg.bingo_compile_dfg("FlashAttention, KV-sharded over 4 clusters with an in-fabric merge",
                          args.output_dir, args.output_offload_file_name,
                          extra_include_header_list=["fa_data.h"])
    # Every buffer _build_cluster allocates, with its real multiplicity. The previous
    # version of this line counted ONE K buffer, ONE V buffer and a hardcoded two score
    # buffers, and did not scale with NQ -- so it under-reported by ~70 kB and would not
    # have moved at all when a buffer was added. A budget line that cannot go up is worse
    # than none, because the L1 heap is the thing that silently bounds this workload.
    l1 = (2 * (M * K * MESH_ROW * TILE_SIZE)              # k8, double buffered
          + 2 * (S2_M * S2_K * MESH_ROW * TILE_SIZE)      # v8, double buffered
          + NQ * (N * K * MESH_COL * TILE_SIZE)           # q8, one per query tile
          + MESH_ROW * MESH_COL * 4                       # cz, never read
          + NSCORE * (64 + BC * BR * 2)                   # s16
          + NSCORE * (BC * BR + 64)                       # p8
          + NQ * (BR * DHEAD * 4)                         # oacc
          + NQ * SnaxBingoKernelSimdFaSoftmaxArgs.arena_bytes(BC, DHEAD)
          + SnaxBingoKernelPackFaPartialArgs.packed_bytes(BR, MONOID_SLOTS))
    print(f"Generated FlashAttention: Br={BR} Bc={BC} d={DHEAD} NKV={NKV}, "
          f"qshift={QSHIFT}, NSCORE={NSCORE}, L1 buffers {l1:,} B of 514,816 B "
          f"({100 * l1 / 514816:.0f}%)")


if __name__ == "__main__":
    main()
