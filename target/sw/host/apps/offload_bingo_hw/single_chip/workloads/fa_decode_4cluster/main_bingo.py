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
# nodes a tile; a BINGO edge costs more than a SIMD task, so that is more scheduling than
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
#   O        the INT32 PV accumulator
#
# m and rowsum cover the reduce, the subtract, the exponential, the tap and the l
# recurrence. O is the only one that reaches the second matmul, and it is always checked
# (a prefix of O_CHECK_ELEMS elements).

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
from bingo_dfg import BingoDFG                            # noqa: E402
from bingo_platform import core_roles, guard_cluster_count, parse_platform_cfg  # noqa: E402
from bingo_node import BingoNode                          # noqa: E402
from bingo_mem_handle import (BingoMemAlloc, BingoMemAllocView,   # noqa: E402
                              BingoMemFixedAddr, BingoMemSymbol)
from bingo_data_staging import DataStaging                     # noqa: E402
from sim_golden_models import block_gemm_golden_model     # noqa: E402
from bingo_kernel_args import (                           # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdma1dCopyArgs,
    SnaxBingoKernelXdmaMulticastArgs,
    SnaxBingoKernelXdmaMemsetArgs,
    SnaxBingoKernelSimdFaSoftmaxArgs,
    SnaxBingoKernelGemmFaQkArgs,
    SnaxBingoKernelGemmFaPvArgs,
    SnaxBingoKernelGemmPerfReportArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelIdmaMultiArgs,
    HostBingoKernelCheckResultArgs,
    HostBingoKernelCheckResultF16RelArgs,
    SnaxBingoKernelPackFaPartialArgs,
    SnaxBingoKernelXdmaChainGatherArgs,
    XDMA_SSJ_IDENTITY,
    XDMA_SSJ_ROWS,
    xdma_softmax_stats_csrs,
    xdma_softmax_stats_groups,
)

# Core placement, from the generated map (snax_core_roles_defs.h).
_ROLES = core_roles()
GEMM_CORE = _ROLES["gemm"]
SIMD_CORE = _ROLES["simd"]
DMA_CORE = _ROLES["dm"]
XDMA_CORE = _ROLES["xdma"]
HOST_CORE = _ROLES["host"]
CHECK_BYTE_EXACT = 0        # data_size IS the byte count for this mode
CHECK_FP16_TOL = 2
# Signed-int32, relative tolerance -- for an ACCUMULATOR (see host_kernel_args.h).
CHECK_INT32_RELTOL = 5

# ---- the tile ------------------------------------------------------------------------
# VersaCore shape 0 on this cluster; the same numbers the device reads
# out of gemm_shapes.h. Shapes below are in ARRAY BLOCKS, as the streamer states them.
MESH_ROW, TILE_SIZE, MESH_COL = 16, 4, 16

# Shape 1, S^T = K.Q^T. From params.hjson, whose defaults match the reference's own
# data/params.hjson exactly, so the cycle counts are comparable point for point. Shrink M
# there to get a fast debug loop -- same graph, same tasks, a fraction of the sim time.
M = K = N = NKV = None   # filled by _load_params() before anything derived is computed
# K streams on the iDMA, V on the xDMA.
NQ = 1                   # params.hjson: query tiles sharing one K/V pass

BC = BR = DHEAD = S2_M = S2_K = S2_N = QSHIFT = None

# Clusters the KV axis is split over. FA's softmax recurrence runs ALONG KV, so splitting
# KV gives each cluster an independent recurrence and leaves exactly ONE cross-cluster
# fold at the end -- which is the whole point of this workload. Splitting the GEMM
# contraction instead would make every cluster need every KV tile.
# ACTIVE_CLUSTERS in params.hjson overrides it, which is what makes a scaling sweep
# possible on ONE binary: the shards are independent, so running 1 or 2 of them measures
# the same per-cluster pipeline with less of the fabric and the task manager contended
# for. The default is the full count.
NCL = 4
NKV_PER = None           # KV tiles per cluster; set by _load_params()

# HOW THE WORK IS SPLIT ACROSS THE CLUSTERS OF ONE QUADRANT:
#
#   "kvsplit"  Each cluster owns a disjoint run of KV tiles and its own K. The softmax
#              recurrence runs ALONG KV, so every cluster carries an independent (m, l)
#              and ONE cross-cluster fold closes the run -- which is what exercises the
#              ChainGather over the SoftmaxStatsJunction.
#
#   "headpar"  Each cluster owns one QUERY HEAD of a GQA group. The group shares its KV
#              head by definition, so K and V are byte-identical on all four clusters and
#              only Q differs. Each KV tile is read from main memory ONCE, by its owner
#              cluster (PULL_ROTATE), and the other three clusters pull it from the owner's
#              L1. No fold: each cluster's output is already a complete attention result
#              for its own head.
#
# WHY THE DEFAULT IS headpar. FA decode's arithmetic intensity is Br exactly -- each KV byte
# feeds Br query rows -- and Br is pinned by the SIMD beat. All clusters mux onto ONE wide
# path, so the machine's balance point is well above that intensity and the kernel is
# memory-bound by construction. One cluster on its own is not, because the same intensity
# then sits against a much lower balance point.
#
# kvsplit cannot fix that from software. It divides the compute by the cluster count and
# multiplies the KV traffic by it, so the ratio is a property of the DECOMPOSITION, not of
# the schedule. headpar instead amortises one KV read over every cluster's query rows, so the
# effective intensity rises by the cluster count and the balance point moves back to the
# compute-bound side. Same total work, a fraction of the main-memory traffic.
#
# headpar needs the shared read: several clusters reading the SAME bytes from main memory
# concurrently is worse than the same number reading disjoint bytes. A multicast of a shared
# operand is not used: it loads only the issuing cluster's xDMA, and the adapter holds ONE
# remote-write context, which is also what makes BCAST_SPREAD hang.
#
# What binds afterwards is the SIMD softmax, not memory -- but only for as long as the
# softmax is EXPOSED. Whether raising NQ pays therefore depends on the cluster hardware and
# not on this file: it doubles array work on the same delivered bytes, which is a large win
# when the array is the constraint and close to nothing when the softmax it also doubles is
# already the constraint.
#
# The cross-chiplet half of the idea is KV-split + the in-fabric fold: chiplets do not share
# a port.
#
# CARRYING AN UNUSED KERNEL IS NOT FREE. Adding one SNAX_EXPORT_FUNC entry shifts every
# kernel address and the .rodata dispatch table, and .rodata reads from a cluster kernel are
# blocking fabric round trips. A device library entry costs something even on the arms that
# never call it; #ifdef it out if that matters.
DECOMP = "headpar"

# WHICH operand rides a multicast, under headpar. Both are False: every operand moves as an
# ordinary DMA -- one cluster reads main memory, the other three read that cluster's L1 (see
# K_PULL_FROM_CL0 and V_PULL_FROM_CL0). The multicast stalls mid-stream on a long KV axis,
# and it loads only the issuing cluster's xDMA, which makes that cluster the slowest of the
# group; under one remote-write context there is no way to spread it (see BCAST_SPREAD).
#
# If it is re-enabled, K is the operand to broadcast: the first QK waits for it, so the
# shared read sits on the critical path's own operand while V, needed later by PV, streams
# underneath on an engine that head-parallelism otherwise leaves idle. Broadcasting BOTH
# operands makes one issuer serialise two streams into a single window during which every
# array idles.
BCAST_K = False
BCAST_V = False

# WHO ISSUES THE BROADCASTS. LEAVE THIS FALSE -- IT HANGS.
#
# True spreads tile j's broadcast onto cluster (j % NCL), so every engine both sends its own
# tiles and receives the others'. The xDMA adapter holds ONE remote-write context, and a
# receiver's grant manager wedges when two sources write into it: a full fan-out and a
# two-issuer fan-out wedge at the same point in the program, and the first to wedge belongs
# to a pure RECEIVER. No read-stall watchdog fires, so this is not the finish-manager read
# path.
#
# Symptom if you trip it anyway: two xDMA harts spinning forever on `csrr 0x409` in
# xdma_wait_task, and one .dasm growing past a gigabyte.
BCAST_SPREAD = False


# WHICH ENGINE GETS TILE j's V, RELATIVE TO ITS K.
#
# BCAST_SPREAD alone does not help the FIRST tile. A COLD transfer costs several times what
# the same transfer costs warm, and when the same cluster owns both operands of tile 0 the
# two cold transfers are SERIAL, so the head is two cold transfers long before any compute
# starts. A non-zero skew puts tile j's V on a different engine from its K, so the two cold
# transfers overlap instead of queueing. NCL // 2 is the maximum separation on a quadrant.
BCAST_V_SKEW = 0

# WARM THE MULTICAST PATH BEFORE THE FIRST REAL TILE, with a small transfer into scratch.
# Leave False.
#
# A cold broadcast runs several times longer than a warm one, and the head warm-ups elsewhere
# in this file exist because a core's first dispatch of a kernel is dominated by icache
# refills. But a small transfer is a different descriptor shape from a full tile, so it does
# not warm the path a real tile takes, and the cold broadcast most likely reflects the whole
# quadrant starting up (contention, not refills); the warm-up is only added work on the
# issuing engine. A useful warm-up would have to be a FULL-SIZE transfer.
BCAST_WARMUP = False

# Let head-parallel put V on the cluster's OWN xDMA, the way kvsplit does.
#
# The `bcast is not None` branch below forces V onto the iDMA under head-parallel, because a
# cluster cannot absorb an incoming broadcast while its own xDMA is running a transfer. That
# rule is right WHILE ANYTHING IS BEING BROADCAST; with BCAST_K and BCAST_V both off there is
# no incoming remote write at all, and this knob puts V on the otherwise idle xDMA so K and V
# are paired on separate engines (both on one engine makes that engine bind). Do not set this
# True together with a broadcast: that is the arm whose receiving xDMAs all wedge.
V_ON_XDMA = False

# DO NOT BROADCAST THE FIRST TILE -- pull it per-cluster like KV-split does. Leave False.
#
# A cold first multicast costs several times what a warm one does, in every phase (kernel
# entry, slot config and the transfer itself), and it sits on the startup critical path, so
# loading tile 0 per-cluster shrinks the head. But tile 0 then lands at the head of the same
# iDMA chain that feeds V(0); with a two-deep V buffer that chain is gated by PV completion,
# so a delay at its head propagates down the whole chain, and the cross-cluster WAR edges
# make every cluster wait for the slowest of the group, which per-cluster loading
# de-synchronises. The iDMA is not spare capacity here: it is already the critical engine for
# V. With the rotated pull (PULL_ROTATE) the head's memory traffic is already spread over the
# clusters, and the two knobs attack the same serialisation, so they are not additive.
BCAST_SKIP_FIRST = False

# PULL V FROM CLUSTER 0's L1 INSTEAD OF FROM MAIN MEMORY.
#
# Under headpar every cluster's V is the same bytes. Read independently by all four, that is
# four main-memory reads of every tile. The alternative: cluster 0 reads a tile once, the
# other three copy it out of cluster 0's L1 with their OWN iDMAs. Same number of transfers
# per engine, a quarter of the main-memory traffic, and the inter-cluster links carry the
# rest.
#
# bingo_l1_alloc(chip, cluster, size) returns a full SoC address, so naming another
# cluster's buffer needs no address arithmetic.
#
# NOT the same as a broadcast: this is a PULL, so the three copies proceed in parallel on
# three engines rather than being issued serially by one.
#
# What the pull relieves is STARTUP CONTENTION, not main-memory bandwidth (the shared port is
# only half used): with per-cluster reads every cluster's first fetch hits main memory at the
# same instant, and moving most of them onto the inter-cluster links cuts the head.
#
# It costs one extra dependency per tile, since the other clusters wait for the owner's copy,
# which lengthens the pipeline. The two effects scale oppositely: the head saving is FIXED,
# the pipeline penalty is PER TILE, so beyond some sequence length reading main memory
# per-cluster should win.
V_PULL_FROM_CL0 = True

# PULL K CROSS-CLUSTER TOO, ON THE OTHERWISE IDLE xDMAs -- no multicast at all.
#
# The same move as V_PULL_FROM_CL0, applied to K: routing traffic off main memory and onto
# the inter-cluster links cuts the head, which is a CONTENTION problem. It also removes the
# last multicast. A multicast's FIRST invocation costs several times what a warm one does, in
# every phase, and it makes the issuing cluster asymmetric: the only one whose xDMA has work,
# and the slowest of the group.
#
# The arrangement, with both engine types busy on every cluster (PULL_ROTATE rotates which
# cluster is the source):
#   cluster 0     xDMA: K(j) from main memory      iDMA: V(j) from main memory
#   clusters 1-3  xDMA: K(j) from cluster 0's L1   iDMA: V(j) from cluster 0's L1
#
# Every cluster then issues one K-sized and one V-sized transfer per tile on separate
# engines, and main-memory traffic is one K and one V read per tile. This is only legal
# because nothing is being broadcast any more: the rule that a receiving cluster's xDMA must
# stay idle exists to keep it free to absorb an incoming multicast, and there is no longer one
# to absorb.
#
# On a long KV axis the pull is the arm that finishes: the head is a fixed cost that
# amortises, while the multicast's per-tile risk does not.
K_PULL_FROM_CL0 = True

# ROTATE WHICH CLUSTER IS THE SOURCE, per tile.
#
# Without rotation, cluster 0 is the source for EVERY tile of both operands: its TCDM serves
# three remote readers on top of its own array, and it is the only cluster reading main
# memory. That is the same asymmetry the broadcast had, moved from the write path to the read
# path.
#
# With rotation, tile j is fetched from main memory by cluster j % NCL and pulled by the other
# three. Every cluster then reads a quarter of the tiles from memory and sources a quarter for
# its neighbours. Main-memory traffic is unchanged; what changes is that no single TCDM is the
# hotspot and no single cluster carries the fetch latency for the whole stream.
#
# SAFE AGAINST THE MULTI-ISSUER DEADLOCK, unlike spreading the broadcast. That hang is in the
# remote-WRITE path -- a receiver's grant manager wedges when two sources write into it. A pull
# is a READ issued by the consumer's own engine, so rotating the source adds no concurrent
# remote writers.
PULL_ROTATE = True


# HOW MANY LEADING TILES EVERY CLUSTER READS FROM MAIN MEMORY ITSELF, instead of pulling.
# Leave at 0.
#
# A pulled tile is a TWO-HOP dependency: main memory -> the owner's L1 -> the other three L1s.
# In steady state the second hop overlaps the previous tile's compute. For the FIRST tile
# nothing overlaps it: the other clusters sit idle for the whole of the owner's read before
# their own pull can even start. Reading the leading tiles directly shrinks that head, but
# tile j < this is then read from main memory NCL times instead of once, and at the fill every
# cluster reads at the same instant, so the reads serialise on the single quadrant exit. The
# two-hop pull is slower per tile but it STAGGERS; four direct reads are faster per tile and
# arrive together. The exit's AVERAGE occupancy is low, but an average never justifies adding
# a burst.
PULL_SKIP_FIRST = 0


# HOW MANY OF O'S ELEMENTS THE HOST CHECKS, as a fraction of the BR x DHEAD accumulator.
#
# WHY IT IS WORTH A KNOB. The O checks are the most expensive thing in the simulation --
# together they can dominate the whole run. The host walks the accumulator out of L3 with no
# D-cache, so every element is a blocking fabric round trip. Shrinking them is the cheapest
# way to make an iteration faster, and iteration speed is what a sweep is limited by.
#
# WHY NOT ZERO. O is the only check that covers ACCUMULATION: m and l can be right while the
# recurrence across tiles is wrong. Never turn it off; shrink it.
#
# WHAT A PREFIX COVERS. The accumulator is [BR][DHEAD] row-major, so a prefix of N elements is
# the first N/DHEAD query rows in full. Every KV tile contributes to every row, so one whole
# row already exercises the full accumulation chain; more rows buy independent samples of it,
# not extra depth. 512 = 4 of the 32 rows.
O_CHECK_ELEMS = 512

# DEBUG ONLY -- see the tol_m/tol_s override. Set False to restore real tolerances.
DEBUG_LOOSE_ML = False

# DIAGNOSTIC for a wrong running max (cluster 2 only; off by default). Stores and checks m
# after EVERY KV tile against the final m golden, which names the first tile at which it
# differs and with it the s16/p8 buffer (i % NSCORE) responsible. The final golden is right
# after every tile only when all KV tiles carry the same bytes (m then reaches its final value
# at tile 0), so this needs identical tiles. It adds host round-trips between tiles, so it
# suits a deterministic failure; a race could hide behind them.
DEBUG_PER_TILE_M = False

# DIAGNOSTIC for a wrong running max (cluster 2 only; off by default). QK(i) and
# QK(i + NSCORE) write the SAME s16 buffer; when both KV tiles carry the same K and Q bytes
# their score tiles must be BYTE-IDENTICAL, so storing the whole tile after each and
# comparing them needs no golden: the first store IS the golden for the second. With distinct
# KV tiles the two differ by construction, so this too needs identical tiles.
#
#   they match    -> the score tile is fine and m goes wrong later, in the SIMD recurrence
#   they differ   -> QK is not overwriting s16
DEBUG_S16_COMPARE = False
# Which tile's score buffer to compare against its NEXT use of the same buffer: tile
# S16_CMP_BUF against tile S16_CMP_BUF + NSCORE. Pick a tile that uses the buffer under
# suspicion.
S16_CMP_BUF = 2


def _pull_owner(j):
    """Which cluster fetches tile j from main memory; the rest pull it from that cluster."""
    return (j % NCL) if PULL_ROTATE else 0


def _bcast_owner(j, skew=0):
    """Which cluster's xDMA issues the broadcast of KV tile j.

    With BCAST_SPREAD the skew rotates tile ownership; without it the skew still selects a
    FIXED engine, which is how K and V get one issuer each. That two-issuer split is the
    useful middle point: it halves the head serialisation without asking four clusters to
    issue at once, and it is also the bisect the adapter hang wants -- if TWO concurrent
    issuers already hang, the star multicast is irrelevant and any pair reproduces it.
    """
    return ((j + skew) % NCL) if BCAST_SPREAD else (skew % NCL)

# Score/probability buffers in the QK -> softmax -> PV rotation. Two is the minimum that
# lets QK(i+1) run while the softmax reads QK(i)'s tile; a third breaks the WAR edge that
# otherwise stalls the array for the whole of the first softmax.
#
# A third buffer means a third QK streaming A, B and D through TCDM alongside the softmax,
# through the same ports the cross-cluster V pull already uses. TCDM BANDWIDTH is shared
# whether or not either engine is "the critical path", and two engines idling in lockstep are
# evidence that something they SHARE is the constraint. It also fits in L1 with little to
# spare, and NSCORE = 3 WEDGES: all cluster engines retire, every check up to the O checks
# passes, and the host then hangs in the first O check's tolerance scan on a load from L3
# that never returns (not an unmapped address, which returns 0xBADCAB1E). The fabric never
# answers, as in the teardown wedges seen elsewhere on this platform -- diagnose that, not the
# score buffer, before using 3.
#
# A third buffer would let two QKs cover one softmax where two buffers give only one, which
# the softmax overruns; whether that pays depends on the cluster hardware.
NSCORE = 2

# DIAGNOSTIC. The WAR depth, normally == NSCORE. Setting it BELOW NSCORE keeps the extra
# score/probability buffers but restores the tighter dependency of a smaller NSCORE, which
# separates two explanations of a failure with NSCORE > 2 that are otherwise confounded:
#   passes with NSCORE_WAR < NSCORE -> the buffer INDEXING is fine and the failure is a race
#                                      opened by the extra scheduling freedom
#   fails   with NSCORE_WAR < NSCORE -> the failure is in the 3-buffer indexing itself
# It is never correct to set it ABOVE NSCORE.
NSCORE_WAR = NSCORE


# FP16's most negative finite value, packed twice into 32 bits (no single BYTE repeats into
# 0xFBFF). It is the neutral element for a max and the value the arena seeds `m` with. Module
# scope because two separate blocks in _build_head need it: the m seed and the s16 prefix fill.
NEG_INF16 = 0xFBFFFBFF

# THE SOFTMAX TEMPERATURE, as a multiplier on the raw INT32 score: P = exp(a*(S - m)) and
# corr = exp(a*(m_old - m_new)), both on the SIMD's own map at no extra task. params.hjson
# SCORE_SCALE; a real layer passes 1/(s_q*s_k*sqrt(d)). 1.0 treats the integer score as
# the logit, which on full-range INT8 operands is a one-hot softmax.
SCORE_SCALE = 1.0

# THE SCORE'S RANGE, k (params.hjson SCORE_SHIFT): the D port writes the score tile as
# RNE(S * 2^-k) (Int32ToFp16Converter built with shift: 1, k in 0..14). A full-range INT8
# score over d = 128 reaches 127^2 * 128 = 2,064,512, 31x past FP16's 65,504, and the
# converter writes +-Inf past that; with the shift the operands need not be shrunk. A power
# of two only moves the exponent, so no precision is lost, and the softmax takes the factor
# back: EXP_SCALE = a * 2^k is what it multiplies the converted scores by. m (and the
# per-shard m the gather folds) is then in the shifted domain; a*m is the same either way.
SCORE_SHIFT = 0
EXP_SCALE = 1.0

# WHERE P8 LANDS (params.hjson P8_PITCH, P8_NEST). The softmax writes P8 as a flat stream
# and can only choose the stride between its 64 B blocks. Dense (64), PV's k-major B walk
# steps round the 256 B bank rotation twice as fast as its A walk and the two streams keep
# colliding; 160 matches their rates. P8_NEST puts the second P buffer of the ping-pong
# pair P8_NEST_OFF bytes into the first one's region, its blocks in the first one's gaps.
# See "THE B PITCH" in offload_hw_kernels/gemm_fa.h.
P8_PITCH = 0
P8_NEST = False
P8_NEST_OFF = 96

# THE P HANDOFF AND THE O RESCALE (see simd.h "THE P HANDOFF" / "THE O RESCALE"). Not
# knobs: without either O is not attention. The softmax writes P already in PV's B layout
# and exports corr next to it; PV walks P k-major and, from the second KV tile on, scales
# the O it reads back by that corr on the C read path.
P_MODE = SnaxBingoKernelSimdFaSoftmaxArgs.EXACT
PV_KMAJOR = SnaxBingoKernelGemmFaPvArgs.B_KMAJOR
PV_RESCALE = SnaxBingoKernelGemmFaPvArgs.B_KMAJOR | SnaxBingoKernelGemmFaPvArgs.C_COLSCALE





# K buffers in the load->QK rotation. TWO is not a tuning choice -- it is what L1 fits.
#
# A third buffer would remove the 2-deep WAR edge that idles the iDMA on EVERY cluster before
# the last K tile, with the engine free and only the buffer missing. But it does not fit:
# named L1 grows past what the heap has once the runtime's per-task scratchpad/args arena and
# a per-cluster copy of the SoC-wide task list are counted -- neither of which appears in the
# budget line printed by main(). Allocation is name-sorted, so the arena lands last and runs
# off the end: a core then reads its own args as X and trips RegWriteKnown.
#
# To revisit: free L1 elsewhere first (NSCORE, or Bc), then a third K buffer is worth having.
NKBUF = 2

# Read VersaCore's own busy/stall counters and print them per cluster. The paper metric is
# array utilisation, and a timed trace span is not the same quantity as the array's busy
# counter -- a span also contains the START writes and the retire poll. See the perf_addr
# comment in device_kernel_args.h for the full argument.
MEASURE_ARRAY = False

# STREAM V ON THE SYSTEM iDMA (a PUSH) INSTEAD OF THE CLUSTER xDMA (a PULL). V_PUSH_TILES
# selects which V tiles ride the push; empty (the default) = all V on the cluster xDMAs.
#
# THE IDEA. The cluster pull (quadrant_wide_out) and a push into the quadrant
# (quadrant_wide_in) are separate 512-bit = 64 B/cc paths, so a system-iDMA push into a
# cluster's L1 travels a different set of wires from a cluster-initiated pull. Splitting
# K (pull) from V (push) would use two pipes concurrently instead of queueing both on one.
#
# ARITHMETIC. This kernel's intensity is fixed by the tile shape. ONE array against one wide
# port balances below it, so a single cluster is compute-bound; FOUR arrays against the SAME
# port balance above it, so the multi-cluster case is memory-bound by exactly the
# cluster-count factor. Two pipes would put the balance back at the workload's own intensity.
#
# WHY IT IS OFF. All host kernels run on one cluster's host core, so the V pushes serialise
# on ONE issuer, and each task on it pays a manager round trip in series. Per transfer the
# push is several times faster than a contended pull share, but across a whole KV axis that
# per-task dispatch cost drags its effective rate below the four concurrent cluster xDMAs'
# aggregate: ONE SERIAL ISSUER LOSES TO FOUR CONCURRENT ONES, even on a faster pipe. The
# array's busy time does not move; the array just waits longer for V. Making the push pay off
# needs the per-task dispatch overhead removed, not more tiles moved.
#
# V(0) deliberately stays on the PULL when tiles are pushed: it is the tile PV(0) needs
# first, and the pull path is the concurrent one, so it arrives soonest there.
V_PUSH_TILES = set()

# HOW MANY V BUFFERS. This is the prefetch depth, and the prefetch depth is what decides how
# much of the K/V stream lands in the FILL rather than inside the pipeline.
#
# The metric is array busy / pipeline, with pipeline = first full-size QK to last PV end, so a
# tile fetched before the first QK does not count against it. Each extra buffer moves one more
# tile per cluster out of the pipeline and into the fill.
#
# V(j) reuses v8[j % NVBUF], which PV(j - NVBUF) was the last to read, so the WAR edge walks
# back with the depth. L1 is the cap: at the shipped depth the named allocations already use
# most of the heap, and a V tile is large enough (64 KB of L1) that exactly one more buffer
# fits. Two more would overflow.
#
# Keep 2. A third buffer lowers the array's wait on V, but the end-to-end window does not
# shrink: the head grows by what the pipeline saves, because one more V tile is loaded before
# the pipeline starts. Buffering depth at this working point transfers work between head and
# pipeline rather than removing any, and when the bottleneck is aggregate port bandwidth,
# rescheduling cannot help -- only less traffic or more bandwidth can. Retry only if the head
# is shortened enough that its V loads stop being critical.
NVBUF = 2

# STAGGER THE FIRST REAL V TILE ACROSS CLUSTERS: chain the first V load from one cluster to
# the next. Leave False.
#
# Every cluster enters one long engine-config span at the same moment, each costing several
# times what the same configuration costs on a single cluster. Nothing is transferring during
# it and it retires very few instructions for its length -- that is cold instruction fetch,
# not engine backpressure. It is the FIRST REAL V TILE's config: the xDMA's memset path is
# warmed by the arena fills, its 1d_copy path is not (see the WarmGemm comment -- a warm-up
# COPY hangs, undiagnosed), so four cold fetches of the same code land on the fabric together.
#
# Chaining the first V load from one cluster to the next serialises those cold fetches, but
# it does not make any of them cheaper: their total cost is invariant to when they run, which
# points at contention with the K/V transfers that stream throughout (the config code is
# fetched from L3 while the fabric is busy carrying tiles). The fix is to warm the
# xdma_1d_copy path or make its config resident; the undiagnosed warm-up-copy hang is what to
# diagnose.
STAGGER_FIRST_V = False

# PUSH V IN PAIRS ON THE SYSTEM iDMA, one BINGO task per PAIR. Leave False.
#
# A host iDMA transfer rides quadrant_wide_in, a pipe disjoint from the clusters' pull, but
# each task on the host costs a manager round trip, so ONE tile per task leaves the push
# losing on effective rate. Batching more tiles into a task raises it. V(j) also waits on
# PV(j-NVBUF), so the V chain serialises THROUGH THE COMPUTE, not just through the issuer;
# pairing attacks that too -- {V2,V3} issue together after PV(1) instead of V(3) waiting
# behind PV(2).
#
# PAIRS, NOT QUADS, DELIBERATELY. V(j) and V(j+1) live in DIFFERENT buffers (NVBUF=2), so a
# pair is expressible inside one cluster. Batching four would have to cross clusters, which
# couples their dependencies -- every cluster's batch would wait on the slowest -- and that is
# exactly the coupling that makes a wider split imbalanced across clusters. Giving up some
# effective rate to keep the shards independent is the right trade.
#
# Batching beats one tile per task, but the push tasks are NOT issue-bound: the issuer sits
# idle for more than half the window, waiting on the WAR edge (V(j) cannot start until
# PV(j-NVBUF) has freed its buffer), and deeper buffering is capped by L1. Four concurrent
# cluster xDMAs beat one host issuer, so V stays on the pull.
V_PUSH_PAIRS = False

# PUSH K TO ALL FOUR CLUSTERS ON THE SYSTEM iDMA -- the second wide pipe, used for the one
# operand that is identical on every cluster. Leave False.
#
# WHY K AND NOT V. V is the operand whose WAR edge is tightest: V(j) cannot start until
# PV(j-NVBUF) has freed its buffer, so a single host issuer idles waiting on compute. K under
# head-parallelism is a different shape. The four clusters want the SAME BYTES, so one task
# with four destinations delivers a whole tile-step (four is the batch size the struct caps
# at), and because the push writes every cluster's k8 directly, the cross-cluster RAW edge
# that the pull needs (three clusters waiting on the owner's copy) disappears: K stops being a
# two-hop operand.
#
# WHAT IT COSTS. The same coupling the broadcast had: one producer feeding four consumers must
# wait for every cluster's WAR edge, so the slowest cluster gates the refill. The existing
# broadcast WAR machinery states those edges, and this reuses it unchanged. The delivery moves
# off the cluster DMA engines and onto the critical path, where QK and softmax wait longer.
#
# FABRIC USE. K rides quadrant_wide_in (a PUSH from the system iDMA) while V rides
# quadrant_wide_out (the clusters' own PULL): disjoint 512-bit pipes.
K_PUSH_ALL = False

# PUT ONLY THE MAIN-MEMORY FETCH ON THE SYSTEM iDMA, and leave the cross-cluster pulls where
# they are. Leave False.
#
# K_PUSH_ALL delivers a tile to all four clusters from one host task, which is one producer
# feeding four consumers: the task waits for every cluster's WAR edge and every cluster then
# waits for it. These knobs make the SAME move without the coupling. Under rotation exactly
# one cluster reads main memory for each tile; that one transfer becomes a host push with a
# SINGLE destination, so it takes only the owner's own WAR edge, exactly as the owner's own
# load did. The other three clusters keep pulling from the owner's L1 on their own engines.
#
# That splits the two wide pipes cleanly: every byte that touches main memory rides
# quadrant_wide_in (the system iDMA's PUSH) and every byte that does not rides
# quadrant_wide_out (the clusters' PULL).
#
# Only meaningful with the matching *_PULL_FROM_CL0 knob on -- they replace the owner's load
# inside the pull scheme.
#
# It stays off: the owner's fetch is the HEAD of every tile's chain, so putting it behind the
# host's per-task dispatch delays all three pullers as well as the owner, and only one
# transfer in four leaves the cluster engines.
#
# K_PUSH_OWNER WITHOUT V_PUSH_OWNER HANGS. With K pushed and V left on the cluster iDMA, a
# puller's xDMA spins forever on its completion CSR while every other hart is frozen; both on
# or both off is fine. Not diagnosed -- do not enable one alone.
#
# The rule behind every host-push knob: four concurrent cluster engines beat one host issuer,
# and the reason is never bandwidth. The system iDMA is the faster pipe per transfer; what it
# cannot do is sit on a chain that four clusters wait on, behind a per-task dispatch cost.
# Work would have to be prefetched far enough ahead to hide that, and NKBUF/NVBUF are capped
# by L1.
K_PUSH_OWNER = False
V_PUSH_OWNER = False


def _load_params(param):
    """Derive the whole geometry from params.hjson and the array, in ONE place.

    Everything below -- the tile, the second matmul's shape, the operand bound, the
    golden -- is a function of (M, K, N) and the mesh. Deriving it here is what stops the
    kernel's idea of the tile and the descriptors' idea of it from drifting apart.
    """
    global M, K, N, NKV, NQ, BC, BR, DHEAD, S2_M, S2_K, S2_N, QSHIFT, NKV_PER, NCL
    global DECOMP, SCORE_SCALE, SCORE_SHIFT, EXP_SCALE, P8_PITCH, P8_NEST, MEASURE_ARRAY
    MEASURE_ARRAY = bool(param.get("MEASURE_ARRAY", MEASURE_ARRAY))
    DECOMP = str(param.get("DECOMP", DECOMP))
    if DECOMP not in ("headpar", "kvsplit"):
        raise ValueError(f"DECOMP={DECOMP!r} must be 'headpar' or 'kvsplit'")
    M, K, N = int(param["M"]), int(param["K"]), int(param["N"])
    NKV = int(param["NKV"])
    # Extra right-shift on the INT8 operands, BEYOND the minimum that keeps a score inside
    # FP16. It exists for the cross-cluster fold, not for the arithmetic.
    #
    # At the minimum shift the scores span thousands, so exp(S - m) underflows for every
    # key but the argmax and every shard's l is exactly 1.0. The softmax-statistics junction
    # folds a key (m, a max) and an exp-weighted value (l); with l identically 1 the value
    # beat is folding 1 + 0 + 0 + 0 on every lane, so a junction that dropped the value
    # entirely would still produce the right answer. Shrinking the operands
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
    SCORE_SHIFT = int(param.get("SCORE_SHIFT", 0))
    if not 0 <= SCORE_SHIFT <= SnaxBingoKernelGemmFaQkArgs.DSHIFT_MAX:
        raise ValueError(f"SCORE_SHIFT={SCORE_SHIFT}: the D-port converter scales by 2^-k "
                         f"for k in 0..{SnaxBingoKernelGemmFaQkArgs.DSHIFT_MAX}")
    QSHIFT = qshift(DHEAD, SCORE_SHIFT) + int(param.get("SCORE_SHIFT_EXTRA", 0))
    SCORE_SCALE = float(param.get("SCORE_SCALE", 1.0))
    if not (np.isfinite(SCORE_SCALE) and SCORE_SCALE > 0):
        raise ValueError(f"SCORE_SCALE={SCORE_SCALE} must be positive and finite")
    EXP_SCALE = float(np.float32(SCORE_SCALE) * np.float32(2.0 ** SCORE_SHIFT))
    P8_PITCH = int(param.get("P8_PITCH", 0))
    P8_NEST = bool(param.get("P8_NEST", 0))
    pitch = SnaxBingoKernelSimdFaSoftmaxArgs.pitch(P8_PITCH)
    if P8_NEST and pitch - 64 < P8_NEST_OFF:
        raise ValueError(f"P8_NEST needs P8_PITCH >= {P8_NEST_OFF + 64} so the other "
                         f"buffer's blocks fit in the gaps, got {pitch}")
    # Bc IS the beat count of the score tile -- the transposed layout puts one key, all
    # Br queries, in each 64-B beat -- and the quantiser writes P in PV's B layout by
    # interleaving FOUR key beats per 32-bit atom, so it must be a multiple of 4.
    if BC % 4:
        raise ValueError(f"Bc={BC} must be a multiple of 4: the quantiser interleaves four "
                         f"key beats into PV's B layout")
    if NKV % NCL:
        raise ValueError(
            f"NKV={NKV} must divide across {NCL} clusters: each cluster owns a disjoint "
            f"run of KV tiles and the merge assumes every shard covers the same count.")
    NKV_PER = NKV // NCL
    if BR > 32 and BR % 32:
        raise ValueError(
            f"Br={BR} must be at most 32 or a multiple of 32: the softmax-statistics "
            f"junction folds 32 query rows a group, with one count of live rows for every "
            f"group of a gather.")


def qshift(d, k=0):
    """How far to bound the INT8 operands so a score cannot leave FP16.

    A score is a sum of d products of two shifted INT8s, so |S| <= (128>>q)^2 * d, and
    the D port writes S * 2^-k. Int32ToFp16 writes Inf past 65504 and exp(inf - inf) is
    NaN, so an overflowing row fails the softmax rather than degrading. With the
    converter's shift (k >= 6 at d = 128) this is 0: full-range operands.
    """
    for q in range(8):
        if (128 >> q) ** 2 * d <= 65504 * 2 ** k:
            return q
    raise ValueError(f"no INT8 shift keeps d={d} scores inside FP16")



def _at(handle, nbytes):
    """The same buffer, `nbytes` further in -- whatever kind of handle it is.

    A staged array arrives as a SYMBOL on the host path and as a FIXED ADDRESS on the
    memory-chiplet path, and an L1 buffer as an allocation; all of them have to offset the
    same way, or the per-tile slicing works on one platform and silently reads tile 0 on
    the other.
    """
    if not nbytes:
        return handle
    if isinstance(handle, BingoMemAlloc):
        return handle.view(nbytes)
    if isinstance(handle, BingoMemAllocView):
        return BingoMemAllocView(handle.base, handle.offset + nbytes)
    if isinstance(handle, BingoMemSymbol):
        return BingoMemSymbol(handle.symbol_name, handle.offset + nbytes)
    if isinstance(handle, BingoMemFixedAddr):
        return BingoMemFixedAddr(handle.address + nbytes)
    raise TypeError(f"cannot offset a {type(handle).__name__}")


def _f16(x):
    """Round ONCE to FP16 from whatever wider value it was computed in (numpy rounds a
    float64 straight to half with round-to-nearest-even, so there is no double rounding)."""
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(x, dtype=np.float64).astype(np.float16)


def _colscale(o, corr):
    """Int32ColumnScale: y = rne(x * corr[col]), saturated to INT32. Exact in float64: an
    INT32 times an 11-bit FP16 significand needs 42 bits, and np.rint rounds half to even."""
    y = np.rint(np.asarray(o, dtype=np.float64) *
                np.asarray(corr, dtype=np.float16).astype(np.float64)[None, :])
    return np.clip(y, -2**31, 2**31 - 1).astype(np.int64)


def d32_order(o_t):
    """Canonical O^T [d, Br] -> the INT32 words in the order the D32 port writes them.

    The C/D spatial map is chosen so the FP16 score tile lands row-major -- one key per
    64 B beat, which is what the LANEWISE reduce needs. C and D share those strides and the
    INT32 side is twice as wide, so at INT32 the SAME strides do NOT come out row-major: O
    lands PERMUTED. That is deliberate and harmless to the arithmetic (C and D permute
    identically, so O += P.V still accumulates against itself -- the cluster cfg says
    exactly this), but it means a golden in canonical block order is not comparable with
    what the host reads back.

    The map is DERIVED from the descriptors, not assumed: the array serialises a
    meshRow x meshCol block into chunks of serial_c_d_width, each chunk into channels of
    bankWidth, and the AGU places channel i at sl0*(i%4) + sl1*((i/4)%4). Ported from the
    reference's data/datagen.py, which is the model that makes the cluster app's own
    exact-equality O check pass.
    """
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
    # reproduce the plain [key][query] row-major layout the score tile is known to have.
    # If the serialisation model were wrong this assert fires instead of the golden
    # silently disagreeing with the hardware.
    for (mm, nn, r, c), byte in scatter(16, 1, S2_N * 16 * MESH_ROW * MESH_COL // 8, 2).items():
        want = ((mm * MESH_ROW + r) * (S2_N * MESH_COL) + nn * MESH_COL + c) * 2
        assert byte == want, "D32 address model disagrees with the FP16 row-major layout"

    ts2 = S2_N * OUT_W * MESH_ROW * MESH_COL // 8
    o_t = np.asarray(o_t, dtype=np.int64).reshape(DHEAD, BR)
    o_mem = np.zeros(o_t.size, dtype=np.int64)
    seen = np.zeros(o_t.size, dtype=bool)
    for (mm, nn, r, c), byte in scatter(OUT_W, S2_M, ts2, 4).items():
        w = byte // 4
        assert byte % 4 == 0 and not seen[w], "D32 INT32 address map is not a bijection"
        seen[w] = True
        o_mem[w] = o_t[mm * MESH_ROW + r, nn * MESH_COL + c]
    assert seen.all(), "D32 INT32 address map does not cover the output"
    return o_mem.astype(np.int32)


def shard_golden(q8, k8_tiles, v8_tiles):
    """One query tile's (m, l, O) over a run of KV tiles, as the hardware computes it.

    q8        A-layout bytes of Q [Br, d]  (== B layout of Q^T, the score matmul's B)
    k8_tiles  per KV tile, A-layout bytes of K_t [Bc, d]
    v8_tiles  per KV tile, A-layout bytes of V_t^T [d, Bc] (PV's A operand)

    Follows the device's own sequence at the precision each stage works in:

        S       exact INT32 out of the mesh, converted to FP16 on the D32 port as
                RNE(S * 2^-k); every step below works on that, with a' = a * 2^k
        m_new   max(m_old, max over this tile's KEYS), per query -- exact
        corr    exp(a * (m_old - m_new)): the difference in FP16, the exponential in FP32
        P       exp(a * (S - m_new)): EW0 subtracts in FP16, the map exponentiates in FP32
        rowsum  over KEYS in the FP32 accumulator, narrowed once to FP16
        P8      sat(rne(P * 127)), written in PV's B layout by the INTERLEAVE quantiser
        l_new   corr * l_old + rowsum, each step rounded to FP16
        O       colscale(O, corr) + V^T . P8^T in INT32 -- the C-path scaler, then the MAC

    m is seeded at -65504, l at 0, and PV's first tile reads no C, so tile 0 needs no
    special case: corr = exp(-inf) = 0 and 0 * l = 0.

    Mirrored from libs/blocks/flash_attention.py:shard_golden, where it is checked against
    an independent row-major model (unpacked matrices, plain matmuls).
    """
    a32 = np.float32(EXP_SCALE)
    m = np.full(BR, -65504.0, dtype=np.float16)
    l = np.zeros(BR, dtype=np.float16)
    o = np.zeros((DHEAD, BR), dtype=np.int64)
    for k8, v8 in zip(k8_tiles, v8_tiles):
        d32 = block_gemm_golden_model(
            M, K, N, MESH_ROW, TILE_SIZE, MESH_COL, np.asarray(k8), np.asarray(q8), 0, 0,
            np.zeros(M * N * MESH_ROW * MESH_COL, dtype=np.int64))
        # Block order [M][N][meshRow][meshCol] -> [key][query]: how the D port lays the
        # tile down (its [4, 4] channel grouping interleaves the two N blocks in one key row).
        s = np.asarray(d32).reshape(M, N, MESH_ROW, MESH_COL).transpose(0, 2, 1, 3)
        # Exact: a power of two only moves the exponent, so float64 holds S * 2^-k and
        # numpy's one rounding to half is the converter's RNE (Inf past 65504, as the RTL).
        s16 = _f16(s.reshape(BC, BR).astype(np.float64) * 2.0 ** -SCORE_SHIFT)
        m_new = np.maximum(m, s16.max(axis=0))
        with np.errstate(over="ignore", invalid="ignore"):
            delta = _f16(m.astype(np.float64) - m_new.astype(np.float64))
            corr = _f16(np.exp(a32 * delta.astype(np.float32)))
            d16 = _f16(s16.astype(np.float64) - m_new.astype(np.float64)[None, :])
            p16 = _f16(np.exp(a32 * d16.astype(np.float32)))
        rowsum = _f16(p16.astype(np.float32).sum(axis=0, dtype=np.float32))
        p8 = np.clip(np.rint(p16.astype(np.float32) * np.float32(127.0)),
                     -127, 127).astype(np.int8)                          # [key][query]
        l = _f16(_f16(corr.astype(np.float32) * l.astype(np.float32)).astype(np.float32)
                 + rowsum.astype(np.float32))
        # B = P^T [Bc, Br] in B layout: block (n, k) holds P[key = 4k + s][query = 16n + c].
        p_b = p8.reshape(BC // TILE_SIZE, TILE_SIZE, BR // MESH_COL, MESH_COL) \
                .transpose(2, 0, 3, 1).reshape(-1)
        pv = np.asarray(block_gemm_golden_model(
            S2_M, S2_K, S2_N, MESH_ROW, TILE_SIZE, MESH_COL, np.asarray(v8), p_b, 0, 0,
            np.zeros(S2_M * S2_N * MESH_ROW * MESH_COL, dtype=np.int64)), dtype=np.int64)
        pv = pv.reshape(S2_M, S2_N, MESH_ROW, MESH_COL).transpose(0, 2, 1, 3).reshape(DHEAD, BR)
        o = _colscale(o, corr) + pv
        m = m_new
    return m, l, o


def build_shards():
    """Random Q, K and V for the whole attention, and every shard's golden (m, l, O).

    EVERY KV TILE IS DIFFERENT. Feeding every tile the SAME K and V would stop the running
    maximum moving after tile 0, make every correction factor exactly 1, and reduce the O
    golden to "one tile times NKV_PER" -- data that hides a PV that reads P in the wrong
    layout (a golden built the same wrong way agrees with it) and an O that is never
    rescaled across tiles (with corr = 1 there is nothing to rescale). Distinct tiles make
    the maximum move, so corr < 1 on a real fraction of the queries and O can only match if
    the recurrence is actually carried out.

    Every cluster still gets its own (m_c, l_c): under kvsplit its own KV shard, under
    headpar its own query head over the group's shared KV head (the GQA relation). Distinct
    partials are what make the gather's result evidence that the gather happened.

    Layouts, one tile after another, each tile contiguous:
        q   A(Q)       per query tile [Br, d]  -- heads stacked under headpar
        k   A(K_t)     per KV tile [Bc, d]     -- shards stacked under kvsplit
        v   A(V_t^T)   per KV tile [d, Bc]     -- PV's A operand, split like k
    """
    rng = np.random.RandomState(1234)
    T = NKV_PER
    n_heads = NCL if DECOMP == "headpar" else 1
    n_kv = T * (NCL if DECOMP == "kvsplit" else 1)
    qb, kb = BR * DHEAD, BC * DHEAD

    def rnd(n):
        return rng.randint(-128, 128, size=n).astype(np.int8) >> QSHIFT

    q8, k8, v8 = rnd(n_heads * NQ * qb), rnd(n_kv * kb), rnd(n_kv * kb)
    m_c, l_c, o_c = [], [], []
    for c in range(NCL):
        head = c if DECOMP == "headpar" else 0
        t0 = c * T if DECOMP == "kvsplit" else 0
        # The CHECKED query tile is the last one; query tiles never interact.
        qt = head * NQ + NQ - 1
        m, l, o = shard_golden(q8[qt * qb:(qt + 1) * qb],
                               [k8[t * kb:(t + 1) * kb] for t in range(t0, t0 + T)],
                               [v8[t * kb:(t + 1) * kb] for t in range(t0, t0 + T)])
        m_c.append(m)
        l_c.append(l)
        o_c.append(d32_order(o))
    m_c, l_c = np.stack(m_c), np.stack(l_c)

    merged = None
    if DECOMP == "kvsplit":
        # The online-softmax combine the softmax-statistics junction implements, on the raw
        # running max and running sum the pack copies, computed in FP32 on FP16 inputs:
        #     m* = max_c m_c                l* = sum_c exp(a'*(m_c - m*)) * l_c
        # with a' = EXP_SCALE, the factor the softmax's exp applies and the junction's CSR(1).
        # m is the max of the converted scores, S * 2^-k, and m* comes back in those units.
        # Laid out in the junction's groups: per 32 query rows a beat of 32 FP16 keys then a
        # beat of 32 FP16 values, rows past Br the identity row (key 0xFBFF, value 0), as
        # FP16 bit patterns.
        ms = np.asarray(m_c, dtype=np.float16).astype(np.float32)
        ls = np.asarray(l_c, dtype=np.float16).astype(np.float32)
        m_star = ms.max(axis=0)
        l_star = (np.exp(np.float32(EXP_SCALE) * (ms - m_star)) * ls).sum(axis=0)
        groups, _ = xdma_softmax_stats_groups(BR)
        merged = np.zeros((groups, 2, XDMA_SSJ_ROWS), dtype=np.uint16)
        merged[:, 0, :] = XDMA_SSJ_IDENTITY
        for g in range(groups):
            r0, r1 = g * XDMA_SSJ_ROWS, min(BR, (g + 1) * XDMA_SSJ_ROWS)
            merged[g, 0, :r1 - r0] = m_star[r0:r1].astype(np.float16).view(np.uint16)
            merged[g, 1, :r1 - r0] = l_star[r0:r1].astype(np.float16).view(np.uint16)
        merged = merged.reshape(-1)
    return {"q8": q8, "k8": k8, "v8": v8, "m": m_c, "l": l_c, "o": o_c, "merged": merged}


def stage(st, data):
    """Hand every array to the staging helper and return the handles.

    WHERE these land is the platform's business, not this workload's: a config with a
    memory chiplet gets a mempool.bin, one without gets C arrays in the host image. See
    mini_compiler/mem/bingo_data_staging.py -- addressing a memory chiplet that a config
    does not have reads unmapped memory rather than faulting, which surfaces as an
    arithmetic bug a long way from the cause.

    ONE ARRAY PER TENSOR, sliced per cluster by offset: a shared operand is the SAME handle
    for every cluster (that is what the broadcast and pull paths key on), a per-cluster one
    is that cluster's run of tiles.
    """
    T, qb, kb = NKV_PER, BR * DHEAD, BC * DHEAD
    q = st.put("fa_q8", "int8_t", data["q8"].astype(np.int8))
    k = st.put("fa_k8", "int8_t", data["k8"].astype(np.int8))
    v = st.put("fa_v8", "int8_t", data["v8"].astype(np.int8))
    headpar = DECOMP == "headpar"
    return {
        "a": [k if headpar else _at(k, c * T * kb) for c in range(NCL)],
        "b": [_at(q, c * NQ * qb) if headpar else q for c in range(NCL)],
        "v": [v if headpar else _at(v, c * T * kb) for c in range(NCL)],
        # Per-shard goldens, one per cluster. Keeping them is what separates "the fold is
        # wrong" from "a shard is wrong": a cluster whose recurrence quietly did nothing
        # still produces a well-formed partial, and the merged result alone cannot say
        # which of the four it came from.
        "m": [st.put(f"fa_m_golden_c{c}", "uint16_t",
                     np.asarray(data["m"][c]).astype(np.float16).view(np.uint16))
              for c in range(NCL)],
        "l": [st.put(f"fa_l_golden_c{c}", "uint16_t",
                     np.asarray(data["l"][c]).astype(np.float16).view(np.uint16))
              for c in range(NCL)],
        "o": [st.put(f"fa_o_golden_c{c}", "int32_t", np.asarray(data["o"][c]).astype(np.int32))
              for c in range(NCL)],
    }


def stage_merged(st, merged):
    """The gathered (m*, l*), in the junction's own groups, as FP16 bit patterns."""
    return st.put("fa_ml_merged_golden", "uint16_t", np.asarray(merged, dtype=np.uint16))


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
                         "the softmax-statistics junction to fold the shards in the fabric")
    names = list(jcts.keys())
    if name not in names:
        raise ValueError(f"cluster cfg has writer junctions {names} but this workload "
                         f"needs {name}")
    return names.index(name)


def _alloc_cluster(dfg, c):
    """Every L1 buffer one cluster's pipeline owns, and nothing else -- no nodes.

    Split out of _build_cluster because a BROADCAST load has to name its destinations in
    all four clusters, so every cluster's buffers must exist before the first load node is
    created. This moves no address: the compiler sorts handles by NAME when it lays out a
    heap (bingo_dfg._collect_memory_handles), so creation order across clusters -- and
    within one -- does not decide the layout. The ordering comment below is about which
    NAME sorts first, and that is unchanged.
    """
    g = G(dfg, c)
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
    # main memory while the previous tile computes. Every tile is a different slice of the
    # operand and is RE-FETCHED per tile, so the memory traffic is what a real KV stream
    # would cost. Loading once and replaying leaves the iDMA idle and makes any utilisation
    # figure exclude memory.
    # TWO K buffers. A third would let the loads run one tile further ahead (every QK is
    # gated by its own K load), but it also raises the number of simultaneously-ready DMA
    # nodes past what the manager's ready set tolerates.
    k8 = [g.l1(f"fa_k8_{i}", M * K * MESH_ROW * TILE_SIZE) for i in range(NKBUF)]
    q8 = [g.l1(f"fa_q8_{q}", N * K * MESH_COL * TILE_SIZE)      # B of the score matmul
          for q in range(NQ)]
    v8 = [g.l1(f"fa_v8_{i}", S2_M * S2_K * MESH_ROW * TILE_SIZE) for i in range(NVBUF)]
    # There is no C buffer: every C channel of the score matmul is masked (see gemm_fa.h),
    # so the AGU walks addresses that are never dereferenced, and the kernel substitutes
    # D_addr when C_addr is 0 (the same path PV's first tile takes).
    cz = 0
    # TWO score buffers. A third would let QK run three deep and remove the WAR edge it
    # targets, but the extra QK streams A, B and D through TCDM alongside the softmax, and
    # the softmax is the critical path. Slack bought for the GEMM is paid for in SIMD
    # bandwidth.
    # ONE BEAT OF HEADROOM in front of each score buffer. The softmax's fused exp pass reads
    # [-m_new][the tile] as a single contiguous stream. Reserving the beat here lets that
    # pass read the GEMM's own output buffer directly, instead of copying the whole tile
    # into the arena beside a -m_new slot -- a copy comparable in size to the K/V stream
    # itself. The GEMM writes its D at +64; the beat below it is the prefix.
    s16 = [g.l1(f"fa_s16_{i}", 64 + BC * BR * 2) for i in range(NSCORE)]  # score tile, fp16
    # ONE TRAILING BEAT past the quantised P. The fused exp pass emits bc/2 INT8 beats and
    # then the tapped row sum, unnarrowed, as ONE contiguous stream -- and no shape can span
    # two allocations, so the row sum lives here rather than in the arena.
    # + the row-sum beat + this tile's corr, which PV loads into its C-path scaler: living in
    # the SAME buffer as P, it is protected by exactly the edges that protect P.
    # At P8_PITCH the beats (and the row sum and corr) sit one pitch apart; with P8_NEST
    # consecutive buffers pair up in one region, the second at +P8_NEST_OFF in the first
    # one's gaps. Each buffer is still its own handle to every node that touches it.
    p8_len = SnaxBingoKernelSimdFaSoftmaxArgs.p8_bytes(BC, BR, P_MODE, P8_PITCH)
    if P8_NEST:
        p8 = []
        for i in range(0, NSCORE, 2):
            if i + 1 < NSCORE:
                pair = g.l1(f"fa_p8_{i}{i + 1}", P8_NEST_OFF + p8_len)
                p8 += [pair.view(0), pair.view(P8_NEST_OFF)]
            else:
                p8.append(g.l1(f"fa_p8_{i}", p8_len))
    else:
        p8 = [g.l1(f"fa_p8_{i}", p8_len) for i in range(NSCORE)]
    # One arena, one Q buffer and one O accumulator PER QUERY TILE -- that is the
    # entire cost of the reuse. K, V, the score tile and P stay single sets: they are
    # exactly what we are trying not to re-read.
    oacc = [g.l1(f"fa_oacc32_{q}", BR * DHEAD * 4) for q in range(NQ)]
    # Destination of the BCAST_WARMUP multicast. Its own allocation rather than a corner of
    # fa_warm_buf: that one is the SIMD warm-up's, and a broadcast landing in it while the
    # SIMD warm-up reads it would be a data race for no reason. 64 B x 4 clusters.
    bwarm = g.l1("fa_bcast_warm", 64)
    return dict(arena=arena, k8=k8, q8=q8, v8=v8, cz=cz, s16=s16, p8=p8, oacc=oacc,
                bwarm=bwarm)


def _shard_handles(h_all, c):
    """The staged handles this cluster reads, with the per-cluster lists resolved.

    Which of these are genuinely per-cluster depends on DECOMP: kvsplit gives every cluster
    its own K, headpar its own Q. stage() already collapsed whatever is shared onto one
    handle, so indexing is uniform here either way.
    """
    h = dict(h_all)
    for key in ("a", "b", "v", "m", "l", "o"):
        h[key] = h_all[key][c]
    return h


def _build_head(dfg, c, h_all, buf):
    """Everything a cluster does BEFORE its first KV tile: warm-ups, Q, the arena fills.

    Split from the body, and called at the start of each cluster's own pass, so that a
    broadcast placed on cluster c's xDMA is created AFTER c's own fills: a core dispatches in
    creation order, so it sits BEHIND them rather than in front of them. In front, a
    broadcast would wait on a PV that waits on a softmax that waits on the very fills queued
    behind it -- a genuine cycle, not a slowdown.
    """
    g = G(dfg, c)
    h = _shard_handles(h_all, c)
    arena, k8, q8, v8 = buf["arena"], buf["k8"], buf["q8"], buf["v8"]
    cz, s16, p8, oacc = buf["cz"], buf["s16"], buf["p8"], buf["oacc"]
    # `off` is the tile's byte offset inside this cluster's operand: every KV tile and every
    # query tile is a different slice, so the tile index has to reach the source address.
    def load(tag, key, dst, nbytes, after=None, off=0):
        return g.node(f"Load_{tag}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                      SnaxBingoKernelIdma1dCopyArgs(_at(h[key], off), dst, nbytes), after)

    def xload(tag, key, dst, nbytes, after=None, off=0):
        """Same transfer on the xDMA hart instead of the DM core's iDMA."""
        return g.node(f"Load_{tag}", XDMA_CORE, "__snax_bingo_kernel_xdma_1d_copy",
                      SnaxBingoKernelXdma1dCopyArgs(_at(h[key], off), dst, nbytes), after)


    # The loads are chained: one iDMA engine, and serialising them here keeps the graph's
    # ready set small rather than handing the manager four nodes that must queue anyway.
    # ORDER MATTERS, and it is not the order the buffers are declared in. There is one
    # iDMA engine so these serialise regardless; what the chain decides is WHICH of them
    # QK(0) has to wait behind. QK reads K and Q. V and the O zero are
    # PV's, and PV cannot run until SM(0) is done anyway -- so they belong AFTER the
    # loads QK needs, where they stream underneath QK(0) and SM(0) instead of delaying
    # the first matmul by their own duration.
    # ---- warm every core's config path BEFORE the first byte moves ---------------------
    # A core's FIRST dispatch of a given kernel costs far more than its next, and the
    # difference is instruction-cache line refills. A refill is far cheaper on an idle
    # fabric than while the iDMA is streaming a tile through the same path, so what the
    # first dispatch costs is decided by WHEN it runs, not by what it does. Left to itself
    # every core takes its first dispatch with the load chain already saturating the
    # fabric, and all three pay the contended price at once.
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
    # allocation per cluster. Both are small against a dispatch, but the point of the
    # measurement IS the dispatch cost, so it should be possible to take the instrument out
    # and confirm the number did not move.
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
    warm_a_z = g.node("WarmZeroA", XDMA_CORE, "__snax_bingo_kernel_xdma_memset",
                      SnaxBingoKernelXdmaMemsetArgs(
                          warm_a, MESH_ROW * TILE_SIZE,
                          SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), warm_z)
    # warm_b needs the same fill as warm_a: WarmGemm below takes it as BOTH its B and its C
    # operand, and TCDM that nothing has written reads back X. A node, not a knob -- feeding
    # the array X is never correct, and the pair is what the name WarmZeroAB always implied.
    warm_ab = g.node("WarmZeroB", XDMA_CORE, "__snax_bingo_kernel_xdma_memset",
                     SnaxBingoKernelXdmaMemsetArgs(
                         warm_b, MESH_COL * TILE_SIZE,
                         SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), warm_a_z)

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
    # Only the CONFIG paths are warmed, not the SIMD task-firing path: warming that too --
    # a full tiny softmax here -- would re-time the pipeline so that later softmax runs land
    # under the K/V stream instead.
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
    # The prologue does not run the FULL kernel (firing the tasks as well): its own run and
    # drain would then sit on the critical path ahead of the first real tile on the same
    # in-order core.
    # One per query tile: the descriptors hold absolute pointers into their own arena.
    warm = [g.node(f"Geom{q}", SIMD_CORE, "__snax_bingo_kernel_simd_fa_softmax",
                   SnaxBingoKernelSimdFaSoftmaxArgs(
                       s16[0].view(64), p8[0], arena[q],
                       bc=BC, dhead=DHEAD, tile_idx=0, seed_state=0,
                       geom_mode=SnaxBingoKernelSimdFaSoftmaxArgs.GEOM_PROLOGUE,
                       score_scale=EXP_SCALE, p_mode=P_MODE, p8_pitch=P8_PITCH))
            for q in range(NQ)]

    # Q is the query tile: fixed for the whole run, loaded once. Query tile q is its own
    # slice of this cluster's Q -- NQ tiles of one head, not NQ copies of one tile.
    QBYTES = N * K * MESH_COL * TILE_SIZE
    ld_q = [load(f"Q{q}", "b", q8[q], QBYTES, [warm_gemm], off=q * QBYTES)
            for q in range(NQ)]
    # No Czero load: with the C channels masked the score matmul never reads this buffer,
    # so no zeros are loaded ahead of QK(0).
    # O starts at zero: the O matmul sets take_in_new_c, so every output block starts from
    # C, and C is oacc itself. A prefix of the same zero region does it.

    # The softmax's OWN running O lives in the arena and is a different buffer from the
    # INT32 oacc32 above. Zeroing it with the SIMD core's stores would be dhead beats on the
    # serial path, so the xDMA core zeroes it in place (below). The device side then only
    # has to seed mrun/lrun, which is two beats.
    _lay = SnaxBingoKernelSimdFaSoftmaxArgs.layout(BC, DHEAD)
    # ZERO THE O ACCUMULATOR ON THE xDMA, not the iDMA.
    #
    # This is a CONSTANT, so fetching it from main memory is wasted traffic on the one
    # engine that is on the critical head: at NQ query tiles it is NQ separate transfers
    # standing in front of K(0), which is the largest single gap in the fill. The xDMA core
    # is otherwise idle -- it owns nothing but its exit node -- so it generates the zeros in
    # place, with the reader channels disabled so no TCDM read is issued either.
    # THE WHOLE RECURRENCE STATE IS GENERATED, NOT LOADED.
    #
    # m = -inf, l = 0 and O = 0 are constants. Seeded by the SIMD kernel or carried by the
    # iDMA they would sit on the critical head: at NQ query tiles that is NQ transfers
    # standing in front of K(0) plus NQ scalar fills in front of the first softmax. None of
    # it needs an operand, so the idle xDMA core generates all three in place -- reader
    # channels disabled, so not even a TCDM read.
    #
    # -65504 is FP16's most negative finite value, which is what the kernel used; a true
    # -inf would make the first exp(S - m) NaN rather than 0. The pattern is 32 bits
    # because no single BYTE repeats into 0xFBFF.

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
        # m and l are one beat each and NOT adjacent (mrun, corrL, lrun), so they are two
        # fills rather than one. Chained after the O fill: they share the one xDMA engine,
        # and stating the order costs nothing at run time while leaving the dep-tag
        # allocator one chain instead of three concurrent ones.
        mfill = g.node(f"SeedM{q}", XDMA_CORE, "__snax_bingo_kernel_xdma_memset",
                       SnaxBingoKernelXdmaMemsetArgs(
                           arena[q].view(_lay["mrun"]), 64, NEG_INF16), z)
        lfill = g.node(f"SeedL{q}", XDMA_CORE, "__snax_bingo_kernel_xdma_memset",
                       SnaxBingoKernelXdmaMemsetArgs(
                           arena[q].view(_lay["lrun"]), 64,
                           SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), mfill)
        ld_az.append(lfill)

    return dict(warm_gemm=warm_gemm, warm=warm, ld_q=ld_q, ld_az=ld_az,
                perf=perf, load=load, xload=xload)


def _build_cluster(dfg, c, h_all, m_all, l_all, buf, bcast=None, all_bufs=None):
    """The KV-tile pipeline and this cluster's own checks, on top of _build_head.

    The one-cluster graph -- same double buffering, same load-chain order -- with three
    differences: it runs NKV_PER tiles instead of NKV, it reads this cluster's operands, and
    it ends at the partial rather than at a check.

    `bcast` is None under kvsplit, where this cluster loads its own K and V. Under headpar
    it is {"k": [...], "v": [...]}: the MULTICAST nodes, already created by build(), that
    fill every cluster's k8/v8 from one read. This cluster then only depends on them.
    """
    g = G(dfg, c)
    h = _shard_handles(h_all, c)
    m, l_gold = m_all[c], l_all[c]
    arena, k8, q8, v8 = buf["arena"], buf["k8"], buf["q8"], buf["v8"]
    cz, s16, p8, oacc = buf["cz"], buf["s16"], buf["p8"], buf["oacc"]

    # HEAD FIRST, IN THIS CLUSTER'S OWN PASS -- do not hoist it into build().
    #
    # Building every cluster's head before any body changes the GLOBAL node creation order,
    # and that order IS the task-descriptor list the BINGO manager streams, even with the
    # same per-core kernel sequences in every cell and the same allocations. Dispatch is
    # order-sensitive; the list order is not cosmetic.
    head = _build_head(dfg, c, h_all, buf)
    warm_gemm, warm = head["warm_gemm"], head["warm"]
    ld_q, ld_az, perf = head["ld_q"], head["ld_az"], head["perf"]
    load, xload = head["load"], head["xload"]

    KBYTES = M * K * MESH_ROW * TILE_SIZE
    VBYTES = S2_M * S2_K * MESH_ROW * TILE_SIZE

    # One tiny multicast, on the engine that owns tile 0, behind this cluster's own head
    # chain so it runs on a quiet fabric. Tile 0's broadcast then waits on it instead of on
    # ld_q, which both orders the two and gives the real transfer a warm config path.
    bwarm_node = None
    if bcast is not None and BCAST_WARMUP and c == _bcast_owner(0):
        bwarm_node = g.node(
            f"BcastWarm_c{c}", XDMA_CORE, "__snax_bingo_kernel_xdma_multicast",
            SnaxBingoKernelXdmaMulticastArgs(
                h_all["a"][0], [bf["bwarm"] for bf in all_bufs], 64),
            ld_q)

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
    # (index into qk/pv, tile) for consumers whose broadcast producer is owned by a
    # cluster that has not been built yet. Closed in build().
    pending_k, pending_v = [], []
    TOTAL = NKV_PER * NQ          # this cluster's shard, not the whole KV axis
    for i in range(TOTAL + 1):
        if i < TOTAL:
            j, q = divmod(i, NQ)
            bcast_this_k = (bcast is not None and BCAST_K and not K_PULL_FROM_CL0
                            and not (BCAST_SKIP_FIRST and j == 0))
            if q == 0 and bcast is not None and K_PUSH_ALL:
                # One host task delivers this tile's K into all four clusters' k8. Built on
                # cluster 0's pass only -- like the broadcast, and for the same reason: the
                # node order is load-bearing, so the producer must appear where the operand
                # is first consumed rather than hoisted into a prologue.
                if c == 0:
                    k_after = ld_q if j < NKBUF else [qk[(j - NKBUF + 1) * NQ - 1]]
                    bcast["k"][j] = g.node(
                        f"PushK{j}", HOST_CORE, "__host_bingo_kernel_idma_multi",
                        HostBingoKernelIdmaMultiArgs(
                            [(_at(h_all["a"][0], j * KBYTES), bf["k8"][j % NKBUF])
                             for bf in all_bufs],
                            KBYTES),
                        k_after, cluster=0)
                ld_k.append(bcast["k"].get(j))
            elif q == 0 and bcast is not None and K_PULL_FROM_CL0:
                # K on the xDMA: cluster 0 reads main memory, the rest read cluster 0's L1.
                k_after = ld_q if j < NKBUF else [qk[(j - NKBUF + 1) * NQ - 1]]
                own = _pull_owner(j)
                if j < PULL_SKIP_FIRST:
                    # Every cluster reads this tile itself. xload, not load: K lives on the
                    # xDMA in the steady state and moving it to the iDMA would put it behind V.
                    n = xload(f"K{j}", "a", k8[j % NKBUF], KBYTES, k_after, off=j * KBYTES)
                    if c == own:
                        bcast["ksrc"][j] = n
                elif c == own:
                    if K_PUSH_OWNER:
                        n = g.node(f"PushK{j}_c{c}", HOST_CORE,
                                   "__host_bingo_kernel_idma",
                                   HostBingoKernelIdmaArgs(_at(h["a"], j * KBYTES),
                                                           k8[j % NKBUF], KBYTES),
                                   k_after, cluster=0)
                    else:
                        n = xload(f"K{j}", "a", k8[j % NKBUF], KBYTES, k_after,
                                  off=j * KBYTES)
                    bcast["ksrc"][j] = n
                else:
                    # Under rotation the owner may be a cluster built LATER, so its node can be
                    # missing here. Create the pull without that edge and record the hole;
                    # build() closes it once every pass has run.
                    src = bcast["ksrc"].get(j)
                    after = list(k_after) + ([src] if src is not None else [])
                    n = g.node(f"PullK{j}", XDMA_CORE,
                               "__snax_bingo_kernel_xdma_1d_copy",
                               SnaxBingoKernelXdma1dCopyArgs(
                                   all_bufs[own]["k8"][j % NKBUF], k8[j % NKBUF], KBYTES),
                               after)
                    bcast["kpull"].append((j, n))
                    if src is None:
                        bcast.setdefault("kpend", []).append((j, n))
                ld_k.append(n)
            elif q == 0 and bcast_this_k:
                # --- headpar: K arrives by broadcast, from ONE read ----------------------
                # The GQA group shares its KV head, so this tile's K is the same bytes on
                # every cluster. Cluster 0 issues the multicast HERE, inside the tile loop
                # and with the SAME predecessors the unicast load would have had; the other
                # three clusters only take the dependency.
                #
                # THOSE PREDECESSORS ARE LOAD-BEARING. Created dependency-free -- so K0, K1,
                # V0 and V1 are all ready at once on this single xDMA -- the run WEDGES IN
                # TEARDOWN: the whole pipeline completes and the host epilogue never runs.
                # That is exactly the failure the k8 double-buffer comment above predicts
                # for "more simultaneously-ready DMA nodes than the manager's ready set
                # tolerates".
                if c == _bcast_owner(j):
                    k_after = ld_q if j < NKBUF else [qk[(j - NKBUF + 1) * NQ - 1]]
                    if j == 0 and bwarm_node is not None:
                        k_after = [bwarm_node]
                    bcast["k"][j] = g.node(
                        f"BcastK{j}", XDMA_CORE, "__snax_bingo_kernel_xdma_multicast",
                        SnaxBingoKernelXdmaMulticastArgs(
                            _at(h_all["a"][0], j * KBYTES),
                            [bf["k8"][j % NKBUF] for bf in all_bufs],
                            KBYTES),
                        k_after)
                # Under BCAST_SPREAD a cluster built LATER owns some tiles, so the node
                # can be missing here. Leave a hole and let build() close it once every
                # pass has run -- the same deferral the cross-cluster WAR edges already
                # need, and for the same reason.
                ld_k.append(bcast["k"].get(j))
            elif q == 0:
                # Tile 0 under BCAST_SKIP_FIRST lands here too, and wants exactly what this
                # branch already does: this cluster's own iDMA pulling the shared K into its
                # own k8. stage() collapsed headpar's K onto one handle, so h["a"] is that
                # shared array for every cluster -- four reads of the same bytes, in parallel.
                # --- stream this KV tile's K and V --------------------------------------
                # Into the buffer every query tile of step j-2 has finished with, so it
                # waits for the LAST of them. Before tile 2 there is nothing but Q.
                k_after = ld_q if j < NKBUF else [qk[(j - NKBUF + 1) * NQ - 1]]
                ld_k.append(load(f"K{j}", "a", k8[j % NKBUF], KBYTES, k_after,
                                 off=j * KBYTES))
            if q == 0 and bcast is not None and BCAST_V:
                if c == _bcast_owner(j, BCAST_V_SKEW):
                    v_after = list(ld_az) if j < NVBUF \
                        else [pv[(j - NVBUF + 1) * NQ - 1]]
                    bcast["v"][j] = g.node(
                        f"BcastV{j}", XDMA_CORE, "__snax_bingo_kernel_xdma_multicast",
                        SnaxBingoKernelXdmaMulticastArgs(
                            _at(h_all["v"][0], j * VBYTES),
                            [bf["v8"][j % NVBUF] for bf in all_bufs], VBYTES),
                        v_after)
                ld_v.append(bcast["v"].get(j))
            elif q == 0:
                # Behind the arena fills, not just behind Q. Issuing V(0) BEFORE the
                # fills would clear the first softmax of V traffic, but it would push the
                # fills back and grow the head by more than the contention cost. The fills
                # go first. V and the fills share the xDMA engine, and the fills are needed
                # by the FIRST softmax while V is not needed until the first PV. Left to
                # the tie-break the V stream gets in front of them and the first softmax
                # waits out two whole V transfers. The fills are short, so stating the
                # order costs nothing.
                v_after = list(ld_az) if j < NVBUF else [pv[(j - NVBUF + 1) * NQ - 1]]
                # Under kvsplit V streams on the xDMA and K on the iDMA (under headpar it is
                # the reverse, see K_PULL_FROM_CL0). The two engines then move their tiles
                # CONCURRENTLY instead of queueing on one, which takes the K/V pair off the
                # critical path whenever the array is the longer of the two.
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
                        pairs = [(_at(h["v"], j * VBYTES), v8[j % NVBUF])]
                        if nxt < NKV_PER:
                            pairs.append((_at(h["v"], nxt * VBYTES), v8[nxt % NVBUF]))
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
                elif bcast is not None and V_PULL_FROM_CL0:
                    # HEADPAR, V PULLED: cluster 0 reads main memory, the rest read
                    # cluster 0's L1. Every cluster still issues exactly one transfer per
                    # tile on its own iDMA, so no engine gains work -- only the SOURCE
                    # changes, and with it three quarters of the main-memory traffic.
                    own = _pull_owner(j)
                    if j < PULL_SKIP_FIRST:
                        # Same as K: the leading tiles have nothing to overlap their second
                        # hop, so every cluster reads them itself. V stays on the iDMA.
                        n = load(f"V{j}", "v", v8[j % NVBUF], VBYTES, v_after,
                                 off=j * VBYTES)
                        if c == own:
                            bcast["vsrc"][j] = n
                    elif c == own:
                        if V_PUSH_OWNER:
                            n = g.node(f"PushV{j}_c{c}", HOST_CORE,
                                       "__host_bingo_kernel_idma",
                                       HostBingoKernelIdmaArgs(h["v"], v8[j % NVBUF], VBYTES),
                                       v_after, cluster=0)
                        else:
                            n = load(f"V{j}", "v", v8[j % NVBUF], VBYTES, v_after,
                                     off=j * VBYTES)
                        bcast["vsrc"][j] = n
                    else:
                        # wait for cluster 0's copy of THIS tile as well as our own WAR edge
                        src = bcast["vsrc"].get(j)
                        after = list(v_after) + ([src] if src is not None else [])
                        n = g.node(f"PullV{j}", DMA_CORE,
                                   "__snax_bingo_kernel_idma_1d_copy",
                                   SnaxBingoKernelIdma1dCopyArgs(
                                       all_bufs[own]["v8"][j % NVBUF], v8[j % NVBUF], VBYTES),
                                   after)
                        bcast["vpull"].append((j, n))
                        if src is None:
                            bcast.setdefault("vpend", []).append((j, n))
                    ld_v.append(n)
                elif bcast is not None and not V_ON_XDMA:
                    # HEADPAR, V NOT BROADCAST: pull it on the iDMA, never the xDMA.
                    #
                    # A RECEIVING CLUSTER'S xDMA MUST BE IDLE. An incoming multicast write
                    # is handled by the destination cluster's own xDMA finish manager -- the
                    # same block whose i_read_stall_watchdog fires in the 4-issuer deadlock
                    # -- so a cluster cannot be running its own transfer and absorbing a
                    # broadcast at the same time.
                    #
                    # With V on the cluster xDMAs and K broadcast into them, ALL FOUR xDMA
                    # harts wedge in their wait loops; with every cluster xDMA doing nothing
                    # but a few tiny memsets while the broadcasts land, the run completes.
                    # Under headpar the iDMA is free anyway -- K no longer rides it, it
                    # carries only Q -- so this costs nothing and keeps the receiving xDMAs
                    # clear.
                    ld_v.append(load(f"V{j}", "v", v8[j % NVBUF], VBYTES, v_after,
                                     off=j * VBYTES))
                else:
                    ld_v.append(xload(f"V{j}", "v", v8[j % NVBUF], VBYTES, v_after,
                                      off=j * VBYTES))

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
            if q != 0:
                deps = [qk[i - 1]]
            elif ld_k[j] is not None:
                deps = [ld_k[j]]
            else:
                deps = []
                pending_k.append((len(qk), j))
            # UNDER headpar THIS EDGE IS NOT REDUNDANT. In the unicast case Q and K share
            # the iDMA and the load chain puts K(0) behind Q, so QK inherits the wait. The
            # broadcast moves K to the xDMA, which breaks that chain -- without this edge
            # QK(0) would be free to read a q8 the iDMA has not filled yet, and TCDM reads
            # X rather than faulting (SimInit="none"), which kills the hart silently.
            if bcast is not None and j == 0:
                deps.append(ld_q[q])
            # WAR on the score buffer: QK(i) overwrites the tile SM(i-NSCORE) read.
            if i >= NSCORE_WAR:
                deps.append(sm[i - NSCORE_WAR])
            qk.append(g.node(f"QK_{j}_{q}", GEMM_CORE,
                             "__snax_bingo_kernel_gemm_fa_qk",
                             SnaxBingoKernelGemmFaQkArgs(k8[j % NKBUF], q8[q], cz,
                                                         s16[i % NSCORE].view(64), M, K, N,
                                                         perf_addr=perf, d_shift=SCORE_SHIFT),
                             deps))

            deps = [qk[i]]
            if j == 0:
                deps.append(ld_az[q])
                deps.append(warm[q])
            if i >= 1:
                deps.append(sm[i - 1])
            # WAR on the probability buffer: SM(i) overwrites what PV(i-NSCORE) read.
            if i >= NSCORE_WAR:
                deps.append(pv[i - NSCORE_WAR])
            sm.append(g.node(f"SM_{j}_{q}", SIMD_CORE,
                             "__snax_bingo_kernel_simd_fa_softmax",
                             SnaxBingoKernelSimdFaSoftmaxArgs(
                                 s16[i % NSCORE].view(64), p8[i % NSCORE], arena[q],
                                 bc=BC, dhead=DHEAD, tile_idx=j,
                                 seed_state=0,
                                 # THE FULL CSR PROGRAM RUNS ONCE PER CORE, NOT PER TILE.
                                 # snax_simd_program_fast writes 21 CSRs that depend only
                                 # on (bc, dhead), so every softmax after the first on this
                                 # core re-writes the same values. CSR_PRIMED says 'the
                                 # block still holds a same-geometry program, write only the
                                 # six per-task CSRs'. Only the host may assert it: this
                                 # cluster's SIMD core runs nothing but these softmaxes and
                                 # the prologue that programmed the CSRs in the first place.
                                 geom_mode=(SnaxBingoKernelSimdFaSoftmaxArgs.GEOM_PRIMED
                                            if i == 0 else
                                            SnaxBingoKernelSimdFaSoftmaxArgs.GEOM_CSR_PRIMED),
                                 score_scale=EXP_SCALE, p_mode=P_MODE, p8_pitch=P8_PITCH),
                             deps))

        if i >= 1:
            # PV for the PREVIOUS step, emitted after this step's QK.
            p = i - 1
            pj, pq = divmod(p, NQ)
            deps = [sm[p]] if p == 0 else [sm[p], pv[p - 1]]
            if pq == 0:
                if ld_v[pj] is not None:
                    deps.append(ld_v[pj])
                else:
                    pending_v.append((len(pv), pj))
            # Tile 0 starts O from nothing (C masked off, no read); every later tile reads
            # O back through the C-path scaler with THIS tile's corr, which the softmax left
            # in the beat after its row sum.
            pbuf = p8[p % NSCORE]
            pv.append(g.node(f"PV_{pj}_{pq}", GEMM_CORE,
                             "__snax_bingo_kernel_gemm_fa_pv",
                             SnaxBingoKernelGemmFaPvArgs(
                                 v8[pj % NVBUF], pbuf, oacc[pq] if pj else 0, oacc[pq],
                                 S2_M, S2_K, S2_N, perf_addr=perf,
                                 flags=PV_RESCALE if pj else PV_KMAJOR,
                                 corr_addr=(pbuf.view(SnaxBingoKernelSimdFaSoftmaxArgs.corr_offset(
                                     BC, BR, P8_PITCH)) if pj else 0),
                                 b_pitch=P8_PITCH),
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
    # expected exact, so two steps is already slack. l is the RUNNING sum: every tile adds
    # a row sum narrowed from FP32 and rescales the old value by an FP16 corr from the
    # hardware exponential, so it gets eight steps.
    tol_m = float(2 * np.max(np.spacing(m.astype(np.float16))))
    tol_s = float(8 * np.max(np.spacing(np.abs(l_gold).astype(np.float16))))
    if DEBUG_LOOSE_ML:
        # DEBUG ONLY. m and rowsum are the FIRST checks dispatched, and a failing host
        # kernel breaks the scheduler loop (bingo_hw_scheduler_run in bingo_api.c), so one
        # bad m costs the other eleven results. Widening these two lets the run reach the O check, which is the
        # one that says whether the COMPUTATION is wrong or only the m/rowsum readback.
        tol_m = tol_s = 1.0e4

    if DEBUG_PER_TILE_M and c == 2:
        # One store+check per tile on cluster 2 only. Ordered sm[i] -> store ->
        # sm[i+1] so the read happens after THIS tile's commit and before the next tile
        # overwrites mrun. The run aborts at the first failing check, so the UART names the
        # tile directly.
        for _i, _smn in enumerate(sm):
            _l3 = BingoMemAlloc(f"dbg_m_c{c}_t{_i}", size=64, mem_level="L3")
            _st = host(f"DbgStoreM_c{c}_t{_i}", "__host_bingo_kernel_idma",
                       HostBingoKernelIdmaArgs(arena[_i % NQ].view(lay["mrun"]), _l3, 64),
                       _smn)
            host(f"DbgCheckM_c{c}_t{_i}", "__host_bingo_kernel_check_result",
                 HostBingoKernelCheckResultArgs(h["m"], _l3, name=f"dbg_m_c{c}_t{_i}",
                                                check_type=CHECK_FP16_TOL,
                                                num_elements=BR, tolerance=tol_m), _st)
            if _i + 1 < len(sm):
                g.dfg.bingo_add_edge(_st, sm[_i + 1])

    if DEBUG_S16_COMPARE and c == 2 and len(qk) > S16_CMP_BUF + NSCORE:
        # THE WHOLE score tile, not one beat. Comparing one beat says nothing about the other
        # beats the rowmax also reads, and the rowmax is what feeds m.
        _n = 64 + BC * BR * 2
        _a = BingoMemAlloc(f"dbg_s16_first_c{c}", size=_n, mem_level="L3")
        _b = BingoMemAlloc(f"dbg_s16_again_c{c}", size=_n, mem_level="L3")
        _i0, _i1 = S16_CMP_BUF, S16_CMP_BUF + NSCORE
        _sa = host(f"DbgS16First_c{c}", "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(s16[_i0 % NSCORE], _a, _n), qk[_i0])
        _sb = host(f"DbgS16Again_c{c}", "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(s16[_i1 % NSCORE], _b, _n), qk[_i1])
        # the first store must complete before QK(_i1) overwrites the buffer
        g.dfg.bingo_add_edge(_sa, qk[_i1])
        host(f"DbgS16Cmp_c{c}", "__host_bingo_kernel_check_result",
             HostBingoKernelCheckResultArgs(_a, _b, name=f"dbg_s16_c{c}",
                                            check_type=CHECK_BYTE_EXACT,
                                            data_size=_n), _sb)

    l3_m = BingoMemAlloc(f"out_fa_m_c{c}", size=64, mem_level="L3")
    st_m = host(f"Store_m_c{c}", "__host_bingo_kernel_idma",
                HostBingoKernelIdmaArgs(arena[NQ - 1].view(lay["mrun"]), l3_m, 64), last)
    ck_m = host(f"Check_m_c{c}", "__host_bingo_kernel_check_result",
                HostBingoKernelCheckResultArgs(h["m"], l3_m, name=f"fa_m_c{c}",
                                               check_type=CHECK_FP16_TOL,
                                               num_elements=BR, tolerance=tol_m), st_m)

    # l is the arena's RUNNING sum, committed by the last softmax -- NOT the row sum in the
    # last tile's p8 buffer, which is one tile's contribution.
    l3_rs = BingoMemAlloc(f"out_fa_l_c{c}", size=64, mem_level="L3")
    st_rs = host(f"Store_l_c{c}", "__host_bingo_kernel_idma",
                 HostBingoKernelIdmaArgs(arena[NQ - 1].view(lay["lrun"]), l3_rs, 64),
                 [last, ck_m])
    ck_rs = host(f"Check_l_c{c}", "__host_bingo_kernel_check_result",
                 HostBingoKernelCheckResultArgs(h["l"], l3_rs,
                                                name=f"fa_l_c{c}",
                                                check_type=CHECK_FP16_TOL,
                                                num_elements=BR, tolerance=tol_s), st_rs)

    # CHECK O. m and l cover the softmax statistics, but only O depends on P reaching PV in
    # the right layout and on every tile's corr having rescaled the accumulator. O is the only
    # checked value that depends on every tile having been folded in correctly.
    #
    # It is also what makes a layout experiment trustworthy. A write that lands past the end
    # of a buffer (say fa_v8_1) damages whichever buffer the static packer put next: with
    # fa_s16_0 there, every softmax row maximum is wrong and the suite says so; with
    # fa_oacc32_0 there, only the O check sees the damage. A green run whose checks cannot see
    # the failure mode under test is not evidence.
    o_bytes = BR * DHEAD * 4
    l3_o = BingoMemAlloc(f"out_fa_o_c{c}", size=o_bytes, mem_level="L3")
    # WAITS ON THE LAST PV, NOT ON `last`. `last` is the last SOFTMAX, and every other store
    # here reads something a softmax wrote (m lives in the arena, rowsum in p8) -- but O is
    # written by the last PV, which is a SIBLING of this node, not an ancestor: PV(TOTAL-1) is
    # emitted in the loop's +1 iteration and depends on sm[TOTAL-1] exactly as this store does.
    # So with `last` the host iDMA would be free to read fa_oacc32_0 while the VersaCore is
    # still accumulating into it.
    #
    # It is a RAW hazard, so the visible failure is not a wrong number -- the run HANGS, with
    # the host parked, because the first Store_o/Check_o pair is where the host iDMA first
    # overlaps a live VersaCore write to the same TCDM.
    #
    # A small NSCORE hides it: SM(i) takes a WAR edge on PV(i-NSCORE), so a smaller NSCORE
    # drains the PV chain further before `last` retires. At NSCORE=2 only PV(TOTAL-1) can
    # still be outstanding at that point; at NSCORE=3 three PVs can be, and the window is
    # wide enough to hit every time.
    st_o = host(f"Store_o_c{c}", "__host_bingo_kernel_idma",
                HostBingoKernelIdmaArgs(buf["oacc"][NQ - 1], l3_o, o_bytes), [pv[-1], ck_rs])
    ck_o = host(f"Check_o_c{c}", "__host_bingo_kernel_check_result",
                HostBingoKernelCheckResultArgs(h["o"], l3_o, name=f"fa_o_c{c}",
                                               check_type=CHECK_INT32_RELTOL,
                                               num_elements=min(O_CHECK_ELEMS, BR * DHEAD),
                                               tolerance=0.02), st_o)

    # The shard checks are chained rather than left as unordered peers. There is one host
    # core, so they run serially regardless; saying so costs nothing at run time and keeps
    # the per-edge dep tags affordable -- four shards of unordered store->check pairs all
    # land in the same (cluster 0, host, host) cell and each would otherwise need its own.
    return {
        # the first REAL V tile (index 0 of ld_v), so build() can serialise the cold
        # xdma_1d_copy config across clusters
        "first_v": ld_v[0] if ld_v else None,
        "last_sm": last,
        "arena": arena[NQ - 1],
        "p8_last": p8[(NKV_PER * NQ - 1) % NSCORE],
        "checks": ck_o,   # the tail of this shard's store->check chain
        "first_store": st_m,  # the head of it -- build() gates this on EVERY cluster
        # The consumers of the broadcast buffers. build() reaches back through these to
        # close the cross-cluster WAR edges, which cannot be stated while this cluster is
        # being built because three quarters of the consumers do not exist yet.
        "qk": qk,
        "pv": pv,
        # The per-tile K/V FILL nodes, indexed by tile. Whatever kind of node filled the
        # slot -- an owner's load or a pull from another cluster -- ld_k[t] is what wrote
        # THIS cluster's k8[t % NKBUF]. The cross-cluster WAR closure needs exactly that,
        # and it cannot be reconstructed from bcast["ksrc"], which holds owner loads only.
        "ld_k": ld_k,
        "ld_v": ld_v,
        "pending_k": pending_k,
        "pending_v": pending_v,
    }


def build(dfg, h, m_all, l_all, merged_h, jct, tol_ml=0.02):
    """Four cluster pipelines, split either over KV or over the GQA group's query heads.

    Under DECOMP = "headpar" (the default) the four clusters hold four query heads of one
    GQA group. They share a KV head by construction, so each KV tile is read from main
    memory ONCE (by its owner cluster) and pulled by the other three clusters, and each
    cluster's (m, l, O) is already a complete result -- there is nothing to fold. That is the
    configuration that keeps the quadrant compute-bound; see the DECOMP comment at the top of
    this file.

    Under DECOMP = "kvsplit" the rest of this docstring applies.

    The shards are independent: each is the single-cluster pipeline over its own KV
    tiles, producing its own (m_c, l_c). What makes this workload different from four
    copies of the one-cluster run is the epilogue -- the online-softmax merge

        m* = max_c m_c        l* = sum_c exp(a' * (m_c - m*)) * l_c

    is not gathered to one cluster and reduced there. It is computed BY THE FABRIC: each
    cluster copies its partial into the softmax-statistics junction's group, and one
    ChainGather walks the four of them, folding at each hop, so the collector's buffer
    receives the answer rather than the operands.
    """
    # Every cluster's L1 first, with no nodes: a broadcast names destinations in all four,
    # so the handles have to exist before the first load node does.
    bufs = [_alloc_cluster(dfg, c) for c in range(NCL)]

    if DECOMP == "headpar":
        # ---- one main-memory read per tile, shared by the four clusters ---------------
        # ALL BROADCASTS ON ONE ENGINE (cluster 0). Spreading them over the four clusters'
        # xDMAs DEADLOCKS the fabric: the adapter has a single from_remote context, so all
        # four clusters' receive windows open together and none ever closes, and every
        # finish manager sticks in ReadBusy. BCAST_SPREAD reproduces it; never enable it.
        #
        # The nodes themselves are created inside cluster 0's own _build_cluster pass, so
        # the global node order is the one the note there requires.
        # "vsrc" carries cluster 0's V load nodes so the other three can depend on them,
        # the same deferral the broadcast nodes already use.
        bcast = {"k": {}, "v": {}, "vsrc": {}, "vpull": [],
                 "ksrc": {}, "kpull": []}
        shards = [_build_cluster(dfg, c, h, m_all, l_all, bufs[c], bcast, bufs)
                  for c in range(NCL)]

        # ---- close the forward references ----------------------------------------------
        # Under BCAST_SPREAD cluster c consumes tiles issued by clusters built after it, so
        # those RAW edges could not be stated inline. They are ordinary edges; only their
        # statement is deferred.
        for sh in shards:
            for idx, j in sh["pending_k"]:
                dfg.bingo_add_edge(bcast["k"][j], sh["qk"][idx])
            for idx, j in sh["pending_v"]:
                dfg.bingo_add_edge(bcast["v"][j], sh["pv"][idx])

        # ---- close the cross-cluster WAR edges ------------------------------------------
        # A broadcast buffer is free only when EVERY cluster's consumer of the tile 2 (resp.
        # NVBUF) steps back has retired. Those consumers do not exist while the broadcast
        # node is being created, so the edges are added here. One producer, four consumers:
        # this is the coupling that head-parallelism buys its bandwidth with.
        # Only the BROADCAST needs this: one producer writes every cluster's k8, so it must
        # wait for every cluster's consumer. Under K_PULL each cluster fills its own buffer
        # and its own k_after already covers its own consumer -- the only extra edge needed is
        # puller -> cluster 0's refill, added below.
        # The K push has exactly the broadcast's shape -- one producer writing every
        # cluster's k8 -- so it needs exactly the broadcast's WAR edges.
        if (BCAST_K or K_PUSH_ALL) and not K_PULL_FROM_CL0:
            for j in range(2, NKV_PER):
                for sh in shards:
                    dfg.bingo_add_edge(sh["qk"][(j - 1) * NQ - 1], bcast["k"][j])
        if BCAST_V:
            for j in range(NVBUF, NKV_PER):
                for sh in shards:
                    dfg.bingo_add_edge(sh["pv"][(j - NVBUF + 1) * NQ - 1], bcast["v"][j])
        # A pull whose source cluster is built later than the puller could not name its
        # producer inline. These are ordinary RAW edges; only their statement is deferred.
        for key, srcmap in (("kpend", "ksrc"), ("vpend", "vsrc")):
            for j, n in bcast.get(key, []):
                dfg.bingo_add_edge(bcast[srcmap][j], n)

        # WAR ACROSS CLUSTERS. A puller of tile j reads the OWNER's buffer,
        # all_bufs[own(j)][slot j % NBUF]. That buffer is next overwritten by the owner's
        # OWN fill of the same slot, which is its ld_k/ld_v[j + NBUF] -- so the edge must
        # be stated against that node, on that cluster.
        #
        # It is NOT stated against `bcast["ksrc"][j + NBUF]`, which is wrong twice over once
        # PULL_ROTATE is on. `ksrc[t]` is the node on cluster t % NCL, so with NCL=4 and
        # NKBUF=2 it would name cluster (j+2) % 4 -- a DIFFERENT cluster, whose k8 is a
        # different physical buffer. And the node that really overwrites own(j)'s slot is
        # frequently a PULL (own(j+2) != own(j)), which is recorded in bcast["kpull"] and
        # never appears in ksrc at all.
        #
        # A missing edge leaves a source buffer overwritten mid-pull: the puller gets a later
        # tile's K, its scores are wrong, and the running max m comes out far too large. The
        # window opens only when the schedule has slack for it (a deeper NSCORE, more tiles
        # per cluster).
        def _close_pull_war(pulls, key, nbuf):
            for j, n in pulls:
                own = _pull_owner(j)
                fills = shards[own].get(key) or []
                t = j + nbuf
                nxt = fills[t] if t < len(fills) else None
                if nxt is not None and nxt is not n:
                    dfg.bingo_add_edge(n, nxt)

        if K_PULL_FROM_CL0:
            _close_pull_war(bcast["kpull"], "ld_k", NKBUF)
        if V_PULL_FROM_CL0:
            _close_pull_war(bcast["vpull"], "ld_v", NVBUF)
        # THE HOST EPILOGUE WAITS FOR EVERY CLUSTER, NOT JUST ITS OWN.
        #
        # Each shard's stores need only their OWN cluster's compute, so with four clusters
        # skewed -- increasingly so with more tiles -- Store_o_c2 would issue a 16 KB host
        # iDMA read out of cluster 2's L1 while clusters 0/1/3 are still streaming K and V
        # through the same fabric. That combination wedges the machine: host and all sixteen
        # snitch traces freeze together while the simulator keeps burning CPU. These edges
        # are the SW way around that RTL fragility.
        #
        # They cost nothing in the utilisation figures: every window here ends at the LAST
        # PV, so the host tail sits outside the measurement; all this does is stop the tail
        # from overlapping live traffic.
        last_pvs = [sh["pv"][-1] for sh in shards if sh["pv"]]
        for sh in shards:
            for lp in last_pvs:
                if lp is not sh["pv"][-1]:
                    dfg.bingo_add_edge(lp, sh["first_store"])

        # No fold: under head-parallelism each cluster's (m, l, O) is already the complete
        # answer for its own query head. The per-shard checks are the whole verification.
        return

    shards = [_build_cluster(dfg, c, h, m_all, l_all, bufs[c])
              for c in range(NCL)]
    g = G(dfg, 0)

    # De-synchronise the cold xdma_1d_copy config (see STAGGER_FIRST_V). One edge per
    # consecutive pair; the transfers still overlap, only the cold CONFIGS are serialised.
    if STAGGER_FIRST_V:
        prev = None
        for sh in shards:
            fv = sh.get("first_v")
            if fv is None:
                continue
            if prev is not None:
                dfg.bingo_add_edge(prev, fv)
            prev = fv

    # One shard is the control arm, not a degenerate merge: with nothing to fold, m* = m_0
    # and l* = l_0, so a gather would only copy. The per-shard checks still run, and the
    # pipeline it measures is the same one each shard runs in the four-cluster case.
    if NCL < 2:
        return

    # ---- copy each shard's (m, l) into the junction's group ------------------------------
    # The junction reads a key beat followed by a value beat; the arena keeps m and l apart,
    # so each shard copies them into one buffer, on its own xDMA core, so that the gather
    # that consumes it is the very next thing that core does. m is the running max as the
    # softmax leaves it and l the running sum (lrun), committed by the last softmax.
    part_bytes = SnaxBingoKernelPackFaPartialArgs.packed_bytes(BR)
    _, n_valid = xdma_softmax_stats_groups(BR)
    csr0, csr1 = xdma_softmax_stats_csrs(n_valid, EXP_SCALE)
    lay = SnaxBingoKernelSimdFaSoftmaxArgs.layout(BC, DHEAD)
    parts, packs = [], []
    for c, sh in enumerate(shards):
        gc = g.at(c)
        part = gc.l1("fa_ml_partial", part_bytes)
        parts.append(part)
        packs.append(gc.node(
            f"PackPartial_c{c}", XDMA_CORE, "__snax_bingo_kernel_pack_fa_partial",
            SnaxBingoKernelPackFaPartialArgs(
                src_m=sh["arena"].view(lay["mrun"]), src_l=sh["arena"].view(lay["lrun"]),
                dst=part, n_rows=BR),
            [sh["last_sm"]]))

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
            # The identifier WRITER_JCT_SOFTMAXSTATSJUNCTION is device-side only; see
            # writer_junction_index() for why the host emits the derived number instead.
            junction=f"{jct} /* WRITER_JCT_SOFTMAXSTATSJUNCTION */",
            # nValid is the number of query rows folded in every group; CSR(1) is the
            # softmax's own temperature, applied to the key difference before the exp.
            jct_csr0=csr0, jct_csr1=csr1),
        packs)

    # ---- check the merged result --------------------------------------------------------
    # Compared in the junction's own group order as FP16, which is what the collector's
    # buffer holds -- per 32 rows a key beat then a value beat.
    l3_ml = BingoMemAlloc("out_fa_ml_merged", size=part_bytes, mem_level="L3")
    st_ml = g.node("Store_ml", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(merged, l3_ml, part_bytes),
                   [gather] + [sh["checks"] for sh in shards], cluster=0)
    g.node("Check_ml", HOST_CORE, "__host_bingo_kernel_check_result",
           HostBingoKernelCheckResultF16RelArgs(merged_h, l3_ml, part_bytes // 2,
                                                rtol=tol_ml, name="fa_ml_merged"),
           st_ml, cluster=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    p.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    p.add_argument("--hwcfg", type=pathlib.Path, required=True)
    p.add_argument("--platformcfg", type=pathlib.Path, required=True)
    p.add_argument("--data_h", type=pathlib.Path, default=None)
    p.add_argument("--configs_out", type=pathlib.Path, default=None)
    # OFF BY DEFAULT. With this flag the mini-compiler places every L1 buffer itself and the
    # kernels get constant offsets instead of runtime bingo_l1_alloc handles; without it,
    # nothing about the emitted addresses changes. It is a flag rather than an environment
    # variable on purpose -- it changes generated code, and the SW build runs inside a
    # container, so an ambient switch can be set where the build is launched and absent where
    # the compiler actually runs. That failure is silent: the build succeeds and the run
    # passes, having quietly ignored it.
    p.add_argument("--static-l1", action="store_true",
                   help="let the compiler place L1 buffers statically (default: off)")
    p.add_argument("--desc-list-in-wide-spm", action="store_true",
                   help="put the BINGO task-descriptor list in the wide SPM "
                        "(default: narrow SPM)")
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
    data = build_shards()
    m, l, merged = data["m"], data["l"], data["merged"]
    st = DataStaging(platform)
    h = stage(st, data)
    # headpar has nothing to fold, so there is no merged golden to stage.
    merged_h = stage_merged(st, merged) if merged is not None else None
    # The merged check is relative: the junction rounds l* to FP16 at every hop and its
    # exponential is a table, not libm's, so l* lands a few FP16 steps from the golden at
    # any magnitude; m* is exact given exact per-shard maxima.
    tol_ml = 0.02
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
                                    "qshift": QSHIFT, "score_scale": SCORE_SCALE,
                                    "score_shift": SCORE_SHIFT, "p8_pitch": P8_PITCH,
                                    "mesh": [MESH_ROW, TILE_SIZE, MESH_COL]}]},
                      f, indent=2)

    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return

    dfg = BingoDFG(num_chiplets=1,
                   num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
                   num_cores_per_cluster=platform["num_cores_per_cluster"],
                   is_host_as_acc=True, chiplet_ids=[0x00],
                   dep_tag_width=platform["dep_tag_width"])
    # The junction index is only meaningful for the fold. headpar has no fold, so it must
    # not demand a cfg feature it never uses -- a head-parallel quadrant is a legitimate
    # target on hardware built without the softmax-statistics junction.
    jct = writer_junction_index(hw, "HasSoftmaxStatsJunction") if DECOMP == "kvsplit" else None
    build(dfg, h, m, l, merged_h, jct, tol_ml)

    os.makedirs(args.output_dir, exist_ok=True)
    dfg.bingo_compile_dfg(
        "FlashAttention, GQA head-parallel over 4 clusters with a broadcast KV stream"
        if DECOMP == "headpar" else
        "FlashAttention, KV-sharded over 4 clusters with an in-fabric merge",
        args.output_dir, args.output_offload_file_name,
        extra_include_header_list=["fa_data.h"],
        static_l1=args.static_l1,
        desc_list_in_narrow_spm=not args.desc_list_in_wide_spm)
    # Every buffer _build_cluster allocates, with its real multiplicity (it scales with NKBUF,
    # NVBUF, NSCORE and NQ). A budget line that cannot go up is worse than none, because the
    # L1 heap is the thing that silently bounds this workload.
    p8_len = SnaxBingoKernelSimdFaSoftmaxArgs.p8_bytes(BC, BR, P_MODE, P8_PITCH)
    l1 = (NKBUF * (M * K * MESH_ROW * TILE_SIZE)          # k8, NKBUF-deep
          + NVBUF * (S2_M * S2_K * MESH_ROW * TILE_SIZE)  # v8, NVBUF-deep
          + NQ * (N * K * MESH_COL * TILE_SIZE)           # q8, one per query tile
          + 0                                             # cz removed
          + NSCORE * (64 + BC * BR * 2)                   # s16
          + (NSCORE // 2 * (P8_NEST_OFF + p8_len) + NSCORE % 2 * p8_len
             if P8_NEST else NSCORE * p8_len)            # p8
          + NQ * (BR * DHEAD * 4)                         # oacc
          + NQ * SnaxBingoKernelSimdFaSoftmaxArgs.arena_bytes(BC, DHEAD)
          # headpar folds nothing, so it allocates no partial.
          + (0 if DECOMP == "headpar"
             else SnaxBingoKernelPackFaPartialArgs.packed_bytes(BR)))
    # Main-memory reads for the whole run, which is the quantity the decomposition
    # changes: kvsplit pays NCL x per-cluster, headpar pays it once.
    kb, vb = M * K * MESH_ROW * TILE_SIZE, S2_M * S2_K * MESH_ROW * TILE_SIZE
    qb = N * K * MESH_COL * TILE_SIZE
    if DECOMP == "headpar":
        # A broadcast tile is read once; a per-cluster tile is read NCL times. Under
        # BCAST_SKIP_FIRST tile 0 is the latter and the rest the former, so K costs
        # NCL + (NKV_PER - 1) reads rather than NKV_PER.
        k_reads = (NKV_PER if not BCAST_K
                   else (NCL + NKV_PER - 1) if BCAST_SKIP_FIRST else NKV_PER)
        k_reads = NCL * NKV_PER if not BCAST_K else k_reads
        # V_PULL_FROM_CL0 reads each V tile from main memory ONCE; the other three copies
        # come out of cluster 0's L1 and never touch the main-memory port.
        v_reads = NKV_PER if (BCAST_V or V_PULL_FROM_CL0) else NCL * NKV_PER
        traffic = (k_reads * kb + v_reads * vb
                   + NCL * qb)
    else:
        traffic = NCL * (NKV_PER * (kb + vb) + qb)
    mac = 2 * BC * BR * DHEAD * NKV_PER * NQ * NCL
    print(f"Generated FlashAttention ({DECOMP}): Br={BR} Bc={BC} d={DHEAD} NKV={NKV}, "
          f"qshift={QSHIFT}, a={SCORE_SCALE}, k={SCORE_SHIFT} (a'={EXP_SCALE:g}), "
          f"P8 pitch {SnaxBingoKernelSimdFaSoftmaxArgs.pitch(P8_PITCH)}{' nested' if P8_NEST else ''}, "
          f"NSCORE={NSCORE}, L1 buffers {l1:,} B of 514,816 B "
          f"({100 * l1 / 514816:.0f}%)")
    print(f"  main-memory reads {traffic:,} B for {mac:,} MAC "
          f"= {mac / traffic:.0f} MAC/byte (one 512-bit port balances at "
          f"{NCL * MESH_ROW * MESH_COL * TILE_SIZE // 64} MAC/byte)")


if __name__ == "__main__":
    main()
