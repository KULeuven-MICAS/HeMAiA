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
    SnaxBingoKernelXdmaMulticastArgs,
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
CHECK_BYTE_EXACT = 0        # data_size IS the byte count for this mode
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

# HOW THE WORK IS SPLIT ACROSS THE CLUSTERS OF ONE QUADRANT. See
# docs/fa_decomposition_hierarchy.md for the full argument; the short version:
#
#   "kvsplit"  Each cluster owns a disjoint run of KV tiles and its own K. The softmax
#              recurrence runs ALONG KV, so every cluster carries an independent (m, l)
#              and ONE cross-cluster fold closes the run -- which is what exercises the
#              ChainGather over the MonoidJunction.
#
#   "headpar"  Each cluster owns one QUERY HEAD of a GQA group. The group shares its KV
#              head by definition, so K and V are byte-identical on all four clusters and
#              only Q differs. They are read from main memory ONCE and multicast into the
#              four L1s. No fold: each cluster's output is already a complete attention
#              result for its own head.
#
# WHY THE DEFAULT IS headpar. FA decode's arithmetic intensity is Br exactly -- each KV byte
# feeds Br query rows -- and Br is pinned by the SIMD beat. All clusters mux onto ONE wide
# path, so the machine's balance point is well above that intensity and the kernel is
# memory-bound by construction. One cluster on its own is not, because the same intensity
# then sits against a much lower balance point.
#
# kvsplit cannot fix that from software. It divides the compute by the cluster count and
# multiplies the KV traffic by it, so the ratio is a property of the DECOMPOSITION, not of
# the schedule -- which is why every arm that merely REORDERED traffic was reabsorbed.
# headpar instead amortises one KV read over every cluster's query rows, so the effective
# intensity rises by the cluster count and the balance point flips back to the compute-bound
# side. Same total work, a fraction of the main-memory traffic.
#
# THE RESULT IS NOT "head-parallelism wins". Head-parallelism WITHOUT a broadcast is worse
# than kvsplit at identical traffic volume and identical engine pairing: several clusters
# reading the SAME bytes concurrently is worse than the same number reading disjoint bytes.
# Every bit of headpar's advantage comes from the broadcast, not from the decomposition.
#
# But broadcasting BOTH operands is worse again, because one issuer then serialises two
# streams into a single window during which every array idles, and that one gap dominates
# the array idle for the run.
#
# SO THE OPTIMUM IS EXACTLY ONE BROADCAST OPERAND. Zero throws away the read sharing; two
# serialises the only engine allowed to issue. That is a direct consequence of the adapter
# holding ONE remote-write context -- the same constraint that makes BCAST_SPREAD hang --
# and it is the whole design rule this file encodes.
#
# WHAT BINDS AFTERWARDS IS THE SIMD SOFTMAX, not memory -- but only for as long as the
# softmax is EXPOSED. Whether raising NQ pays therefore depends on the cluster hardware and
# not on this file: it doubles array work on the same delivered bytes, which is a large win
# when the array is the constraint and close to nothing when the softmax it also doubles is
# already the constraint. Re-measure it against the current build rather than trusting a
# figure recorded here.
#
# The cross-chiplet half of the idea is UNTOUCHED and still right -- chiplets do not share a
# port, so KV-split + the in-fabric fold belongs there.
#
# CARRYING AN UNUSED KERNEL IS NOT FREE. Adding one SNAX_EXPORT_FUNC entry shifts every
# kernel address and the .rodata dispatch table, and .rodata reads from a cluster kernel are
# blocking fabric round trips. A device library entry costs something even on the arms that
# never call it; #ifdef it out if that matters.
DECOMP = "headpar"

# WHICH operand rides the broadcast, under headpar. EXACTLY ONE SHOULD.
#
# Both ends of the range are worse than the middle -- neither operand broadcast, and both
# operands broadcast, each lose to exactly one. This is a genuine optimum, not a monotone
# knob, so do not "improve" it by moving further in either direction.
#
# WHY K AND NOT V, WHEN THE BROADCAST WAS USED. Either single choice moves one stream off
# the broadcaster, but K is the one the first QK waits for, so broadcasting K puts the shared
# read on the critical path's own operand while V -- needed later, by PV -- streams underneath
# on an engine that head-parallelism otherwise leaves idle. That pairing removed the head
# stall entirely.
#
# THE COST WAS AN ASYMMETRY. Only the issuing cluster's xDMA is loaded, so it becomes the
# slowest of the group and drags the mean below what the other clusters reach on their own.
# Under one remote-write context there is no way to spread it -- see BCAST_SPREAD.
#
# BOTH ARE NOW FALSE: the multicast is no longer used. It does not survive a long KV axis --
# at 16 tiles per cluster the broadcast arm stalls mid-stream around tile 11 while the
# all-pull arm completes, and a stream that does not finish is not a faster stream. Every
# operand now moves as an ordinary DMA: one cluster reads main memory, the other three read
# that cluster's L1. See K_PULL_FROM_CL0 and V_PULL_FROM_CL0.
BCAST_K = False
BCAST_V = False

# WHO ISSUES THE BROADCASTS. LEAVE THIS FALSE -- IT HANGS.
#
# True spreads tile j's broadcast onto cluster (j % NCL), so every engine both sends its own
# tiles and receives the others'. `02-hemaia-integration-summary.md` names this as the
# configuration its defects 1-3 make legal and asks for it to be re-run. It was, on a build
# carrying those fixes (xdma_axi_adapter 0b96648, full RTL regen). IT STILL HANGS.
#
# Both a full fan-out and a two-issuer fan-out wedge at the SAME point in the program and
# within a few hundred nanoseconds of each other, so neither the issuer count beyond two nor
# the fan-out width matters. And with two issuers the FIRST casualty is a cluster that issues
# nothing: the grant manager that wedges first belongs to a pure RECEIVER. No read-stall
# watchdog fires anywhere, so this is not the finish-manager read path.
#
# Symptom if you trip it anyway: two xDMA harts spinning forever on `csrr 0x409` in
# xdma_wait_task, and one .dasm growing past a gigabyte. Full write-up with timings in
# docs/xdma_multi_issuer_multicast_hang.md.
BCAST_SPREAD = False


# WHICH ENGINE GETS TILE j's V, RELATIVE TO ITS K.
#
# BCAST_SPREAD alone does not help the FIRST tile, and the first tile is where headpar
# loses. A COLD transfer costs several times what the same transfer costs warm, and when the
# same cluster owns both operands of tile 0 the two cold transfers are SERIAL -- so the head
# is two cold transfers long before any compute starts. Steady state was never the problem.
#
# A non-zero skew puts tile j's V on a different engine from its K, so the two cold
# transfers overlap instead of queueing. NCL // 2 is the maximum separation on a quadrant.
BCAST_V_SKEW = 0

# WARM THE MULTICAST PATH BEFORE THE FIRST REAL TILE. MEASURED AND REFUTED -- leave False.
#
# The idea: a cold broadcast runs several times longer than a warm one, and the head
# warm-ups elsewhere in this file exist because a core's first dispatch of a kernel is
# dominated by icache refills. So warm the multicast too, with a small transfer into scratch.
#
# It does not work, in BOTH configurations, and it costs. The per-cluster split is the tell:
# it hurts exactly the cluster that runs it and nobody else, so it is pure added work on the
# issuing engine with no compensating saving.
#
# TWO REASONS IT FAILED, and the second is a flaw in the experiment rather than the idea.
# (a) a small transfer is a completely different descriptor shape from a full tile, so it
#     would not warm the path that actually costs anything even if first-dispatch cost were
#     the issue.
# (b) more likely it is not first-dispatch cost at all: the cold broadcast runs while the
#     whole quadrant is starting up. Contention, not refills.
# If anyone retries this, warm it with a FULL-SIZE transfer and check whether the first real
# broadcast's run span actually shrinks -- that is the measurement that decides it, not the
# end-to-end number.
BCAST_WARMUP = False

# Let head-parallel put V on the cluster's OWN xDMA, the way kvsplit does.
#
# The `bcast is not None` branch below forces V onto the iDMA under head-parallel, because a
# cluster cannot absorb an incoming broadcast while its own xDMA is running a transfer. That
# rule is right WHILE ANYTHING IS BEING BROADCAST, but with BCAST_K and BCAST_V both off
# there is no incoming remote write at all, and leaving four xDMAs idle would make the
# no-broadcast control unfair to itself.
#
# EXISTS FOR THAT CONTROL, which is the one that justifies the whole broadcast. Pairing K
# and V on separate engines is the honest floor for head-parallelism without a broadcast --
# putting both on one engine is far worse, because that engine then binds. And that floor is
# BELOW what kvsplit reaches at identical traffic and identical engine use: several clusters
# reading the same bytes at the same time is worse than the same number reading disjoint
# bytes. Do not set this True together with a broadcast: that is the arm whose receiving
# xDMAs all wedged.
V_ON_XDMA = False

# DO NOT BROADCAST THE FIRST TILE -- pull it per-cluster like KV-split does.
# MEASURED AND REFUTED. Leave this False.
#
# The premise was sound and the first half of it held. A cold first multicast costs several
# times what a warm one does, in every phase -- kernel entry, slot config and the transfer
# itself -- and it sits on the startup critical path. Loading tile 0 per-cluster instead DID
# shrink the head, as predicted.
#
# It lost anyway, because the pipeline got worse by more than the head gained. And it is not
# bandwidth: the iDMAs took on only the one extra tile's worth of work. It is SERIALISATION.
# Tile 0 lands at the head of the same iDMA chain that feeds V(0); with a two-deep V buffer
# that chain is gated by PV completion, so a delay at its head propagates down the whole
# chain. The cross-cluster WAR edges then make every cluster wait for the slowest of the
# group, which per-cluster loading has just de-synchronised.
#
# The lesson, for anything else that tries to move work off the broadcaster: the iDMA is not
# spare capacity here. It is already the critical engine for V, and the WAR chain amplifies
# whatever is put in front of it.
#
# RE-MEASURED WITH THE ROTATED PULL AND IT STILL LOSES, but the shape of the loss changed:
# the head win has nearly vanished while the pipeline cost has not. ROTATION ALREADY TOOK MOST
# OF WHAT THIS WAS REACHING FOR -- with tile j fetched by cluster j % NCL the head's memory
# traffic is already spread over four clusters -- and it still pays the full WAR-chain price.
# Two knobs that each look like a head optimisation are not additive: they attack the same
# serialisation, and rotation is the better of the two.
BCAST_SKIP_FIRST = False

# PULL V FROM CLUSTER 0's L1 INSTEAD OF FROM MAIN MEMORY.
#
# Under headpar every cluster's V is the same bytes, and today all four read them
# independently from main memory -- 16 tile-reads, 79% of all main-memory read traffic.
# The alternative: cluster 0 reads a tile once, the other three copy it out of cluster 0's
# L1 with their OWN iDMAs. Same number of transfers per engine, a quarter of the
# main-memory traffic, and the inter-cluster links carry the rest.
#
# bingo_l1_alloc(chip, cluster, size) returns a full SoC address, so naming another
# cluster's buffer needs no address arithmetic.
#
# NOT the same as a broadcast: this is a PULL, so the three copies proceed in parallel on
# three engines rather than being issued serially by one.
#
# MEASURED, and it wins on the metric that counts data movement, for a reason that is NOT
# the one predicted. Main-memory read traffic does fall sharply, but the shared port was only
# half used, so bandwidth was never the constraint. What the pull relieves is STARTUP
# CONTENTION: in the baseline every cluster's first fetch hits main memory at the same
# instant. Moving most of them onto the inter-cluster links cuts the head -- the cold-start
# cost nothing else has touched.
#
# The cost is real and visible: one extra dependency per tile, since the other clusters wait
# for the owner's copy, which lengthens the pipeline and costs streaming efficiency. It is
# outweighed comfortably here.
#
# THE TWO EFFECTS SCALE OPPOSITELY, so this is not unconditionally right. The head saving is
# FIXED; the pipeline penalty is PER TILE. There is therefore a sequence length beyond which
# reading main memory per-cluster should win, and the only way to know where it sits on a
# given machine is to measure both arms there.
V_PULL_FROM_CL0 = True

# PULL K CROSS-CLUSTER TOO, ON THE OTHERWISE IDLE xDMAs -- no multicast at all.
#
# The V pull showed the head is a CONTENTION problem: routing traffic off main memory and onto
# the inter-cluster links cut it even though the memory port was only half used.
# This applies the same move to K, and removes the last multicast in the process.
#
# The multicast is worth removing on its own terms. Its FIRST invocation costs several times
# what a warm one does, in every phase -- it is the single largest item left in the head. It
# also makes the issuing cluster asymmetric: the only one whose xDMA has work, and the
# slowest of the group.
#
# The arrangement, with both engine types busy on every cluster:
#   cluster 0     xDMA: K(j) from main memory      iDMA: V(j) from main memory
#   clusters 1-3  xDMA: K(j) from cluster 0's L1   iDMA: V(j) from cluster 0's L1
#
# Every cluster then issues 4 K-sized transfers and 4 V-sized ones on separate engines, and
# main-memory traffic is unchanged at 4 K + 4 V tiles. Note this is only legal because nothing
# is being broadcast any more: the rule that a receiving cluster's xDMA must stay idle exists
# to keep it free to absorb an incoming multicast, and there is no longer one to absorb.
#
# AT FOUR TILES PER CLUSTER IT LOSES, and that is where the rule "head-parallel wants exactly
# one broadcast operand" was measured: zero broadcasts (this knob) and two were both worse
# than one.
#
# AT A LONG KV AXIS IT IS THE ONLY ARM THAT FINISHES. The broadcast arm stalls mid-stream;
# this one passes every check. The rule above was a statement about the HEAD, and the head is
# a fixed cost that amortises -- measured as the same absolute number at four times the
# sequence length -- while the multicast's per-tile risk does not. Long KV wins the argument,
# so K is pulled.
K_PULL_FROM_CL0 = True

# ROTATE WHICH CLUSTER IS THE SOURCE, per tile.
#
# With the pulls as first written, cluster 0 is the source for EVERY tile of both operands: its
# TCDM serves three remote readers on top of its own array, and it is the only cluster reading
# main memory. That is the same asymmetry the broadcast had, just moved from the write path to
# the read path -- and cluster 0 is measurably the slowest of the four.
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
#
# MEASURED AND IT IS THE BEST CONFIGURATION FOUND. Serving every pulled tile from one
# cluster costs about as much as taking K off the multicast does; rotating the server gives it
# back. Combined with the multicast it is the only arm that improves every metric at once, and
# head-parallel then passes KV-split end to end while keeping the streaming lead it had.
PULL_ROTATE = True


# HOW MANY LEADING TILES EVERY CLUSTER READS FROM MAIN MEMORY ITSELF, instead of pulling.
#
# THE PROBLEM IT SOLVES. A pulled tile is a TWO-HOP dependency: main memory -> the owner's L1
# -> the other three L1s. That is fine in steady state, where the second hop overlaps the
# previous tile's compute. It is not fine for the FIRST tile, because nothing overlaps it:
# the other clusters sit idle for the whole of the owner's read before their own pull can even
# start. Measured, the first GEMM gap is far larger on the pullers than on the owner, and
# their own xDMA is only a fraction busy inside it -- they are not moving data, they are
# waiting for somebody else's.
#
# WHAT IT COSTS. Tile j < this is read from main memory NCL times instead of once. That is
# only affordable if main memory has the headroom, and on average it appears to: the
# quadrant's single wide exit measures a small fraction busy across the pipeline, so the
# extra reads look like they should land in slack.
#
# MEASURED AND IT LOSES. The head did shrink as predicted,
# and the PIPELINE grew by more, for a net loss of window.
#
# The reasoning that justified it was wrong in a specific, reusable way. "Main memory has the
# headroom, its wide exit is barely busy" is an AVERAGE over the pipeline. At the fill every
# cluster reads at the same instant, so the instantaneous demand is the cluster count over and
# they serialise on the single quadrant exit. The two-hop pull is slower per tile but it
# STAGGERS; four direct reads are faster per tile and arrive together.
#
# An average occupancy never justifies adding a burst. Leave this at 0.
PULL_SKIP_FIRST = 0


# HOW MANY OF O'S ELEMENTS THE HOST CHECKS, as a fraction of the BR x DHEAD accumulator.
#
# WHY IT IS WORTH A KNOB. The O checks are the single most expensive thing in the simulation
# -- together they can dominate the whole run. They cost that because the host walks the
# accumulator out of L3 with no D-cache, so every element is a blocking fabric round trip. Shrinking them is the cheapest way to make an iteration faster,
# and iteration speed is what a sweep is limited by.
#
# WHY NOT ZERO. O is the only check that covers ACCUMULATION. m and rowsum are per-tile and
# come out identical on every tile, so both pass even when the recurrence across tiles is
# wrong -- adding the O check is what found two such bugs. Never turn it off; shrink it.
#
# WHAT A PREFIX COVERS. The accumulator is [BR][DHEAD] row-major, so a prefix of N elements is
# the first N/DHEAD query rows in full. Every KV tile contributes to every row, so one whole
# row already exercises the full accumulation chain; more rows buy independent samples of it,
# not extra depth. 512 = 4 of the 32 rows.
O_CHECK_ELEMS = 512

# DEBUG ONLY -- see the tol_m/tol_s override. Set False to restore real tolerances.
DEBUG_LOOSE_ML = False

# DIAGNOSTIC for bug A (NSCORE=3 gives m values above the arithmetic maximum once
# TOTAL >= 8). m is a running max and every KV tile is fed IDENTICAL bytes, so m must
# reach its final value at tile 0 and never move again -- which means the same golden is
# valid after EVERY tile. Storing and checking m per tile therefore names the exact tile
# at which it first goes wrong, and with it the s16/p8 buffer (i % NSCORE) responsible.
#
# Safe to use precisely because bug A is DETERMINISTIC: war2 reproduced it bit-identically
# with a different dependency depth, so adding host round-trips between tiles cannot hide
# it the way it would hide a race.
DEBUG_PER_TILE_M = False

# DIAGNOSTIC for bug A, the decisive one. Every KV tile is fed IDENTICAL K and Q bytes, so
# QK(i) and QK(i + NSCORE) -- which write the SAME s16 buffer -- must produce BYTE-IDENTICAL
# score tiles. Storing one beat of s16 after each and comparing them needs no golden at all:
# the first store IS the golden for the second.
#
#   they match    -> the score tile is fine and m goes wrong later, in the SIMD recurrence
#   they differ   -> QK is not overwriting s16, which is what a device m of 386 (= 3 x the
#                    128 arithmetic maximum for one tile) has been pointing at all along
#
# Only valid because bug A is DETERMINISTIC (war2 reproduced it bit-identically at a
# different WAR depth), so the extra host round-trips cannot hide it.
DEBUG_S16_COMPARE = False
# Which tile's score buffer to compare against its NEXT use of the same buffer. Bug A first
# shows at TILE 5, which uses s16[2] -- so 2, not 0. The first run of this diagnostic compared
# s16[0] (tiles 0 vs 3), and BOTH of those tiles pass their m check, so it proved nothing
# about the buffer that actually fails. Corrected.
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
# TESTED AT FOUR CLUSTERS AND IT STILL LOSES, on both the pipeline and the head, so unlike
# NVBUF=3 this is not a transfer between them but a straight loss. The flaw in the argument
# above is that TCDM BANDWIDTH is what is being traded, and it is shared whether or not either
# engine is "the critical path": a third score buffer means a third QK streaming A, B and D
# through the same ports the cross-cluster V pull already uses. Two engines idling in lockstep
# is evidence that something they SHARE is the constraint, not that nothing is.
#
# It does fit in L1, with little to spare. It WEDGES, twice, reproducibly -- whether it would
# pay is still unknown, because no run has ever reached the end.
#
# THE EVIDENCE, on the second run, after a first diagnosis that was right by luck and a
# retraction that was wrong:
#   - all 16 Snitch traces frozen, AND the host trace (hart 0, trace_chip_00_hart_00000.LOG,
#     not .dasm) frozen in both size and simulated timestamp across 50 s
#   - VCS still burning CPU throughout, so the simulator is alive and the design is not
#   - the host's last retired instruction is `c.lw a5, 0(a4)` at 0x80000cb2, a5=0x...fc94,
#     a4=0x8000e978 -- a load from L3 that never returns
#   - every cluster core reached MGR_WRITE_DONE first: the compute finished, all 12 checks
#     up to the O checks PASS, and it dies in the first O check's tolerance scan
#
# A load that never returns is not an unmapped address (that returns 0xBADCAB1E) and not a
# slow check (the check is bounded, and shrinking it 8x changed nothing). Something in the
# fabric never answers, with every cluster engine already retired -- the teardown-wedge family
# this platform has seen before. Diagnose that, not the score buffer.
#
# WHY IT IS STILL WORTH FIXING: the SIMD is a large component of the array's idle, and a third
# buffer lets two QKs cover one softmax where two buffers give only one -- which the softmax
# overruns. Whether that still holds depends on the cluster hardware; re-measure before
# spending a run on it.
NSCORE = 2

# DIAGNOSTIC. The WAR depth, normally == NSCORE. Setting it BELOW NSCORE keeps the extra
# score/probability buffers but restores the tighter dependency of a smaller NSCORE, which
# separates two explanations of the NKV=64 failure that are otherwise confounded:
#   passes with NSCORE_WAR < NSCORE -> the buffer INDEXING is fine and the failure is a race
#                                      opened by the extra scheduling freedom
#   fails   with NSCORE_WAR < NSCORE -> the failure is in the 3-buffer indexing itself
# It is never correct to set it ABOVE NSCORE.
NSCORE_WAR = NSCORE


# FP16's most negative finite value, packed twice into 32 bits (no single BYTE repeats into
# 0xFBFF). It is the neutral element for a max and the value the arena seeds `m` with. Module
# scope because two separate blocks in _build_head need it: the m seed and the s16 prefix fill.
NEG_INF16 = 0xFBFFFBFF





# K buffers in the load->QK rotation. TWO is not a tuning choice -- it is what L1 fits.
#
# MEASURED, the case FOR a third: the 2-deep WAR edge idles the iDMA on EVERY cluster before
# the last K tile, with the engine free and only the buffer missing. K traffic overlapping the
# GEMM core's argument reads costs more again -- that core stalls heavily per instruction while
# the iDMA streams, and most long prologues overlap the iDMA rather than the xDMA.
#
# MEASURED, the case AGAINST: a third buffer FAILS on capacity. Named L1 grows past what the
# heap has once the runtime's per-task scratchpad/args arena and a per-cluster copy of the
# SoC-wide task list are counted -- neither of which appears in the budget line below.
# Allocation is name-sorted, so the arena lands last and runs off the end: a core then reads
# its own args as X and trips RegWriteKnown. The prologue term DID improve as predicted, so
# the idea is right and the capacity is not there.
#
# To revisit: free L1 elsewhere first (NSCORE, or Bc), then a third K buffer is worth having.
NKBUF = 2

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
# The wall is specific and it is not the only pipe. The quadrant has THREE independent wide
# paths, and this workload puts all of K+V on one of them:
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
# ARITHMETIC. This kernel's intensity is fixed by the tile shape. ONE array against one wide
# port balances below it, so a single cluster is compute-bound with margin -- which is why the
# one-cluster case runs so much closer to peak. FOUR arrays against the SAME port balance
# above it, so the multi-cluster case is memory-bound by exactly the cluster-count factor, and
# that factor IS the drop. Two pipes put the balance back at the workload's own intensity.
#
# COST: all host kernels run on one cluster's host core, so the V pushes serialise there. That
# is the correct shape anyway -- there is one system iDMA -- and the push fits alongside the
# pulls rather than extending them.
# WHICH V TILES RIDE THE PUSH. Measured, both extremes. Put every tile on the cluster xDMAs
# and four engines run concurrently at a modest rate each, but they overlap, so the effective
# aggregate is good. Put every tile on the host push and each transfer is several times
# faster -- the two-pipe model was right about bandwidth -- but there is ONE issuer, so every
# transfer also pays a manager round trip in series, and the effective aggregate is WORSE.
#
# One serial issuer loses to four concurrent ones even on a faster pipe, because the
# per-transfer dispatch overhead is paid in series. The answer is not to pick a pipe but to
# USE BOTH: split the traffic in proportion to each path's EFFECTIVE rate so the two finish at
# the same time, which is worth a large fraction of the fabric time either alone costs.
#
# V(0) deliberately stays on the PULL: it is the tile PV(0) needs first, and the pull path is
# the concurrent one, so it arrives soonest there. Tiles 1..3 go on the push.
# MEASURED, three arms, and the trend is monotonic -- every tile moved to the push COSTS, and
# ARRAY BUSY NEVER MOVED across them. The entire effect is pipeline stretch, i.e. the array
# waiting longer for V. The push pipe is REAL -- per transfer it is several times a contended
# pull share -- but there is exactly ONE issuer, and each task on it pays a manager round trip
# in series. Across a whole KV axis that dispatch cost alone drags the push's effective rate
# below the cluster xDMAs' aggregate.
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
# dispatch while the array runs. Measured, total load-engine busy EXCEEDS the pipeline, so a
# substantial part of the loading already hides in the fill. Each extra buffer moves one more
# tile per cluster out of the pipeline and into it.
#
# V(j) reuses v8[j % NVBUF], which PV(j - NVBUF) was the last to read, so the WAR edge walks
# back with the depth. L1 is the cap: at the shipped depth the named allocations already use
# most of the heap, and a V tile is large enough that exactly one more buffer fits. Two more
# would overflow.
#
# MEASURED: 3 is WORSE than 2. The extra buffer did exactly what it was supposed to -- the
# array's wait on V fell -- but the pipeline did not shrink, because the port is already
# saturated at close to its full width. Prefetching earlier does not create bandwidth; it
# only moves which engine is waiting, and here the SIMD and dispatch absorbed the whole gain.
#
# This is the same wall the V-push arms hit. When the bottleneck is aggregate bandwidth,
# RESCHEDULING CANNOT HELP -- only less traffic or more bandwidth can.
# NVBUF = 3 IS A WASH, measured twice. A third V buffer is a real streaming win -- it is the
# best ideal/pipeline any arm has produced -- and the end-to-end window does not move, because
# the head grows by what the pipeline saves: one more V tile is loaded before the pipeline
# starts. Buffering depth at this working point transfers work between head and pipeline
# rather than removing any. Keep 2; it also costs 64 KB of L1.
#
# Retry only if the head is ever shortened enough that its V loads stop being critical.
#
# Two lessons, each of which cost a run: an attribution is not a cause (the iDMA being busy
# across the array's idle does not mean the array waits for it), and "X did not help" is not
# the same finding as "X did nothing".
NVBUF = 2

# STAGGER THE FIRST REAL V TILE ACROSS CLUSTERS.
#
# The Gantt shows every cluster entering one long engine-config span at the SAME moment, each
# costing several times what the same configuration costs on a single cluster. Nothing is
# transferring during it (measured: zero overlap with any transfer), and the span retires very
# few instructions for its length -- that is cold instruction fetch, not engine backpressure.
#
# It is the FIRST REAL V TILE's config: the xDMA's memset path is warmed by the arena fills,
# its 1d_copy path is not (see the WarmGemm comment -- a warm-up COPY hangs, undiagnosed).
# So four cold fetches of the same code land on the fabric together.
#
# Chaining the first V load from one cluster to the next serialises those cold fetches instead.
# If the model is right, each then costs what it costs uncontended and the band shrinks. The
# price is that the last cluster starts its V stream later; whether that trade is positive is
# exactly what the measurement decides.
#
# MEASURED AND DISPROVEN. The mechanism failed too, not just the outcome -- the SUM of the
# stalls barely moved while the band they occupy GREW, and the last cluster's own stall got
# worse. Serialising the cold fetches did not make any of them cheaper.
#
# So the stall is NOT the four clusters colliding with each other. Their total cost is
# invariant to when they run, which points at contention with the K/V TRANSFERS that stream
# throughout -- the config code is fetched from L3 while the fabric is busy carrying tiles,
# and that traffic is identical in both arms.
#
# That relocates the fix: warm the xdma_1d_copy path or make its config resident, rather than
# rescheduling clusters. The source already notes a warm-up COPY hangs, undiagnosed -- that
# is the thing to diagnose.
STAGGER_FIRST_V = False

# PUSH V IN PAIRS ON THE SYSTEM iDMA, one BINGO task per PAIR.
#
# Two measured facts make this the shape to try:
#
#  1. A host iDMA transfer moves 64 KiB at 56.4 B/cc -- three times a contended pull share,
#     because it rides quadrant_wide_in, a pipe disjoint from the clusters' pull. But each
#     task on the host costs a manager round trip, so ONE tile per task leaves the push
#     losing on effective rate. Batching more tiles into a task raises it.
#
#  2. The all-push arm was worse than the model predicted, because V(j) waits on
#     PV(j-NVBUF): the V chain serialises THROUGH
#     THE COMPUTE, not just through the issuer. Pairing attacks that too -- {V2,V3} issue
#     together after PV(1) instead of V(3) waiting behind PV(2).
#
# PAIRS, NOT QUADS, DELIBERATELY. V(j) and V(j+1) live in DIFFERENT buffers (NVBUF=2), so a
# pair is expressible inside one cluster. Batching four would have to cross clusters, which
# couples their dependencies -- every cluster's batch would wait on the slowest -- and that
# is exactly the coupling that made a wider split arm imbalanced across clusters. Giving up
# some effective rate to keep the shards independent is the right trade.
#
# MEASURED: batching beats one tile per task and is still far below leaving V on the cluster
# xDMAs. It did exactly what it was designed to do -- the pair is genuinely overlapped rather
# than serialised, so the rate WHILE RUNNING exceeds the port width -- and it was not enough,
# because the push tasks are NOT issue-bound. The issuer sits idle for more than half the
# window, waiting on the WAR edge: V(j) cannot start until PV(j-NVBUF) has freed its buffer.
# Deeper buffering is the only thing that relaxes that, and L1 caps it at a depth that
# measured worse on its own.
#
# FIVE ARMS, one conclusion: 4 concurrent cluster xDMAs beat one host issuer every time.
# Leave V on the pull.
V_PUSH_PAIRS = False

# PUSH K TO ALL FOUR CLUSTERS ON THE SYSTEM iDMA -- the second wide pipe, used for the one
# operand that is identical on every cluster.
#
# WHY THIS AND NOT THE FIVE V-PUSH ARMS ABOVE. Those all moved V, and V is the operand whose
# WAR edge is tightest: V(j) cannot start until PV(j-NVBUF) has freed its buffer, so the
# single host issuer sat idle more than half the window waiting on compute. Batching helped
# the issue rate and changed nothing about the wait.
#
# K under head-parallelism is a different shape. The four clusters want the SAME BYTES, so
# one task with four destinations delivers a whole tile-step -- the batch of four that
# measures 51.6 B/cc -- and it is the batch size the struct already caps at. And because the
# push writes every cluster's k8 directly, the cross-cluster RAW edge that the pull needs
# (three clusters waiting on the owner's copy) disappears: K stops being a two-hop operand.
#
# WHAT IT COSTS. The same coupling the broadcast had: one producer feeding four consumers
# must wait for every cluster's WAR edge, so the slowest cluster gates the refill. The
# existing broadcast WAR machinery states those edges, and this reuses it unchanged.
#
# WHAT IT BUYS ON THE FABRIC. K then rides quadrant_wide_in (a PUSH from the system iDMA)
# while V rides quadrant_wide_out (the clusters' own PULL). Those are disjoint 512-bit pipes;
# today the push pipe is idle and every byte of both operands queues on the pull.
#
# MEASURED, AND IT DOES EXACTLY WHAT IT WAS DESIGNED TO DO AND STILL LOSES. The cluster DMA
# engines' busy time roughly HALVES -- the push really is carrying the traffic -- and the run
# gets 3% longer anyway, because QK and softmax both wait longer. The delivery moved off the
# engines and onto the critical path. Keep it False.
K_PUSH_ALL = False

# PUT ONLY THE MAIN-MEMORY FETCH ON THE SYSTEM iDMA, and leave the cross-cluster pulls where
# they are.
#
# K_PUSH_ALL above delivers a tile to all four clusters from one host task, which is one
# producer feeding four consumers: the task waits for every cluster's WAR edge and every
# cluster then waits for it. MEASURED, that halves the cluster DMA engines' busy time and
# still loses, because the coupling puts the delivery back on the critical path -- QK and
# softmax both wait longer than they did on the pull.
#
# These knobs make the SAME move without the coupling. Under rotation exactly one cluster
# reads main memory for each tile; that one transfer becomes a host push with a SINGLE
# destination, so it takes only the owner's own WAR edge, exactly as the owner's own load
# did. The other three clusters keep pulling from the owner's L1 on their own engines.
#
# What it buys is a clean split of the two wide pipes: every byte that touches main memory
# then rides quadrant_wide_in (the system iDMA's PUSH) and every byte that does not rides
# quadrant_wide_out (the clusters' PULL). Today both queue on the pull pipe while the push
# pipe sits idle.
#
# Only meaningful with the matching *_PULL_FROM_CL0 knob on -- they replace the owner's
# load inside the pull scheme.
#
# MEASURED AND IT ALSO LOSES, by about two points. Removing the coupling was not enough: the
# owner's fetch is the HEAD of every tile's chain, so putting it behind ~433 cc of host
# dispatch delays all three pullers as well as the owner. The cluster engines barely got any
# freer, because only one transfer in four moved.
#
# K_PUSH_OWNER WITHOUT V_PUSH_OWNER HANGS. With K pushed and V left on the cluster iDMA, a
# puller's xDMA spins forever on its completion CSR while every other hart is frozen; both
# on or both off is fine. Not diagnosed -- the arm loses on time regardless -- but do not
# enable one alone.
#
# TAKEN WITH THE FIVE V-PUSH ARMS ABOVE AND K_PUSH_ALL, that is seven measurements of one
# rule: four concurrent cluster engines beat one host issuer, and the reason is never
# bandwidth. The system iDMA is genuinely the faster pipe per transfer; what it cannot do is
# sit on a chain that four clusters wait on, behind a per-task dispatch cost. Work would have
# to be prefetched far enough ahead to hide that, and NKBUF/NVBUF are capped by L1.
K_PUSH_OWNER = False
V_PUSH_OWNER = False


def _load_params(param):
    """Derive the whole geometry from params.hjson and the array, in ONE place.

    Everything below -- the tile, the second matmul's shape, the operand bound, the
    golden -- is a function of (M, K, N) and the mesh. Deriving it here is what stops the
    kernel's idea of the tile and the descriptors' idea of it from drifting apart.
    """
    global M, K, N, NKV, CHECK_O, NQ, BC, BR, DHEAD, S2_M, S2_K, S2_N, QSHIFT, NKV_PER, NCL
    global DECOMP
    DECOMP = str(param.get("DECOMP", DECOMP))
    if DECOMP not in ("headpar", "kvsplit"):
        raise ValueError(f"DECOMP={DECOMP!r} must be 'headpar' or 'kvsplit'")
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

    if DECOMP == "headpar":
        # ONE K and ONE V for the whole group -- that IS the GQA relation, not a
        # simplification: the group's query heads share a KV head by construction. What
        # varies per cluster is Q, so every cluster still gets its own (m_c, l_c) and the
        # per-shard checks stay evidence that the shard ran. There is nothing to fold.
        #
        # The distinct-shard argument the kvsplit docstring makes does not apply here for
        # the opposite reason: no fold is being tested, so identical partials would prove
        # nothing either way. Distinct Q is what keeps the four checks independent.
        a_shared = (rng.randint(-128, 127, size=M * K * MESH_ROW * TILE_SIZE)
                       .astype(np.int8) >> QSHIFT)
        shards, b_list = [], []
        for c in range(NCL):
            b_c = (rng.randint(-128, 127, size=N * K * MESH_COL * TILE_SIZE)
                      .astype(np.int8) >> QSHIFT)
            b_list.append(b_c)
            shards.append(build_data(a=a_shared, b=b_c, v=v))
        m_c = np.stack([np.asarray(sh[3], dtype=np.float16) for sh in shards])
        l_c = np.stack([np.asarray(sh[4], dtype=np.float16) for sh in shards])
        # a_list holds the SAME array object NCL times so stage() can see, by identity,
        # that one staged copy serves every cluster.
        # O PER CLUSTER, not shards[0]'s. Under headpar every cluster gets its own b_c
        # (its own query head), so every cluster's O is different and one golden would
        # only ever have validated cluster 0.
        return [a_shared] * NCL, b_list, v, m_c, l_c, None, [sh[5] for sh in shards]

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
    # b is shared under kvsplit; the caller indexes per cluster either way.
    return a_list, [b] * NCL, v, m_c, l_c, merged, [sh[5] for sh in shards]


def stage(st, a, b, v, m, rowsum, o):
    """Hand every array to the staging helper and return the handles.

    WHERE these land is the platform's business, not this workload's: a config with a
    memory chiplet gets a mempool.bin, one without gets C arrays in the host image. See
    util/sim/common/bingo_data_staging.py -- addressing a memory chiplet that a config
    does not have reads unmapped memory rather than faulting, which surfaces as an
    arithmetic bug a long way from the cause.
    """
    # STAGE EACH DISTINCT ARRAY ONCE. Under headpar every cluster's K is the same object,
    # and staging it four times would put four copies in the image (or on the memory chip)
    # AND give the broadcast four different source addresses to read -- which is exactly
    # the redundancy the decomposition exists to remove. Keyed by object identity, so
    # "shared" is decided by build_shards() rather than restated here.
    def put_per_cluster(tag, arrays, ctype, conv):
        # An array every cluster shares keeps the BARE name; only a genuinely per-cluster
        # one takes the _c<n> suffix. That is not cosmetic: the staged names reach the
        # generated header, and keeping them stable is what makes a kvsplit build after
        # this change byte-identical to one before it.
        shared = len({id(x) for x in arrays}) == 1
        handles, by_id = [], {}
        for c in range(NCL):
            key = id(arrays[c])
            if key not in by_id:
                by_id[key] = st.put(tag if shared else f"{tag}_c{c}",
                                    ctype, conv(arrays[c]))
            handles.append(by_id[key])
        return handles

    return {
        "a": put_per_cluster("fa_k8", a, "int8_t",
                             lambda x: np.asarray(x).astype(np.int8)),
        # Q is shared under kvsplit and per-head under headpar, by the same rule.
        "b": put_per_cluster("fa_q8", b, "int8_t",
                             lambda x: np.asarray(x).astype(np.int8)),
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
        "o": [st.put(f"fa_o_golden_c{c}", "int32_t", np.asarray(o[c]).astype(np.int32))
              for c in range(NCL)],
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
    # main memory while the previous tile computes. Every tile is fed the same bytes -- the
    # golden relies on that, and m stops moving after tile 0 -- but they are RE-FETCHED per
    # tile, so the memory traffic is what a real KV stream would cost. Loading once and
    # replaying leaves the iDMA idle and makes any utilisation figure exclude memory.
    # TWO K buffers. A third would let the loads run one tile further ahead, which is the
    # right diagnosis -- every QK is gated by its own K load -- but it also raises the
    # number of simultaneously-ready DMA nodes past what the manager's ready set tolerates,
    # and the run wedges in teardown. Revisit only alongside that limit.
    k8 = [g.l1(f"fa_k8_{i}", M * K * MESH_ROW * TILE_SIZE) for i in range(NKBUF)]
    q8 = [g.l1(f"fa_q8_{q}", N * K * MESH_COL * TILE_SIZE)      # B of the score matmul
          for q in range(NQ)]
    v8 = [g.l1(f"fa_v8_{i}", S2_M * S2_K * MESH_ROW * TILE_SIZE) for i in range(NVBUF)]
    # The score matmul masks every C channel off (see gemm_fa.h), so nothing is ever
    # READ through this pointer -- the AGU walks addresses that are never dereferenced.
    # It exists only because the descriptor needs a base pointer. One block, not M*N.
    # fa_cz REMOVED. Every C channel of the score matmul is masked, so the AGU walks
    # addresses that are never dereferenced -- and the kernel already substitutes D_addr
    # when C_addr is 0 (the same path PV's first tile takes). A whole 1,024 B allocation
    # existed only to give that never-read pointer somewhere to point.
    cz = 0
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
    for key in ("a", "b", "m", "rowsum", "o"):
        h[key] = h_all[key][c]
    return h


def _build_head(dfg, c, h_all, buf):
    """Everything a cluster does BEFORE its first KV tile: warm-ups, Q, the arena fills.

    Split from the body so that under headpar EVERY cluster's head nodes are created before
    ANY broadcast node is. That ordering is what makes it safe to spread the broadcasts
    over all four xDMAs: a core dispatches in creation order, so a broadcast placed on
    cluster c's xDMA sits BEHIND c's own fills rather than in front of them. In front, a
    later broadcast would wait on a PV that waits on a softmax that waits on the very fills
    queued behind it -- a genuine cycle, not a slowdown.
    """
    g = G(dfg, c)
    h = _shard_handles(h_all, c)
    arena, k8, q8, v8 = buf["arena"], buf["k8"], buf["q8"], buf["v8"]
    cz, s16, p8, oacc = buf["cz"], buf["s16"], buf["p8"], buf["oacc"]
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

    return dict(warm_gemm=warm_gemm, warm=warm, ld_q=ld_q, ld_az=ld_az,
                perf=perf, load=load, xload=xload)


def _build_cluster(dfg, c, h_all, m_all, rowsum_all, buf, bcast=None, all_bufs=None):
    """The KV-tile pipeline and this cluster's own checks, on top of _build_head.

    Byte-for-byte the tuned one-cluster graph -- same double buffering, same load-chain
    order -- with three differences and no others: it runs NKV_PER tiles instead of NKV, it
    reads this cluster's operands, and it ends at the partial rather than at a check.

    `bcast` is None under kvsplit, where this cluster loads its own K and V. Under headpar
    it is {"k": [...], "v": [...]}: the MULTICAST nodes, already created by build(), that
    fill every cluster's k8/v8 from one read. This cluster then only depends on them.
    """
    g = G(dfg, c)
    h = _shard_handles(h_all, c)
    m, rowsum = m_all[c], rowsum_all[c]
    arena, k8, q8, v8 = buf["arena"], buf["k8"], buf["q8"], buf["v8"]
    cz, s16, p8, oacc = buf["cz"], buf["s16"], buf["p8"], buf["oacc"]

    # HEAD FIRST, IN THIS CLUSTER'S OWN PASS -- do not hoist it into build().
    #
    # MEASURED: building every cluster's head before any body changes the GLOBAL node
    # creation order, and that order IS the task-descriptor list the BINGO manager streams.
    # It cost several points of array utilisation on a graph that was otherwise identical --
    # same per-core kernel sequences in every cell, same allocations. Dispatch is
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
                            [(h_all["a"][0], bf["k8"][j % NKBUF]) for bf in all_bufs],
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
                    n = xload(f"K{j}", "a", k8[j % NKBUF], KBYTES, k_after)
                    if c == own:
                        bcast["ksrc"][j] = n
                elif c == own:
                    if K_PUSH_OWNER:
                        n = g.node(f"PushK{j}_c{c}", HOST_CORE,
                                   "__host_bingo_kernel_idma",
                                   HostBingoKernelIdmaArgs(h["a"], k8[j % NKBUF], KBYTES),
                                   k_after, cluster=0)
                    else:
                        n = xload(f"K{j}", "a", k8[j % NKBUF], KBYTES, k_after)
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
                # tolerates". Measured twice, from two different directions.
                if c == _bcast_owner(j):
                    k_after = ld_q if j < NKBUF else [qk[(j - NKBUF + 1) * NQ - 1]]
                    if j == 0 and bwarm_node is not None:
                        k_after = [bwarm_node]
                    bcast["k"][j] = g.node(
                        f"BcastK{j}", XDMA_CORE, "__snax_bingo_kernel_xdma_multicast",
                        SnaxBingoKernelXdmaMulticastArgs(
                            h_all["a"][0], [bf["k8"][j % NKBUF] for bf in all_bufs],
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
                ld_k.append(load(f"K{j}", "a", k8[j % NKBUF], KBYTES, k_after))
            if q == 0 and bcast is not None and BCAST_V:
                if c == _bcast_owner(j, BCAST_V_SKEW):
                    v_after = list(ld_az) if j < NVBUF \
                        else [pv[(j - NVBUF + 1) * NQ - 1]]
                    bcast["v"][j] = g.node(
                        f"BcastV{j}", XDMA_CORE, "__snax_bingo_kernel_xdma_multicast",
                        SnaxBingoKernelXdmaMulticastArgs(
                            h_all["v"], [bf["v8"][j % NVBUF] for bf in all_bufs], VBYTES),
                        v_after)
                ld_v.append(bcast["v"].get(j))
            elif q == 0:
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
                elif bcast is not None and V_PULL_FROM_CL0:
                    # HEADPAR, V PULLED: cluster 0 reads main memory, the rest read
                    # cluster 0's L1. Every cluster still issues exactly one transfer per
                    # tile on its own iDMA, so no engine gains work -- only the SOURCE
                    # changes, and with it three quarters of the main-memory traffic.
                    own = _pull_owner(j)
                    if j < PULL_SKIP_FIRST:
                        # Same as K: the leading tiles have nothing to overlap their second
                        # hop, so every cluster reads them itself. V stays on the iDMA.
                        n = load(f"V{j}", "v", v8[j % NVBUF], VBYTES, v_after)
                        if c == own:
                            bcast["vsrc"][j] = n
                    elif c == own:
                        if V_PUSH_OWNER:
                            n = g.node(f"PushV{j}_c{c}", HOST_CORE,
                                       "__host_bingo_kernel_idma",
                                       HostBingoKernelIdmaArgs(h["v"], v8[j % NVBUF], VBYTES),
                                       v_after, cluster=0)
                        else:
                            n = load(f"V{j}", "v", v8[j % NVBUF], VBYTES, v_after)
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
                    # MEASURED. With V on the cluster xDMAs and K broadcast into them, ALL
                    # FOUR xDMA harts wedged in their wait loops (three at 1.8 GB of trace).
                    # The arm where every cluster xDMA did nothing but five tiny memsets
                    # while the broadcasts landed is the one that PASSED. Under headpar the
                    # iDMA is free anyway -- K no longer rides it, it carries only Q -- so
                    # this costs nothing and keeps the receiving xDMAs clear.
                    ld_v.append(load(f"V{j}", "v", v8[j % NVBUF], VBYTES, v_after))
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
                                                         perf_addr=perf),
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
                                            SnaxBingoKernelSimdFaSoftmaxArgs.GEOM_CSR_PRIMED)),
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
    if DEBUG_LOOSE_ML:
        # DEBUG ONLY. m and rowsum are the FIRST checks dispatched, and a failing host
        # kernel breaks the scheduler loop (bingo_api.c:906), so one bad m costs the other
        # eleven results. Widening these two lets the run reach the O check, which is the
        # one that says whether the COMPUTATION is wrong or only the m/rowsum readback.
        tol_m = tol_s = 1.0e4

    if DEBUG_PER_TILE_M and c == 2:
        # One store+check per tile on the failing cluster only. Ordered sm[i] -> store ->
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
        # THE WHOLE score tile, not one beat. Comparing 64 B of 32,832 B says nothing about
        # the 511 beats the rowmax also reads, and the rowmax is what feeds m.
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

    # CHECK O. Without this the suite validates m and rowsum only, and both are per-tile
    # quantities that this workload makes IDENTICAL on every tile -- so neither covers the
    # accumulation across KV tiles, which is the entire point of FlashAttention. O is the
    # only checked value that depends on every tile having been folded in correctly.
    #
    # It is also what makes a layout experiment trustworthy. The static-L1 work found a
    # write that lands past the end of fa_v8_1, and which buffer it damages depends on what
    # the packer put next: with fa_s16_0 there, every softmax row maximum is wrong and the
    # suite says so; with fa_oacc32_0 there, nothing looked at the damage. A green run whose
    # checks cannot see the failure mode under test is not evidence, and adding this check
    # is cheaper than reasoning about which layouts happen to be observable.
    o_bytes = BR * DHEAD * 4
    l3_o = BingoMemAlloc(f"out_fa_o_c{c}", size=o_bytes, mem_level="L3")
    # WAITS ON THE LAST PV, NOT ON `last`. `last` is the last SOFTMAX, and every other store
    # here reads something a softmax wrote (m lives in the arena, rowsum in p8) -- but O is
    # written by the last PV, which is a SIBLING of this node, not an ancestor: PV(TOTAL-1) is
    # emitted in the loop's +1 iteration and depends on sm[TOTAL-1] exactly as this store did.
    # So with `last` the host iDMA was free to read fa_oacc32_0 while the VersaCore was still
    # accumulating into it.
    #
    # It is a RAW hazard, so the visible failure is not a wrong number -- the run HANGS, with
    # the host parked and the UART stopping after the eighth check (all four m, all four
    # rowsum) because the first Store_o/Check_o pair is where the host iDMA first overlaps a
    # live VersaCore write to the same TCDM.
    #
    # It was latent at NSCORE=2: SM(i) takes a WAR edge on PV(i-NSCORE), so a smaller NSCORE
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


def build(dfg, h, m_all, rowsum_all, merged_h, jct_monoid):
    """Four cluster pipelines, split either over KV or over the GQA group's query heads.

    Under DECOMP = "headpar" (the default) the four clusters hold four query heads of one
    GQA group. They share a KV head by construction, so each KV tile is read from main
    memory ONCE and multicast into all four L1s, and each cluster's (m, l, O) is already a
    complete result -- there is nothing to fold. That is the configuration that keeps the
    quadrant compute-bound; see the DECOMP comment at the top of this file.

    Under DECOMP = "kvsplit" the rest of this docstring applies.

    The shards are independent: each is the tuned single-cluster pipeline over its own KV
    tiles, producing its own (m_c, l_c). What makes this workload different from four
    copies of the one-cluster run is the epilogue -- the online-softmax merge

        m* = max_c m_c        l* = sum_c exp(m_c - m*) * l_c

    is not gathered to one cluster and reduced there. It is computed BY THE FABRIC: each
    cluster packs its partial into the monoid junction's lane geometry, and one
    ChainGather walks the four of them, folding at each hop, so the collector's buffer
    receives the answer rather than the operands.
    """
    # Every cluster's L1 first, with no nodes: a broadcast names destinations in all four,
    # so the handles have to exist before the first load node does.
    bufs = [_alloc_cluster(dfg, c) for c in range(NCL)]

    if DECOMP == "headpar":
        # ---- one read per tile, fanned out in the writer --------------------------------
        # ALL BROADCASTS ON ONE ENGINE (cluster 0). Spreading them over the four clusters'
        # xDMAs DEADLOCKS the fabric -- measured, and root-caused on a waveform to the
        # adapter's single from_remote context: all four clusters' receive windows open
        # inside 1.3 us and none ever closes, so every finish manager sticks in ReadBusy.
        # See docs/xdma_per_source_remote_contexts.md. BCAST_SPREAD reproduces it on
        # purpose; never enable it for a measurement.
        #
        # The nodes themselves are created inside cluster 0's own _build_cluster pass, so
        # the global node order stays what it was -- see the note there for what hoisting
        # them cost.
        # "vsrc" carries cluster 0's V load nodes so the other three can depend on them,
        # the same deferral the broadcast nodes already use.
        bcast = {"k": {}, "v": {}, "vsrc": {}, "vpull": [],
                 "ksrc": {}, "kpull": []}
        shards = [_build_cluster(dfg, c, h, m_all, rowsum_all, bufs[c], bcast, bufs)
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
        # It used to be stated as `bcast["ksrc"][j + NBUF]`, and that is wrong twice over
        # once PULL_ROTATE is on. `ksrc[t]` is the node on cluster t % NCL, so with NCL=4
        # and NKBUF=2 it names cluster (j+2) % 4 -- a DIFFERENT cluster, whose k8 is a
        # different physical buffer. And the node that really overwrites own(j)'s slot is
        # frequently a PULL (own(j+2) != own(j)), which is recorded in bcast["kpull"] and
        # never appears in ksrc at all, so the loop could not have found it either way.
        #
        # The effect is a source buffer overwritten mid-pull: the puller gets a later
        # tile's K, its scores are wrong, and the running max m comes out far too large.
        # It is latent at NSCORE=2 and at four tiles per cluster -- the schedule has no
        # slack to open the window -- and fires every time at NSCORE=3 with sixteen.
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
        # Each shard's stores only needed its OWN cluster's compute, so with four clusters
        # skewed -- which they are, increasingly so with more tiles -- Store_o_c2 would issue
        # a 16 KB host iDMA read out of cluster 2's L1 while clusters 0/1/3 were still
        # streaming K and V through the same fabric. That combination wedges the machine:
        # host and all sixteen snitch traces freeze together and VCS keeps burning CPU. It is
        # the RTL fragility in docs/soc_bottlenecks.md section 9, and this is the SW way
        # around it.
        #
        # It costs nothing measurable. Every utilisation figure here ends its window at the
        # LAST PV, so the host tail sits outside the measurement; all this does is stop the
        # tail from overlapping live traffic.
        last_pvs = [sh["pv"][-1] for sh in shards if sh["pv"]]
        for sh in shards:
            for lp in last_pvs:
                if lp is not sh["pv"][-1]:
                    dfg.bingo_add_edge(lp, sh["first_store"])

        # No fold: under head-parallelism each cluster's (m, l, O) is already the complete
        # answer for its own query head. The per-shard checks are the whole verification.
        return

    shards = [_build_cluster(dfg, c, h, m_all, rowsum_all, bufs[c])
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
    a, b, v, m, rowsum, merged, o = build_shards()
    st = DataStaging(platform)
    h = stage(st, a, b, v, m, rowsum, o)
    # headpar has nothing to fold, so there is no merged golden to stage.
    merged_h = stage_merged(st, merged) if merged is not None else None
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
    # The junction index is only meaningful for the fold. headpar has no fold, so it must
    # not demand a cfg feature it never uses -- a head-parallel quadrant is a legitimate
    # target on hardware built without the monoid junction.
    jct = writer_junction_index(hw, "HasMonoidJunction") if DECOMP == "kvsplit" else None
    build(dfg, h, m, rowsum, merged_h, jct)

    os.makedirs(args.output_dir, exist_ok=True)
    dfg.bingo_compile_dfg(
        "FlashAttention, GQA head-parallel over 4 clusters with a broadcast KV stream"
        if DECOMP == "headpar" else
        "FlashAttention, KV-sharded over 4 clusters with an in-fabric merge",
        args.output_dir, args.output_offload_file_name,
        extra_include_header_list=["fa_data.h"],
        static_l1=args.static_l1,
        desc_list_in_narrow_spm=not args.desc_list_in_wide_spm)
    # Every buffer _build_cluster allocates, with its real multiplicity. The previous
    # version of this line counted ONE K buffer, ONE V buffer and a hardcoded two score
    # buffers, and did not scale with NQ -- so it under-reported by ~70 kB and would not
    # have moved at all when a buffer was added. A budget line that cannot go up is worse
    # than none, because the L1 heap is the thing that silently bounds this workload.
    l1 = (NKBUF * (M * K * MESH_ROW * TILE_SIZE)          # k8, NKBUF-deep
          + NVBUF * (S2_M * S2_K * MESH_ROW * TILE_SIZE)  # v8, NVBUF-deep
          + NQ * (N * K * MESH_COL * TILE_SIZE)           # q8, one per query tile
          + 0                                             # cz removed
          + NSCORE * (64 + BC * BR * 2)                   # s16
          + NSCORE * (BC * BR + 64)                       # p8
          + NQ * (BR * DHEAD * 4)                         # oacc
          + NQ * SnaxBingoKernelSimdFaSoftmaxArgs.arena_bytes(BC, DHEAD)
          # headpar folds nothing, so it allocates no partial.
          + (0 if DECOMP == "headpar"
             else SnaxBingoKernelPackFaPartialArgs.packed_bytes(BR, MONOID_SLOTS)))
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
          f"qshift={QSHIFT}, NSCORE={NSCORE}, L1 buffers {l1:,} B of 514,816 B "
          f"({100 * l1 / 514816:.0f}%)")
    print(f"  main-memory reads {traffic:,} B for {mac:,} MAC "
          f"= {mac / traffic:.0f} MAC/byte (one 512-bit port balances at "
          f"{NCL * MESH_ROW * MESH_COL * TILE_SIZE // 64} MAC/byte)")


if __name__ == "__main__":
    main()
