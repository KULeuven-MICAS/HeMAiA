# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Llama-3-8B's grouped-query attention for ONE decode token at position ctx over a ctx-position KV cache, the attention
SPLIT BY KEY over every cluster of the 16 + 4 chiplet platform (the FA core of fa/common/fa_llama_mc.py: 32 query heads as
two halves of 16 SIMD lanes, a block-diagonal query, K / V tiles per half, the softmax statistics met in the fabric, the
partials reduced by a ChainGather), every projection on VersaCore's one-token GEMV. The platform, the rings, the
stamps, the broadcast and the checks are chiplet_mc.py's; the projection, W_O and append helpers llama3_decode_mc.py's
(DBuild); the data LlamaDecodeData's (a fresh causal prompt of ctx + 1 positions, the decode token the last), the
decode token's attention replaced by the key split's arithmetic (KsDecodeData below).

THE SPLIT. The cache holds ctx / 64 tiles of 64 keys; chip i of the chip list holds tiles i NT .. i NT + NT - 1 (NT = ctx
/ (64 x chips)) and its cluster c takes half c of them: the 16 query lanes of heads 16 c .. 16 c + 15 over the key /
value groups 4 c .. 4 c + 3. A half's K tile is its four groups' keys side by side, [64, 512] (A layout, the q / k
columns in rope_pairs' chunk order), its V tile V^T [512, 64] (A layout); the query half is block diagonal, [16, 512]:
lane i is head 16 c + i on its group's 128 columns, zero on the others. QK is [64, 512] x [512, 16], PV [512, 64] x [64,
16], the shapes of the FA core. Each chip holds both halves of its keys, so the chip's two clusters' partials are its
partial side by side (O_0 | O_1 | l_0 | l_1): the chip fold is a concatenation, not a sum.

THE NEW KEY (position ctx) is a tile of NR = 16 keys on the clusters of the chip that owns tile ctx / 64 in the round
robin (chip (ctx / 64 / NT) mod 16, or params new_chip): the token's key in row 0, rows 1..15 zero and masked by the QK's
INT32 C (mask_c: MASKED_SCORE at the D port's scale). Its K tile is [16, 512] (8 KiB, A layout: a group's chip half one
contiguous 1 KiB run), its V tile V^T [512, 16] (8 KiB, A layout: a group's chip half one contiguous 1 KiB run); each
chip's clusters put their runs there behind a flag each.

THE STEPS, per cluster unless named:
    x          the token's INT8 row in a_row (value c at (c // 4) 8 + c % 4, the upper 4 bytes of each 8-byte word
               zero; 8 KiB) from the HBM of the root's memory chip: the root's host has that memory chip's push engine
               multicast it into every chip's L3 (<TAG>_xland) behind one flag (<TAG>_xtflag); every cluster fetches it
               into L1
    q          4 GEMVs (W_q's 32-column chunks of the cluster's head: head 4 m + 2 hh + c on chip (m, hh) of group m's
               chip pair, the prefill's assignment), each dequantised, RoPE of each chunk pair at the token's position.
               Cluster 1 copies its row next to cluster 0's; cluster 0 interleaves the two rows' 8-byte atoms (4 values)
               on the xDMA and quantises them: the chip's 256-B piece, one 8-byte word a 64-B block of the query's A
               layout (its two lanes). Cluster 0 puts it into the root's staging (symmetric buffer) behind a flag
    q halves   the root waits for the 16 pieces and scatters them on the xDMA into the two block-diagonal halves (zero
               elsewhere: the symmetric buffer is zeroed before the window); one multicast a half into its cluster
               index (half 0 into every cluster 0, half 1 into every cluster 1), the second behind the first
    cache      each chip's memory chip pushes the chip's shard (K tiles of both halves, then V tiles: two parts behind a
               flag each) into its L3 between the W_q chunks and the rest of its weights (the prefetcher split around
               the push); each cluster copies its half's tiles into L1
    pass 1     per K tile S^T = K Q^T (gemm_fa_qk) and the FA softmax: the shard's row maximum (passes 2), or the
               online softmax and PV (passes 1)
    k / v      2 GEMVs, the chip's half of its group's K (cluster 0: dequantised, RoPE, row 0 of a [16, 64] A tile,
               quantised: the 1 KiB run) or V (cluster 1: each chunk's row as row 0 of a zeroed [16, 32] FP16 block,
               transposed, reshaped into V^T's A layout, quantised: two 512-B runs); the run put into the new key's
               tile of the new key's chip, the cluster of its half, behind a flag
    new key    the new key's chip's cluster c waits for its 16 runs (8 K, 8 V), then its pass 1 over that tile
    stats      each chip's statistics on its cluster 0's symmetric buffer: stats max, the two halves' row maxima [m_0 |
               m_1] (64 B), one ChainGather through every chip with the Elementwise junction's FP16 MAX into the root:
               m*; stats merge, the halves' [m_0 | m_1 | l_0 | l_1] (128 B), one ChainGather with the softmax-statistics
               junction (m* = max m_c, l* = sum exp(a (m_c - m*)) l_c) into the root. m* multicast into every
               cluster
    pass 2     (passes 2) per V tile the softmax seeded at m* and PV, the last tile leaving FP16; (passes 1) O re-based
               onto m* by one more softmax step over FIX_ROWS rows of m* and a PV with zero values and the step's corr
               as C's column scale
    partial    PV's [512, 16] FP16 output compacted on the xDMA to the half's [128, 16] O^T (lane i keeps its group's
               rows); cluster 0's into its OFF_LOC, cluster 1's copied next to it, the row sums after them (stats max)
    reduce     one ChainGather through every chip with the Elementwise junction's FP16 ADD into the root: per half the
               root normalises (o~ = O c / l, l the reduced row sums or l*) and transposes: o~ [32, 128], one head a row
    o          the root's o~ row into a_row (xDMA, 8-byte atoms 16 bytes apart in FP16), quantised (s_ot): the W_O
               GEMV's A operand, multicast into every cluster (one multicast a cluster index, in turn)
    W_O        by output columns: 4 GEMVs of 32 of the cluster's 128 columns over the full depth; the cluster's columns of
               y put into the root's L3 row behind a flag; the root waits for all 32 and writes y (FP16, before W_O's
               column dequantisation) into the HBM of the root's memory chip
    append     the new key's chip's cluster 1, behind its y put, writes both halves' new K tiles (8 KiB each: the first
               16-key block of the 64-key tile ctx / 64) and V^T tiles (each 16-dimension block's first four 64-B blocks)
               into the cache image's next tile in its memory chip's HBM

THE STREAM. Each chip's host runs its weight prefetcher in three parts: W_q's 4 chunks of each ring, the cache push, the
k / v's 2 chunks, W_O's 4 chunks. params wo_push:
    after_reduce   (default) each chip's W_O part waits for the partials' ChainGather too: every chunk of the earlier
                   parts has landed before the statistics' ChainGather (the new key needs every chip's k / v chunks),
                   and no W_O chunk is pushed before the reduce has ended, so no ChainGather runs beside a push
    after_kv       W_O's part right behind the k / v part: its pushes run beside both ChainGathers
    before_chains  W_O's part right behind the k / v part, every cluster's W_O chunks loaded into its slabs (params
                   w_buffers at least W_O's chunks a cluster) right after its k / v run (and the new key's chip's run
                   wait), and each chip's statistics handed to the chain only once both its clusters have loaded their
                   last W_O chunk: every push of every part has landed before the statistics' ChainGather, so no
                   ChainGather runs beside a push, and the W_O GEMVs read their weights from L1 once o lands

THE COLLECTIVES. Every multicast is the root cluster 0's DM core's (the q halves, m*, o), each a dependency descendant
of the one before; the host's x multicast comes first; the two ChainGathers sit between them (q -> pass 1 -> stats
chain -> m* -> pass 2 -> reduce chain -> o). _check_collectives proves every pair ordered by a path of the graph, and
with wo_push after_reduce or before_chains every push part ordered against every ChainGather.

THE ORDERS. The compiler orders every device core's program (mini_compiler passes/bingo_dfg_schedule.py). Each DM core,
in the order the tasks are made: x, the W_q chunks,
the q piece, the q wait, the cache tiles, the k / v chunks, the run put, (the new key's chip) the run wait, the stats
copies, the m* wait and its copies, the partial copies, the first W_O chunks (the root's cluster 0: after the o
multicast), the o wait, the rest of W_O, the y put, (the root's cluster 0) the y wait and the HBM write, (the new key's
chip's cluster 1) the appends. wo_push before_chains loads every W_O chunk right before the stats copies instead.

PARAMETERS (params.hjson): ctx (cached positions, 64 x 16 x NT), rows and bc (the prompt's prefill passes and key tiles
for the golden's static scales, as llama3_decode_mc.py; bc 64 is also the cached tile), seed, wbits 4, w4_paired true,
passes (1 | 2), stats (max | merge), wo_push (after_reduce | after_kv | before_chains), new_chip, checks (y | stages),
async_exec, and the platform's and the graph's switches (w_chunk_bytes = one 32-column chunk, w_buffers,
weight_ring, static_l1, l1_capacity, compact_id_maps, gid_lists, per_chip_image, desc_in_image, d2d_ddr, d2d_rx_yield,
fast_host_init, args_phase_bytes). bcast bcast_l1, reduce chain and sync edge only.

IN A LAYER the block runs with shared=the layer, as llama3_decode_mc.py's: x_on_host false (the layer's input norm on
the root's cluster 0 makes x8 in a_row and multicasts it from L1 into every chip's L3 at <TAG>_xland behind <TAG>_xtflag;
x_multicast returns that put) and y_to_hbm false (y stays in the root's L3 row <TAG>_yrow; y_out returns the root's
wait). prefetch[k] is the chip's last prefetcher part (W_O), behind which a layer chains its next block's. append_now
(deps) makes the append where the block does (behind the new key's chip's cluster-1 y put); a layer that writes into
HBM only after all its collectives and pushes overrides it to return [] and calls kv_append(deps) at its end.
"""

import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# llama3_decode_mc puts util/sim/multichip, mini_compiler and ffn/common on sys.path (via llama3_data)
import llama3_decode_mc as lmd  # noqa: E402
from llama3_attn_mc import BR, QC, run_driver  # noqa: E402

import networkx as nx  # noqa: E402
from bingo_kernel_args import (  # noqa: E402
    HostBingoKernelHbmUcastArgs, HostBingoKernelWeightPrefetchRangeArgs, SnaxBingoKernelGemmFaPvArgs,
    SnaxBingoKernelGemmFaQkArgs, SnaxBingoKernelIdma2dCopyArgs, SnaxBingoKernelIdmaFetchFlaggedArgs,
    SnaxBingoKernelIdmaPutFlaggedArgs, SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelSimdFaSoftmaxArgs,
    SnaxBingoKernelSimdMlaNormaliseArgs, SnaxBingoKernelSimdMulF16Args, SnaxBingoKernelWaitFlagsArgs,
    SnaxBingoKernelXdma6dArgs,
    SnaxBingoKernelXdmaChainGatherArgs, SnaxBingoKernelXdmaTranspose2dArgs, XDMA_EWJCT_FMT_FP16, XDMA_EWJCT_OP_MAX,
    XDMA_SSJ_ROWS, simd_fp16_lanes, xdma_ewjct_csr0, xdma_softmax_stats_csrs)
from bingo_mem_handle import BingoMemFixedAddr  # noqa: E402
from libs import at_offset  # noqa: E402
from libs.comm.nest import convert_args  # noqa: E402

import chiplet_mc as cm  # noqa: E402

FA = SnaxBingoKernelSimdFaSoftmaxArgs
PVK = SnaxBingoKernelGemmFaPvArgs
BEAT = FA.beat()                       # one SIMD beat: BR FP16 lanes
XFER = max(64, BEAT)                   # the xDMA's and the D2D paths' smallest transfer
ARENA = FA.arena_bytes(64, 1)
LAY = FA.layout(64, 1)
MRUN, LRUN = LAY["mrun"], LAY["lrun"]
EW_F16_MAX = xdma_ewjct_csr0(XDMA_EWJCT_OP_MAX, XDMA_EWJCT_FMT_FP16)
ATOM, LANES = lmd.ATOM, lmd.LANES      # the xDMA's 8-byte atom, atoms a beat
BC = 64                                # keys a cached tile
NR = 16                                # keys of the new key's tile
NEW = -1                               # the new key's tile in a cluster's tile list
FIX_ROWS = 4                           # passes 1: the re-base step's tile of m*
NHF = 2                                # query halves, one a cluster index
QPB = 256                              # a chip's q piece: one 8-byte word a 64-B block of a group's query columns
if FA.layout(NR, 1) != LAY or FA.layout(FIX_ROWS, 1) != LAY:
    raise ValueError("the FA softmax arena's layout depends on its tile height")
if ARENA % XFER:
    raise ValueError(f"the FA softmax arena is {ARENA} B, not whole {XFER}-B units")


class KsDecodeData(lmd.LlamaDecodeData):
    """LlamaDecodeData with the decode token's attention as the key split runs it (key_split): the prompt, the weights,
    the cache and every static scale are LlamaDecodeData's; s_ot (the absolute maximum of o~ over every position) and y
    take the key split's o~ of the decode token.

    `plan`: tiles {(chip, c): [cached tile index or NEW]} (cluster c holds half c), order (the chips in the reduce's
    fold order: the ChainGather's data order, the root last), passes (1 | 2), merge (the softmax-statistics junction's
    fold), fix_rows."""

    def __init__(self, plan, **kw):
        self.plan, self.ks = plan, None
        super().__init__(**kw)

    def decode_tile(self, h):
        if self.ks is None:
            self.ks = self.key_split()
        t = np.zeros((BR, self.HD), dtype=np.float16)
        t[0] = self.ks["xt"][h]
        return t

    def half_rows(self, h):
        """Half h's K rows and V rows [T, D] (its groups' columns side by side, natural order) and its block-diagonal
        query [BR, D] (lane i: head BR h + i's q8 at the token, on its group's columns), INT64."""
        HD, gph = self.HD, self.NKV // NHF
        K = np.concatenate([self.k8[gph * h + j] for j in range(gph)], axis=1).astype(np.int64)
        V = np.concatenate([self.v8[gph * h + j] for j in range(gph)], axis=1).astype(np.int64)
        Q = np.zeros((BR, gph * HD), dtype=np.int64)
        for i in range(BR):
            j = i // self.GQ
            Q[i, j * HD:(j + 1) * HD] = self.q8[BR * h + i][self.ctx]
        return K, V, Q

    def compact(self, o16):
        """[D, BR] FP16 -> [HD, BR]: lane i's rows of its group."""
        out = np.empty((self.HD, BR), dtype=np.float16)
        for i in range(BR):
            j = i // self.GQ
            out[:, i] = o16[j * self.HD:(j + 1) * self.HD, i]
        return out

    def key_split(self):
        """The decode token's attention as the device runs the key split: per cluster its tiles of its half (pass 1,
        the statistics met, pass 2 or the re-base), the chips' partials [O_0 | O_1 (| l_0 | l_1)] added in FP16 in the
        fold order, each half normalised. -> mstar or ml (the root's statistics), loc {chip: partial}, tot, xt [NH, HD],
        stats {(chip, c): (m, l)}."""
        P = self.plan
        fp, hm, simd = self.fp, self.hm, self.simd
        F16, F32, F64 = np.float16, np.float32, np.float64
        lane_max = lambda a: simd.reduce_lanewise(a, simd.MAX)          # noqa: E731
        seed = np.full(BR, -65504.0, dtype=F16)
        a32 = F32(self.a_exp)
        ctx, passes, merge = self.ctx, P["passes"], P["merge"]
        halves = [self.half_rows(h) for h in range(NHF)]
        bias = np.zeros((NR, BR), dtype=np.int64)
        bias[1:] = self.mask_bias

        def tile(h, t):
            K, V, Q = halves[h]
            if t == NEW:
                Kt, Vt = np.zeros((NR, K.shape[1]), np.int64), np.zeros((NR, V.shape[1]), np.int64)
                Kt[0], Vt[0] = K[ctx], V[ctx]
                return fp.d_port(Kt @ Q.T + bias, self.k_s), Vt
            Kt, Vt = K[t * BC:(t + 1) * BC], V[t * BC:(t + 1) * BC]
            return fp.d_port(Kt @ Q.T, self.k_s), Vt

        def colscale(o, corr16):
            return np.clip(np.rint(o.astype(F64) * corr16.astype(F64)[None, :]), -2**31, 2**31 - 1).astype(np.int64)

        def sweep(h, tiles, m16, pv_on):
            l16, o = np.zeros(BR, dtype=F16), None
            for t in tiles:
                S16, V = tile(h, t)
                if not np.all(np.isfinite(S16)):
                    raise FloatingPointError("a decode score overflows FP16 after the D-port shift")
                mnew = lane_max(np.stack([lane_max(S16), m16]))
                corr16 = hm.exp16(fp.add16(m16, -mnew), a32)
                p16 = hm.exp16(fp.add16(S16, -mnew[None, :]), a32)
                rsum16 = simd.reduce_lanewise(p16, simd.ADD)
                l16 = simd.reduce_lanewise(np.stack([rsum16, fp.mul16(corr16, l16)]), simd.ADD)
                if pv_on:
                    pv = V.T @ fp.quant_i8(p16, hm.P8_SCALE).astype(np.int64)
                    o = pv if o is None else colscale(o, corr16) + pv
                m16 = mnew
            return m16, l16, o

        cl = sorted(P["tiles"])
        chips = sorted({k for k, _ in cl})
        st = {kc: sweep(kc[1], P["tiles"][kc], seed, passes == 1) for kc in cl}
        out = {"stats": {kc: st[kc][:2] for kc in cl}}
        if merge:
            # the softmax-statistics junction's arithmetic, bit for bit (fa/common/fa_llama_data.py ssj_fold)
            sys.path.insert(0, os.path.join(HERE, "..", "..", "fa", "common"))
            import fa_llama_data
            grp ={k: np.concatenate([st[k, h][0] for h in range(NHF)] + [st[k, h][1] for h in range(NHF)])
                   for k in chips}
            ml = fa_llama_data.ssj_fold([grp[k] for k in P["order"]], a32)
            out["ml"] = ml
            mstar = [ml[h * BR:(h + 1) * BR] for h in range(NHF)]
            lstar = [ml[XDMA_SSJ_ROWS + h * BR:XDMA_SSJ_ROWS + (h + 1) * BR] for h in range(NHF)]
        else:
            mstar = [lane_max(np.stack([lane_max(np.stack([st[k, h][0] for k in chips])), seed])) for h in range(NHF)]
            out["mstar"] = np.concatenate(mstar).astype(F16)
        part = {}
        for k, h in cl:
            if passes == 2:
                _, l16, o = sweep(h, P["tiles"][k, h], mstar[h], True)
            else:
                m_c, l_c, o = st[k, h]
                S = np.repeat(mstar[h][None, :], P["fix_rows"], axis=0)
                mnew = lane_max(np.stack([lane_max(S), m_c]))
                corr16 = hm.exp16(fp.add16(m_c, -mnew), a32)
                o = colscale(o, corr16)
                l16 = fp.mul16(l_c, corr16)
            part[k, h] = (self.compact(fp.d_port(o, self.k_o)), l16)
        loc = {k: np.concatenate([part[k, h][0].reshape(-1) for h in range(NHF)]
                                 + ([] if merge else [part[k, h][1] for h in range(NHF)])).astype(F16) for k in chips}
        order = P["order"]
        tot = loc[order[0]]
        for k in order[1:]:
            tot = fp.add16(tot, loc[k])
        n = self.HD * BR
        xt = np.empty((self.NH, self.HD), dtype=F16)
        for h in range(NHF):
            O = tot[h * n:(h + 1) * n].reshape(self.HD, BR)
            lt = lstar[h] if merge else tot[NHF * n + h * BR:NHF * n + (h + 1) * BR]
            t16 = simd.stream_map(lt, self.a_n, 0.0, simd.RSQRT)
            xt[h * BR:(h + 1) * BR] = fp.mul16(O, fp.mul16(t16, t16)[None, :]).T
        out.update(loc=loc, tot=tot, xt=xt)
        return out


class KsBuild(lmd.DBuild):
    TAG = "lk"
    x_on_host = True
    y_to_hbm = True

    def __init__(self, args, p, plat, hw, shared=None):
        # DBuild's builder is not run (its cache, its buffers and its sweep are the head-parallel mapping's); the helpers
        # taken from it and from ABuild (gemv, chunks, loads, rope_row, row_to_a, l3, x_multicast, y_out, _a_row,
        # _pack_b, kv_cc, l1_of, nd, mul, add, quant, xd, copy, copy2d, _load) read only what is set here
        cm.ChipletBuild.__init__(self, args, p, plat, hw, shared=shared)
        if self.T != 1:
            raise ValueError(f"params tokens={self.T}: the decode runs one token")
        if simd_fp16_lanes() != BR:
            raise ValueError(f"the SIMD has {simd_fp16_lanes()} FP16 lanes; the softmax takes one query a lane")
        if p.get("x_bcast", "memchip") != "memchip":
            raise ValueError(f"params x_bcast={p.get('x_bcast')!r}: x comes from HBM (memchip)")
        # THE ABLATION (workloads/fa/llama3_decode_full): params x_cast, bcast, reduce and sync take the paper's mechanisms
        # off one at a time, on the same mapping and buffers.
        #   x_cast   multicast: x by one multicast of the root's memory chip; unicast: a copy a chip in turn, the same
        #            landing area and flag (__host_bingo_kernel_hbm_ucast)
        #   bcast    bcast_l1: the q halves, m* and o multicast by the root; pull: every cluster reads the root's copy
        #   reduce   chain: the statistics and the partials folded along a ChainGather; root: the root pulls every chip's
        #            statistics (the halves' row maxima: with no junction stats merge falls back to max, the row sums
        #            re-based and reduced with O) and runs their max, then pulls every chip's partial into two L1 slots
        #            and adds it on the SIMD
        #   sync     edge: the cross-chip dependencies as the scheduler's edges; barrier: a point-to-point software chip
        #            barrier (on cluster 0's xDMA core) after every kernel -- x and q, the q gather and its broadcast,
        #            pass 1, the token's k / v, the new key and the statistics, their exchange and m*, pass 2, the reduce,
        #            o's normalise and broadcast, W_O and y -- fencing every core of the chip, no dependency edge across
        #            chips inside the window
        self.x_cast = str(p.get("x_cast", "multicast"))
        if self.x_cast not in ("multicast", "unicast"):
            raise ValueError(f"params x_cast={self.x_cast!r}: multicast or unicast")
        if self.bcast_impl not in ("bcast_l1", "pull"):
            raise ValueError(f"params bcast={self.bcast_impl!r}: bcast_l1 or pull")
        if self.x_cast == "unicast" and not self.x_on_host:
            raise ValueError("params x_cast unicast: x from the root memory chip's HBM (the block alone)")
        self.per_kernel = self.sync == "barrier"
        if self.per_kernel:
            if str(p.get("wo_push", "after_reduce")) == "after_kv":
                raise ValueError("params sync barrier: wo_push after_reduce (each chip's W_O part behind its barrier "
                                 "after the reduce) or before_chains (every W_O chunk loaded before the barrier after "
                                 "the statistics)")
            self.barrier_role = "xdma"
        self.w4_paired = bool(p.get("w4_paired", True))
        if not self.w4_paired or self.wbits != 4:
            raise ValueError("params w4_paired, wbits 4: the GEMV reads INT4 weights in the paired layout")
        self.restream, self.kv_split, self.rope_pairs, self.mask_c = False, True, "chunks", True
        self.checks_kind = str(p.get("checks", "y"))
        if self.checks_kind not in ("y", "stages"):
            raise ValueError(f"params checks={self.checks_kind!r}: y or stages")
        self.passes = int(p.get("passes", 2))
        if self.passes not in (1, 2):
            raise ValueError(f"params passes={self.passes}: 1 or 2")
        stats = str(p.get("stats", "max"))
        if stats not in ("max", "merge"):
            raise ValueError(f"params stats={stats!r}: max or merge")
        # the softmax-statistics junction folds the statistics only on a chain; a root has none and takes the maximum
        self.merge = stats == "merge" and self.reduce_impl == "chain"
        self.wo_push = str(p.get("wo_push", "after_reduce"))
        if self.wo_push not in ("after_reduce", "after_kv", "before_chains"):
            raise ValueError(f"params wo_push={self.wo_push!r}: after_reduce, after_kv or before_chains")
        # cache_order own: each chip's cache copy (kv_ucast) right behind its own W_q part. after_wq: also behind the
        # W_q parts of every other chip its memory chip feeds, so no cache copy runs on a memory chip while W_q still
        # streams from it to another chip (a running copy holds back the other chips' last W_q chunk until it ends).
        # sync edge: an edge from each of those W_q parts; sync barrier: the chip's barrier after x and q.
        # after_q: behind q in every cluster, so no cache copy runs while the root multicasts q either (traffic through
        # a memory chip waits while one of its long pushes runs). sync edge: a probe on the root after every cluster's
        # q; sync barrier: the chip's barrier after the q gather and its broadcast.
        self.cache_order = str(p.get("cache_order", "own"))
        if self.cache_order not in ("own", "after_wq", "after_q"):
            raise ValueError(f"params cache_order={self.cache_order!r}: own, after_wq or after_q")
        # hbm_checks direct: the root's host compares y and the appended K / V^T tiles where they are, in HBM.
        # stash: after the end stamp, one iDMA copy each on the root's cluster 0 brings them into the root's L3,
        # where the host compares them (the same bytes, read locally).
        self.hbm_checks = str(p.get("hbm_checks", "direct"))
        if self.hbm_checks not in ("direct", "stash"):
            raise ValueError(f"params hbm_checks={self.hbm_checks!r}: direct or stash")
        self.R, self.bc, self.ctx = int(p.get("rows", 64)), int(p.get("bc", 64)), int(p.get("ctx", 1024))
        if self.bc != BC:
            raise ValueError(f"params bc={self.bc}: the key split's cached tile is {BC} keys")
        nch = len(self.chips)
        if self.ctx % (BC * nch):
            raise ValueError(f"params ctx={self.ctx}: whole {BC}-key tiles, the same number on each of {nch} chips")
        self.NT = self.ctx // (BC * nch)                     # cached tiles a cluster
        # ---- the split: the prefill's key/value groups on chip pairs and heads on clusters (the projections) --------
        pairs = sorted({(k, k + 1) for k in self.chips if (k & 15) % 2 == 0 and k + 1 in self.chips})
        c = lmd.llama3_data.LLAMA3_8B
        if len(pairs) != c["kv_heads"] or nch * self.NC != c["heads"]:
            raise ValueError(f"{len(pairs)} chip pairs, {nch * self.NC} clusters: the attention has {c['kv_heads']} "
                             f"key/value groups and {c['heads']} query heads")
        self.pairs = pairs
        self.group, self.head, self.kv_half, self.kv_mate = {}, {}, {}, {}
        for m, (ka, kb) in enumerate(pairs):
            self.kv_half[ka], self.kv_half[kb] = 0, 1
            self.kv_mate[ka], self.kv_mate[kb] = kb, ka
            for n, (k, cc) in enumerate(((ka, 0), (ka, 1), (kb, 0), (kb, 1))):
                self.group[k] = m
                self.head[self.chips.index(k) * self.NC + cc] = m * (c["heads"] // c["kv_heads"]) + n
        self.kind = {g: "K" if self.c(g) == 0 else "V" for g in range(self.G)}
        # ---- the shards: chip i tiles i NT .., cluster c half c; the new key on the owner of tile ctx / BC ------------
        self.tile_of = {k: list(range(i * self.NT, (i + 1) * self.NT)) for i, k in enumerate(self.chips)}
        nc = p.get("new_chip")
        self.NEWCHIP = (int(nc, 16) if isinstance(nc, str) else int(nc)) if nc is not None else \
            self.chips[(self.ctx // BC // self.NT) % nch]
        if self.NEWCHIP not in self.chips:
            raise ValueError(f"params new_chip={nc}: not a compute chip")
        self.tiles = {(k, cc): self.tile_of[k] + ([NEW] if k == self.NEWCHIP else [])
                      for k in self.chips for cc in range(self.NC)}
        plan = dict(tiles=self.tiles, order=self.fold_order(), passes=self.passes, merge=self.merge, fix_rows=FIX_ROWS)
        d = self.d = KsDecodeData(plan, ctx=self.ctx, rows=self.R, bc=self.bc,
                                  seed=int(p.get("seed", 1)), wbits=self.wbits, mask_c=True)
        self.gold = d.ks
        self.lay, self.mesh = d.layout, d.mesh
        self.HD, self.H = d.HD, d.H
        self.NQC = self.HD // QC
        self.NKVC = self.NQC // 2
        self.WO0 = self.NQC + self.NKVC
        self.WOC = self.H // self.G                          # y's columns a cluster
        self.NWS = self.WOC // 32
        self.GPH = d.NKV // NHF                              # groups a half
        self.D = self.GPH * self.HD                          # a half's K / V row
        first = np.arange(QC) * 2
        self.perm = np.concatenate([np.concatenate([t * 2 * QC + first, t * 2 * QC + first + 1])
                                    for t in range(self.NQC // 2)])
        self.slot = {}
        D = self.D
        self.KB, self.VB = BC * D, D * BC                    # a half's K tile, V tile
        self.NKB = NR * D                                    # a half's new-key K tile, V^T tile
        self.RUN = self.NKVC * QC * BR                       # a chip's run of a new-key tile: 1 KiB
        self.QH = BR * D                                     # a query half
        self.FB = D * BR * 2                                 # PV's FP16 output
        self.OH = self.HD * BR * 2                           # a half's compacted O^T
        self.OB = NHF * self.OH
        self.PBE = self.OB + (0 if self.merge else NHF * BEAT)   # a chip's partial
        self.P, self.PB = 2, self.NT * NHF * self.KB         # the cache push: K tiles, then V tiles (a part each)
        self.sym_zero = True
        for line in d.report(d.y16(d.y_raw)):
            print(f"[{self.TAG}] {line}")
        print(f"[{self.TAG}] key split: {self.NT} cached tile(s) of {BC} keys a cluster (cluster c half c of its chip's "
              f"tiles), the new key in a tile of {NR} on chip {self.NEWCHIP:#04x}; passes {self.passes}, stats "
              f"{'merge' if self.merge else 'max'}; chain order {[f'{k:02x}' for k in self.fold_order()]}; wo_push "
              f"{self.wo_push}")
        self._stage()
        self.make_streams()
        if self.wo_push == "before_chains" and len(self.ls[0].slabs) < self.NWS:
            raise ValueError(f"params wo_push before_chains: w_buffers={len(self.ls[0].slabs)}, the {self.NWS} W_O chunks "
                             f"of a cluster wait in its slabs for o")
        self.make_rings()
        self._buffers()

    # ---- data -------------------------------------------------------------------------------------------------------
    def _half_cols(self, h, j0, rows, kind):
        """Half h's K rows (q / k columns in rope_pairs' order) or V rows of positions j0 .., [rows, D] INT8."""
        arr = self.d.k8 if kind == "K" else self.d.v8
        cols = self.perm if kind == "K" else np.arange(self.HD)
        return np.concatenate([arr[self.GPH * h + j][j0:j0 + rows][:, cols] for j in range(self.GPH)], axis=1)

    def k_tile(self, h, t):
        """Half h's K tile t ([BC, D], A layout), or (NEW) its new-key K tile ([NR, D], key ctx in row 0)."""
        if t == NEW:
            k = np.zeros((NR, self.D), dtype=np.int8)
            k[0] = self._half_cols(h, self.ctx, 1, "K")[0]
        else:
            k = self._half_cols(h, t * BC, BC, "K")
        return self.lay.to_a(k, self.mesh).astype(np.int8)

    def v_tile(self, h, t, keys=None):
        """Half h's V tile t (V^T [D, BC], A layout), or (NEW) the new key's V^T [D, keys] (keys NR, or BC: the cache's
        next tile with the new key in column 0)."""
        if t == NEW:
            v = np.zeros((keys or NR, self.D), dtype=np.int8)
            v[0] = self._half_cols(h, self.ctx, 1, "V")[0]
        else:
            v = self._half_cols(h, t * BC, BC, "V")
        return self.lay.to_a(np.ascontiguousarray(v.T), self.mesh).astype(np.int8)

    def q_half(self, h):
        """The block-diagonal query half h ([BR, D], A layout, q columns in rope_pairs' order)."""
        q = np.zeros((BR, self.D), dtype=np.int8)
        for i in range(BR):
            j = i // self.d.GQ
            q[i, j * self.HD:(j + 1) * self.HD] = self.d.q8[BR * h + i][self.ctx][self.perm]
        return self.lay.to_a(q, self.mesh).astype(np.int8)

    def _stage(self):
        st, d, T, HD = self.st, self.d, self.TAG, self.HD
        mR = self.mem[self.ROOT]
        root_mem = (mR >> 4, mR & 15)
        # x: the token's INT8 row in a_row, in the root memory chip's HBM; its landing area in every chip's L3
        xr = self._a_row(d.hq[self.ctx])
        self.XB = xr.size
        if self.x_on_host:
            self.h_xhbm = st.put_hbm(f"{T}_xhbm", xr, mem_chip=root_mem)
        st.put_zeros(f"{T}_xland", "int8_t", self.XB)
        st.put_zeros(f"{T}_xtflag", "uint8_t", 64)
        if self.x_cast == "unicast":                       # the root's progress flags, one a chip
            st.put_zeros(f"{T}_xuprog", "uint8_t", len(self.chips) * 64)
        # y: the token's row in the root memory chip's HBM; the row the clusters fill in the root's L3, a flag each
        self.YB = 2 * self.H
        if self.y_to_hbm:
            self.h_yhbm = st.put_hbm(f"{T}_yhbm", np.zeros(self.YB, dtype=np.uint8), mem_chip=root_mem)
        st.put_zeros(f"{T}_yrow", "uint8_t", self.YB)
        st.put_zeros(f"{T}_yflag", "uint8_t", self.G * 64)
        # the cache: per chip, in its memory chip's HBM, its tiles as the push lands them in its L3 (both halves' K tiles,
        # then both halves' V tiles); the new key's chip one more tile of each, zero until the append
        st.put_zeros(f"{T}_kvc", "uint8_t", self.P * self.PB)
        st.put_zeros(f"{T}_kvcflag", "uint8_t", self.P * 64)
        st.put_zeros(f"{T}_kvcprog", "uint8_t", self.P * 64)
        self.h_kv, self.APP = {}, self.P * self.PB
        for k in self.chips:
            ts = self.tile_of[k]
            parts = [self.k_tile(h, t) for h in range(NHF) for t in ts] + [self.v_tile(h, t) for h in range(NHF) for t in ts]
            if k == self.NEWCHIP:
                parts.append(np.zeros(NHF * (self.KB + self.VB), dtype=np.int8))
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            self.h_kv[k] = st.put_hbm(f"{T}_kv_k{k:02x}", np.concatenate(parts), mem_chip=loc)
        # per cluster, in its memory chip's HBM: its weights in the order it streams them (W_q's chunks of its head, its
        # chip's half of W_k or W_v, its 32-column chunks of W_O) and its constants (each projection's column factors, the
        # RoPE tables at the token's position, the new-key tile's QK bias)
        self.img, self.chunk_off, self.cst = {}, {}, {}
        bias = np.zeros((NR, BR), dtype=np.int32)
        bias[1:] = d.mask_bias
        for g in range(self.G):
            k, h = self.chip(g), self.head[g]
            m = self.group[k]
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            wkv, fkv = (d.wk_q, d.k_f[m]) if self.kind[g] == "K" else (d.wv_q, d.v_f[m])
            qcol = lambda cc: self.perm[cc * QC:(cc + 1) * QC]                    # noqa: E731
            kcol = qcol if self.kind[g] == "K" else (lambda cc: np.arange(cc * QC, (cc + 1) * QC))
            parts = [self._pack_b(d.wq_q[:, h * HD + qcol(cc)]) for cc in range(self.NQC)]
            parts += [self._pack_b(wkv[:, m * HD + kcol(cc)]) for cc in self.kv_cc(g)]
            for s in range(self.NWS):
                o0 = self.WOC * g + s * 32
                parts.append(self._pack_b(d.wo_q[:, o0:o0 + 32]))
            offs, off = [], 0
            for b in parts:
                if b.size > self.chunk:
                    raise ValueError(f"a weight chunk of {b.size} B; params w_chunk_bytes is {self.chunk}")
                offs.append((off, b.size))
                off += b.size
            self.img[g] = st.put_hbm(f"{T}_w_g{g}", np.concatenate(parts).astype(np.int8), mem_chip=loc)
            self.chunk_off[g] = offs
            f16 = lambda a: np.asarray(a, dtype=np.float16).reshape(-1).view(np.int8)   # noqa: E731
            blob = np.concatenate([f16(d.q_f[h][self.perm]),
                                   f16(np.concatenate([fkv[kcol(cc)] for cc in self.kv_cc(g)])),
                                   f16(d.cos_rep[self.ctx][self.perm]), f16(d.sin_sgn[self.ctx][self.perm]),
                                   bias.reshape(-1).view(np.int8)])
            self.cst[g] = (st.put_hbm(f"{T}_c_g{g}", blob, mem_chip=loc), blob.nbytes)
        self.C_QF, self.C_KVF = 0, 2 * HD
        self.C_COS = self.C_KVF + 2 * self.NKVC * QC
        self.C_SIN = self.C_COS + 2 * HD
        self.C_MASK = self.C_SIN + 2 * HD
        if any(o % 64 for o in (self.C_KVF, self.C_COS, self.C_SIN, self.C_MASK)):
            raise ValueError("the constants' parts are not 64-byte aligned")

    def _buffers(self):
        HD, NC, D = self.HD, self.NKVC * QC, self.D
        # ---- every cluster's symmetric buffer, on whole transfer units (64 B) ---------------------------------------
        o = 0

        def take(nb):
            nonlocal o
            at, o = o, o + -(-nb // XFER) * XFER
            return at
        self.OFF_X = take(self.QH)                 # the cluster's query half, as the multicast leaves it
        self.OFF_FLAG, self.OFF_MFLAG, self.OFF_OFLAG = take(64), take(64), take(64)
        self.OFF_QIN = take(NHF * self.QH)         # the root: both halves, the multicasts' source
        self.OFF_QST = take(len(self.chips) * QPB)  # the root: the chips' q pieces
        self.OFF_QSFL = take(len(self.chips) * 64)
        self.OFF_MLOC = take(2 * XFER)             # cluster 0: the chip's statistics, the stats chain's source
        self.OFF_MS = take(XFER)                   # m*, as the multicast leaves it
        self.OFF_MR = take(2 * XFER)               # the root: the stats chain's destination, m*'s multicast source
        self.OFF_AR1, self.OFF_AR2 = take(ARENA), take(ARENA)   # the arenas: below every score and P buffer
        if self.reduce_impl == "root":
            self.OFF_ART = take(NHF * ARENA)       # the root: its max over the chips' row maxima, a half each
        self.OFF_ZV = take(D * FIX_ROWS)           # passes 1: the re-base PV's zero values
        self.OFF_NK, self.OFF_NV = take(self.NKB), take(self.NKB)   # the new key's chip: its half's new-key tiles
        self.OFF_NFL = take(2 * self.GPH * 2 * 64)  # ... and their runs' flags (K, then V)
        self.OFF_LOC = take(self.PBE)              # cluster 0: the chip's partial, the reduce chain's source
        self.OFF_O = take(self.XB)                 # o8 in a_row, as the multicast leaves it: W_O's A operand
        if self.bcast_impl == "pull":
            self.OFF_O8 = take(self.XB)            # bcast pull: the root's o8, which every cluster reads (a symmetric address)
        self.sym_bytes = o
        self.sym = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_sym", self.sym_bytes)
                    for k in self.chips for c in range(self.NC)}
        # the root's reduce destination; its normalise's latch beat below it
        self.dst_lead = BEAT
        self.dst_buf = self.cx_chip(self.ROOT).at(0).l1(f"{self.TAG}_dst", self.PBE + BEAT)
        self.dst = self.dst_buf.view(BEAT)
        self.consts, self.xa, self.q16, self.rt, self.rs, self.qa16, self.qpc = {}, {}, {}, {}, {}, {}, {}
        self.kv16, self.a16k, self.run8, self.vr, self.vtt, self.va = {}, {}, {}, {}, {}, {}
        self.ktile, self.vtile, self.s16, self.p8a, self.p8b, self.oacc, self.full, self.part1 = ({} for _ in range(8))
        self.p8f, self.fx, self.ywo = {}, {}, {}
        for g in range(self.G):
            l1 = self.l1_of(g)
            c = self.c(g)
            k = self.chip(g)
            self.consts[g] = l1("consts", self.cst[g][1])
            self.xa[g] = l1("x", self.XB)                        # the token's row, a_row
            self.q16[g] = l1("q16", (2 if c == 0 else 1) * 2 * HD)   # q's row (cluster 0: then cluster 1's)
            # RoPE scratch, q's and k's each their own (their RoPE may run in either order): x cos, chunk by chunk;
            # the pair's other chunk x sin
            self.rt[g] = {sd: l1(f"rt{sd}", 2 * HD) for sd in ("q", "k")}
            self.rs[g] = {sd: l1(f"rs{sd}", 2 * HD) for sd in ("q", "k")}
            if c == 0:
                self.qa16[g] = l1("qa16", 4 * HD)                # the two rows' atoms interleaved, FP16
                self.qpc[g] = l1("qpc", QPB)                     # the chip's q piece
            self.kv16[g] = l1("kv16", 2 * NC)                    # the chip's half of k or v, the GEMVs' row
            self.run8[g] = l1("run8", self.RUN)                  # its run of the new-key tile
            if self.kind[g] == "K":
                self.a16k[g] = l1("a16k", BR * NC * 2)           # k's half in row 0 of a 16-row FP16 A tile
            else:
                for n in range(self.NKVC):
                    self.vr[g, n] = l1(f"vr{n}", NR * QC * 2)    # v's chunk in row 0 of a zeroed [NR, 32] block
                    self.vtt[g, n] = l1(f"vtt{n}", QC * NR * 2)  # transposed
                    self.va[g, n] = l1(f"va{n}", QC * NR * 2)    # in the A layout
            self.ktile[g] = l1("ktile", self.NT * self.KB)       # the half's cached K tiles
            self.vtile[g] = l1("vtile", self.NT * self.VB)       # its V tiles
            ts = self.tiles[k, c]
            self.s16[g] = {t: l1(f"s16_{self._tn(t)}", BEAT + self._rows(t) * BR * 2) for t in ts}
            self.p8a[g] = l1("p8a", FA.p8_bytes(BC, BR, FA.EXACT))
            self.p8b[g] = {t: l1(f"p8b_{self._tn(t)}", FA.p8_bytes(self._rows(t), BR, FA.EXACT)) for t in ts}
            self.oacc[g] = l1("oacc", D * BR * 4)
            self.full[g] = l1("ofull", self.FB)
            if c == 1:
                self.part1[g] = l1("part1", self.OH + BEAT)      # cluster 1's compacted O^T and its row sums
            if self.passes == 1:
                self.p8f[g] = l1("p8f", FA.p8_bytes(FIX_ROWS, BR, FA.EXACT) + BEAT)
                self.fx[g] = l1("fx", (1 + FIX_ROWS) * BEAT)
            self.ywo[g] = l1("ywo", 2 * self.WOC)                # the cluster's columns of y
        r0 = self.cx_chip(self.ROOT).at(0)
        self.ot = r0.l1(f"{self.TAG}_ot", self.OB)               # the root: o~^T of both halves
        self.xt = r0.l1(f"{self.TAG}_xt", self.OB)               # o~ [32, 128]
        self.oa16 = r0.l1(f"{self.TAG}_oa16", 2 * self.XB)       # o~ in a_row, FP16
        self.o8 = (self.sv(self.ROOT, 0, self.OFF_O8) if self.bcast_impl == "pull" else
                   r0.l1(f"{self.TAG}_o8", self.XB))         # o8 in a_row
        if self.reduce_impl == "root":
            n = len(self.chips)
            self.mstg = r0.l1(f"{self.TAG}_mstg", n * XFER)      # reduce root: the chips' row maxima as pulled
            self.mrows = [r0.l1(f"{self.TAG}_mrows_h{h}", BEAT + n * BEAT) for h in range(NHF)]   # a half's, a row each
            self.p8r = [r0.l1(f"{self.TAG}_p8r_h{h}", FA.p8_bytes(n, BR, 0)) for h in range(NHF)]
            self.rslots = r0.l1(f"{self.TAG}_rslots", 2 * self.PBE)  # the pulled partials, two slots

    @staticmethod
    def _tn(t):
        return "new" if t == NEW else f"t{t}"

    @staticmethod
    def _rows(t):
        return NR if t == NEW else BC

    # ---- helpers ----------------------------------------------------------------------------------------------------
    def at_chip(self, k_from, k, c, off):
        """Chip k cluster c's symmetric buffer at `off`, as a task on chip k_from names it."""
        return self.sv(k, c, off) if k == k_from else cm.remote(k, self.sx(c, off))

    def _prefetch_parts(self, k, after):
        """Chip k's prefetcher in three parts: W_q's chunks of each ring (a task of its own, behind `after`), the k / v
        chunks (a task of its own), W_O's (the prefetcher task itself). Returns (the first part, the second)."""
        t, b = self.prefetch[k], self.prefetch[k].kernel_args
        if not 0 < self.NQC <= self.n_slots:
            raise ValueError(f"params weight_ring.slots={self.n_slots}: the W_q chunks of a ring must fit its slots")
        part = lambda lo, hi: HostBingoKernelWeightPrefetchRangeArgs(                  # noqa: E731
            b.sched_addr, b.ring_addr, b.flag_addr, b.release_addr, b.seq_addr, b.rec_addr, b.rec_slots, b.n_rings,
            b.n_slots, b.slot_bytes, b.memchip_id, batch=b.batch, policy=b.policy, engine=b.engine,
            batch_routed=b.batch_routed, queued=b.queued, queued_inflight_routed=b.queued_inflight_routed, head=b.head,
            batch_head=b.batch_head, chunk_lo=lo, chunk_hi=hi)
        t.kernel_args = part(self.WO0, 0)
        t.kernel_name = t.kernel_args.KERNEL_NAME
        host = self.cx_chip(k)
        head = host.host_node(f"weight_prefetch_k{k:02x}_wq", t.kernel_name, part(0, self.NQC), after=list(after))
        mid = host.host_node(f"weight_prefetch_k{k:02x}_kv", t.kernel_name, part(self.NQC, self.WO0), after=[])
        return head, mid

    def x_multicast(self, go):
        """x's row into every chip's L3 (xland) behind the flag xtflag; params x_cast unicast: the root memory chip's push
        engine writes a copy into each chip in turn (a progress flag a chip on the root), else DBuild's multicast."""
        if self.x_cast != "unicast":
            return super().x_multicast(go)
        R = self.ROOT
        mR = self.mem[R]
        return self.cx_chip(R).host_node(
            "x_hbm_ucast", "__host_bingo_kernel_hbm_ucast",
            HostBingoKernelHbmUcastArgs(self.h_xhbm.address, self.l3("xland"), self.l3("xtflag"), self.l3("xuprog"),
                                        self.h_seq[mR].address + 64, 1, self.XB, 1, mR, self.engine[R], self.order),
            after=[go[R]])

    def kernel_barrier(self, after=None):
        """sync barrier: one chip barrier on every chip after the kernel whose tasks were made before it (the fence ties
        every device task to the barriers around it, in the order made). Returns {chip: barrier} or {}."""
        if not self.per_kernel:
            return {}
        return self.barrier({k: list((after or {}).get(k, [])) for k in self.chips})

    def _max(self, name, rows, n, p8, arena, deps):
        """The lanewise maximum of n beats at `rows` (a one-beat prefix below them) into `arena`'s mrun, on the root's
        cluster 0: the FA softmax over an n-key tile, its own seed."""
        g = self.chips.index(self.ROOT) * self.NC
        return self.nd(g, "simd", name, "__snax_bingo_kernel_simd_fa_softmax",
                       FA(rows, p8, arena, n, 1, tile_idx=0, seed_state=1, geom_mode=FA.GEOM_SELF, score_scale=1.0,
                          p_mode=0, p8_pitch=0), deps)

    def _stats_root(self, before):
        """reduce root: the root pulls every chip's row maxima [m_0 | m_1] in the fold order into one tile, splits it
        into the halves' rows and runs the max over each, then packs m* into OFF_MR. Returns the pack."""
        R, n = self.ROOT, len(self.chips)
        g = self.chips.index(R) * self.NC
        pulls = []
        for i, k in enumerate(self.order):
            src = self.sv(R, 0, self.OFF_MLOC) if k == R else cm.remote(k, self.sx(0, self.OFF_MLOC))
            pulls.append(self.copy(g, f"mg_pull{i}", src, at_offset(self.mstg, i * XFER), XFER, [before[k]] + pulls[-1:]))
        pk = []
        for h in range(NHF):
            sp = self.copy2d(g, f"mg_split_h{h}", at_offset(self.mstg, h * BEAT), at_offset(self.mrows[h], BEAT), BEAT,
                             XFER, BEAT, n, [pulls[-1]] + pk[-1:])
            art = self.sv(R, 0, self.OFF_ART + h * ARENA)
            mx = self._max(f"mg_rowmax_h{h}", at_offset(self.mrows[h], BEAT), n, self.p8r[h], art, [sp])
            pk.append(self.copy(g, f"mg_pack_h{h}", at_offset(art, MRUN), self.sv(R, 0, self.OFF_MR + h * BEAT), BEAT,
                                [mx] + pk[-1:]))
        return pk[-1]

    def _reduce_root(self, before):
        """reduce root: the root pulls every other chip's partial in the fold order into two L1 slots and adds it on the
        SIMD, its own first, into dst. Returns the last add."""
        R, xb = self.ROOT, self.PBE
        g = self.chips.index(R) * self.NC
        pulls, adds = [], []
        for i, k in enumerate(self.order[1:]):
            slot = self.rslots.view((i % 2) * xb)
            pulls.append(self.copy(g, f"y_pull{i}", cm.remote(k, self.sx(0, self.OFF_LOC)), slot, xb,
                                   [before[k]] + pulls[-1:] + ([adds[i - 2]] if i >= 2 else [])))
            a_in = self.sv(R, 0, self.OFF_LOC) if i == 0 else self.dst
            adds.append(self.nd(g, "simd", f"y_add{i}", "__snax_bingo_kernel_simd_stream_elementwise",
                                SnaxBingoKernelSimdAddF16Args(a_in, slot, self.dst, xb // 64, 32),
                                [pulls[-1], before[R]] + adds[-1:]))
        return adds[-1]

    def qk(self, g, name, K, rows, bias, s16, deps):
        k, c = self.chip(g), self.c(g)
        return self.nd(g, "gemm", name, "__snax_bingo_kernel_gemm_fa_qk",
                       SnaxBingoKernelGemmFaQkArgs(K, self.sv(k, c, self.OFF_X), bias, at_offset(s16, BEAT),
                                                   M=rows // 16, K=self.D // 4, N=BR // 16, d_shift=self.d.k_s), deps)

    def softmax(self, g, name, s16, p8, arena, rows, i, seed_state, deps):
        return self.nd(g, "simd", name, "__snax_bingo_kernel_simd_fa_softmax",
                       FA(at_offset(s16, BEAT), p8, arena, rows, 1, tile_idx=i, seed_state=seed_state,
                          geom_mode=FA.GEOM_SELF, score_scale=self.d.a_exp, p_mode=FA.EXACT, p8_pitch=0), deps)

    def pv(self, g, name, V, p8, rows, first, last, oacc, out, deps):
        flags = PVK.B_KMAJOR | (0 if first else PVK.C_COLSCALE) | (PVK.D_FP16 if last else 0)
        return self.nd(g, "gemm", name, "__snax_bingo_kernel_gemm_fa_pv",
                       PVK(V, p8, 0 if first else oacc, out if last else oacc, M=self.D // 16, K=rows // 4,
                           N=BR // 16, flags=flags, corr_addr=0 if first else at_offset(p8, FA.corr_offset(rows, BR)),
                           d_shift=self.d.k_o if last else 0), deps)

    def tile_kv(self, g, t):
        """Cluster g's K and V tile t, and the tasks after which each is in L1."""
        k, c = self.chip(g), self.c(g)
        if t == NEW:
            return self.sv(k, c, self.OFF_NK), self.sv(k, c, self.OFF_NV), self.nk_wait[g], self.nk_wait[g]
        i = self.tile_of[k].index(t)
        return (self.ktile[g].view(i * self.KB), self.vtile[g].view(i * self.VB), self.kc_ready[g],
                self.vc_ready[g])

    def compact(self, g, src, dst, deps):
        """PV's [D, BR] FP16 output -> the half's [HD, BR] O^T: lane i keeps its group's rows, 8-byte atoms (four lanes
        of one group), on the xDMA."""
        return self.xd(g, "o_compact", SnaxBingoKernelXdma6dArgs(
            src, dst, BR * 2, BR * 2, [LANES * BR * 2, self.HD * BR * 2 + ATOM], [self.HD // LANES, self.d.GQ],
            [LANES * BR * 2, ATOM], [self.HD // LANES, self.d.GQ]), deps)

    # ---- the block --------------------------------------------------------------------------------------------------
    def build(self):
        d, R, HD = self.d, self.ROOT, self.HD
        dm, xdma = self.roles["dm"], self.roles["xdma"]
        NC = self.NKVC * QC
        N = self.NEWCHIP
        for ls in self.ls:
            ls.strict = True
        # ---- before the window: the constants into L1; the zero-padded operands zeroed --------------------------------
        pre = {k: [] for k in self.chips}
        for g in range(self.G):
            k = self.chip(g)
            h, nb = self.cst[g]
            pre[k].append(self.copy(g, f"const_ld_g{g}", h.address, self.consts[g], nb, []))
            z = [(self.a16k[g], BR * NC * 2)] if self.kind[g] == "K" else \
                [(self.vr[g, n], NR * QC * 2) for n in range(self.NKVC)]
            if (k, self.c(g)) == (R, 0):
                z.append((self.oa16, 2 * self.XB))
            prev = []
            for i, (buf, n) in enumerate(z):
                prev = [self.memset(g, f"zero{i}", buf, n, prev)]
            pre[k] += prev
        go = self.setup(extra=pre)
        self.dfg.relay_one_consumer_cores = [xdma]
        # ---- the hosts: x's multicast (the root); per chip the W_q part of its prefetcher, its cache, the k / v part,
        # the W_O part -------------------------------------------------------------------------------------------------
        one = lambda mm: self.h_seq[mm].address + 64     # noqa: E731  a word holding 1 in the memory chip's HBM
        self.x_put = self.x_multicast(go)
        self.ucast, self.pf_parts = {}, {}
        for k in self.chips:
            first = [go[k]] + ([self.x_put] if k == R and self.x_on_host else [])
            head, mid = self._prefetch_parts(k, first)
            mk = self.mem[k]
            self.ucast[k] = self.cx_chip(k).host_node(
                f"kv_ucast_k{k:02x}", "__host_bingo_kernel_hbm_ucast",
                HostBingoKernelHbmUcastArgs(self.h_kv[k].address, self.l3("kvc"), self.l3("kvcflag"), self.l3("kvcprog"),
                                            one(mk), 1, self.PB, self.P, mk, self.engine[k], [k]), after=[head])
            self.edge(self.ucast[k], mid)
            self.edge(mid, self.prefetch[k])
            self.pf_parts[k] = (head, mid, self.prefetch[k])
        if self.cache_order == "after_wq" and not self.per_kernel:
            for k in self.chips:
                for j in self.chips:
                    if j != k and self.mem[j] == self.mem[k]:
                        self.edge(self.pf_parts[j][0], self.ucast[k])
        # ---- x into every cluster's L1 ---------------------------------------------------------------------------------
        xf = {}
        for g in range(self.G):
            xf[g] = self.nd(g, "dm", f"x_wait_g{g}", "__snax_bingo_kernel_idma_fetch_flagged",
                            SnaxBingoKernelIdmaFetchFlaggedArgs(self.l3("xland"), self.xa[g], self.XB, self.l3("xtflag"),
                                                                1, 1), [go[self.chip(g)]])
            self.dfg.bingo_add_order_edge(self.x_put, xf[g])
        # ---- q: the GEMVs, the column factors, RoPE ------------------------------------------------------------------
        qrot = {}
        for g in range(self.G):
            cst, q16 = self.consts[g], self.q16[g]
            dq, rot = {}, []

            def deq(n, gm, g=g, cst=cst, q16=q16, dq=dq, rot=rot):
                dq[n] = self.mul(g, f"dqq{n}", q16.view(2 * QC * n), cst.view(self.C_QF + 2 * QC * n),
                                 q16.view(2 * QC * n), QC, [gm])
                if n % 2:
                    rot.extend(self.rope_row(g, "q", q16, n - 1, n, n - 1, n, [dq[n - 1], dq[n]]))
            self.chunks(g, "gq", list(range(self.NQC)), self.xa[g], [q16.view(2 * QC * n) for n in range(self.NQC)],
                        d.k_x, [xf[g]], then=deq)
            qrot[g] = rot
        # ---- the chips' q pieces into the root, the halves assembled and multicast ----------------------------------
        qput = {}
        for k in self.chips:
            g0, g1 = self.gs(k)
            cp = self.copy(g1, "q_to_c0", self.q16[g1], self.q16[g0].view(2 * HD), 2 * HD, qrot[g1])
            # atom j of row r (4 values, 8 bytes) at 16 j + 8 r: one 64-B block's word of the two lanes after quantising
            sc = self.xd(g0, "q_pair", SnaxBingoKernelXdma6dArgs(
                self.q16[g0], self.qa16[g0], ATOM, 2 * ATOM, [LANES * ATOM, 2 * HD], [HD // 4 // LANES, 2],
                [LANES * 2 * ATOM, ATOM], [HD // 4 // LANES, 2]), qrot[g0] + [cp])
            qz = self.quant(g0, "q_pair8", self.qa16[g0], self.qpc[g0], 4 * HD, d.s_q.inv, [sc])
            slot = 2 * self.group[k] + self.kv_half[k]
            qput[k] = self.nd(g0, "dm", "q_put", "__snax_bingo_kernel_idma_put_flagged",
                              SnaxBingoKernelIdmaPutFlaggedArgs(
                                  self.qpc[g0], self.at_chip(k, R, 0, self.OFF_QST + slot * QPB), QPB,
                                  self.at_chip(k, R, 0, self.OFF_QSFL + slot * 64), value=1), [qz])
        b_xq = self.kernel_barrier()                      # sync barrier: x and q end
        if self.cache_order == "after_wq" and b_xq:
            for k in self.chips:
                self.edge(b_xq[k], self.ucast[k])
        qw = self.node(R, 0, dm, "q_gather", "__snax_bingo_kernel_wait_flags",
                       SnaxBingoKernelWaitFlagsArgs(self.sv(R, 0, self.OFF_QSFL), len(self.chips), 1), [qput[R]])
        for k in self.chips:
            if k != R:
                self.dfg.bingo_add_order_edge(qput[k], qw)
        # piece (m, hh) word w -> half m // 4, block 32 (m % 4) + w, byte 16 (m % 4) + 8 hh: pieces 8 hf + 2 j + hh
        gq, gpb = self.GPH, HD // 4                           # groups a half, blocks a group's columns
        qs = self.node(R, 0, xdma, "q_scatter", "__snax_bingo_kernel_xdma_6d", SnaxBingoKernelXdma6dArgs(
            self.sv(R, 0, self.OFF_QST), self.sv(R, 0, self.OFF_QIN), ATOM, 64,
            [LANES * ATOM, QPB, 2 * QPB, 2 * gq * QPB], [gpb // LANES, 2, gq, NHF],
            [LANES * 64, ATOM, gpb * 64 + 2 * ATOM, self.QH], [gpb // LANES, 2, gq, NHF]), [qw])
        # bcast pull: every cluster reads its half from the root once the scatter is done (src_ready)
        pull = self.bcast_impl == "pull"
        ready = {"src_ready": qs} if pull else {}
        self.qin = self.broadcast("q0", self.sv(R, 0, self.OFF_QIN), self.QH, self.OFF_X, self.OFF_FLAG,
                                  {k: (qs if k == R else qput[k]) for k in self.chips}, clusters=[0], **ready)
        put0 = qs if pull else self.bcast_puts[-1]
        self.qin.update(self.broadcast("q1", self.sv(R, 0, self.OFF_QIN + self.QH), self.QH, self.OFF_X, self.OFF_FLAG,
                                       {k: (put0 if k == R else qput[k]) for k in self.chips}, clusters=[1], **ready))
        self.q_puts = [] if pull else [put0, self.bcast_puts[-1]]
        if self.cache_order == "after_q" and not self.per_kernel:
            # sync edge: one probe on the root once q is in every cluster; every chip's cache copy behind it
            q_all = self.probe(R, "q_all", list(dict.fromkeys(self.qin.values())))
            for k in self.chips:
                self.edge(q_all, self.ucast[k])
        b_qb = self.kernel_barrier()                      # sync barrier: the q gather and its broadcast end
        if self.cache_order == "after_q" and b_qb:
            for k in self.chips:
                self.edge(b_qb[k], self.ucast[k])
        # ---- the cache: each cluster's half of its chip's tiles into L1 ---------------------------------------------
        self.kc_ready, self.vc_ready = {}, {}
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            n = self.NT * self.KB
            self.kc_ready[g] = self.nd(g, "dm", "kc_fetch", "__snax_bingo_kernel_idma_fetch_flagged",
                                       SnaxBingoKernelIdmaFetchFlaggedArgs(self.l3("kvc", c * n), self.ktile[g], n,
                                                                           self.l3("kvcflag"), 1, 1), [])
            self.vc_ready[g] = self.nd(g, "dm", "vc_fetch", "__snax_bingo_kernel_idma_fetch_flagged",
                                       SnaxBingoKernelIdmaFetchFlaggedArgs(self.l3("kvc", self.PB + c * n), self.vtile[g],
                                                                           n, self.l3("kvcflag", 64), 1, 1), [])
            for t in (self.kc_ready[g], self.vc_ready[g]):
                self.dfg.bingo_add_order_edge(self.ucast[k], t)
        # ---- pass 1 over the cached tiles ---------------------------------------------------------------------------
        self.sm1, self.pv1 = {g: [] for g in range(self.G)}, {g: [] for g in range(self.G)}
        for g in range(self.G):
            self._pass1(g, [t for t in self.tiles[self.chip(g), self.c(g)] if t != NEW])
        self.kernel_barrier()                             # sync barrier: pass 1 over the cache ends
        # ---- the token's k / v: the chip's half, its run put into the new key's tile ----------------------------------
        runs = {}
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            cst, kv16 = self.consts[g], self.kv16[g]
            js = [self.NQC + n for n in range(self.NKVC)]
            if self.kind[g] == "K":
                dq = {}

                def deqk(n, gm, g=g, cst=cst, kv16=kv16, dq=dq):
                    dq[n] = self.mul(g, f"dqk{n}", kv16.view(2 * QC * n), cst.view(self.C_KVF + 2 * QC * n),
                                     kv16.view(2 * QC * n), QC, [gm])
                self.chunks(g, "gk", js, self.xa[g], [kv16.view(2 * QC * n) for n in range(self.NKVC)], d.k_x, [],
                            then=deqk)
                c0 = self.kv_cc(g)[0]
                rot = self.rope_row(g, "k", kv16, c0, c0 + 1, 0, 1, [dq[0], dq[1]])
                sc = self.row_to_a(g, "k2a", kv16, self.a16k[g], NC, list(rot))
                made = [self.quant(g, "k8", self.a16k[g], self.run8[g], BR * NC * 2, d.s_k.inv, [sc])]
            else:
                tas = {}

                def deqv(n, gm, g=g, cst=cst, tas=tas):
                    vr, vt, va = self.vr[g, n], self.vtt[g, n], self.va[g, n]
                    dqv = self.mul(g, f"dqv{n}", vr, cst.view(self.C_KVF + 2 * QC * n), vr, QC, [gm])
                    tr = self.nd(g, "xdma", f"vtr{n}", "__snax_bingo_kernel_xdma_transpose_2d",
                                 SnaxBingoKernelXdmaTranspose2dArgs(vr, vt, NR, QC, 2), [dqv])
                    tas[n] = self.xd(g, f"vta{n}", convert_args("row_major", "A", QC, NR, self.mesh, 2, vt, va), [tr])
                self.chunks(g, "gv", js, self.xa[g], [self.vr[g, n] for n in range(self.NKVC)], d.k_x, [], then=deqv)
                made = [self.quant(g, f"vq{n}", self.va[g, n], self.run8[g].view(n * QC * NR), QC * NR * 2, d.s_v.inv,
                                   [tas[n]]) for n in range(self.NKVC)]
            m, hh = self.group[k], self.kv_half[k]
            hf, slot = m // self.GPH, 2 * (m % self.GPH) + hh
            base, fl = (self.OFF_NK, 0) if self.kind[g] == "K" else (self.OFF_NV, 2 * self.GPH)
            runs[g] = (hf, self.nd(g, "dm", "run_put", "__snax_bingo_kernel_idma_put_flagged",
                                   SnaxBingoKernelIdmaPutFlaggedArgs(
                                       self.run8[g], self.at_chip(k, N, hf, base + slot * self.RUN), self.RUN,
                                       self.at_chip(k, N, hf, self.OFF_NFL + (fl + slot) * 64), value=1), made))
        self.kernel_barrier()                             # sync barrier: the token's k / v end
        # ---- the new key's chip: its runs, then pass 1 over the new key's tile ----------------------------------------
        self.nk_wait = {}
        for c, g in enumerate(self.gs(N)):
            mine = [t for gg, (hf, t) in runs.items() if hf == c]
            self.nk_wait[g] = self.nd(g, "dm", "run_wait", "__snax_bingo_kernel_wait_flags",
                                      SnaxBingoKernelWaitFlagsArgs(self.sv(N, c, self.OFF_NFL), len(mine), 1),
                                      [t for t in mine if t.assigned_chiplet_id == N])
            for t in mine:
                if t.assigned_chiplet_id != N:
                    self.dfg.bingo_add_order_edge(t, self.nk_wait[g])
            self._pass1(g, [NEW])
        # ---- wo_push before_chains: every W_O chunk into the cluster's slabs now, the statistics behind the last -------
        wo_js = [self.WO0 + s for s in range(self.NWS)]
        wl, wo_last = {}, {}
        if self.wo_push == "before_chains":
            for g in range(self.G):
                wl[g] = self.loads(g, wo_js)
                wo_last[g] = [wl[g][self.NWS - 1][0]]
        # ---- the statistics: each chip's on its cluster 0, one ChainGather into the root, m* multicast ---------------
        self.mloc = {}
        lkeep = {}
        for k in self.chips:
            g0, g1 = self.gs(k)
            reps = 2 if self.merge else 1                  # m (and l) of the cluster's half
            cps = []
            for c, g in ((1, g1), (0, g0)):
                a = self.sv(k, c, self.OFF_AR1)
                cps.append(self.nd(g, "dm", "st_pack", "__snax_bingo_kernel_idma_2d_copy",
                                   SnaxBingoKernelIdma2dCopyArgs(at_offset(a, MRUN),
                                                                 self.at_chip(k, k, 0, self.OFF_MLOC + c * BEAT),
                                                                 BEAT, LRUN - MRUN, XFER, reps),
                                   [self.sm1[g][-1]] + cps[-1:] + wo_last.get(g, [])
                                   # wo_push after_reduce / before_chains: every cache tile fetched before the
                                   # statistics' chain
                                   + ([self.kc_ready[g], self.vc_ready[g]]
                                      if self.wo_push in ("after_reduce", "before_chains") else [])))
            self.mloc[k] = cps[-1]
            if self.passes == 1 and not self.merge:
                for c, g in enumerate((g0, g1)):
                    # l_c beside the re-base factor, before the re-base step moves the arena's lrun
                    lkeep[g] = self.copy(g, "l_keep", at_offset(self.sv(k, c, self.OFF_AR1), LRUN),
                                         at_offset(self.p8f[g], FA.p8_bytes(FIX_ROWS, BR, FA.EXACT)), BEAT,
                                         [self.sm1[g][-1]])
        others = self.order[1:]
        # sync barrier: the chips' statistics are there once every chip has passed the barrier after them
        bm = self.kernel_barrier({k: [self.mloc[k]] for k in self.chips})
        before = {k: bm[R] for k in self.chips} if bm else dict(self.mloc)
        if self.reduce_impl == "root":
            self.m_made = self._stats_root(before)
        else:
            if self.merge:
                csr0, csr1 = xdma_softmax_stats_csrs(XDMA_SSJ_ROWS, d.a_exp)
                ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                    cm.remote, R, 0, others, src=self.sx(0, self.OFF_MLOC), dst=self.sx(0, self.OFF_MR), size=2 * XFER,
                    junction=self.jct["SoftmaxStats"], jct_csr0=csr0, jct_csr1=csr1)
            else:
                ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                    cm.remote, R, 0, others, src=self.sx(0, self.OFF_MLOC), dst=self.sx(0, self.OFF_MR), size=XFER,
                    junction=self.jct["Elementwise"], jct_csr0=EW_F16_MAX)
            self.m_made = self.node(R, 0, xdma, "st_chain", "__snax_bingo_kernel_xdma_chain_gather", ga,
                                    list(dict.fromkeys(before[k] for k in self.chips)))
        self.bcast_puts = []
        marr = self.broadcast("m", self.sv(R, 0, self.OFF_MR), XFER, self.OFF_MS, self.OFF_MFLAG,
                              {k: (self.m_made if k == R else self.mloc[k]) for k in self.chips}, src_ready=self.m_made)
        self.m_puts = list(self.bcast_puts)
        self.kernel_barrier()                             # sync barrier: the statistics' exchange and m* end
        # ---- pass 2 (or the re-base), the partials ------------------------------------------------------------------
        # cluster 1's O^T and row sums next to cluster 0's in its OFF_LOC; cluster 0's last task of the partial (its row
        # sums' write, or with stats merge its compaction) follows them all: the reduce's one producer a chip
        self.join, self.sm2 = {}, {}
        for k in self.chips:
            g0, g1 = self.gs(k)
            o1 = self._pass2(g1, marr[(k, 1)], lkeep.get(g1))
            cmp1 = self.compact(g1, self.full[g1], self.part1[g1], o1)
            last1 = self.copy(g1, "o_to_c0", self.part1[g1], self.sv(k, 0, self.OFF_LOC + self.OH), self.OH, [cmp1])
            if not self.merge:
                l1 = self._l_task(g1, at_offset(self.part1[g1], self.OH), lkeep.get(g1), [])
                last1 = self.copy(g1, "l_to_c0", at_offset(self.part1[g1], self.OH),
                                  self.sv(k, 0, self.OFF_LOC + self.OB + BEAT), BEAT, [l1, last1])
            o0 = self._pass2(g0, marr[(k, 0)], lkeep.get(g0))
            cmp0 = self.compact(g0, self.full[g0], self.sv(k, 0, self.OFF_LOC), o0 + ([last1] if self.merge else []))
            self.join[k] = cmp0 if self.merge else \
                self._l_task(g0, self.sv(k, 0, self.OFF_LOC + self.OB), lkeep.get(g0), [cmp0, last1])
        # ---- the reduce into the root, each half normalised and transposed ----------------------------------------
        bj = self.kernel_barrier({k: [self.join[k]] for k in self.chips})     # sync barrier: every chip's partial
        before = {k: bj[R] for k in self.chips} if bj else dict(self.join)
        if self.reduce_impl == "root":
            self.o_chain = self._reduce_root(before)
        else:
            ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                cm.remote, R, 0, others, src=self.sx(0, self.OFF_LOC), dst=self.dst_expr(), size=self.PBE,
                junction=self.jct["Elementwise"], jct_csr0=cm.EW_F16_ADD, holds=[self.dst_buf])
            self.o_chain = self.node(R, 0, xdma, "o_chain", "__snax_bingo_kernel_xdma_chain_gather", ga,
                                     list(dict.fromkeys(before[k] for k in self.chips)))
        # sync barrier: each chip's W_O part behind its own barrier after the reduce, not the root's reduce itself
        br = self.kernel_barrier({R: [self.o_chain]})
        if self.wo_push == "after_reduce":
            for k in self.chips:
                self.edge(br[k] if br else self.o_chain, self.prefetch[k])
        nrm, tr = [], []
        for h in range(NHF):
            o_in = self.dst.view(h * self.OH)
            latch = self.dst_buf if h == 0 else self.dst.view(h * self.OH - BEAT)
            l_src = self.sv(R, 0, self.OFF_MR + XFER + h * BEAT) if self.merge else self.dst.view(self.OB + h * BEAT)
            # half 1's latch beat is half 0's last row, which half 0's normalise has read
            nrm.append(self.node(R, 0, self.roles["simd"], f"o_norm_h{h}", "__snax_bingo_kernel_simd_mla_normalise",
                                 SnaxBingoKernelSimdMlaNormaliseArgs(l_src, latch, o_in, self.ot.view(h * self.OH), HD,
                                                                     d.a_n_bits), [self.o_chain] + nrm[-1:]))
            tr.append(self.node(R, 0, xdma, f"o_transpose_h{h}", "__snax_bingo_kernel_xdma_transpose_2d",
                                SnaxBingoKernelXdmaTranspose2dArgs(self.ot.view(h * self.OH), self.xt.view(h * self.OH),
                                                                   HD, BR, 2), [nrm[h]] + tr[-1:]))
        # ---- o: a_row, quantised, multicast into every cluster --------------------------------------------------------
        g_r0 = self.chips.index(R) * self.NC
        na = d.NH * HD // 4 // LANES
        toar = self.xd(g_r0, "o_toar", SnaxBingoKernelXdma6dArgs(self.xt, self.oa16, ATOM, 2 * ATOM, [LANES * ATOM], [na],
                                                                 [LANES * 2 * ATOM], [na]), [tr[-1]])
        oq = self.quant(g_r0, "o8", self.oa16, self.o8, 2 * self.XB, d.s_ot.inv, [toar])
        early = self.wo_push == "before_chains"           # its W_O chunks are in the slabs already
        if not early:
            wl = {g: self.loads(g, wo_js) for g in range(self.G) if g != g_r0}
        self.bcast_puts = []
        oarr = self.broadcast("o", self.o8, self.XB, self.OFF_O, self.OFF_OFLAG,
                              {k: (oq if k == R else self.join[k]) for k in self.chips}, src_ready=oq)
        self.o_puts = list(self.bcast_puts)
        self.kernel_barrier()                             # sync barrier: o's normalise and broadcast end
        if not early:
            wl[g_r0] = self.loads(g_r0, wo_js)
        # ---- W_O by columns, y into the root's row, the root's write into HBM ------------------------------------------
        far = lambda k: None if k == R else R           # noqa: E731
        rg = self.gs(R)
        yput = {}
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            gms = self.chunks(g, "gwo", wo_js, self.sv(k, c, self.OFF_O), [self.ywo[g].view(64 * s) for s in range(self.NWS)],
                              d.k_wo, [oarr[(k, c)]], lds=wl[g])
            yput[g] = self.nd(g, "dm", "y_put", "__snax_bingo_kernel_idma_put_flagged",
                              SnaxBingoKernelIdmaPutFlaggedArgs(self.ywo[g], self.l3("yrow", 2 * self.WOC * g, far(k)),
                                                                2 * self.WOC, self.l3("yflag", g * 64, far(k)), value=1),
                              list(gms))                    # every W_O GEMM writes its part of ywo
        self.kernel_barrier()                             # sync barrier: W_O and y end
        # ---- the append: the new key's chip's cluster 1 writes both halves' new tiles into the cache's next tile in
        # HBM, behind its y put (the root's cluster 0 keeps its DM core for the y wait and write) -----------------------
        app = {k: [] for k in self.chips}
        app[N] = self.append_now([yput[self.gs(N)[1]]])
        yw = self.node(R, 0, dm, "y_wait", "__snax_bingo_kernel_wait_flags",
                       SnaxBingoKernelWaitFlagsArgs(self.l3("yflag"), self.G, 1), [yput[g] for g in rg])
        for g in range(self.G):
            if g not in rg:
                self.dfg.bingo_add_order_edge(yput[g], yw)
        yh = self.y_out(yw)
        fin = {k: app[k] + [yput[g] for g in self.gs(k)] + ([yh] if k == R else []) for k in self.chips}
        self.end_window([self.probe(k, f"fin{k:02x}", fin[k]) for k in self.chips])
        self._link_report()
        self._checks()

    def kv_append(self, deps):
        """The append: the new key's chip's cluster 1 writes both halves' new K tiles (the first 16-key block of the
        64-key tile ctx / 64) and V^T tiles (each 16-dimension block's first NR / 4 blocks of 64 B) into the cache
        image's next tile in its memory chip's HBM, each half behind its run wait and `deps`. Returns the tasks."""
        N = self.NEWCHIP
        base = self.h_kv[N].address + self.APP
        ga = self.gs(N)[1]
        app = []
        for c, g in enumerate(self.gs(N)):
            app.append(self.copy(ga, f"k_app_h{c}", self.sv(N, c, self.OFF_NK), base + c * self.KB, self.NKB,
                                 [self.nk_wait[g]] + list(deps) + app[-1:]))
            # V^T [D, NR] -> V^T [D, BC]: each 16-row block's NR / 4 blocks of 64 B at the head of its BC / 4
            app.append(self.nd(ga, "dm", f"v_app_h{c}", "__snax_bingo_kernel_idma_2d_copy",
                               SnaxBingoKernelIdma2dCopyArgs(self.sv(N, c, self.OFF_NV), base + NHF * self.KB + c * self.VB,
                                                             NR * 64 // 4, NR * 64 // 4, BC * 64 // 4, self.D // 16),
                               [app[-1]]))
        return app

    def append_now(self, deps):
        """The append where build() makes it: behind `deps` (the new key's chip's cluster-1 y put). Returns its tasks,
        which the chip's end follows. A layer that writes into HBM only after its own collectives returns [] here and
        calls kv_append at its end."""
        return self.kv_append(deps)

    # ---- the passes of one cluster ---------------------------------------------------------------------------------
    def _pass1(self, g, ts):
        """Pass 1 over cluster g's tiles ts: per tile QK and the FA softmax (its own seed at the cluster's first tile);
        passes 1 also PV, the accumulator scaled by the softmax's corr after the first tile."""
        k, c = self.chip(g), self.c(g)
        all_ts = self.tiles[k, c]
        ar1 = self.sv(k, c, self.OFF_AR1)
        for t in ts:
            i = all_ts.index(t)
            K, V, kr, vr = self.tile_kv(g, t)
            rows = self._rows(t)
            bias = self.consts[g].view(self.C_MASK) if t == NEW else 0
            q = self.qk(g, f"qk_{self._tn(t)}", K, rows, bias, self.s16[g][t], [kr, self.qin[(k, c)]])
            p8 = self.p8b[g][t] if self.passes == 1 else self.p8a[g]
            sm = self.softmax(g, f"s1_{self._tn(t)}", self.s16[g][t], p8, ar1, rows, i, 1, [q] + self.sm1[g][-1:])
            self.sm1[g].append(sm)
            if self.passes == 1:
                self.pv1[g].append(self.pv(g, f"pv_{self._tn(t)}", V, p8, rows, i == 0, False, self.oacc[g], None,
                                           [sm, vr] + self.pv1[g][-1:]))

    def _pass2(self, g, marr, lkeep):
        """Cluster g after m*'s arrival: passes 2, m* into its pass-2 arena, per tile the softmax seeded at m* and PV
        (the last tile FP16 into ofull); passes 1, the re-base step over FIX_ROWS rows of m* (its corr exp(a (m_c - m*)))
        and a PV with zero values and that corr as C's column scale, FP16 into ofull. Returns [O's producer]."""
        k, c = self.chip(g), self.c(g)
        ts = self.tiles[k, c]
        m_in = self.sv(k, c, self.OFF_MS + c * BEAT)
        if self.passes == 2:
            ar2 = self.sv(k, c, self.OFF_AR2)
            un = self.copy(g, "m_unpack", m_in, at_offset(ar2, MRUN), BEAT, [marr])
            sm, pv = [], []
            for i, t in enumerate(ts):
                K, V, kr, vr = self.tile_kv(g, t)
                rows = self._rows(t)
                sm.append(self.softmax(g, f"s2_{self._tn(t)}", self.s16[g][t], self.p8b[g][t], ar2, rows, i, 0,
                                       [un, self.sm1[g][-1]] + sm[-1:]))
                pv.append(self.pv(g, f"pv_{self._tn(t)}", V, self.p8b[g][t], rows, i == 0, i == len(ts) - 1,
                                  self.oacc[g], self.full[g], [sm[i], vr] + pv[-1:]))
            self.sm2[g] = sm[-1]
            return [pv[-1]]
        ar1 = self.sv(k, c, self.OFF_AR1)
        fx, p8f = self.fx[g], self.p8f[g]
        cpf = self.copy2d(g, "m_fix", m_in, at_offset(fx, BEAT), BEAT, 0, BEAT, FIX_ROWS, [marr])
        fs = self.softmax(g, "s_fix", fx, p8f, ar1, FIX_ROWS, len(ts), 1,
                          [cpf, self.sm1[g][-1]] + ([lkeep] if lkeep is not None else []))
        self.sm2[g] = fs
        fp_ = self.nd(g, "gemm", "pv_fix", "__snax_bingo_kernel_gemm_fa_pv",
                      PVK(self.sv(k, c, self.OFF_ZV), p8f, self.oacc[g], self.full[g], M=self.D // 16, K=FIX_ROWS // 4,
                          N=BR // 16, flags=PVK.B_KMAJOR | PVK.C_COLSCALE | PVK.D_FP16,
                          corr_addr=at_offset(p8f, FA.corr_offset(FIX_ROWS, BR)), d_shift=self.d.k_o),
                      [fs, self.pv1[g][-1]])
        return [fp_]

    def _l_task(self, g, dst, lkeep, deps):
        """(stats max) Cluster g's row sums of its half into dst: passes 2, pass 2's l from its arena; passes 1, l_c
        times the re-base step's corr (a SIMD multiply)."""
        k, c = self.chip(g), self.c(g)
        if self.passes == 2:
            return self.copy(g, "l_cp", at_offset(self.sv(k, c, self.OFF_AR2), LRUN), dst, BEAT, [self.sm2[g]] + deps)
        p8f = self.p8f[g]
        return self.nd(g, "simd", "l_fix", "__snax_bingo_kernel_simd_stream_elementwise",
                       SnaxBingoKernelSimdMulF16Args(at_offset(p8f, FA.corr_offset(FIX_ROWS, BR)),
                                                     at_offset(p8f, FA.p8_bytes(FIX_ROWS, BR, FA.EXACT)), dst, 1, BR),
                       [self.sm2[g], lkeep] + deps)

    # ---- reports and proofs -----------------------------------------------------------------------------------------
    def _link_report(self):
        """The bytes each memory link carries in the window, and the busiest link's time at the links' rate."""
        per = {}
        for k in self.chips:
            w = sum(n for g in self.gs(k) for _, n in self.chunk_off[g])
            per.setdefault(self.link[k], [0, 0])
            per[self.link[k]][0] += w
            per[self.link[k]][1] += self.P * self.PB
        gbps = 231.2 / len(per)                         # the memory chips' links together, GB/s, split evenly
        tot_w = sum(v[0] for v in per.values())
        tot_c = sum(v[1] for v in per.values())
        (m, side), (bw, bc) = max(per.items(), key=lambda kv: sum(kv[1]))
        print(f"[{self.TAG}] stream: weights {tot_w / 2**20:.2f} MiB, cache {tot_c / 2**20:.2f} MiB over {len(per)} memory "
              f"links; the busiest (memory chip {m:#04x}, side {side:+d}) {(bw + bc) / 2**20:.3f} MiB "
              f"({bw / 2**20:.3f} weights, {bc / 2**20:.3f} cache): {(bw + bc) / (gbps * 1e3):.1f} us at {gbps:.1f} GB/s")

    def _check_collectives(self):
        """Every multicast (a device bcast put, a memory chip's HBM multicast) and every ChainGather is ordered against
        every other by a path of dependency and order edges. wo_push after_reduce: every ring load and cache fetch of
        the first two prefetcher parts precedes the first ChainGather, and every chip's W_O part follows the last;
        before_chains: every ring load of every part (W_O's included) and every cache fetch precedes the first.
        Returns (multicasts, gathers)."""
        G = self.dfg.bingo_order_view()
        if self.per_kernel:
            # sync barrier: a barrier round orders every chip's tasks before it against every chip's after it
            G = G.copy()
            for n, rnd in enumerate(self.barrier_rounds):
                v = ("barrier_round", n)
                for b in rnd.values():
                    G.add_edge(b, v)
                    for s in list(G.successors(b)):
                        if s != v:
                            G.add_edge(v, s)
        mc = [t for t in self.dfg.node_list
              if (t.kernel_name == "__snax_bingo_kernel_idma_put_flagged" and t.kernel_args.bcast & 1)
              or t.kernel_name == "__host_bingo_kernel_hbm_bcast"]
        ch = [t for t in self.dfg.node_list if t.kernel_name == "__snax_bingo_kernel_xdma_chain_gather"]
        items = sorted(mc + ch, key=lambda t: t.node_id)
        desc = {t: nx.descendants(G, t) for t in items}
        bad = [(a.node_name, b.node_name) for a, b in itertools.combinations(items, 2)
               if b not in desc[a] and a not in desc[b]]
        if bad:
            raise ValueError(f"{len(bad)} unordered collective pairs, e.g. {bad[:4]}")
        msg = f"{len(mc)} multicasts and {len(ch)} ChainGathers, every pair ordered by a path"
        if self.wo_push == "after_reduce":
            anc, after = nx.ancestors(G, self.m_made), nx.descendants(G, self.o_chain)
            early = [t for t, j in self.load_chunk.items() if j < self.WO0 and t not in anc]
            fetch = [t for g in range(self.G) for t in (self.kc_ready[g], self.vc_ready[g]) if t not in anc]
            late = [k for k in self.chips if self.prefetch[k] not in after]
            after_m = desc[self.m_made] if self.m_made in desc else nx.descendants(G, self.m_made)
            if early or fetch or late or self.o_chain not in after_m:
                raise ValueError(f"wo_push after_reduce: {len(early)} W_q / k / v ring loads and {len(fetch)} cache "
                                 f"fetches not before the statistics' ChainGather, {len(late)} W_O parts not after the "
                                 f"reduce's")
            msg += ("; every W_q / k / v chunk and cache tile is loaded before the first, every chip's W_O part waits "
                    "for the last")
        elif self.wo_push == "before_chains":
            anc = nx.ancestors(G, self.m_made)
            early = [t for t in self.load_chunk if t not in anc]
            fetch = [t for g in range(self.G) for t in (self.kc_ready[g], self.vc_ready[g]) if t not in anc]
            nwo = sum(1 for j in self.load_chunk.values() if j >= self.WO0)
            after_m = desc[self.m_made] if self.m_made in desc else nx.descendants(G, self.m_made)
            if early or fetch or nwo != self.G * self.NWS or self.o_chain not in after_m:
                raise ValueError(f"wo_push before_chains: {len(early)} ring loads and {len(fetch)} cache fetches not "
                                 f"before the statistics' ChainGather ({nwo} W_O ring loads)")
            msg += (f"; every ring chunk of every part ({len(self.load_chunk)} loads, {nwo} of them W_O's) and every "
                    f"cache tile is loaded before the first")
        else:
            msg += "; wo_push after_kv: the W_O pushes run beside both ChainGathers"
        print(f"[{self.TAG}] collectives: {msg}")
        return len(mc), len(ch)

    def _load(self, g, i, j, deps=()):
        ld, si, slab = super()._load(g, i, j, deps)
        if not hasattr(self, "load_chunk"):
            self.load_chunk = {}
        self.load_chunk[ld] = j                         # the ring load's chunk index in the cluster's image
        return ld, si, slab

    # ---- checks -------------------------------------------------------------------------------------------------------
    def _checks(self):
        """Byte for byte: y in the root memory chip's HBM (or the root's L3 row), the new key's chip's appended K and
        V^T tiles in HBM; with params checks stages first the query halves on the root and the last chip's cluster 1's
        half, the root's statistics (m* or (m*, l*)), and a far chip's partial."""
        d, R, gd, N = self.d, self.ROOT, self.gold, self.NEWCHIP
        u8 = lambda a: np.ascontiguousarray(a).view(np.uint8).reshape(-1)   # noqa: E731
        checks = []
        if self.checks_kind == "stages":
            qh = np.concatenate([self.q_half(h) for h in range(NHF)])
            last = self.chips[-1]
            stage = [("q_halves", R, 0, self.sv(R, 0, self.OFF_QIN), u8(qh)),
                     (f"q1_k{last:02x}", last, 1, self.sv(last, 1, self.OFF_X), u8(self.q_half(1)))]
            sn = "ml" if self.merge else "mstar"
            stage.append((sn, R, 0, self.sv(R, 0, self.OFF_MR), u8(np.asarray(gd[sn], np.float16))))
            far = self.order[len(self.order) // 2]
            stage.append((f"loc_k{far:02x}", far, 0, self.sv(far, 0, self.OFF_LOC), u8(gd["loc"][far])))
            for nm, k, c, src, img in stage:
                checks.append((nm, self.stash(nm, k, c, src, img.size, []), img, img.size))
        # hbm_checks stash: an HBM buffer through the root's L3 (one iDMA copy after the end stamp), else read in place
        hbm = (lambda nm, h, n: self.stash(nm, R, 0, h, n, [])) if self.hbm_checks == "stash" else \
            (lambda nm, h, n: h)                                                   # noqa: E731
        y = u8(d.y_raw.astype(np.float16))
        if self.y_to_hbm:
            checks.append(("y_hbm", hbm("y_hbm", self.h_yhbm, y.size), y, y.size))
        else:
            checks.append(("y_l3", self.l3("yrow", 0, R), y, y.size))
        base = self.h_kv[N].address + self.APP
        for c in range(NHF):
            kg = u8(self.k_tile(c, NEW))
            vg = u8(self.v_tile(c, NEW, keys=BC))
            checks.append((f"kapp_h{c}", hbm(f"kapp_h{c}", BingoMemFixedAddr(base + c * self.KB, mem_level="HBM"),
                                             kg.size), kg, kg.size))
            checks.append((f"vapp_h{c}", hbm(f"vapp_h{c}", BingoMemFixedAddr(base + NHF * self.KB + c * self.VB,
                                                                             mem_level="HBM"), vg.size), vg, vg.size))
        self.run_checks(checks)

    # ---- emission -------------------------------------------------------------------------------------------------------
    def finish(self, app_name, x_prefixes=("x_wait_",)):
        bad = [t.node_name for t in self.dfg.node_list if (t.kernel_name or "") in
               ("__snax_bingo_kernel_gemm_full", "__snax_bingo_kernel_gemm_minimal")]
        if bad:
            raise ValueError(f"gemm_full / gemm_minimal tasks in a GEMV block: {bad[:4]}")
        if self.per_kernel:
            if self.barrier_fence:                  # the fences first: the proofs and the core order see them
                self.fence_barriers()
                self.barrier_fence = False
            cross = self.check_window_local()
            if cross:
                raise ValueError(f"sync barrier: {len(cross)} dependency edges cross chips inside the window, e.g. "
                                 + "; ".join(f"{u.node_name} -> {v.node_name}" for u, v in cross[:4]))
            print(f"[{self.TAG}] sync barrier: {self.n_barrier} barriers a chip, no edge crosses chips inside the window")
        self._check_collectives()
        self._args_report()
        lmd.DBuild.finish(self, app_name, x_prefixes)

    def _args_report(self):
        """Each cluster's task args (the compiler's arena estimate: each struct 64-byte aligned), beside its static L1."""
        from bingo_args_phases import _args_bytes
        dev = {self.roles[r] for r in ("gemm", "simd", "xdma", "dm")}
        per = {}
        for t in self.dfg.node_list:
            if t.assigned_core_id in dev and (t.kernel_name or "").startswith("__snax"):
                key = (t.assigned_chiplet_id, t.assigned_cluster_id)
                per[key] = per.get(key, 0) + _args_bytes(t)
        (k, c), nb = max(per.items(), key=lambda kv: kv[1])
        print(f"[{self.TAG}] task args: at most {nb:,} B a cluster (chip {k:#04x} cluster {c}), estimate")


def _app_name(S):
    arms = "" if (S.x_cast, S.bcast_impl, S.reduce_impl, S.sync) == ("multicast", "bcast_l1", "chain", "edge") else \
        f" x_cast={S.x_cast} bcast={S.bcast_impl} reduce={S.reduce_impl} sync={S.sync}"
    arms += "" if S.cache_order == "own" else f" cache_order={S.cache_order}"
    arms += "" if S.hbm_checks == "direct" else f" hbm_checks={S.hbm_checks}"
    return (f"Llama-3-8B attention decode, key split (synthetic weights), one token over a {S.ctx}-position cache, "
            f"{len(S.chips)} chips x {S.NC} clusters, passes={S.passes} stats={'merge' if S.merge else 'max'} "
            f"wo_push={S.wo_push}" + arms)


def run():
    run_driver("llama3 attention decode (key split) workload generator", KsBuild, _app_name)
