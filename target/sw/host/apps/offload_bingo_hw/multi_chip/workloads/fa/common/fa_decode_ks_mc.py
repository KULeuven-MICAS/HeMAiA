# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite's layer-1 MLA attention as ONE DECODE STEP with the attention SPLIT BY KEY (flash decoding) and
every weight multiply on VersaCore's one-token GEMV ((1, 4, 32), __snax_bingo_kernel_gemv, INT4 in the paired layout),
on the 16 + 4 chiplet platform. The key-split core is fa_mc.py's (the 16 heads as the 16 FP16 lanes of one query tile,
two passes, m* exchanged, the partials folded into the root, normalised); the projections around it are made here, and
every input comes from HBM.

The token at position P (params cache, default 1,024) attends to keys 0 .. P: the P cached keys and its own. The data
and the scales are fa_decode_data.FaDecodeData's (a fresh causal prompt of P + 1 tokens, the token its last; rope_perm;
the mask as the score matmul's INT32 C); the golden is the arithmetic of the steps below (KsGolden).

THE MAPPING. Chip i (chips[i], i = 0 .. 15) owns head i, the cached key tiles i, i + 16, .. (64 keys each: one tile a
chip at P = 1,024) and W_O's columns of its two clusters; the root C00 (chips[0]) also owns the new key and its tile.
Each chip's memory chip pushes ONE ring holding all of the chip's chunks in the order they are needed (the push order
is the ring order).

    x          the token's INT8 row in a_row (value c at byte (c // 4) * 8 + c % 4, the other bytes zero; 4 KiB) in
               the HBM of the root's memory chip; that memory chip's push engine multicasts it into every chip's L3
               (fk_xland) behind one flag (fk_xtflag), and every cluster fetches it into L1
    new key    W_KVA's 18 chunks of 32 columns: chunk n on chip n's cluster 1 (n < 16) and on the clusters 1 of params
               kva_extra (the RoPE chunks 16, 17); a GEMV each, its FP16 row put into C00's L3 behind its flag. C00's
               cluster 1 waits for the 18, dequantises the row, RMSNorm of c (simd_rmsnorm, one row), RoPE of k_pe at
               P, then makes the piece of 16 keys whose row 0 is the token and the rest zero: its key rows (the row in
               row 0 of a zeroed FP16 A tile [16, 576], quantised in two parts, c at s_c and k_pe at s_kpe) are the
               first 16 keys of the new tile's K; its value columns (c~ in row 0 of a zeroed [16, 512] tile,
               transposed, the A layout, quantised) the first four columns of each 16-row block of the new tile's V
               (one 2-D copy)
    q          on every chip's cluster 0, head h: W_q's six chunks of the head (the two RoPE chunks first, then the four
               q_nope chunks), a GEMV each into the head's raw row; q_pe dequantised and rotated at P (rope_perm's
               halves: x cos + the other half x sin); q_nope spread into a_row, dequantised and quantised at s_qn in
               one SIMD pass (SimdMulI8 over the spread factors); the absorb q~ = q_nope W_UK,h, one GEMV (K 128,
               N 512), dequantised. The head's FP16 row [q~ | q_pe] (1,152 B) is put into C00's L3, row h, behind its
               flag
    query      C00's cluster 0 waits for the 16 rows, converts the FP16 rows [16, 576] to the A layout (xDMA) and
               quantises them in two parts (q~ at s_qt, q_pe at s_qpe): Q8, the query tile of 16 lanes, lane h head h,
               multicast into every chip's cluster-0 exchange buffer (fa_mc's broadcast, cluster 0 only); C00's
               cluster 1 copies it from its cluster 0
    pass 1     every chip's cluster 0 over its cached tiles (K from the ring) and C00's cluster 1 over the new tile (K
               made on the device, its keys after P masked by the QK's C): S^T = K Q^T and the FA softmax with its own
               seed, the shard's lanewise maximum; C00 takes the maximum of its two clusters' (the softmax over the two
               beats)
    m*         params m_exchange gather (default): every chip puts its maximum beat into C00's L3 behind its flag; C00
               waits for the 16, gathers them into one tile and runs the softmax's lanewise maximum over it. chain:
               fa_mc's ChainGather with the Elementwise junction's FP16 MAX. m* is multicast into every chip's pass-2
               arena (cluster 0 only; C00's cluster 1 copies it)
    pass 2     the softmax seeded at m* and PV (V from the ring, or the new tile's), O^T leaving the D port as FP16;
               each cluster's partial (O^T, then its row sums) at its chip's OFF_LOC; C00 adds its two clusters'
    reduce     one ChainGather through every chip (chiplet_mc's fold order) with the Elementwise junction's FP16 ADD
               into C00; C00 normalises (o~ = O c / l), transposes ([16, 512], a head a row), spreads every head's row
               into a_row, quantises at s_ot and multicasts the 16 heads' rows (16 KiB) into every chip's L3
    W_UV       every chip's cluster 0 fetches its head's row and runs o_h = o~_h W_UV,h, one GEMV (K 512, N 128),
               spread into a_row, dequantised and quantised at s_o in one pass: o8_h, put into C00's L3 at its head's
               place behind its flag
    W_O        C00 waits for the 16 and multicasts every head's o8 (one a_row of 4 KiB) into every chip's L3; every
               cluster runs W_O BY COLUMNS: its 64 columns of y over the full depth of 2,048, two GEMVs of 32 columns,
               one exact INT32 sum each, and puts its 64 FP16 values into C00's L3 row behind its flag; C00 waits for
               the 32 and writes y (FP16 [2048], before W_O's per-column dequantisation, as the prefill leaves it)
               into the HBM of the root's memory chip (stage_store)
    append     C00's cluster 1 writes the new piece into its cache shard in HBM at the new tile's slot: the key rows (a
               1-D copy), then the value columns (32 runs of 256 B, one 16-row block apart), after y

THE SPLIT AT P = 1,024. 1,025 keys: 16 whole cached tiles of 64 and the token's. One cached tile a chip on its cluster
0 and the new tile on C00's cluster 1: every attention cluster runs one tile, and the pass-1 maxima and the partials of
the 16 chips meet once each (a chain of 16 entries, BINGO_XDMA_CHAIN_MAX). Splitting a tile's PV by latent halves over
a chip's two clusters would repeat its QK and its key stream on both; 32-key tiles would add a chip fold of 16 KiB
partials on every chip; either halves only the array passes of one tile's matmuls, whose dispatch does not shrink with
them. More cached tiles (params cache a multiple of 1,024) go to the same clusters, two passes over each shard as
fa_mc's recurrence.

THE CACHE. Chip i's shard in its memory chip's HBM (fk_kv_k<chip>): its tiles in order, each [K tile, the A layout of
[64, 576] | V tile, the A layout of V^T [512, 64]], the layouts the attention reads; after them on C00 the slot of tile
P / 64, which holds the appended piece. Each tile streams through the chip's ring like a weight chunk, K and V one chunk
each: no chip's cache crosses another memory chip's links, and no multicast carries it.

THE ORDERS. The compiler orders every device core's program (mini_compiler passes/bingo_dfg_schedule.py), each cluster's
ring loads in the ring's order; a ring load blocks its DM core until its chunk lands. A chip's ring order: W_KVA's
chunk(s) (cluster 1), W_q x 6 and W_UK (cluster 0), its cached K tiles, then its V tiles, W_UV, cluster 1's two W_O
chunks, cluster 0's two W_O chunks: cluster 0's last W_O chunk is the chip's last push.

THE COLLECTIVES. Every two multicasts and every multicast / chain pair are ordered by the graph's dependency and flag
edges (_check_collectives proves it): x (the memory chip's, ahead of the root's ring on its engine), the query, m*, o~
and the o8 row are each made on C00 after every chip's part of the step before has landed there. params
chains_after_stream (default true): every chip's partial is handed to the chain only after its DM core has loaded the
chip's last ring chunk, so the chain runs after every push of every memory chip has landed and no ChainGather runs
beside a memory-chip push (m_exchange gather makes no other chain; m_exchange chain's maximum chain is held the same
way). Every write into HBM (y, the append) follows every collective and every chip's last ring load. The gathers (the
W_KVA rows, the query rows, the maxima, the o8 rows, the y columns) are unicast iDMA puts behind flags.

IN A LAYER (hooks): x_multicast(go, bcast) returns the task the x fetches are ordered after (a
layer's input norm puts x into fk_xland on every chip behind fk_xtflag: the a_row above, INT8 at the block's s_x,
4,096 B); stage_store(yw) stores y (a layer copies the root's L3 row fk_yrow into its own buffer instead, after `yw`);
kv_append(deps) makes the append (a layer may make it at its own end). What a layer reads: d (FaDecodeData), gold
(y_raw), _chains, _y_stores, append, XB, P.

PARAMETERS (params.hjson): cache (P, a multiple of 64 x 16), n_calib, m_exchange (gather | chain), chains_after_stream,
kva_extra (the chips whose cluster 1 takes W_KVA's chunks 16, 17; hex strings), checks (y | stages), stats (max) and
passes (2) (the only values: the statistics' merge and the one-pass rebase are not built here), w_chunk_bytes (one K
tile, 36,864), weight_ring (slots, batch, head, batch_head, queue), and the platform's and the graph's switches (static_l1,
l1_capacity, l1_order, compact_id_maps, gid_lists, per_chip_image, desc_in_image, fast_host_init, d2d_ddr,
d2d_rx_yield, relay_per_gather).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# fa_mc puts util/sim/multichip and mini_compiler on sys.path
from fa_mc import (ARENA, BC, BEAT, BR, D_QK, D_V, EW_F16_MAX, FA, LRUN, MRUN, PVK, REPO, XFER, FaBuild,  # noqa: E402
                   cm, run_driver)
import fa_decode_data as fdd  # noqa: E402

from bingo_kernel_args import (  # noqa: E402
    HostBingoKernelHbmBcastArgs, SnaxBingoKernelGemmFaQkArgs,
    SnaxBingoKernelGemvArgs, SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdma2dCopyArgs,
    SnaxBingoKernelIdmaFetchFlaggedArgs, SnaxBingoKernelIdmaPutFlaggedArgs, SnaxBingoKernelIdmaRingLoadArgs,
    SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelSimdFp16ToInt8Args, SnaxBingoKernelSimdMlaNormaliseArgs,
    SnaxBingoKernelSimdMulF16Args, SnaxBingoKernelSimdMulI8Args, SnaxBingoKernelSimdRmsnormArgs,
    SnaxBingoKernelWaitFlagsArgs, SnaxBingoKernelXdma6dArgs, SnaxBingoKernelXdmaChainGatherArgs,
    SnaxBingoKernelXdmaMemsetArgs, SnaxBingoKernelXdmaTranspose2dArgs, simd_beats, simd_fp16_lanes)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from libs import at_offset  # noqa: E402
from libs.blocks import WeightRings  # noqa: E402
from libs.comm.nest import convert_args  # noqa: E402

HIDDEN, KV_RANK, ROPE, Q_NOPE, V_HEAD = 2048, 512, 64, 128, 128
Q_HEAD = Q_NOPE + ROPE
QC = 32                              # a GEMV chunk: 32 columns (the paired INT4 layout's unit)
WCH = HIDDEN * QC // 2               # a chunk of 32 columns over the hidden depth, INT4
ATOM, LANES = 8, 8                   # the xDMA's lane (an 8-byte atom) and its lanes a beat
KB, VB = BC * D_QK, D_V * BC         # a K tile, a V tile
SB = BEAT + BC * BR * 2              # a score tile with its prefix beat
QB = BR * D_QK                       # the query tile Q8, A layout
OB = D_V * BR * 2                    # O^T, FP16
PB = OB + XFER                       # a partial: O^T, then its row sums
PK, PVB = 16 * D_QK, D_V * 16        # the new piece: its key rows, its value columns
QROW = 2 * D_QK                      # a head's query row, FP16 [q~ | q_pe]
XB = 2 * HIDDEN                      # x in a_row
LAT = 2 * KV_RANK                    # a head's o~ in a_row
OAR = 2 * V_HEAD                     # a head's o8 in a_row
WQ_ORDER = (4, 5, 0, 1, 2, 3)        # W_q's chunks of a head as the ring streams them: q_pe's two, then q_nope's four
N_KVA = D_QK // QC                   # W_KVA's chunks
LINK_BW = 231.2e9 / 8                # bytes a second of one memory-chip link (the eight links' aggregate over eight)

# the constants of a cluster 0 (FP16, in L1 before the window): q_nope's factors spread into a_row, q_pe's, the RoPE
# tables at P, q~'s factors, W_UV's factors spread into a_row
C0_NSP, C0_FPE, C0_COS, C0_SIN, C0_FQT, C0_UVSP = 0, 512, 640, 768, 896, 1920
C0_BYTES = 2432
# the constants of the new key's cluster: W_KVA's factors, the RoPE tables at P, the new tile's mask (INT32 [64, 16])
C1_FKV, C1_COS, C1_SIN, C1_MASK = 0, 1152, 1280, 1408
C1_BYTES = C1_MASK + BC * BR * 4


def a_row(v):
    """A row of values in a_row: value c at (c // 4) * 8 + c % 4, the other places zero (the GEMV's a_blk 8)."""
    v = np.asarray(v).reshape(-1)
    c = np.arange(v.size)
    out = np.zeros(2 * v.size, dtype=v.dtype)
    out[(c // 4) * 8 + c % 4] = v
    return out


def f32bits(v):
    return int(np.asarray(v, dtype=np.float32).reshape(()).view(np.uint32))


class KsGolden:
    """The device's arithmetic of the step for FaDecodeData d (hwmodel's operators, fp's roundings), on the mapping's
    shards and fold order: units [(chip, cluster, [tile, ..])] (tile NTC the new one), fold the chips in the order the
    reduce adds them, NTC the new tile's index."""

    def __init__(self, d, units, chips, fold, NTC):
        fp, hm, simd, lay, mesh = d.fp, d.hm, d.simd, d.layout, d.mesh
        F16, F32, F64 = np.float16, np.float32, np.float64
        P, H = d.P, d.H
        mm = lambda a, b: np.rint(np.asarray(a, F64) @ np.asarray(b, F64)).astype(np.int64)    # noqa: E731
        cos, sin = np.asarray(d.cos_rep[0], F16), np.asarray(d.sin_sgn[0], F16)                 # position P
        half = ROPE // 2

        def rope(x):
            xs = np.concatenate([x[half:], x[:half]])
            return fp.add16(fp.mul16(x, cos), fp.mul16(xs, sin))

        self.cos, self.sin = cos, sin
        x8 = self.x8 = np.asarray(d.hq_all[P], np.int8)
        # ---- the query: per head the q path, then the tile of the 16 heads -------------------------------------------
        self.qrow, q8 = [], []
        for h in range(H):
            q_raw = fp.d_port(mm(x8, d.wq.q[:, h * Q_HEAD:(h + 1) * Q_HEAD]), d.k_x)
            q16 = fp.mul16(q_raw, d.q_scale(h))
            qpe = rope(q16[Q_NOPE:])
            qn8 = fp.quant_i8(q16[:Q_NOPE], d.s_qn.inv)
            qt16 = fp.mul16(fp.d_port(mm(qn8, d.wuk.q[h]), d.k_uk), d.qt_scale(h))
            self.qrow.append(np.concatenate([qt16, qpe]).astype(F16))
            q8.append(np.concatenate([fp.quant_i8(qt16, d.s_qt.inv), fp.quant_i8(qpe, d.s_qpe.inv)]))
        self.Q8 = np.stack(q8).astype(np.int8)                                      # [16 lanes, 576]
        self.q8_tile = lay.to_a(self.Q8, mesh).astype(np.int8)
        # ---- the new key: W_KVA by GEMV, dequantised, RMSNorm of c, RoPE of k_pe, quantised --------------------------
        self.kv_raw = fp.d_port(mm(x8, d.wkva.q), d.k_x)
        kv16 = fp.mul16(self.kv_raw, d.kva_scale())
        cn16 = hm.rmsnorm(kv16[:KV_RANK])
        kpe16 = rope(kv16[KV_RANK:])
        c8, kpe8 = fp.quant_i8(cn16, d.s_c.inv), fp.quant_i8(kpe16, d.s_kpe.inv)
        self.new_row = np.concatenate([c8, kpe8]).astype(np.int8)
        if not (np.array_equal(self.new_row, d.rows_all[P]) and np.array_equal(c8, d.c8_all[P])):
            raise AssertionError("the GEMV path's new key differs from FaDecodeData's key of position P")
        Kn, Vn = np.zeros((BC, D_QK), np.int8), np.zeros((BC, D_V), np.int8)
        Kn[0], Vn[0] = self.new_row, c8
        self.newK, self.newV = Kn, Vn
        self.ktile_new = lay.to_a(Kn, mesh).astype(np.int8)                         # [64, 576], A layout
        self.vtile_new = lay.to_a(np.ascontiguousarray(Vn.T), mesh).astype(np.int8)  # V^T [512, 64], A layout
        self.piece_k = lay.to_a(Kn[:16], mesh).astype(np.int8)
        self.piece_v = lay.to_a(np.ascontiguousarray(Vn[:16].T), mesh).astype(np.int8)
        if not np.array_equal(self.ktile_new[:PK], self.piece_k):
            raise AssertionError("a K tile's first 16 keys are not its first 9 KiB (A layout)")
        vt = self.vtile_new.reshape(D_V // 16, 16 * BC)[:, :256].reshape(-1)
        if not np.array_equal(vt, self.piece_v):
            raise AssertionError("a piece's value columns are not the first 256 B of every row block of its V tile")
        # ---- the key-split attention: two passes, m* the lanewise maximum, the partials folded -----------------------
        lane_max = lambda a: simd.reduce_lanewise(a, simd.MAX)                       # noqa: E731
        seed = np.full(BR, -65504.0, dtype=F16)
        a32 = F32(d.a_exp)
        Qt = self.Q8.astype(F64).T
        bias = np.zeros((BC, BR), dtype=np.int64)
        bias[1:] = d.mask_bias                                                      # every key after P, every lane
        self.mask = bias.astype(np.int32)

        def tile(t):
            if t == NTC:
                K, V, b = Kn, Vn, bias
            else:
                K, V, b = d.rows_all[t * BC:(t + 1) * BC], d.c8_all[t * BC:(t + 1) * BC], 0
            S16 = fp.d_port(mm(K, Qt) + b, d.k_s)
            if t != NTC and not np.all(np.isfinite(S16)):
                raise FloatingPointError(f"tile {t}: a score overflows FP16 after the D-port shift")
            return S16, np.asarray(V, np.int64)

        tiles = {t: tile(t) for _, _, ts in units for t in ts}
        self.m_u = {}
        for k, c, ts in units:
            m16 = seed
            for t in ts:
                m16 = lane_max(np.stack([lane_max(tiles[t][0]), m16]))
            self.m_u[(k, c)] = m16
        self.mstar = lane_max(np.stack([lane_max(np.stack(list(self.m_u.values()))), seed]))
        parts = {}
        for k, c, ts in units:
            m16, l16, o = self.mstar, np.zeros(BR, dtype=F16), None
            for t in ts:
                S16, V = tiles[t]
                mnew = lane_max(np.stack([lane_max(S16), m16]))
                corr16 = hm.exp16(fp.add16(m16, -mnew), a32)
                p16 = hm.exp16(fp.add16(S16, -mnew[None, :]), a32)
                rsum16 = simd.reduce_lanewise(p16, simd.ADD)
                l16 = simd.reduce_lanewise(np.stack([rsum16, fp.mul16(corr16, l16)]), simd.ADD)
                pv = V.T @ fp.quant_i8(p16, hm.P8_SCALE).astype(np.int64)
                o = pv if o is None else np.clip(np.rint(o.astype(F64) * corr16.astype(F64)[None, :]),
                                                 -2**31, 2**31 - 1).astype(np.int64) + pv
                m16 = mnew
            parts[(k, c)] = (fp.d_port(o, d.k_o), l16)
        self.loc = {}
        for k in chips:
            us = [(k, c) for c in range(2) if (k, c) in parts]
            o16, l16 = parts[us[0]]
            for u in us[1:]:
                o16, l16 = fp.add16(o16, parts[u][0]), fp.add16(l16, parts[u][1])
            self.loc[k] = (o16, l16)
        Ot, lt = self.loc[fold[0]]
        for k in fold[1:]:
            Ot, lt = fp.add16(Ot, self.loc[k][0]), fp.add16(lt, self.loc[k][1])
        t16 = simd.stream_map(lt, d.a_n, 0.0, simd.RSQRT)
        ot = fp.mul16(Ot, fp.mul16(t16, t16)[None, :])                               # [512, 16]
        self.xt = np.ascontiguousarray(ot.T).astype(F16)                             # [16, 512], a head a row
        # ---- W_UV per head, W_O by columns -------------------------------------------------------------------------
        self.o8 = []
        for h in range(H):
            ot8 = fp.quant_i8(self.xt[h], d.s_ot.inv)
            o16 = fp.mul16(fp.d_port(mm(ot8, d.wuv.q[h]), d.k_uv), d.uv_scale(h))
            self.o8.append(fp.quant_i8(o16, d.s_o.inv))
        self.o8_all = np.concatenate(self.o8).astype(np.int8)                       # [2048]
        self.y_raw = fp.d_port(mm(self.o8_all, d.wo.q), d.k_wo).astype(F16)          # [2048], before W_O's factors

    def part_bytes(self, k):
        """Chip k's partial as the chain reads it: O^T, then its row-sum beat twice (lrun and lnew)."""
        o16, l16 = self.loc[k]
        l16 = np.asarray(l16, np.float16).reshape(-1)
        return np.concatenate([np.asarray(o16, np.float16).reshape(-1), l16, l16])

    def report(self, d):
        y = d.fp.mul16(self.y_raw, d.out_scale()).astype(np.float64)
        ref = np.asarray(d.y_ref, dtype=np.float64)
        cos = float(y @ ref / max(np.linalg.norm(y) * np.linalg.norm(ref), 1e-30))
        rel = float(np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-30))
        return [f"key split, GEMV projections: device y of the token vs the float attention: cosine {cos:.5f}, "
                f"relative L2 error {rel:.4f}"]


class FaDecodeKs(FaBuild):
    """The key-split MLA decode step with GEMV projections (the module's docstring). run_driver makes the program in
    four steps: __init__ (the parameters), mem_handles() (the data and the memory, no task), build() (the task graph:
    every task and edge) and finish() (the compile into the C header and the data image)."""
    TAG = "fk"
    QC, KC = 0, 1                    # every chip's query / attention cluster, and its W_KVA cluster
    # x_on_host: x's row sits in the root memory chip's HBM and its push engine multicasts it (x_multicast); y_to_hbm:
    # the root writes y into that HBM (stage_store). A whole layer makes x on the device and keeps y on the root
    x_on_host = True
    y_to_hbm = True

    def __init__(self, args, p, plat, hw, shared=None):
        """The parameters; no data, memory or task yet. ChipletBuild.__init__, not FaBuild's (fa_mc's shards and data
        are not this step's): the platform's and the graph's switches, the chips, the empty DFG, the data staging (a
        layer's block shares the layer's). Then the step's: the cache's length P and its tiles, m_exchange,
        chains_after_stream, the checks, the data's inputs, the head of each chip, the attention's units and the
        chunks of W_KVA a chip."""
        cm.ChipletBuild.__init__(self, args, p, plat, hw, shared=shared)
        if self.T != 1:
            raise ValueError("params tokens: the decode runs one token")
        if BR != simd_fp16_lanes():
            raise ValueError(f"the softmax takes one query per FP16 lane: {simd_fp16_lanes()} on this cluster, BR {BR}")
        if self.wbits != 4:
            raise ValueError("params wbits 4: the GEMV streams paired INT4")
        for key, want in (("bcast", "bcast_l1"), ("reduce", "chain"), ("sync", "edge"), ("stats", "max"),
                          ("passes", 2)):
            if key in p and p[key] != want:
                raise ValueError(f"params {key}={p[key]!r}: the key-split decode builds {want!r}")
        if self.chunk < KB:
            raise ValueError(f"params w_chunk_bytes={self.chunk}: a K tile is {KB} B")
        self.P = int(p.get("cache", 1024))
        nchip = len(self.chips)
        if self.P % (BC * nchip):
            raise ValueError(f"params cache={self.P}: whole tiles of {BC} keys, the same number on each of {nchip} "
                             f"chips")
        self.NTC = self.P // BC                    # the cached tiles; the new tile is tile NTC
        self.m_exchange = str(p.get("m_exchange", "gather"))
        if self.m_exchange not in ("gather", "chain"):
            raise ValueError(f"params m_exchange={self.m_exchange!r}: gather or chain")
        self.after_stream = bool(p.get("chains_after_stream", True))
        self.checks_kind = str(p.get("checks", "y"))
        if self.checks_kind not in ("y", "stages"):
            raise ValueError(f"params checks={self.checks_kind!r}: y or stages")
        # the data's inputs: the cache of hidden states, the checkpoint's weights, the calibration tokens
        self.kv_path = os.path.join(REPO, p.get("kv_cache", "target/sim/automation/sweep/llm_blocks/fa/cache/"
                                                            "dsv2_l1_kv_8193.npz"))
        self.w_cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/common/weights/cache"))
        self.n_calib = int(p.get("n_calib", 32))
        R = self.ROOT
        self.head = {k: i for i, k in enumerate(self.chips)}
        self.tiles = {k: [i + nchip * t for t in range(self.NTC // nchip)] for i, k in enumerate(self.chips)}
        self.units = [(k, self.QC, self.tiles[k]) for k in self.chips] + [(R, self.KC, [self.NTC])]
        # W_KVA's chunks: chunk n on chip n's cluster 1, the rest on params kva_extra (default _kva_extra_default)
        extra = [int(str(x), 16) for x in p.get("kva_extra", [])] or self._kva_extra_default(N_KVA - nchip)
        if len(extra) != N_KVA - nchip or any(x not in self.chips for x in extra):
            raise ValueError(f"params kva_extra={p.get('kva_extra')}: {N_KVA - nchip} compute chips")
        self.kva_extra = extra
        self.kva = {k: [i] for i, k in enumerate(self.chips)}
        for n, k in enumerate(extra, start=nchip):
            self.kva[k].append(n)

    def mem_handles(self):
        """The data and the memory; no task yet. FaBuild.mem_handles is not run, as FaBuild.__init__ is not: the data
        (fa_decode_data.FaDecodeData) and its golden (KsGolden) with their report and the cross-check of head 0's Q8;
        _stage (x, y, the gathers' and multicasts' areas in L3, every chip's weights, cache shard and constants in its
        memory chip's HBM); one weight ring a chip (_make_ring); every cluster's exchange buffer (its offsets: the
        query, the flags, the arenas, the partial)."""
        R, nchip, extra = self.ROOT, len(self.chips), self.kva_extra
        d = self.d = fdd.FaDecodeData(self.kv_path, self.w_cache, cache=self.P, bc=BC, n_calib=self.n_calib,
                                      rope_perm=True, mask_c=True, wbits=4)
        if d.H != nchip:
            raise ValueError(f"{nchip} chips: the attention has {d.H} heads, one a chip")
        self.gold = KsGolden(d, self.units, self.chips, self.fold_order(), self.NTC)
        sc = lambda q: f"{q.s:.4g}"     # noqa: E731
        print(f"[{self.TAG}] DeepSeek-V2-Lite layer 1 MLA, one decode token at position {self.P} over keys "
              f"0..{self.P}, {d.H} heads, INT{d.wbits} per column; D-port shifts k_x={d.k_x} k_uk={d.k_uk} "
              f"k_uv={d.k_uv} k_s={d.k_s} k_o={d.k_o} k_wo={d.k_wo}; scales s_x={sc(d.s_x)} s_qn={sc(d.s_qn)} "
              f"s_qt={sc(d.s_qt)} s_qpe={sc(d.s_qpe)} s_c={sc(d.s_c)} s_ot={sc(d.s_ot)} s_o={sc(d.s_o)}")
        for line in self.gold.report(d):
            print(f"[{self.TAG}] {line}")
        self._cross_check()
        print(f"[{self.TAG}] key split: {self.NTC} cached tiles of {BC} keys, {self.NTC // nchip} a chip on its "
              f"cluster 0, the new tile (key {self.P}) on {R:02x}'s cluster 1; W_KVA's chunks 16.. on "
              f"{[f'{k:02x}' for k in extra]}; m* by {self.m_exchange}; fold order {[f'{k:02x}' for k in self.order]}")
        self._stage()
        self._make_ring()
        # ---- every cluster's exchange buffer, on whole transfer units -----------------------------------------------
        self.OFF_X, self.OFF_FLAG, self.OFF_MFLAG = 0, QB, QB + XFER
        self.OFF_MX = QB + 2 * XFER                  # an FA arena: the pass-1 arena of a chip's cluster 0 (the root's:
        self.OFF_MLOC = self.OFF_MX + MRUN           # its two clusters' maximum), whose mrun is the chip's maximum
        self.OFF_A2 = self.OFF_MX + ARENA            # the pass-2 arena: m* lands in its mrun
        self.OFF_M = self.OFF_A2 + MRUN
        self.OFF_MR = self.OFF_A2 + ARENA            # the root's arena of m*
        self.OFF_LOC = self.OFF_MR + ARENA           # the chip's partial for the chain
        self.make_xch(self.OFF_LOC + PB, PB, dst_lead=BEAT)

    def _kva_extra_default(self, n):
        """One chip a memory link, the links of the memory chips other than the root's (whose links also carry x and the
        root's multicasts) in link order, each the chip nearest to its memory chip."""
        R, out = self.ROOT, []
        for ln in sorted(set(self.link.values())):
            if ln[0] == self.mem[R] or len(out) == n:
                continue
            out.append(min((k for k in self.chips if self.link[k] == ln), key=lambda k: (cm.dist(k, self.mem[k]), k)))
        return out

    def _cross_check(self):
        """Head 0's Q8 against FaDecodeData's own chain of the head (the 16-row prefill arithmetic, its row 0)."""
        ref = np.asarray(self.d.head(0)["q8"][0], np.int8)
        if not np.array_equal(ref, self.gold.Q8[0]):
            raise AssertionError("head 0's Q8 on the GEMV path differs from FaDecodeData.head(0)")

    # ---- data --------------------------------------------------------------------------------------------------------
    def _stage(self):
        st, d, gold, T = self.st, self.d, self.gold, self.TAG
        R, nchip = self.ROOT, len(self.chips)
        loc = lambda k: (self.mem[k] >> 4, self.mem[k] & 15)       # noqa: E731
        f16 = lambda a: np.asarray(a, dtype=np.float16).reshape(-1).view(np.uint8)    # noqa: E731
        # x: the token's INT8 row in a_row in the root memory chip's HBM; its landing area and flag in every chip's L3
        if self.x_on_host:
            self.h_xhbm = st.put_hbm(f"{T}_xhbm", a_row(gold.x8), mem_chip=loc(R))
        st.put_zeros(f"{T}_xland", "int8_t", XB)
        st.put_zeros(f"{T}_xtflag", "uint8_t", 64)
        # y: FP16 [2048] in the root memory chip's HBM; the row the clusters fill in the root's L3, a flag each
        if self.y_to_hbm:
            self.h_yhbm = st.put_hbm(f"{T}_yhbm", np.zeros(2 * HIDDEN, dtype=np.uint8), mem_chip=loc(R))
        st.put_zeros(f"{T}_yrow", "uint8_t", 2 * HIDDEN)
        st.put_zeros(f"{T}_yflag", "uint8_t", self.N_CL * 64)
        # the gathers into the root's L3: the W_KVA rows, the query rows, the maxima, the o8 rows; the multicasts'
        # landing areas in every chip's L3: o~ and the o8 row
        for name, nb in (("kvraw", N_KVA * 2 * QC), ("kvrflag", N_KVA * 64), ("qrows", nchip * QROW),
                         ("qflag", nchip * 64), ("mstg", nchip * XFER), ("mflag", nchip * 64), ("olat", nchip * LAT),
                         ("olflag", 64), ("oland", nchip * OAR), ("oflag", nchip * 64), ("omflag", 64)):
            st.put_zeros(f"{T}_{name}", "uint8_t", nb)
        # per chip, in its memory chip's HBM: its weights in ring order, its cache shard, its constants
        self.w_off, self.h_w, self.h_kv, self.h_c0, self.kv_new = {}, {}, {}, {}, None
        for i, k in enumerate(self.chips):
            h = self.head[k]
            chunks = [(f"kva{n}", d.wblob(d.wkva.q[:, QC * n:QC * (n + 1)])) for n in self.kva[k]]
            chunks += [(f"wq{c}", d.wblob(d.wq.q[:, h * Q_HEAD + QC * c:h * Q_HEAD + QC * (c + 1)])) for c in WQ_ORDER]
            chunks += [("wuk", d.wblob(d.wuk.q[h])), ("wuv", d.wblob(d.wuv.q[h]))]
            for c in (self.KC, self.QC):
                for s in range(2):
                    c0 = 64 * (i * self.NC + c) + QC * s
                    chunks.append((f"wo{c}{s}", d.wblob(d.wo.q[:, c0:c0 + QC])))
            self.w_off[k], off = {}, 0
            for nm, b in chunks:
                if b.size != WCH:
                    raise ValueError(f"chunk {nm}: {b.size} B, not {WCH}")
                self.w_off[k][nm] = (off, b.size)
                off += b.size
            self.h_w[k] = st.put_hbm(f"{T}_w_k{k:02x}", np.concatenate([b for _, b in chunks]).astype(np.int8),
                                     mem_chip=loc(k))
            tiles = []
            for t in self.tiles[k]:
                tiles += [d.layout.to_a(d.rows_all[t * BC:(t + 1) * BC], d.mesh),
                          d.layout.to_a(np.ascontiguousarray(d.c8_all[t * BC:(t + 1) * BC].T), d.mesh)]
            if k == R:
                self.kv_new = len(self.tiles[k]) * (KB + VB)          # the next tile's slot: the appended piece
                tiles.append(np.zeros(KB + VB, dtype=np.int8))
            self.h_kv[k] = st.put_hbm(f"{T}_kv_k{k:02x}", np.concatenate(tiles).astype(np.int8), mem_chip=loc(k))
            qs = np.asarray(d.q_scale(h), np.float16)
            c0b = np.concatenate([f16(a_row(qs[:Q_NOPE])), f16(qs[Q_NOPE:]), f16(gold.cos), f16(gold.sin),
                                  f16(d.qt_scale(h)), f16(a_row(np.asarray(d.uv_scale(h), np.float16)))])
            if c0b.size != C0_BYTES:
                raise ValueError(f"cluster 0's constants are {c0b.size} B, not {C0_BYTES}")
            self.h_c0[k] = st.put_hbm(f"{T}_c0_k{k:02x}", c0b, mem_chip=loc(k))
        c1b = np.concatenate([f16(d.kva_scale()), f16(gold.cos), f16(gold.sin), gold.mask.reshape(-1).view(np.uint8)])
        if c1b.size != C1_BYTES:
            raise ValueError(f"the new key's constants are {c1b.size} B, not {C1_BYTES}")
        self.h_c1 = st.put_hbm(f"{T}_c1_k{R:02x}", c1b, mem_chip=loc(R))

    def ring_order(self, k):
        """Chip k's ring: (name, HBM address, bytes, the cluster that reads it), in push order."""
        w, kv = self.h_w[k].address, self.h_kv[k].address
        wo = lambda nm: (nm, w + self.w_off[k][nm][0], self.w_off[k][nm][1])       # noqa: E731
        out = [wo(f"kva{n}") + (self.KC,) for n in self.kva[k]]
        out += [wo(f"wq{c}") + (self.QC,) for c in WQ_ORDER] + [wo("wuk") + (self.QC,)]
        out += [(f"K{t}", kv + i * (KB + VB), KB, self.QC) for i, t in enumerate(self.tiles[k])]
        out += [(f"V{t}", kv + i * (KB + VB) + KB, VB, self.QC) for i, t in enumerate(self.tiles[k])]
        out += [wo("wuv") + (self.QC,)]
        out += [wo(f"wo{c}{s}") + (c,) for c in (self.KC, self.QC) for s in range(2)]
        return out

    def _make_ring(self):
        """One ring a chip (every chunk of both clusters, in ring order), pushed by its memory chip's engine; each run
        posted by the DM core of the cluster that reads its first chunk (WeightRings post_chunk, made by load() in ring
        order), the queue's weight_ring.queue entries the ring's share; each chunk read by one cluster."""
        p, st, T = self.p, self.st, self.TAG
        wr = cm.ring_params(p)
        self.ring_queue = int(wr.get("queue", 16))
        if not 2 <= self.ring_queue <= 32:
            raise ValueError(f"params weight_ring.queue={self.ring_queue}: 2..32 entries (two a run, one ring)")
        self.n_slots, self.ring_batch = int(wr.get("slots", 16)), int(wr.get("batch", 4))
        self.ring_head, self.ring_batch_head = int(wr.get("head", 0)), int(wr.get("batch_head", 0))
        seq = np.zeros((1024, 8), dtype=np.uint64)
        seq[:, 0] = np.arange(1024, dtype=np.uint64)
        self.h_seq = {m: st.put_hbm(f"{T}_ring_seq_m{m:02x}", seq.view(np.int8).reshape(-1), mem_chip=(m >> 4, m & 15))
                      for m in sorted(set(self.mem.values()))}
        self.h_flags = st.put_zeros(f"{T}_ring_flags", "uint8_t", self.n_slots * 64)
        self.h_wslots = st.put_zeros(f"{T}_wring_slots", "uint8_t", self.n_slots * self.slot_bytes, preload=True)
        self.rings, self.tk, self.ring_bytes = {}, {}, {}
        self.ring_posted = {k: 0 for k in self.chips}       # the chunks posted, in ring order
        self.last_chunk = {k: None for k in self.chips}
        for k in self.chips:
            rings = self.rings[k] = WeightRings(
                slots=self.h_wslots, flags=self.h_flags, n_rings=1, n_slots=self.n_slots, slot_bytes=self.slot_bytes,
                batch=self.ring_batch, head=self.ring_head, batch_head=self.ring_batch_head)
            order = self.ring_order(k)
            if len(order) > self.n_slots:
                raise ValueError(f"params weight_ring.slots={self.n_slots}: chip {k:02x} streams {len(order)} chunks "
                                 f"(every chunk keeps its slot)")
            self.tk[k] = {}
            for nm, src, nb, c in order:
                if nb > self.slot_bytes:
                    raise ValueError(f"chunk {nm}: {nb} B over a {self.slot_bytes}-B slot")
            self.ring_bytes[k] = sum(nb for _, _, nb, _ in order)
            rings.push_from(memchip=self.mem[k], engine=self.engine[k], seq=self.h_seq[self.mem[k]], edge=self.edge,
                            depth=self.ring_queue)
            self.queue_poster((self.mem[k], self.engine[k]), f"{T} ring, chip {k:#04x}", self.ring_queue,
                              rings.queue_span)
        self.last_name = {k: self.ring_order(k)[-1][0] for k in self.chips}

    # ---- nodes -------------------------------------------------------------------------------------------------------
    def nd(self, k, c, role, name, kernel, args, deps=()):
        return self.node(k, c, self.roles[role], f"{name}_c{c}", kernel, args,
                         list(dict.fromkeys(t for t in deps if t is not None)))

    def l1(self, k, c, name, nb):
        return self.cx_chip(k).at(c).l1(f"{self.TAG}_{name}", nb)

    def l3(self, name, off=0, chip=None):
        """An L3 symbol of this block at byte `off`, as chip `chip` names it (None: its own)."""
        return BingoMemSymbol(f"{self.TAG}_{name}", offset=off, chip_id=chip)

    def far(self, k):
        """How chip k names the root's L3."""
        return None if k == self.ROOT else self.ROOT

    def load(self, k, name, dst, deps=()):
        """Chip k's ring chunk `name` into the L1 buffer dst, on its reader's DM core: waits for the chunk's flag. Every
        chunk of the ring up to this one is posted first, in ring order (WeightRings post_chunk on the DM core of the
        cluster that reads the run's first chunk), and the load follows its post."""
        order = self.ring_order(k)
        i = next(j for j, o in enumerate(order) if o[0] == name)
        while self.ring_posted[k] <= i:
            nm, src, nb_, c_ = order[self.ring_posted[k]]
            tk, post = self.rings[k].post_chunk(
                lambda n, core, kern, args, after, c_=c_: self.node(k, c_, core, n, kern, args, after), 0,
                f"Ld_{nm}", src=src, nbytes=nb_, dm=self.roles["dm"])
            self.tk[k][nm] = tk + (c_, self.rings[k].last_take, post)
            self.ring_posted[k] += 1
        sl, fl, seq, nb, c, ex, post = self.tk[k][name]
        ld = self.nd(k, c, "dm", f"Ld_{name}", "__snax_bingo_kernel_idma_ring_load",
                     SnaxBingoKernelIdmaRingLoadArgs(fl, seq, sl, dst, nb, 0), list(deps) + [post])
        self.rings[k].read_chunk(0, seq - 1, ld, last=True)
        ld.exec_chunk = ex
        if name == self.last_name[k]:
            self.last_chunk[k] = ld
        return ld

    def gemv(self, k, c, name, x, w, out, K, N, k_shift, deps):
        """x (a_row, K values) . w (paired INT4, K x N) into the FP16 row out."""
        return self.nd(k, c, "gemm", name, "__snax_bingo_kernel_gemv",
                       SnaxBingoKernelGemvArgs(x, w, out, K=K, N=N, d_shift=k_shift, a_blk=8, w4=True), deps)

    def mul(self, k, c, name, a, b, out, n, deps):
        return self.nd(k, c, "simd", name, "__snax_bingo_kernel_simd_stream_elementwise",
                       SnaxBingoKernelSimdMulF16Args(a, b, out, n // 32, 32), deps)

    def add(self, k, c, name, a, b, out, n, deps):
        return self.nd(k, c, "simd", name, "__snax_bingo_kernel_simd_stream_elementwise",
                       SnaxBingoKernelSimdAddF16Args(a, b, out, n // 32, 32), deps)

    def muli8(self, k, c, name, a, b, out, n, inv, deps):
        """out = quant_i8(a (.) b) over n FP16 values: a dequantisation and the quantiser in one SIMD pass."""
        return self.nd(k, c, "simd", name, "__snax_bingo_kernel_simd_stream_elementwise",
                       SnaxBingoKernelSimdMulI8Args(a, b, out, n // 32, 32, f32bits(inv)), deps)

    def quant(self, k, c, name, src, dst, nbytes, inv, deps):
        return self.nd(k, c, "simd", name, "__snax_bingo_kernel_simd_fp16_to_int8",
                       SnaxBingoKernelSimdFp16ToInt8Args(src, dst, beats=simd_beats(nbytes), rows=1,
                                                         inv_scale_f32bits=f32bits(inv)), deps)

    def xd(self, k, c, name, args, deps):
        return self.nd(k, c, "xdma", name, "__snax_bingo_kernel_xdma_6d", args, deps)

    def spread(self, k, c, name, src, dst, n, deps):
        """An FP16 row of n values into a_row's places, still FP16 (each 8-byte atom 16 B apart), on the xDMA; the
        buffer's other atoms stay as they are (zeroed before the window)."""
        if n % (4 * LANES):
            raise ValueError(f"spread: {n} values are not whole xDMA beats of {LANES} atoms")
        b = n // 4 // LANES
        return self.xd(k, c, name, SnaxBingoKernelXdma6dArgs(src, dst, ATOM, 2 * ATOM, [LANES * ATOM], [b],
                                                             [2 * LANES * ATOM], [b]), deps)

    def row_to_a(self, k, c, name, src, dst, cols, deps):
        """An FP16 row of `cols` values into row 0 of a 16-row FP16 A tile: its 8-byte atoms a 128-byte block apart."""
        blk, b = 4 * BR * 2, cols // 4 // LANES
        return self.xd(k, c, name, SnaxBingoKernelXdma6dArgs(src, dst, ATOM, blk, [LANES * ATOM], [b],
                                                             [LANES * blk], [b]), deps)

    def copy(self, k, c, name, src, dst, nb, deps):
        return self.nd(k, c, "dm", name, "__snax_bingo_kernel_idma_1d_copy",
                       SnaxBingoKernelIdma1dCopyArgs(src, dst, nb), deps)

    def copy2d(self, k, c, name, src, dst, size, sst, dstr, reps, deps):
        return self.nd(k, c, "dm", name, "__snax_bingo_kernel_idma_2d_copy",
                       SnaxBingoKernelIdma2dCopyArgs(src, dst, size, sst, dstr, reps), deps)

    def put(self, k, c, name, src, dst, nb, flag, deps, bcast=False, also_here=False):
        return self.nd(k, c, "dm", name, "__snax_bingo_kernel_idma_put_flagged",
                       SnaxBingoKernelIdmaPutFlaggedArgs(src, dst, nb, flag, value=1, bcast=bcast,
                                                         also_here=also_here), deps)

    def _flagged(self, w, puts):
        """The puts a flag wait w follows: a dependency from a task of w's chip, an ordering edge from another's (the
        flag proves the landing)."""
        for t in puts:
            if t.assigned_chiplet_id == w.assigned_chiplet_id:
                if not self.dfg.has_edge(t, w):
                    self.edge(t, w)
            else:
                self.dfg.bingo_add_order_edge(t, w)
        return w

    def wait(self, k, c, name, flags, n, puts, deps=()):
        w = self.nd(k, c, "dm", name, "__snax_bingo_kernel_wait_flags", SnaxBingoKernelWaitFlagsArgs(flags, n, 1),
                    list(deps) + [t for t in puts if t.assigned_chiplet_id == k])
        return self._flagged(w, puts)

    def fetch(self, k, c, name, src, dst, nb, flags, n, puts, deps=()):
        w = self.nd(k, c, "dm", name, "__snax_bingo_kernel_idma_fetch_flagged",
                    SnaxBingoKernelIdmaFetchFlaggedArgs(src, dst, nb, flags, n, 1),
                    list(deps) + [t for t in puts if t.assigned_chiplet_id == k])
        return self._flagged(w, puts)

    def memset(self, k, c, name, buf, nb, deps=()):
        return self.nd(k, c, "xdma", name, "__snax_bingo_kernel_xdma_memset",
                       SnaxBingoKernelXdmaMemsetArgs(buf, nb, SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), deps)

    # ---- the hooks a layer replaces ----------------------------------------------------------------------------------
    def x_multicast(self, go, bcast):
        """x's a_row into every chip's L3 (fk_xland) behind fk_xtflag: the root memory chip's push engine from HBM
        (`bcast`, the root host's task maker). Returns the task the x fetches are ordered after."""
        return bcast("x_hbm_bcast", self.h_xhbm.address, f"{self.TAG}_xland", f"{self.TAG}_xtflag", XB, 1,
                     [go[self.ROOT]])

    def stage_store(self, yw):
        """y's row from the root's L3 into the HBM of the root's memory chip, after yw (the root's wait for every
        cluster's columns). Returns the task."""
        st = self.copy(self.ROOT, 0, "y_store", self.l3("yrow"), self.h_yhbm.address, 2 * HIDDEN, [yw])
        self._y_stores.append(st)
        return st

    def kv_append(self, deps):
        """The new piece from the new key's cluster into its slot of the root's cache shard in HBM: the key rows, then
        the value columns (32 runs of 256 B one 16-row block of the V tile apart). Returns the last task."""
        R, c = self.ROOT, self.KC
        base = self.h_kv[R].address + self.kv_new
        ak = self.copy(R, c, "kv_app_k", self.b1["kt"], base, PK, deps)
        return self.copy2d(R, c, "kv_app_v", self.b1["vp8"], base + KB, 256, 256, 16 * BC, D_V // 16, [ak])

    # ---- the build ---------------------------------------------------------------------------------------------------
    def _preload(self):
        """Before the window: every cluster 0's constants, the new key's, and the buffers whose unwritten places must
        be zero (the a_row spreads, the new tile, the zero rows of the new piece's tiles). One chain a cluster."""
        R = self.ROOT
        pre = {k: [] for k in self.chips}
        self.b0 = {}
        for k in self.chips:
            b = self.b0[k] = dict(c=self.l1(k, 0, "c0", C0_BYTES), nsp=self.l1(k, 0, "nsp", 4 * Q_NOPE),
                                  osp=self.l1(k, 0, "ohsp", 4 * V_HEAD))
            prev = [self.copy(k, 0, "c0_ld", self.h_c0[k].address, b["c"], C0_BYTES, [])]
            prev = [self.memset(k, 0, "z_nsp", b["nsp"], 4 * Q_NOPE, prev)]
            prev = [self.memset(k, 0, "z_ohsp", b["osp"], 4 * V_HEAD, prev)]
            if k == R:
                self.br = dict(osp=self.l1(R, 0, "osp", 2 * BR * LAT))
                prev = [self.memset(R, 0, "z_osp", self.br["osp"], 2 * BR * LAT, prev)]
            pre[k] += prev
        sizes = dict(c=C1_BYTES, kt=KB, vt=VB, ka16=2 * PK, vr16=2 * PVB)
        b = self.b1 = {nm: self.l1(R, 1, {"kt": "kt_new", "vt": "vt_new"}.get(nm, nm), nb) for nm, nb in sizes.items()}
        prev = [self.copy(R, 1, "c1_ld", self.h_c1.address, b["c"], C1_BYTES, [])]
        for nm in ("kt", "vt", "ka16", "vr16"):
            prev = [self.memset(R, 1, f"z_{nm}", b[nm], sizes[nm], prev)]
        pre[R] += prev
        return pre

    def build(self):
        d, R, NC = self.d, self.ROOT, self.NC
        QC_, KC_ = self.QC, self.KC
        dm, simd, gemm, xdma = (self.roles[r] for r in ("dm", "simd", "gemm", "xdma"))
        self._y_stores, self._chains = [], []
        pre = self._preload()
        go = self.setup(extra=pre)
        if self.p.get("relay_per_gather", True):
            self.dfg.relay_one_consumer_cores = [xdma]
        m = self.mem[R]

        def bcast(name, src, dst, flag, nb, n, after):
            # ChipletBuild.hbm_cast: its posts and waits
            return self.hbm_cast(
                R, f"{name}_k{R:02x}",
                HostBingoKernelHbmBcastArgs(src, BingoMemSymbol(dst), BingoMemSymbol(flag), self.h_seq[m].address + 64,
                                            1, nb, n, m, self.engine[R]), after=after)
        xm = self.x_put = self.x_multicast(go, bcast)
        if self.x_on_host:                                  # x first on the root memory chip's engine, then its ring
            self.rings[R].start_after(self.done_of(xm))
        # ---- x into every cluster's L1 ------------------------------------------------------------------------------
        xa, xw = {}, {}
        for k in self.chips:
            for c in range(NC):
                xa[k, c] = self.l1(k, c, "x", XB)
                w = self.nd(k, c, "dm", "x_wait", "__snax_bingo_kernel_idma_fetch_flagged",
                            SnaxBingoKernelIdmaFetchFlaggedArgs(self.l3("xland"), xa[k, c], XB, self.l3("xtflag"), 1,
                                                                1), [go[k]])
                self.dfg.bingo_add_order_edge(self.queued_of(xm), w)
                xw[k, c] = w
        # ---- the new key's columns: a GEMV a W_KVA chunk on the clusters 1, each row put into the root's L3 ----------
        kv_puts = []
        for k in self.chips:
            for n in self.kva[k]:
                wb = self.l1(k, KC_, f"wkva{n}", WCH)
                ld = self.load(k, f"kva{n}", wb, [xw[k, KC_]])
                ob = self.l1(k, KC_, f"kvo{n}", 2 * QC)
                gm = self.gemv(k, KC_, f"gkva{n}", xa[k, KC_], wb, ob, HIDDEN, QC, d.k_x, [ld, xw[k, KC_]])
                kv_puts.append(self.put(k, KC_, f"kva_put{n}", ob, self.l3("kvraw", 2 * QC * n, self.far(k)), 2 * QC,
                                        self.l3("kvrflag", 64 * n, self.far(k)), [gm]))
        # ---- q: every cluster 0, its head's W_q GEMVs, RoPE of q_pe, the absorb, the row into the root's L3 ----------
        q_puts = {}
        for k in self.chips:
            h, c, b = self.head[k], QC_, self.b0[k]
            cst = b["c"]
            qraw, qrow = self.l1(k, c, "qraw", 2 * Q_HEAD), self.l1(k, c, "qrow", QROW)
            gq = {}
            for cc in WQ_ORDER:
                wb = self.l1(k, c, f"wq{cc}", WCH)
                ld = self.load(k, f"wq{cc}", wb, [xw[k, c]])
                gq[cc] = self.gemv(k, c, f"gq{cc}", xa[k, c], wb, qraw.view(2 * QC * cc), HIDDEN, QC, d.k_x,
                                   [ld, xw[k, c]])
                if cc == WQ_ORDER[1]:
                    # q_pe: dequantised, then x cos + (its other half) x sin, at position P
                    pe, rt, rs = self.l1(k, c, "qpe", 2 * ROPE), self.l1(k, c, "qrt", 2 * ROPE), self.l1(k, c, "qrs",
                                                                                                         2 * ROPE)
                    hf = ROPE // 2
                    dq = self.mul(k, c, "dq_pe", qraw.view(2 * Q_NOPE), cst.view(C0_FPE), pe, ROPE, [gq[4], gq[5]])
                    tc = self.mul(k, c, "rp_cos", pe, cst.view(C0_COS), rt, ROPE, [dq])
                    sa = self.mul(k, c, "rp_sa", pe.view(2 * hf), cst.view(C0_SIN), rs, hf, [dq])
                    sb = self.mul(k, c, "rp_sb", pe, cst.view(C0_SIN + 2 * hf), rs.view(2 * hf), hf, [dq])
                    rope = self.add(k, c, "rp_add", rt, rs, qrow.view(2 * KV_RANK), ROPE, [tc, sa, sb])
            # q_nope into a_row, dequantised and quantised at s_qn; the absorb q~ = q_nope W_UK, dequantised
            sp = self.spread(k, c, "qn_spread", qraw, b["nsp"], Q_NOPE, [gq[cc] for cc in range(4)])
            qn8 = self.l1(k, c, "qn8", 2 * Q_NOPE)
            qn = self.muli8(k, c, "qn8", b["nsp"], cst.view(C0_NSP), qn8, 2 * Q_NOPE, d.s_qn.inv, [sp])
            wuk = self.l1(k, c, "wuk", WCH)
            ld = self.load(k, "wuk", wuk, [xw[k, c]])
            qtraw = self.l1(k, c, "qtraw", 2 * KV_RANK)
            gu = self.gemv(k, c, "guk", qn8, wuk, qtraw, Q_NOPE, KV_RANK, d.k_uk, [ld, qn])
            dqt = self.mul(k, c, "dq_qt", qtraw, cst.view(C0_FQT), qrow, KV_RANK, [gu])
            q_puts[k] = self.put(k, c, "q_put", qrow, self.l3("qrows", QROW * h, self.far(k)), QROW,
                                 self.l3("qflag", 64 * h, self.far(k)), [dqt, rope])
        # ---- the new key on the root's cluster 1: the 18 rows, dequantised, RMSNorm, RoPE, the piece, the new tile ---
        c, b = KC_, self.b1
        b1 = b
        kvl, kv16, krow = self.l1(R, c, "kvraw", 2 * D_QK), self.l1(R, c, "kv16", 2 * D_QK), self.l1(R, c, "krow",
                                                                                                    2 * D_QK)
        kw = self.fetch(R, c, "kv_wait", self.l3("kvraw"), kvl, 2 * D_QK, self.l3("kvrflag"), N_KVA, kv_puts)
        dkv = self.mul(R, c, "dq_kv", kvl, b["c"].view(C1_FKV), kv16, D_QK, [kw])
        rms_a = SnaxBingoKernelSimdRmsnormArgs(kv16, krow, 1, KV_RANK)
        rscr = self.l1(R, c, "kvn_scr", rms_a.scratch_bytes())
        rms = self.nd(R, c, "simd", "kv_rms", "__snax_bingo_kernel_simd_rmsnorm",
                      SnaxBingoKernelSimdRmsnormArgs(kv16, krow, 1, KV_RANK, scratch_addr=rscr), [dkv])
        kpe, rt, rs = kv16.view(2 * KV_RANK), self.l1(R, c, "krt", 2 * ROPE), self.l1(R, c, "krs", 2 * ROPE)
        hf = ROPE // 2
        tc = self.mul(R, c, "kp_cos", kpe, b["c"].view(C1_COS), rt, ROPE, [dkv])
        sa = self.mul(R, c, "kp_sa", kpe.view(2 * hf), b["c"].view(C1_SIN), rs, hf, [dkv])
        sb = self.mul(R, c, "kp_sb", kpe, b["c"].view(C1_SIN + 2 * hf), rs.view(2 * hf), hf, [dkv])
        kr = self.add(R, c, "kp_add", rt, rs, krow.view(2 * KV_RANK), ROPE, [tc, sa, sb, rms])
        # the key rows: row 0 of the FP16 A tile [16, 576], quantised (c at s_c, k_pe at s_kpe) into the new tile's K
        r2a = self.row_to_a(R, c, "k_toa", krow, b["ka16"], D_QK, [rms, kr])
        qk1 = self.quant(R, c, "k8_c", b["ka16"], b["kt"], 2 * 16 * KV_RANK, d.s_c.inv, [r2a])
        qk2 = self.quant(R, c, "k8_pe", b["ka16"].view(2 * 16 * KV_RANK), b["kt"].view(16 * KV_RANK), 2 * 16 * ROPE,
                         d.s_kpe.inv, [r2a])
        # the value columns: c~ in row 0 of the FP16 [16, 512] tile, transposed, the A layout, quantised at s_c, then
        # the first four columns of every row block of the new tile's V
        cpv = self.copy(R, c, "v_row", krow, b["vr16"], 2 * KV_RANK, [rms])
        vt16, va16 = self.l1(R, c, "vt16", 2 * PVB), self.l1(R, c, "va16", 2 * PVB)
        b["vp8"] = self.l1(R, c, "vp8", PVB)
        trv = self.nd(R, c, "xdma", "v_tr", "__snax_bingo_kernel_xdma_transpose_2d",
                      SnaxBingoKernelXdmaTranspose2dArgs(b["vr16"], vt16, 16, KV_RANK, 2), [cpv])
        cva = self.xd(R, c, "v_toa", convert_args("row_major", "A", KV_RANK, 16, cm.MESH, 2, vt16, va16), [trv])
        qv = self.quant(R, c, "v8", va16, b["vp8"], 2 * PVB, d.s_c.inv, [cva])
        vsc = self.copy2d(R, c, "v_tile", b["vp8"], b["vt"], 256, 256, 16 * BC, D_V // 16, [qv])
        new_ready = {"K": [qk1, qk2], "V": [vsc]}
        # ---- the query tile: the 16 rows on the root, the A layout, Q8, multicast into every cluster 0 ---------------
        qrm, qa16, qa8 = self.l1(R, 0, "qrm", BR * QROW), self.l1(R, 0, "qa16", BR * QROW), self.l1(R, 0, "qa8", QB)
        qg = self.fetch(R, 0, "qg_wait", self.l3("qrows"), qrm, BR * QROW, self.l3("qflag"), len(self.chips),
                        list(q_puts.values()))
        qcv = self.xd(R, 0, "q_toa", convert_args("row_major", "A", BR, D_QK, cm.MESH, 2, qrm, qa16), [qg])
        qq1 = self.quant(R, 0, "q8_lat", qa16, qa8, 2 * BR * KV_RANK, d.s_qt.inv, [qcv])
        qq2 = self.quant(R, 0, "q8_pe", qa16.view(2 * BR * KV_RANK), qa8.view(BR * KV_RANK), 2 * BR * ROPE,
                         d.s_qpe.inv, [qcv])
        qin = self.broadcast("q", qa8, QB, self.OFF_X, self.OFF_FLAG,
                             {k: ([qq1, qq2][-1] if k == R else q_puts[k]) for k in self.chips}, clusters=[0])
        self.edge(qq1, self.bcast_puts[0])
        qin[(R, 1)] = self.copy(R, 1, "q_cp", self.xch_view(R, 0, self.OFF_X), self.xch_view(R, 1, self.OFF_X), QB,
                                [qin[(R, 0)]])
        self.q_mc = list(self.bcast_puts)
        # ---- pass 1: the row maximum of every attention cluster over its tiles ---------------------------------------
        self.kbuf, self.vbuf, s16, ar1, sm1 = {}, {}, {}, {}, {}
        for k, c, ts in self.units:
            q8 = self.xch_view(k, c, self.OFF_X)
            ar1[k, c] = self.l1(R, 0, "ar1", ARENA) if (k, c) == (R, 0) else self.xch_view(k, c, self.OFF_MX)
            s16[k, c] = self.l1(k, c, "s16", len(ts) * SB)
            p8 = self.l1(k, c, "p8a", FA.p8_bytes(BC, BR, FA.EXACT))
            sm = []
            for i, t in enumerate(ts):
                if t == self.NTC:
                    kb, ready, bias = b1["kt"], new_ready["K"], b1["c"].view(C1_MASK)
                else:
                    kb = self.kbuf[k, t] = self.l1(k, c, f"kt{i}", KB)
                    ready, bias = [self.load(k, f"K{t}", kb, [qin[(k, c)]])], 0
                qk = self.nd(k, c, "gemm", f"qk{i}", "__snax_bingo_kernel_gemm_fa_qk",
                             SnaxBingoKernelGemmFaQkArgs(kb, q8, bias, at_offset(s16[k, c], i * SB + BEAT), M=BC // 16,
                                                         K=D_QK // 4, N=BR // 16, d_shift=d.k_s),
                             ready + [qin[(k, c)]])
                sm.append(self.nd(k, c, "simd", f"s1_{i}", "__snax_bingo_kernel_simd_fa_softmax",
                                  FA(at_offset(s16[k, c], i * SB + BEAT), p8, ar1[k, c], BC, 1, tile_idx=i,
                                     seed_state=1, geom_mode=FA.GEOM_SELF, score_scale=d.a_exp, p_mode=FA.EXACT,
                                     p8_pitch=0),
                                  [qk] + sm[-1:]))
            sm1[k, c] = sm[-1]
        # ---- each chip's maximum (the root: its two clusters'), m* on the root ---------------------------------------
        mloc = {k: sm1[k, QC_] for k in self.chips if k != R}
        mt, p8m = self.l1(R, 0, "mt", 3 * BEAT), self.l1(R, 0, "p8m", FA.p8_bytes(2, BR, 0))
        cp0 = self.copy(R, 0, "m_cp0", at_offset(ar1[R, 0], MRUN), at_offset(mt, BEAT), BEAT, [sm1[R, 0]])
        cp1 = self.copy(R, 1, "m_cp1", self.xch_view(R, 1, self.OFF_MLOC), at_offset(mt, 2 * BEAT), BEAT, [sm1[R, 1]])
        mloc[R] = self.nd(R, 0, "simd", "m_max", "__snax_bingo_kernel_simd_fa_softmax",
                          FA(at_offset(mt, BEAT), p8m, self.xch_view(R, 0, self.OFF_MX), 2, 1, tile_idx=0, seed_state=1,
                             geom_mode=FA.GEOM_SELF, score_scale=1.0, p_mode=0, p8_pitch=0), [cp0, cp1])
        vld = {}
        if self.m_exchange == "gather":
            m_puts = {}
            for i, k in enumerate(self.chips):
                m_puts[k] = self.put(k, 0, "m_put", self.xch_view(k, 0, self.OFF_MLOC),
                                     self.l3("mstg", XFER * i, self.far(k)), XFER,
                                     self.l3("mflag", 64 * i, self.far(k)), [mloc[k]])
                vld.update(self._v_loads(k, qin))
            rows = self.l1(R, 0, "mrows", BEAT + len(self.chips) * BEAT)
            p8r = self.l1(R, 0, "p8r", FA.p8_bytes(len(self.chips), BR, 0))
            mw = self.wait(R, 0, "mg_wait", self.l3("mflag"), len(self.chips), list(m_puts.values()))
            msp = self.copy2d(R, 0, "mg_split", self.l3("mstg"), at_offset(rows, BEAT), BEAT, XFER, BEAT,
                              len(self.chips), [mw])
            made = self.nd(R, 0, "simd", "mg_max", "__snax_bingo_kernel_simd_fa_softmax",
                           FA(at_offset(rows, BEAT), p8r, self.xch_view(R, 0, self.OFF_MR), len(self.chips), 1,
                              tile_idx=0,
                              seed_state=1, geom_mode=FA.GEOM_SELF, score_scale=1.0, p_mode=0, p8_pitch=0), [msp])
        else:
            for k in self.chips:
                vld.update(self._v_loads(k, qin))
            if self.after_stream:
                for k in self.chips:
                    self._stream_tail(k)
            ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                cm.remote, R, 0, self.order[1:], src=self.xch_addr(0, self.OFF_MLOC),
                dst=self.xch_addr(0, self.OFF_MR + MRUN),
                size=XFER, junction=self.jct["Elementwise"], jct_csr0=EW_F16_MAX)
            deps = [mloc[k] for k in self.chips] + ([self.last_chunk[k] for k in self.chips]
                                                    if self.after_stream else [])
            made = self.nd(R, 0, "xdma", "mg_chain", "__snax_bingo_kernel_xdma_chain_gather", ga, deps)
            self._chains.append(made)
        self.m_made, self.msrc = made, self.xch_view(R, 0, self.OFF_MR + MRUN)
        marr = self.broadcast("m", self.msrc, XFER, self.OFF_M, self.OFF_MFLAG,
                              {k: (made if k == R else vld[k, self.tiles[k][-1]]) for k in self.chips}, clusters=[0])
        self.m_mc = list(self.bcast_puts)
        marr[(R, 1)] = self.copy(R, 1, "m_cp", self.msrc, self.xch_view(R, 1, self.OFF_M), XFER, [made])
        # ---- pass 2: the softmax seeded at m*, PV, the partial -------------------------------------------------------
        part, lcp = {}, {}
        for k, c, ts in self.units:
            ar2 = self.xch_view(k, c, self.OFF_A2)
            p8 = [self.l1(k, c, f"p8b{j}", FA.p8_bytes(BC, BR, FA.EXACT)) for j in range(2)]
            oacc = self.l1(k, c, "oacc", D_V * BR * 4) if len(ts) > 1 else None
            single = sum(1 for kk, _, _ in self.units if kk == k) == 1
            part[k, c] = self.xch_view(k, 0, self.OFF_LOC) if single else self.l1(k, c, "part", PB)
            corr = FA.corr_offset(BC, BR)
            sm, pv = [], []
            for i, t in enumerate(ts):
                first, last = i == 0, i == len(ts) - 1
                vb, ready = (b1["vt"], new_ready["V"]) if t == self.NTC else (self.vbuf[k, t], [vld[k, t]])
                sm.append(self.nd(k, c, "simd", f"s2_{i}", "__snax_bingo_kernel_simd_fa_softmax",
                                  FA(at_offset(s16[k, c], i * SB + BEAT), p8[i % 2], ar2, BC, 1, tile_idx=i,
                                     seed_state=0, geom_mode=FA.GEOM_SELF, score_scale=d.a_exp, p_mode=FA.EXACT,
                                     p8_pitch=0),
                                  [marr[(k, c)], sm1[k, c]] + sm[-1:] + (pv[i - 2:i - 1] if i >= 2 else [])))
                flags = PVK.B_KMAJOR | (0 if first else PVK.C_COLSCALE) | (PVK.D_FP16 if last else 0)
                pv.append(self.nd(k, c, "gemm", f"pv{i}", "__snax_bingo_kernel_gemm_fa_pv",
                                  PVK(vb, p8[i % 2], 0 if first else oacc, part[k, c] if last else oacc, M=D_V // 16,
                                      K=BC // 4, N=BR // 16, flags=flags,
                                      corr_addr=0 if first else at_offset(p8[i % 2], corr),
                                      d_shift=d.k_o if last else 0),
                                  [sm[i]] + ready + pv[-1:]))
            if c == QC_ and self.after_stream and self.m_exchange == "gather":
                self._stream_tail(k)
            lcp[k, c] = self.copy(k, c, "l_cp", self.xch_view(k, c, self.OFF_A2 + LRUN), at_offset(part[k, c], OB),
                                  XFER, [sm[-1], pv[-1]])
        if not self.after_stream:
            for k in self.chips:
                self._stream_tail(k)
        # ---- the partials: the root's two clusters added, then one chain through every chip into the root ------------
        loc = {k: lcp[k, QC_] for k in self.chips}
        pp = self.copy(R, 1, "part_push", part[R, 1], self.p1[R], PB, [lcp[R, 1]])
        loc[R] = self.nd(R, 0, "simd", "loc_add", "__snax_bingo_kernel_simd_stream_elementwise",
                         SnaxBingoKernelSimdAddF16Args(part[R, 0], self.p1[R], self.xch_view(R, 0, self.OFF_LOC),
                                                       PB // 64, 32), [lcp[R, 0], pp])
        ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
            cm.remote, R, 0, self.order[1:], src=self.xch_addr(0, self.OFF_LOC), dst=self.dst_expr(), size=PB,
            junction=self.jct["Elementwise"], jct_csr0=cm.EW_F16_ADD, holds=[self.dst_buf])
        chain = self.nd(R, 0, "xdma", "o_chain", "__snax_bingo_kernel_xdma_chain_gather", ga,
                        [loc[k] for k in self.chips])
        self._chains.append(chain)
        # ---- o~: normalised, a head a row, a_row, quantised at s_ot, multicast into every chip's L3 ------------------
        self.ot, self.xt = self.l1(R, 0, "ot", OB), self.l1(R, 0, "xt", OB)
        nrm = self.nd(R, 0, "simd", "o_norm", "__snax_bingo_kernel_simd_mla_normalise",
                      SnaxBingoKernelSimdMlaNormaliseArgs(self.dst.view(OB), self.dst_buf, self.dst, self.ot, D_V,
                                                          d.a_n_bits), [chain])
        tr = self.nd(R, 0, "xdma", "o_tr", "__snax_bingo_kernel_xdma_transpose_2d",
                     SnaxBingoKernelXdmaTranspose2dArgs(self.ot, self.xt, D_V, BR, 2), [nrm])
        osp = self.br["osp"]
        ospr = self.spread(R, 0, "o_spread", self.xt, osp, BR * KV_RANK, [tr])
        self.oar = self.l1(R, 0, "oar", BR * LAT)
        oq = self.quant(R, 0, "o_q8", osp, self.oar, 2 * BR * LAT, d.s_ot.inv, [ospr])
        self.ol_mc = self.put(R, 0, "ol_bput", self.oar, self.l3("olat"), BR * LAT, self.l3("olflag"), [oq], bcast=True)
        # ---- W_UV: every cluster 0 its head's o~ row, o_h, o8_h into the root's L3 -----------------------------------
        o_puts = {}
        for k in self.chips:
            h, c, bk = self.head[k], QC_, self.b0[k]
            if k == R:
                ola, got = self.oar.view(LAT * h), oq
            else:
                ola = self.l1(k, c, "ola", LAT)
                got = self.fetch(k, c, "ol_get", self.l3("olat", LAT * h), ola, LAT, self.l3("olflag"), 1, [self.ol_mc])
            ohr, o8 = self.l1(k, c, "ohraw", 2 * V_HEAD), self.l1(k, c, "o8", OAR)
            gv = self.gemv(k, c, "guv", ola, self.wuv[k], ohr, KV_RANK, V_HEAD, d.k_uv, [got, self.wuv_ld[k]])
            osr = self.spread(k, c, "oh_spread", ohr, bk["osp"], V_HEAD, [gv])
            oqk = self.muli8(k, c, "o8", bk["osp"], bk["c"].view(C0_UVSP), o8, 2 * V_HEAD, d.s_o.inv, [osr])
            o_puts[k] = self.put(k, c, "o8_put", o8, self.l3("oland", OAR * h, self.far(k)), OAR,
                                 self.l3("oflag", 64 * h, self.far(k)), [oqk])
        ow = self.wait(R, 0, "o8_wait", self.l3("oflag"), len(self.chips), list(o_puts.values()))
        self.o8_mc = self.put(R, 0, "o8_bput", self.l3("oland"), self.l3("oland"), len(self.chips) * OAR,
                              self.l3("omflag"), [ow], bcast=True)
        # ---- W_O by columns: every cluster its 64 columns over the full depth, into the root's y row -----------------
        y_puts = []
        for i, k in enumerate(self.chips):
            for c in range(NC):
                cl = i * NC + c
                ofl, yw_ = self.l1(k, c, "ofull", XB), self.l1(k, c, "ywo", 4 * QC)
                if k == R:
                    get = self.copy(R, c, "o8_get", self.l3("oland"), ofl, len(self.chips) * OAR, [ow])
                else:
                    get = self.fetch(k, c, "o8_get", self.l3("oland"), ofl, len(self.chips) * OAR, self.l3("omflag"), 1,
                                     [self.o8_mc])
                gms = [self.gemv(k, c, f"gwo{s}", ofl, self.wo[k, c, s], yw_.view(2 * QC * s), HIDDEN, QC, d.k_wo,
                                 [get, self.wo_ld[k, c, s]]) for s in range(2)]
                y_puts.append(self.put(k, c, "y_put", yw_, self.l3("yrow", 4 * QC * cl, self.far(k)), 4 * QC,
                                       self.l3("yflag", 64 * cl, self.far(k)), gms))
        yw = self.wait(R, 0, "y_wait", self.l3("yflag"), self.N_CL, y_puts)
        ys = self.stage_store(yw)
        self.append = self.kv_append([ys] if ys is not None else [yw])
        self.link_report()
        self.end_window([t for t in (ys, self.append, yw) if t is not None])
        self._checks()

    def _v_loads(self, k, qin):
        """Chip k's V tiles into L1 on its cluster 0, after its maximum's put (pass 2's tiles, loaded while m* is
        made)."""
        out = {}
        for i, t in enumerate(self.tiles[k]):
            vb = self.vbuf[k, t] = self.l1(k, self.QC, f"vt{i}", VB)
            out[k, t] = self.load(k, f"V{t}", vb, [qin[(k, self.QC)]])
        return out

    def _stream_tail(self, k):
        """Chip k's last ring chunks into L1: W_UV and W_O on cluster 0, W_O on cluster 1 (cluster 0's last W_O chunk is
        the chip's last push)."""
        if not hasattr(self, "wuv"):
            self.wuv, self.wuv_ld, self.wo, self.wo_ld = {}, {}, {}, {}
        if k in self.wuv:
            return
        self.wuv[k] = self.l1(k, self.QC, "wuv", WCH)
        self.wuv_ld[k] = self.load(k, "wuv", self.wuv[k])
        for c in (self.KC, self.QC):
            for s in range(2):
                self.wo[k, c, s] = self.l1(k, c, f"wo{s}", WCH)
                self.wo_ld[k, c, s] = self.load(k, f"wo{c}{s}", self.wo[k, c, s])

    # ---- checks ------------------------------------------------------------------------------------------------------
    def _checks(self):
        """Byte for byte: y in HBM, the appended piece in HBM (the head and the tail of its key rows and of its V slot),
        the query tile as its multicast left it on the chip in the middle of the fold order, that chip's partial, m*
        and o~ on the root; checks stages adds that chip's query row in the root's L3, the root's o8 row of every head
        and the new tile's K on the root's cluster 1."""
        R, gold = self.ROOT, self.gold
        far = self.order[len(self.order) // 2]
        n = 2048
        checks = [("q8", self.stash("q8", far, 0, self.xch_view(far, 0, self.OFF_X), QB, []),
                   gold.q8_tile.view(np.uint8), QB),
                  ("mstar", self.stash("mstar", R, 0, self.msrc, BEAT, [self.m_made]),
                   np.asarray(gold.mstar, np.float16).view(np.uint8), BEAT),
                  (f"loc_k{far:02x}", self.stash(f"loc_k{far:02x}", far, 0, self.xch_view(far, 0, self.OFF_LOC), PB,
                                                 []),
                   gold.part_bytes(far).view(np.uint8), PB),
                  ("xt", self.stash("xt", R, 0, self.xt, OB, []), gold.xt.view(np.uint8), OB)]
        if self.checks_kind == "stages":
            h = self.head[far]
            checks += [("qrow", self.stash("qrow", R, 0, self.l3("qrows", QROW * h), QROW, []),
                        gold.qrow[h].view(np.uint8), QROW),
                       ("o8row", self.stash("o8row", R, 0, self.l3("oland"), len(self.chips) * OAR, []),
                        a_row(gold.o8_all).view(np.uint8), len(self.chips) * OAR),
                       ("ktnew", self.stash("ktnew", R, 1, self.b1["kt"], PK, []), gold.piece_k.view(np.uint8), PK)]
        if self.y_to_hbm:
            y = gold.y_raw.view(np.uint8)
            checks.append(("y", self.stash("y", R, 0, self.h_yhbm.address, y.size, list(self._y_stores)), y, y.size))
        if self.append is not None:
            base = self.h_kv[R].address + self.kv_new
            vt = gold.vtile_new.view(np.uint8)
            for nm, off, img in (("kapp_a", 0, gold.piece_k[:n]), ("kapp_b", PK - n, gold.piece_k[PK - n:]),
                                 ("vapp_a", KB, vt[:n]), ("vapp_b", KB + VB - n, vt[VB - n:])):
                img = np.ascontiguousarray(img).view(np.uint8)
                checks.append((nm, self.stash(nm, R, 0, base + off, n, [self.append]), img, n))
        self.run_checks(checks)

    def _check_collectives(self):
        """Every two multicasts, every multicast and chain, and every chain and memory-chip push ordered by the graph
        (dependency and flag edges); every write into HBM after all of them and after every chip's last ring load.
        Raises otherwise."""
        import networkx as nx
        spans = self.collectives()      # each collective by its first task, descendants of its done one; before the copy
        gr = nx.DiGraph(self.dfg)
        gr.add_edges_from(self.dfg.order_edges)
        done = {s: e for s, e, _ in spans}
        mc = [s for s, _, kind in spans if kind != "chain"]
        ch = [s for s, _, kind in spans if kind == "chain"]
        coll = mc + ch
        desc = {t: nx.descendants(gr, done.get(t, t)) for t in coll + [self.last_chunk[k] for k in self.chips]}
        bad = [(a.node_name, b.node_name) for i, a in enumerate(coll) for b in coll[i + 1:]
               if (a in mc or b in mc) and b not in desc[a] and a not in desc[b]]
        if bad:
            raise ValueError(f"{len(bad)} multicast / chain pair(s) may run at once: {bad[:6]}")
        beside = [(t.node_name, f"{k:02x}") for t in ch for k in self.chips if t not in desc[self.last_chunk[k]]]
        if beside and self.after_stream:
            raise ValueError(f"{len(beside)} chain(s) not after a chip's last push: {beside[:6]}")
        hbm = list(self._y_stores) + ([self.append] if self.append is not None else [])
        late = [(a.node_name, w.node_name) for a in coll + [self.last_chunk[k] for k in self.chips] for w in hbm
                if w not in desc[a]]
        if late:
            raise ValueError(f"{len(late)} HBM write(s) not after a collective or a push: {late[:6]}")
        print(f"[{self.TAG}] collectives: {len(mc)} multicasts and {len(ch)} chain(s), each ordered with every "
              f"multicast; the chains "
              f"{'after every push' if not beside else f'beside the pushes of {len(beside)} chips'}; "
              f"{len(hbm)} HBM writes after all of them")

    def link_report(self):
        """The bytes each D2D link carries in the step (both directions): the rings (memory chip to chip), x's
        multicast from the memory chip, the root's multicasts (query, m*, o~, o8), the gathers into the root, the
        chain's hops, y and the append into the root memory chip. Prints the busiest links and the stream floor (the
        busiest link at LINK_BW)."""
        R, nchip = self.ROOT, len(self.chips)
        grid = set(self.chips) | set(self.mem.values())
        load, kind = {}, {}

        def add(links, nb, what):
            for ln in links:
                load[ln] = load.get(ln, 0) + nb
                kind.setdefault(ln, {}).setdefault(what, 0)
                kind[ln][what] += nb

        for k in self.chips:
            add(fdd._route(self.mem[k], k), self.ring_bytes[k], "ring")
        add(fdd._bcast_tree(self.mem[R], grid), XB, "x")
        for nb, what in ((QB, "query"), (XFER, "m*"), (BR * LAT, "o~"), (nchip * OAR, "o8")):
            add(fdd._bcast_tree(R, grid), nb, what)
        for k in self.chips:
            add(fdd._route(k, R), QROW + XFER + OAR + self.NC * 4 * QC + 2 * QC * len(self.kva[k]), "gathers")
        hops = self.order[1:] + [R]
        for a, b in zip(hops, hops[1:]):
            add(fdd._route(a, b), PB, "chain")
        add(fdd._route(R, self.mem[R]), 2 * HIDDEN + PK + PVB, "y+append")
        top = sorted(load, key=lambda ln: -load[ln])[:4]
        self.floor = load[top[0]] / LINK_BW
        print(f"[{self.TAG}] streams: ring {sum(self.ring_bytes.values()):,} B over {nchip} chips "
              f"({min(self.ring_bytes.values()):,}..{max(self.ring_bytes.values()):,} B a chip); stream floor "
              f"{self.floor * 1e6:.1f} us (the busiest link at {LINK_BW / 1e9:.1f} GB/s)")
        for ln in top:
            parts = ", ".join(f"{w} {b:,}" for w, b in sorted(kind[ln].items(), key=lambda x: -x[1]))
            print(f"[{self.TAG}] link {ln[0]:02x}-{ln[1]:02x}: {load[ln]:,} B ({parts})")

    def l1_report(self):
        """Per cluster: the static plan's peak (static_l1_layout.csv) plus every device task's args struct plus the
        scheduler's lists (12 B a task), against l1_capacity; prints the fullest clusters."""
        import csv
        from bingo_args_phases import _args_bytes
        peak = {}
        with open(os.path.join(self.args.output_dir, "static_l1_layout.csv")) as f:
            for r in csv.DictReader(f):
                key = (int(r["chip"], 16), int(r["cluster"]))
                peak[key] = max(peak.get(key, 0), int(r["end"]))
        args, ntask = {}, {}
        for t in self.dfg.node_list:
            if (t.kernel_name or "").startswith("__snax"):
                key = (t.assigned_chiplet_id, t.assigned_cluster_id)
                args[key] = args.get(key, 0) + _args_bytes(t)
                ntask[key] = ntask.get(key, 0) + 1
        cap = self.dfg.l1_capacity_bytes
        tot = {key: peak.get(key, 0) + args.get(key, 0) + 12 * ntask.get(key, 0) for key in set(peak) | set(args)}
        top = sorted(tot, key=lambda key: -tot[key])[:4]
        print(f"[{self.TAG}] L1 estimate (static peak + args + lists of {cap:,} B): "
              + "; ".join(f"{k:02x}/{c} {peak.get((k, c), 0):,} + {args.get((k, c), 0):,} + "
                          f"{12 * ntask.get((k, c), 0):,} = {tot[(k, c)]:,}" for k, c in top))
        if tot[top[0]] > cap:
            over = sum(v > cap for v in tot.values())
            print(f"[{self.TAG}] WARNING: L1 estimate over l1_capacity on {over} clusters")

    # ---- emission ----------------------------------------------------------------------------------------------------
    def finish(self, app_name):
        self._check_collectives()
        cm.ChipletBuild.finish(self, app_name)
        self.l1_report()


def _app_name(S):
    return (f"DeepSeek-V2-Lite layer 1 MLA attention decode, key split, GEMV projections, one token at position "
            f"{S.P} over {S.P} cached keys, {len(S.chips)} chips x {S.NC} clusters, x, weights and the cache from HBM, "
            f"y into HBM")


def run():
    run_driver("fa key-split decode workload generator", FaDecodeKs, _app_name)
