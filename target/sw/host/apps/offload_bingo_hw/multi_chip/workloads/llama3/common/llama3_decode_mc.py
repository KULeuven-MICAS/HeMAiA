# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Llama-3-8B's grouped-query attention for ONE decode token over a KV cache of `ctx` positions, on the prefill's
head-parallel mapping (llama3_attn_mc.py): a query head a cluster, a key/value group on a column's chip pair (x, 2 y),
(x, 2 y + 1), each chip of the pair projecting half of the group's K (cluster 0) and half of its V (cluster 1), whole
RoPE chunk pairs (kv_split, rope_pairs chunks). The platform, the rings, the stamps and the checks are chiplet_mc.py's;
the data and the golden are LlamaDecodeData's below (LlamaAttnData over a fresh causal prompt of ctx + 1 positions, the
decode token the last).

THE TOKEN. Every projection is VersaCore's one-row GEMV ((1, 4, 32), __snax_bingo_kernel_gemv), one task a weight chunk
of 32 columns as it lands, on the prefill's paired INT4 chunks; its A operand is a_row (the 8-byte word of each A block
the GEMV reads) and its output a plain FP16 row. The row kernels of simd_row.h are written for a 64-B SIMD beat and
refuse this cluster's, so nothing here uses them: a row goes into a 16-row tile by an xDMA scatter of its 8-byte atoms
into row 0 of a zeroed FP16 A tile, quantised whole by the SIMD's Fp16ToInt8 (the zero rows stay zero).

    x          the token's INT8 row (a_row) from the HBM of the root's memory chip: the root's host has that memory
               chip's push engine multicast it into every chip's L3 behind one flag; every cluster fetches it into L1
    q          4 GEMVs (W_q's chunks of the cluster's head, in rope_pairs' column order), each dequantised (x the
               chunk's column factors); RoPE of each chunk pair at the token's position (x cos + the pair's other
               chunk x sin, the prefill's _rope_pair on one row); the row into row 0 of the query tile, quantised
    k or v     2 GEMVs, the chip's half of the group's K (cluster 0: dequantised, RoPE, into row 0 of the new key tile's
               first 16 keys, that half's run) or V (cluster 1: each chunk's row as row 0 of a zeroed [bc, 32] FP16
               block, transposed, reshaped into V^T's A layout, quantised into the new value tile at the chunk's
               place, as the prefill makes a value tile)
    trade      each chip's halves of the new key and value through its L3 into its other cluster and, by remote
               reads, into both clusters of its mate (stage_trade_kv_split's pattern for one tile)
    sweep      fa_sweep over ctx / bc cached key tiles and the tile holding the new key (key ctx at its offset 0, the
               keys after it zero and masked, mask_c), one 16-lane query tile (lane 0 the token, lanes 1..15 a zero
               query), o~ quantised into o8 in the A layout (ot8 fused)
    o          each cluster's o8 row compacted to a_row on the xDMA (rows 0 and 1 of every block; the GEMV takes row 0)
               and put into the root's L3 behind a flag; the root waits for every head's and multicasts the row of all
               32 heads (8 KiB) into every chip's L3; every cluster copies it into L1 on its xDMA
    W_O        by output columns (the prefill's wo_cols): 4 GEMVs of 32 of the cluster's 128 columns over the full
               depth; the cluster's 128 columns of y put into the root's L3 row behind a flag; the root waits for all
               32 and writes y (FP16, before W_O's column dequantisation, as the prefill leaves it) into the HBM of the
               root's memory chip
    append     each chip writes its halves of the new key (the first 16 keys' run of its half) and value (its half of
               V^T) into its cache in its memory chip's HBM, at the new key's tile: cluster 0 the key's, cluster 1 the
               value's, each behind its cluster's last weight chunk and y put on the DM core

THE CACHE. Each chip's memory chip holds that chip's half of the group's cache, in the layouts the prefill's key and
value buffers use (K: each key tile in the A layout, rope_pairs' column order, a half being one 1 KiB run of every
16-key block; V: each tile's V^T in the A layout, a half being one contiguous 4 KiB) and in the order the prefill's
trade buffer kvx holds a pass: [K halves | V halves], in params cache_parts parts of whole tiles, then the new key's
tile (zero until the append). Each half crosses its chip's memory link once: the chip's host queues its memory chip's
push engine (__host_bingo_kernel_hbm_ucast, a part a tile behind its flag) between the two parts of its weight
prefetcher (params cache_after: the chunks of each ring before it), so the cache lands after the q weights and before
the k/v and W_O weights; each cluster copies the half into its key and value tiles, and the chip relays it into its
mate's L3 (cluster 0 the K half, cluster 1 the V half, a flagged put each), where both of the mate's clusters copy it.
Per pair link: one half each way; per memory link: the halves of the chips it feeds. No chip copies remote data into its
own L3, and no multicast meets a chain (there is none): the two multicasts (x at the start, o after every sweep) are
ordered.

THE ORDERS. The compiler orders every device core's program (mini_compiler passes/bingo_dfg_schedule.py): the token's
k / v projection is made before the sweep (the sweep's last tile needs it), and its weights land after the cache, so
the cached tiles' sweep can run while they stream. The compiler splits the matmuls, SIMD and xDMA tasks into
configure / start / wait parts where its schedule gains (params async_exec false: every task whole). Each DM core: x, the q chunks, the cache, the k / v
chunks, the trade, the first W_O chunks, the o put, the rest of W_O, the y put, the append.

PARAMETERS (params.hjson): ctx (cached positions, whole key tiles and whole prefill passes), rows and bc (the prefill's
passes and key tiles: the golden's static scales are the ctx + 1-position prompt's), cache_parts, cache_after,
w_chunk_bytes (one 32-column chunk), w_buffers, weight_ring, seed, w4_paired (true: the GEMV's INT4 layout), mask_c,
sm_geom_xdma, async_exec, checks (y | stages), and the platform's and the graph's switches
(static_l1, l1_capacity, compact_id_maps, gid_lists, per_chip_image, desc_in_image, d2d_ddr, d2d_rx_yield,
fast_host_init, args_phase_bytes).

IN A LAYER (llama3_decode_layer_mc.py) the block runs with shared=the layer: x_on_host false (the layer's input norm makes
x on the root and multicasts it from the root's L1: x_multicast returns that put, and the root's host pushes its cache and
weights without waiting for it) and y_to_hbm false (y stays in the root's L3, where the layer's glue reads it: y_out
returns the root's wait, and the check reads the row there).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# llama3_attn_mc puts util/sim/multichip and mini_compiler on sys.path (llama3_data adds ffn/common)
from llama3_attn_mc import BR, QC, L1Tiles, ABuild, run_driver  # noqa: E402

from bingo_kernel_args import (  # noqa: E402
    HostBingoKernelHbmBcastArgs, HostBingoKernelHbmUcastArgs, HostBingoKernelWeightPrefetchRangeArgs,
    SnaxBingoKernelGemvArgs, SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdmaFetchFlaggedArgs,
    SnaxBingoKernelIdmaPutFlaggedArgs,
    SnaxBingoKernelWaitFlagsArgs, SnaxBingoKernelXdma1dCopyArgs, SnaxBingoKernelXdma6dArgs,
    SnaxBingoKernelXdmaMemsetArgs, SnaxBingoKernelXdmaTranspose2dArgs, simd_fp16_lanes)
from bingo_mem_handle import BingoMemFixedAddr, BingoMemSymbol  # noqa: E402
from libs.blocks import FaSweepCfg, fa_sweep, fa_sweep_alloc  # noqa: E402
from libs.comm.nest import convert_args  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import llama3_data  # noqa: E402
import model_data  # noqa: E402

ATOM = 8                      # the xDMA's lane: one 8-byte atom
LANES = 8                     # the xDMA's lanes a beat
OAR = 2                       # a_row bytes a value: one 8-byte word for 4 values


class LlamaDecodeData(llama3_data.LlamaAttnData):
    """LlamaAttnData over a fresh causal prompt of ctx + 1 positions whose last is the decode token: the same synthetic
    weights and the same prompt draw, every static scale the absolute maximum over all ctx + 1 positions (s_ot over every
    head's o~ of every position), so positions 0 .. ctx - 1 of K and V are the cache the prefill of this prompt leaves.

    Positions 0 .. ctx - 1 run the prefill's passes of `rows` queries over key tiles of `bc` (LlamaAttnData.sweep); the
    decode token runs as the device does: one 16-lane query tile, lane 0 its q8 and lanes 1..15 zero, over ctx / bc + 1
    key tiles, the last holding key ctx at offset 0 and zero keys after it, every key after ctx masked for every lane.
    Each lane's arithmetic is its own, so lane 0 is the token's row of the prompt's golden."""

    def __init__(self, ctx=1024, rows=64, bc=64, seed=1, wbits=4, mask_c=False):
        self.mask_c = bool(mask_c)
        fp, hm, layout, pack, ref = model_data.import_golden()
        import simd
        self.fp, self.hm, self.layout, self.pack, self.ref, self.simd = fp, hm, layout, pack, ref, simd
        self.mesh = layout.MESH
        c = llama3_data.LLAMA3_8B
        self.H, self.NH, self.NKV, self.HD = c["hidden"], c["heads"], c["kv_heads"], c["head_dim"]
        self.GQ = self.NH // self.NKV
        self.ctx, self.R, self.bc, self.wbits = int(ctx), int(rows), int(bc), int(wbits)
        ctx, R, bc, H, HD = self.ctx, self.R, self.bc, self.H, self.HD
        if ctx < R or ctx % R or R % 16 or R % bc or bc % 16:
            raise ValueError(f"ctx {ctx} rows {R} bc {bc}: the cache in whole prefill passes and whole key tiles")
        self.T = T = ctx + 1
        self.NP = ctx // R                                # the prefill's passes over the cached positions
        self.NT = ctx // bc + 1                           # the decode's key tiles: the cached ones and the new key's
        F16, F32 = np.float16, np.float32
        # ---- the synthetic weights: LlamaAttnData's draws, in its order --------------------------------------------
        r = llama3_data._rng(seed, 3)
        draw = lambda shape: r.standard_normal(shape, dtype=F32) * F32(llama3_data.INIT_STD)
        Wq, Wk, Wv = draw((H, self.NH * HD)), draw((H, self.NKV * HD)), draw((H, self.NKV * HD))
        Wo = draw((self.NH * HD, H))
        self.gain = F32(1.0) + r.standard_normal(H, dtype=F32) * F32(llama3_data.GAIN_STD)
        self._float = dict(wq=Wq, wk=Wk, wv=Wv, wo=Wo)
        qc = lambda W: pack.quantize_columns(W, wbits=wbits)
        (self.wq_q, self.wq_s), (self.wk_q, self.wk_s) = qc(Wq), qc(Wk)
        (self.wv_q, self.wv_s), (self.wo_q, self.wo_s) = qc(Wv), qc(Wo)
        # ---- the prompt (its first ctx rows are the ctx-token prompt's draw) and its normed INT8 --------------------
        self.x16 = llama3_data._rng(seed, 4).standard_normal((T, H)).astype(F16)
        self.gain16 = self.gain.astype(F16)
        xg16 = fp.mul16(hm.rmsnorm(self.x16), self.gain16[None, :])
        QS = hm.QScale
        amax = lambda a: float(np.max(np.abs(np.asarray(a, dtype=np.float64))))
        self.s_x = QS.for_amax(amax(xg16))
        self.hq = fp.quant_i8(xg16, self.s_x.inv)
        s = fp.d_shift_for_depth
        self.k_x, self.k_s, self.k_o, self.k_wo = s(H), s(HD), s(T), s(self.NH * HD)
        inv = 1.0 / (c["rope_theta"] ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
        ang = np.arange(T, dtype=np.float64)[:, None] * inv[None, :]
        cos, sin = np.cos(ang).astype(F16), np.sin(ang).astype(F16)
        self.cos_rep = np.repeat(cos, 2, axis=1)
        self.sin_sgn = np.empty((T, HD), dtype=F16)
        self.sin_sgn[:, 0::2], self.sin_sgn[:, 1::2] = -sin, sin
        # ---- the projections of every position --------------------------------------------------------------------
        dq = lambda ws, cols, k: hm.dequant_scale(self.s_x.s, ws[cols], k)
        self.q_f = [dq(self.wq_s, slice(h * HD, (h + 1) * HD), self.k_x) for h in range(self.NH)]
        self.k_f = [dq(self.wk_s, slice(m * HD, (m + 1) * HD), self.k_x) for m in range(self.NKV)]
        self.v_f = [dq(self.wv_s, slice(m * HD, (m + 1) * HD), self.k_x) for m in range(self.NKV)]
        proj = lambda wq, f, cols: fp.mul16(fp.d_port(llama3_data._mm(self.hq, wq[:, cols]), self.k_x), f[None, :])
        hc = lambda i: slice(i * HD, (i + 1) * HD)
        self.q16 = [proj(self.wq_q, self.q_f[h], hc(h)) for h in range(self.NH)]
        self.k16 = [proj(self.wk_q, self.k_f[m], hc(m)) for m in range(self.NKV)]
        self.v16 = [proj(self.wv_q, self.v_f[m], hc(m)) for m in range(self.NKV)]
        self.q_rot = [self.rope(q) for q in self.q16]
        self.k_rot = [self.rope(k) for k in self.k16]
        self.s_q = QS.for_amax(max(amax(q) for q in self.q_rot))
        self.s_k = QS.for_amax(max(amax(k) for k in self.k_rot))
        self.s_v = QS.for_amax(max(amax(v) for v in self.v16))
        self.q8 = [fp.quant_i8(q, self.s_q.inv) for q in self.q_rot]
        self.k8 = [fp.quant_i8(k, self.s_k.inv) for k in self.k_rot]
        self.v8 = [fp.quant_i8(v, self.s_v.inv) for v in self.v16]
        self.a_exp = float(F32((1.0 / np.sqrt(HD)) * self.s_q.s * self.s_k.s * 2.0 ** self.k_s))
        self.a_n = float(F32(hm.P8_SCALE / (2.0 ** self.k_o * self.s_v.s)))
        self.a_n_bits = fp.f32bits(self.a_n)
        # ---- the masks: the prefill passes' (the golden of positions 0 .. ctx - 1), and the decode tile's ----------
        self.mask_tiles = self._mask_tiles()
        keys = (self.NT - 1) * bc + np.arange(bc)
        self.dec_mask = np.where(np.repeat((keys > ctx)[:, None], BR, axis=1), F16(hm.MASKED_SCORE),
                                 F16(0)).astype(F16)                        # [bc, BR], a key a row, a query a lane
        self.dec_bias = np.where(self.dec_mask != 0, self.mask_bias, 0).astype(np.int64)
        # ---- the sweeps, o~ and o8 -------------------------------------------------------------------------------
        self.dec_ot16 = {}
        self.ot16 = [self.head_attention(h) for h in range(self.NH)]
        self.s_ot = QS.for_amax(max(amax(o) for o in self.ot16))
        self.ot8 = [fp.quant_i8(o, self.s_ot.inv) for o in self.ot16]
        self.dec_ot8 = {h: fp.quant_i8(self.dec_ot16[h], self.s_ot.inv) for h in range(self.NH)}
        self.wo_f = hm.dequant_scale(self.s_ot.s, self.wo_s, self.k_wo)
        # ---- the token's y: every head's o8 row through W_O, off the D port (W_O by columns: no fold) ------------
        orow = np.concatenate([self.ot8[h][ctx] for h in range(self.NH)])[None, :]
        self.y_raw = fp.d_port(llama3_data._mm(orow, self.wo_q), self.k_wo)[0]
        self.y_ref = self._float_last()

    def head_attention(self, h):
        """o~ of query head h for every position, FP16 [T, HD]: the prefill's passes over the cached positions, then the
        decode tile's lane 0 (its 16 lanes kept in dec_ot16)."""
        tile = self.decode_tile(h)
        self.dec_ot16[h] = tile
        return np.concatenate([super().head_attention(h), tile[:1]])

    def decode_tile(self, h):
        """The decode token's sweep as the device runs it: o~ of the 16-lane query tile, FP16 [BR, HD]."""
        fp, hm, simd = self.fp, self.hm, self.simd
        F16, F32, F64 = np.float16, np.float32, np.float64
        bc, HD, NT = self.bc, self.HD, self.NT
        m = h // self.GQ
        q = np.zeros((HD, BR), dtype=F64)
        q[:, 0] = self.q8[h][self.ctx]
        K8, V8 = np.zeros((NT * bc, HD), dtype=F64), np.zeros((NT * bc, HD), dtype=F64)
        K8[:self.T], V8[:self.T] = self.k8[m], self.v8[m]
        a32 = F32(self.a_exp)
        m16 = np.full(BR, -65504.0, dtype=F16)
        l16 = np.zeros(BR, dtype=F16)
        o = None
        for j in range(NT):
            S = np.rint(K8[j * bc:(j + 1) * bc] @ q).astype(np.int64)
            if j < NT - 1:
                S16 = fp.d_port(S, self.k_s)
            elif self.mask_c:
                S16 = fp.d_port(S + self.dec_bias, self.k_s)
            else:
                S16 = fp.add16(fp.d_port(S, self.k_s), self.dec_mask)
            if not np.all(np.isfinite(S16)):
                raise FloatingPointError("a decode score overflows FP16 after the D-port shift")
            rmax = simd.reduce_lanewise(S16, simd.MAX)
            mnew = simd.reduce_lanewise(np.stack([rmax, m16]), simd.MAX)
            corr16 = hm.exp16(fp.add16(m16, -mnew), a32)
            p16 = hm.exp16(fp.add16(S16, -mnew[None, :]), a32)
            rsum16 = simd.reduce_lanewise(p16, simd.ADD)
            l16 = simd.reduce_lanewise(np.stack([rsum16, fp.mul16(corr16, l16)]), simd.ADD)
            p8 = fp.quant_i8(p16, hm.P8_SCALE).astype(F64)
            pv = np.rint(V8[j * bc:(j + 1) * bc].T @ p8).astype(np.int64)
            if j == 0:
                o = pv
            else:
                o = np.clip(np.rint(o.astype(F64) * corr16.astype(F64)[None, :]), -2**31, 2**31 - 1).astype(np.int64) + pv
            m16 = mnew
        O16 = fp.d_port(o, self.k_o)
        t16 = simd.stream_map(l16, self.a_n, 0.0, simd.RSQRT)
        return np.ascontiguousarray(fp.mul16(O16, fp.mul16(t16, t16)[None, :]).T)

    def _float_last(self):
        """The decode token's attention output in float64 on the same prompt and weights, [H]."""
        F64 = np.float64
        W, c, HD, ctx = self._float, llama3_data.LLAMA3_8B, self.HD, self.ctx
        x = self.x16.astype(F64)
        xg = x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + c["rms_eps"]) * self.gain.astype(F64)[None, :]
        cos, sin = self.cos_rep.astype(F64), self.sin_sgn.astype(F64)

        def rot(v, r0):
            vs = np.empty_like(v)
            vs[:, 0::2], vs[:, 1::2] = v[:, 1::2], v[:, 0::2]
            n = v.shape[0]
            return v * cos[r0:r0 + n] + vs * sin[r0:r0 + n]
        kv = {}
        outs = []
        for h in range(self.NH):
            m = h // self.GQ
            if m not in kv:
                kv[m] = (rot(xg @ W["wk"][:, m * HD:(m + 1) * HD].astype(F64), 0),
                         xg @ W["wv"][:, m * HD:(m + 1) * HD].astype(F64))
            k, v = kv[m]
            q = rot(xg[ctx:] @ W["wq"][:, h * HD:(h + 1) * HD].astype(F64), ctx)
            s = q @ k.T / np.sqrt(HD)
            p = np.exp(s - s.max())
            outs.append((p / p.sum()) @ v)
        return (np.concatenate(outs, axis=1) @ W["wo"].astype(F64))[0]

    def report(self, y16):
        ref = np.asarray(self.y_ref, dtype=np.float64).reshape(-1)
        got = np.asarray(y16, dtype=np.float64).reshape(-1)
        cos = float(got @ ref / max(np.linalg.norm(got) * np.linalg.norm(ref), 1e-30))
        rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30))
        sc = lambda q: f"{q.s:.4g}"
        return [f"{llama3_data.MODEL}: attention of one layer, one decode token at position {self.ctx} over a cache of "
                f"{self.ctx} positions (a fresh causal prompt of {self.T}, the token the last), {self.NH} query / "
                f"{self.NKV} key-value heads of {self.HD}, INT{self.wbits} weights per column; {self.NT} key tiles of "
                f"{self.bc}",
                f"D-port shifts k_x={self.k_x} k_s={self.k_s} k_o={self.k_o} k_wo={self.k_wo}; scales s_x={sc(self.s_x)} "
                f"s_q={sc(self.s_q)} s_k={sc(self.s_k)} s_v={sc(self.s_v)} s_ot={sc(self.s_ot)}",
                f"device y vs the float attention (the token's row): cosine {cos:.5f}, relative L2 error {rel:.4f}"]


class DBuild(ABuild):
    TAG = "ld"
    # x_on_host: x's row sits in the root memory chip's HBM and the root's host multicasts it (x_multicast) ahead of its
    # cache and weight pushes; y_to_hbm: the root writes y into that HBM (y_out). A whole layer
    # (llama3_decode_layer_mc.py) makes x on the device and keeps y in the root's L3
    x_on_host = True
    y_to_hbm = True

    def __init__(self, args, p, plat, hw, shared=None):
        """The parameters; no data, memory or task yet. The decode runs in ABuild's four steps (__init__, mem_handles(),
        build(), finish()), but neither ABuild.__init__ nor ABuild.mem_handles: their data, staging and buffers are a
        prompt's passes. The helpers taken from ABuild (_load, nd, copy, copy2d, mul, add, quant, xd, kv_cc, l1_of,
        ot8_bits, _pack_b) read only what is set here and in mem_handles. ChipletBuild.__init__ (a layer's block shares
        the layer's staging and graph), then the decode's switches with their checks, the rows, the key tile, the
        cache's length and parts."""
        cm.ChipletBuild.__init__(self, args, p, plat, hw, shared=shared)
        if self.T != 1:
            raise ValueError(f"params tokens={self.T}: the decode runs one token")
        if simd_fp16_lanes() != BR:
            raise ValueError(f"the SIMD has {simd_fp16_lanes()} FP16 lanes; the softmax takes one query a lane")
        if p.get("x_bcast", "memchip") != "memchip":
            raise ValueError(f"params x_bcast={p.get('x_bcast')!r}: x comes from HBM (memchip)")
        self.w4_paired = bool(p.get("w4_paired", True))
        if not self.w4_paired or self.wbits != 4:
            raise ValueError("params w4_paired, wbits 4: the GEMV reads INT4 weights in the paired layout")
        self.restream, self.ot8_fused, self.kv_split, self.rope_pairs = False, True, True, "chunks"
        self.mask_c = bool(p.get("mask_c", True))
        self.sm_geom = "xdma" if bool(p.get("sm_geom_xdma", False)) else None
        self.checks_kind = str(p.get("checks", "y"))
        if self.checks_kind not in ("y", "stages"):
            raise ValueError(f"params checks={self.checks_kind!r}: y or stages")
        self.R = int(p.get("rows", 64))
        self.bc = int(p.get("bc", 64))
        self.ctx = int(p.get("ctx", 1024))
        self.seed = int(p.get("seed", 1))               # the synthetic weights' and prompt's draws
        # params cache_parts: the cache in this many parts of whole key tiles, a push and a flag each, so the sweep's
        # first tiles start while the later parts land
        self.P = int(p.get("cache_parts", 2))
        self.slot = {}
        self.xch_zero = False                                # no exchange buffer: x and o land in L3

    def mem_handles(self):
        """The data and the memory; no task yet. ABuild.mem_handles is not run, as ABuild.__init__ is not: the data
        and its golden (LlamaDecodeData), the shapes that follow from the model (the key tiles, the weight chunks of a
        head, W_O's columns a cluster, the cache's parts), the key/value groups on chip pairs and a query head a
        cluster, the report; _stage (x, y, o and the cache's areas and flags in L3, every chip's half of its group's
        cache and every cluster's weights and constants in HBM); the load streams' L1 slabs, the weight rings and the
        L1 buffers (_buffers)."""
        p = self.p
        d = self.d = LlamaDecodeData(ctx=self.ctx, rows=self.R, bc=self.bc, seed=self.seed,
                                     wbits=self.wbits, mask_c=self.mask_c)
        self.lay, self.mesh = d.layout, d.mesh
        self.HD, self.H = d.HD, d.H
        self.NT, self.CT = d.NT, d.NT - 1                    # key tiles; the cached ones
        self.LAST = self.CT                                  # the new key's tile
        self.NQC = self.HD // QC
        self.NKVC = self.NQC // 2
        self.WO0 = self.NQC + self.NKVC
        if len(self.chips) * self.NC != d.NH:
            raise ValueError(f"{len(self.chips) * self.NC} clusters: the attention has {d.NH} query heads, one a cluster")
        self.WOC = self.H // (len(self.chips) * self.NC)     # y's columns a cluster
        self.NWS = self.WOC // 32                            # its W_O chunks of 32 columns
        first = np.arange(QC) * 2
        self.perm = np.concatenate([np.concatenate([t * 2 * QC + first, t * 2 * QC + first + 1])
                                    for t in range(self.NQC // 2)])
        if self.P < 1 or self.CT % self.P:
            raise ValueError(f"params cache_parts={self.P}: whole key tiles of the {self.CT} cached ones a part")
        self.PT = self.CT // self.P                          # key tiles a part
        # params cache_after: each ring's chunks the prefetcher pushes before the cache (its first part)
        self.cache_after = int(p.get("cache_after", self.NQC))
        # ---- the split: key/value groups on chip pairs of a column, a query head a cluster (the prefill's) -----------
        pairs = sorted({(k, k + 1) for k in self.chips if (k & 15) % 2 == 0 and k + 1 in self.chips})
        if len(pairs) != d.NKV:
            raise ValueError(f"{len(pairs)} chip pairs (x, 2y), (x, 2y + 1): the attention has {d.NKV} key/value groups")
        self.pairs = pairs
        self.group, self.head = {}, {}
        for m, (ka, kb) in enumerate(pairs):
            for n, (k, c) in enumerate(((ka, 0), (ka, 1), (kb, 0), (kb, 1))):
                cl = self.chips.index(k) * self.NC + c
                self.group[k] = m
                self.head[cl] = m * d.GQ + n
        self.kind = {cl: "K" if self.local(cl) == 0 else "V" for cl in range(self.N_CL)}
        self.kv_half, self.kv_mate = {}, {}
        for ka, kb in pairs:
            self.kv_half[ka], self.kv_half[kb] = 0, 1
            self.kv_mate[ka], self.kv_mate[kb] = kb, ka
        self.TILE_B = self.bc * self.HD                      # a key or value tile, INT8
        self.HB = self.TILE_B // 2                           # a tile's half
        self.KRUN = self.NKVC * QC * BR                      # a key half's run in a 16-key block
        self.PB = 2 * self.PT * self.HB                      # a cache part: [K halves | V halves]
        for line in d.report(d.y16(d.y_raw)):
            print(f"[{self.TAG}] {line}")
        print(f"[{self.TAG}] decode: {self.CT} cached key tiles of {self.bc} in {self.P} parts and the new key's tile; "
              f"key/value groups {[(f'{a:02x}', f'{b:02x}') for a, b in pairs]}; W_O by columns, {self.WOC} a cluster")
        self._stage()
        self.make_streams()
        self.make_rings()
        self._buffers()

    # ---- data -------------------------------------------------------------------------------------------------------
    def _a_row(self, row):
        """A [K] INT8 row in a_row: value c at (c // 4) * 8 + c % 4, the upper 4 bytes of each word zero."""
        c = np.arange(row.size)
        out = np.zeros(2 * row.size, dtype=np.int8)
        out[(c // 4) * 8 + c % 4] = row
        return out

    def _ktile(self, m, j):
        """Key tile j of group m in the A layout, rope_pairs' column order (keys past the prompt zero), INT8 flat."""
        d, bc = self.d, self.bc
        k = np.zeros((bc, self.HD), dtype=np.int8)
        rows = d.k8[m][j * bc:(j + 1) * bc]
        k[:rows.shape[0]] = rows[:, self.perm]
        return self.lay.to_a(k, self.mesh).astype(np.int8)

    def _vtile(self, m, j):
        """Value tile j of group m, V^T in the A layout (keys past the prompt zero), INT8 flat."""
        d, bc = self.d, self.bc
        v = np.zeros((bc, self.HD), dtype=np.int8)
        rows = d.v8[m][j * bc:(j + 1) * bc]
        v[:rows.shape[0]] = rows
        return self.lay.to_a(np.ascontiguousarray(v.T), self.mesh).astype(np.int8)

    def _khalf(self, m, j, hh):
        """Half hh of key tile j: its run of every 16-key block, compact (the prefill's k8h)."""
        return self._ktile(m, j).reshape(self.bc // BR, 2, self.KRUN)[:, hh].reshape(-1)

    def _vhalf(self, m, j, hh):
        """Half hh of value tile j: its contiguous half of V^T."""
        return self._vtile(m, j).reshape(2, self.HB)[hh]

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
        # y: the token's row in the root memory chip's HBM; the row the clusters fill in the root's L3, a flag each
        self.YB = 2 * self.H
        if self.y_to_hbm:
            self.h_yhbm = st.put_hbm(f"{T}_yhbm", np.zeros(self.YB, dtype=np.uint8), mem_chip=root_mem)
        st.put_zeros(f"{T}_yrow", "uint8_t", self.YB)
        st.put_zeros(f"{T}_yflag", "uint8_t", self.N_CL * 64)
        # o: every head's o8 row (a_row) gathered in the root's L3 and multicast to the same place on every chip
        self.OB = OAR * HD
        st.put_zeros(f"{T}_oland", "uint8_t", d.NH * self.OB)
        st.put_zeros(f"{T}_oflag", "uint8_t", d.NH * 64)
        st.put_zeros(f"{T}_omflag", "uint8_t", 64)
        # the cache: the chip's half lands here (a flag and a progress flag a part), the mate's half is relayed here
        # (a flag a part and kind), the new tile's halves trade through kvx
        st.put_zeros(f"{T}_kvc", "uint8_t", self.P * self.PB)
        st.put_zeros(f"{T}_kvcflag", "uint8_t", self.P * 64)
        st.put_zeros(f"{T}_kvcprog", "uint8_t", self.P * 64)
        st.put_zeros(f"{T}_kvm", "uint8_t", self.P * self.PB)
        st.put_zeros(f"{T}_kvmflag", "uint8_t", 2 * self.P * 64)
        self.KVX = self.KRUN + self.HB
        st.put_zeros(f"{T}_kvx", "uint8_t", self.KVX)
        # per chip, in its memory chip's HBM: its half of the group's cache, part by part, then the new key's tile
        self.h_kv, self.APP = {}, self.P * self.PB
        for k in self.chips:
            m, hh = self.group[k], self.kv_half[k]
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            parts = []
            for q in range(self.P):
                js = range(q * self.PT, (q + 1) * self.PT)
                parts += [self._khalf(m, j, hh) for j in js] + [self._vhalf(m, j, hh) for j in js]
            parts.append(np.zeros(2 * self.HB, dtype=np.int8))
            self.h_kv[k] = st.put_hbm(f"{T}_kv_k{k:02x}", np.concatenate(parts), mem_chip=loc)
        # per cluster, in its memory chip's HBM: its weights in the order it streams them (W_q's chunks, then its half of
        # W_k or W_v, then its 32-column chunks of W_O) and its constants (each projection's column factors, the RoPE
        # tables at the token's position, the new key tile's mask)
        self.img, self.chunk_off, self.cst = {}, {}, {}
        masks = (d.dec_bias.astype(np.int32) if self.mask_c else d.dec_mask)
        for cl in range(self.N_CL):
            k, h = self.chip(cl), self.head[cl]
            m = self.group[k]
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            wkv, fkv = (d.wk_q, d.k_f[m]) if self.kind[cl] == "K" else (d.wv_q, d.v_f[m])
            qcol = lambda cc: self.perm[cc * QC:(cc + 1) * QC]
            kcol = qcol if self.kind[cl] == "K" else (lambda cc: np.arange(cc * QC, (cc + 1) * QC))
            parts = [self._pack_b(d.wq_q[:, h * HD + qcol(cc)]) for cc in range(self.NQC)]
            parts += [self._pack_b(wkv[:, m * HD + kcol(cc)]) for cc in self.kv_cc(cl)]
            for s in range(self.NWS):
                o0 = self.WOC * cl + s * 32
                parts.append(self._pack_b(d.wo_q[:, o0:o0 + 32]))
            offs, off = [], 0
            for b in parts:
                if b.size > self.chunk:
                    raise ValueError(f"a weight chunk of {b.size} B; params w_chunk_bytes is {self.chunk}")
                offs.append((off, b.size))
                off += b.size
            self.img[cl] = st.put_hbm(f"{T}_w_cl{cl}", np.concatenate(parts).astype(np.int8), mem_chip=loc)
            self.chunk_off[cl] = offs
            f16 = lambda a: np.asarray(a, dtype=np.float16).reshape(-1).view(np.int8)
            blob = np.concatenate([f16(d.q_f[h][self.perm]),
                                   f16(np.concatenate([fkv[kcol(cc)] for cc in self.kv_cc(cl)])),
                                   f16(d.cos_rep[self.ctx][self.perm]), f16(d.sin_sgn[self.ctx][self.perm]),
                                   np.ascontiguousarray(masks).reshape(-1).view(np.int8)])
            self.cst[cl] = (st.put_hbm(f"{T}_c_cl{cl}", blob, mem_chip=loc), blob.nbytes)
        self.C_QF, self.C_KVF = 0, 2 * HD
        self.C_COS = self.C_KVF + 2 * self.NKVC * QC
        self.C_SIN = self.C_COS + 2 * HD
        self.C_MASK = self.C_SIN + 2 * HD
        if any(o % 64 for o in (self.C_KVF, self.C_COS, self.C_SIN, self.C_MASK)):
            raise ValueError("the constants' parts are not 64-byte aligned")

    def _buffers(self):
        HD, NC = self.HD, self.NKVC * QC
        self.kbuf, self.vbuf, self.consts, self.sweep_bufs = {}, {}, {}, {}
        self.xa, self.q16, self.rt, self.rs, self.a16q, self.q8 = {}, {}, {}, {}, {}, {}
        self.kv16, self.a16k, self.vr, self.vtt, self.va = {}, {}, {}, {}, {}
        self.oar, self.ofull, self.ywo = {}, {}, {}
        for cl in range(self.N_CL):
            l1 = self.l1_of(cl)
            self.kbuf[cl] = l1("kt", self.NT * self.TILE_B)      # every key tile, A layout
            self.vbuf[cl] = l1("vt", self.NT * self.TILE_B)      # every value tile, V^T in the A layout
            self.consts[cl] = l1("consts", self.cst[cl][1])
            self.xa[cl] = l1("x", self.XB)                       # the token's row, a_row
            self.q16[cl] = l1("q16", 2 * HD)                     # q: the GEMVs' rows, dequantised and rotated in place
            # RoPE scratch, q's and k's each their own (their RoPE may run in either order): x cos, chunk by chunk;
            # the pair's other chunk x sin
            self.rt[cl] = {sd: l1(f"rt{sd}", 2 * HD) for sd in ("q", "k")}
            self.rs[cl] = {sd: l1(f"rs{sd}", 2 * HD) for sd in ("q", "k")}
            self.a16q[cl] = l1("a16q", BR * HD * 2)              # q in row 0 of a 16-row FP16 A tile
            self.q8[cl] = l1("q8", BR * HD)                      # the sweep's query tile
            self.kv16[cl] = l1("kv16", 2 * NC)                   # the chip's half of k or v, the GEMVs' row
            if self.kind[cl] == "K":
                self.a16k[cl] = l1("a16k", BR * NC * 2)          # k's half in row 0 of a 16-row FP16 A tile
            else:
                for n in range(self.NKVC):
                    self.vr[cl, n] = l1(f"vr{n}", self.bc * QC * 2)   # v's chunk in row 0 of a zeroed [bc, 32] block
                    self.vtt[cl, n] = l1(f"vtt{n}", QC * self.bc * 2)  # transposed
                    self.va[cl, n] = l1(f"va{n}", QC * self.bc * 2)   # in the A layout
            self.sweep_bufs[cl] = fa_sweep_alloc(lambda name, nb, l1=l1: l1(name, nb), self._sweep_cfg())
            self.oar[cl] = l1("oar", self.OB)                    # the head's o8 row, a_row
            self.ofull[cl] = l1("ofull", self.d.NH * self.OB)    # every head's: W_O's A operand
            self.ywo[cl] = l1("ywo", 2 * self.WOC)               # the cluster's columns of y

    def _sweep_cfg(self):
        d = self.d
        return FaSweepCfg(bc=self.bc, nt=self.NT, nq=1, br=BR, k_s=d.k_s, k_o=d.k_o, a_exp=d.a_exp,
                          a_n_f32bits=d.a_n_bits, d_qk=self.HD, d_v=self.HD)

    # ---- helpers ----------------------------------------------------------------------------------------------------
    def gemv(self, cl, name, x, w, out, n, k_shift, deps):
        """One weight chunk of n columns: the token's a_row x . w (paired INT4) into the FP16 row out."""
        return self.nd(cl, "gemm", name, "__snax_bingo_kernel_gemv",
                       SnaxBingoKernelGemvArgs(x, w, out, K=self.H, N=n, d_shift=k_shift, a_blk=8, w4=True), deps)

    def memset(self, cl, name, buf, nb, deps=()):
        return self.nd(cl, "xdma", name, "__snax_bingo_kernel_xdma_memset",
                       SnaxBingoKernelXdmaMemsetArgs(buf, nb, SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), list(deps))

    def row_to_a(self, cl, name, src, dst, cols, deps):
        """An FP16 row of `cols` values into row 0 of a 16-row FP16 A tile: its 8-byte atoms (4 values, one A block's
        row) 4 * BR * 2 bytes apart, on the xDMA."""
        blk = 4 * BR * 2
        atoms = cols // 4
        if atoms % LANES:
            raise ValueError(f"row_to_a: {cols} values are not whole xDMA beats of {LANES} atoms")
        return self.xd(cl, name, SnaxBingoKernelXdma6dArgs(src, dst, ATOM, blk, [LANES * ATOM], [atoms // LANES],
                                                          [LANES * blk], [atoms // LANES]), deps)

    def a_to_row(self, cl, name, src, dst, cols, deps):
        """Rows 0 and 1 of a 16-row INT8 A tile of `cols` columns into a_row: the first 8-byte word of every 64-byte
        block, on the xDMA (the GEMV takes row 0 of each word)."""
        blk = 4 * BR
        words = cols // 4
        if words % LANES:
            raise ValueError(f"a_to_row: {cols} columns are not whole xDMA beats of {LANES} words")
        return self.xd(cl, name, SnaxBingoKernelXdma6dArgs(src, dst, blk, ATOM, [LANES * blk], [words // LANES],
                                                          [LANES * ATOM], [words // LANES]), deps)

    def rope_row(self, cl, side, x, ca, cb, la_, lb, deps):
        """RoPE of one row's chunk pair in place (rope_pairs chunks): chunk ca of the head (the pairs' first
        dimensions) at x + 64 la_, chunk cb (their second) at x + 64 lb; a = a cos_a + b sin_a, b = b cos_b + a sin_b
        (sin signed), the tables the constants' rows at the token's position. Returns the two adds."""
        cst, rt, rs, n = self.consts[cl], self.rt[cl][side], self.rs[cl][side], QC
        cos = lambda cc: cst.view(self.C_COS + 2 * QC * cc)
        sin = lambda cc: cst.view(self.C_SIN + 2 * QC * cc)
        xa, xb = x.view(2 * QC * la_), x.view(2 * QC * lb)
        dep = list(deps)
        ta = self.mul(cl, f"{side}rc{ca}", xa, cos(ca), rt.view(2 * QC * la_), n, dep)
        sa = self.mul(cl, f"{side}rs{ca}", xb, sin(ca), rs.view(2 * QC * la_), n, dep)
        tb = self.mul(cl, f"{side}rc{cb}", xb, cos(cb), rt.view(2 * QC * lb), n, dep)
        sb = self.mul(cl, f"{side}rs{cb}", xa, sin(cb), rs.view(2 * QC * lb), n, dep)
        ra = self.add(cl, f"{side}ra{ca}", rt.view(2 * QC * la_), rs.view(2 * QC * la_), xa, n, [ta, sa, sb])
        rb = self.add(cl, f"{side}ra{cb}", rt.view(2 * QC * lb), rs.view(2 * QC * lb), xb, n, [tb, sb, sa])
        return ra, rb

    def loads(self, cl, js):
        """The loads of cluster cl's weight chunks js that fit its slabs now: the first w_buffers of them."""
        return {n: self._load(cl, 0, js[n]) for n in range(min(len(self.ls[cl].slabs), len(js)))}

    def chunks(self, cl, name, js, x, outs, k_shift, deps, then=None, lds=None):
        """Cluster cl's weight chunks js, chunk n a GEMV of x into the row outs[n] behind its load; the first w_buffers
        loads made here unless given (lds), each later one right after the GEMV that frees its slab. then(n, gemv)
        makes what follows chunk n's GEMV. Returns the GEMVs."""
        ls, nb = self.ls[cl], len(self.ls[cl].slabs)
        lds = dict(lds) if lds is not None else self.loads(cl, js)
        gms = []
        for n, j in enumerate(js):
            ld, si, slab = lds[n]
            gm = self.gemv(cl, f"{name}{n}", x, slab, outs[n], 32, k_shift, [ld] + list(deps))
            ls.release(si, gm)
            gms.append(gm)
            if n + nb < len(js):
                lds[n + nb] = self._load(cl, 0, js[n + nb])
            if then is not None:
                then(n, gm)
        return gms

    def l3(self, name, off=0, chip=None):
        """An L3 symbol of this block at byte `off`, on chip `chip` (None: the chip that names it)."""
        return BingoMemSymbol(f"{self.TAG}_{name}", offset=off, chip_id=chip)

    def _prefetch_head(self, k, n, after):
        """Chip k's prefetcher's first n chunks of each ring in a task of their own behind `after`; the prefetcher then
        starts at the next chunk (the prefill's split_root_prefetch on any chip). Returns the first part."""
        t, b = self.prefetch[k], self.prefetch[k].kernel_args
        if not 0 < n <= self.n_slots:
            raise ValueError(f"params cache_after={n}: 1..{self.n_slots} (the ring's slots: the first part never waits)")
        part = lambda lo, hi: HostBingoKernelWeightPrefetchRangeArgs(
            b.sched_addr, b.ring_addr, b.flag_addr, b.release_addr, b.seq_addr, b.rec_addr, b.rec_slots, b.n_rings,
            b.n_slots, b.slot_bytes, b.memchip_id, batch=b.batch, policy=b.policy, engine=b.engine,
            batch_routed=b.batch_routed, queued=b.queued, queued_inflight_routed=b.queued_inflight_routed, head=b.head,
            batch_head=b.batch_head, chunk_lo=lo, chunk_hi=hi)
        t.kernel_args = part(n, 0)
        t.kernel_name = t.kernel_args.KERNEL_NAME
        return self.cx_chip(k).host_node(f"weight_prefetch_k{k:02x}_head", t.kernel_name, part(0, n), after=list(after))

    def x_multicast(self, go):
        """x's row into every chip's L3 (xland) behind the flag xtflag: the root's host has its memory chip's push engine
        multicast it from HBM, behind the root's go. Returns the task the x fetches follow."""
        R = self.ROOT
        mR = self.mem[R]
        return self.cx_chip(R).host_node(
            "x_hbm_bcast", "__host_bingo_kernel_hbm_bcast",
            HostBingoKernelHbmBcastArgs(self.h_xhbm.address, self.l3("xland"), self.l3("xtflag"), self.h_seq[mR].address + 64,
                                        1, self.XB, 1, mR, self.engine[R]), after=[go[R]])

    def y_out(self, yw):
        """y's row from the root's L3 into the HBM of the root's memory chip, behind yw (the root's wait for every
        cluster's columns). Returns the task the root's end follows."""
        return self.node(self.ROOT, 0, self.roles["dm"], "y_hbm", "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(self.l3("yrow"), self.h_yhbm, self.YB), [yw])

    # ---- the block --------------------------------------------------------------------------------------------------
    def build(self):
        d, R, HD = self.d, self.ROOT, self.HD
        dm, xdma = self.roles["dm"], self.roles["xdma"]
        NC = self.NKVC * QC
        L16 = self.LAST * self.TILE_B                     # the new key's tile in kbuf / vbuf
        self.tile_ready = {}
        for ls in self.ls:
            ls.strict = True
        # ---- before the window: the constants into L1; the new tile and the zero-padded operands zeroed -----------
        pre = {k: [] for k in self.chips}
        for cl in range(self.N_CL):
            k = self.chip(cl)
            h, nb = self.cst[cl]
            pre[k].append(self.copy(cl, f"const_ld", h.address, self.consts[cl], nb, []))
            z = [(self.kbuf[cl].view(L16), self.TILE_B), (self.vbuf[cl].view(L16), self.TILE_B),
                 (self.a16q[cl], BR * HD * 2)]
            z += ([(self.a16k[cl], BR * NC * 2)] if self.kind[cl] == "K" else
                  [(self.vr[cl, n], self.bc * QC * 2) for n in range(self.NKVC)])
            prev = []
            for i, (buf, n) in enumerate(z):
                prev = [self.memset(cl, f"zero{i}", buf, n, prev)]
            pre[k] += prev
        go = self.setup(extra=pre)
        self.dfg.relay_one_consumer_cores = [xdma]
        # ---- the hosts: x's multicast (the root), the weight prefetcher's first part, the chip's cache half, the rest
        one = lambda mm: self.h_seq[mm].address + 64     # a word holding 1 in the memory chip's HBM: the flags' value
        self.x_put = self.x_multicast(go)
        self.ucast = {}
        for k in self.chips:
            first = [go[k]] + ([self.x_put] if k == R and self.x_on_host else [])
            if self.cache_after:
                first = [self._prefetch_head(k, self.cache_after, first)]
            mk = self.mem[k]
            self.ucast[k] = self.cx_chip(k).host_node(
                f"kv_ucast_k{k:02x}", "__host_bingo_kernel_hbm_ucast",
                HostBingoKernelHbmUcastArgs(self.h_kv[k].address, self.l3("kvc"), self.l3("kvcflag"), self.l3("kvcprog"),
                                            one(mk), 1, self.PB, self.P, mk, self.engine[k], [k]), after=first)
            self.edge(self.ucast[k], self.prefetch[k])
        # ---- x into every cluster's L1 ---------------------------------------------------------------------------------
        xf = {}
        for cl in range(self.N_CL):
            xf[cl] = self.nd(cl, "dm", f"x_wait", "__snax_bingo_kernel_idma_fetch_flagged",
                            SnaxBingoKernelIdmaFetchFlaggedArgs(self.l3("xland"), self.xa[cl], self.XB,
                                                                self.l3("xtflag"), 1, 1), [go[self.chip(cl)]])
            self.dfg.bingo_add_order_edge(self.x_put, xf[cl])
        # ---- q: the GEMVs, the column factors, RoPE, the query tile ---------------------------------------------------
        q8t = {}
        for cl in range(self.N_CL):
            cst, q16 = self.consts[cl], self.q16[cl]
            dq, rot = {}, []

            def deq(n, gm, cl=cl, cst=cst, q16=q16, dq=dq, rot=rot):
                # each chunk dequantised behind its GEMV; a chunk pair's RoPE behind its second chunk
                dq[n] = self.mul(cl, f"dqq{n}", q16.view(2 * QC * n), cst.view(self.C_QF + 2 * QC * n),
                                 q16.view(2 * QC * n), QC, [gm])
                if n % 2:
                    rot.extend(self.rope_row(cl, "q", q16, n - 1, n, n - 1, n, [dq[n - 1], dq[n]]))
            self.chunks(cl, "gq", list(range(self.NQC)), self.xa[cl], [q16.view(2 * QC * n) for n in range(self.NQC)],
                        d.k_x, [xf[cl]], then=deq)
            sc = self.row_to_a(cl, "q2a", q16, self.a16q[cl], HD, rot)
            q8t[cl] = self.quant(cl, "q8", self.a16q[cl], self.q8[cl], BR * HD * 2, d.s_q.inv, [sc])
        # ---- the cache: each chip's half from its memory chip into its tiles and relayed to its mate, part by part ----
        relay = {}
        for p in range(self.P):
            t0 = p * self.PT
            for k in self.chips:
                hh, mt = self.kv_half[k], self.kv_mate[k]
                for cl in self.clusters(k):
                    c = self.local(cl)
                    w = self.nd(cl, "dm", f"kvc_wait{p}", "__snax_bingo_kernel_wait_flags",
                                SnaxBingoKernelWaitFlagsArgs(self.l3("kvcflag", p * 64), 1, 1), [])
                    self.dfg.bingo_add_order_edge(self.ucast[k], w)
                    # cluster 0 relays the part's K halves, cluster 1 its V halves, into the mate's L3 behind a flag each
                    off = p * self.PB + c * self.PT * self.HB
                    relay[k, p, c] = self.nd(cl, "dm", f"kv_relay{p}", "__snax_bingo_kernel_idma_put_flagged",
                                             SnaxBingoKernelIdmaPutFlaggedArgs(
                                                 self.l3("kvc", off), self.l3("kvm", off, mt), self.PT * self.HB,
                                                 self.l3("kvmflag", (2 * p + c) * 64, mt), value=1), [w])
                    self._cache_copies(cl, "own", self.l3("kvc", p * self.PB), t0, hh, [w])
            for k in self.chips:
                hh, mt = self.kv_half[k], self.kv_mate[k]
                for cl in self.clusters(k):
                    w = self.nd(cl, "dm", f"kvm_wait{p}", "__snax_bingo_kernel_wait_flags",
                                SnaxBingoKernelWaitFlagsArgs(self.l3("kvmflag", 2 * p * 64), 2, 1), [])
                    for c in range(self.NC):
                        self.dfg.bingo_add_order_edge(relay[mt, p, c], w)
                    ck, cv = self._cache_copies(cl, "mate", self.l3("kvm", p * self.PB), t0, 1 - hh, [w])
                    for j in range(t0, t0 + self.PT):
                        self.tile_ready[(cl, "K", j)], self.tile_ready[(cl, "V", j)] = ck, cv
        # ---- the token's k / v: the chip's half, into the new tile, then traded within the pair --------------------
        own_q = {}
        for cl in range(self.N_CL):
            cst, kv16 = self.consts[cl], self.kv16[cl]
            hh = self.kv_half[self.chip(cl)]
            js = [self.NQC + n for n in range(self.NKVC)]
            if self.kind[cl] == "K":
                dq = {}

                def deqk(n, gm, cl=cl, cst=cst, kv16=kv16, dq=dq):
                    dq[n] = self.mul(cl, f"dqk{n}", kv16.view(2 * QC * n), cst.view(self.C_KVF + 2 * QC * n),
                                     kv16.view(2 * QC * n), QC, [gm])
                self.chunks(cl, "gk", js, self.xa[cl], [kv16.view(2 * QC * n) for n in range(self.NKVC)], d.k_x, [],
                            then=deqk)
                c0 = self.kv_cc(cl)[0]
                rot = self.rope_row(cl, "k", kv16, c0, c0 + 1, 0, 1, [dq[0], dq[1]])
                sc = self.row_to_a(cl, "k2a", kv16, self.a16k[cl], NC, list(rot))
                own_q[cl] = self.quant(cl, "k8", self.a16k[cl], self.kbuf[cl].view(L16 + hh * self.KRUN), BR * NC * 2,
                                      d.s_k.inv, [sc])
            else:
                tas = {}

                def deqv(n, gm, cl=cl, cst=cst, tas=tas):
                    # each chunk's row (row 0 of its zeroed [bc, 32] block) dequantised, transposed and reshaped into V^T's
                    # A layout behind its GEMV; the quantisations after both, so the SIMD core takes the second chunk's
                    # dequantisation while the xDMA reshapes the first
                    vr, vt, va = self.vr[cl, n], self.vtt[cl, n], self.va[cl, n]
                    dqv = self.mul(cl, f"dqv{n}", vr, cst.view(self.C_KVF + 2 * QC * n), vr, QC, [gm])
                    tr = self.nd(cl, "xdma", f"vtr{n}", "__snax_bingo_kernel_xdma_transpose_2d",
                                 SnaxBingoKernelXdmaTranspose2dArgs(vr, vt, self.bc, QC, 2), [dqv])
                    tas[n] = self.xd(cl, f"vta{n}", convert_args("row_major", "A", QC, self.bc, self.mesh, 2, vt, va),
                                     [tr])
                self.chunks(cl, "gv", js, self.xa[cl], [self.vr[cl, n] for n in range(self.NKVC)], d.k_x, [], then=deqv)
                for n, cc in enumerate(self.kv_cc(cl)):
                    own_q[cl] = self.quant(cl, f"vq{n}", self.va[cl, n], self.vbuf[cl].view(L16 + cc * QC * self.bc),
                                          QC * self.bc * 2, d.s_v.inv, [tas[n]])
        kput, vput = {}, {}
        for k in self.chips:
            cl0, cl1 = self.clusters(k)
            hh = self.kv_half[k]
            kput[k] = self.copy(cl0, "kx_put", self.kbuf[cl0].view(L16 + hh * self.KRUN), self.l3("kvx"), self.KRUN,
                                [own_q[cl0]])
            vput[k] = self.copy(cl1, "vx_put", self.vbuf[cl1].view(L16 + hh * self.HB), self.l3("kvx", self.KRUN),
                                self.HB, [own_q[cl1]])
        for k in self.chips:
            cl0, cl1 = self.clusters(k)
            hh, mt = self.kv_half[k], self.kv_mate[k]
            self.copy(cl1, "kx_get", self.l3("kvx"), self.kbuf[cl1].view(L16 + hh * self.KRUN), self.KRUN, [kput[k]])
            self.copy(cl0, "vx_get", self.l3("kvx", self.KRUN), self.vbuf[cl0].view(L16 + hh * self.HB), self.HB,
                      [vput[k]])
            for cl in (cl0, cl1):
                kr = self.copy(cl, "kx_rget", self.l3("kvx", 0, mt), self.kbuf[cl].view(L16 + (1 - hh) * self.KRUN),
                               self.KRUN, [kput[mt]])
                vr = self.copy(cl, "vx_rget", self.l3("kvx", self.KRUN, mt),
                               self.vbuf[cl].view(L16 + (1 - hh) * self.HB),
                               self.HB, [vput[mt]])
                self.tile_ready[(cl, "K", self.LAST)], self.tile_ready[(cl, "V", self.LAST)] = kr, vr
        # ---- W_O's first chunks, behind the trade on the DM core ------------------------------------------------------
        wo_js = [self.WO0 + s for s in range(self.NWS)]
        wl = {cl: self.loads(cl, wo_js) for cl in range(self.N_CL)}
        # ---- the sweep: the cached tiles, then the new key's -----------------------------------------------------------
        sw = {}
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            mk = self.consts[cl].view(self.C_MASK)
            sw[cl] = fa_sweep(self.at(k, c), self._sweep_cfg(), self.sweep_bufs[cl], L1Tiles(self, cl),
                              {0: (self.q8[cl], [q8t[cl]])},
                             self.roles, name="s_", early=1, mask=lambda j, u, mk=mk: mk if j == self.LAST else None,
                             mask_c=self.mask_c, ot8=self.ot8_bits(), geom_core=self.sm_geom)
        # ---- o: every head's row into the root, its multicast, every cluster's copy -----------------------------------
        rcl = [cl for cl in range(self.N_CL) if self.chip(cl) == R]
        far = lambda k: None if k == R else R
        comp, puts = {}, {}
        for cl in range(self.N_CL):
            k, h = self.chip(cl), self.head[cl]
            comp[cl] = self.a_to_row(cl, "o2r", sw[cl]["xt"][0], self.oar[cl], HD, [sw[cl]["tr"][0]])
            puts[cl] = self.nd(cl, "dm", "o_put", "__snax_bingo_kernel_idma_put_flagged",
                              SnaxBingoKernelIdmaPutFlaggedArgs(self.oar[cl], self.l3("oland", h * self.OB, far(k)),
                                                                self.OB,
                                                                self.l3("oflag", h * 64, far(k)), value=1), [comp[cl]])
        ow = self.node(R, 0, dm, "o_wait", "__snax_bingo_kernel_wait_flags",
                       SnaxBingoKernelWaitFlagsArgs(self.l3("oflag"), d.NH, 1), [puts[cl] for cl in rcl])
        for cl in range(self.N_CL):
            if cl not in rcl:
                self.dfg.bingo_add_order_edge(puts[cl], ow)
        # one multicast source at a time: x's has ended once every cluster holds x, which every o row follows
        mc = self.node(R, 0, dm, "o_bput", "__snax_bingo_kernel_idma_put_flagged",
                       SnaxBingoKernelIdmaPutFlaggedArgs(self.l3("oland"), self.l3("oland"), d.NH * self.OB,
                                                         self.l3("omflag"), value=1, bcast=True), [ow, self.x_put])
        get = {}
        for cl in range(self.N_CL):
            if cl in rcl:
                after = [ow]
            else:
                w = self.nd(cl, "xdma", "om_wait", "__snax_bingo_kernel_wait_flags",
                            SnaxBingoKernelWaitFlagsArgs(self.l3("omflag"), 1, 1), [comp[cl]])
                self.dfg.bingo_add_order_edge(mc, w)
                after = [w]
            get[cl] = self.nd(cl, "xdma", "o_get", "__snax_bingo_kernel_xdma_1d_copy",
                             SnaxBingoKernelXdma1dCopyArgs(self.l3("oland"), self.ofull[cl], d.NH * self.OB), after)
        # ---- W_O by columns, y into the root's row, the root's write into HBM ------------------------------------------
        ylast = {}
        for cl in range(self.N_CL):
            gms = self.chunks(cl, "gwo", wo_js, self.ofull[cl], [self.ywo[cl].view(64 * s) for s in range(self.NWS)],
                              d.k_wo, [get[cl]], lds=wl[cl])
            ylast[cl] = gms[-1]
        yput = {}
        for cl in range(self.N_CL):
            k = self.chip(cl)
            yput[cl] = self.nd(cl, "dm", "y_put", "__snax_bingo_kernel_idma_put_flagged",
                              SnaxBingoKernelIdmaPutFlaggedArgs(self.ywo[cl], self.l3("yrow", 2 * self.WOC * cl,
                                                                                      far(k)),
                                                                2 * self.WOC, self.l3("yflag", cl * 64, far(k)),
                                                                value=1),
                              [ylast[cl]])
        yw = self.node(R, 0, dm, "y_wait", "__snax_bingo_kernel_wait_flags",
                       SnaxBingoKernelWaitFlagsArgs(self.l3("yflag"), self.N_CL, 1), [yput[cl] for cl in rcl])
        for cl in range(self.N_CL):
            if cl not in rcl:
                self.dfg.bingo_add_order_edge(yput[cl], yw)
        yh = self.y_out(yw)
        # ---- the append: each chip's halves of the new key and value into its cache in HBM, behind the y puts -------
        app = {}
        for k in self.chips:
            cl0, cl1 = self.clusters(k)
            hh = self.kv_half[k]
            base = self.h_kv[k].address + self.APP
            app[k] = [self.copy(cl0, "k_app", self.kbuf[cl0].view(L16 + hh * self.KRUN), base, self.KRUN, [kput[k]]),
                      self.copy(cl1, "v_app", self.vbuf[cl1].view(L16 + hh * self.HB), base + self.HB, self.HB,
                                [vput[k]])]
        self.end_window([self.probe(k, f"fin{k:02x}", app[k] + ([yh] if k == R else [])) for k in self.chips])
        self._link_report()
        self._checks()

    def _cache_copies(self, cl, tag, src, t0, half, deps):
        """A cache part's K and V halves `half` (src: [K halves | V halves], each tile's 4 KiB) into cluster cl's tiles
        t0 ..: the K half a 1 KiB run of every 16-key block, the V half its contiguous half of V^T. Returns both."""
        rk = self.PT * self.bc // BR
        ck = self.copy2d(cl, f"kc_{tag}{t0}", src, self.kbuf[cl].view(t0 * self.TILE_B + half * self.KRUN), self.KRUN,
                         self.KRUN, 2 * self.KRUN, rk, deps)
        cv = self.copy2d(cl, f"vc_{tag}{t0}", BingoMemSymbol(src.symbol_name, offset=src.offset + self.PT * self.HB),
                         self.vbuf[cl].view(t0 * self.TILE_B + half * self.HB), self.HB, self.HB, self.TILE_B, self.PT,
                         deps)
        return ck, cv

    def _link_report(self):
        """The bytes each memory link carries in the window: the weights and the cache halves of the chips behind it."""
        per = {}
        for k in self.chips:
            w = sum(n for cl in self.clusters(k) for _, n in self.chunk_off[cl])
            per.setdefault(self.link[k], [0, 0])
            per[self.link[k]][0] += w
            per[self.link[k]][1] += self.P * self.PB
        tot_w = sum(v[0] for v in per.values())
        tot_c = sum(v[1] for v in per.values())
        (m, side), (bw, bc) = max(per.items(), key=lambda kv: sum(kv[1]))
        print(f"[{self.TAG}] stream: weights {tot_w / 2**20:.2f} MiB, cache {tot_c / 2**20:.2f} MiB over {len(per)} memory "
              f"links; the busiest (memory chip {m:#04x}, side {side:+d}) {(bw + bc) / 2**20:.3f} MiB "
              f"({bw / 2**20:.3f} weights, {bc / 2**20:.3f} cache); each pair link one cache half "
              f"({self.P * self.PB / 2**10:.0f} KiB) each way")

    # ---- checks -------------------------------------------------------------------------------------------------------
    def _checks(self):
        """Byte for byte against LlamaDecodeData: y in the root memory chip's HBM (every column) and the appended halves
        of the new key and value in the HBM of the root's pair and of the last pair; with params checks stages first the
        root cluster 0's query tile, both root clusters' new key and value tiles, the root's mate's last cached key block
        and value tile (its half and the relayed one), and the root cluster 0's o8 row (a_row)."""
        d, R, lay, mesh = self.d, self.ROOT, self.lay, self.mesh
        L16 = self.LAST * self.TILE_B
        checks = []
        if self.checks_kind == "stages":
            m0, mate = self.group[R], self.kv_mate[R]
            cl0, cl1 = 0, 1
            gm = self.chips.index(mate) * self.NC + 1
            h0 = self.head[cl0]
            q = np.zeros((BR, self.HD), dtype=np.int8)
            q[0] = d.q8[h0][self.ctx][self.perm]
            kt16, vt16 = self._ktile(m0, self.LAST), self._vtile(m0, self.LAST)
            ktc, vtc = self._ktile(m0, self.CT - 1), self._vtile(m0, self.CT - 1)
            orow = lay.to_a(d.dec_ot8[h0], mesh).astype(np.int8).reshape(-1, 4 * BR)[:, :ATOM].reshape(-1)
            blk = BR * self.HD
            stage = [("q8", R, 0, self.q8[cl0], lay.to_a(q, mesh).astype(np.int8)),
                     ("k16_c0", R, 0, self.kbuf[cl0].view(L16), kt16[:blk]),
                     ("k16_c1", R, 1, self.kbuf[cl1].view(L16), kt16[:blk]),
                     ("v16_c0", R, 0, self.vbuf[cl0].view(L16), vt16),
                     ("v16_c1", R, 1, self.vbuf[cl1].view(L16), vt16),
                     ("kc_mate", mate, 1, self.kbuf[gm].view(self.CT * self.TILE_B - blk), ktc[-blk:]),
                     ("vc_mate", mate, 1, self.vbuf[gm].view((self.CT - 1) * self.TILE_B), vtc),
                     ("o8row", R, 0, self.oar[cl0], orow)]
            for nm, k, c, src, img in stage:
                img = np.ascontiguousarray(img).view(np.uint8)
                checks.append((nm, self.stash(nm, k, c, src, img.size, []), img, img.size))
        y = np.ascontiguousarray(d.y_raw.astype(np.float16)).view(np.uint8)
        if self.y_to_hbm:
            checks.append(("y_hbm", self.h_yhbm, y, y.size))
        else:                                           # y stays in the root's L3
            checks.append(("y_l3", self.l3("yrow", 0, R), y, y.size))
        for k in sorted({R, self.kv_mate[R], *self.pairs[-1]}):
            m, hh = self.group[k], self.kv_half[k]
            base = self.h_kv[k].address + self.APP
            kg = self._khalf(m, self.LAST, hh)[:self.KRUN].view(np.uint8)
            vg = self._vhalf(m, self.LAST, hh).view(np.uint8)
            checks.append((f"kapp_k{k:02x}", BingoMemFixedAddr(base, mem_level="HBM"), kg, kg.size))
            checks.append((f"vapp_k{k:02x}", BingoMemFixedAddr(base + self.HB, mem_level="HBM"), vg, vg.size))
        self.run_checks(checks)

    # ---- emission -------------------------------------------------------------------------------------------------------
    def finish(self, app_name, x_prefixes=("x_wait_",)):
        self.dfg.split_chip_functions = True
        cm.ChipletBuild.finish(self, app_name, x_prefixes)


def _app_name(S):
    return (f"Llama-3-8B attention decode (synthetic weights), one token over a {S.ctx}-position cache, "
            f"{len(S.chips)} chips x {S.NC} clusters")


def run():
    run_driver("llama3 attention decode workload generator", DBuild, _app_name)
