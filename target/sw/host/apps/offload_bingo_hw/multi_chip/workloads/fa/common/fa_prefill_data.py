# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Data of the fa prefill workloads: DeepSeek-V2-Lite's layer-1 MLA attention for a prompt of T tokens,
from the checkpoint, to the device's arithmetic. Two cases:

    cached (default)   the last T positions of a real 8K-token cache: the queries attend to every cached key,
                       the causal block among their own keys left out; the cache is an input
    fresh              a fresh prompt (no cache): the first T positions of the same sequence, causal, and the
                       keys are the prompt's own, made from its hidden states as the device makes them

THE INPUTS are real: sweep/llm_blocks/fa/real_kv_cache.py's cache (per token the normalised latent
c_unit [512] and the rotated k_pe [64], and the hidden state x1 entering layer 1), and the
checkpoint's layer-1 q_proj, kv_b_proj, o_proj and the two norm gains (sweep/common/weights); fresh
adds kv_a_proj_with_mqa (W_KVA, 2048 -> 576). A fresh prompt's float cache is the file's first T rows:
the model is causal, so a token's latent does not depend on what follows it.

THE WEIGHTS are quantised with util/sim/common/pack.py per output column, INT8 or (wbits 4) INT4 in the paired
layout the ring streams (pack._blob: B layout in blocks of 32 columns, nibble-packed). The
input_layernorm gain is NOT folded into W_Q's rows (a real gain spans a wide range, so folded, a few
rows would set every column's scale): the host applies it to the normed input before the input is
quantised, as the FFN's data does. The kv_a_layernorm gain is folded as the dsv2 golden folds it,
into W_UK's columns and W_UV's rows.

THE GOLDEN follows the mapping (fa_prefill_mc): head h on chip h, a cluster's rows = tokens. Per
head, all T tokens at once (every stage is per row, and the attention per query lane):

    hq        = quant_i8(rmsnorm(x16) (.) gain, s_x)                       staged on the root
    q         = d_port(hq @ Wq[:, head h], k_x) (.) s_q                     192 columns
    q_pe      = RoPE at the token's position                                three elementwise stages
    q~        = d_port(quant_i8(q_nope, s_qn) @ W_UK,h, k_uk) (.) s_uk      512 columns
    Q8        = [quant_i8(q~, s_qt) | quant_i8(q_pe, s_qpe)]                576 columns
    O^T, l    = the online softmax over key tiles of Bc, per lane           hwmodel.mla_attention
                (fresh: a query block of `chunk` / 2 rows sweeps the key tiles up to its chunk's end, and
                MASKED_SCORE is added to a key after a query, tile_plan)
    o~        = d_port(O, k_o) (.) c / l                                    per token
    o_h       = d_port(quant_i8(o~, s_ot) @ W_UV,h, k_uv) (.) s_uv
    part_h    = d_port(quant_i8(o_h, s_o) @ W_O[head h rows, :], k_wo)      FP16, 2048 columns
    y_raw     = part summed over the heads in the fold order (FP16 adds)

An FP16 sum depends on its order, so `reduce` takes the order the device folds in. The dequantisation
of the output (s_o s_wo 2^k_wo per column) is applied after the reduce, as the FFN's.

THE KEYS of a fresh prompt (fa_stages_mc, stages 1 and 2, per 16-token tile):
    kv        = d_port(hq @ W_KVA, k_x) (.) f_kva                           576 columns
    c~        = rmsnorm(kv[:, :512])                                        the gain folded into W_UK, W_UV
    k_pe      = RoPE(kv[:, 512:]) at the token's position
    c8, kpe8  = quant_i8(c~, s_c), quant_i8(k_pe, s_kpe)                    the rows [c8 | kpe8]
f_kva's latent columns carry a power of two that brings the latent to rms ~1: RMSNorm ignores a scale,
and the SIMD's FP16 sum of squares over 512 columns must not overflow.
"""

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.abspath(__file__).split("/target/sw/host/")[0], "util", "sim", "llm"))
from model_data import import_golden, read_bf16  # noqa: E402

D_QK, D_V, KV_RANK, ROPE, Q_NOPE, V_HEAD = 576, 512, 512, 64, 128, 128
Q_HEAD = Q_NOPE + ROPE
HIDDEN = 2048
LAYER = 1


class FaPrefillData:
    def __init__(self, kv_path, weights_cache, tokens=256, keys=8192, bc=256, n_calib=32,
                 model="dsv2-lite-bf16", threads=None, fresh=False, chunk=None, rope_perm=False, mask_c=False,
                 wbits=8, skip_masked=False):
        fp, hm, layout, pack, ref = import_golden()
        import dsv2_model as mdl
        import simd
        self.fp, self.hm, self.layout, self.pack, self.ref, self.simd, self.mdl = \
            fp, hm, layout, pack, ref, simd, mdl
        self.mesh = layout.MESH
        self.skip_masked = skip_masked
        d = self.dims = mdl.load()
        self.H, self.T, self.keys, self.bc = d.heads, int(tokens), int(keys), int(bc)
        self.fresh = bool(fresh)
        # rope_perm: every rotary part's 64 columns as the pairs' first elements, then their second (0, 2, .., 62,
        # 1, 3, .., 63), in W_q's and W_KVA's columns and the RoPE tables alike, so a pair's partner sits 32 columns
        # away instead of next to it. q_pe and k_pe share the order, so their dot product, and every score, keeps
        # its value; the swap RoPE needs moves whole halves instead of adjacent FP16 values (fa_prefill_mc)
        self.rope_perm = bool(rope_perm)
        if self.rope_perm and not self.fresh:
            raise ValueError("rope_perm: needs fresh (a cache holds its keys' rotary part in pair order)")
        self.rp = (np.concatenate([np.arange(0, ROPE, 2), np.arange(1, ROPE, 2)]) if self.rope_perm
                   else np.arange(ROPE))
        # mask_c: the causal mask as the score matmul's INT32 C (mask_bias where masked, mask_bias_patterns), added
        # before the D port's conversion: a masked score is RNE(S * 2^-k_s + MASKED_SCORE), one rounding, where the
        # SIMD add makes it RNE(RNE(S * 2^-k_s) + MASKED_SCORE). Either is P = 0 after the softmax's exponential
        self.mask_c = bool(mask_c)
        if self.mask_c and not self.fresh:
            raise ValueError("mask_c: needs fresh (a cache's sweep has no causal mask)")
        if keys % bc or bc % 16:
            raise ValueError(f"{keys} keys are not whole tiles of {bc} (a multiple of 16)")
        if self.fresh:
            # a query block is a cluster's rows of a chunk; both blocks of a chunk sweep every tile up to its end, so a
            # tile may straddle the two blocks (its keys past a block's rows are masked)
            self.C = int(chunk or 0)
            if self.keys != self.T or not self.C or self.T % self.C or self.C % bc:
                raise ValueError(f"fresh: keys {keys} must be the {self.T} tokens, chunk {chunk} a divisor of them "
                                 f"that is whole tiles of {bc}")
        z = np.load(kv_path)
        if z["c_unit"].shape[0] < keys or self.T > keys:
            raise ValueError(f"{kv_path}: {z['c_unit'].shape[0]} tokens; need {keys} keys and "
                             f"{self.T} query tokens")
        self.c_unit = z["c_unit"][:keys].astype(np.float32)
        self.kpe = z["kpe"][:keys].astype(np.float32)
        self.pos = np.arange(0, self.T) if self.fresh else np.arange(keys - self.T, keys)
        x1 = z["x1"][self.pos].astype(np.float32)
        # ---- the checkpoint's layer-1 attention ----------------------------------------------
        with open(os.path.join(weights_cache, "manifest.json")) as f:
            m = json.load(f)[model]
        t = m["tensors"]
        base = os.path.join(weights_cache, m["repo"].replace("/", "__"))
        pre = f"model.layers.{LAYER}."
        rd = lambda n: read_bf16(os.path.join(base, pre + n + ".bin"), t[pre + n]["shape"])
        self.in_norm = rd("input_layernorm.weight")
        self.kv_norm = rd("self_attn.kv_a_layernorm.weight")
        wq = rd("self_attn.q_proj.weight").T                          # [2048, 3072]
        wkvb = rd("self_attn.kv_b_proj.weight").T                     # [512, 4096]
        wo = rd("self_attn.o_proj.weight").T                          # [2048 (heads x 128), 2048]
        H = self.H
        self.w_uk = [wkvb[:, h * (Q_NOPE + V_HEAD):h * (Q_NOPE + V_HEAD) + Q_NOPE] for h in range(H)]
        self.w_uv = [wkvb[:, h * (Q_NOPE + V_HEAD) + Q_NOPE:(h + 1) * (Q_NOPE + V_HEAD)]
                     for h in range(H)]
        wq_dev = wq.copy() if self.rope_perm else wq
        if self.rope_perm:
            for h in range(H):
                pe = h * Q_HEAD + Q_NOPE
                wq_dev[:, pe:pe + ROPE] = wq[:, pe + self.rp]
        if wbits not in (8, 4):
            raise ValueError(f"wbits {wbits}: the weights are INT8 or INT4")
        self.wbits = wbits
        self.wq = pack.pack("wq", wq_dev, wbits=wbits)
        self.wuk = pack.pack_heads("wuk", [w.T for w in self.w_uk], col_gain=self.kv_norm, wbits=wbits)
        self.wuv = pack.pack_heads("wuv", self.w_uv, row_gain=self.kv_norm, wbits=wbits)
        self.wo = pack.pack("wo", wo, wbits=wbits)
        self._float = dict(wq=wq, wo=wo)
        if self.fresh:
            wkva = rd("self_attn.kv_a_proj_with_mqa.weight").T              # [2048, 576]
            wkva_dev = wkva.copy() if self.rope_perm else wkva
            if self.rope_perm:
                wkva_dev[:, KV_RANK:] = wkva[:, KV_RANK + self.rp]
            self.wkva = pack.pack("wkva", wkva_dev, wbits=wbits)
            self._float["wkva"] = wkva
        # ---- the D-port shifts: no INT8 GEMM of the depth can overflow after them ---------------
        s = fp.d_shift_for_depth
        self.k_x, self.k_uk, self.k_uv, self.k_s = s(HIDDEN), s(Q_NOPE), s(KV_RANK), s(D_QK)
        self.k_o = s(keys)                                            # the last PV, depth = keys
        self.k_wo = s(H * V_HEAD)                                     # a head's partial: the sum of 16 fits
        # ---- scales: the calibration tokens' absolute maxima onto 127 --------------------------
        self._calibrate(x1, n_calib)
        # ---- the device's arithmetic -----------------------------------------------------------
        self.hq = np.stack([self._input(x) for x in x1])              # [T, 2048] int8
        self._cos_sin()
        if self.fresh:
            self._device_kv()
            self._tile_plans()
        self._heads = {}
        self.x1 = x1
        # a head's chain takes about five seconds: kept on disk, keyed by everything it depends on
        st = os.stat(kv_path)
        with open(os.path.abspath(__file__), "rb") as f:
            src = f.read()
        self._cache_dir = os.path.join(os.path.dirname(os.path.abspath(kv_path)), "fa_prefill_golden")
        self._cache_key = hashlib.sha1(repr((st.st_size, int(st.st_mtime), model, self.T, self.keys, self.bc,
                                             int(n_calib), self.fresh, getattr(self, "C", None), self.rope_perm,
                                             self.mask_c, self.wbits,
                                             hashlib.sha1(src).hexdigest()) + ((1,) if self.skip_masked else ())
                                             ).encode()
                                       ).hexdigest()[:16]

    # ---- the input and the float reference ------------------------------------------------------
    def _gained(self, x):
        """The normed, gained input as the host produces it (the device's RMSNorm model)."""
        hm, fp = self.hm, self.fp
        return fp.mul16(hm.rmsnorm(np.asarray(x, dtype=np.float16)), self.in_norm.astype(np.float16))

    def _input(self, x):
        return self.fp.quant_i8(self._gained(x), self.s_x.inv)

    def _float_query(self, x1, pos):
        """q~ (with the latent gain), the rotated q_pe, q_nope of one token, float, all heads."""
        d, ref, mdl = self.dims, self.ref, self.mdl
        xg = ref.rmsnorm(x1, self.in_norm, d.rms_eps)
        q = (xg @ self._float["wq"]).astype(np.float32).reshape(self.H, Q_HEAD)
        cos, sin = mdl.rope_cos_sin(d, pos)
        qpe = mdl.rope_interleaved(q[:, Q_NOPE:].astype(np.float64), cos, sin).astype(np.float32)
        qn = q[:, :Q_NOPE]
        qlat = np.stack([qn[h] @ self.w_uk[h].T for h in range(self.H)]).astype(np.float32)
        return qlat * self.kv_norm[None, :], qpe, qn, xg

    def _calibrate(self, x1, n):
        hm, fp, d = self.hm, self.fp, self.dims
        QS = hm.QScale
        amax = lambda a: float(np.max(np.abs(np.asarray(a, dtype=np.float64))))
        s_c, s_kpe = QS.for_amax(amax(self.c_unit)), QS.for_amax(amax(self.kpe))
        a_xg = a_qn = a_qt = a_qpe = a_ot = a_o = 0.0
        self.y_ref = None
        for i in range(max(self.T - int(n), 0), self.T):
            qt, qpe, qn, _ = self._float_query(x1[i], int(self.pos[i]))
            a_xg = max(a_xg, amax(self._gained(x1[i])))
            a_qn, a_qt, a_qpe = max(a_qn, amax(qn)), max(a_qt, amax(qt)), max(a_qpe, amax(qpe))
            kk = int(self.pos[i]) + 1 if self.fresh else self.keys        # fresh: causal, the keys up to the token
            sc = (qt @ self.c_unit[:kk].T + qpe @ self.kpe[:kk].T) * d.softmax_scale     # [H, keys]
            p = self.ref.softmax(sc)
            ot = (p @ self.c_unit[:kk]).astype(np.float32)                    # [H, 512] unit latent
            o = np.stack([(ot[h] * self.kv_norm) @ self.w_uv[h] for h in range(self.H)])
            a_ot, a_o = max(a_ot, amax(ot)), max(a_o, amax(o))
            if i == self.T - 1:
                self.y_ref = (o.reshape(-1) @ self._float["wo"]).astype(np.float32)
        prod = max(a_qt * s_c.s, a_qpe * s_kpe.s) / 127.0
        self.s_c, self.s_kpe = s_c, s_kpe
        self.s_qt = QS(float(np.float32(s_c.s / prod)))
        self.s_qpe = QS(float(np.float32(s_kpe.s / prod)))
        self.s_x, self.s_qn = QS.for_amax(a_xg), QS.for_amax(a_qn)
        self.s_ot, self.s_o = QS.for_amax(a_ot), QS.for_amax(a_o)
        self.a_exp = float(np.float32(d.softmax_scale * self.s_qt.s * s_c.s * 2.0 ** self.k_s))
        self.a_n = float(np.float32(hm.P8_SCALE / (2.0 ** self.k_o * s_c.s)))
        self.a_n_bits = fp.f32bits(self.a_n)
        # the cache as the device holds it: [keys, 576] rows = [c8 | kpe8], and the value copy's c8
        self.c8 = fp.quant_i8(self.c_unit.astype(np.float16), s_c.inv)
        self.kpe8 = fp.quant_i8(self.kpe.astype(np.float16), s_kpe.inv)
        self.rows = np.concatenate([self.c8, self.kpe8], axis=1).astype(np.int8)

    def _cos_sin(self):
        """Per token: cos repeated per pair and sin with the pair's sign folded in, FP16 [T, 64]."""
        F16 = np.float16
        cs = [self.mdl.rope_cos_sin(self.dims, int(p)) for p in self.pos]
        cos = np.stack([c for c, _ in cs]).astype(F16)
        sin = np.stack([s for _, s in cs]).astype(F16)
        self.cos_rep = np.repeat(cos, 2, axis=1)
        self.sin_sgn = np.empty((self.T, ROPE), dtype=F16)
        self.sin_sgn[:, 0::2] = -sin
        self.sin_sgn[:, 1::2] = sin
        if self.rope_perm:
            self.cos_rep, self.sin_sgn = self.cos_rep[:, self.rp], self.sin_sgn[:, self.rp]

    # ---- the device's arithmetic, one head at a time -------------------------------------------
    @staticmethod
    def _mm(a, b):
        """The array's exact INT32 accumulation of int8 operands, through float64 (exact: the sums
        stay below 2^53)."""
        return np.rint(np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)).astype(np.int64)

    def rope(self, x16):
        """RoPE of [T, 64] FP16 rows at their positions: x cos + swap(x) sin_signed, one rounding a stage. The swap
        exchanges each pair's two values: neighbours, or under rope_perm the two halves."""
        fp = self.fp
        xs = np.empty_like(x16)
        if self.rope_perm:
            h = ROPE // 2
            xs[:, :h], xs[:, h:] = x16[:, h:], x16[:, :h]
        else:
            xs[:, 0::2], xs[:, 1::2] = x16[:, 1::2], x16[:, 0::2]
        return fp.add16(fp.mul16(x16, self.cos_rep), fp.mul16(xs, self.sin_sgn))

    # ---- a fresh prompt's keys and its causal sweep -------------------------------------------------
    def kva_scale(self):
        """Dequantisation of W_KVA's 576 columns, FP16; the latent's 512 carry c_fold."""
        s = np.asarray(self.hm.dequant_scale(self.s_x.s, self.wkva.s, self.k_x), dtype=np.float64)
        s[:KV_RANK] *= self.c_fold
        return s.astype(np.float16)

    def _device_kv(self):
        """The prompt's keys as stage_kv makes them (THE KEYS): kv16, c_unit16, kpe16, and the rows c8 | kpe8."""
        fp, hm = self.fp, self.hm
        raw = fp.d_port(self._mm(self.hq, self.wkva.q), self.k_x)
        lat = np.asarray(fp.mul16(raw[:, :KV_RANK], np.asarray(
            hm.dequant_scale(self.s_x.s, self.wkva.s[:KV_RANK], self.k_x), dtype=np.float16)[None, :]), dtype=np.float64)
        rms = float(np.max(np.sqrt(np.mean(lat ** 2, axis=1))))
        self.c_fold = float(2.0 ** -np.ceil(np.log2(rms))) if rms > 0 else 1.0
        self.kv16 = fp.mul16(raw, self.kva_scale()[None, :])
        if not np.all(np.isfinite(self.kv16)):
            raise FloatingPointError("the KV projection overflows FP16")
        self.c_unit16 = hm.rmsnorm(self.kv16[:, :KV_RANK])
        if not np.all(np.isfinite(self.c_unit16)):
            raise FloatingPointError("the latent's RMSNorm overflows FP16")
        self.kpe16 = self.rope(np.ascontiguousarray(self.kv16[:, KV_RANK:]))
        self.c8 = fp.quant_i8(self.c_unit16, self.s_c.inv)
        self.kpe8 = fp.quant_i8(self.kpe16, self.s_kpe.inv)
        self.rows = np.concatenate([self.c8, self.kpe8], axis=1).astype(np.int8)

    def _tile_plans(self):
        """Per query block (r0 a multiple of chunk / 2, a cluster's rows of a chunk): the key tiles it sweeps -- every
        tile up to its chunk's end, so a chunk's two blocks sweep the same tiles; skip_masked: every tile up to the
        block's own last row, so no tile is wholly masked -- and its masked (tile j, query tile u) pairs, each a
        pattern of mask_patterns: FP16 [bc, 16], 0 or MASKED_SCORE where the key comes after the query (S^T: a key a
        row, a query a lane)."""
        R, bc = self.C // 2, self.bc
        self.mask_patterns, self._plans = [], {}
        for r0 in range(0, self.T, R):
            nt = -(-(r0 + R) // bc) if self.skip_masked else (r0 // self.C + 1) * self.C // bc
            plan = {}
            for j in range(nt):
                for u in range(R // 16):
                    keys = j * bc + np.arange(bc)[:, None]
                    qs = r0 + 16 * u + np.arange(16)[None, :]
                    m = np.where(keys > qs, np.float16(self.hm.MASKED_SCORE), np.float16(0)).astype(np.float16)
                    if not np.any(m):
                        continue
                    for i, p in enumerate(self.mask_patterns):
                        if np.array_equal(p, m):
                            plan[(j, u)] = i
                            break
                    else:
                        plan[(j, u)] = len(self.mask_patterns)
                        self.mask_patterns.append(m)
            self._plans[r0] = (nt, plan)

    @property
    def mask_bias(self):
        """(mask_c) The INT32 bias of a masked score: MASKED_SCORE at the D port's scale 2^k_s."""
        return int(round(self.hm.MASKED_SCORE * 2.0 ** self.k_s))

    def mask_bias_patterns(self):
        """(mask_c) mask_patterns as the score matmul's C: INT32 [bc, 16], 0 or mask_bias."""
        return [np.where(m != 0, self.mask_bias, 0).astype(np.int32) for m in self.mask_patterns]

    def tile_plan(self, r0):
        """(fresh) (key tiles, {(j, u): mask pattern}) of the query block whose first row is r0."""
        return self._plans[r0]

    def wblob(self, q):
        """Weight columns as the ring streams them: B layout (INT8), or wbits 4 the paired INT4 layout."""
        return self.pack._blob(q, self.wbits, self.mesh).astype(np.int8)

    def kva_chunks(self):
        """W_KVA as the ring streams it (fresh): blocks of 64 columns."""
        return [self.wblob(self.wkva.q[:, 64 * i:64 * (i + 1)]) for i in range(D_QK // 64)]

    def kva_cols(self, c0, c1):
        """W_KVA's columns c0 .. c1 - 1 as one ring chunk, B layout (fresh: a chip's slice)."""
        return self.wblob(self.wkva.q[:, c0:c1])

    def kv_piece(self, t0):
        """Tokens t0 .. t0 + 15 as stage_kv multicasts them: their key rows (the A layout of [16, 576], one 16-row
        block of the key tiles), then their value columns (the A layout of c8^T [512, 16])."""
        lay, mesh = self.layout, self.mesh
        k = lay.to_a(self.rows[t0:t0 + 16], mesh)
        v = lay.to_a(np.ascontiguousarray(self.c8[t0:t0 + 16].T), mesh)
        return np.concatenate([k, v]).astype(np.int8)

    def attention(self, q8):
        """hwmodel.mla_attention for the lanes of q8 [L, 576] (one query a lane, independent) over
        every key, in tiles of bc: (O^T int64 [512, L], l FP16 [L]). Fresh: lane t is token t, and each
        query block sweeps its tile_plan."""
        if not self.fresh:
            return self._sweep(q8, self.keys // self.bc, {})
        R, L = self.C // 2, q8.shape[0]
        o = np.zeros((D_V, L), dtype=np.int64)
        l16 = np.zeros(L, dtype=np.float16)
        for r0 in range(0, L, R):
            nt, plan = self.tile_plan(r0)
            o[:, r0:r0 + R], l16[r0:r0 + R] = self._sweep(q8[r0:r0 + R], nt, plan)
        return o, l16

    def _sweep(self, q8, nt, plan):
        """The online softmax of the lanes of q8 over key tiles 0 .. nt - 1, plan's masks added to the scores."""
        hm, fp, simd = self.hm, self.fp, self.simd
        F16, F32, F64 = np.float16, np.float32, np.float64
        bc, L = self.bc, q8.shape[0]
        a32 = F32(self.a_exp)
        Q = q8.astype(F64).T                                           # [576, L]
        K8, V8 = self.rows.astype(F64), self.c8.astype(F64)
        m16 = np.full(L, -65504.0, dtype=F16)
        l16 = np.zeros(L, dtype=F16)
        o = None
        for j in range(nt):
            S = np.rint(K8[j * bc:(j + 1) * bc] @ Q).astype(np.int64)  # [bc, L], S^T
            S16 = fp.d_port(S, self.k_s)
            if not np.all(np.isfinite(S16)):
                raise FloatingPointError("a score overflows FP16 after the D-port shift")
            for u in range(L // 16):
                pi = plan.get((j, u))
                if pi is None:
                    continue
                if self.mask_c:
                    bias = np.where(self.mask_patterns[pi] != 0, self.mask_bias, 0)
                    S16[:, 16 * u:16 * (u + 1)] = fp.d_port(S[:, 16 * u:16 * (u + 1)] + bias, self.k_s)
                else:
                    S16[:, 16 * u:16 * (u + 1)] = fp.add16(S16[:, 16 * u:16 * (u + 1)], self.mask_patterns[pi])
            rmax = simd.reduce_lanewise(S16, simd.MAX)
            mnew = simd.reduce_lanewise(np.stack([rmax, m16]), simd.MAX)
            corr16 = hm.exp16(fp.add16(m16, -mnew), a32)
            p16 = hm.exp16(fp.add16(S16, -mnew[None, :]), a32)
            rsum16 = simd.reduce_lanewise(p16, simd.ADD)
            l16 = simd.reduce_lanewise(np.stack([rsum16, fp.mul16(corr16, l16)]), simd.ADD)
            p8 = fp.quant_i8(p16, hm.P8_SCALE).astype(F64)             # [bc, L]
            pv = np.rint(V8[j * bc:(j + 1) * bc].T @ p8).astype(np.int64)   # [512, L]
            if j == 0:
                o = pv
            else:
                o = np.clip(np.rint(o.astype(F64) * corr16.astype(F64)[None, :]),
                            -2**31, 2**31 - 1).astype(np.int64) + pv
            m16 = mnew
        return o, l16

    def head(self, h):
        """Head h's chain for every token (cached): dict of q8 [T, 576], ot16 [T, 512], o8 [T, 128],
        part [T, 2048] and the stages between."""
        if h in self._heads:
            return self._heads[h]
        path = os.path.join(self._cache_dir, f"{self._cache_key}_h{h:02d}.npz")
        if os.path.exists(path):
            try:
                with np.load(path) as z:
                    self._heads[h] = {k: z[k] for k in z.files}
                return self._heads[h]
            except (OSError, ValueError):
                pass
        hm, fp, simd = self.hm, self.fp, self.simd
        F16 = np.float16
        r = {}
        cols = slice(h * Q_HEAD, (h + 1) * Q_HEAD)
        q_raw = fp.d_port(self._mm(self.hq, self.wq.q[:, cols]), self.k_x)
        q16 = fp.mul16(q_raw, self.q_scale(h)[None, :])
        qpe_rot = self.rope(np.ascontiguousarray(q16[:, Q_NOPE:]))
        qn8 = fp.quant_i8(q16[:, :Q_NOPE], self.s_qn.inv)
        qt_raw = fp.d_port(self._mm(qn8, self.wuk.q[h]), self.k_uk)
        qt16 = fp.mul16(qt_raw, self.qt_scale(h)[None, :])
        q8 = np.concatenate([fp.quant_i8(qt16, self.s_qt.inv), fp.quant_i8(qpe_rot, self.s_qpe.inv)],
                            axis=1)
        o, l16 = self.attention(q8)
        O16 = fp.d_port(o, self.k_o)                                    # [512, T]
        t16 = simd.stream_map(l16, self.a_n, 0.0, simd.RSQRT)
        rsc16 = fp.mul16(t16, t16)
        ot16 = np.ascontiguousarray(fp.mul16(O16, rsc16[None, :]).T)    # [T, 512]
        ot8 = fp.quant_i8(ot16, self.s_ot.inv)
        oh_raw = fp.d_port(self._mm(ot8, self.wuv.q[h]), self.k_uv)
        o16 = fp.mul16(oh_raw, self.uv_scale(h)[None, :])
        o8 = fp.quant_i8(o16, self.s_o.inv)
        part = fp.d_port(self._mm(o8, self.wo.q[h * V_HEAD:(h + 1) * V_HEAD, :]), self.k_wo)
        r.update(q_raw=q_raw, q16=q16, qpe_rot=qpe_rot, qn8=qn8, qt_raw=qt_raw, qt16=qt16, q8=q8,
                 l=l16, O16=O16, ot16=ot16, ot8=ot8, oh_raw=oh_raw, o16=o16, o8=o8, part=part)
        self._heads[h] = r
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            np.savez(path + f".{os.getpid()}.tmp.npz", **r)
            os.replace(path + f".{os.getpid()}.tmp.npz", path)
        except OSError:
            pass
        return r

    # ---- the per-column factors and the constants the device holds -----------------------------
    def q_scale(self, h):
        """Dequantisation of head h's q, FP16 [192]."""
        return self.hm.dequant_scale(self.s_x.s, self.wq.s[h * Q_HEAD:(h + 1) * Q_HEAD], self.k_x)

    def qt_scale(self, h):
        return self.hm.dequant_scale(self.s_qn.s, self.wuk.s[h], self.k_uk)          # [512]

    def uv_scale(self, h):
        return self.hm.dequant_scale(self.s_ot.s, self.wuv.s[h], self.k_uv)          # [128]

    def out_scale(self):
        """Dequantisation of the reduced output, FP16 [2048]."""
        return self.hm.dequant_scale(self.s_o.s, self.wo.s, self.k_wo)

    def scale_tile(self, vec):
        """One row tile of a per-column factor in the D layout: `vec` repeated on 16 rows."""
        return self.layout.to_d(np.tile(np.asarray(vec, dtype=np.float16), (16, 1)), self.mesh)

    def scale_tile_a(self, vec, rows):
        """A per-column factor in the A layout of `rows` rows (a multiple of 16): `vec` repeated on every row, as
        the D -> A reshape leaves the values it scales (fa_prefill_mc: a dequantisation folded into its quantise)."""
        return self.layout.to_a(np.tile(np.asarray(vec, dtype=np.float16), (rows, 1)), self.mesh)

    def rope_tables(self, r0, rows):
        """cos and sin_signed of tokens r0 .. r0 + rows - 1 in the D layout (FP16 [rows, 64])."""
        lay = self.layout
        return (lay.to_d(self.cos_rep[r0:r0 + rows], self.mesh),
                lay.to_d(self.sin_sgn[r0:r0 + rows], self.mesh))

    # ---- the device layouts --------------------------------------------------------------------
    def x_chunk_a(self, r0, rows):
        """Rows r0 .. r0 + rows - 1 of the quantised input in the A layout (INT8)."""
        return self.layout.to_a(self.hq[r0:r0 + rows], self.mesh).astype(np.int8)

    def weight_chunks(self, h, wo_block=None):
        """Head h's weights as the ring streams them, in order: [(name, int8 bytes)]: W_q in three
        blocks of 64 columns, W_UK, W_UV, W_O in two blocks of 1024 columns, B layout. wo_block q (W_O by
        columns): W_O's columns 128 q .. 128 q + 127 over every head's rows, in two blocks of 64."""
        b = self.wblob
        wq = self.wq.q[:, h * Q_HEAD:(h + 1) * Q_HEAD]
        out = [(f"wq{i}", b(wq[:, 64 * i:64 * (i + 1)])) for i in range(3)]
        out.append(("wuk", b(self.wuk.q[h])))
        out.append(("wuv", b(self.wuv.q[h])))
        if wo_block is None:
            wo = self.wo.q[h * V_HEAD:(h + 1) * V_HEAD, :]
            out += [(f"wo{i}", b(wo[:, 1024 * i:1024 * (i + 1)])) for i in range(2)]
        else:
            c0 = 128 * wo_block
            out += [(f"wo{i}", b(self.wo.q[:, c0 + 64 * i:c0 + 64 * (i + 1)])) for i in range(2)]
        return out

    def cache_tiles(self):
        """[(K tile, V tile)] of the cache as the sweep reads them in L1: K the key copy's rows
        (A layout, bc x 576), V the value copy's columns (V^T in the A layout, 512 x bc)."""
        lay, bc, cap = self.layout, self.bc, self.keys
        key = lay.key_copy(self.rows, cap, self.mesh).astype(np.int8).reshape(-1)
        val = lay.value_copy(self.c8, cap, self.mesh).astype(np.int8).reshape(-1)
        kb = bc * D_QK
        out = []
        for j in range(cap // bc):
            k = key[j * kb:(j + 1) * kb]
            v = np.concatenate([val[r * 16 * cap + j * 16 * bc: r * 16 * cap + (j + 1) * 16 * bc]
                                for r in range(D_V // 16)])
            if k.size != kb or v.size != D_V * bc:
                raise ValueError(f"tile {j}: {k.size} / {v.size} bytes")
            out.append((k, v))
        return out

    def q8_a(self, h, r0):
        """Head h's Q8 of tokens r0 .. r0 + 15 in the A layout: q~ (128 blocks of 4 columns), then
        the rotated q_pe (16 blocks), each block 16 rows x 4 bytes, as the score matmul's B operand."""
        lay = self.layout
        q8 = self.head(h)["q8"][r0:r0 + 16]
        return np.concatenate([lay.to_a(q8[:, :KV_RANK], self.mesh),
                               lay.to_a(q8[:, KV_RANK:], self.mesh)]).astype(np.int8)

    # ---- the reduce ---------------------------------------------------------------------------
    def reduce(self, heads_in_order, r0, rows):
        """Rows r0 .. r0 + rows - 1 of the partials added in FP16 in the given order of heads:
        the device's fold. FP16 [rows, 2048]."""
        add16 = self.fp.add16
        acc = self.head(heads_in_order[0])["part"][r0:r0 + rows]
        for h in heads_in_order[1:]:
            acc = add16(acc, self.head(h)["part"][r0:r0 + rows])
        return acc

    def y_cols(self, r0, rows):
        """Rows r0 .. r0 + rows - 1 of y as W_O by columns makes it: every head's o8 side by side through all of W_O,
        one INT32 sum over the full depth, off the D port. FP16 [rows, 2048]."""
        o8 = np.concatenate([self.head(h)["o8"][r0:r0 + rows] for h in range(self.H)], axis=1)
        return self.fp.d_port(self._mm(o8, self.wo.q), self.k_wo)

    def y16(self, y_raw):
        return self.fp.mul16(y_raw, self.out_scale()[None, :])

    def report(self, heads_in_order):
        """How far the device's last token is from the float attention of the checkpoint."""
        y = self.y16(self.reduce(heads_in_order, self.T - 1, 1))[0].astype(np.float64)
        ref = np.asarray(self.y_ref, dtype=np.float64)
        cos = float(y @ ref / max(np.linalg.norm(y) * np.linalg.norm(ref), 1e-30))
        rel = float(np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-30))
        sc = lambda q: f"{q.s:.4g}"
        if self.fresh:
            a, b = np.asarray(self.c_unit16, dtype=np.float64).ravel(), self.c_unit.astype(np.float64).ravel()
            ccos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
            what = (f"a fresh prompt of {self.T} tokens (causal, no cache): the keys made on the device (W_KVA, "
                    f"RMSNorm, RoPE; latent vs the model's cosine {ccos:.5f}, c_fold {self.c_fold:g}), "
                    f"{len(self.mask_patterns)} mask patterns, {self.H} heads, INT{self.wbits} per column")
        else:
            what = (f"the last {self.T} of {self.keys} cached tokens against all {self.keys} keys, {self.H} heads, "
                    f"INT{self.wbits} per column")
        return [f"DeepSeek-V2-Lite layer {LAYER} MLA attention, {what}",
                f"D-port shifts k_x={self.k_x} k_uk={self.k_uk} k_uv={self.k_uv} k_s={self.k_s} "
                f"k_o={self.k_o} k_wo={self.k_wo}; scales s_x={sc(self.s_x)} s_qn={sc(self.s_qn)} "
                f"s_qt={sc(self.s_qt)} s_qpe={sc(self.s_qpe)} s_c={sc(self.s_c)} s_ot={sc(self.s_ot)} "
                f"s_o={sc(self.s_o)}",
                f"device y of the last token vs the float attention: cosine {cos:.5f}, "
                f"relative L2 error {rel:.4f}"]
