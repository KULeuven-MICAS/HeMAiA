# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Data of the llama3 workloads: one decoder layer of Llama-3-8B with SYNTHETIC weights of the model's own shapes.

THE MODEL is Meta-Llama-3-8B's configuration (config.json of the released checkpoint): hidden 4096, 32 query heads
and 8 key/value heads of 128 (grouped-query attention, 4 query heads a group), a SwiGLU FFN of 14336, RMSNorm with
eps 1e-5, RoPE with theta 500000. No checkpoint is read: every weight is a seeded normal draw (std INIT_STD, the
configuration's initializer range) and every RMSNorm gain a seeded draw around 1, so the shapes, the bytes streamed
and the work are the model's and the values only need to stay in the device's ranges.

RoPE runs on adjacent pairs of each head's dimensions (hwmodel.rope), as the device's elementwise stages do.
Llama's checkpoints rotate halves; the two are the same rotation after a fixed permutation of each head's columns
of W_q and W_k, which synthetic weights absorb.

THE FFN is registered with workloads/ffn/common/ffn_data.py (`register_ffn`), so the FFN driver runs it unchanged.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.abspath(__file__).split("/target/sw/host/")[0], "util", "sim", "llm"))
sys.path.insert(0, os.path.join(os.path.abspath(__file__).split("/target/sw/host/")[0], "util", "sim", "gemm"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "ffn", "common"))

import ffn_data  # noqa: E402
import gemv_utils  # noqa: E402
import model_data  # noqa: E402

# Meta-Llama-3-8B config.json
LLAMA3_8B = dict(hidden=4096, inter=14336, heads=32, kv_heads=8, head_dim=128, rms_eps=1e-5, rope_theta=500000.0,
                 layers=32)
INIT_STD = 0.02                 # the configuration's initializer_range
GAIN_STD = 0.05
MODEL = "llama3-8b-synthetic"


def _rng(seed, *tag):
    return np.random.default_rng([int(seed), 0x11a3, *[int(t) for t in tag]])


def ffn_weights(inter=None, seed=1):
    """The FFN's synthetic weights in load_weights' form: gate, up [hidden, inter], down [inter, hidden], gain."""
    H, I = LLAMA3_8B["hidden"], LLAMA3_8B["inter"]
    n = I if inter is None else int(inter)
    if not 0 < n <= I:
        raise ValueError(f"params inter={inter}: Llama-3-8B has {I} intermediate columns")
    r = _rng(seed, 2)
    draw = lambda shape: (r.standard_normal(shape, dtype=np.float32) * np.float32(INIT_STD))
    return dict(model=MODEL, repo="meta-llama/Meta-Llama-3-8B (shapes only, synthetic values)", revision="synthetic",
                layer=0, hidden=H, inter=n,
                gate=draw((H, n)), up=draw((H, n)), down=draw((n, H)), gain=ffn_gain(seed))


def ffn_gain(seed=1):
    """The post-attention RMSNorm's gain (float32 [hidden]): its own draw, whatever the FFN's size."""
    H = LLAMA3_8B["hidden"]
    return np.float32(1.0) + _rng(seed, 5).standard_normal(H, dtype=np.float32) * np.float32(GAIN_STD)


def exact_gemv(xq, Wq, k):
    """hwmodel.gemv's signature over gemv_utils.exact_gemv (float64 BLAS, the INT32 accumulation bit for bit)."""
    import fp
    return gemv_utils.exact_gemv(fp, xq, Wq, k)


def use_exact_gemv():
    """Point hwmodel.gemv (the golden's GEMM) at exact_gemv."""
    import hwmodel as hm
    hm.gemv = exact_gemv


def register_ffn(seed=1):
    """Make `model: llama3-8b-synthetic` an FFN the ffn driver can build."""
    def build(inter):
        use_exact_gemv()
        return ffn_weights(inter, seed=seed)
    ffn_data.SYNTHETIC[MODEL] = build


def register_ffn_shape(hidden, inter, seed=1):
    """Make `model: llama3-style-<hidden>x<inter>-synthetic` an FFN the ffn driver can build: Llama 3's SwiGLU FFN with
    another hidden and intermediate size (a sweep's fixed workload), its weights drawn as ffn_weights draws them and a
    post-attention gain of its own size. Returns the model name."""
    H, I = int(hidden), int(inter)
    name = f"llama3-style-{H}x{I}-synthetic"

    def build(n):
        n = I if n is None else int(n)
        if not 0 < n <= I:
            raise ValueError(f"params inter={n}: {name} has {I} intermediate columns")
        use_exact_gemv()
        r = _rng(seed, 2, H, I)
        draw = lambda shape: (r.standard_normal(shape, dtype=np.float32) * np.float32(INIT_STD))
        gain = np.float32(1.0) + _rng(seed, 5, H).standard_normal(H, dtype=np.float32) * np.float32(GAIN_STD)
        return dict(model=name, repo=f"Llama-3-style SwiGLU FFN {H} -> {I} (synthetic values)", revision="synthetic",
                    layer=0, hidden=H, inter=n, gate=draw((H, n)), up=draw((H, n)), down=draw((n, H)), gain=gain)
    ffn_data.SYNTHETIC[name] = build
    return name


# ---- the attention ---------------------------------------------------------------------------------------------------

def _mm(a, b):
    """The array's exact INT32 accumulation of integer operands, through float64 (every sum stays below 2^53)."""
    return np.rint(np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)).astype(np.int64)


class LlamaAttnData:
    """Grouped-query attention of one Llama-3-8B layer over a fresh prompt of `tokens` tokens: synthetic weights, the
    device's arithmetic stage by stage, and the golden of every stage the driver checks.

    THE PROMPT runs in passes of `rows` tokens. Pass i's queries (rows i .. i + rows - 1 of the prompt) attend to every
    key of passes 0 .. i, in tiles of `bc` keys, with an online softmax (hwmodel.mla_attention's arithmetic); the
    keys after a query (the causal mask) get -65504 added to their scores before the softmax, so their P is exactly 0.

    THE STAGES, per query head h and its key/value group m = h // 4:

        hq        = quant_i8(rmsnorm(x16) (.) gain, s_x)                     the normed input, INT8
        q16       = d_port(hq @ Wq[:, h], k_x) (.) q_f[h]                    dequantised per column
        k16, v16  = the same with Wk[:, m], Wv[:, m]
        q_rot     = rope(q16), k_rot = rope(k16)                             adjacent pairs, x cos + swap(x) sin_signed
        q8, k8    = quant_i8(q_rot, s_q), quant_i8(k_rot, s_k); v8 = quant_i8(v16, s_v)
        o, l      = the sweep of pass i over key tiles 0 .. (i + 1) rows / bc - 1   (O^T INT32, the row sums)
        ot16      = (d_port(o, k_o) (.) rsqrt(l a_n)^2)^T                     o~ = O / l, one query a row
        ot8       = quant_i8(ot16, s_ot)
        part[k]   = d_port([ot8 of chip k's two heads] @ Wo[their rows, :], k_wo)   a chip's partial over every output
        y_raw     = the partials folded in FP16 in the chain's order;  y16 = y_raw (.) wo_f

    Every static scale is the absolute maximum of the tensor it quantises over the whole prompt, onto 127.
    """

    def __init__(self, tokens=256, rows=64, bc=32, seed=1, wbits=4, mask_c=False):
        # mask_c: the causal mask as the score matmul's INT32 C (mask_bias where masked), added before the D port's
        # conversion: a masked score is RNE(S * 2^-k_s + MASKED_SCORE), one rounding, instead of the SIMD add's two
        self.mask_c = bool(mask_c)
        fp, hm, layout, pack, ref = model_data.import_golden()
        import simd
        self.fp, self.hm, self.layout, self.pack, self.ref, self.simd = fp, hm, layout, pack, ref, simd
        self.mesh = layout.MESH
        c = LLAMA3_8B
        self.H, self.NH, self.NKV, self.HD = c["hidden"], c["heads"], c["kv_heads"], c["head_dim"]
        self.GQ = self.NH // self.NKV                     # query heads a key/value group
        self.T, self.R, self.bc, self.wbits = int(tokens), int(rows), int(bc), int(wbits)
        T, R, bc, H, HD = self.T, self.R, self.bc, self.H, self.HD
        if T % R or R % 16 or R % bc or bc % 16:
            raise ValueError(f"tokens {T} rows {R} bc {bc}: passes of whole 16-row query tiles and whole key tiles")
        self.NP = T // R
        F16, F32 = np.float16, np.float32
        # ---- synthetic weights, quantised per output column -------------------------------------------------
        r = _rng(seed, 3)
        draw = lambda shape: r.standard_normal(shape, dtype=F32) * F32(INIT_STD)
        Wq, Wk, Wv = draw((H, self.NH * HD)), draw((H, self.NKV * HD)), draw((H, self.NKV * HD))
        Wo = draw((self.NH * HD, H))
        self.gain = F32(1.0) + r.standard_normal(H, dtype=F32) * F32(GAIN_STD)
        self._float = dict(wq=Wq, wk=Wk, wv=Wv, wo=Wo)
        qc = lambda W: pack.quantize_columns(W, wbits=wbits)
        (self.wq_q, self.wq_s), (self.wk_q, self.wk_s) = qc(Wq), qc(Wk)
        (self.wv_q, self.wv_s), (self.wo_q, self.wo_s) = qc(Wv), qc(Wo)
        # ---- the prompt: the residual stream entering the layer, and its normed INT8 ----------------------------
        self.x16 = _rng(seed, 4).standard_normal((T, H)).astype(F16)
        self.gain16 = self.gain.astype(F16)
        xg16 = fp.mul16(hm.rmsnorm(self.x16), self.gain16[None, :])
        QS = hm.QScale
        amax = lambda a: float(np.max(np.abs(np.asarray(a, dtype=np.float64))))
        self.s_x = QS.for_amax(amax(xg16))
        self.hq = fp.quant_i8(xg16, self.s_x.inv)                                   # [T, H] int8
        # ---- the D-port shifts: no INT8 GEMM of its depth overflows FP16 ------------------------------------------
        s = fp.d_shift_for_depth
        self.k_x, self.k_s, self.k_o, self.k_wo = s(H), s(HD), s(T), s(self.NH * HD)
        # ---- RoPE (theta of the model), cos repeated per pair and sin with the pair's sign ---------------------------
        inv = 1.0 / (c["rope_theta"] ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
        ang = np.arange(T, dtype=np.float64)[:, None] * inv[None, :]               # [T, HD / 2]
        cos, sin = np.cos(ang).astype(F16), np.sin(ang).astype(F16)
        self.cos_rep = np.repeat(cos, 2, axis=1)
        self.sin_sgn = np.empty((T, HD), dtype=F16)
        self.sin_sgn[:, 0::2], self.sin_sgn[:, 1::2] = -sin, sin
        # ---- the projections ---------------------------------------------------------------------------------------
        dq = lambda ws, cols, k: hm.dequant_scale(self.s_x.s, ws[cols], k)
        self.q_f = [dq(self.wq_s, slice(h * HD, (h + 1) * HD), self.k_x) for h in range(self.NH)]
        self.k_f = [dq(self.wk_s, slice(m * HD, (m + 1) * HD), self.k_x) for m in range(self.NKV)]
        self.v_f = [dq(self.wv_s, slice(m * HD, (m + 1) * HD), self.k_x) for m in range(self.NKV)]
        proj = lambda wq, f, cols: fp.mul16(fp.d_port(_mm(self.hq, wq[:, cols]), self.k_x), f[None, :])
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
        # ---- the masks the sweep adds: per (key tile, query tile) of a pass, the pattern's index or None ------------
        self.mask_tiles = self._mask_tiles()
        # ---- the sweep, the normalisation and o8 ------------------------------------------------------------------
        self.ot16 = [self.head_attention(h) for h in range(self.NH)]
        self.s_ot = QS.for_amax(max(amax(o) for o in self.ot16))
        self.ot8 = [fp.quant_i8(o, self.s_ot.inv) for o in self.ot16]
        self.wo_f = hm.dequant_scale(self.s_ot.s, self.wo_s, self.k_wo)          # FP16 [H]
        self.y_ref = self._float_attention()

    # ---- operators ---------------------------------------------------------------------------------------------------
    def rope(self, x16, r0=0):
        """RoPE of FP16 rows r0 .. at their positions: x cos + swap(x) sin_signed, one rounding a stage."""
        fp = self.fp
        n = x16.shape[0]
        xs = np.empty_like(x16)
        xs[:, 0::2], xs[:, 1::2] = x16[:, 1::2], x16[:, 0::2]
        return fp.add16(fp.mul16(x16, self.cos_rep[r0:r0 + n]), fp.mul16(xs, self.sin_sgn[r0:r0 + n]))

    @property
    def mask_bias(self):
        """(mask_c) The INT32 bias of a masked score: MASKED_SCORE at the D port's scale 2^k_s."""
        return int(round(self.hm.MASKED_SCORE * 2.0 ** self.k_s))

    def mask_bias_patterns(self):
        """(mask_c) mask_patterns as the score matmul's C: INT32 [bc, 16], 0 or mask_bias."""
        return [np.where(m != 0, self.mask_bias, 0).astype(np.int32) for m in self.mask_patterns]

    def _mask_tiles(self):
        """The patterns of a pass's masked (key tile, query tile) pairs, FP16 [bc, 16] each (0 or -65504, S^T: a key
        a row, a query a lane), and per pair (t, u) -- t the key tile's index from the pass's first, u the query tile's
        in the pass -- the pattern index or None. A pair is masked where some key is after some query; the earlier key
        tiles of a pass have none."""
        R, bc = self.R, self.bc
        pats, which = [], {}
        for t in range(R // bc):
            for u in range(R // 16):
                keys = t * bc + np.arange(bc)[:, None]               # key offsets in the pass
                qs = u * 16 + np.arange(16)[None, :]
                m = np.where(keys > qs, np.float16(self.hm.MASKED_SCORE), np.float16(0)).astype(np.float16)
                if not np.any(m):
                    which[(t, u)] = None
                    continue
                for i, p in enumerate(pats):
                    if np.array_equal(p, m):
                        which[(t, u)] = i
                        break
                else:
                    which[(t, u)] = len(pats)
                    pats.append(m)
        self.mask_patterns = pats
        return which

    def head_attention(self, h):
        """o~ of query head h for every token, FP16 [T, HD]: each pass's sweep, then O / l."""
        fp, hm, simd = self.fp, self.hm, self.simd
        m = h // self.GQ
        out = []
        for i in range(self.NP):
            o, l16 = self.sweep(h, i)
            O16 = fp.d_port(o, self.k_o)                                          # [HD, R]
            t16 = simd.stream_map(l16, self.a_n, 0.0, simd.RSQRT)
            out.append(np.ascontiguousarray(fp.mul16(O16, fp.mul16(t16, t16)[None, :]).T))
        return np.concatenate(out)

    def sweep(self, h, i):
        """Pass i of head h: the online softmax over key tiles 0 .. (i + 1) R / bc - 1 (O^T int64 [HD, R], l FP16 [R])."""
        fp, hm, simd = self.fp, self.hm, self.simd
        F16, F32, F64 = np.float16, np.float32, np.float64
        R, bc = self.R, self.bc
        m = h // self.GQ
        q = self.q8[h][i * R:(i + 1) * R].astype(F64).T                             # [HD, R]
        K8, V8 = self.k8[m].astype(F64), self.v8[m].astype(F64)
        a32 = F32(self.a_exp)
        m16 = np.full(R, -65504.0, dtype=F16)
        l16 = np.zeros(R, dtype=F16)
        o = None
        nt = (i + 1) * R // bc
        first = i * R // bc                                                         # the pass's own first key tile
        for j in range(nt):
            S = np.rint(K8[j * bc:(j + 1) * bc] @ q).astype(np.int64)              # [bc, R], S^T
            S16 = fp.d_port(S, self.k_s)
            if not np.all(np.isfinite(S16)):
                raise FloatingPointError("a score overflows FP16 after the D-port shift")
            if j >= first:
                for u in range(R // 16):
                    pi = self.mask_tiles[(j - first, u)]
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
            p8 = fp.quant_i8(p16, hm.P8_SCALE).astype(F64)                         # [bc, R]
            pv = np.rint(V8[j * bc:(j + 1) * bc].T @ p8).astype(np.int64)          # [HD, R]
            if j == 0:
                o = pv
            else:
                o = np.clip(np.rint(o.astype(F64) * corr16.astype(F64)[None, :]), -2**31, 2**31 - 1).astype(np.int64) + pv
            m16 = mnew
        return o, l16

    # ---- the output projection and the fold -----------------------------------------------------------------------
    def part(self, heads):
        """A chip's partial over every output: its heads' o8 side by side (in `heads` order) through their rows of Wo,
        off the D port, FP16 [T, H]."""
        a = np.concatenate([self.ot8[h] for h in heads], axis=1)
        rows = np.concatenate([np.arange(h * self.HD, (h + 1) * self.HD) for h in heads])
        return self.fp.d_port(_mm(a, self.wo_q[rows, :]), self.k_wo)

    def fold(self, parts_in_order):
        """The partials added in FP16 in the chain's order: y_raw FP16 [T, H]."""
        y = parts_in_order[0]
        for p in parts_in_order[1:]:
            y = self.fp.add16(y, p)
        return y

    def y16(self, y_raw):
        return self.fp.mul16(y_raw, self.wo_f[None, :])

    def _float_attention(self):
        """The layer's attention in float64 on the same prompt and weights: RMSNorm, the projections, RoPE on adjacent
        pairs, causal softmax, Wo."""
        F64 = np.float64
        W = self._float
        c = LLAMA3_8B
        x = self.x16.astype(F64)
        xg = x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + c["rms_eps"]) * self.gain.astype(F64)[None, :]
        cos, sin = self.cos_rep.astype(F64), self.sin_sgn.astype(F64)

        def rot(v):
            vs = np.empty_like(v)
            vs[:, 0::2], vs[:, 1::2] = v[:, 1::2], v[:, 0::2]
            return v * cos + vs * sin
        HD, T = self.HD, self.T
        outs = []
        mask = np.triu(np.full((T, T), -np.inf), 1)
        for h in range(self.NH):
            m = h // self.GQ
            q = rot(xg @ W["wq"][:, h * HD:(h + 1) * HD].astype(F64))
            k = rot(xg @ W["wk"][:, m * HD:(m + 1) * HD].astype(F64))
            v = xg @ W["wv"][:, m * HD:(m + 1) * HD].astype(F64)
            s = q @ k.T / np.sqrt(HD) + mask
            p = np.exp(s - s.max(axis=1, keepdims=True))
            outs.append((p / p.sum(axis=1, keepdims=True)) @ v)
        return np.concatenate(outs, axis=1) @ W["wo"].astype(F64)

    def report(self, y16):
        ref = np.asarray(self.y_ref, dtype=np.float64).reshape(-1)
        got = np.asarray(y16, dtype=np.float64).reshape(-1)
        cos = float(got @ ref / max(np.linalg.norm(got) * np.linalg.norm(ref), 1e-30))
        rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30))
        sc = lambda q: f"{q.s:.4g}"
        return [f"{MODEL}: attention of one layer, {self.T} tokens (a fresh prompt, causal), {self.NH} query / {self.NKV} "
                f"key-value heads of {self.HD}, INT{self.wbits} weights per column; passes of {self.R}, key tiles of {self.bc}",
                f"D-port shifts k_x={self.k_x} k_s={self.k_s} k_o={self.k_o} k_wo={self.k_wo}; scales s_x={sc(self.s_x)} "
                f"s_q={sc(self.s_q)} s_k={sc(self.s_k)} s_v={sc(self.s_v)} s_ot={sc(self.s_ot)}; "
                f"{len(self.mask_patterns)} mask patterns",
                f"device y vs the float attention: cosine {cos:.5f}, relative L2 error {rel:.4f}"]
