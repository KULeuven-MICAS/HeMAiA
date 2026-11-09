# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Data of the Llama-3-8B decode attention (workloads/fa/llama3_decode, driver fa_llama_mc.py): the grouped-query
attention of one decode token over a KV cache of `keys` positions, with SYNTHETIC INT4 weights.

THE WEIGHTS AND THE PROMPT are llama3_data's (workloads/llama3/common): the same seeded draw of W_q, W_k, W_v (and W_o,
drawn for the gain that follows it), the input RMSNorm's gain, and the prompt's residual stream x16 (a standard normal
row a position), so the cache is the one the Llama blocks' prefill of that prompt leaves.

THE CACHE, position by position as llama3_data.LlamaAttnData computes it: hq = quant_i8(rmsnorm(x16) (.) gain, s_x);
k16 = d_port(hq @ Wk_int4[:, group]) (.) its column factors, RoPE (adjacent pairs, theta 500000) at the position;
v16 the same with Wv and no RoPE; K and V INT8 at one static scale each (the absolute maximum over every position and
group onto 127).

THE QUERY is position keys - 1's (its own key is the cache's last): its normed row through Wq_int4, RoPE at its
position, INT8 at s_q, calibrated on the last n_calib positions' queries.

THE HALVES. The softmax takes one query a FP16 lane of a SIMD beat (BR = 16) and a key tile is shared by the four query
heads of its group, so the 32 heads run as two halves of 16 lanes: half h holds query heads 16 h .. 16 h + 15 and the
key / value groups 4 h .. 4 h + 3; lane i is head 16 h + i of group 4 h + i // 4.
    K tile    the half's four groups' keys side by side, [BC, 512] (the cache's own row layout), A layout
    query     block diagonal, [16, 512]: lane i is its head's q8 on its group's 128 columns and zero on the others,
              so S^T = K Q^T scores every lane against its own group's keys only
    V tile    the half's four groups' V^T stacked, [512, BC], A layout
    PV        leaves [512, 16] FP16; lane i's O^T is rows 128 (i // 4) .. + 127 of its column. The compaction keeps
              those: [128, 16] a half, lane i head 16 h + i
The block-diagonal operands spend four times the multiplies a group needs; at one decode token the array is idle anyway.

THE GOLDEN is the device's arithmetic (util/sim/common/hwmodel.py) for the key split fa_llama_mc.py builds, half by half, as
fa_data.FaData.golden: every cluster's shard of key tiles; pass 1 the shard's row maximum; m* the lanewise maximum
(exact in any order); pass 2 the softmax seeded at m* and PV, O^T leaving the D port as FP16 at k_o; a cluster's partial
[O_0 | O_1 | l_0 | l_1] (the compacted halves, then the two row-sum beats) added in FP16 in the reduce's order; per half
o~ = O * rsqrt(l a_n)^2; xt [32, 128] one head a row.

golden_v covers the other mappings fa_llama_mc.py builds (params passes, stats): the softmax statistics (m, l) of
every cluster folded by the softmax-statistics junction (ssj_hop, SoftmaxStatsJunction's arithmetic bit for bit, as
host_kernel_lib.h repeats it) into (m*, l*), the root normalising with l*; and the one-pass split, every cluster's
online softmax and PV over its own tiles, its O re-based onto m* by one more softmax step whose only scores are m*
(corr = exp(a (m_c - m*))) and the PV's column scale.
"""

import os
import re
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "llama3", "common"))

import llama3_data  # noqa: E402  (puts util/sim/llm, util/sim/gemm and ffn/common on sys.path)
import model_data  # noqa: E402

# BC keys a tile; BR query lanes (one SIMD beat of FP16, fa_llama_mc.py checks it against the cluster); HD a head's
# dimensions; GPH key/value groups a half; D a half's K / V row: its groups' columns side by side
BC, BR = 64, 16
CFG = llama3_data.LLAMA3_8B
HD, NH, NKV = CFG["head_dim"], CFG["heads"], CFG["kv_heads"]
GQ = NH // NKV
HALVES = NH // BR
GPH = NKV // HALVES
D = GPH * HD
CACHE_VERSION = 2
# the softmax-statistics junction's group: a key beat m[0..31] and a value beat l[0..31], FP16; here the two halves'
# 16 lanes side by side
SSJ_ROWS = 32
HOST_LIB = os.path.join(HERE, "..", "..", "..", "..", "..", "..", "runtime", "host_kernel_lib.h")


# ---- the softmax-statistics junction, bit for bit (host_kernel_lib.h bingo_host_ssj_hop) -----------------------------
def _f32_bits(f):
    return int(np.asarray(f, dtype=np.float32).view(np.uint32))


def _f32_of(u):
    return np.asarray(u & 0xFFFFFFFF, dtype=np.uint32).view(np.float32)[()]


def _f32_round(q):
    """The FP32 nearest the rational q, ties to even (an FMA's one rounding)."""
    if q == 0:
        return np.float32(0.0)
    a = -q if q < 0 else q
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    quantum = Fraction(2) ** (max(e, -126) - 23)
    m = a / quantum
    n = m.numerator // m.denominator
    r = m - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n % 2):
        n += 1
    v = float(n * quantum)
    return np.float32(-v if q < 0 else v)


def _fma32(a, b, c):
    return _f32_round(Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c)))


def _a_wins(a, b):
    """FpHelpers.fp32aWins: a >= b on FP32 bit patterns, a tie keeps a."""
    sa, sb = a >> 31, b >> 31
    if sa != sb:
        return not sa
    return b >= a if sa else a >= b


_SSJ_LUT = []


def _ssj_lut():
    if not _SSJ_LUT:
        txt = open(HOST_LIB).read()
        body = re.search(r"bingo_host_ssj_exp_lut\[128\]\s*=\s*\{(.*?)\};", txt, re.S).group(1)
        _SSJ_LUT.extend(int(x, 16) for x in re.findall(r"0x([0-9A-Fa-f]{8})u", body))
        if len(_SSJ_LUT) != 128:
            raise ValueError(f"{HOST_LIB}: {len(_SSJ_LUT)} entries in the junction's exponential table")
    return _SSJ_LUT


def ssj_exp(x):
    """FpActivation's EXP at the junction's table depth (bingo_host_ssj_exp)."""
    hi_e, lo_e, magic = 0x42B0C0A5, 0xC2B0C0A5, 0x4B400000
    u = _f32_bits(x)
    if _a_wins(u, hi_e):
        u = hi_e
    if not _a_wins(u, lo_e):
        u = lo_e
    s0 = _fma32(_f32_of(u), _f32_of(0x4338AA3B), np.float32(0.0))
    rm = np.float32(s0 + _f32_of(magic))
    rnd = np.float32(rm + _f32_of(magic | 0x80000000))
    frac = np.float32(s0 + _f32_of(_f32_bits(rnd) ^ 0x80000000))
    im = _f32_bits(rm) - magic
    interp = _fma32(frac, _f32_of(0x3BB17218), _f32_of(0x3F800000))
    r3 = np.float32(_f32_of(_ssj_lut()[im & 127]) * interp)
    ne = im >> 7
    pow2 = 0 if ne < -127 else ((ne + 127) & 0xFF) << 23
    return np.float32(r3 * _f32_of(pow2))


def ssj_hop(ma, la, mb, lb, scale):
    """One junction hop on one row, FP16 bit patterns in and out, (ma, la) operand A -> (m*, l*)."""
    f16 = lambda b: np.float32(np.uint16(b).view(np.float16))      # noqa: E731
    ka, kb = _f32_bits(f16(ma)), _f32_bits(f16(mb))
    aw = _a_wins(ka, kb)
    d = _f32_bits(np.float32(_f32_of(kb if aw else ka) + _f32_of((ka if aw else kb) ^ 0x80000000)))
    if (d & 0x7F800000) == 0x7F800000 and d & 0x7FFFFF:
        d = 0
    alpha = ssj_exp(np.float32(_f32_of(d) * np.float32(scale)))
    los, win = f16(lb if aw else la), f16(la if aw else lb)
    lo = win if _f32_bits(los) == 0 else _fma32(los, alpha, win)
    return (ma if aw else mb), int(np.float32(lo).astype(np.float16).view(np.uint16))


def ssj_fold(groups, scale):
    """Groups (FP16 [2 * SSJ_ROWS]: m then l) folded in order, each later one into the running result (operand A):
    a chain in data order ending at its collector, or the host fold's part order."""
    acc = np.asarray(groups[0], np.float16).view(np.uint16).copy()
    for g in groups[1:]:
        b = np.asarray(g, np.float16).view(np.uint16)
        for r in range(SSJ_ROWS):
            acc[r], acc[SSJ_ROWS + r] = ssj_hop(int(acc[r]), int(acc[SSJ_ROWS + r]), int(b[r]),
                                                int(b[SSJ_ROWS + r]), scale)
    return acc.view(np.float16)


class LlamaFaData:
    def __init__(self, cache_dir, keys=8192, seed=1, wbits=4, n_calib=64):
        fp, hm, layout, pack, ref = model_data.import_golden()
        import simd
        self.fp, self.hm, self.layout, self.simd = fp, hm, layout, simd
        if keys % BC:
            raise ValueError(f"{keys} keys are not whole tiles of {BC}")
        if BR % GQ or GQ * GPH != BR:
            raise ValueError(f"{BR} lanes do not hold {GPH} groups of {GQ} heads")
        self.keys, self.seed, self.wbits = int(keys), int(seed), int(wbits)
        path = os.path.join(cache_dir, f"llama3_synth_kv_{keys}_s{seed}_w{wbits}_c{n_calib}_v{CACHE_VERSION}.npz")
        if not os.path.exists(path):
            os.makedirs(cache_dir, exist_ok=True)
            tmp = path + f".tmp{os.getpid()}.npz"
            np.savez(tmp, **self._make(pack, n_calib))
            os.replace(tmp, path)
        z = np.load(path)
        self.k8, self.v8, self.q8 = z["k8"], z["v8"], z["q8"]           # [NKV, keys, HD], [NKV, keys, HD], [NH, HD]
        self.s_q, self.s_k, self.s_v = (hm.QScale(float(z[n])) for n in ("inv_q", "inv_k", "inv_v"))
        self.o_ref = z["o_ref"]                                          # [NH, HD] float32
        F32 = np.float32
        self.k_s = fp.d_shift_for_depth(HD)
        self.k_o = fp.d_shift_for_depth(keys)
        self.a_exp = float(F32((1.0 / np.sqrt(HD)) * self.s_q.s * self.s_k.s * 2.0 ** self.k_s))
        self.a_n = float(F32(hm.P8_SCALE / (2.0 ** self.k_o * self.s_v.s)))
        self.a_n_bits = fp.f32bits(self.a_n)
        # every half's rows [keys, D] (its groups side by side) and block-diagonal query [BR, D]
        self.rows = [np.concatenate([self.k8[GPH * h + j] for j in range(GPH)], axis=1) for h in range(HALVES)]
        self.vrows = [np.concatenate([self.v8[GPH * h + j] for j in range(GPH)], axis=1) for h in range(HALVES)]
        self.Q8 = []
        for h in range(HALVES):
            q = np.zeros((BR, D), dtype=np.int8)
            for i in range(BR):
                j = i // GQ
                q[i, j * HD:(j + 1) * HD] = self.q8[BR * h + i]
            self.Q8.append(q)

    # ---- the cache, the query and the float reference (made once, kept in cache_dir) -------------------------------
    def _make(self, pack, n_calib):
        fp, hm = self.fp, self.hm
        T, H = self.keys, CFG["hidden"]
        F16, F32, F64 = np.float16, np.float32, np.float64
        r = llama3_data._rng(self.seed, 3)
        draw = lambda shape: r.standard_normal(shape, dtype=F32) * F32(llama3_data.INIT_STD)  # noqa: E731
        Wq, Wk, Wv = draw((H, NH * HD)), draw((H, NKV * HD)), draw((H, NKV * HD))
        draw((NH * HD, H))                                                # W_o: drawn so the gain is the blocks'
        gain = F32(1.0) + r.standard_normal(H, dtype=F32) * F32(llama3_data.GAIN_STD)
        qc = lambda W: pack.quantize_columns(W, wbits=self.wbits)         # noqa: E731
        (wq_q, wq_s), (wk_q, wk_s), (wv_q, wv_s) = qc(Wq), qc(Wk), qc(Wv)
        x16 = llama3_data._rng(self.seed, 4).standard_normal((T, H)).astype(F16)
        xg16 = fp.mul16(hm.rmsnorm(x16), gain.astype(F16)[None, :])
        amax = lambda a: float(np.max(np.abs(np.asarray(a, dtype=F64))))  # noqa: E731
        s_x = hm.QScale.for_amax(amax(xg16))
        hq = fp.quant_i8(xg16, s_x.inv)
        k_x = fp.d_shift_for_depth(H)
        # RoPE tables: cos repeated per pair, sin with the pair's sign (llama3_data.LlamaAttnData)
        inv = 1.0 / (CFG["rope_theta"] ** (np.arange(0, HD, 2, dtype=F64) / HD))
        ang = np.arange(T, dtype=F64)[:, None] * inv[None, :]
        cos_rep = np.repeat(np.cos(ang).astype(F16), 2, axis=1)
        sin_sgn = np.empty((T, HD), dtype=F16)
        sin_sgn[:, 0::2], sin_sgn[:, 1::2] = -np.sin(ang).astype(F16), np.sin(ang).astype(F16)

        def rope(x, r0):
            n = x.shape[0]
            xs = np.empty_like(x)
            xs[:, 0::2], xs[:, 1::2] = x[:, 1::2], x[:, 0::2]
            return fp.add16(fp.mul16(x, cos_rep[r0:r0 + n]), fp.mul16(xs, sin_sgn[r0:r0 + n]))

        def proj(a, wq, ws, cols):
            f = hm.dequant_scale(s_x.s, ws[cols], k_x)
            return fp.mul16(fp.d_port(llama3_data._mm(a, wq[:, cols]), k_x), f[None, :])

        hc = lambda i: slice(i * HD, (i + 1) * HD)                       # noqa: E731
        k_rot = [rope(proj(hq, wk_q, wk_s, hc(m)), 0) for m in range(NKV)]
        v16 = [proj(hq, wv_q, wv_s, hc(m)) for m in range(NKV)]
        c0 = T - n_calib
        q_rot = [rope(proj(hq[c0:], wq_q, wq_s, hc(h)), c0) for h in range(NH)]
        s_k = hm.QScale.for_amax(max(amax(k) for k in k_rot))
        s_v = hm.QScale.for_amax(max(amax(v) for v in v16))
        s_q = hm.QScale.for_amax(max(amax(q) for q in q_rot))
        k8 = np.stack([fp.quant_i8(k, s_k.inv) for k in k_rot])
        v8 = np.stack([fp.quant_i8(v, s_v.inv) for v in v16])
        q8 = np.stack([fp.quant_i8(q[-1:], s_q.inv)[0] for q in q_rot])
        # the float reference: the same token's attention over the float cache (float weights, float RoPE)
        x = x16.astype(F64)
        xg = x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + CFG["rms_eps"]) * gain.astype(F64)[None, :]
        cs, sn = cos_rep.astype(F64), sin_sgn.astype(F64)

        def rot(v, r0):
            vs = np.empty_like(v)
            vs[:, 0::2], vs[:, 1::2] = v[:, 1::2], v[:, 0::2]
            return v * cs[r0:r0 + v.shape[0]] + vs * sn[r0:r0 + v.shape[0]]
        o_ref = np.zeros((NH, HD), dtype=F32)
        for m in range(NKV):
            kf = rot(xg @ Wk[:, hc(m)].astype(F64), 0)
            vf = xg @ Wv[:, hc(m)].astype(F64)
            for h in range(GQ * m, GQ * (m + 1)):
                qf = rot(xg[-1:] @ Wq[:, hc(h)].astype(F64), T - 1)[0]
                s = kf @ qf / np.sqrt(HD)
                p = np.exp(s - s.max())
                o_ref[h] = (p / p.sum()) @ vf
        return dict(k8=k8, v8=v8, q8=q8, inv_q=s_q.inv, inv_k=s_k.inv, inv_v=s_v.inv, o_ref=o_ref)

    # ---- the device layouts -----------------------------------------------------------------------------------------
    def q8_image(self):
        """The query as the attention reads it: per half the block-diagonal [BR, D] in the A layout (QK's B)."""
        return np.concatenate([self.layout.to_a(q, self.layout.MESH) for q in self.Q8]).astype(np.int8)

    def tiles(self, t0, nt):
        """Tiles t0 .. t0 + nt - 1 per half: (K tile [BC, D] A layout, V tile = V^T [D, BC] A layout), bytes."""
        lay, cap = self.layout, self.keys
        out = []
        for h in range(HALVES):
            key = lay.key_copy(self.rows[h], cap, lay.MESH).astype(np.int8).reshape(-1)
            val = lay.value_copy(self.vrows[h], cap, lay.MESH).astype(np.int8).reshape(-1)
            kb, vb = BC * D, D * BC
            th = []
            for j in range(t0, t0 + nt):
                k = key[j * kb:(j + 1) * kb]
                v = np.concatenate([val[r * 16 * cap + j * 16 * BC: r * 16 * cap + (j + 1) * 16 * BC]
                                    for r in range(D // 16)])
                if k.size != kb or v.size != vb:
                    raise ValueError(f"half {h} tile {j}: {k.size} / {v.size} bytes")
                th.append((k, v))
            out.append(th)
        return out                                                       # [half][tile] -> (k, v)

    # ---- the golden -------------------------------------------------------------------------------------------------
    def _tile(self, h, j):
        K = self.rows[h][j * BC:(j + 1) * BC].astype(np.int64)
        V = self.vrows[h][j * BC:(j + 1) * BC].astype(np.int64)
        return self.hm.d_port(K @ self.Q8[h].astype(np.int64).T, self.k_s), V

    @staticmethod
    def compact(o16):
        """[D, BR] FP16 -> [HD, BR]: lane i's rows of its group."""
        out = np.empty((HD, BR), dtype=np.float16)
        for i in range(BR):
            j = i // GQ
            out[:, i] = o16[j * HD:(j + 1) * HD, i]
        return out

    def golden(self, shards, chips, order):
        """shards {g: (t0, nt)}, chips {chip: [g, ...]} (cluster 0 first), order: the chips in fold order. -> per half
        the clusters' maxima, m* (packed [m*_0 | m*_1]), every cluster's partial and each chip's sum (flat FP16
        [O_0 | O_1 | l_0 | l_1]), the folded partial, xt [NH, HD]."""
        hm, fp, simd = self.hm, self.fp, self.hm.simd
        F16, F32, F64 = np.float16, np.float32, np.float64
        lane_max = lambda a: simd.reduce_lanewise(a, simd.MAX)          # noqa: E731
        seed = np.full(BR, -65504.0, dtype=F16)
        a32 = F32(self.a_exp)
        mstar, part = [], {g: [[], []] for g in shards}
        for h in range(HALVES):
            m_c = {}
            for g, (t0, nt) in shards.items():
                m16 = seed
                for j in range(t0, t0 + nt):
                    m16 = lane_max(np.stack([lane_max(self._tile(h, j)[0]), m16]))
                m_c[g] = m16
            ms = lane_max(np.stack([lane_max(np.stack(list(m_c.values()))), seed]))
            mstar.append(ms)
            for g, (t0, nt) in shards.items():
                m16, l16, o = ms, np.zeros(BR, dtype=F16), None
                for j in range(t0, t0 + nt):
                    S16, V = self._tile(h, j)
                    mnew = lane_max(np.stack([lane_max(S16), m16]))
                    corr16 = hm.exp16(fp.add16(m16, -mnew), a32)
                    p16 = hm.exp16(fp.add16(S16, -mnew[None, :]), a32)
                    rsum16 = simd.reduce_lanewise(p16, simd.ADD)
                    l16 = simd.reduce_lanewise(np.stack([rsum16, fp.mul16(corr16, l16)]), simd.ADD)
                    pv = V.T @ fp.quant_i8(p16, hm.P8_SCALE).astype(np.int64)
                    o = pv if o is None else np.clip(
                        np.rint(o.astype(F64) * corr16.astype(F64)[None, :]),
                        -2**31, 2**31 - 1).astype(np.int64) + pv
                    m16 = mnew
                part[g][0].append(self.compact(hm.d_port(o, self.k_o)))
                part[g][1].append(l16)
        flat = {g: np.concatenate([o.reshape(-1) for o in os_] + list(ls)).astype(F16)
                for g, (os_, ls) in part.items()}
        loc = {}
        for k, gs in chips.items():
            acc = flat[gs[0]]
            for g in gs[1:]:
                acc = fp.add16(acc, flat[g])
            loc[k] = acc
        tot = loc[order[0]]
        for k in order[1:]:
            tot = fp.add16(tot, loc[k])
        n = HD * BR
        xt = np.empty((NH, HD), dtype=F16)
        for h in range(HALVES):
            O = tot[h * n:(h + 1) * n].reshape(HD, BR)
            lt = tot[HALVES * n + h * BR:HALVES * n + (h + 1) * BR]
            t16 = simd.stream_map(lt, self.a_n, 0.0, simd.RSQRT)
            xt[h * BR:(h + 1) * BR] = fp.mul16(O, fp.mul16(t16, t16)[None, :]).T
        return dict(mstar=np.concatenate(mstar).astype(F16), flat=flat, loc=loc, tot=tot, xt=xt)

    def golden_v(self, shards, chips, order, passes=2, merge=False, fix_rows=4):
        """golden for the other mappings. passes 2: pass 1's online softmax leaves every cluster's (m, l); pass 2 is
        golden's. passes 1: one online softmax and PV over the shard; O re-based onto m* by a softmax step over
        `fix_rows` scores all m* (corr = exp(a (m_c - m*))) and a PV with C = O scaled by corr and zero values.
        merge: each cluster's group [m_0 | m_1 | l_0 | l_1] folded by the junction, its chip's two clusters (cluster
        1's arriving into cluster 0's), then the chips in `order` (the chain's data order, the root last); the
        partials are O alone and the root normalises with l*. Without merge m* is the lanewise maximum and the
        partials carry the row sums (pass 2's, or the one pass's re-based: l_c (.) corr). -> ml (the folded group)
        or mstar, flat, loc, tot, xt."""
        hm, fp, simd = self.hm, self.fp, self.hm.simd
        F16, F32, F64 = np.float16, np.float32, np.float64
        lane_max = lambda a: simd.reduce_lanewise(a, simd.MAX)          # noqa: E731
        seed = np.full(BR, -65504.0, dtype=F16)
        a32 = F32(self.a_exp)

        def sweep(h, t0, nt, m16, pv_on):
            l16, o = np.zeros(BR, dtype=F16), None
            for j in range(t0, t0 + nt):
                S16, V = self._tile(h, j)
                mnew = lane_max(np.stack([lane_max(S16), m16]))
                corr16 = hm.exp16(fp.add16(m16, -mnew), a32)
                p16 = hm.exp16(fp.add16(S16, -mnew[None, :]), a32)
                rsum16 = simd.reduce_lanewise(p16, simd.ADD)
                l16 = simd.reduce_lanewise(np.stack([rsum16, fp.mul16(corr16, l16)]), simd.ADD)
                if pv_on:
                    pv = V.T @ fp.quant_i8(p16, hm.P8_SCALE).astype(np.int64)
                    o = pv if o is None else self._colscale(o, corr16) + pv
                m16 = mnew
            return m16, l16, o

        st = {(g, h): sweep(h, t0, nt, seed, passes == 1) for g, (t0, nt) in shards.items() for h in range(HALVES)}
        out = {}
        if merge:
            grp = {g: np.concatenate([st[g, h][0] for h in range(HALVES)] + [st[g, h][1] for h in range(HALVES)])
                   for g in shards}
            chip = {k: ssj_fold([grp[g] for g in reversed(gs)], a32) for k, gs in chips.items()}
            ml = ssj_fold([chip[k] for k in order], a32)
            out["ml"] = ml
            mstar = [ml[h * BR:(h + 1) * BR] for h in range(HALVES)]
            lstar = [ml[SSJ_ROWS + h * BR:SSJ_ROWS + (h + 1) * BR] for h in range(HALVES)]
        else:
            mstar = [lane_max(np.stack([lane_max(np.stack([st[g, h][0] for g in shards])), seed]))
                     for h in range(HALVES)]
            out["mstar"] = np.concatenate(mstar).astype(F16)
        part = {g: [[], []] for g in shards}
        for g, (t0, nt) in shards.items():
            for h in range(HALVES):
                if passes == 2:
                    _, l16, o = sweep(h, t0, nt, mstar[h], True)
                else:
                    m_c, l_c, o = st[g, h]
                    S = np.repeat(mstar[h][None, :], fix_rows, axis=0)
                    mnew = lane_max(np.stack([lane_max(S), m_c]))
                    corr16 = hm.exp16(fp.add16(m_c, -mnew), a32)
                    o = self._colscale(o, corr16)
                    l16 = fp.mul16(l_c, corr16)
                part[g][0].append(self.compact(hm.d_port(o, self.k_o)))
                part[g][1].append(l16)
        flat = {g: np.concatenate([o.reshape(-1) for o in os_] + ([] if merge else list(ls))).astype(F16)
                for g, (os_, ls) in part.items()}
        loc = {}
        for k, gs in chips.items():
            acc = flat[gs[0]]
            for g in gs[1:]:
                acc = fp.add16(acc, flat[g])
            loc[k] = acc
        tot = loc[order[0]]
        for k in order[1:]:
            tot = fp.add16(tot, loc[k])
        n = HD * BR
        xt = np.empty((NH, HD), dtype=F16)
        for h in range(HALVES):
            O = tot[h * n:(h + 1) * n].reshape(HD, BR)
            lt = lstar[h] if merge else tot[HALVES * n + h * BR:HALVES * n + (h + 1) * BR]
            t16 = simd.stream_map(lt, self.a_n, 0.0, simd.RSQRT)
            xt[h * BR:(h + 1) * BR] = fp.mul16(O, fp.mul16(t16, t16)[None, :]).T
        out.update(flat=flat, loc=loc, tot=tot, xt=xt)
        return out

    @staticmethod
    def _colscale(o, corr16):
        """PV's C-path column scale: the INT32 accumulator times its column's FP16 factor, rounded to even."""
        return np.clip(np.rint(o.astype(np.float64) * corr16.astype(np.float64)[None, :]),
                       -2**31, 2**31 - 1).astype(np.int64)

    def report(self, xt):
        dev = np.asarray(xt, np.float64).reshape(-1)
        ref = self.o_ref.astype(np.float64).reshape(-1)
        cos = float(dev @ ref / max(np.linalg.norm(dev) * np.linalg.norm(ref), 1e-30))
        return [f"Llama-3-8B (synthetic INT{self.wbits} weights, seed {self.seed}) grouped-query attention, one query "
                f"at position {self.keys - 1} over a {self.keys}-position cache; {NH} query / {NKV} key-value heads of "
                f"{HD}, two halves of {BR} lanes",
                f"scales s_q={self.s_q.s:.4g} s_k={self.s_k.s:.4g} s_v={self.s_v.s:.4g}; k_s={self.k_s} k_o={self.k_o} "
                f"a'={self.a_exp:.5g}",
                f"device o~ vs the float attention over the float cache: cosine {cos:.5f}"]
