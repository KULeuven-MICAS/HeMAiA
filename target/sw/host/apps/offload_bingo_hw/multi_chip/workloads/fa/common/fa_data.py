# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Data of the fa workloads: DeepSeek-V2-Lite's layer-1 MLA attention for one decode token over a
REAL 8K-token latent cache.

THE CACHE is sweep/llm_blocks/fa/real_kv_cache.py's: 8,193 tokens of real text through the checkpoint's
embedding, layer 0 and layer 1's KV projection -- per token the normalised latent c_unit [512]
(without the kv_a_layernorm gain) and the rotated k_pe [64], and the hidden state x1 entering
layer 1. The device cache is INT8 with one static scale per part (the cache's own absolute
maximum mapped onto 127, as util/calib.py does), in the two copies MLA reads: the key copy
(A layout of [keys, 576]) and the value copy (A layout of V^T = c8^T).

THE QUERY is the last token's (position T - 1, whose row is the cache's last): x1 through layer 1's
input RMSNorm and q_proj, q_pe rotated at its position, q_nope absorbed into the latent space
(W_UK, the kv_a_layernorm gain moved onto the query), quantised with util/calib.py's tied scales
(s_qt * s_c = s_qpe * s_kpe, so each score is one exact INT32 dot product over [c | k_pe]),
calibrated on the last 64 tokens' queries. The 16 heads fill the BR query lanes -- one SIMD beat
of FP16 lanes, a whole multiple of the heads -- repeated if BR is wider (a zero lane would sum
every key's value at weight 1, near the FP16 range).

THE GOLDEN is the device's arithmetic (util/hwmodel.py, the snax FA kernels' models) for the key
split the driver builds -- dsv2_datagen.split_attention_golden's mode "key", in this platform's
fold order: every cluster's shard of key tiles; pass 1, the shard's row maximum; m* their
lanewise maximum (exact); pass 2, the shard's softmax seeded at m* (no correction) and PV, O^T
leaving the D port as FP16 at k_o, the row sums FP16; the shards' (O, l) added in FP16 in the
fold order (each chip's two clusters, then the chips, in the order the reduce adds them); o~ =
O * rsqrt(l^2 / c^2).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ffn",
                                "common"))
from ffn_data import RMS_EPS, _bf16, import_util  # noqa: E402

# BR: the query lanes, one SIMD beat of FP16 values (the softmax reduces one query per lane) --
# the cluster's SIMD beat, which fa_mc.py checks against the cluster cfg.
BC, BR, D_QK, D_V = 64, 16, 576, 512


class FaData:
    def __init__(self, kv_path, weights_cache, dsv2_dir, keys=8192, layer=1,
                 model="dsv2-lite-bf16", n_calib=64):
        fp, hm, layout, pack, ref = import_util(dsv2_dir)
        self.fp, self.hm, self.layout, self.ref = fp, hm, layout, ref
        from util import model as mdl
        self.mdl = mdl
        d = self.dims = mdl.load()
        z = np.load(kv_path)
        if z["c_unit"].shape[0] < keys:
            raise ValueError(f"{kv_path}: {z['c_unit'].shape[0]} tokens, fewer than {keys}")
        if keys % BC:
            raise ValueError(f"{keys} keys are not whole tiles of {BC}")
        self.keys = keys
        c_unit, kpe, x1 = z["c_unit"][:keys], z["kpe"][:keys], z["x1"][:keys]
        # ---- the layer-1 weights the query needs ---------------------------------------
        with open(os.path.join(weights_cache, "manifest.json")) as f:
            m = json.load(f)[model]
        t = m["tensors"]
        base = os.path.join(weights_cache, m["repo"].replace("/", "__"))
        pre = f"model.layers.{layer}."
        rd = lambda n: _bf16(os.path.join(base, pre + n + ".bin"), t[pre + n]["shape"])
        in_norm = rd("input_layernorm.weight")
        wq = rd("self_attn.q_proj.weight").T                         # [2048, 3072]
        kv_norm = rd("self_attn.kv_a_layernorm.weight")              # [512]
        wkvb = rd("self_attn.kv_b_proj.weight").T                    # [512, 4096]
        H = d.heads
        F32 = np.float32

        def query(pos):
            xn = ref.rmsnorm(x1[pos], in_norm, d.rms_eps)
            q = (xn @ wq).astype(F32).reshape(H, d.q_head)
            cos, sin = mdl.rope_cos_sin(d, pos)
            qpe = mdl.rope_interleaved(q[:, d.q_nope:].astype(np.float64), cos, sin).astype(F32)
            qlat = np.stack([wkvb[:, h * 256:h * 256 + d.q_nope] @ q[h, :d.q_nope]
                             for h in range(H)]).astype(F32)
            return (qlat * kv_norm[None, :]).astype(F32), qpe

        # ---- scales (util/calib.py: amax onto 127; the query's two parts tied) -----------
        amax = lambda a: float(np.max(np.abs(np.asarray(a, dtype=np.float64))))
        self.s_c = hm.QScale.for_amax(amax(c_unit))
        self.s_kpe = hm.QScale.for_amax(amax(kpe))
        cal = [query(p) for p in range(keys - n_calib, keys)]
        prod = max(max(amax(a) for a, _ in cal) * self.s_c.s,
                   max(amax(b) for _, b in cal) * self.s_kpe.s) / 127.0
        self.s_qt = hm.QScale(float(F32(self.s_c.s / prod)))
        self.s_qpe = hm.QScale(float(F32(self.s_kpe.s / prod)))
        # ---- the device cache and query ---------------------------------------------------
        self.c8 = fp.quant_i8(c_unit.astype(np.float16), self.s_c.inv)
        self.kpe8 = fp.quant_i8(kpe.astype(np.float16), self.s_kpe.inv)
        self.rows = np.concatenate([self.c8, self.kpe8], axis=1).astype(np.int8)  # [keys, 576]
        qt, qpe = query(keys - 1)
        Q8 = np.zeros((BR, D_QK), dtype=np.int8)
        Q8[:H, :d.kv_rank] = fp.quant_i8(qt.astype(np.float16), self.s_qt.inv)
        Q8[:H, d.kv_rank:] = fp.quant_i8(qpe.astype(np.float16), self.s_qpe.inv)
        if BR % H:
            raise ValueError(f"{H} heads do not tile {BR} query lanes")
        for r in range(H, BR, H):
            Q8[r:r + H] = Q8[:H]
        self.Q8 = Q8
        self.k_s = fp.d_shift_for_depth(D_QK)
        self.k_o = fp.d_shift_for_depth(keys)
        self.a_exp = float(F32(d.softmax_scale * self.s_qt.s * self.s_c.s * 2.0 ** self.k_s))
        self.a_n = float(F32(hm.P8_SCALE / (2.0 ** self.k_o * self.s_c.s)))
        self.a_n_bits = fp.f32bits(self.a_n)
        # the float reference: the same token's absorbed attention over the float cache
        s = ((qt @ c_unit.T) + (qpe @ kpe.T)).astype(np.float64) * d.softmax_scale
        p = np.exp(s - s.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        self.o_ref = (p @ c_unit.astype(np.float64)).astype(np.float32)          # [H, 512]
        self.heads = H

    # ---- the device layouts ---------------------------------------------------------------
    def q8_a(self):
        return self.layout.to_a(self.Q8, self.layout.MESH).astype(np.int8)

    def tiles(self, t0, nt):
        """Tiles t0 .. t0 + nt - 1 as the attention reads them in L1: per tile its K tile (the
        key copy's rows, A layout, BC x 576 bytes) then its V tile (the value copy's columns,
        D_V x BC bytes, as MlaShardScores' 2-D load leaves them)."""
        lay, cap = self.layout, self.keys
        key = lay.key_copy(self.rows, cap, lay.MESH).astype(np.int8).reshape(-1)
        val = lay.value_copy(self.rows[:, :D_V], cap, lay.MESH).astype(np.int8).reshape(-1)
        kb, vb = BC * D_QK, D_V * BC
        out = []
        for j in range(t0, t0 + nt):
            k = key[j * kb:(j + 1) * kb]
            v = np.concatenate([val[r * 16 * cap + j * 16 * BC: r * 16 * cap + (j + 1) * 16 * BC]
                                for r in range(D_V // 16)])
            if k.size != kb or v.size != vb:
                raise ValueError(f"tile {j}: {k.size} / {v.size} bytes")
            out.append((k, v))
        return out

    # ---- the golden -----------------------------------------------------------------------
    def _tile(self, j):
        K = self.rows[j * BC:(j + 1) * BC].astype(np.int64)
        V = self.c8[j * BC:(j + 1) * BC].astype(np.int64)
        return self.hm.d_port(K @ self.Q8.astype(np.int64).T, self.k_s), V

    def golden(self, shards, chips, order):
        """shards {g: (t0, nt)}, chips {chip: [g, ...]} (its clusters, cluster 0 first),
        order: the chips in fold order. -> m per cluster, m*, (O16, l) per cluster, the folded
        O and l, o~^T [512, 32] and xt [32, 512]."""
        hm, fp, simd = self.hm, self.fp, self.hm.simd
        F16, F32, F64 = np.float16, np.float32, np.float64
        lane_max = lambda a: simd.reduce_lanewise(a, simd.MAX)
        seed = np.full(BR, -65504.0, dtype=F16)
        a32 = F32(self.a_exp)
        m_c = {}
        for g, (t0, nt) in shards.items():
            m16 = seed
            for j in range(t0, t0 + nt):
                m16 = lane_max(np.stack([lane_max(self._tile(j)[0]), m16]))
            m_c[g] = m16
        # each chip's two clusters, then the chips: the lanewise max is exact in any order
        mstar = lane_max(np.stack([lane_max(np.stack(list(m_c.values()))), seed]))
        O16, lc = {}, {}
        for g, (t0, nt) in shards.items():
            m16, l16, o = mstar, np.zeros(BR, dtype=F16), None
            for j in range(t0, t0 + nt):
                S16, V = self._tile(j)
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
            O16[g], lc[g] = hm.d_port(o, self.k_o), l16
        loc = {}
        for k, gs in chips.items():
            acc_o, acc_l = O16[gs[0]], lc[gs[0]]
            for g in gs[1:]:
                acc_o, acc_l = fp.add16(acc_o, O16[g]), fp.add16(acc_l, lc[g])
            loc[k] = (acc_o, acc_l)
        Ot, lt = loc[order[0]]
        for k in order[1:]:
            Ot, lt = fp.add16(Ot, loc[k][0]), fp.add16(lt, loc[k][1])
        t16 = simd.stream_map(lt, self.a_n, 0.0, simd.RSQRT)
        rsc16 = fp.mul16(t16, t16)
        ot = fp.mul16(Ot, rsc16[None, :])                                     # [512, BR]
        return dict(m_c=m_c, mstar=mstar, O16=O16, l=lc, loc=loc, Ot=Ot, lt=lt, ot=ot,
                    xt=np.ascontiguousarray(ot.T))

    def part_bytes(self, O16, l16, l_beats=1):
        """A cluster's (O16, l) as the reduce adds it: O^T then its row-sum beat, FP16, repeated
        l_beats times (the partial carries l in a whole transfer unit, the running l and the
        new one of the arena, equal after the commit)."""
        l16 = np.asarray(l16, np.float16).reshape(-1)
        return np.concatenate([np.asarray(O16, np.float16).reshape(-1)] + [l16] * l_beats)

    def report(self, xt):
        H = self.heads
        dev = np.asarray(xt, np.float64)[:H]
        ref = self.o_ref.astype(np.float64)
        cos = float(np.sum(dev * ref) / max(np.linalg.norm(dev) * np.linalg.norm(ref), 1e-30))
        return [f"DeepSeek-V2-Lite layer-1 MLA, one query at position {self.keys - 1} over a real "
                f"{self.keys}-token cache; {H} heads",
                f"scales s_c={self.s_c.s:.4g} s_kpe={self.s_kpe.s:.4g} s_qt={self.s_qt.s:.4g} "
                f"s_qpe={self.s_qpe.s:.4g}; k_s={self.k_s} k_o={self.k_o} a'={self.a_exp:.5g}",
                f"device o~ vs the float attention over the float cache: cosine {cos:.5f}"]
