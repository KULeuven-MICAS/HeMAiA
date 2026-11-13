# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The golden of the key-split decode step: the device's arithmetic, step by step (KsGolden)."""

import numpy as np

from .consts import BC, BR, D_QK, D_V, KV_RANK, PK, Q_HEAD, Q_NOPE, ROPE


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
