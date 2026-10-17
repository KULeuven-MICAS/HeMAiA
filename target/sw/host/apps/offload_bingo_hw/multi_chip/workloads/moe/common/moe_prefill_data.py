# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Data of the moe prefill workloads: DeepSeek-V2-Lite's layer-1 MoE over a prompt of T tokens,
from its checkpoint, and the golden of every stage as the device computes it.

THE WEIGHTS are the checkpoint's own (sweep/common/weights' cache): the router [64, 2048], the
64 routed experts and the two shared ones as one MLP, quantised per output column -- the router
INT8, the experts and the shared MLP INT4. `inter_e` / `inter_sh` keep the first columns of every
expert / of the shared MLP (a smaller layer with the same router, for the proof of concept).

THE TOKENS are the FFN's seeded draws (ffn_data.py): n_calib calibration rows, then the T prompt
rows, normal FP16; each normed (RMSNorm, then the post-attention gain) and quantised to INT8 with
one static scale, hq. After them n_cal_act more rows calibrate the activation scales.

THE ROUTE, as the root's cluster 0 computes it: logits = d_port(hq @ Wr) times Wr's column
factors (FP16), p = softmax16 per token (the SIMD softmax), the top 6 of p (ties to the lower
index). Expert e's rows are the tokens that chose it, in token order, padded with the zero row
to a whole 8-row tile; its route weight w is p[token, e].

THE EXPERTS (and the shared MLP, every token with weight 1), per row of its input:
    g16  = d_port(x @ [Wg | Wu], k_x) (.) f_gu                 the gate|up GEMM, dequantised
    a16  = silu(g16_gate) (.) g16_up                            SwiGLU
    a8   = quant_i8(map(w * a16), s_a)                          the route weight on the input
                                                                of the down projection
    d16  = d_port(a8 @ Wd, k_dn) (.) f_dn                       the down GEMM, dequantised
The route weight multiplies the SwiGLU output (the down projection is linear, so this is the
weighted sum of the experts' outputs); s_a is calibrated on weighted outputs.

A CHIP'S PARTIAL over a range of output columns: the shared MLP's down over the chip's shared
columns, then each of its experts' d16 rows added into the rows of their tokens, expert by
expert in the chip's slot order, rows in token order (FP16 adds, the SIMD's). The chips'
partials are folded into the root in the reduce's order, as the FFN's are.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ffn",
                                "common"))
from ffn_data import RMS_EPS, _bf16, import_util  # noqa: E402

N_EXP, TOP_K = 64, 6
TILE_ROWS = 8                 # an A / D row tile of array shape 3
MESH3 = (8, 8, 16)            # array shape 3: (Mu, Ku, Nu)


def pad8(n):
    return -(-int(n) // TILE_ROWS) * TILE_ROWS


class MoePrefillData:
    def __init__(self, cache_dir, dsv2_dir, hw=None, seed=1, n_calib=8, n_cal_act=32, tokens=256,
                 inter_e=None, inter_sh=None, wbits=4, layer=1, model="dsv2-lite-bf16"):
        fp, hm, layout, pack, ref = import_util(dsv2_dir)
        self.fp, self.hm, self.layout, self.packmod, self.ref = fp, hm, layout, pack, ref
        self.simd = hm.simd
        if hw is not None:
            self.simd.use_hwcfg(hw)
        if wbits != 4:
            raise ValueError(f"params wbits={wbits}: the experts stream INT4 weights")
        self.wbits, self.T = wbits, int(tokens)
        with open(os.path.join(cache_dir, "manifest.json")) as f:
            m = json.load(f)[model]
        t = m["tensors"]
        base = os.path.join(cache_dir, m["repo"].replace("/", "__"))
        pre = f"model.layers.{layer}."

        def rd(name):
            d = t[pre + name]
            if d["dtype"] != "BF16":
                raise ValueError(f"{name} is {d['dtype']}")
            return _bf16(os.path.join(base, pre + name + ".bin"), d["shape"])

        self.repo, self.revision, self.layer = m["repo"], m["revision"], layer
        self._rd = rd
        gain = self.gain = rd("post_attention_layernorm.weight")
        wr = self.wr = np.ascontiguousarray(rd("mlp.gate.weight").T)            # [H, 64]
        H = self.H = wr.shape[0]
        sh = {p: rd(f"mlp.shared_experts.{p}_proj.weight") for p in ("gate", "up", "down")}
        I_sh_full = sh["gate"].shape[0]
        self.I_sh = I_sh_full if inter_sh is None else int(inter_sh)
        self.I_e = 1408 if inter_e is None else int(inter_e)
        if not 0 < self.I_sh <= I_sh_full or not 0 < self.I_e <= 1408:
            raise ValueError(f"params inter_e={inter_e} (<= 1408), inter_sh={inter_sh} (<= {I_sh_full})")
        if self.I_e % 32 or self.I_sh % 32:
            raise ValueError(f"inter_e={self.I_e}, inter_sh={self.I_sh}: whole pairs of 16-column blocks")
        n = self.I_sh
        self.shared_w = {"gate": sh["gate"][:n].T, "up": sh["up"][:n].T, "down": sh["down"][:, :n].T}
        self.k_x = fp.d_shift_for_depth(H)
        self.k_dn_e = fp.d_shift_for_depth(self.I_e)
        self.k_dn_sh = fp.d_shift_for_depth(self.I_sh)
        # ---- the tokens and the input scale ------------------------------------------------
        rng = np.random.default_rng([int(seed), 2])
        cal = rng.normal(0.0, 1.0, size=(n_calib, H)).astype(np.float16).astype(np.float32)
        self.x16 = rng.normal(0.0, 1.0, size=(self.T, H)).astype(np.float16)   # [T, H]
        cal_act = rng.normal(0.0, 1.0, size=(n_cal_act, H)).astype(np.float16).astype(np.float32)
        self.gain16 = gain.astype(np.float16)
        xg16 = lambda x: fp.mul16(hm.rmsnorm(np.asarray(x, dtype=np.float16)), self.gain16)
        self.s_h = hm.QScale.for_amax(max(float(np.max(np.abs(xg16(x).astype(np.float32))))
                                          for x in cal))
        self.hq = np.stack([fp.quant_i8(xg16(x), self.s_h.inv) for x in self.x16])   # [T, H]
        self.xg_cal = (ref.rmsnorm_unit(cal_act, RMS_EPS) * gain).astype(np.float32)
        self.xg_ref = (ref.rmsnorm_unit(self.x16.astype(np.float32), RMS_EPS) * gain).astype(np.float32)
        # ---- the route ---------------------------------------------------------------------
        self.wr_q, self.wr_s = pack.quantize_columns(wr, wbits=8)
        lg_raw, _ = hm.gemv(self.hq, self.wr_q, self.k_x)
        self.wr_f = hm.dequant_scale(self.s_h.s, self.wr_s, self.k_x)                 # FP16 [64]
        self.logits16 = fp.mul16(lg_raw, self.wr_f[None, :])                          # [T, 64]
        self.p16 = np.stack([hm.softmax16(r) for r in self.logits16])                 # [T, 64]
        self.ids = np.stack([hm.top_k16(r, TOP_K) for r in self.p16]).astype(np.int64)  # [T, 6]
        self.w16 = np.take_along_axis(self.p16, self.ids, 1)                          # [T, 6]
        self.tok = {e: [] for e in range(N_EXP)}
        self.wtok = {e: [] for e in range(N_EXP)}
        for tt in range(self.T):
            for j in range(TOP_K):
                e = int(self.ids[tt, j])
                self.tok[e].append(tt)
                self.wtok[e].append(self.w16[tt, j])
        self.cnt = np.array([len(self.tok[e]) for e in range(N_EXP)])
        self.wtok = {e: np.array(v, dtype=np.float16) for e, v in self.wtok.items()}
        # the float router on the calibration rows: each expert's weight there, for its scale
        lg_cal = self.xg_cal @ wr
        p_cal = np.exp(lg_cal - lg_cal.max(1, keepdims=True))
        self.p_cal = p_cal / p_cal.sum(1, keepdims=True)                              # [n_cal, 64]
        # ---- the shared MLP ----------------------------------------------------------------
        self.sh = self._pack_mlp(self.shared_w, np.ones(len(cal_act)), self.k_dn_sh)
        self._experts, self._a8 = {}, {}

    # ---- packing -------------------------------------------------------------------------
    def _pack_mlp(self, w, w_cal, k_dn):
        """An MLP's INT4 pack and scales: gu_q [H, 2 I], gu_f FP16 [2 I], dn_q [I, H], dn_f FP16
        [H], s_a (calibrated on the calibration rows' SwiGLU outputs times their weights w_cal)."""
        fp, hm, pack, ref = self.fp, self.hm, self.packmod, self.ref
        gu_q, gu_s = pack.quantize_columns(np.concatenate([w["gate"], w["up"]], axis=1),
                                           wbits=self.wbits)
        dn_q, dn_s = pack.quantize_columns(w["down"], wbits=self.wbits)
        act = ref.silu(self.xg_cal @ w["gate"]) * (self.xg_cal @ w["up"])
        s_a = hm.QScale.for_amax(float(np.max(np.abs(act * np.asarray(w_cal)[:, None]))))
        return dict(I=w["gate"].shape[1], gu_q=gu_q, gu_f=hm.dequant_scale(self.s_h.s, gu_s, self.k_x),
                    dn_q=dn_q, dn_f=hm.dequant_scale(s_a.s, dn_s, k_dn), s_a=s_a, k_dn=k_dn,
                    inv_a=fp.f32bits(s_a.inv), w=w)

    def expert(self, e):
        if e not in self._experts:
            rd, n = self._rd, self.I_e
            w = {p: rd(f"mlp.experts.{e}.{p}_proj.weight") for p in ("gate", "up", "down")}
            w = {"gate": w["gate"][:n].T, "up": w["up"][:n].T, "down": w["down"][:, :n].T}
            self._experts[e] = self._pack_mlp(w, self.p_cal[:, e], self.k_dn_e)
        return self._experts[e]

    # ---- the device's arithmetic -----------------------------------------------------------
    def rows_of(self, e):
        """Expert e's rows: its tokens, then the zero row (index T) to a whole tile; and their
        FP16 weights (0 on the padding)."""
        n = len(self.tok[e])
        rows = list(self.tok[e]) + [self.T] * (pad8(n) - n)
        w = np.concatenate([self.wtok[e], np.zeros(pad8(n) - n, dtype=np.float16)])
        return rows, w

    def x_rows(self, rows):
        """The INT8 input rows `rows` (index T: the zero row)."""
        hz = np.concatenate([self.hq, np.zeros((1, self.H), dtype=self.hq.dtype)])
        return hz[np.asarray(rows)]

    def a8(self, mp, xq, w16):
        """An MLP's down operand: INT8 [rows, I] of input rows xq and route weights w16."""
        fp, hm, simd = self.fp, self.hm, self.simd
        I = mp["I"]
        g_raw, _ = hm.gemv(xq, mp["gu_q"], self.k_x)
        g16 = fp.mul16(g_raw, mp["gu_f"][None, :])
        a16 = fp.mul16(hm.silu16(g16[:, :I]), g16[:, I:])
        aw = np.stack([simd.stream_map(a16[r], float(np.float32(w16[r])), 0.0, simd.LINEAR)
                       for r in range(a16.shape[0])])
        return fp.quant_i8(aw, mp["s_a"].inv)

    def a8_expert(self, e):
        if e not in self._a8:
            rows, w = self.rows_of(e)
            self._a8[e] = self.a8(self.expert(e), self.x_rows(rows), w)
        return self._a8[e]

    def a8_shared(self):
        if not hasattr(self, "_a8_sh"):
            self._a8_sh = self.a8(self.sh, self.hq, np.ones(self.T, dtype=np.float16))
        return self._a8_sh

    def d16(self, mp, a8, k0, k1, c0, w):
        """The down GEMM over rows k0:k1 of the MLP's intermediate, output columns c0:c0+w,
        dequantised (FP16 [rows, w])."""
        raw, _ = self.hm.gemv(a8[:, k0:k1], mp["dn_q"][k0:k1, c0:c0 + w], mp["k_dn"])
        return self.fp.mul16(raw, mp["dn_f"][None, c0:c0 + w])

    def chip_partial(self, sh_cols, experts, c0, w):
        """A chip's partial over output columns c0:c0+w (FP16 [T, w]): the shared MLP's down over
        its shared intermediate columns sh_cols = (k0, k1), then its experts in slot order."""
        add16 = self.fp.add16
        y = self.d16(self.sh, self.a8_shared(), sh_cols[0], sh_cols[1], c0, w).copy()
        for e in experts:
            d = self.d16(self.expert(e), self.a8_expert(e), 0, self.I_e, c0, w)
            for i, tt in enumerate(self.tok[e]):
                y[tt] = add16(y[tt], d[i])
        return y

    # ---- the float reference ---------------------------------------------------------------
    def y_ref(self):
        """The checkpoint's layer on the same rows: float router, top 6, BF16 weights in FP32."""
        ref, xg = self.ref, self.xg_ref
        lg = xg @ self.wr
        p = np.exp(lg - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        w = self.shared_w
        y = (ref.silu(xg @ w["gate"]) * (xg @ w["up"])) @ w["down"]
        top = np.argsort(-p, 1)[:, :TOP_K]
        for e in range(N_EXP):
            sel = np.nonzero((top == e).any(1))[0]
            if not sel.size:
                continue
            we = self.expert(e)["w"]
            h = xg[sel]
            y[sel] += p[sel, e][:, None] * ((ref.silu(h @ we["gate"]) * (h @ we["up"])) @ we["down"])
        return y

    def report(self, y16):
        ref = np.asarray(self.y_ref(), dtype=np.float64).reshape(-1)
        got = np.asarray(y16, dtype=np.float64).reshape(-1)
        rel = float(np.max(np.abs(got - ref)) / max(np.max(np.abs(ref)), 1e-12))
        cos = float(np.dot(got, ref) / max(np.linalg.norm(got) * np.linalg.norm(ref), 1e-30))
        return [f"{self.repo}@{self.revision[:12]} layer {self.layer} MoE: {self.T} tokens, 64 experts of "
                f"{self.I_e} (top {TOP_K}), shared {self.I_sh}; tokens per expert {self.cnt.min()}-"
                f"{self.cnt.max()}, {sum(pad8(c) for c in self.cnt)} padded rows for {int(self.cnt.sum())}",
                f"D-port shifts k_x={self.k_x} k_dn={self.k_dn_e} (shared {self.k_dn_sh}); s_h={self.s_h.s:.4g}",
                f"device y vs the checkpoint's float layer: max |err| / max |ref| = {rel:.4f}, cosine {cos:.5f}"]
