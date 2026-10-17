# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Data of the moe workloads: DeepSeek-V2-Lite's layer-1 MoE at its real size, from its checkpoint.

THE WEIGHTS are the checkpoint's own, read from sweep/common/weights' cache (the same pinned commit as
the FFN): the router [64, 2048], 64 routed experts (gate, up [1408, 2048], down [2048, 1408]),
the two shared experts as one MLP of 2816, and the post-attention RMSNorm gain. As for the FFN
(ffn_data.py) the experts are INT4 per output column, the router INT8 per column, and the gain
is applied to the normed input before it is quantised rather than folded into the weights.

THE TOKEN is a seeded normal draw, as ffn_data draws it; the static scales are calibrated on 8
more: the input's, and per expert its SwiGLU output's.

THE ROUTE, two ways, each computed the way its device arm computes it:

  gather (baseline)  every chip's 4 logits gathered on the root, softmax over the 64 there
                     (hwmodel.softmax16, the SIMD row kernel) and the top 6 of the
                     probabilities (moe_route): w = p[ids]
  fold (in fabric)   every chip's own softmax partial over its 4 logits, (m_k, l_k) -- the
                     SIMD row kernel's max and sum -- and its 4 (logit, id) pairs are folded
                     across the chips: the TopK junction keeps the 8 largest logits with their
                     ids, the softmax-statistics junction gives (m*, l*) = (max m_k,
                     sum l_k exp(m_k - m*)), l* rounded to FP16 at every hop of the chain.
                     The root applies them to the 8 candidates (the SIMD row kernel with the
                     stats given: p = exp(l - m*) * rsqrt(l*^2)) and takes the top 6 of those.

Both pick the same experts unless two logits nearly tie (checked); the weights can differ in
their last bits, so each arm has its own golden. The softmax-statistics junction's exponential
is its own table (FpActivation with 128 entries), which ssj_exp models bit for bit, and ssj_fold
the junction's combine: both arms are checked byte for byte.

THE EXPERTS (ffn_mc's mapping, per expert): every cluster takes a range of each expert's
intermediate columns (gate and up) and the same rows of its down projection, so each cluster's
down GEMV gives a partial over every output. Per expert the partial is dequantised (its own
per-column factors) and weighted, and the cluster adds them in slot order after the shared
expert's; then the chips' partials are folded in the fold order, as the FFN's are.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ffn",
                                "common"))
from ffn_data import RMS_EPS, _bf16, a_row_bytes, import_util  # noqa: E402

N_EXP, TOP_K, CAND = 64, 6, 8          # experts, the route's k, the TopK junction's candidates
PAD16 = np.float16(-65504.0)           # a lane no softmax or max may pick
ROUTER_COLS = 32                       # a chip's router block: its logits, then zero columns
SSJ_EXP_LUT_N = 128                    # SoftmaxStatsJunction's exponential table (expLutN)


def _f32_a_wins(a, b):
    """FpHelpers.fp32aWins: a >= b, compared on the FP32 bit patterns (a tie keeps a)."""
    ua = int(np.asarray(a, dtype=np.float32).view(np.uint32))
    ub = int(np.asarray(b, dtype=np.float32).view(np.uint32))
    sa, sb = ua >> 31, ub >> 31
    if sa != sb:
        return sa == 0
    return ub >= ua if sa else ua >= ub


class MoeData:
    """`chips_experts`: per chip index, the 4 expert ids whose logits it computes (in order)."""

    def __init__(self, cache_dir, dsv2_dir, seed=1, n_calib=8, wbits=4,
                 model="dsv2-lite-bf16", layer=1):
        fp, hm, layout, pack, ref = import_util(dsv2_dir)
        self.fp, self.hm, self.layout, self.packmod, self.ref = fp, hm, layout, pack, ref
        if wbits != 4:
            raise ValueError(f"params wbits={wbits}: the experts stream INT4 weights")
        self.wbits = wbits
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
        self.gain = rd("post_attention_layernorm.weight")
        self.wr = np.ascontiguousarray(rd("mlp.gate.weight").T)                  # [2048, 64]
        H = self.H = self.wr.shape[0]
        sh = {p: rd(f"mlp.shared_experts.{p}_proj.weight") for p in ("gate", "up", "down")}
        self.shared = {"gate": sh["gate"].T, "up": sh["up"].T, "down": sh["down"].T}
        self.I_sh = self.shared["gate"].shape[1]
        self._rd = rd
        self._experts = {}
        self.I_e = 1408
        self.k_x = fp.d_shift_for_depth(H)
        # ---- the token, the calibration samples, the scales -----------------------------
        rng = np.random.default_rng([int(seed), 2])
        cal = rng.normal(0.0, 1.0, size=(n_calib, H)).astype(np.float16).astype(np.float32)
        self.h16 = rng.normal(0.0, 1.0, size=H).astype(np.float16)
        self.gain16 = self.gain.astype(np.float16)
        xg16 = lambda x: fp.mul16(hm.rmsnorm(np.asarray(x, dtype=np.float16)), self.gain16)
        self.s_h = hm.QScale.for_amax(max(float(np.max(np.abs(xg16(x).astype(np.float32))))
                                          for x in cal))
        self.hq = fp.quant_i8(xg16(self.h16), self.s_h.inv)
        self._cal_xg = [(ref.rmsnorm_unit(x, RMS_EPS) * self.gain).astype(np.float32)
                        for x in cal]
        # ---- the router: INT8 per column -----------------------------------------------
        self.wr_q, self.wr_s = pack.quantize_columns(self.wr, wbits=8)
        lg_raw, _ = hm.gemv(self.hq, self.wr_q, self.k_x)
        self.wr_f = hm.dequant_scale(self.s_h.s, self.wr_s, self.k_x)              # FP16 [64]
        self.logits16 = fp.mul16(lg_raw, self.wr_f)                                 # [64]
        # ---- the shared expert -----------------------------------------------------------
        self.sh_pack = self._pack_mlp(self.shared)

    # ---- experts --------------------------------------------------------------------------
    def _pack_mlp(self, w):
        """An MLP's INT4 pack, factors and SwiGLU scale: gu_q [H, 2 I], gu_f FP16 [2 I], dn_q
        [I, H], dn_s float [H], the SwiGLU output's static scale, the down GEMV's shift."""
        fp, hm, pack, ref = self.fp, self.hm, self.packmod, self.ref
        I = w["gate"].shape[1]
        gu_q, gu_s = pack.quantize_columns(np.concatenate([w["gate"], w["up"]], axis=1),
                                           wbits=self.wbits)
        dn_q, dn_s = pack.quantize_columns(w["down"], wbits=self.wbits)
        acts = [ref.silu(x @ w["gate"]) * (x @ w["up"]) for x in self._cal_xg]
        s_a = hm.QScale.for_amax(max(float(np.max(np.abs(a))) for a in acts))
        k_dn = fp.d_shift_for_depth(I)
        return dict(I=I, gu_q=gu_q, gu_f=hm.dequant_scale(self.s_h.s, gu_s, self.k_x),
                    dn_q=dn_q, dn_f=hm.dequant_scale(s_a.s, dn_s, k_dn), s_a=s_a, k_dn=k_dn,
                    inv_a=fp.f32bits(s_a.inv), w=w)

    def expert(self, e):
        if e not in self._experts:
            rd = self._rd
            w = {p: rd(f"mlp.experts.{e}.{p}_proj.weight").T for p in ("gate", "up", "down")}
            self._experts[e] = self._pack_mlp(w)
        return self._experts[e]

    def gu_blob(self, mp, c0, n):
        I = mp["I"]
        q = np.concatenate([mp["gu_q"][:, c0:c0 + n], mp["gu_q"][:, I + c0:I + c0 + n]], axis=1)
        return self.packmod._blob(q, self.wbits, self.layout.MESH).astype(np.int8)

    def dn_blob(self, mp, c0, n):
        return self.packmod._blob(mp["dn_q"][c0:c0 + n, :], self.wbits,
                                  self.layout.MESH).astype(np.int8)

    def gu_factors(self, mp, c0, n):
        I = mp["I"]
        return np.concatenate([mp["gu_f"][c0:c0 + n], mp["gu_f"][I + c0:I + c0 + n]])

    def router_blob(self, ids):
        """The router columns of `ids`, padded to 32 columns (a SIMD beat of logits) with zero
        columns, INT8 B layout: a chip's own logits."""
        q = np.zeros((self.H, ROUTER_COLS), dtype=np.int8)
        q[:, :len(ids)] = self.wr_q[:, ids]
        return self.layout.to_b(q, self.layout.MESH).astype(np.int8)

    def router_factors(self, ids):
        f = np.zeros(ROUTER_COLS, dtype=np.float16)
        f[:len(ids)] = self.wr_f[ids]
        return f

    def x8(self):
        return a_row_bytes(self.hq)

    # ---- the route ------------------------------------------------------------------------
    def route_gather(self):
        """(ids, w16, p16): the baseline's softmax over all 64 logits and its top 6."""
        p = self.hm.softmax16(self.logits16)
        ids = self.hm.top_k16(p, TOP_K)
        return [int(e) for e in ids], p[ids], p

    def chip_partial(self, ids):
        """A chip's softmax partial over its logits `ids`, as the SIMD row kernel leaves it on a
        32-lane row padded with -65504: (m_k, l_k) FP16."""
        simd, fp = self.hm.simd, self.fp
        x = np.full(32, PAD16, dtype=np.float16)
        x[:len(ids)] = self.logits16[ids]
        m = simd.reduce_row(simd.as_beats(x), simd.MAX)
        e = self.hm.exp16(fp.add16(x, -m))
        s = simd.reduce_row(simd.as_beats(e), simd.ADD)
        return np.float16(m), np.float16(s)

    def ssj_exp(self, t, n=SSJ_EXP_LUT_N):
        """The softmax-statistics junction's exponential, bit for bit: FpActivation's EXP with
        an n-entry table (dsv2 util/simd.py's model of the same unit, at the junction's table
        depth)."""
        simd, F32 = self.hm.simd, np.float32
        log2ef = F32(1.44269504088896341 * n)
        ln2 = F32(0.6931471805599453 / n)
        lut = np.array([F32(2.0 ** (i / n)) for i in range(n)], dtype=F32)
        with np.errstate(over="ignore", under="ignore"):
            pre = np.maximum(np.minimum(F32(t), simd.HI_E), simd.LO_E)
            s0 = simd.fma32(pre, log2ef, F32(0.0))
            rM = (s0 + simd.MAGIC).astype(F32)
            frac = (s0 - (rM - simd.MAGIC).astype(F32)).astype(F32)
            iM = int(np.asarray(rM).view(np.int32)) - 0x4B400000
            interp = simd.fma32(frac, ln2, F32(1.0))
            r3 = F32(lut[iM & (n - 1)] * interp)
            nE = iM >> (n.bit_length() - 1)
            pow2 = np.uint32(0 if nE < -127 else ((nE + 127) & 0xFF) << 23).view(F32)
            return F32(r3 * pow2)

    def ssj_fold(self, parts, scale=1.0):
        """The softmax-statistics ChainGather over one row's FP16 (m, l) partials, in chain
        order, bit for bit (SoftmaxStatsJunction); each hop's output is the next hop's A
        operand. The keys are compared on their FP32 bit patterns, A winning a tie, and the
        winner's FP16 key is kept exactly; delta = loser - winner in FP32 (a NaN, from two
        equal infinite keys, is 0), times `scale` (the CSR(1) factor); l = fma(l_loser,
        exp(delta), l_winner) in FP32, or l_winner where l_loser is +0, rounded to FP16
        (nearest even) on the way out. -> (m*, l*) FP16."""
        simd, F16, F32 = self.hm.simd, np.float16, np.float32
        m, l = F16(parts[0][0]), F16(parts[0][1])
        for mb, lb in parts[1:]:
            mb, lb = F16(mb), F16(lb)
            wm, wl, lm, ll = (m, l, mb, lb) if _f32_a_wins(m, mb) else (mb, lb, m, l)
            with np.errstate(invalid="ignore", over="ignore"):
                delta = F32(F32(lm) - F32(wm))
            if np.isnan(delta):
                delta = F32(0.0)
            alpha = self.ssj_exp(F32(delta * F32(scale)))
            los, win = F32(ll), F32(wl)
            l32 = win if los.view(np.uint32) == 0 else F32(simd.fma32(los, alpha, win))
            m, l = wm, F16(simd.narrow16(l32))
        return m, l

    def route_fold(self, chips_experts, order):
        """(ids, w16, p8, cand_ids, m*, l*): the in-fabric arm, with the softmax-statistics
        fold modelled bit for bit (ssj_fold) in `order`, the order the chain combines the chips
        in."""
        fp, simd = self.fp, self.hm.simd
        lg = self.logits16.astype(np.float32)
        cand = sorted(range(N_EXP), key=lambda e: (-lg[e], e))[:CAND]
        parts = {k: self.chip_partial(ids) for k, ids in chips_experts.items()}
        m_star, l_star = self.ssj_fold([parts[k] for k in order])
        x = np.full(32, PAD16, dtype=np.float16)
        x[:CAND] = self.logits16[cand]
        e = self.hm.exp16(fp.add16(x, -m_star))
        r = simd.stream_map(fp.mul16(l_star, l_star), 1.0, 0.0, simd.RSQRT)
        p = fp.mul16(e, r)
        top = self.hm.top_k16(p[:CAND], TOP_K)
        ids = [int(cand[i]) for i in top]
        return ids, p[top], p[:CAND], cand, m_star, l_star

    # ---- the experts' output ---------------------------------------------------------------
    def cluster_out(self, slots, w16, cols_of, g):
        """Cluster g's partial of the MoE output (FP16 [H]): the shared expert's, then each
        slot's weighted, added in slot order. cols_of(mp, g) -> (c0, n)."""
        fp, hm = self.fp, self.hm

        def part(mp):
            c0, n = cols_of(mp, g)
            gr, _ = hm.gemv(self.hq, np.concatenate([mp["gu_q"][:, c0:c0 + n],
                                                     mp["gu_q"][:, mp["I"] + c0:mp["I"] + c0 + n]],
                                                    axis=1), self.k_x)
            g16 = fp.mul16(gr, self.gu_factors(mp, c0, n))
            a8 = fp.quant_i8(fp.mul16(hm.silu16(g16[:n]), g16[n:]), mp["s_a"].inv)
            y, _ = hm.gemv(a8, mp["dn_q"][c0:c0 + n, :], mp["k_dn"])
            return fp.mul16(y, mp["dn_f"])

        acc = part(self.sh_pack)
        for e, w in zip(slots, w16):
            acc = fp.add16(acc, fp.mul16(part(self.expert(e)), np.float16(w)))
        return acc

    def report(self, ids_g, ids_f, w_g, w_f):
        p_ref = self.ref.softmax(self.logits16.astype(np.float32))
        return [f"{self.repo}@{self.revision[:12]} layer {self.layer} MoE: 64 experts top-6 + "
                f"shared {self.I_sh}; INT4 experts, INT8 router",
                f"route (gather) {ids_g}, weights {[round(float(w), 4) for w in w_g]}",
                f"route (fold)   {ids_f}, weights {[round(float(w), 4) for w in w_f]}",
                f"float softmax of the device logits on those: "
                f"{[round(float(p_ref[e]), 4) for e in ids_g]}"]
