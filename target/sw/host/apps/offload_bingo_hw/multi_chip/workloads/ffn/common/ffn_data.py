# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Data of the ffn workloads: one LLM's dense SwiGLU FFN at its real size, from its checkpoint.

THE WEIGHTS are the checkpoint's own (sweep/llm_blocks/ffn/fetch_weights.py caches them as stored, BF16),
quantised with snax_cluster's util/pack.py: symmetric INT4 per output column. Unlike the dsv2
golden, the RMSNorm gain in front of the FFN is NOT folded into the rows of gate and up: a
real checkpoint's gain spans 30x across channels, so folded, a few rows set every column's
scale and most weights round to zero. The gain is applied to the normed input instead, before
it is quantised (the token's producer does it; here, the host).

THE TOKENS are seeded draws, as the dsv2 golden draws its hidden states (normal, FP16), and so
are the calibration samples; T tokens are T rows of every activation below. Every activation the device quantises has one static scale, the
calibration samples' absolute maximum mapped onto 127 (util/calib.py): the normed input hq and
the SwiGLU output a8.

THE GOLDEN is the device's own arithmetic (util/hwmodel.py, util/fp.py), stage by stage, for the
tensor-parallel mapping the driver builds (ffn_mc.py):

    hq        = quant_i8(rmsnorm(x16) (.) gain, s_h)       on the root, staged
    g16[g]    = d_port(hq @ Wgu[:, cols g], k_x) (.) s_gu   every cluster g its columns
    a8[g]     = quant_i8(silu(gate) * up, s_a)              SwigluARow
    part[g]   = d_port(a8[g] @ Wdn[rows g, :], k_dn)        the down projection, by rows
    loc[k]    = part[k, 0] + part[k, 1]                     each chip folds its clusters (FP16)
    y_raw     = ((loc[root] + loc[o_0]) + loc[o_1]) + ...   the fold order over the chips
    y16       = y_raw (.) s_dn                              on the root

An FP16 sum depends on its order, so the golden takes the order the device folds in; both
implementations of the reduce fold in that order, so they are checked against one golden,
byte for byte. Every FP16 add is the device's one rounding of an FP32 sum (fp.add16), which is
what the SIMD stream adder and the xDMA writer junction both compute.
"""

import json
import os
import sys

import numpy as np

RMS_EPS = 1e-6


def import_util(dsv2_dir):
    """snax_cluster's dsv2 golden package (`util`): fp, hwmodel, layout, pack, reference."""
    dsv2_dir = os.path.abspath(dsv2_dir)
    if dsv2_dir not in sys.path:
        sys.path.insert(0, dsv2_dir)
    import util
    from util import fp, hwmodel, layout, pack, reference
    where = os.path.dirname(os.path.abspath(util.__file__))
    if where != os.path.join(dsv2_dir, "util"):
        raise ImportError(f"imported `util` from {where}, not from {dsv2_dir}/util")
    return fp, hwmodel, layout, pack, reference


def _bf16(path, shape):
    u = np.fromfile(path, dtype="<u2")
    if u.size != int(np.prod(shape)):
        raise ValueError(f"{path}: {u.size} values, expected {shape}")
    return (u.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def load_weights(cache_dir, model, inter=None):
    """The FFN of `model` (a sweep/llm_blocks/ffn/fetch_weights.py key) in [in, out] orientation, float32
    values of the BF16 checkpoint: gate, up [hidden, inter], down [inter, hidden], gain [hidden].
    `inter` keeps only the first `inter` intermediate columns (a smaller FFN of the same layer)."""
    with open(os.path.join(cache_dir, "manifest.json")) as f:
        man = json.load(f)
    if model not in man:
        raise ValueError(f"params model={model!r}: not in {cache_dir}/manifest.json "
                         f"({sorted(man)}); run sweep/llm_blocks/ffn/fetch_weights.py")
    m = man[model]
    H, I = int(m["hidden"]), int(m["inter"])
    t = m["tensors"]
    base = os.path.join(cache_dir, m["repo"].replace("/", "__"))
    for role, d in t.items():
        if d["dtype"] != "BF16":
            raise ValueError(f"{model}: {d['name']} is {d['dtype']}, the FFN data reads BF16")
    rd = lambda role, shape: _bf16(os.path.join(base, t[role]["name"] + ".bin"), shape)
    n = I if inter is None else int(inter)
    if not 0 < n <= I:
        raise ValueError(f"params inter={inter}: {model} has {I} intermediate columns")
    return dict(model=model, repo=m["repo"], revision=m["revision"], layer=int(m["layer"]),
                hidden=H, inter=n,
                gate=np.ascontiguousarray(rd("gate", (I, H))[:n].T),
                up=np.ascontiguousarray(rd("up", (I, H))[:n].T),
                down=np.ascontiguousarray(rd("down", (H, I))[:, :n].T),
                gain=rd("gain", (H,)))


def a_row_bytes(v):
    """int8 values as a_row: value c at (c // 4) * 8 + c % 4, the rest 0 (the GEMV's x)."""
    v = np.asarray(v, dtype=np.int8).reshape(-1)
    o = np.zeros(2 * v.size, dtype=np.int8)
    o.reshape(-1, 8)[:, :4] = v.reshape(-1, 4)
    return o


class FfnData:
    """The FFN's weights packed for the device, its scales, and the golden of one token.

    `cols`: per global cluster (c0, n), its intermediate columns -- of gate and of up -- and the
    same rows of down. `chips`: per chip index its clusters (global numbers). `order`: the chip
    indices in the order the reduce folds them into the root's (the root first)."""

    def __init__(self, cache_dir, model, dsv2_dir, seed=1, n_calib=8, wbits=4, tokens=1, inter=None):
        fp, hm, layout, pack, ref = import_util(dsv2_dir)
        self.fp, self.hm, self.layout, self.packmod = fp, hm, layout, pack
        if wbits != 4:
            raise ValueError(f"params wbits={wbits}: the FFN streams INT4 weights")
        self.wbits = wbits
        W = self.W = load_weights(cache_dir, model, inter)
        H, I = self.H, self.I = W["hidden"], W["inter"]
        self.mesh = layout.MESH
        # ---- the pack: per output column --------------------------------------------------
        self.gu_q, self.gu_s = pack.quantize_columns(np.concatenate([W["gate"], W["up"]], axis=1),
                                                     wbits=wbits)
        self.dn_q, self.dn_s = pack.quantize_columns(W["down"], wbits=wbits)
        self.k_x = fp.d_shift_for_depth(H)
        self.k_dn = fp.d_shift_for_depth(I)
        # ---- the token and the static scales ---------------------------------------------
        rng = np.random.default_rng([int(seed), 2])
        cal = rng.normal(0.0, 1.0, size=(n_calib, H)).astype(np.float16).astype(np.float32)
        self.T = int(tokens)
        self.x16 = rng.normal(0.0, 1.0, size=(self.T, H)).astype(np.float16)    # [T, H]
        self.gain16 = W["gain"].astype(np.float16)
        # the normed, gained input as the device would compute it (RMSNorm, then the gain)
        xg16 = lambda x: fp.mul16(hm.rmsnorm(np.asarray(x, dtype=np.float16)), self.gain16)
        a_cal = []
        for x in cal:
            xg = (ref.rmsnorm_unit(x, RMS_EPS) * W["gain"]).astype(np.float32)
            a_cal.append(ref.silu(xg @ W["gate"]) * (xg @ W["up"]))
        self.s_h = hm.QScale.for_amax(max(float(np.max(np.abs(xg16(x).astype(np.float32))))
                                          for x in cal))
        self.s_a = hm.QScale.for_amax(max(float(np.max(np.abs(a))) for a in a_cal))
        self.inv_a = fp.f32bits(self.s_a.inv)
        # ---- the device's arithmetic, up to the per-cluster cut ---------------------------
        self.hq = np.stack([fp.quant_i8(xg16(x), self.s_h.inv) for x in self.x16])   # [T, H]
        g_raw, _ = hm.gemv(self.hq, self.gu_q, self.k_x)
        self.gu_f = hm.dequant_scale(self.s_h.s, self.gu_s, self.k_x)       # FP16 [2 I]
        g16 = fp.mul16(g_raw, self.gu_f[None, :])
        a16 = fp.mul16(hm.silu16(g16[:, :I]), g16[:, I:])
        self.a8 = fp.quant_i8(a16, self.s_a.inv)                             # [T, I]
        self.dn_f = hm.dequant_scale(self.s_a.s, self.dn_s, self.k_dn)      # FP16 [H]
        # the float reference: the checkpoint's FFN on the same token (BF16 weights, FP32)
        xg = (ref.rmsnorm_unit(self.x16.astype(np.float32), RMS_EPS) * W["gain"]).astype(np.float32)
        self.y_ref = (ref.silu(xg @ W["gate"]) * (xg @ W["up"])) @ W["down"]       # [T, H]

    # ---- per cluster ---------------------------------------------------------------------------
    def x8(self):
        """The root's input, as every cluster's GEMV reads it: each token's hq in a_row, the
        tokens back to back."""
        return np.concatenate([a_row_bytes(h) for h in self.hq])

    def gu_blob(self, c0, n):
        """Cluster weights of gate|up: its gate columns, then its up columns, B_W4 bytes."""
        q = np.concatenate([self.gu_q[:, c0:c0 + n], self.gu_q[:, self.I + c0:self.I + c0 + n]],
                           axis=1)
        return self.packmod._blob(q, self.wbits, self.mesh).astype(np.int8)

    def dn_blob(self, c0, n):
        """Cluster weights of down: its rows, every output column, B_W4 bytes."""
        return self.packmod._blob(self.dn_q[c0:c0 + n, :], self.wbits, self.mesh).astype(np.int8)

    def gu_factors(self, c0, n):
        """The dequantisation factors of the cluster's gate|up columns, FP16 [2 n]."""
        return np.concatenate([self.gu_f[c0:c0 + n], self.gu_f[self.I + c0:self.I + c0 + n]])

    def part(self, c0, n):
        """The cluster's partial y: its rows of down, off the D port (FP16 [T, H])."""
        p, _ = self.hm.gemv(self.a8[:, c0:c0 + n], self.dn_q[c0:c0 + n, :], self.k_dn)
        return p

    def golden(self, cols, chips, order):
        """{part, loc, y_raw, y16} for the mapping: per cluster, per chip, the fold, the output."""
        add16, mul16 = self.fp.add16, self.fp.mul16
        part = {g: self.part(c0, n) for g, (c0, n) in enumerate(cols)}
        loc = {}
        for k, gs in chips.items():
            acc = part[gs[0]]
            for g in gs[1:]:
                acc = add16(acc, part[g])
            loc[k] = acc
        y = loc[order[0]]
        for k in order[1:]:
            y = add16(y, loc[k])
        y16 = mul16(y, self.dn_f[None, :])
        return dict(part=part, loc=loc, y_raw=y, y16=y16)

    def report(self, y16):
        """How far the device's FFN is from the checkpoint's own (float, BF16 weights)."""
        ref = np.asarray(self.y_ref, dtype=np.float64).reshape(-1)
        got = np.asarray(y16, dtype=np.float64).reshape(-1)
        rel = float(np.max(np.abs(got - ref)) / max(np.max(np.abs(ref)), 1e-12))
        cos = float(np.dot(got, ref) / max(np.linalg.norm(got) * np.linalg.norm(ref), 1e-30))
        W = self.W
        return [f"{W['model']}: {W['repo']}@{W['revision'][:12]} layer {W['layer']} FFN "
                f"{self.H} -> {self.I} -> {self.H}, {self.T} token(s), INT{self.wbits} per column; "
                f"{(self.gu_q.size + self.dn_q.size) * self.wbits / 8 / 2**20:.2f} MiB of weights",
                f"D-port shifts k_x={self.k_x} k_dn={self.k_dn}; scales s_h={self.s_h.s:.4g} "
                f"s_a={self.s_a.s:.4g}",
                f"device y vs the checkpoint's float FFN: max |err| / max |ref| = {rel:.4f}, "
                f"cosine {cos:.5f}"]
