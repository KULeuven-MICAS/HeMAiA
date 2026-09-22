#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Operands and goldens for one toy transformer layer, shared by the bring-up ladder.

Every rung of the ladder -- llm_s1_norm through llm_layer -- generates from THIS
module, so a golden is computed once and every rung checks the same numbers. Two
copies of the chain would drift the moment one stage changed, and the drift would
look like a hardware failure on whichever rung was not updated.

THE GOLDEN MIRRORS THE DEVICE STAGE FOR STAGE, including every narrowing that loses
information: int32 -> fp16 after each GEMM, fp16 -> int8 before each projection. Skipping
one would make the golden more accurate than the hardware, and every check would then fail
by a margin that looks like a bug in the kernel rather than a bug in the model.

Everything is generated ROW-MAJOR and converted with util/sim/xdma/layout_convert.py.
Generating directly in a blocked layout is how the two sides come to agree on the wrong
thing: the old MoE datagen emitted B as (K, N, tileSize, meshCol), which is not B-layout
but has the same flat length, and nothing caught it because the same flat buffer went to
the golden.
"""

import math
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _usg_paths  # noqa: F401,E402  (registers util/sim/{common,gemm,xdma,ara})
from layout_convert import row_major_to_a, row_major_to_b, row_major_to_d  # noqa: E402
from sim_golden_models import int32_to_fp16_golden  # noqa: E402


def f32bits(x):
    return int(np.asarray(x, dtype=np.float32).view(np.uint32))


def int8_scale_for(peak):
    """The largest power of two that keeps `peak` inside the int8 range.

    Derived from the data, not fixed. A scale sized for one stage's range saturates
    another's, and a tensor whose every value is +-127 agrees with its golden whatever
    produced it. A power of two is exact in fp16, so it adds no error of its own.
    """
    if peak <= 0:
        return 1.0
    return float(2.0 ** math.floor(math.log2(127.0 / peak)))


def _to_fp16(x_i32):
    """The GEMM D-port's int32 -> fp16 narrowing, bit for bit as the hardware does it."""
    flat = np.asarray(x_i32, dtype=np.int64).reshape(-1)
    return np.array([int32_to_fp16_golden(int(v)) for v in flat],
                    dtype=np.uint16).view(np.float16).reshape(np.shape(x_i32))


def _rmsnorm(x_f16):
    """RMSNorm as the fused kernel computes it: no learnable gain, fp32 internals.

    The device has no FPU on the engine cores, so the kernel does sum-of-squares in the
    SIMD, then an integer sqrt and reciprocal. The arithmetic below is the fp32 reference
    that describes; the tolerance on the check is what absorbs the difference.
    """
    x = x_f16.astype(np.float32)
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-6)
    return (x / rms).astype(np.float16)


def _rope(x_f16, cos_f16, sin_f16):
    """RoPE over adjacent pairs: out = x*cos + swap(x)*sin, with swap negating the even lane.

    The kernel builds `swap` with an iDMA adjacent-pair swap and folds the sign into the
    SIN TABLE, which is why the caller stages `sin_signed` rather than sin. Reproducing
    that here -- rather than writing the textbook rotation -- is what makes the golden
    describe the kernel instead of the maths.
    """
    x = x_f16.astype(np.float32)
    swapped = x.reshape(x.shape[0], -1, 2)[:, :, ::-1].reshape(x.shape)
    return (x * cos_f16.astype(np.float32)
            + swapped * sin_f16.astype(np.float32)).astype(np.float16)


# THE WEIGHT QUANTISATION SCALE. The int8 weight tensors below are quantisations of real
# weights at this scale -- w_real = w_q / W_SCALE, so |w_real| <= 2/16 = 0.125, an ordinary
# transformer weight magnitude. It exists because a layer has to DEQUANTISE after each
# GEMM, and the dequant factor is 1/(scale_x * W_SCALE).
#
# WHY 16 AND NOT SOMETHING ELSE. Two fp16 limits bracket the GEMM and neither reports when
# it is exceeded:
#   the D port narrows the int32 accumulation to fp16   -> |x_q @ w_q| < 65504
#   RMSNorm reduces SUM(x^2) into an fp16 scalar        -> sum_j x[j]^2 < 65504
# Measured over this data at d=128: the accumulation peaks at ~3.7e3 (fine) and 16 puts the
# dequantised activations at ~3.6 with a worst row sum-of-squares of ~315, i.e. 200x under
# the reduce limit. Without the dequant that sum reaches 1.3e9 and RMSNorm returns a
# well-formed tensor scaled by 2^16 rather than an error.
W_SCALE = 16.0


def _dequant(x_f16, scale):
    """The device's dequantise: multiply the D port's fp16 output by an fp32 scalar."""
    return (x_f16.astype(np.float32) * np.float32(scale)).astype(np.float16)


def _quant(x_f16, scale):
    q = np.rint(x_f16.astype(np.float32) * scale)
    return np.clip(q, -128, 127).astype(np.int8)


def _softmax_rows(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def generate_layer_data(p):
    """Every array the layer needs, keyed by name. Shapes are in ELEMENTS."""
    T, d, h = p["tokens"], p["d_model"], p["d_hidden"]
    E, k = p["num_experts"], p["top_k"]
    mr, ts, mc = p["meshRow"], p["tileSize"], p["meshCol"]
    rng = np.random.default_rng(seed=7)
    out = {}

    # ---- the layer input ---------------------------------------------------------------
    # Small ranges on purpose: this layer narrows to fp16 four times and to int8 three
    # times, and two int8 GEMMs back to back over d reach ~1e6 with full-range operands --
    # past fp16's 65504, where the golden and the device agree only on inf.
    x = (rng.integers(-4, 4, size=(T, d)).astype(np.float32) / 4.0).astype(np.float16)
    out["x_packed"] = x

    # ================= attention =========================================================
    n1 = _rmsnorm(x)
    out["norm1_golden"] = n1

    s1 = int8_scale_for(float(np.abs(n1.astype(np.float32)).max()))
    out["scale_n1"], out["scale_n1_bits"] = s1, f32bits(s1)
    n1_q = _quant(n1, s1)
    out["norm1_a"] = row_major_to_a(n1_q, T // mr, d // ts, mr, ts)

    # Q, K, V projections. Square, because this toy has one head and d_head == d_model.
    dq1 = 1.0 / (s1 * W_SCALE)
    out["dq_proj"], out["dq_proj_bits"] = dq1, f32bits(dq1)
    proj = {}
    for nm in ("q", "k", "v"):
        w = rng.integers(-2, 2, size=(d, d), dtype=np.int8)
        out[f"w_{nm}"] = row_major_to_b(w, d // ts, d // mc, ts, mc)
        # D port first, THEN the scale -- that is the order the device runs them in, and
        # the narrowing is not distributive over the multiply.
        y = _dequant(_to_fp16(n1_q.astype(np.int32) @ w.astype(np.int32)), dq1)
        proj[nm] = y
        out[f"proj_{nm}_golden"] = row_major_to_d(y, T // mr, d // mc, mr, mc)

    # RoPE tables. Precomputed because the device cores have no FPU, and the sign is
    # folded into sin for the same reason -- see _rope().
    pos = np.arange(T)[:, None]
    inv = (1.0 / (10000.0 ** (np.arange(0, d, 2) / d)))[None, :]
    ang = pos * inv
    cos_full = np.repeat(np.cos(ang), 2, axis=1).astype(np.float16)
    sin_half = np.repeat(np.sin(ang), 2, axis=1)
    sin_signed = (sin_half * np.tile([-1.0, 1.0], d // 2)[None, :]).astype(np.float16)
    out["rope_cos"], out["rope_sin"] = cos_full, sin_signed

    q_r = _rope(proj["q"], cos_full, sin_signed)
    k_r = _rope(proj["k"], cos_full, sin_signed)
    out["rope_q_golden"], out["rope_k_golden"] = q_r, k_r

    # FlashAttention operands are int8 in the array's own layouts: Q is B-layout (it is the
    # score matmul's B operand), K and V are A-layout.
    s_qk = int8_scale_for(max(float(np.abs(q_r.astype(np.float32)).max()),
                              float(np.abs(k_r.astype(np.float32)).max()),
                              float(np.abs(proj["v"].astype(np.float32)).max())))
    out["scale_qkv"], out["scale_qkv_bits"] = s_qk, f32bits(s_qk)
    qi, ki, vi = (_quant(t, s_qk) for t in (q_r, k_r, proj["v"]))
    out["fa_q_b"] = row_major_to_b(qi, T // ts, d // mc, ts, mc)
    out["fa_v_a"] = row_major_to_a(vi, T // mr, d // ts, mr, ts)

    # K IS THE FULL KV CACHE, NOT THE 32 CURRENT TOKENS. Under kvsplit FlashAttention
    # declares k as (bc * clusters, dhead) -- one disjoint shard per cluster, stacked --
    # and each cluster reads its own shard at its own offset. Staging only bc rows there
    # does not fault and does not fail a check: the three clusters past the first read
    # whatever staged array happens to follow K in L3, and the run completes on garbage.
    # So the cache is built at its declared length: ncl*bc past positions, projected and
    # rotated exactly like the current tokens, with their own position indices.
    ncl = int(p["num_clusters"])
    kv_len = ncl * T
    x_kv = (rng.integers(-4, 4, size=(kv_len, d)).astype(np.float32) / 4.0).astype(np.float16)
    kv_pos = np.arange(kv_len)[:, None]
    kv_ang = kv_pos * inv
    kv_cos = np.repeat(np.cos(kv_ang), 2, axis=1).astype(np.float16)
    kv_sin = (np.repeat(np.sin(kv_ang), 2, axis=1)
              * np.tile([-1.0, 1.0], d // 2)[None, :]).astype(np.float16)
    w_k_rm = rng.integers(-2, 2, size=(d, d), dtype=np.int8)
    out["w_k_cache"] = row_major_to_b(w_k_rm, d // ts, d // mc, ts, mc)
    k_cache = _rope(_to_fp16(_quant(_rmsnorm(x_kv), s1).astype(np.int32)
                             @ w_k_rm.astype(np.int32)), kv_cos, kv_sin)
    out["fa_k_a"] = row_major_to_a(_quant(k_cache, s_qk), kv_len // mr, d // ts, mr, ts)

    # The attention itself, in fp32 -- this is the REFERENCE, not the device's online
    # recurrence. NO CAUSAL MASK, because the block has none: every query attends to every
    # key. See the TODO on FlashAttention.
    #
    # IT IS COMPUTED OVER `ki`, THE CURRENT TOKENS, NOT OVER THE CACHE STAGED ABOVE, and
    # that is deliberate. This tensor exists to produce `attn_ctx_a`, the operand stage 4
    # consumes -- it is a self-consistent stand-in whose job is to make proj_o and
    # everything after it checkable. It is NOT a golden for what FA computes: the ladder's
    # stage 3 is unchecked, and FA's own arithmetic is covered by fa_decode_4cluster.
    # Making this one describe FA would also mean reconciling FA's kvsplit asymmetry --
    # k stacked per cluster, v shared -- which belongs with the block, not here.
    scores = qi.astype(np.float32) @ ki.astype(np.float32).T
    attn = _softmax_rows(scores / math.sqrt(d))
    # AND THE ATTENTION OUTPUT IS DEQUANTISED TOO. `vi` is V quantised by s_qk, so the
    # weighted sum above lands in the int8 domain (O(100)); the output projection and the
    # residual both expect the activation domain. Leaving it unscaled is what put resid1
    # at ~326 and overflowed RMSNorm's fp16 sum-of-squares.
    ctx_ = _dequant((attn @ vi.astype(np.float32)).astype(np.float16), 1.0 / s_qk)
    out["attn_golden"] = ctx_

    s_ctx = int8_scale_for(float(np.abs(ctx_.astype(np.float32)).max()))
    out["scale_ctx"], out["scale_ctx_bits"] = s_ctx, f32bits(s_ctx)
    ctx_q = _quant(ctx_, s_ctx)
    # The output projection's A-layout operand, STAGED. FlashAttention's own result is
    # int32 in the D-port scatter layout and, under kvsplit, still un-folded across four
    # clusters, so the layer cannot consume it yet -- see the stage-3 note in
    # llm_layer_stages.py. Staging the reference context is what keeps proj_o, resid1 and
    # everything after them checkable against a golden that describes the same numbers:
    # feeding proj_o some other tensor makes proj_o_golden wrong and silently poisons
    # every later rung.
    out["attn_ctx_a"] = row_major_to_a(ctx_q, T // mr, d // ts, mr, ts)

    w_o = rng.integers(-2, 2, size=(d, d), dtype=np.int8)
    out["w_o"] = row_major_to_b(w_o, d // ts, d // mc, ts, mc)
    dqo = 1.0 / (s_ctx * W_SCALE)
    out["dq_proj_o"], out["dq_proj_o_bits"] = dqo, f32bits(dqo)
    attn_out = _dequant(_to_fp16(ctx_q.astype(np.int32) @ w_o.astype(np.int32)), dqo)
    out["proj_o_golden"] = row_major_to_d(attn_out, T // mr, d // mc, mr, mc)

    resid1 = (x.astype(np.float32) + attn_out.astype(np.float32)).astype(np.float16)
    out["resid1_golden"] = resid1

    # ================= the feed-forward half =============================================
    n2 = _rmsnorm(resid1)
    out["norm2_golden"] = n2
    s2 = int8_scale_for(float(np.abs(n2.astype(np.float32)).max()))
    out["scale_n2"], out["scale_n2_bits"] = s2, f32bits(s2)
    n2_q = _quant(n2, s2)
    out["norm2_a"] = row_major_to_a(n2_q, T // mr, d // ts, mr, ts)

    # Fixed router logits, not random: which experts win has to be known at generation
    # time or the per-expert goldens cannot be computed. These make experts 3 and 1 the
    # top two -- neither the first two nor adjacent, so a combine that quietly folded
    # slots 0..k-1 would pass on an easier set.
    logits = np.array([-1.5, 0.3, -0.8, 2.1], dtype=np.float32)[:E]
    if E > 4:
        logits = np.concatenate([logits, rng.uniform(-2, 2, E - 4).astype(np.float32)])
    out["router_logits"] = logits
    prob = _softmax_rows(logits[None, :])[0].astype(np.float32)
    out["softmax_golden"] = prob
    winners = np.argsort(-prob, kind="stable")[:k]
    weight = np.zeros(E, dtype=np.float32)
    weight[winners] = prob[winners] / prob[winners].sum()
    out["weight_golden"], out["winners"] = weight, winners

    combined = np.zeros((T, d), dtype=np.float32)
    acts = {}
    W = {}
    for e in range(E):
        W[e] = (rng.integers(-2, 2, size=(d, h), dtype=np.int8),
                rng.integers(-2, 2, size=(h, d), dtype=np.int8))
        up = _to_fp16(n2_q.astype(np.int32) @ W[e][0].astype(np.int32))
        g = up.astype(np.float32)
        sig = np.where(g >= 0, 1.0 / (1.0 + np.exp(-np.abs(g))),
                       np.exp(-np.abs(g)) / (1.0 + np.exp(-np.abs(g))))
        acts[e] = (g * sig).astype(np.float16)

    s_act = int8_scale_for(max(float(np.abs(a.astype(np.float32)).max())
                               for a in acts.values()))
    out["scale_act"], out["scale_act_bits"] = s_act, f32bits(s_act)
    for e in range(E):
        w_up, w_down = W[e]
        out[f"w_up_{e}"] = row_major_to_b(w_up, d // ts, h // mc, ts, mc)
        out[f"w_down_{e}"] = row_major_to_b(w_down, h // ts, d // mc, ts, mc)
        y = _to_fp16(_quant(acts[e], s_act).astype(np.int32) @ w_down.astype(np.int32))
        out[f"ffn_y_golden_{e}"] = row_major_to_d(y, T // mr, d // mc, mr, mc)
        combined += weight[e] * y.astype(np.float32)

    # The dense up-projection the ladder's stage 5 actually builds. The MoE combine below
    # is the eventual shape; this is what a single Linear(d -> h) produces, and the rung
    # has to be checked against what it computes rather than what it will compute.
    dq2 = 1.0 / (s2 * W_SCALE)
    out["dq_ffn"], out["dq_ffn_bits"] = dq2, f32bits(dq2)
    up0 = _dequant(_to_fp16(n2_q.astype(np.int32) @ W[0][0].astype(np.int32)), dq2)
    out["ffn_up_golden"] = row_major_to_d(up0, T // mr, h // mc, mr, mc)
    # ...and the layer output the LADDER produces: resid1 + that up-projection. The
    # `layer_golden` below is the eventual shape, with the MoE combine in place of the
    # dense projection; a rung has to be checked against what it computes.
    out["ladder_out_golden"] = (resid1.astype(np.float32)
                                + up0.astype(np.float32)).astype(np.float16)

    ffn_out = combined.astype(np.float16)
    out["ffn_golden"] = row_major_to_d(ffn_out, T // mr, d // mc, mr, mc)

    layer_out = (resid1.astype(np.float32) + ffn_out.astype(np.float32)).astype(np.float16)
    out["layer_golden"] = layer_out
    return out


def stage(st, data, p):
    """Hand every array to DataStaging and return the handles the DFG names.

    WHERE these land is the platform's business: a config with a memory chiplet gets a
    mempool.bin, one without gets C arrays in the host image. Addressing a memory chiplet
    a config does not have reads unmapped memory rather than faulting.
    """
    E = p["num_experts"]
    u16 = lambda a: np.ascontiguousarray(a).astype(np.float16).view(np.uint16)  # noqa: E731
    h = {
        "x": st.put("llm_x", "uint16_t", u16(data["x_packed"])),
        "rope_cos": st.put("llm_rope_cos", "uint16_t", u16(data["rope_cos"])),
        "rope_sin": st.put("llm_rope_sin", "uint16_t", u16(data["rope_sin"])),
        "norm1_a": st.put("llm_norm1_a", "int8_t", data["norm1_a"]),
        "norm2_a": st.put("llm_norm2_a", "int8_t", data["norm2_a"]),
        "fa_q": st.put("llm_fa_q", "int8_t", data["fa_q_b"]),
        "fa_k": st.put("llm_fa_k", "int8_t", data["fa_k_a"]),
        "fa_v": st.put("llm_fa_v", "int8_t", data["fa_v_a"]),
        "attn_ctx": st.put("llm_attn_ctx", "int8_t", data["attn_ctx_a"]),
        "w_o": st.put("llm_w_o", "int8_t", data["w_o"]),
        "logits": st.put("llm_logits", "uint32_t", data["router_logits"].view(np.uint32)),
    }
    for nm in ("q", "k", "v"):
        h[f"w_{nm}"] = st.put(f"llm_w_{nm}", "int8_t", data[f"w_{nm}"])
    for e in range(E):
        h[f"w_up_{e}"] = st.put(f"llm_w_up_{e}", "int8_t", data[f"w_up_{e}"])
        h[f"w_down_{e}"] = st.put(f"llm_w_down_{e}", "int8_t", data[f"w_down_{e}"])
    # goldens, one per checked stage
    for nm in ("norm1_golden", "rope_q_golden", "attn_golden", "proj_o_golden",
               "resid1_golden", "norm2_golden", "ffn_golden", "ffn_up_golden", "ladder_out_golden",
               "layer_golden", "proj_q_golden", "proj_k_golden", "proj_v_golden"):
        h[nm] = st.put(f"llm_{nm}", "uint16_t", u16(data[nm]))
    h["softmax_golden"] = st.put("llm_softmax_golden", "uint32_t",
                                 data["softmax_golden"].view(np.uint32))
    return h
