#!/usr/bin/env python3
"""Operands and goldens for one MoE feed-forward layer.

Everything is generated ROW-MAJOR and converted to the array's blocked layouts with
util/sim/xdma/layout_convert.py, rather than being generated blocked. The old version
generated B directly as ``(K, N, tileSize, meshCol)``, which is not B-layout -- B is
``[n][k][c][s]``, a different permutation with the same flat length. Nothing caught it
because the data was random and the same flat buffer went to the golden model, so both
sides were wrong identically. The moment a real weight matrix is involved that stops
being true, which is exactly what an FFN is.

The golden chain mirrors the device stage for stage, including the two narrowings that
lose information (int32 -> fp16 after each GEMM, fp16 -> int8 before the down
projection). Skipping either would make the golden more accurate than the hardware and
every check would fail by a margin that looks like a bug.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
import _usg_paths  # noqa: F401,E402  (registers util/sim/{common,gemm,xdma,ara} on sys.path)
from layout_convert import row_major_to_a, row_major_to_b, row_major_to_d  # noqa: E402
from sim_golden_models import int32_to_fp16_golden  # noqa: E402

def int8_scale_for(peak):
    """The largest power of two that keeps `peak` inside the int8 range.

    Derived from the data rather than fixed. The baked 16.0 the *_f16_i8 kernels use is
    sized for activations around [-8, 8]; a SwiGLU is the PRODUCT of two GEMM outputs, so
    here it reaches the thousands and 16.0 saturated 83% of the tensor -- a chain whose
    every value is +-127 agrees with its golden no matter what the SwiGLU or the reshape
    did. A power of two keeps the scaling exact in fp16, so it adds no error of its own.

    This workload drives the standalone simd_fp16_to_int8 kernel, whose inv_scale is an
    ordinary argument, so it is free to choose.
    """
    import math
    if peak <= 0:
        return 1.0
    return float(2.0 ** math.floor(math.log2(127.0 / peak)))


def f32bits(x):
    return int(np.asarray(x, dtype=np.float32).view(np.uint32))


def _to_fp16(x_i32):
    """The GEMM's D-port int32 -> fp16 narrowing, bit for bit as the hardware does it."""
    flat = np.asarray(x_i32, dtype=np.int64).reshape(-1)
    return np.array([int32_to_fp16_golden(int(v)) for v in flat],
                    dtype=np.uint16).view(np.float16).reshape(np.shape(x_i32))


def _swiglu(gate_f16, up_f16):
    """out = silu(gate) * up. FP32 internal, FP16 in and out -- the SIMD's own contract.

    The sigmoid is written branch-wise rather than as 1/(1+exp(-x)): a GEMM over 64 int8
    terms reaches the low thousands, and exp(+2000) overflows fp32 long before silu does
    anything interesting. The two branches are algebraically the same function and agree
    to the last bit where both are finite.
    """
    g = gate_f16.astype(np.float32)
    sig = np.where(g >= 0.0, 1.0 / (1.0 + np.exp(-np.abs(g))),
                   np.exp(-np.abs(g)) / (1.0 + np.exp(-np.abs(g))))
    sg = (g * sig).astype(np.float16)
    return (sg.astype(np.float32) * up_f16.astype(np.float32)).astype(np.float16)


def _quantise_i8(x_f16, scale):
    """round-to-nearest then saturate, which is what the Fp16ToInt8 leaf does."""
    q = np.rint(x_f16.astype(np.float32) * scale)
    return np.clip(q, -128, 127).astype(np.int8)


def generate_moe_data(p):
    """Every array this workload needs, keyed by name. Shapes are in ELEMENTS.

    p carries tokens / d_model / d_hidden / num_experts / top_k and the array's
    (meshRow, tileSize, meshCol).
    """
    T, d, h = p["tokens"], p["d_model"], p["d_hidden"]
    E, k = p["num_experts"], p["top_k"]
    mr, ts, mc = p["meshRow"], p["tileSize"], p["meshCol"]
    rng = np.random.default_rng(seed=42)
    out = {}

    # ---- the layer's input, shared by every expert ------------------------------------
    # Small operand ranges on purpose. This layer narrows to FP16 twice and to INT8 once,
    # and two int8 GEMMs back to back over d then h reach ~1e6 with full-range operands --
    # past fp16's 65504, where the golden and the device would agree only on inf.
    x = rng.integers(-4, 4, size=(T, d), dtype=np.int8)
    out["x_a"] = row_major_to_a(x, T // mr, d // ts, mr, ts)

    # ---- the router ------------------------------------------------------------------
    # Fixed logits rather than random ones: which experts win has to be known at
    # generation time, or the per-expert goldens cannot be computed at all. These make
    # experts 3 and 1 the top two, so the winners are neither the first two nor
    # adjacent -- a combine that quietly folded slots 0..k-1 would pass on an easier set.
    logits = np.array([-1.5, 0.3, -0.8, 2.1], dtype=np.float32)[:E]
    if E > 4:
        logits = np.concatenate([logits, rng.uniform(-2, 2, E - 4).astype(np.float32)])
    out["router_logits"] = logits

    ex = np.exp(logits - logits.max())
    prob = (ex / ex.sum()).astype(np.float32)
    out["softmax_golden"] = prob

    winners = np.argsort(-prob, kind="stable")[:k]
    weight = np.zeros(E, dtype=np.float32)
    weight[winners] = prob[winners] / prob[winners].sum()
    out["weight_golden"] = weight
    out["winners"] = winners

    # ---- each expert's FFN -------------------------------------------------------------
    # Two passes: the activations first, because the int8 scale is chosen from how large
    # they actually get, and only then the projection that consumes them.
    W, act = {}, {}
    for e in range(E):
        W[e] = (rng.integers(-2, 2, size=(d, h), dtype=np.int8),
                rng.integers(-2, 2, size=(d, h), dtype=np.int8),
                rng.integers(-2, 2, size=(h, d), dtype=np.int8))
        up = _to_fp16(x.astype(np.int32) @ W[e][0].astype(np.int32))
        gate = _to_fp16(x.astype(np.int32) @ W[e][1].astype(np.int32))
        act[e] = _swiglu(gate, up)                    # fp16 [T, h]

    peak = max(float(np.abs(a.astype(np.float32)).max()) for a in act.values())
    scale = int8_scale_for(peak)
    out["int8_scale"] = scale
    out["int8_scale_f32bits"] = f32bits(scale)

    combined = np.zeros((T, d), dtype=np.float32)
    for e in range(E):
        w_up, w_gate, w_down = W[e]
        out[f"w_up_{e}"] = row_major_to_b(w_up, d // ts, h // mc, ts, mc)
        out[f"w_gate_{e}"] = row_major_to_b(w_gate, d // ts, h // mc, ts, mc)
        out[f"w_down_{e}"] = row_major_to_b(w_down, h // ts, d // mc, ts, mc)

        act_i8 = _quantise_i8(act[e], scale)          # int8 [T, h]
        # NO dequantisation here. The device's down projection writes its int32
        # accumulator straight out as fp16 and nothing divides the scale back out, so a
        # golden that did would disagree with it by exactly that factor -- everywhere, by
        # a constant, which reads like a broken kernel rather than a bad golden.
        y = _to_fp16(act_i8.astype(np.int32) @ w_down.astype(np.int32))

        # Per-expert goldens in the layouts the device actually writes.
        out[f"swiglu_golden_{e}"] = row_major_to_d(act[e], T // mr, h // mc, mr, mc)
        out[f"y_golden_{e}"] = row_major_to_d(y, T // mr, d // mc, mr, mc)
        combined += weight[e] * y.astype(np.float32)

    out["out_golden"] = row_major_to_d(combined.astype(np.float16),
                                       T // mr, d // mc, mr, mc)

    # The whole landing area as the combine's operands should look: a winner's slot holds
    # its y, a loser's is still the zeros the workload memsets it with. Checking it in one
    # piece is what verifies the SKIP itself -- an expert that ran when the router did not
    # pick it shows up here as a non-zero slot, and nowhere else.
    out["ybuf_golden"] = np.concatenate([
        out[f"y_golden_{e}"] if e in set(winners.tolist())
        else np.zeros_like(out[f"y_golden_{e}"])
        for e in range(E)])
    return out


def stage(st, data, p):
    """Hand every array to DataStaging and return the handles the DFG names.

    DataStaging decides between the host image and the memory chiplet from the platform,
    which the hand-rolled `.wide_spm` emitter this replaced could not do.
    """
    E = p["num_experts"]
    h = {
        "x_a": st.put("moe_x_a", "int8_t", data["x_a"]),
        "logits": st.put("moe_router_logits", "uint32_t",
                         data["router_logits"].view(np.uint32)),
        "softmax_golden": st.put("moe_softmax_golden", "uint32_t",
                                 data["softmax_golden"].view(np.uint32)),
        "weight_golden": st.put("moe_weight_golden", "uint32_t",
                                data["weight_golden"].view(np.uint32)),
        "out_golden": st.put("moe_out_golden", "uint16_t",
                             data["out_golden"].view(np.uint16)),
        "ybuf_golden": st.put("moe_ybuf_golden", "uint16_t",
                              data["ybuf_golden"].view(np.uint16)),
        # Zeros for the landing area. put_zeros keeps them out of the image: on the host
        # path the array is declared and not defined, so it lands in .bss.
        "ybuf_zero": st.put_zeros("moe_ybuf_zero", "uint16_t",
                                  data["ybuf_golden"].size),
        # expert -> CERF group. The compiler fills this in; the gating kernel indexes it
        # by the router's score index, so it is one entry per EXPERT.
        "cerf_gids": None,
    }
    for e in range(E):
        for w in ("w_up", "w_gate", "w_down"):
            h[f"{w}_{e}"] = st.put(f"moe_{w}_{e}", "int8_t", data[f"{w}_{e}"])
        h[f"swiglu_golden_{e}"] = st.put(f"moe_swiglu_golden_{e}", "uint16_t",
                                         data[f"swiglu_golden_{e}"].view(np.uint16))
        h[f"y_golden_{e}"] = st.put(f"moe_y_golden_{e}", "uint16_t",
                                    data[f"y_golden_{e}"].view(np.uint16))
    return h
