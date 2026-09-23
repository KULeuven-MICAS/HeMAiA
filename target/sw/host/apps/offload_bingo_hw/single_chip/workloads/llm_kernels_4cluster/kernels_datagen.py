#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Operands and goldens for the LLM layer's kernel unit tests.

ONE GOLDEN PER KERNEL, at the layer's own toy shape. Each golden describes what THAT
kernel does and nothing else, so a failure names the kernel rather than the layer. The
layer's own datagen chains them; this one deliberately does not.

Every golden mirrors the device's arithmetic including the narrowings, because a golden
that is more accurate than the hardware fails by a margin that reads like a kernel bug.
"""

import math
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
import _usg_paths  # noqa: F401,E402
from layout_convert import row_major_to_a, row_major_to_b, row_major_to_d  # noqa: E402
from sim_golden_models import int32_to_fp16_golden  # noqa: E402


def f32bits(x):
    return int(np.asarray(x, dtype=np.float32).view(np.uint32))


def _to_fp16(x_i32):
    """The GEMM D-port's int32 -> fp16 narrowing, bit for bit."""
    flat = np.asarray(x_i32, dtype=np.int64).reshape(-1)
    return np.array([int32_to_fp16_golden(int(v)) for v in flat],
                    dtype=np.uint16).view(np.float16).reshape(np.shape(x_i32))


def generate(p):
    """Every operand and golden, keyed by name. Shapes in ELEMENTS."""
    T, d, h = p["tokens"], p["d_model"], p["d_hidden"]
    mr, ts, mc = p["meshRow"], p["tileSize"], p["meshCol"]
    rng = np.random.default_rng(seed=11)
    o = {}

    # ---- 1. RMSNorm -------------------------------------------------------------------
    # Small integers so the fp32 reference and the device's integer sqrt/reciprocal agree
    # within a tolerance that still means something.
    x = (rng.integers(-8, 9, size=(T, d)).astype(np.float32) / 4.0).astype(np.float16)
    o["rms_x"] = x
    xf = x.astype(np.float32)
    rms = np.sqrt((xf * xf).mean(axis=-1, keepdims=True) + 1e-6)
    o["rms_golden"] = (xf / rms).astype(np.float16)

    # ---- 2. RoPE ----------------------------------------------------------------------
    # out = x*cos + swap(x)*sin_signed, with the sign folded into the table because the
    # kernel builds `swap` with an iDMA adjacent-pair swap and has no negate.
    rx = (rng.integers(-8, 9, size=(T, d)).astype(np.float32) / 8.0).astype(np.float16)
    pos = np.arange(T)[:, None]
    inv = (1.0 / (10000.0 ** (np.arange(0, d, 2) / d)))[None, :]
    ang = pos * inv
    cos_full = np.repeat(np.cos(ang), 2, axis=1).astype(np.float16)
    sin_signed = (np.repeat(np.sin(ang), 2, axis=1)
                  * np.tile([-1.0, 1.0], d // 2)[None, :]).astype(np.float16)
    o["rope_x"], o["rope_cos"], o["rope_sin"] = rx, cos_full, sin_signed
    swapped = rx.astype(np.float32).reshape(T, -1, 2)[:, :, ::-1].reshape(T, d)
    o["rope_golden"] = (rx.astype(np.float32) * cos_full.astype(np.float32)
                        + swapped * sin_signed.astype(np.float32)).astype(np.float16)

    # ---- 3. Quantise ------------------------------------------------------------------
    # inv_scale = 1.0 so the compare is BYTE-EXACT: any other scale makes the check a
    # tolerance question and stops it being a test of the kernel's rounding.
    qx = rng.integers(-100, 101, size=(T, d)).astype(np.float16)
    qx2 = qx.copy()
    qx2[:, 0] = np.float16(300.0)      # -> saturates to +127
    qx2[:, 1] = np.float16(-300.0)     # -> saturates to -127
    o["quant_x"] = qx2
    o["quant_scale_bits"] = f32bits(1.0)
    prod = np.clip(qx2.astype(np.float32), -128.0, 128.0)
    o["quant_golden"] = np.clip(np.rint(prod.astype(np.float64)), -127, 127).astype(np.int8)

    # ---- 4. Elementwise add (the residual) --------------------------------------------
    # Integer-valued operands whose sum is exact in fp16, so this too is byte-exact.
    a = rng.integers(-40, 41, size=(T, d)).astype(np.float16)
    b = rng.integers(-40, 41, size=(T, d)).astype(np.float16)
    o["add_a"], o["add_b"] = a, b
    o["add_golden"] = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16)

    # ---- 5. GEMM (the projection) ------------------------------------------------------
    ga = rng.integers(-4, 4, size=(T, d), dtype=np.int8)
    gb = rng.integers(-2, 2, size=(d, d), dtype=np.int8)
    o["gemm_a"] = row_major_to_a(ga, T // mr, d // ts, mr, ts)
    o["gemm_b"] = row_major_to_b(gb, d // ts, d // mc, ts, mc)
    gd = _to_fp16(ga.astype(np.int32) @ gb.astype(np.int32))
    o["gemm_golden"] = row_major_to_d(gd, T // mr, d // mc, mr, mc)

    # ---- 6. The two reshapes the layer needs -------------------------------------------
    # D -> row_major and row_major -> A, both at FP16. They are separate tests because they are
    # separate derived nests, and a nest that is wrong moves the right NUMBER of bytes to
    # the wrong offsets -- which a byte count would not catch, but a byte-exact compare
    # against a permuted golden does.
    r = (rng.integers(-64, 65, size=(T, d)).astype(np.float16))
    o["rs_d_src"] = row_major_to_d(r, T // mr, d // mc, mr, mc)   # D-layout input
    o["rs_d2p_golden"] = r                                        # row-major output
    o["rs_p_src"] = r                                             # row-major input
    o["rs_p2a_golden"] = row_major_to_a(r, T // mr, d // ts, mr, ts)
    return o


def stage(st, g, p):
    """Hand every array to DataStaging; return the handles the DFG names."""
    u16 = lambda a: np.ascontiguousarray(a).astype(np.float16).view(np.uint16)  # noqa: E731
    h = {}
    for nm in ("rms_x", "rms_golden", "rope_x", "rope_cos", "rope_sin", "rope_golden",
               "quant_x", "add_a", "add_b", "add_golden",
               "rs_d_src", "rs_d2p_golden", "rs_p_src", "rs_p2a_golden"):
        h[nm] = st.put(f"k_{nm}", "uint16_t", u16(g[nm]))
    h["gemm_golden"] = st.put("k_gemm_golden", "uint16_t", u16(g["gemm_golden"]))
    h["quant_golden"] = st.put("k_quant_golden", "int8_t", g["quant_golden"])
    for nm in ("gemm_a", "gemm_b"):
        h[nm] = st.put(f"k_{nm}", "int8_t", g[nm])
    return h
