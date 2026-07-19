#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Shared golden-reference for the F3 cross-cluster in-fabric collective
# (StreamMomentMergeRt): the online-softmax moment-merge monoid
#   (m,l) = ( max(m_a,m_b), l_win + l_los * exp(m_los - m_win) )
# Double-precision reference, mirroring the SAME formula used by the Chisel
# testers (hw/chisel/.../DataPathExtension/{MomentMergeTester,
# StreamMomentMergeRtTester}.scala, `local()`/`global()`) and the snax_cluster
# S0a/S0b smoke tests -- keep this the SINGLE source of truth for the golden
# so every tier (Tier-1 Chisel, S0a, S0b, S1/S2/S3 HeMAiA workloads) checks
# against numerically the same reference.
#
# Beat layout (StreamMomentMergeRt.scala): 16 FP32 lanes in a 512-bit (64B)
# beat; m_k = lane k, l_k = lane (8+k), for k in 0..7 (maxPairs=8). Lanes
# >= nValid are masked to the monoid identity by the RTL and are don't-care
# on input.
#
# Acceptance bound (matches the Chisel testers' exp-LUT-depth-128 tolerance):
#   m* : exact (bit-level max of already-materialized values, no rounding)
#   l* : relative error <= L_REL_TOL

import numpy as np

MAX_PAIRS = 8
L_REL_TOL = 1.5e-2


def f32_bits(x):
    """Python float/np.float64 -> the uint32 FP32 bit pattern."""
    return int(np.float32(x).view(np.uint32))


def bits_to_f32(bits):
    """uint32 FP32 bit pattern -> Python float."""
    return float(np.uint32(bits).view(np.float32))


def moment_merge_golden(m, l):
    """m, l: sequences of the SAME length (the live lanes only), double precision.
    Returns (m*, l*) as Python floats, computed exactly like the RTL's monoid:
    m* = max_i(m_i), l* = sum_i( l_i * exp(m_i - m*) )."""
    m = np.asarray(m, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    m_star = float(np.max(m))
    l_star = float(np.sum(l * np.exp(m - m_star)))
    return m_star, l_star


def pack_beat(m, l):
    """Pack up to MAX_PAIRS (m_k, l_k) FP32 pairs into one 16xuint32 (64B) beat
    array, in StreamMomentMergeRt's lane layout. len(m) == len(l) <= MAX_PAIRS;
    unused lanes are zeroed (don't-care, masked by nValid on the RTL side)."""
    n = len(m)
    assert n == len(l) and n <= MAX_PAIRS
    beat = np.zeros(2 * MAX_PAIRS, dtype=np.uint32)
    for k in range(n):
        beat[k] = f32_bits(m[k])
        beat[MAX_PAIRS + k] = f32_bits(l[k])
    return beat


def check_pair(m_hw_bits, l_hw_bits, m_gold, l_gold):
    """Check a hardware-measured (m*,l*) FP32-bit-pattern pair against the
    golden. Returns (m_ok, l_ok, rel_err_l)."""
    m_hw = bits_to_f32(m_hw_bits)
    l_hw = bits_to_f32(l_hw_bits)
    m_ok = (m_hw == m_gold) or abs(m_hw - m_gold) <= 1e-6 * max(1.0, abs(m_gold))
    rel_err_l = abs(l_hw - l_gold) / abs(l_gold) if l_gold != 0 else abs(l_hw)
    l_ok = rel_err_l <= L_REL_TOL
    return m_ok, l_ok, rel_err_l


if __name__ == "__main__":
    # Quick self-check.
    rng = np.random.default_rng(42)
    m = rng.uniform(-8.0, 8.0, size=5)
    l = rng.uniform(0.01, 4.0, size=5)
    m_star, l_star = moment_merge_golden(m, l)
    beat = pack_beat(m, l)
    print(f"m={m}\nl={l}\ngolden (m*,l*) = ({m_star:.6f}, {l_star:.6f})")
    print(f"packed beat (hex): {[hex(int(x)) for x in beat]}")
