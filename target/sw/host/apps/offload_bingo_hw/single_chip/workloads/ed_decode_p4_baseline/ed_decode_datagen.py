#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Data generator for E-d: the KV-sharded long-context DECODE softmax epilogue.
#
# One decode step attends a single query over a length-`SEQ_LEN` KV cache that is
# SHARDED across P clusters (S_p = SEQ_LEN/P keys per cluster). Each cluster computes
# its OWN partial softmax statistics on the SIMD datapath:
#     m_p = max_j s_pj                 (running max over its shard's scores)
#     l_p = sum_j exp(s_pj - m_p)      (shifted exp-sum)
# and the P partials are combined by the in-fabric moment-merge collective into the
# global (m*, l*). This is the flash-decoding cross-shard combine, and it is exactly
# the online-softmax monoid StreamMomentMergeRt folds in-transit.
#
# We measure the EPILOGUE only (SIMD softmax-stats + the collective), not QK^T/PV --
# those are GEMM and are deliberately out of scope. The scores s_pj are the workload
# input; think of them as the already-computed q.k_j for this decode step.
#
# The golden is computed from the FP16-ROUNDED scores (the SIMD datapath transports
# FP16), so the on-device (m*, l*) matches the reference to LUT-exp tolerance.
#
# TWO invariants this datagen MUST enforce, or the experiment is worthless:
#   1. The per-shard maxima m_p must genuinely DIFFER across shards. If they were equal
#      every cross-shard delta would be 0, exp(delta)=1, and the collective would
#      degenerate to a plain sum -- the nonlinearity that puts it beyond SHARP would
#      never be exercised. We inject a distinct PEAK per shard (PEAK_P[p]).
#   2. exp(s) must stay finite in the datapath. The device computes the SHIFTED sum
#      via l_p = exp(-m_p) * sum exp(s) with the SUM carried in FP32 (a_addr path), so
#      exp(s) itself is evaluated UNSHIFTED in FP16 -- bound the scores so max exp(s)
#      stays well under fp16 max (65504). SCORE_MAX <= ~10 keeps exp(s) <= ~2.2e4.

import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(os.path.join(ROOT_DIR, "util", "sim"))
from xdma_moment_merge_golden import moment_merge_golden, f32_bits  # noqa: E402

# ---- configuration (overridable via params.hjson) --------------------------------
SEQ_LEN = 2048          # total KV length attended this decode step
NUM_SHARDS = 2          # P clusters; must match params.hjson num_clusters
SCORE_MAX = 10.0        # scores clamped to [-inf, SCORE_MAX]; keeps unshifted exp(s) finite
SEED = 0xED

# The bulk of each shard's scores sits LOW (so the injected peak is unambiguously the max),
# and each shard gets a distinct injected PEAK so the cross-shard maxima differ by a clean
# margin regardless of the random draw (invariant #1). Without an injected peak the shard max
# is a random extreme (~base + 4.9 for S_p=1024), which collides across shards once it clamps
# at SCORE_MAX -- that made the min gap ~0.18 and would have made the fold near-trivial.
SCORE_MEAN = -1.0
SCORE_SIGMA = 1.0
PEAK_P = [4.0, 5.5, 7.0, 8.5]  # distinct by 1.5, all < SCORE_MAX; one per shard

# ABSOLUTE tol on (m*, l*). The host check is absolute (host_kernel_lib.h FP32_TOL), so this
# must scale with the checked magnitude. l* ~ 17 here; the chain has TWO LUT-exp hops (the
# on-device sum_exp, then the collective's exp(m_p - m*)), so budget ~0.5% -- comfortably under
# the moment-merge module's documented 1.5% LUT bound, but tight enough to catch a real error.
CHECK_TOLERANCE = 0.09


def _fp16(x):
    return np.float16(x)


def _gen_scores(rng, s_p, peak):
    """One shard's FP16 score vector: a low-mean bulk plus one injected `peak` at index 0,
    so the shard maximum is deterministically `peak` (invariant #1) and the rest sit well
    below it. Everything is clamped to <= SCORE_MAX so unshifted exp(s) stays finite."""
    s = rng.normal(SCORE_MEAN, SCORE_SIGMA, size=s_p).astype(np.float32)
    s = np.minimum(s, peak - 0.5)   # keep the bulk strictly below the injected peak
    s[0] = peak
    s = np.minimum(s, SCORE_MAX)
    return _fp16(s).astype(np.float16)


def _shard_stats_fp16(scores_f16):
    """Per-shard (m_p, l_p) computed the way the DEVICE does it, so the golden matches:
    m_p = max(scores)                     (FP16 in, exact max)
    l_p = exp(-m_p) * sum(exp(scores))    (exp evaluated UNSHIFTED, sum in FP32)
    The datapath widens FP16->FP32 internally, so we reduce in float64 here for a clean
    reference and rely on the LUT-exp tolerance to absorb the small format gap."""
    s = scores_f16.astype(np.float64)
    m_p = float(np.max(s))
    sum_exp = float(np.sum(np.exp(s)))         # unshifted, matches device step C
    l_p = float(np.exp(-m_p) * sum_exp)        # == sum(exp(s - m_p)), the shifted sum
    return m_p, l_p


def _build(seq_len=SEQ_LEN, num_shards=None, num_clusters=None, **_):
    # params.hjson provides `num_clusters` (the platform-guard key); the shard count IS
    # the cluster count. Fall back to num_shards / the module default only if neither given.
    if num_shards is None:
        num_shards = num_clusters if num_clusters is not None else NUM_SHARDS
    assert seq_len % num_shards == 0, "SEQ_LEN must divide evenly across shards"
    s_p = seq_len // num_shards
    assert s_p % 32 == 0, "per-shard length must be a whole number of 64B FP16 beats"
    assert num_shards <= len(PEAK_P), "add more PEAK_P levels for more shards"

    rng = np.random.default_rng(SEED)
    scores = [_gen_scores(rng, s_p, PEAK_P[p]) for p in range(num_shards)]
    stats = [_shard_stats_fp16(scores[p]) for p in range(num_shards)]
    m_vals = [st[0] for st in stats]
    l_vals = [st[1] for st in stats]

    # Invariant #1: the shard maxima must be distinct enough to exercise the fold.
    m_sorted = sorted(m_vals)
    min_gap = min(b - a for a, b in zip(m_sorted, m_sorted[1:]))
    assert min_gap > 0.5, f"shard maxima too close (min gap {min_gap:.3f}); fold would be ~trivial"

    # Invariant #2: unshifted exp(s) stays finite in fp16.
    assert max(m_vals) <= SCORE_MAX + 1e-6
    assert np.exp(SCORE_MAX) < 65504.0, "SCORE_MAX too large for unshifted fp16 exp"

    m_star, l_star = moment_merge_golden(m_vals, l_vals)
    return s_p, scores, m_vals, l_vals, m_star, l_star


def _u16_beats(name, scores_f16):
    """Emit one shard's scores as an fp16 array, zero-padded to a whole 64B beat count.
    (S_p is already a multiple of 32, so no padding is actually added -- the assert in
    _build guarantees it -- but keep the helper beat-aware for future shapes.)"""
    u16 = scores_f16.view(np.uint16)
    lines = [f"static uint16_t {name}[{len(u16)}] __attribute__((aligned(64))) = {{"]
    row = "    " + ", ".join(f"0x{v:04x}" for v in u16.tolist())
    lines.append(row + ",")
    lines.append("};")
    return lines


def _u32_array(name, vals):
    lines = [f"static uint32_t {name}[{len(vals)}] __attribute__((aligned(4))) = {{"]
    lines.append("    " + ", ".join(f"0x{v:08x}" for v in vals) + ",")
    lines.append("};")
    return lines


def emit_header_file(**kwargs):
    s_p, scores, m_vals, l_vals, m_star, l_star = _build(**kwargs)
    num_shards = len(scores)

    lines = [
        "// Auto-generated by ed_decode_datagen.py -- KV-sharded decode softmax epilogue.",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define ED_SEQ_LEN {s_p * num_shards}",
        f"#define ED_NUM_SHARDS {num_shards}",
        f"#define ED_SHARD_LEN {s_p}",
        f"#define ED_SHARD_BEATS {s_p // 32}",
        "",
    ]
    for p in range(num_shards):
        lines += _u16_beats(f"ed_scores_shard{p}", scores[p])
        lines.append("")
    # Golden global (m*, l*) for the on-device check.
    lines += _u32_array("ed_golden_pair", [f32_bits(m_star), f32_bits(l_star)])
    lines.append("")
    # Per-shard golden stats, for the Gate-2 prologue check.
    lines += _u32_array("ed_golden_m", [f32_bits(v) for v in m_vals])
    lines.append("")
    lines += _u32_array("ed_golden_l", [f32_bits(v) for v in l_vals])
    lines.append("")
    lines.append(f"// per-shard m_p = {[round(v, 4) for v in m_vals]}")
    lines.append(f"// per-shard l_p = {[round(v, 4) for v in l_vals]}")
    lines.append(f"// golden global (m*, l*) = ({m_star:.6f}, {l_star:.6f})")
    return "\n".join(lines) + "\n"


def _self_check():
    """Gate 0: no sim. Recompute the golden two independent ways and cross-check, and
    re-assert both invariants loudly."""
    s_p, scores, m_vals, l_vals, m_star, l_star = _build()
    # Independent recompute of the global softmax denominator directly from all scores,
    # then compare to the merge-of-partials -- they must agree (the whole point of the
    # collective is that per-shard-then-merge == global).
    all_s = np.concatenate([sc.astype(np.float64) for sc in scores])
    m_glob = float(np.max(all_s))
    l_glob = float(np.sum(np.exp(all_s - m_glob)))
    assert abs(m_glob - m_star) <= 1e-9, f"m* mismatch: {m_glob} vs {m_star}"
    rel = abs(l_glob - l_star) / abs(l_glob)
    assert rel <= 1e-6, f"l* merge-of-partials disagrees with global by {rel:.2e}"
    print(f"[Gate 0] self-check OK: S_p={s_p}, shards={len(scores)}")
    print(f"         m_p={[round(v, 3) for v in m_vals]}  (distinct)")
    print(f"         l_p={[round(v, 3) for v in l_vals]}")
    print(f"         golden (m*,l*)=({m_star:.6f}, {l_star:.6f}); "
          f"merge==global within {rel:.2e}")


if __name__ == "__main__":
    _self_check()
