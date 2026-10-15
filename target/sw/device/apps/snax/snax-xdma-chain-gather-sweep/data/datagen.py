#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the cross-die ChainGather SCALING SWEEP on the 16-chiplet platform.
#
# One binary measures the in-fabric fold at several chain widths P. The chiplet sitting at
# snake index i owns partial i (see the snake order in the app source), so the participants
# of a width-P round are exactly the first P partials and the golden for that round is the
# fold over rows [0 .. P-1]. That prefix property is why a single data set serves every P.
#
# Two folds are emitted per P, matching the two junctions the app exercises:
#
#   LINEAR    ElementwiseJunction, per-element FP32 ADD.
#             golden_lin[P] = sum(data[0:P], axis=0)
#             Values are small integers so the fp32 sum is EXACT and the on-device check
#             can be a byte-exact bit-pattern compare. An exact sum does not depend on the
#             order of the fold, so the chain and the tree share this golden.
#
#   SMS       SoftmaxStatsJunction, the online-softmax statistics merge
#             (m1,l1) (+) (m2,l2) = (max(m1,m2), l1*exp(a(m1-m*)) + l2*exp(a(m2-m*)))
#             A partial is one GROUP: the key beat m[0..31], then the value beat l[0..31],
#             FP16, lane r = row r. Rows 0 .. SMS_NVALID-1 are folded and the junction writes
#             the identity (key 0xFBFF, value 0) over the others, whatever the sources hold
#             there. CSR(1) carries a = SMS_SCALE.
#             The junction rounds l to FP16 at every hop, so the answer depends on the order
#             of the fold: golden_sms_chain[P] folds in the chain's order, golden_sms_tree[P]
#             in the tree's (each row into x=0, then the x=0 column into chip 0x00), both with
#             sms_fold() below, a bit-exact model of one hop. Both are byte-exact compares.

import argparse
import math
import pathlib
import hjson
import numpy as np

np.random.seed(320)

F16 = np.float16
F32 = np.float32
F64 = np.float64

SMS_ROWS = 32                       # FP16 lanes of a 64 B beat: the rows of one group
SMS_HALFS = 2 * SMS_ROWS            # a group: the key beat, then the value beat
SMS_NVALID = 30                     # rows folded; rows 30, 31 carry data the output must drop
SMS_SCALE = 1.0 / math.sqrt(128.0)  # a in exp(a (m - m*)), the softmax scale of a 128-wide head
SMS_ID_KEY = 0xFBFF                 # the lowest finite FP16: the identity key

# ---- SoftmaxStatsJunction, one hop, bit for bit ------------------------------------------------
#
# Every FP unit on the junction is a port of fpnew (IEEE binary32, round to nearest even,
# subnormals kept), so a numpy float32 add or multiply models one exactly, and fma32 models the
# fused multiply-add, which rounds once. Per row, with A the partial arriving from the previous
# hop and B this node's own (XDMADataSwitch: a = remote, b = local):
#
#   win     A if m_A >= m_B on the bit patterns of the widened FP32 keys (fp32aWins), else B;
#           equal keys keep A
#   delta   m_lose - m_win in FP32, forced to 0 if that is NaN (equal infinite keys)
#   alpha   FpActivation's EXP of delta * a (an FP32 multiply), table of EXP_LUT_N entries
#   l       fma(l_lose, alpha, l_win) in FP32, or l_win when l_lose is +0; narrowed to FP16
#   m       m_win, the FP16 key itself
#
# and every row at or above nValid leaves as the identity, key 0xFBFF and value 0.

# FpActivation's EXP as the junction builds it: the constants and the table are computed as the
# Scala generator computes them, a double rounded once to float.
EXP_LUT_N = 128
EXP_MAGIC = F32(12582912.0)                          # 1.5 * 2^23: float -> integer round
EXP_LOG2EF_N = F32(1.44269504088896341 * EXP_LUT_N)
EXP_LN2_N = F32(0.6931471805599453 / EXP_LUT_N)
EXP_HI = F32(88.3762626647949)
EXP_LO = F32(-88.3762626647949)
EXP_LUT = np.array([F32(math.pow(2.0, i / EXP_LUT_N)) for i in range(EXP_LUT_N)], dtype=F32)


def fma32(a, b, c):
    """a * b + c in binary32 with ONE rounding to nearest even, elementwise.

    The product is exact in float64 (24 x 24 significant bits) and TwoSum gives the float64
    sum's rounding error e, so s + e == a b + c exactly. Rounding s to float32 is then correct
    unless s sits on a float32 midpoint the exact sum is not on; e says which way.
    """
    a64 = np.asarray(a, dtype=F32).astype(F64)
    b64 = np.asarray(b, dtype=F32).astype(F64)
    c64 = np.asarray(c, dtype=F32).astype(F64)
    with np.errstate(over="ignore", invalid="ignore"):
        p = a64 * b64
        s = p + c64
        bb = s - p
        e = (p - (s - bb)) + (c64 - bb)
        r = s.astype(F32)
        up = np.nextafter(r, F32(np.inf))
        dn = np.nextafter(r, F32(-np.inf))
        other = np.where(r.astype(F64) > s, dn, up)
        at_mid = np.isfinite(s) & np.isfinite(r) & (e != 0) & \
            (s == (r.astype(F64) + other.astype(F64)) * 0.5)
    fixed = np.where(e > 0, np.maximum(r, other), np.minimum(r, other))
    return np.where(at_mid, fixed, r).astype(F32)


def _keep_a(a, b):
    """FpHelpers.fp32aWins: a >= b, compared on the FP32 bit patterns (finite operands)."""
    ua = np.asarray(a, dtype=F32).view(np.uint32)
    ub = np.asarray(b, dtype=F32).view(np.uint32)
    sa, sb = ua >> 31, ub >> 31
    return np.where(sa != sb, sa == 0, np.where(sa == 1, ub >= ua, ua >= ub))


def junction_exp(t):
    """FpActivation's EXP on FP32 inputs, FP32 out: clamp to +-HI, split t log2(e) N into
    i + frac with the magic round, then 2^((i mod N) / N) * fma(frac, ln2 / N, 1) * 2^(i div N),
    the power of two flushed to 0 below 2^-127."""
    t = np.ascontiguousarray(np.asarray(t, dtype=F32))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        lo = np.full_like(t, EXP_LO)
        hi = np.full_like(t, EXP_HI)
        clamped = np.where(_keep_a(hi, t), t, hi).astype(F32)            # fp32min(t, HI)
        pre = np.where(_keep_a(clamped, lo), clamped, lo).astype(F32)    # fp32max(_, LO)
        s0 = fma32(pre, EXP_LOG2EF_N, F32(0.0))
        r_m = (s0 + EXP_MAGIC).astype(F32)
        frac = (s0 - (r_m - EXP_MAGIC).astype(F32)).astype(F32)
        i_m = r_m.view(np.int32).astype(np.int64) - 0x4B400000
        interp = fma32(frac, EXP_LN2_N, F32(1.0))
        r3 = (EXP_LUT[i_m & (EXP_LUT_N - 1)] * interp).astype(F32)
        n_e = i_m >> int(math.log2(EXP_LUT_N))
        pow2 = np.where(n_e < -127, 0, ((n_e + 127) & 0xFF) << 23).astype(np.uint32).view(F32)
        return (r3 * pow2).astype(F32)


def sms_fold(a, b, n_valid=SMS_NVALID, scale=SMS_SCALE):
    """One hop of SoftmaxStatsJunction. `a` (the arriving partial) and `b` (the local one) are
    groups of SMS_HALFS FP16 bit patterns; returns the output group the same way."""
    a = np.asarray(a, dtype=np.uint16)
    b = np.asarray(b, dtype=np.uint16)
    ka, la = a[:SMS_ROWS].view(F16).astype(F32), a[SMS_ROWS:].view(F16).astype(F32)
    kb, lb = b[:SMS_ROWS].view(F16).astype(F32), b[SMS_ROWS:].view(F16).astype(F32)
    a_wins = _keep_a(ka, kb)
    with np.errstate(over="ignore", invalid="ignore"):
        d0 = (np.where(a_wins, kb, ka) - np.where(a_wins, ka, kb)).astype(F32)
        delta = np.where(np.isnan(d0), F32(0.0), d0).astype(F32)
        alpha = junction_exp((delta * F32(scale)).astype(F32))
        lose = np.where(a_wins, lb, la).astype(F32)
        win = np.where(a_wins, la, lb).astype(F32)
        l_out = np.where(lose.view(np.uint32) == 0, win, fma32(lose, alpha, win)).astype(F32)
        l16 = l_out.astype(F16).view(np.uint16)
    live = np.arange(SMS_ROWS) < n_valid
    out = np.empty(SMS_HALFS, dtype=np.uint16)
    out[:SMS_ROWS] = np.where(live, np.where(a_wins, a[:SMS_ROWS], b[:SMS_ROWS]), SMS_ID_KEY)
    out[SMS_ROWS:] = np.where(live, l16, 0)
    return out


def fold_chain(parts, **kw):
    """A ChainGather over `parts` in DATA order: the head forwards parts[0] unfolded, and every
    later node folds what arrives (operand A) with its own partial (operand B)."""
    acc = parts[0]
    for p in parts[1:]:
        acc = sms_fold(acc, p, **kw)
    return acc


def snake_index(x, y, grid_x):
    """The app's boustrophedon: row y runs west-to-east when y is even, east-to-west when odd."""
    return y * grid_x + (x if y % 2 == 0 else grid_x - 1 - x)


def tree_rows(width, grid_x, grid_y):
    """How many FULL rows a width-P tree spans, or 0 if this width has no tree (as the app)."""
    if grid_x <= 1 or width % grid_x != 0:
        return 0
    ry = width // grid_x
    return ry if 2 <= ry <= grid_y else 0


def sms_partials(num_chiplets):
    """One group per chiplet, as FP16 bit patterns. Every row is random except a few corners:

      row 0  the same key everywhere: a tie at every hop, l a plain FP32 sum per hop
      row 1  partial 1 (in every width) far ahead: alpha is exactly 0 for every other l
      row 2  l = 0 on every odd partial: the zero-loser bypass
      row 3  keys rising toward the collector: the arriving partial loses at every hop
      row 4  keys falling toward the collector: the arriving partial wins at every hop
    """
    rng = np.random.RandomState(321)
    keys = rng.normal(0.0, 12.0, size=(num_chiplets, SMS_ROWS))
    vals = rng.uniform(1.0, 600.0, size=(num_chiplets, SMS_ROWS))
    idx = np.arange(num_chiplets)
    keys[:, 0] = 7.25
    keys[1, 1] = 1500.0
    vals[idx % 2 == 1, 2] = 0.0
    keys[:, 3] = 4.0 * (num_chiplets - idx)
    keys[:, 4] = 4.0 * idx - 20.0
    groups = np.empty((num_chiplets, SMS_HALFS), dtype=np.uint16)
    groups[:, :SMS_ROWS] = keys.astype(F16).view(np.uint16)
    groups[:, SMS_ROWS:] = vals.astype(F16).view(np.uint16)
    return groups


def _f(v):
    # repr(float) always carries a decimal point or exponent, so "3.0f" and never the
    # invalid "3f", and enough digits to round-trip the exact fp32 value.
    return repr(float(v)) + "f"


def emit_header_file(**kwargs):
    num_chiplets = kwargs["num_chiplets"]
    grid_x = kwargs["num_chiplets_x"]
    partial_elems = kwargs["partial_elems"]
    sweep_p = list(kwargs["sweep_p"])

    assert num_chiplets % grid_x == 0, \
        f"num_chiplets ({num_chiplets}) is not a whole number of rows of {grid_x}"
    grid_y = num_chiplets // grid_x
    assert 1 <= SMS_NVALID <= SMS_ROWS, f"SMS_NVALID ({SMS_NVALID}) outside [1, {SMS_ROWS}]"
    for p in sweep_p:
        assert 2 <= p <= num_chiplets, f"sweep width {p} outside [2, {num_chiplets}]"

    # ---- LINEAR: small integer-valued fp32 partials -> exact fp32 sums.
    data = np.random.randint(0, 64, size=(num_chiplets, partial_elems)).astype(np.float32)
    golden_lin = np.zeros((len(sweep_p), partial_elems), dtype=np.float32)
    for i, p in enumerate(sweep_p):
        golden_lin[i] = np.sum(data[:p], axis=0).astype(np.float32)

    # ---- SMS: one group per chiplet, folded in each shape's own order.
    sms = sms_partials(num_chiplets)
    golden_chain = np.zeros((len(sweep_p), SMS_HALFS), dtype=np.uint16)
    golden_tree = np.zeros((len(sweep_p), SMS_HALFS), dtype=np.uint16)
    for i, p in enumerate(sweep_p):
        # chain: the far source is the head, the collector (snake index 0) folds last
        golden_chain[i] = fold_chain([sms[s] for s in range(p - 1, -1, -1)])
        # tree: each row walks x downwards into (0, y), then the column walks y downwards
        # into (0, 0). Widths with no tree keep a zero group, which the app never reads.
        ry = tree_rows(p, grid_x, grid_y)
        if ry:
            rows = [fold_chain([sms[snake_index(x, y, grid_x)] for x in range(grid_x - 1, -1, -1)])
                    for y in range(ry)]
            golden_tree[i] = fold_chain(rows[::-1])

    scale_bits = int(np.array(SMS_SCALE, dtype=F32).view(np.uint32))

    def fvec(name, arr):
        vals = ", ".join(_f(v) for v in arr.reshape(-1))
        return f"float {name}[{arr.size}] = {{{vals}}};"

    def hvec(name, arr, attr=""):
        vals = ", ".join(f"0x{int(v):04x}" for v in arr.reshape(-1))
        return f"uint16_t {name}[{arr.size}]{attr} = {{{vals}}};"

    def ivec(name, vals):
        body = ", ".join(str(int(v)) for v in vals)
        return f"static const int {name}[{len(vals)}] = {{{body}}};"

    lines = [
        "// Generated by datagen.py -- do not edit.",
        "#include <stdint.h>",
        f"#define NUM_CHIPLETS {num_chiplets}",
        f"#define NUM_CHIPLETS_X {grid_x}",
        f"#define PARTIAL_ELEMS {partial_elems}",
        f"#define NUM_SWEEP_P {len(sweep_p)}",
        f"#define SMS_ROWS {SMS_ROWS}",
        f"#define SMS_NVALID {SMS_NVALID}",
        f"#define SMS_SCALE_BITS 0x{scale_bits:08x}u  // {float(F32(SMS_SCALE))!r}",
        ivec("SWEEP_P", sweep_p),
        fvec("chain_gather_data", data),
        fvec("chain_gather_golden_lin", golden_lin),
        hvec("chain_gather_sms_data", sms, " __attribute__((aligned(64)))"),
        hvec("chain_gather_golden_sms_chain", golden_chain),
        hvec("chain_gather_golden_sms_tree", golden_tree),
    ]
    return "\n\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generating data for the ChainGather scaling sweep")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True,
                        help="Select param config file")
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
