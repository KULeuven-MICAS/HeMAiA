#!/usr/bin/env python3
"""
Golden data generator for attn_test_softmax.
Tests __host_bingo_kernel_fp32_softmax in isolation.

The HW softmax uses a Cephes polynomial approximation of exp() with FMA,
ordered reduction via vfredosum, and multiplication by 1/sum (not direct
division). To produce bit-identical goldens, we mirror the HW algorithm
exactly in Python using math.fma and careful fp32 arithmetic.
"""

import numpy as np
import struct
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

np.random.seed(42)


def _f32(x):
    """Coerce to Python float that round-trips through fp32."""
    return float(np.float32(x))


def _fma(a, b, c):
    """Bit-exact FMA: a*b + c with single rounding to fp32.

    Trick: fp32 inputs fit exactly in fp64 (Python's native float).
    fp32 * fp32 product needs at most 48 mantissa bits, which fits in fp64's
    52-bit mantissa exactly. fp64 + fp64 of these is also exact here.
    Final rounding to fp32 gives the same result as true FMA (single rounding).
    """
    a64 = _f32(a)  # Python float = fp64
    b64 = _f32(b)
    c64 = _f32(c)
    # Intermediate a*b + c computed in fp64 (exact for fp32 inputs)
    return _f32(a64 * b64 + c64)


def _bits_to_f32(bits):
    """Reinterpret 32-bit integer as float32."""
    return struct.unpack('<f', struct.pack('<I', np.uint32(bits) & 0xFFFFFFFF))[0]


# HW exp constants (must be fp32-rounded)
_EXP_HI = _f32(88.3762626647949)
_EXP_LO = _f32(-88.3762626647949)
_LOG2EF = _f32(1.44269504088896341)
_C1 = _f32(0.693359375)
_C2 = _f32(-2.12194440e-4)
_HALF = _f32(0.5)
_ONE = _f32(1.0)
_POLY = [
    _f32(1.9875691500e-4),
    _f32(1.3981999507e-3),
    _f32(8.3334519073e-3),
    _f32(4.1665795894e-2),
    _f32(1.6666665459e-1),
    _f32(5.0000001201e-1),
]


def bingo_exp_f32(x_val):
    """Scalar version of HW __bingo_exp_f32 — bit-exact mirror of the
    RVV Cephes polynomial approximation used in the HW softmax kernel.
    """
    # Clamp
    x = _f32(min(max(x_val, _EXP_LO), _EXP_HI))

    # fx = half + x * LOG2EF  (FMA: half + x*LOG2EF, single rounding)
    # vfmacc(half, x, LOG2EF) = half + x*LOG2EF
    fx = _fma(x, _LOG2EF, _HALF)
    # tmp = (int32)fx  — cast truncates toward zero
    tmp = int(np.int32(fx) if fx >= 0 else -np.int32(-fx))
    # Safer: use numpy truncation directly
    tmp = int(np.float32(fx).astype(np.int32))
    ftmp = _f32(tmp)
    # Floor correction: if fx < ftmp, ftmp -= 1
    if fx < ftmp:
        ftmp = _f32(ftmp - _ONE)
    tmp = int(np.float32(ftmp).astype(np.int32))

    # x = x - ftmp*C1 (plain mul+sub, fp32 rounding at each step)
    x = _f32(x - _f32(ftmp * _C1))
    # x = x - ftmp*C2
    x = _f32(x - _f32(ftmp * _C2))

    z = _f32(x * x)

    # Polynomial (Horner with FMA): y = y*x + coef
    y = _POLY[0]
    y = _fma(y, x, _POLY[1])
    y = _fma(y, x, _POLY[2])
    y = _fma(y, x, _POLY[3])
    y = _fma(y, x, _POLY[4])
    y = _fma(y, x, _POLY[5])
    # Final step: y = (x + 1) + y * z  (vfmacc: vd = vd + vs1*vs2 with vd=x+1)
    # Equivalent to fma(y, z, x+1)
    y = _fma(y, z, _f32(x + _ONE))

    # Scale by 2^n via integer bit manipulation
    shift_bits = (tmp + 127) << 23
    shift_f = _bits_to_f32(shift_bits)
    y = _f32(y * shift_f)
    return y


# HeMAiA Ara configuration: VLEN=128 bits, NrLanes=2.
# For fp32 elements, VLMAX = VLEN / 32 = 4 elements per vector.
# Reduction loop processes `avl` elements in chunks of VLMAX.
_VLMAX_F32 = 4


def bingo_softmax_row(x_row):
    """Mirror HW __host_bingo_kernel_fp32_softmax for a single row.

    Ara has VLMAX=4 fp32 elements. The reduction loop chunks through
    row_length in groups of 4, doing vfredosum per chunk with init=0,
    then adding chunk sum to scalar accumulator.

    Algorithm (matches host_kernel_lib.h:215-275):
      1. max_val = vfredmax reduction over x_row
      2. For each chunk of 4 elements:
           expv = bingo_exp_f32(x_row - max_val)
           rsum = vfredosum(expv, init=0)  [(((0 + e0) + e1) + e2) + e3]
           sum += rsum
      3. inv_sum = 1.0f / sum
      4. out = expv * inv_sum  (per chunk, but same as elementwise here)
    """
    row_length = len(x_row)

    # Pass 1: max reduction. vfredmax is unordered but max is associative.
    max_val = _f32(x_row[0])
    for v in x_row[1:]:
        vv = _f32(v)
        if vv > max_val:
            max_val = vv

    # Pass 2: exp(x - max) per element + chunked ordered reduction
    expv = np.empty(row_length, dtype=np.float32)
    for i in range(row_length):
        shifted = _f32(_f32(x_row[i]) - max_val)
        expv[i] = bingo_exp_f32(shifted)

    # Chunked ordered reduction matching HW: vl=VLMAX per chunk
    sum_val = _f32(0.0)
    for chunk_start in range(0, row_length, _VLMAX_F32):
        chunk_end = min(chunk_start + _VLMAX_F32, row_length)
        # vfredosum: (((init + chunk[0]) + chunk[1]) + chunk[2]) + chunk[3]
        rsum = _f32(0.0)
        for i in range(chunk_start, chunk_end):
            rsum = _f32(rsum + _f32(expv[i]))
        # sum += rsum (scalar accumulation across chunks)
        sum_val = _f32(sum_val + rsum)

    # Pass 3: multiply by 1/sum (NOT direct division — matches HW `inv_sum * expv`)
    inv_sum = _f32(_ONE / sum_val)
    out = np.empty(row_length, dtype=np.float32)
    for i in range(row_length):
        out[i] = _f32(_f32(expv[i]) * inv_sum)
    return out


def emit_header_file(**kwargs):
    lines = ["#include <stdint.h>"]

    seq_len = kwargs["seq_len"]
    num_rows = seq_len
    row_length = seq_len

    # Generate random FP32 input simulating QK_scaled: shape (seq_len, seq_len)
    x = np.random.uniform(-2.0, 2.0, size=(num_rows, row_length)).astype(np.float32)

    # Compute softmax row-wise using HW-exact algorithm
    softmax_out = np.empty_like(x, dtype=np.float32)
    for r in range(num_rows):
        softmax_out[r] = bingo_softmax_row(x[r])

    lines.append(f"// Softmax test: {num_rows} rows x {row_length} cols")
    lines.append(f"// Goldens computed via HW-exact Cephes exp polynomial + ordered sum + mul-by-reciprocal.")
    lines.append(format_vector_definition("float", "softmax_input", x.reshape(-1)))
    lines.append(format_vector_definition("float", "golden_softmax_output", softmax_out.reshape(-1)))
    lines.append(format_scalar_definition("uint32_t", "num_rows", num_rows))
    lines.append(format_scalar_definition("uint32_t", "row_length", row_length))

    return "\n\n".join(lines) + "\n"
