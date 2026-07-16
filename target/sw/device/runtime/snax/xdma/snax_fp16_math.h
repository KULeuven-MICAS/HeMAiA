// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Integer-only FP16 scalar math for the cluster DM core.
//
// The SNAX cluster cores are rv32ima -- NO FPU. A plain-C float op (1.0f/x, sqrtf, a
// cast) compiles to an F/Zfh instruction that TRAPS at runtime. These helpers do the
// arithmetic as integer bit-ops on the fp16 pattern instead (rv32iM has a hardware
// `divu`, which is all the reciprocal / Newton isqrt need). Extracted from the xDMA
// offload kernels so any DM-core kernel can reuse them; they carry NO xDMA dependency.

#ifndef SNAX_FP16_MATH_H
#define SNAX_FP16_MATH_H

#include <stdint.h>

// Integer FP16 reciprocal (1/s) for a POSITIVE, normal fp16 `s`. The cluster DM
// core is rv32ima (NO FPU), so a float `1.0f/x` would trap -- but rv32iM has an
// integer divide, and the reciprocal significand is exactly one `divu`. For
// s = 2^(E-15) * (M/1024) with 11-bit significand M in [1024,2047]:
//   1/s = 2^(15-E) * (1024/M),  and 1024/M in (0.5,1]  -> normalize *2, exp-1:
//   1/s = 2^(15-E-1) * (2048/M),  2048/M in (1,2] is the result significand.
// q = round(2^21 / M) = 1024*(2048/M) gives the Q10 significand in (1024,2048];
// mantissa = q-1024, biased exp = 29-E; the q==2048 carry (M==1024, exact power
// of two) bumps the exponent and zeroes the mantissa. Domain: softmax's Sexp is
// always >= 1 (the max term is exp(0)=1) and <= ~D, so the result is a normal
// fp16 -- no zero/subnormal/inf/sign handling is needed.
static inline uint16_t recip_f16(uint16_t s)
{
    uint32_t E = (s >> 10) & 0x1Fu;             // biased exponent
    uint32_t M = 1024u + (s & 0x3FFu);          // 1.mmm significand in Q10, [1024,2047]
    uint32_t q = ((1u << 21) + (M >> 1)) / M;   // round(2^21 / M), in (1024,2048]
    uint32_t exp_field, mant;
    if (q >= 2048u) {                           // M==1024: 1/s is an exact power of two
        exp_field = 30u - E;
        mant = 0u;
    } else {
        exp_field = 29u - E;
        mant = q - 1024u;
    }
    return (uint16_t)((exp_field << 10) | (mant & 0x3FFu));   // sign = 0 (Sexp > 0)
}

// Integer fp16 sqrt for a POSITIVE, normal fp16 `v`. No FPU: split the exponent (halve
// it, folding an odd bit into the significand), then take the significand's square root
// with a few Newton iterations on the integer significand (rv32iM has divu). rmsnorm's
// mean = Sxx/N is always a positive normal, so no zero/subnormal/negative handling.
static inline uint16_t sqrt_f16(uint16_t v)
{
    uint32_t E = (v >> 10) & 0x1Fu;
    if (E == 0u) return 0u;                            // 0 / subnormal (not a normal mean)
    uint32_t M = 1024u + (v & 0x3FFu);                 // significand [1024,2047], value M/1024
    int32_t  e = (int32_t)E - 15;                      // unbiased exponent
    uint32_t sig; int32_t oe;
    if (e & 1) { sig = 2u * M; oe = (e - 1) >> 1; }    // odd:  sqrt(2*(M/1024)) * 2^((e-1)/2)
    else       { sig = M;      oe =  e      >> 1; }    // even: sqrt(M/1024)     * 2^(e/2)
    // Result significand (Q10) = isqrt(sig << 10); sig<<10 is in [2^20, 2^22] -> isqrt in [1024,2047].
    uint32_t n = sig << 10;
    uint32_t x = 1448u;                                // mid-range seed
    x = (x + n / x) >> 1;
    x = (x + n / x) >> 1;
    x = (x + n / x) >> 1;
    x = (x + n / x) >> 1;
    x = (x + n / x) >> 1;                              // 5 Newton iters converge to isqrt(n)
    uint32_t mant = x - 1024u;                         // x in [1024,2048)
    uint32_t exp_field = (uint32_t)(oe + 15);
    return (uint16_t)((exp_field << 10) | (mant & 0x3FFu));   // sign 0 (v > 0)
}

// f16 -> f32 bit pattern (integer; the DM core has no FPU). softmax's -max and 1/Sexp
// are finite normals, so only the zero + normal cases arise (no subnormal/inf/nan).
static inline uint32_t f16_to_f32bits(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    if (exp == 0u) return sign;                             // +/-0 (softmax scalars: no subnormals)
    return sign | ((exp + 112u) << 23) | (mant << 13);     // normal: rebias 15->127 (+112), mant 10->23
}

#endif  // SNAX_FP16_MATH_H
