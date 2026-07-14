// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Shared cycle-sweep + correctness driver for the multi-precision Ara host
// kernels. Each ara_<op>/src/main.c includes, IN THIS ORDER:
//     #include "host.h"
//     #include "host_kernel_lib.h"
//     #include "op_test_data.h"   // big inputs + golden from util/sim/ara/ara_lib.py
//     #include "ara_sweep.h"
// and then invokes one ARA_MAIN_* macro. The macro sweeps the op's applicable
// precisions x sizes, calls the precision dispatcher __host_bingo_kernel_<op>
// (precision passed as a runtime arg word), and prints per point:
//     CYCLES,<op>,<prec>,<N>,<rep>,<cycles>
//     CHECK,<op>,<prec>,<N>,PASS|FAIL
// gather_ara_luts.py turns the CYCLES lines into a per-(op,prec) cycle CSV.
//
// Precision applicability (matches host_kernel_lib.h):
//   - float ops: {fp32, fp16}.  fp16 reuses the fp32 inputs cast at runtime and
//     is checked against the fp32 golden with a loose tolerance.
//   - integer-meaningful ops also get {int8, int16}: dedicated integer inputs +
//     exact integer golden (op_a_i8/op_b_i8/golden_i8 ... from ara_lib.py).
#pragma once

#define ARA_TOL_FP16   0.05f   // loose: fp16 rounding + (compound) intermediate narrowing
// Sizes must BRACKET the n the model actually queries, or every query is an extrapolation.
// The CVA6 does ONLY the per-row scalar math -- an ordinary elementwise unary over the `rows`
// splatted beats, i.e. n = rows * 32 lanes:
//
//   decode  (rows = 1)       -> n =    32
//   prefill (rows = S = 256) -> n =  8192
//   host-only baseline       -> n = 65536     (the whole [S,D] tensor, xDMA offload disabled)
//
// Small n is where extrapolation hurts most (the fixed per-call overhead dominates there), and n=32 is
// exactly the DECODE operating point, so the low end of the ladder matters most.
// Going ABOVE 4096 needs OP_MAX_LEN raised in util/sim/ara/ara_lib.py (N_BIG) -- each fp32 timing
// buffer is 4*OP_MAX_LEN bytes, so 8192 costs 32 KB/buffer. Add 8192 there when the .data budget allows;
// until then the prefill point (n=8192) is a 2x linear extrapolation, which these curves tolerate.
#define ARA_NSIZES     6

static const uint64_t ara_sizes[ARA_NSIZES] = { 32, 64, 128, 256, 1024, 4096 };

static inline const char *ara_prec_name(uint64_t p) {
    switch (p) {
        case BINGO_PREC_FP32:  return "fp32";
        case BINGO_PREC_FP16:  return "fp16";
        case BINGO_PREC_INT8:  return "int8";
        case BINGO_PREC_INT16: return "int16";
        case BINGO_PREC_INT32: return "int32";
        default:               return "?";
    }
}

#define ARA_SWEEP_INIT(opname)                                                  \
    init_uart((uintptr_t)get_current_chip_baseaddress(), 32, 1);               \
    enable_vec();                                                               \
    asm volatile("fence" ::: "memory");                                        \
    printf("=== ara sweep: " #opname " ===\r\n");                              \
    printf("CYCLES_HEADER,kernel,prec,N,rep,cycles\r\n")

// Relative-tolerance float check of `ov` against golden `g` (s floored at 1).
#define ARA_FCHECK(ov, g, tol, errs)                                            \
    do { float _d = (ov) - (g); if (_d < 0) _d = -_d;                          \
         float _s = (g) < 0 ? -(g) : (g); if (_s < 1.0f) _s = 1.0f;            \
         if (_d / _s > (tol)) (errs)++; } while (0)

// ----- binary elementwise (a, b, out, n, precision) -----
// INT variant: {fp32, fp16, int8, int16, int32}.  golden_big / golden_i8 / golden_i16 / golden_i32.
#define ARA_MAIN_BINARY_INT(opname, TOLF)                                       \
static float    ara_o_f32[OP_MAX_LEN] __attribute__((aligned(8)));             \
static _Float16 ara_a_f16[OP_MAX_LEN] __attribute__((aligned(8)));             \
static _Float16 ara_b_f16[OP_MAX_LEN] __attribute__((aligned(8)));             \
static _Float16 ara_o_f16[OP_MAX_LEN] __attribute__((aligned(8)));             \
static int8_t   ara_o_i8 [OP_MAX_LEN] __attribute__((aligned(8)));             \
static int16_t  ara_o_i16[OP_MAX_LEN] __attribute__((aligned(8)));             \
static int32_t  ara_o_i32[OP_MAX_LEN] __attribute__((aligned(8)));             \
int main(void) {                                                               \
    ARA_SWEEP_INIT(opname);                                                     \
    /* INT32 is a first-class Ara precision: an args class exists for every int-capable op at    */ \
    /* int32, and add_int32 IS the K-split reduce kernel -- so the sweep measures it alongside   */ \
    /* fp32/fp16/int8/int16, giving bingo a cycle curve for all 10 int32 ops.                    */ \
    static const uint64_t precs[5] = { BINGO_PREC_FP32, BINGO_PREC_FP16,        \
                                       BINGO_PREC_INT8, BINGO_PREC_INT16,       \
                                       BINGO_PREC_INT32 };                      \
    for (uint64_t i = 0; i < OP_MAX_LEN; i++) {                                 \
        ara_a_f16[i] = (_Float16)op_a_big[i];                                   \
        ara_b_f16[i] = (_Float16)op_b_big[i]; }                                 \
    uint64_t t[8];                                                              \
    for (int pi = 0; pi < 5; pi++) { uint64_t prec = precs[pi];                 \
        for (int si = 0; si < ARA_NSIZES; si++) { uint64_t N = ara_sizes[si];   \
            const void *A, *B; void *O;                                         \
            if (prec == BINGO_PREC_FP32)      { A = op_a_big; B = op_b_big; O = ara_o_f32; } \
            else if (prec == BINGO_PREC_FP16) { A = ara_a_f16; B = ara_b_f16; O = ara_o_f16; } \
            else if (prec == BINGO_PREC_INT8) { A = op_a_i8;  B = op_b_i8;  O = ara_o_i8; }  \
            else if (prec == BINGO_PREC_INT16){ A = op_a_i16; B = op_b_i16; O = ara_o_i16; } \
            else                              { A = op_a_i32; B = op_b_i32; O = ara_o_i32; } \
            t[0] = (uint64_t)(uintptr_t)A; t[1] = (uint64_t)(uintptr_t)B;       \
            t[2] = (uint64_t)(uintptr_t)O; t[3] = N; t[4] = prec;               \
            uint64_t c0 = ara_get_cycle_count();                               \
            __host_bingo_kernel_##opname(t);                                    \
            uint64_t c1 = ara_get_cycle_count();                               \
            printf("CYCLES," #opname ",%s,%lu,0,%lu\r\n", ara_prec_name(prec), N, c1 - c0); \
            int errs = 0;                                                       \
            if (prec == BINGO_PREC_INT8)  { for (uint64_t i=0;i<N;i++) if (ara_o_i8[i]  != golden_i8[i])  errs++; } \
            else if (prec == BINGO_PREC_INT16) { for (uint64_t i=0;i<N;i++) if (ara_o_i16[i] != golden_i16[i]) errs++; } \
            else if (prec == BINGO_PREC_INT32) { for (uint64_t i=0;i<N;i++) if (ara_o_i32[i] != golden_i32[i]) errs++; } \
            else { float tol = (prec == BINGO_PREC_FP16) ? ARA_TOL_FP16 : (TOLF); \
                for (uint64_t i=0;i<N;i++) { float ov = (prec==BINGO_PREC_FP16)?(float)ara_o_f16[i]:ara_o_f32[i]; \
                    ARA_FCHECK(ov, golden_big[i], tol, errs); } }              \
            printf("CHECK," #opname ",%s,%lu,%s\r\n", ara_prec_name(prec), N, errs ? "FAIL" : "PASS"); \
        } }                                                                     \
    printf("=== ara " #opname " done ===\r\n"); return 0;                       \
}

// FLOAT variant: {fp32, fp16}.  Used by div and silu_mul (silu_mul args share
// the binary layout: gate, up, out, n, precision; golden_big = silu(a)*b).
#define ARA_MAIN_BINARY_FLOAT(opname, TOLF)                                     \
static float    ara_o_f32[OP_MAX_LEN] __attribute__((aligned(8)));             \
static _Float16 ara_a_f16[OP_MAX_LEN] __attribute__((aligned(8)));             \
static _Float16 ara_b_f16[OP_MAX_LEN] __attribute__((aligned(8)));             \
static _Float16 ara_o_f16[OP_MAX_LEN] __attribute__((aligned(8)));             \
int main(void) {                                                               \
    ARA_SWEEP_INIT(opname);                                                     \
    static const uint64_t precs[2] = { BINGO_PREC_FP32, BINGO_PREC_FP16 };      \
    for (uint64_t i = 0; i < OP_MAX_LEN; i++) {                                 \
        ara_a_f16[i] = (_Float16)op_a_big[i];                                   \
        ara_b_f16[i] = (_Float16)op_b_big[i]; }                                 \
    uint64_t t[8];                                                              \
    for (int pi = 0; pi < 2; pi++) { uint64_t prec = precs[pi];                 \
        for (int si = 0; si < ARA_NSIZES; si++) { uint64_t N = ara_sizes[si];   \
            const void *A, *B; void *O;                                         \
            if (prec == BINGO_PREC_FP32) { A = op_a_big; B = op_b_big; O = ara_o_f32; } \
            else                         { A = ara_a_f16; B = ara_b_f16; O = ara_o_f16; } \
            t[0] = (uint64_t)(uintptr_t)A; t[1] = (uint64_t)(uintptr_t)B;       \
            t[2] = (uint64_t)(uintptr_t)O; t[3] = N; t[4] = prec;               \
            uint64_t c0 = ara_get_cycle_count();                               \
            __host_bingo_kernel_##opname(t);                                    \
            uint64_t c1 = ara_get_cycle_count();                               \
            printf("CYCLES," #opname ",%s,%lu,0,%lu\r\n", ara_prec_name(prec), N, c1 - c0); \
            /* Float-only op: this sweep runs {fp32, fp16} only (precs[2]), and     */ \
            /* ara_lib.py emits no int golden for it -- so there is no int arm here. */ \
            int errs = 0; float tol = (prec == BINGO_PREC_FP16) ? ARA_TOL_FP16 : (TOLF); \
            for (uint64_t i=0;i<N;i++) {                                        \
                float ov = (prec==BINGO_PREC_FP16)?(float)ara_o_f16[i]:ara_o_f32[i]; \
                ARA_FCHECK(ov, golden_big[i], tol, errs); }                     \
            printf("CHECK," #opname ",%s,%lu,%s\r\n", ara_prec_name(prec), N, errs ? "FAIL" : "PASS"); \
        } }                                                                     \
    printf("=== ara " #opname " done ===\r\n"); return 0;                       \
}

// ----- unary elementwise (in, out, n, precision) -----
// SRC is the fp32 input array (op_a_big / op_mixed_big) matching the golden.
#define ARA_MAIN_UNARY_INT(opname, SRC, TOLF)                                   \
static float    ara_o_f32[OP_MAX_LEN] __attribute__((aligned(8)));             \
static _Float16 ara_i_f16[OP_MAX_LEN] __attribute__((aligned(8)));             \
static _Float16 ara_o_f16[OP_MAX_LEN] __attribute__((aligned(8)));             \
static int8_t   ara_o_i8 [OP_MAX_LEN] __attribute__((aligned(8)));             \
static int16_t  ara_o_i16[OP_MAX_LEN] __attribute__((aligned(8)));             \
static int32_t  ara_o_i32[OP_MAX_LEN] __attribute__((aligned(8)));             \
int main(void) {                                                               \
    ARA_SWEEP_INIT(opname);                                                     \
    /* INT32 is a first-class Ara precision: an args class exists for every int-capable op at    */ \
    /* int32, and add_int32 IS the K-split reduce kernel -- so the sweep measures it alongside   */ \
    /* fp32/fp16/int8/int16, giving bingo a cycle curve for all 10 int32 ops.                    */ \
    static const uint64_t precs[5] = { BINGO_PREC_FP32, BINGO_PREC_FP16,        \
                                       BINGO_PREC_INT8, BINGO_PREC_INT16,       \
                                       BINGO_PREC_INT32 };                      \
    for (uint64_t i = 0; i < OP_MAX_LEN; i++) ara_i_f16[i] = (_Float16)(SRC)[i]; \
    uint64_t t[8];                                                              \
    for (int pi = 0; pi < 5; pi++) { uint64_t prec = precs[pi];                 \
        for (int si = 0; si < ARA_NSIZES; si++) { uint64_t N = ara_sizes[si];   \
            const void *A; void *O;                                             \
            if (prec == BINGO_PREC_FP32)      { A = (SRC);     O = ara_o_f32; }  \
            else if (prec == BINGO_PREC_FP16) { A = ara_i_f16; O = ara_o_f16; }  \
            else if (prec == BINGO_PREC_INT8) { A = op_a_i8;   O = ara_o_i8; }   \
            else if (prec == BINGO_PREC_INT16){ A = op_a_i16;  O = ara_o_i16; }  \
            else                              { A = op_a_i32;  O = ara_o_i32; }  \
            t[0] = (uint64_t)(uintptr_t)A; t[1] = (uint64_t)(uintptr_t)O;       \
            t[2] = N; t[3] = prec;                                              \
            uint64_t c0 = ara_get_cycle_count();                               \
            __host_bingo_kernel_##opname(t);                                    \
            uint64_t c1 = ara_get_cycle_count();                               \
            printf("CYCLES," #opname ",%s,%lu,0,%lu\r\n", ara_prec_name(prec), N, c1 - c0); \
            int errs = 0;                                                       \
            if (prec == BINGO_PREC_INT8)  { for (uint64_t i=0;i<N;i++) if (ara_o_i8[i]  != golden_i8[i])  errs++; } \
            else if (prec == BINGO_PREC_INT16) { for (uint64_t i=0;i<N;i++) if (ara_o_i16[i] != golden_i16[i]) errs++; } \
            else if (prec == BINGO_PREC_INT32) { for (uint64_t i=0;i<N;i++) if (ara_o_i32[i] != golden_i32[i]) errs++; } \
            else { float tol = (prec == BINGO_PREC_FP16) ? ARA_TOL_FP16 : (TOLF); \
                for (uint64_t i=0;i<N;i++) { float ov = (prec==BINGO_PREC_FP16)?(float)ara_o_f16[i]:ara_o_f32[i]; \
                    ARA_FCHECK(ov, golden_big[i], tol, errs); } }              \
            printf("CHECK," #opname ",%s,%lu,%s\r\n", ara_prec_name(prec), N, errs ? "FAIL" : "PASS"); \
        } }                                                                     \
    printf("=== ara " #opname " done ===\r\n"); return 0;                       \
}

#define ARA_MAIN_UNARY_FLOAT(opname, SRC, TOLF)                                 \
static float    ara_o_f32[OP_MAX_LEN] __attribute__((aligned(8)));             \
static _Float16 ara_i_f16[OP_MAX_LEN] __attribute__((aligned(8)));             \
static _Float16 ara_o_f16[OP_MAX_LEN] __attribute__((aligned(8)));             \
int main(void) {                                                               \
    ARA_SWEEP_INIT(opname);                                                     \
    static const uint64_t precs[2] = { BINGO_PREC_FP32, BINGO_PREC_FP16 };      \
    for (uint64_t i = 0; i < OP_MAX_LEN; i++) ara_i_f16[i] = (_Float16)(SRC)[i]; \
    uint64_t t[8];                                                              \
    for (int pi = 0; pi < 2; pi++) { uint64_t prec = precs[pi];                 \
        for (int si = 0; si < ARA_NSIZES; si++) { uint64_t N = ara_sizes[si];   \
            const void *A; void *O;                                             \
            if (prec == BINGO_PREC_FP32) { A = (SRC);     O = ara_o_f32; }       \
            else                         { A = ara_i_f16; O = ara_o_f16; }       \
            t[0] = (uint64_t)(uintptr_t)A; t[1] = (uint64_t)(uintptr_t)O;       \
            t[2] = N; t[3] = prec;                                              \
            uint64_t c0 = ara_get_cycle_count();                               \
            __host_bingo_kernel_##opname(t);                                    \
            uint64_t c1 = ara_get_cycle_count();                               \
            printf("CYCLES," #opname ",%s,%lu,0,%lu\r\n", ara_prec_name(prec), N, c1 - c0); \
            /* Float-only op: this sweep runs {fp32, fp16} only (precs[2]), and     */ \
            /* ara_lib.py emits no int golden for it -- so there is no int arm here. */ \
            int errs = 0; float tol = (prec == BINGO_PREC_FP16) ? ARA_TOL_FP16 : (TOLF); \
            for (uint64_t i=0;i<N;i++) {                                        \
                float ov = (prec==BINGO_PREC_FP16)?(float)ara_o_f16[i]:ara_o_f32[i]; \
                ARA_FCHECK(ov, golden_big[i], tol, errs); }                     \
            printf("CHECK," #opname ",%s,%lu,%s\r\n", ara_prec_name(prec), N, errs ? "FAIL" : "PASS"); \
        } }                                                                     \
    printf("=== ara " #opname " done ===\r\n"); return 0;                       \
}

// ----- reductions (in, scalar_out, n, precision) -----
// fp32/fp16 write a float; int8/int16/int32 all write an int32 scalar. golden_reduce[]
// (float), golden_reduce_i8/_i16/_i32[] (int32), one entry per sweep size.
#define ARA_MAIN_REDUCE_INT(opname, TOLF)                                       \
static _Float16 ara_i_f16[OP_MAX_LEN] __attribute__((aligned(8)));             \
int main(void) {                                                               \
    ARA_SWEEP_INIT(opname);                                                     \
    static const uint64_t precs[5] = { BINGO_PREC_FP32, BINGO_PREC_FP16,        \
                                       BINGO_PREC_INT8, BINGO_PREC_INT16,       \
                                       BINGO_PREC_INT32 };                      \
    for (uint64_t i = 0; i < OP_MAX_LEN; i++) ara_i_f16[i] = (_Float16)op_a_big[i]; \
    uint64_t t[8]; float sc_f = 0.0f; int32_t sc_i = 0;                         \
    for (int pi = 0; pi < 5; pi++) { uint64_t prec = precs[pi];                 \
        for (int si = 0; si < ARA_NSIZES; si++) { uint64_t N = ara_sizes[si];   \
            const void *A; void *O;                                            \
            if (prec == BINGO_PREC_FP32)      { A = op_a_big;  O = &sc_f; }      \
            else if (prec == BINGO_PREC_FP16) { A = ara_i_f16; O = &sc_f; }      \
            else if (prec == BINGO_PREC_INT8) { A = op_a_i8;   O = &sc_i; }      \
            else if (prec == BINGO_PREC_INT16){ A = op_a_i16;  O = &sc_i; }      \
            else                              { A = op_a_i32;  O = &sc_i; }      \
            t[0] = (uint64_t)(uintptr_t)A; t[1] = (uint64_t)(uintptr_t)O;       \
            t[2] = N; t[3] = prec;                                              \
            uint64_t c0 = ara_get_cycle_count();                               \
            __host_bingo_kernel_##opname(t);                                    \
            uint64_t c1 = ara_get_cycle_count();                               \
            printf("CYCLES," #opname ",%s,%lu,0,%lu\r\n", ara_prec_name(prec), N, c1 - c0); \
            int errs = 0;                                                       \
            if (prec == BINGO_PREC_INT8)       { if (sc_i != golden_reduce_i8[si])  errs++; } \
            else if (prec == BINGO_PREC_INT16) { if (sc_i != golden_reduce_i16[si]) errs++; } \
            else if (prec == BINGO_PREC_INT32) { if (sc_i != golden_reduce_i32[si]) errs++; } \
            else { float tol = (prec == BINGO_PREC_FP16) ? ARA_TOL_FP16 : (TOLF); \
                ARA_FCHECK(sc_f, golden_reduce[si], tol, errs); }              \
            printf("CHECK," #opname ",%s,%lu,%s\r\n", ara_prec_name(prec), N, errs ? "FAIL" : "PASS"); \
        } }                                                                     \
    printf("=== ara " #opname " done ===\r\n"); return 0;                       \
}

#define ARA_MAIN_REDUCE_FLOAT(opname, TOLF)                                     \
static _Float16 ara_i_f16[OP_MAX_LEN] __attribute__((aligned(8)));             \
int main(void) {                                                               \
    ARA_SWEEP_INIT(opname);                                                     \
    static const uint64_t precs[2] = { BINGO_PREC_FP32, BINGO_PREC_FP16 };      \
    for (uint64_t i = 0; i < OP_MAX_LEN; i++) ara_i_f16[i] = (_Float16)op_a_big[i]; \
    uint64_t t[8]; float sc_f = 0.0f;                                           \
    for (int pi = 0; pi < 2; pi++) { uint64_t prec = precs[pi];                 \
        for (int si = 0; si < ARA_NSIZES; si++) { uint64_t N = ara_sizes[si];   \
            const void *A = (prec == BINGO_PREC_FP32) ? (const void*)op_a_big : (const void*)ara_i_f16; \
            t[0] = (uint64_t)(uintptr_t)A; t[1] = (uint64_t)(uintptr_t)&sc_f;   \
            t[2] = N; t[3] = prec;                                              \
            uint64_t c0 = ara_get_cycle_count();                               \
            __host_bingo_kernel_##opname(t);                                    \
            uint64_t c1 = ara_get_cycle_count();                               \
            printf("CYCLES," #opname ",%s,%lu,0,%lu\r\n", ara_prec_name(prec), N, c1 - c0); \
            int errs = 0; float tol = (prec == BINGO_PREC_FP16) ? ARA_TOL_FP16 : (TOLF); \
            ARA_FCHECK(sc_f, golden_reduce[si], tol, errs);                    \
            printf("CHECK," #opname ",%s,%lu,%s\r\n", ara_prec_name(prec), N, errs ? "FAIL" : "PASS"); \
        } }                                                                     \
    printf("=== ara " #opname " done ===\r\n"); return 0;                       \
}

// ----- softmax (in, out, num_rows=1, row_len=N, precision, scratchpad) -----
#define ARA_MAIN_SOFTMAX(TOLF)                                                  \
static float    ara_o_f32[OP_MAX_LEN] __attribute__((aligned(8)));             \
static _Float16 ara_i_f16[OP_MAX_LEN] __attribute__((aligned(8)));             \
static _Float16 ara_o_f16[OP_MAX_LEN] __attribute__((aligned(8)));             \
static bingo_kernel_scratchpad_t ara_sp __attribute__((aligned(8)));           \
int main(void) {                                                               \
    ARA_SWEEP_INIT(softmax);                                                    \
    static const uint64_t precs[2] = { BINGO_PREC_FP32, BINGO_PREC_FP16 };      \
    for (uint64_t i = 0; i < OP_MAX_LEN; i++) ara_i_f16[i] = (_Float16)op_mixed_big[i]; \
    uint64_t t[8];                                                              \
    for (int pi = 0; pi < 2; pi++) { uint64_t prec = precs[pi];                 \
        for (int si = 0; si < ARA_NSIZES; si++) { uint64_t N = ara_sizes[si];   \
            const void *A; void *O;                                            \
            if (prec == BINGO_PREC_FP32) { A = op_mixed_big; O = ara_o_f32; }    \
            else                         { A = ara_i_f16;    O = ara_o_f16; }    \
            t[0] = (uint64_t)(uintptr_t)A; t[1] = (uint64_t)(uintptr_t)O;       \
            t[2] = 1; t[3] = N; t[4] = prec; t[5] = (uint64_t)(uintptr_t)&ara_sp; \
            uint64_t c0 = ara_get_cycle_count();                               \
            __host_bingo_kernel_softmax(t);                                     \
            uint64_t c1 = ara_get_cycle_count();                               \
            printf("CYCLES,softmax,%s,%lu,0,%lu\r\n", ara_prec_name(prec), N, c1 - c0); \
            const float *g = golden_vec[si]; int errs = 0;                      \
            float tol = (prec == BINGO_PREC_FP16) ? ARA_TOL_FP16 : (TOLF);      \
            for (uint64_t i=0;i<N;i++) { float ov = (prec==BINGO_PREC_FP16)?(float)ara_o_f16[i]:ara_o_f32[i]; \
                ARA_FCHECK(ov, g[i], tol, errs); }                            \
            printf("CHECK,softmax,%s,%lu,%s\r\n", ara_prec_name(prec), N, errs ? "FAIL" : "PASS"); \
        } }                                                                     \
    printf("=== ara softmax done ===\r\n"); return 0;                           \
}

// ----- rmsnorm (in, weight=ones, out, hidden=N, num_tokens=1, precision) -----
#define ARA_MAIN_RMSNORM(TOLF)                                                  \
static float    ara_o_f32[OP_MAX_LEN] __attribute__((aligned(8)));             \
static float    ara_w_f32[OP_MAX_LEN] __attribute__((aligned(8)));             \
static _Float16 ara_i_f16[OP_MAX_LEN] __attribute__((aligned(8)));             \
static _Float16 ara_w_f16[OP_MAX_LEN] __attribute__((aligned(8)));             \
static _Float16 ara_o_f16[OP_MAX_LEN] __attribute__((aligned(8)));             \
int main(void) {                                                               \
    ARA_SWEEP_INIT(rmsnorm);                                                    \
    static const uint64_t precs[2] = { BINGO_PREC_FP32, BINGO_PREC_FP16 };      \
    for (uint64_t i = 0; i < OP_MAX_LEN; i++) {                                 \
        ara_i_f16[i] = (_Float16)op_a_big[i];                                   \
        ara_w_f32[i] = 1.0f; ara_w_f16[i] = (_Float16)1.0f; }                   \
    uint64_t t[8];                                                              \
    for (int pi = 0; pi < 2; pi++) { uint64_t prec = precs[pi];                 \
        for (int si = 0; si < ARA_NSIZES; si++) { uint64_t N = ara_sizes[si];   \
            const void *A, *W; void *O;                                        \
            if (prec == BINGO_PREC_FP32) { A = op_a_big;  W = ara_w_f32; O = ara_o_f32; } \
            else                         { A = ara_i_f16; W = ara_w_f16; O = ara_o_f16; } \
            t[0] = (uint64_t)(uintptr_t)A; t[1] = (uint64_t)(uintptr_t)W;       \
            t[2] = (uint64_t)(uintptr_t)O; t[3] = N; t[4] = 1; t[5] = prec;     \
            uint64_t c0 = ara_get_cycle_count();                               \
            __host_bingo_kernel_rmsnorm(t);                                     \
            uint64_t c1 = ara_get_cycle_count();                               \
            printf("CYCLES,rmsnorm,%s,%lu,0,%lu\r\n", ara_prec_name(prec), N, c1 - c0); \
            const float *g = golden_vec[si]; int errs = 0;                      \
            float tol = (prec == BINGO_PREC_FP16) ? ARA_TOL_FP16 : (TOLF);      \
            for (uint64_t i=0;i<N;i++) { float ov = (prec==BINGO_PREC_FP16)?(float)ara_o_f16[i]:ara_o_f32[i]; \
                ARA_FCHECK(ov, g[i], tol, errs); }                            \
            printf("CHECK,rmsnorm,%s,%lu,%s\r\n", ara_prec_name(prec), N, errs ? "FAIL" : "PASS"); \
        } }                                                                     \
    printf("=== ara rmsnorm done ===\r\n"); return 0;                           \
}
