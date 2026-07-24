// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Single-kernel cycle-count sweep + correctness check for the FP32 RVV host
// kernel "quantize" (bingo dispatches it to CVA6+Ara).  Per size it prints:
//   CYCLES,quantize,<N>,<rep>,<cycles>   (timing)
//   CHECK,quantize,<N>,PASS|FAIL         (output vs golden from util/sim/ara/ara_lib.py)
// gather_ara_luts.py turns the CYCLES lines into a CSV.  Golden uses exact math;
// ARA_TOL is loose for exp-based kernels (the HW path uses a poly approximation).

#include "host.h"
#include "host_kernel_lib.h"
#include "op_test_data.h"
#include "ara_sweep.h"   // ara_sizes[] / ARA_NSIZES -- the ONE sweep-size list

#define ARA_TOL 0.001f

// Timing buffers (globals -> live in .data, avoid stack overflow).
// OP_MAX_LEN is 4096 (from util/sim/ara/ara_lib.py) -> 16 KB per fp32 buffer.
static int8_t timing_int8_scratch[OP_MAX_LEN] __attribute__((aligned(8)));
static float timing_scale_scratch __attribute__((aligned(8)));
static bingo_kernel_scratchpad_t timing_scratchpad __attribute__((aligned(8)));
// f16i8 input: the fp32 sweep input cast to fp16 at run time (same convention the
// fp16 arms of ara_sweep.h use -- one input array, cast per precision).
static _Float16 timing_f16_src[OP_MAX_LEN] __attribute__((aligned(8)));

// One measured point: run the conversion, print its CYCLES row, check the int8 codes
// AND the emitted per-tensor scale against this size's golden. A macro (not a function
// taking a kernel pointer) so the kernel call stays DIRECT -- an indirect call would sit
// inside the cycle window and tax the measurement.
//
// PREC_TOK is the in->out pair, which is also the bingo op id suffix (quantize_f32i8 /
// quantize_f16i8) -- both the CYCLES and CHECK rows carry it, so the two conversions of
// this one app stay distinguishable.
#define QUANT_POINT(PREC_TOK, KFN, SRC, GQ, GS)                                     \
    do {                                                                            \
        t_args[0] = (uint64_t)(uintptr_t)(SRC);                                     \
        t_args[1] = (uint64_t)(uintptr_t)timing_int8_scratch;                       \
        t_args[2] = (uint64_t)(uintptr_t)&timing_scale_scratch;                     \
        t_args[3] = N;                                                              \
        t_args[4] = 0;  /* Arg4 = precision: a no-op for the conversion kernels */  \
        t_args[5] = (uint64_t)(uintptr_t)&timing_scratchpad;                        \
        uint64_t _c0 = ara_get_cycle_count();                                       \
        KFN(t_args);                                                                \
        uint64_t _c1 = ara_get_cycle_count();                                       \
        printf("CYCLES,quantize," PREC_TOK ",%lu,%d,%lu\r\n", N, rep, _c1 - _c0);   \
        int _errs = 0;                                                              \
        for (uint64_t i = 0; i < N; i++) {                                          \
            int _d = (int)timing_int8_scratch[i] - (int)(GQ)[si][i];                \
            if (_d < 0) _d = -_d;                                                   \
            if (_d > 1) _errs++;            /* allow +/-1 rounding */               \
        }                                                                           \
        float _ds = timing_scale_scratch - (GS)[si]; if (_ds < 0) _ds = -_ds;       \
        float _gs = (GS)[si] > 1e-9f ? (GS)[si] : 1e-9f;                            \
        if (_ds / _gs > 1e-3f) _errs++;                                             \
        printf("CHECK,quantize," PREC_TOK ",%lu,%s\r\n", N, _errs ? "FAIL" : "PASS"); \
    } while (0)

// Sizes come from ara_sweep.h. They MUST be the same list ara_lib.py emitted the
// per-size goldens for: this app indexes golden_q[si]/golden_scale[si] BY SIZE INDEX,
// so a private size list silently compares size si against the golden of a different
// size (a local {64,256,1024,4096} against a 6-entry golden checked N=64 vs the N=32
// golden, and every quantize point failed).
#define TIMING_NUM_SIZES ARA_NSIZES
#define TIMING_NUM_REPS  1
#define timing_sizes     ara_sizes

int main() {
    uintptr_t address_prefix = (uintptr_t)get_current_chip_baseaddress();
    init_uart(address_prefix, 32, 1);
    asm volatile("fence" ::: "memory");

    printf("=== ara sweep: quantize ===\r\n");
    printf("CYCLES_HEADER,kernel,prec,N,rep,cycles\r\n");

    uint64_t t_args[8];

    // f16i8 reuses the fp32 sweep input, cast once to fp16.
    for (uint64_t i = 0; i < OP_MAX_LEN; i++)
        timing_f16_src[i] = (_Float16)op_mixed_big[i];

    for (int si = 0; si < TIMING_NUM_SIZES; si++) {
        uint64_t N = timing_sizes[si];

        for (int rep = 0; rep < TIMING_NUM_REPS; rep++) {
            // Both quantize conversions bingo can dispatch. f16i8 is the one the
            // fp16 activation path feeds (the host-side sibling of the xDMA
            // Fp16ToInt8 datapath); it has its OWN golden, since rounding the input
            // to fp16 moves max|x| and therefore the scale and the int8 codes.
            QUANT_POINT("f32i8", __host_bingo_kernel_quantize_f32i8,
                        op_mixed_big, golden_q, golden_scale);
            QUANT_POINT("f16i8", __host_bingo_kernel_quantize_f16i8,
                        timing_f16_src, golden_q16, golden_scale16);
        }
    }

    printf("=== ara quantize done ===\r\n");
    return 0;
}
