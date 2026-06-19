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

#define ARA_TOL 0.001f

// Timing buffers (globals -> live in .data, avoid stack overflow).
// OP_MAX_LEN is 4096 (from util/sim/ara/ara_lib.py) -> 16 KB per fp32 buffer.
static int8_t timing_int8_scratch[OP_MAX_LEN] __attribute__((aligned(8)));
static float timing_scale_scratch __attribute__((aligned(8)));
static bingo_kernel_scratchpad_t timing_scratchpad __attribute__((aligned(8)));

#define TIMING_NUM_SIZES 4
#define TIMING_NUM_REPS  1
static const uint64_t timing_sizes[TIMING_NUM_SIZES] = { 64, 256, 1024, 4096 };

int main() {
    uintptr_t address_prefix = (uintptr_t)get_current_chip_baseaddress();
    init_uart(address_prefix, 32, 1);
    enable_vec();
    asm volatile("fence" ::: "memory");

    printf("=== ara sweep: quantize ===\r\n");
    printf("CYCLES_HEADER,kernel,N,rep,cycles\r\n");

    uint64_t t_args[8];

    for (int si = 0; si < TIMING_NUM_SIZES; si++) {
        uint64_t N = timing_sizes[si];

        for (int rep = 0; rep < TIMING_NUM_REPS; rep++) {
            uint64_t c0, c1;

            // Quantize (input, int8_out, scale_out, n, scratchpad).
            t_args[0] = (uint64_t)(uintptr_t)op_mixed_big;
            t_args[1] = (uint64_t)(uintptr_t)timing_int8_scratch;
            t_args[2] = (uint64_t)(uintptr_t)&timing_scale_scratch;
            t_args[3] = N;
            t_args[4] = (uint64_t)(uintptr_t)&timing_scratchpad;
            c0 = ara_get_cycle_count();
            __host_bingo_kernel_fp32_quantize(t_args);
            c1 = ara_get_cycle_count();
            printf("CYCLES,quantize,%lu,%d,%lu\r\n", N, rep, c1 - c0);

            {
                const int8_t *gq = golden_q[si];
                int errs = 0;
                for (uint64_t i = 0; i < N; i++) {
                    int diff = (int)timing_int8_scratch[i] - (int)gq[i];
                    if (diff < 0) diff = -diff;
                    if (diff > 1) errs++;   /* allow +/-1 rounding */
                }
                float ds = timing_scale_scratch - golden_scale[si]; if (ds < 0) ds = -ds;
                float gs = golden_scale[si] > 1e-9f ? golden_scale[si] : 1e-9f;
                if (ds / gs > 1e-3f) errs++;
                printf("CHECK,quantize,%lu,%s\r\n", N, errs ? "FAIL" : "PASS");
            }
        }
    }

    printf("=== ara quantize done ===\r\n");
    return 0;
}
