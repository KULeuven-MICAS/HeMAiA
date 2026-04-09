// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Cycle-count characterization sweep for the FP32 RVV host kernels.
// Emits one MCYCLE-style CSV line per (kernel × size × rep) combination,
// which is then parsed by scripts/characterize_cva6.py to fit a linear
// model `cycles = overhead + slope·N` per op.  This data populates
// inputs/rtl_luts/cva6_ops.csv for the framework's `--cost-model=rtl` mode.
//
// Paired workload: ci_ara (same kernels, but correctness-only with N=32 —
// used as the fast CI regression).
//
// NOTE: This workload is LONG-RUNNING in RTL simulation (many ops × sizes).
// Use ci_ara for quick correctness regressions.

#include "host.h"
#include "host_kernel_lib.h"
#include "op_test_data.h"

// Timing buffers (globals → live in .data, avoid stack overflow).
// OP_MAX_LEN is 8192 (from gen_op_data.py) → 32 KB per buffer.
static float timing_output[OP_MAX_LEN] __attribute__((aligned(8)));
static float timing_fp32_scratch[OP_MAX_LEN] __attribute__((aligned(8)));
static int8_t timing_int8_scratch[OP_MAX_LEN] __attribute__((aligned(8)));
static float timing_scale_scratch __attribute__((aligned(8)));

// Timing sizes that span the TinyLlama workload range.
#define TIMING_NUM_SIZES 5
#define TIMING_NUM_REPS  3
static const uint64_t timing_sizes[TIMING_NUM_SIZES] = { 64, 256, 1024, 4096, 8192 };

int main() {
    uintptr_t address_prefix = (uintptr_t)get_current_chip_baseaddress();
    init_uart(address_prefix, 32, 1);
    enable_vec();
    asm volatile("fence" ::: "memory");

    printf("=== ara_test: CVA6+Ara FP32 cycle sweep ===\r\n");
    printf("Sizes: ");
    for (int si = 0; si < TIMING_NUM_SIZES; si++) {
        printf("%lu ", timing_sizes[si]);
    }
    printf("(reps=%d)\r\n\r\n", TIMING_NUM_REPS);
    printf("CYCLES_HEADER,kernel,N,rep,cycles\r\n");

    uint64_t t_args[8];
    float t_scalar_out;

    for (int si = 0; si < TIMING_NUM_SIZES; si++) {
        uint64_t N = timing_sizes[si];

        for (int rep = 0; rep < TIMING_NUM_REPS; rep++) {
            uint64_t c0, c1;

            // --- Binary elementwise (a, b, out, n) ---
            #define TIME_BINARY(name, kernel_fn) do { \
                t_args[0] = (uint64_t)(uintptr_t)op_a_big; \
                t_args[1] = (uint64_t)(uintptr_t)op_b_big; \
                t_args[2] = (uint64_t)(uintptr_t)timing_output; \
                t_args[3] = N; \
                c0 = ara_get_cycle_count(); \
                kernel_fn(t_args); \
                c1 = ara_get_cycle_count(); \
                printf("CYCLES,%s,%lu,%d,%lu\r\n", name, N, rep, c1 - c0); \
            } while (0)

            TIME_BINARY("add", __host_bingo_kernel_fp32_add);
            TIME_BINARY("sub", __host_bingo_kernel_fp32_sub);
            TIME_BINARY("mul", __host_bingo_kernel_fp32_mul);
            TIME_BINARY("div", __host_bingo_kernel_fp32_div);
            #undef TIME_BINARY

            // --- Unary elementwise (input, out, n) ---
            #define TIME_UNARY(name, kernel_fn, src) do { \
                t_args[0] = (uint64_t)(uintptr_t)(src); \
                t_args[1] = (uint64_t)(uintptr_t)timing_output; \
                t_args[2] = N; \
                c0 = ara_get_cycle_count(); \
                kernel_fn(t_args); \
                c1 = ara_get_cycle_count(); \
                printf("CYCLES,%s,%lu,%d,%lu\r\n", name, N, rep, c1 - c0); \
            } while (0)

            TIME_UNARY("exp", __host_bingo_kernel_fp32_exp, op_a_big);           // positive input
            TIME_UNARY("sigmoid", __host_bingo_kernel_fp32_sigmoid, op_mixed_big);
            TIME_UNARY("sqrt", __host_bingo_kernel_fp32_sqrt, op_a_big);         // positive input
            #undef TIME_UNARY

            // --- Reductions (input, scalar_out, n) ---
            #define TIME_REDUCE(name, kernel_fn) do { \
                t_args[0] = (uint64_t)(uintptr_t)op_a_big; \
                t_args[1] = (uint64_t)(uintptr_t)&t_scalar_out; \
                t_args[2] = N; \
                c0 = ara_get_cycle_count(); \
                kernel_fn(t_args); \
                c1 = ara_get_cycle_count(); \
                printf("CYCLES,%s,%lu,%d,%lu\r\n", name, N, rep, c1 - c0); \
            } while (0)

            TIME_REDUCE("reduce_sum", __host_bingo_kernel_fp32_reduce_sum);
            TIME_REDUCE("reduce_max", __host_bingo_kernel_fp32_reduce_max);
            TIME_REDUCE("reduce_mean", __host_bingo_kernel_fp32_reduce_mean);
            #undef TIME_REDUCE

            // --- Data conversion: quantize (input, int8_out, scale_out, n) ---
            t_args[0] = (uint64_t)(uintptr_t)op_mixed_big;
            t_args[1] = (uint64_t)(uintptr_t)timing_int8_scratch;
            t_args[2] = (uint64_t)(uintptr_t)&timing_scale_scratch;
            t_args[3] = N;
            c0 = ara_get_cycle_count();
            __host_bingo_kernel_fp32_quantize(t_args);
            c1 = ara_get_cycle_count();
            printf("CYCLES,quantize,%lu,%d,%lu\r\n", N, rep, c1 - c0);

            // --- Compound: softmax (input, output, num_rows=1, row_len=N) ---
            t_args[0] = (uint64_t)(uintptr_t)op_mixed_big;
            t_args[1] = (uint64_t)(uintptr_t)timing_output;
            t_args[2] = 1;
            t_args[3] = N;
            c0 = ara_get_cycle_count();
            __host_bingo_kernel_fp32_softmax(t_args);
            c1 = ara_get_cycle_count();
            printf("CYCLES,softmax,%lu,%d,%lu\r\n", N, rep, c1 - c0);

            // --- Compound: rmsnorm (input, weight=ones, output, hidden_dim=N, num_tokens=1) ---
            // Use timing_fp32_scratch as the weight buffer (ones)
            for (uint64_t i = 0; i < N; i++) timing_fp32_scratch[i] = 1.0f;
            t_args[0] = (uint64_t)(uintptr_t)op_a_big;
            t_args[1] = (uint64_t)(uintptr_t)timing_fp32_scratch;
            t_args[2] = (uint64_t)(uintptr_t)timing_output;
            t_args[3] = N;
            t_args[4] = 1;
            c0 = ara_get_cycle_count();
            __host_bingo_kernel_fp32_rmsnorm(t_args);
            c1 = ara_get_cycle_count();
            printf("CYCLES,rmsnorm,%lu,%d,%lu\r\n", N, rep, c1 - c0);
        }
    }

    printf("=== ara_test done ===\r\n");
    return 0;
}
