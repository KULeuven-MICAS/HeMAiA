// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Test all FP32 RVV kernels that the bingo framework dispatches to CVA6+Ara.
// These correspond to the BingoOp set: elementwise binary (ADD/SUB/MUL/DIV/MAX/MIN),
// elementwise unary (RELU/NEG/ABS/EXP/SIGMOID/TANH/SQRT/RECIPROCAL/SILU/GELU),
// reductions (REDUCE_SUM/REDUCE_MAX/REDUCE_MEAN), and compound ops (SOFTMAX/RMSNORM).

#include "host.h"
#include "host_kernel_lib.h"
#include "op_test_data.h"

static int tests_passed = 0;
static int tests_failed = 0;

// Tolerance for FP32 comparison (exp/sigmoid have polynomial approximation error)
#define FP32_TOL 1e-3f
#define FP32_TOL_LOOSE 1e-2f  // for exp-based ops (Cephes polynomial)

static int check_fp32_array(const char *name, float *result, float *golden, int n, float tol) {
    int err = 0;
    for (int i = 0; i < n; i++) {
        float diff = result[i] - golden[i];
        if (diff < 0) diff = -diff;
        float scale = golden[i] < 0 ? -golden[i] : golden[i];
        if (scale < 1.0f) scale = 1.0f;
        if (diff / scale > tol) {
            if (err < 3) {
                printf("  %s[%d]: got %.6f, expected %.6f (diff=%.6f)\r\n",
                       name, i, result[i], golden[i], diff);
            }
            err++;
        }
    }
    if (err == 0) {
        printf("  PASS: %s\r\n", name);
        tests_passed++;
    } else {
        printf("  FAIL: %s (%d/%d mismatches)\r\n", name, err, n);
        tests_failed++;
    }
    return err;
}

static int check_fp32_scalar(const char *name, float result, float golden, float tol) {
    float diff = result - golden;
    if (diff < 0) diff = -diff;
    float scale = golden < 0 ? -golden : golden;
    if (scale < 1.0f) scale = 1.0f;
    if (diff / scale > tol) {
        printf("  FAIL: %s got %.6f, expected %.6f\r\n", name, result, golden);
        tests_failed++;
        return 1;
    }
    printf("  PASS: %s (%.6f)\r\n", name, result);
    tests_passed++;
    return 0;
}

// Helper: build binary kernel args (a, b, output, n)
static void setup_binary_args(uint64_t *args, float *a, float *b, float *out, uint64_t n) {
    args[0] = (uint64_t)(uintptr_t)a;
    args[1] = (uint64_t)(uintptr_t)b;
    args[2] = (uint64_t)(uintptr_t)out;
    args[3] = n;
}

// Helper: build unary kernel args (input, output, n)
static void setup_unary_args(uint64_t *args, float *input, float *out, uint64_t n) {
    args[0] = (uint64_t)(uintptr_t)input;
    args[1] = (uint64_t)(uintptr_t)out;
    args[2] = n;
}

int main() {
    uintptr_t address_prefix = (uintptr_t)get_current_chip_baseaddress();
    init_uart(address_prefix, 32, 1);
    enable_vec();
    asm volatile("fence" ::: "memory");

    printf("=== FP32 Operator Kernel Tests ===\r\n");
    printf("Vector length: %d elements\r\n\r\n", OP_TEST_LEN);

    float output[OP_TEST_LEN];
    uint64_t args[8];

    // ─── Binary elementwise ────────────────────────────────────
    printf("[Binary Elementwise]\r\n");

    setup_binary_args(args, op_a, op_b, output, OP_TEST_LEN);
    __host_bingo_kernel_fp32_add(args);
    check_fp32_array("add", output, golden_add, OP_TEST_LEN, FP32_TOL);

    setup_binary_args(args, op_a, op_b, output, OP_TEST_LEN);
    __host_bingo_kernel_fp32_sub(args);
    check_fp32_array("sub", output, golden_sub, OP_TEST_LEN, FP32_TOL);

    setup_binary_args(args, op_a, op_b, output, OP_TEST_LEN);
    __host_bingo_kernel_fp32_mul(args);
    check_fp32_array("mul", output, golden_mul, OP_TEST_LEN, FP32_TOL);

    setup_binary_args(args, op_a, op_b, output, OP_TEST_LEN);
    __host_bingo_kernel_fp32_div(args);
    check_fp32_array("div", output, golden_div, OP_TEST_LEN, FP32_TOL);

    setup_binary_args(args, op_a, op_b, output, OP_TEST_LEN);
    __host_bingo_kernel_fp32_max(args);
    check_fp32_array("max", output, golden_max, OP_TEST_LEN, FP32_TOL);

    setup_binary_args(args, op_a, op_b, output, OP_TEST_LEN);
    __host_bingo_kernel_fp32_min(args);
    check_fp32_array("min", output, golden_min, OP_TEST_LEN, FP32_TOL);

    // ─── Unary elementwise ─────────────────────────────────────
    printf("\r\n[Unary Elementwise]\r\n");

    setup_unary_args(args, op_mixed, output, OP_TEST_LEN);
    __host_bingo_kernel_fp32_relu(args);
    check_fp32_array("relu", output, golden_relu, OP_TEST_LEN, FP32_TOL);

    setup_unary_args(args, op_mixed, output, OP_TEST_LEN);
    __host_bingo_kernel_fp32_neg(args);
    check_fp32_array("neg", output, golden_neg, OP_TEST_LEN, FP32_TOL);

    setup_unary_args(args, op_mixed, output, OP_TEST_LEN);
    __host_bingo_kernel_fp32_abs(args);
    check_fp32_array("abs", output, golden_abs, OP_TEST_LEN, FP32_TOL);

    setup_unary_args(args, op_a, output, OP_TEST_LEN);  // use positive for exp
    __host_bingo_kernel_fp32_exp(args);
    check_fp32_array("exp", output, golden_exp, OP_TEST_LEN, FP32_TOL_LOOSE);

    setup_unary_args(args, op_mixed, output, OP_TEST_LEN);
    __host_bingo_kernel_fp32_sigmoid(args);
    check_fp32_array("sigmoid", output, golden_sigmoid, OP_TEST_LEN, FP32_TOL_LOOSE);

    setup_unary_args(args, op_mixed, output, OP_TEST_LEN);
    __host_bingo_kernel_fp32_tanh(args);
    check_fp32_array("tanh", output, golden_tanh, OP_TEST_LEN, FP32_TOL_LOOSE);

    setup_unary_args(args, op_a, output, OP_TEST_LEN);  // use positive for sqrt
    __host_bingo_kernel_fp32_sqrt(args);
    check_fp32_array("sqrt", output, golden_sqrt, OP_TEST_LEN, FP32_TOL);

    setup_unary_args(args, op_a, output, OP_TEST_LEN);  // use positive for reciprocal
    __host_bingo_kernel_fp32_reciprocal(args);
    check_fp32_array("reciprocal", output, golden_reciprocal, OP_TEST_LEN, FP32_TOL);

    setup_unary_args(args, op_mixed, output, OP_TEST_LEN);
    __host_bingo_kernel_fp32_silu(args);
    check_fp32_array("silu", output, golden_silu, OP_TEST_LEN, FP32_TOL_LOOSE);

    // ─── Reductions ────────────────────────────────────────────
    printf("\r\n[Reductions]\r\n");
    float scalar_out;

    setup_unary_args(args, op_a, &scalar_out, OP_TEST_LEN);
    __host_bingo_kernel_fp32_reduce_sum(args);
    check_fp32_scalar("reduce_sum", scalar_out, golden_reduce_sum, FP32_TOL);

    setup_unary_args(args, op_a, &scalar_out, OP_TEST_LEN);
    __host_bingo_kernel_fp32_reduce_max(args);
    check_fp32_scalar("reduce_max", scalar_out, golden_reduce_max, FP32_TOL);

    setup_unary_args(args, op_a, &scalar_out, OP_TEST_LEN);
    __host_bingo_kernel_fp32_reduce_mean(args);
    check_fp32_scalar("reduce_mean", scalar_out, golden_reduce_mean, FP32_TOL);

    // ─── Compound ops ──────────────────────────────────────────
    printf("\r\n[Compound Ops]\r\n");

    // Softmax (on op_a, treated as 1 row of OP_TEST_LEN)
    {
        uint64_t sm_args[4];
        float sm_out[OP_TEST_LEN];
        sm_args[0] = (uint64_t)(uintptr_t)op_a;
        sm_args[1] = (uint64_t)(uintptr_t)sm_out;
        sm_args[2] = 1;            // num_rows
        sm_args[3] = OP_TEST_LEN;  // row_length
        __host_bingo_kernel_fp32_softmax(sm_args);
        // Verify: all elements > 0 and sum ~ 1.0
        float sm_sum = 0.0f;
        int sm_ok = 1;
        for (int i = 0; i < OP_TEST_LEN; i++) {
            if (sm_out[i] < 0.0f) sm_ok = 0;
            sm_sum += sm_out[i];
        }
        float sm_diff = sm_sum - 1.0f;
        if (sm_diff < 0) sm_diff = -sm_diff;
        if (sm_diff > FP32_TOL_LOOSE) sm_ok = 0;
        if (sm_ok) {
            printf("  PASS: softmax (sum=%.6f)\r\n", sm_sum);
            tests_passed++;
        } else {
            printf("  FAIL: softmax (sum=%.6f, expected 1.0)\r\n", sm_sum);
            tests_failed++;
        }
    }

    // RMSNorm
    {
        float rms_weight[OP_TEST_LEN];
        float rms_out[OP_TEST_LEN];
        for (int i = 0; i < OP_TEST_LEN; i++) rms_weight[i] = 1.0f;  // identity scale
        uint64_t rms_args[5];
        rms_args[0] = (uint64_t)(uintptr_t)op_a;
        rms_args[1] = (uint64_t)(uintptr_t)rms_weight;
        rms_args[2] = (uint64_t)(uintptr_t)rms_out;
        rms_args[3] = OP_TEST_LEN;  // hidden_dim
        rms_args[4] = 1;            // num_tokens
        __host_bingo_kernel_fp32_rmsnorm(rms_args);
        // Verify: output should have unit RMS (approximately)
        float rms_sq = 0.0f;
        for (int i = 0; i < OP_TEST_LEN; i++) rms_sq += rms_out[i] * rms_out[i];
        rms_sq /= OP_TEST_LEN;
        float rms_diff = rms_sq - 1.0f;
        if (rms_diff < 0) rms_diff = -rms_diff;
        if (rms_diff < 0.1f) {
            printf("  PASS: rmsnorm (output RMS^2=%.4f)\r\n", rms_sq);
            tests_passed++;
        } else {
            printf("  FAIL: rmsnorm (output RMS^2=%.4f, expected ~1.0)\r\n", rms_sq);
            tests_failed++;
        }
    }

    // SiLU * up (compound fused kernel)
    {
        float gate[OP_TEST_LEN], up[OP_TEST_LEN], fused_out[OP_TEST_LEN];
        for (int i = 0; i < OP_TEST_LEN; i++) {
            gate[i] = op_mixed[i];
            up[i] = op_a[i];
        }
        uint64_t sm_args[4];
        sm_args[0] = (uint64_t)(uintptr_t)gate;
        sm_args[1] = (uint64_t)(uintptr_t)up;
        sm_args[2] = (uint64_t)(uintptr_t)fused_out;
        sm_args[3] = OP_TEST_LEN;
        __host_bingo_kernel_fp32_silu_mul(sm_args);
        // Just verify it doesn't crash and produces non-zero output
        int nonzero = 0;
        for (int i = 0; i < OP_TEST_LEN; i++) {
            if (fused_out[i] != 0.0f) nonzero++;
        }
        if (nonzero > 0) {
            printf("  PASS: silu_mul (%d/%d nonzero)\r\n", nonzero, OP_TEST_LEN);
            tests_passed++;
        } else {
            printf("  FAIL: silu_mul (all zeros)\r\n");
            tests_failed++;
        }
    }

    // ─── Summary ───────────────────────────────────────────────
    printf("\r\n=== Results: %d passed, %d failed ===\r\n", tests_passed, tests_failed);
    if (tests_failed > 0) {
        printf("SOME TESTS FAILED!\r\n");
        return 1;
    }
    printf("ALL TESTS PASSED!\r\n");
    return 0;
}
