#pragma once

#include <stdint.h>
#include "uart.h"
#include "heterogeneous_runtime.h"
#include "perf_tracing.h"
#include "libbingo/bingo_utils.h"  // bingo_cerf_update()
#define EXIT_CODE_SUCC 1
#define EXIT_CODE_FAIL 2
// Host Bingo Kernel Implementations
// Normally the functions ret with 0
// Only the exit kernel returns the exit code defined by EXIT_CODE_SUCC, for now it is 1
static inline uint64_t __host_bingo_kernel_dummy(void *arg){
    // Arg[0]: dummy_input, Arg[1]: scratchpad_ptr
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint64_t dummy_input = ((uint64_t *)arg)[0];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[1];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    printf_safe("Chip(%x, %x): [Host] Kernel Dummy: %d\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), dummy_input);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    sp->return_value = 0;
    sp->num_return_values = 0;
    return 0;
}

static inline uint64_t __host_bingo_kernel_exit(void *arg){
    // Arg[0]: exit_code, Arg[1]: scratchpad_ptr
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint64_t exit_code = ((uint64_t *)arg)[0];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[1];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    printf_safe("Chip(%x, %x): [Host] Kernel Exit called with exit code %d\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), exit_code);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    sp->return_value = EXIT_CODE_SUCC;
    sp->num_return_values = 0;
    return EXIT_CODE_SUCC;
}

static inline uint64_t __host_bingo_kernel_entry(void *arg){
    // Arg[0]: start_cc_reg_addr, Arg[1]: scratchpad_ptr
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[1];
    uint64_t start_cc;
    asm volatile("csrr %0, mcycle" : "=r"(start_cc));
    printf_safe("Chip(%x, %x): [Host] Start at %d CC\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), start_cc);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    sp->return_value = (uint32_t)start_cc;
    sp->num_return_values = 0;
    return 0;
}
static inline uint64_t __host_bingo_kernel_check_result(void *arg){
    // Arg0-3: golden, output, size, name; Arg4: scratchpad_ptr
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint8_t* golden_data_addr = (uint8_t*)(((uint64_t *)arg)[0]);
    uint8_t* output_data_addr = (uint8_t*)(((uint64_t *)arg)[1]);
    uint64_t data_size = ((uint64_t *)arg)[2];
    const char* name = (const char*)(((uint64_t *)arg)[3]);
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[4];
    if (!name) name = "?";
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    uint32_t err = 0;
    for (uint64_t i = 0; i < data_size; i++) {
        if (output_data_addr[i] != golden_data_addr[i]) {
            err++;
            printf_safe("[%s] output[%d]=%d, golden[%d]=%d\n",
                   name, i, output_data_addr[i], i, golden_data_addr[i]);
        }
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    sp->return_value = err;
    sp->num_return_values = 0;
    if (err == 0) {
        printf_safe("[Host] Check [%s]: PASS (%d bytes)\r\n", name, data_size);
        return 0;
    } else {
        printf_safe("[Host] Check [%s]: FAIL (%d / %d bytes)\r\n", name, err, data_size);
        return EXIT_CODE_FAIL;
    }
}

static inline uint64_t __host_bingo_kernel_idma(void *arg){
    // Arg0-2: src, dst, size; Arg3: scratchpad_ptr
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint64_t src_addr = ((uint64_t *)arg)[0];
    uint64_t dst_addr = ((uint64_t *)arg)[1];
    uint64_t size = ((uint64_t *)arg)[2];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[3];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_IDMA_CFG_START);
    uint64_t tf_id = sys_dma_memcpy(get_current_chip_id(), dst_addr, src_addr, size);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_IDMA_CFG_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_IDMA_RUN_START);
    while (*(sys_dma_done_ptr(get_current_chip_id())) != tf_id) {
        asm volatile("nop");
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_IDMA_RUN_END);
    return 0;
}

// ============================================================
// FP32 CVA6+Ara SIMD kernels — operations that VersaCore cannot handle
// These run on the CVA6 host processor with Ara RVV acceleration.
// Uses RISC-V Vector (RVV) intrinsics following the Ara softmax/exp patterns.
// Requires: enable_vec() called once at startup.
// ============================================================

#include "riscv_vector.h"

// Cephes polynomial exp approximation for f32 (from Ara exp kernel)
// Matches: ara/apps/exp/kernel/exp.c __exp_2xf32
static inline vfloat32m1_t __bingo_exp_f32(vfloat32m1_t x, size_t vl) {
    // Clamp input to avoid overflow/underflow
    vfloat32m1_t exp_hi = __riscv_vfmv_v_f_f32m1(88.3762626647949f, vl);
    vfloat32m1_t exp_lo = __riscv_vfmv_v_f_f32m1(-88.3762626647949f, vl);
    x = __riscv_vfmin_vv_f32m1(x, exp_hi, vl);
    x = __riscv_vfmax_vv_f32m1(x, exp_lo, vl);

    // Express exp(x) = exp(g + n*log(2))
    vfloat32m1_t cephes_LOG2EF = __riscv_vfmv_v_f_f32m1(1.44269504088896341f, vl);
    vfloat32m1_t cephes_exp_C1 = __riscv_vfmv_v_f_f32m1(0.693359375f, vl);
    vfloat32m1_t cephes_exp_C2 = __riscv_vfmv_v_f_f32m1(-2.12194440e-4f, vl);
    vfloat32m1_t half = __riscv_vfmv_v_f_f32m1(0.5f, vl);

    vfloat32m1_t fx = __riscv_vfmacc_vv_f32m1(half, x, cephes_LOG2EF, vl);
    vint32m1_t tmp = __riscv_vfcvt_x_f_v_i32m1(fx, vl);
    vfloat32m1_t ftmp = __riscv_vfcvt_f_x_v_f32m1(tmp, vl);

    // Correct for floor behavior
    vbool32_t mask = __riscv_vmflt_vv_f32m1_b32(fx, ftmp, vl);
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    ftmp = __riscv_vfsub_vv_f32m1(ftmp, __riscv_vmerge_vvm_f32m1(
        __riscv_vfmv_v_f_f32m1(0.0f, vl), one, mask, vl), vl);
    tmp = __riscv_vfcvt_x_f_v_i32m1(ftmp, vl);

    x = __riscv_vfsub_vv_f32m1(x, __riscv_vfmul_vv_f32m1(ftmp, cephes_exp_C1, vl), vl);
    x = __riscv_vfsub_vv_f32m1(x, __riscv_vfmul_vv_f32m1(ftmp, cephes_exp_C2, vl), vl);
    vfloat32m1_t z = __riscv_vfmul_vv_f32m1(x, x, vl);

    // Polynomial approx
    vfloat32m1_t y = __riscv_vfmv_v_f_f32m1(1.9875691500E-4f, vl);
    y = __riscv_vfmadd_vv_f32m1(y, x, __riscv_vfmv_v_f_f32m1(1.3981999507E-3f, vl), vl);
    y = __riscv_vfmadd_vv_f32m1(y, x, __riscv_vfmv_v_f_f32m1(8.3334519073E-3f, vl), vl);
    y = __riscv_vfmadd_vv_f32m1(y, x, __riscv_vfmv_v_f_f32m1(4.1665795894E-2f, vl), vl);
    y = __riscv_vfmadd_vv_f32m1(y, x, __riscv_vfmv_v_f_f32m1(1.6666665459E-1f, vl), vl);
    y = __riscv_vfmadd_vv_f32m1(y, x, __riscv_vfmv_v_f_f32m1(5.0000001201E-1f, vl), vl);
    y = __riscv_vfmacc_vv_f32m1(__riscv_vfadd_vv_f32m1(x, one, vl), y, z, vl);

    // Scale by 2^n via integer addition to exponent bits
    vint32m1_t shift = __riscv_vadd_vv_i32m1(tmp, __riscv_vmv_v_x_i32m1(127, vl), vl);
    shift = __riscv_vsll_vx_i32m1(shift, 23, vl);
    y = __riscv_vfmul_vv_f32m1(y, __riscv_vreinterpret_v_i32m1_f32m1(shift), vl);
    return y;
}

static inline uint64_t __host_bingo_kernel_fp32_rmsnorm(void *arg){
    // RMSNorm: out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
    // Arg0: float* input_addr
    // Arg1: float* weight_addr (gamma)
    // Arg2: float* output_addr
    // Arg3: uint64_t hidden_dim
    // Arg4: uint64_t num_tokens (seq_len)
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float* input  = (float*)(((uint64_t *)arg)[0]);
    float* weight = (float*)(((uint64_t *)arg)[1]);
    float* output = (float*)(((uint64_t *)arg)[2]);
    uint64_t hidden_dim    = ((uint64_t *)arg)[3];
    uint64_t num_tokens    = ((uint64_t *)arg)[4];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    float eps = 1e-6f;
    for (uint64_t t = 0; t < num_tokens; t++) {
        float *x_row = input + t * hidden_dim;
        float *o_row = output + t * hidden_dim;

        // Pass 1: compute sum(x^2) using RVV
        float ss = 0.0f;
        uint64_t avl = hidden_dim;
        float *ptr = x_row;
        for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0; avl -= vl, ptr += vl) {
            vl = __riscv_vsetvl_e32m1(avl);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr, vl);
            vfloat32m1_t sq = __riscv_vfmul_vv_f32m1(v, v, vl);
            // Scalar reduction
            vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t sum_v = __riscv_vfredosum_vs_f32m1_f32m1(sq, zero, vl);
            ss += __riscv_vfmv_f_s_f32m1_f32(sum_v);
        }
        float rms = 1.0f / __builtin_sqrtf(ss / (float)hidden_dim + eps);

        // Pass 2: normalize and scale using RVV
        avl = hidden_dim;
        float *w_ptr = weight;
        float *i_ptr = x_row;
        float *o_ptr = o_row;
        for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0;
             avl -= vl, i_ptr += vl, w_ptr += vl, o_ptr += vl) {
            vl = __riscv_vsetvl_e32m1(avl);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(i_ptr, vl);
            vfloat32m1_t w = __riscv_vle32_v_f32m1(w_ptr, vl);
            vfloat32m1_t scaled = __riscv_vfmul_vf_f32m1(v, rms, vl);
            vfloat32m1_t result = __riscv_vfmul_vv_f32m1(scaled, w, vl);
            __riscv_vse32_v_f32m1(o_ptr, result, vl);
        }
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    return 0;
}

static inline uint64_t __host_bingo_kernel_fp32_softmax(void *arg){
    // Arg0-3: input, output, num_rows, row_length; Arg4: scratchpad_ptr
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float* input  = (float*)(((uint64_t *)arg)[0]);
    float* output = (float*)(((uint64_t *)arg)[1]);
    uint64_t num_rows      = ((uint64_t *)arg)[2];
    uint64_t row_length    = ((uint64_t *)arg)[3];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[4];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    for (uint64_t r = 0; r < num_rows; r++) {
        float *in_row = input + r * row_length;
        float *out_row = output + r * row_length;
        uint64_t avl = row_length;

        // Pass 1: find max (RVV reduction)
        float max_val = in_row[0];
        float *ptr = in_row;
        uint64_t rem = avl;
        for (size_t vl = __riscv_vsetvl_e32m1(rem); rem > 0; rem -= vl, ptr += vl) {
            vl = __riscv_vsetvl_e32m1(rem);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr, vl);
            vfloat32m1_t init = __riscv_vfmv_v_f_f32m1(max_val, vl);
            vfloat32m1_t rmax = __riscv_vfredmax_vs_f32m1_f32m1(v, init, vl);
            max_val = __riscv_vfmv_f_s_f32m1_f32(rmax);
        }

        // Pass 2: exp(x - max) and accumulate sum (RVV)
        float sum = 0.0f;
        ptr = in_row;
        float *optr = out_row;
        rem = avl;
        for (size_t vl = __riscv_vsetvl_e32m1(rem); rem > 0;
             rem -= vl, ptr += vl, optr += vl) {
            vl = __riscv_vsetvl_e32m1(rem);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr, vl);
            vfloat32m1_t shifted = __riscv_vfsub_vf_f32m1(v, max_val, vl);
            vfloat32m1_t expv = __bingo_exp_f32(shifted, vl);
            __riscv_vse32_v_f32m1(optr, expv, vl);
            vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t rsum = __riscv_vfredosum_vs_f32m1_f32m1(expv, zero, vl);
            sum += __riscv_vfmv_f_s_f32m1_f32(rsum);
        }

        // Pass 3: divide by sum (RVV)
        float inv_sum = 1.0f / sum;
        optr = out_row;
        rem = avl;
        for (size_t vl = __riscv_vsetvl_e32m1(rem); rem > 0; rem -= vl, optr += vl) {
            vl = __riscv_vsetvl_e32m1(rem);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(optr, vl);
            vfloat32m1_t result = __riscv_vfmul_vf_f32m1(v, inv_sum, vl);
            __riscv_vse32_v_f32m1(optr, result, vl);
        }
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    // Write output info to scratchpad for successor gating kernels
    sp->return_value = (uint32_t)(uintptr_t)output;
    sp->num_return_values = num_rows * row_length;
    return 0;
}

static inline uint64_t __host_bingo_kernel_fp32_silu_mul(void *arg){
    // SiLU(gate) * up: out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
    // Uses RVV exp intrinsic for vectorized sigmoid
    // Arg0: float* gate_addr
    // Arg1: float* up_addr
    // Arg2: float* output_addr
    // Arg3: uint64_t num_elements
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float* gate   = (float*)(((uint64_t *)arg)[0]);
    float* up     = (float*)(((uint64_t *)arg)[1]);
    float* output = (float*)(((uint64_t *)arg)[2]);
    uint64_t num_elements  = ((uint64_t *)arg)[3];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    uint64_t avl = num_elements;
    float *g_ptr = gate, *u_ptr = up, *o_ptr = output;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0;
         avl -= vl, g_ptr += vl, u_ptr += vl, o_ptr += vl) {
        vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t gv = __riscv_vle32_v_f32m1(g_ptr, vl);
        vfloat32m1_t uv = __riscv_vle32_v_f32m1(u_ptr, vl);
        // sigmoid(g) = 1 / (1 + exp(-g))
        vfloat32m1_t neg_g = __riscv_vfneg_v_f32m1(gv, vl);
        vfloat32m1_t exp_neg = __bingo_exp_f32(neg_g, vl);
        vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
        vfloat32m1_t denom = __riscv_vfadd_vv_f32m1(one, exp_neg, vl);
        vfloat32m1_t sigmoid = __riscv_vfdiv_vv_f32m1(one, denom, vl);
        // silu(g) * up = g * sigmoid(g) * up
        vfloat32m1_t silu = __riscv_vfmul_vv_f32m1(gv, sigmoid, vl);
        vfloat32m1_t result = __riscv_vfmul_vv_f32m1(silu, uv, vl);
        __riscv_vse32_v_f32m1(o_ptr, result, vl);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    return 0;
}

// ---- Generic binary elementwise: out[i] = op(a[i], b[i]) ----
// All share the same arg layout: a_addr, b_addr, output_addr, num_elements

#define DEFINE_FP32_BINARY_KERNEL(name, vec_op)                                 \
static inline uint64_t __host_bingo_kernel_fp32_##name(void *arg){              \
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);                     \
    float* a      = (float*)(((uint64_t *)arg)[0]);                             \
    float* b      = (float*)(((uint64_t *)arg)[1]);                             \
    float* output = (float*)(((uint64_t *)arg)[2]);                             \
    uint64_t num_elements  = ((uint64_t *)arg)[3];                              \
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);                       \
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);                         \
    uint64_t avl = num_elements;                                                \
    float *a_ptr = a, *b_ptr = b, *o_ptr = output;                             \
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0;                       \
         avl -= vl, a_ptr += vl, b_ptr += vl, o_ptr += vl) {                   \
        vl = __riscv_vsetvl_e32m1(avl);                                         \
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a_ptr, vl);                    \
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b_ptr, vl);                    \
        vfloat32m1_t result = vec_op(va, vb, vl);                              \
        __riscv_vse32_v_f32m1(o_ptr, result, vl);                              \
    }                                                                           \
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);                           \
    return 0;                                                                   \
}

DEFINE_FP32_BINARY_KERNEL(add, __riscv_vfadd_vv_f32m1)
DEFINE_FP32_BINARY_KERNEL(sub, __riscv_vfsub_vv_f32m1)
DEFINE_FP32_BINARY_KERNEL(mul, __riscv_vfmul_vv_f32m1)
DEFINE_FP32_BINARY_KERNEL(div, __riscv_vfdiv_vv_f32m1)
DEFINE_FP32_BINARY_KERNEL(max, __riscv_vfmax_vv_f32m1)
DEFINE_FP32_BINARY_KERNEL(min, __riscv_vfmin_vv_f32m1)

// ---- INT32 elementwise add (for inter-cluster partial-D accumulation) ----
// Used when DSE picks K-split tilings: each cluster produces a partial D
// in INT32, and partial Ds from clusters covering the same (M,N) region
// must be summed with this kernel to yield the final INT32 D.
// Arg layout (same as FP32 binary): a_addr, b_addr, output_addr, num_elements
static inline uint64_t __host_bingo_kernel_int32_add(void *arg){
    // Arg0-3: a, b, output, num_elements; Arg4: scratchpad_ptr
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    int32_t* a      = (int32_t*)(((uint64_t *)arg)[0]);
    int32_t* b      = (int32_t*)(((uint64_t *)arg)[1]);
    int32_t* output = (int32_t*)(((uint64_t *)arg)[2]);
    uint64_t num_elements = ((uint64_t *)arg)[3];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[4];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    uint64_t avl = num_elements;
    int32_t *a_ptr = a, *b_ptr = b, *o_ptr = output;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0;
         avl -= vl, a_ptr += vl, b_ptr += vl, o_ptr += vl) {
        vl = __riscv_vsetvl_e32m1(avl);
        vint32m1_t va = __riscv_vle32_v_i32m1(a_ptr, vl);
        vint32m1_t vb = __riscv_vle32_v_i32m1(b_ptr, vl);
        vint32m1_t result = __riscv_vadd_vv_i32m1(va, vb, vl);
        __riscv_vse32_v_i32m1(o_ptr, result, vl);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    sp->return_value = (uint32_t)(uintptr_t)output;
    sp->num_return_values = num_elements;
    return 0;
}

// ---- Generic unary elementwise: out[i] = op(x[i]) ----
// Arg layout: input_addr, output_addr, num_elements

#define DEFINE_FP32_UNARY_KERNEL(name, body)                                    \
static inline uint64_t __host_bingo_kernel_fp32_##name(void *arg){              \
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);                     \
    float* input  = (float*)(((uint64_t *)arg)[0]);                             \
    float* output = (float*)(((uint64_t *)arg)[1]);                             \
    uint64_t num_elements = ((uint64_t *)arg)[2];                               \
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);                       \
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);                         \
    uint64_t avl = num_elements;                                                \
    float *i_ptr = input, *o_ptr = output;                                      \
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0;                       \
         avl -= vl, i_ptr += vl, o_ptr += vl) {                                \
        vl = __riscv_vsetvl_e32m1(avl);                                         \
        vfloat32m1_t v = __riscv_vle32_v_f32m1(i_ptr, vl);                     \
        vfloat32m1_t result;                                                    \
        body                                                                    \
        __riscv_vse32_v_f32m1(o_ptr, result, vl);                              \
    }                                                                           \
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);                           \
    return 0;                                                                   \
}

// relu: max(0, x)
DEFINE_FP32_UNARY_KERNEL(relu, {
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    result = __riscv_vfmax_vv_f32m1(v, zero, vl);
})

// neg: -x
DEFINE_FP32_UNARY_KERNEL(neg, {
    result = __riscv_vfneg_v_f32m1(v, vl);
})

// abs: |x|  (uses neg + max)
DEFINE_FP32_UNARY_KERNEL(abs, {
    vfloat32m1_t neg_v = __riscv_vfneg_v_f32m1(v, vl);
    result = __riscv_vfmax_vv_f32m1(v, neg_v, vl);
})

// exp: exp(x) using Cephes polynomial
DEFINE_FP32_UNARY_KERNEL(exp, {
    result = __bingo_exp_f32(v, vl);
})

// sigmoid: 1 / (1 + exp(-x))
DEFINE_FP32_UNARY_KERNEL(sigmoid, {
    vfloat32m1_t neg_v = __riscv_vfneg_v_f32m1(v, vl);
    vfloat32m1_t exp_neg = __bingo_exp_f32(neg_v, vl);
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    vfloat32m1_t denom = __riscv_vfadd_vv_f32m1(one, exp_neg, vl);
    result = __riscv_vfdiv_vv_f32m1(one, denom, vl);
})

// tanh: (exp(2x) - 1) / (exp(2x) + 1)
DEFINE_FP32_UNARY_KERNEL(tanh, {
    vfloat32m1_t two = __riscv_vfmv_v_f_f32m1(2.0f, vl);
    vfloat32m1_t two_x = __riscv_vfmul_vv_f32m1(v, two, vl);
    vfloat32m1_t exp_2x = __bingo_exp_f32(two_x, vl);
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    vfloat32m1_t num = __riscv_vfsub_vv_f32m1(exp_2x, one, vl);
    vfloat32m1_t den = __riscv_vfadd_vv_f32m1(exp_2x, one, vl);
    result = __riscv_vfdiv_vv_f32m1(num, den, vl);
})

// sqrt: sqrt(x)
DEFINE_FP32_UNARY_KERNEL(sqrt, {
    result = __riscv_vfsqrt_v_f32m1(v, vl);
})

// reciprocal: 1/x
DEFINE_FP32_UNARY_KERNEL(reciprocal, {
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    result = __riscv_vfdiv_vv_f32m1(one, v, vl);
})

// silu: x * sigmoid(x) = x / (1 + exp(-x))
DEFINE_FP32_UNARY_KERNEL(silu, {
    vfloat32m1_t neg_v = __riscv_vfneg_v_f32m1(v, vl);
    vfloat32m1_t exp_neg = __bingo_exp_f32(neg_v, vl);
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    vfloat32m1_t denom = __riscv_vfadd_vv_f32m1(one, exp_neg, vl);
    vfloat32m1_t sig = __riscv_vfdiv_vv_f32m1(one, denom, vl);
    result = __riscv_vfmul_vv_f32m1(v, sig, vl);
})

// gelu: GELU(x) ~ x * sigmoid(1.702 * x) (fast approximation)
DEFINE_FP32_UNARY_KERNEL(gelu, {
    vfloat32m1_t coeff = __riscv_vfmv_v_f_f32m1(1.702f, vl);
    vfloat32m1_t scaled = __riscv_vfmul_vv_f32m1(v, coeff, vl);
    vfloat32m1_t neg_s = __riscv_vfneg_v_f32m1(scaled, vl);
    vfloat32m1_t exp_neg = __bingo_exp_f32(neg_s, vl);
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    vfloat32m1_t denom = __riscv_vfadd_vv_f32m1(one, exp_neg, vl);
    vfloat32m1_t sig = __riscv_vfdiv_vv_f32m1(one, denom, vl);
    result = __riscv_vfmul_vv_f32m1(v, sig, vl);
})

// ---- Reduction kernels ----
// out[0] = reduce_op(x[0..n])
// Arg layout: input_addr, output_addr, num_elements

static inline uint64_t __host_bingo_kernel_fp32_reduce_sum(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float* input  = (float*)(((uint64_t *)arg)[0]);
    float* output = (float*)(((uint64_t *)arg)[1]);
    uint64_t num_elements = ((uint64_t *)arg)[2];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    float acc = 0.0f;
    uint64_t avl = num_elements;
    float *ptr = input;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0; avl -= vl, ptr += vl) {
        vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr, vl);
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t rsum = __riscv_vfredosum_vs_f32m1_f32m1(v, zero, vl);
        acc += __riscv_vfmv_f_s_f32m1_f32(rsum);
    }
    *output = acc;
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    return 0;
}

static inline uint64_t __host_bingo_kernel_fp32_reduce_max(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float* input  = (float*)(((uint64_t *)arg)[0]);
    float* output = (float*)(((uint64_t *)arg)[1]);
    uint64_t num_elements = ((uint64_t *)arg)[2];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    float max_val = input[0];
    uint64_t avl = num_elements;
    float *ptr = input;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0; avl -= vl, ptr += vl) {
        vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr, vl);
        vfloat32m1_t init = __riscv_vfmv_v_f_f32m1(max_val, vl);
        vfloat32m1_t rmax = __riscv_vfredmax_vs_f32m1_f32m1(v, init, vl);
        max_val = __riscv_vfmv_f_s_f32m1_f32(rmax);
    }
    *output = max_val;
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    return 0;
}

static inline uint64_t __host_bingo_kernel_fp32_reduce_mean(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float* input  = (float*)(((uint64_t *)arg)[0]);
    float* output = (float*)(((uint64_t *)arg)[1]);
    uint64_t num_elements = ((uint64_t *)arg)[2];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    float acc = 0.0f;
    uint64_t avl = num_elements;
    float *ptr = input;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0; avl -= vl, ptr += vl) {
        vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr, vl);
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t rsum = __riscv_vfredosum_vs_f32m1_f32m1(v, zero, vl);
        acc += __riscv_vfmv_f_s_f32m1_f32(rsum);
    }
    *output = acc / (float)num_elements;
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    return 0;
}

// ============================================================
// Data type conversion kernels for mixed-precision inference
// FP32 <-> INT8 at the boundary between CVA6 and VersaCore
// ============================================================

static inline uint64_t __host_bingo_kernel_fp32_quantize(void *arg){
    // Arg0-3: input, output, scale_out, num_elements; Arg4: scratchpad_ptr
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float*   input      = (float*)(((uint64_t *)arg)[0]);
    int8_t*  output     = (int8_t*)(((uint64_t *)arg)[1]);
    float*   scale_out  = (float*)(((uint64_t *)arg)[2]);
    uint64_t num_elements = ((uint64_t *)arg)[3];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[4];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);

    // Pass 1: find max(|x|) using RVV
    float abs_max = 0.0f;
    uint64_t avl = num_elements;
    float *ptr = input;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0; avl -= vl, ptr += vl) {
        vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr, vl);
        vfloat32m1_t neg_v = __riscv_vfneg_v_f32m1(v, vl);
        vfloat32m1_t abs_v = __riscv_vfmax_vv_f32m1(v, neg_v, vl);
        vfloat32m1_t init = __riscv_vfmv_v_f_f32m1(abs_max, vl);
        vfloat32m1_t rmax = __riscv_vfredmax_vs_f32m1_f32m1(abs_v, init, vl);
        abs_max = __riscv_vfmv_f_s_f32m1_f32(rmax);
    }

    // Compute scale
    float scale = abs_max / 127.0f;
    if (scale < 1e-10f) scale = 1e-10f;  // avoid div-by-zero for near-zero input
    *scale_out = scale;
    float inv_scale = 1.0f / scale;

    // Pass 2: quantize — scale, round, clamp, narrow to int8
    avl = num_elements;
    ptr = input;
    int8_t *o_ptr = output;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0;
         avl -= vl, ptr += vl, o_ptr += vl) {
        vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr, vl);
        vfloat32m1_t scaled = __riscv_vfmul_vf_f32m1(v, inv_scale, vl);
        // Round to nearest integer
        vint32m1_t rounded = __riscv_vfcvt_x_f_v_i32m1(scaled, vl);
        // Clamp to [-128, 127]
        vint32m1_t lo = __riscv_vmv_v_x_i32m1(-128, vl);
        vint32m1_t hi = __riscv_vmv_v_x_i32m1(127, vl);
        rounded = __riscv_vmax_vv_i32m1(rounded, lo, vl);
        rounded = __riscv_vmin_vv_i32m1(rounded, hi, vl);
        // Narrow int32 -> int8 via scalar extract (safe for initial bring-up)
        for (size_t i = 0; i < vl; i++) {
            int32_t val = __riscv_vmv_x_s_i32m1_i32(
                __riscv_vslidedown_vx_i32m1(rounded, i, vl));
            o_ptr[i] = (int8_t)val;
        }
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    sp->return_value = (uint32_t)(uintptr_t)output;
    sp->num_return_values = num_elements;
    return 0;
}

static inline uint64_t __host_bingo_kernel_int32_dequantize(void *arg){
    // Dequantize INT32 GEMM accumulator to FP32
    // y[i] = int32_input[i] * combined_scale
    // where combined_scale = scale_a * scale_b (pre-computed, stored at scale_addr)
    // Arg0-3: input, output, scale, num_elements; Arg4: scratchpad_ptr
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    int32_t* input     = (int32_t*)(((uint64_t *)arg)[0]);
    float*   output    = (float*)(((uint64_t *)arg)[1]);
    float*   scale_ptr = (float*)(((uint64_t *)arg)[2]);
    uint64_t num_elements = ((uint64_t *)arg)[3];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[4];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    float combined_scale = *scale_ptr;

    uint64_t avl = num_elements;
    int32_t *i_ptr = input;
    float   *o_ptr = output;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0;
         avl -= vl, i_ptr += vl, o_ptr += vl) {
        vl = __riscv_vsetvl_e32m1(avl);
        vint32m1_t vi = __riscv_vle32_v_i32m1(i_ptr, vl);
        vfloat32m1_t vf = __riscv_vfcvt_f_x_v_f32m1(vi, vl);
        vfloat32m1_t result = __riscv_vfmul_vf_f32m1(vf, combined_scale, vl);
        __riscv_vse32_v_f32m1(o_ptr, result, vl);
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    sp->return_value = (uint32_t)(uintptr_t)output;
    sp->num_return_values = num_elements;
    return 0;
}

// ================================================================
// DARTS Unified CERF Gating Kernel
// ================================================================
// Supports multiple activation modes via args->mode:
//   MODE_TOP_K (0):    Read logits from predecessor scratchpad, select top-k
//   MODE_THRESHOLD (1): Read confidence from predecessor scratchpad, activate if < threshold
//   MODE_STATIC (2):   Use compile-time cerf_write_mask directly
//
// Args layout (__host_bingo_kernel_cerf_gating_args_t):
//   [0] mode
//   [1] pred_scratchpad_addr (unused for static)
//   [2] cerf_controlled_mask
//   [3] top_k_or_threshold: top_k | threshold_bits | cerf_write_mask
//   [4] cerf_group_ids_addr: &cerf_group_ids[] | unused (0)
//   [5] cond_activation_addr: per-expert uint8_t[] (SW guard for group sharing)
//   [6] scratchpad_ptr
//
// Two-level gating (for >32 experts with CERF group sharing):
//   Level 1 (HW CERF): Inactive groups skip entire expert clusters at zero cost.
//   Level 2 (SW guard): Within active groups, only selected experts compute.
//     The gating kernel writes 1/0 per expert to cond_activation_addr[].
//     Expert kernels read their slot and early-return if 0.
//     When experts <= 32 (no sharing), cond_activation_addr can be 0 (skip SW guard).
static inline uint64_t __host_bingo_kernel_cerf_gating(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint64_t *args = (uint64_t *)arg;
    uint32_t mode = (uint32_t)args[0];
    bingo_kernel_scratchpad_t* pred_sp = (bingo_kernel_scratchpad_t*)(uintptr_t)args[1];
    uint32_t cerf_controlled_mask = (uint32_t)args[2];
    uint64_t top_k_or_threshold = args[3];
    uint64_t cerf_group_ids_addr = args[4];
    uint8_t *cond_activation = (uint8_t *)(uintptr_t)args[5];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)args[6];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);

    uint32_t cerf_write_mask = 0;

    if (mode == BINGO_GATING_MODE_TOP_K) {
        uint32_t top_k = (uint32_t)top_k_or_threshold;
        uint8_t *cerf_group_ids = (uint8_t *)(uintptr_t)cerf_group_ids_addr;
        float *logits = (float *)(uintptr_t)pred_sp->return_value;
        uint32_t num_experts = pred_sp->num_return_values;
        if (num_experts > 256) num_experts = 256;  // sanity bound

        // Clear per-expert activation array (all experts start as inactive)
        if (cond_activation) {
            for (uint32_t e = 0; e < num_experts; e++)
                cond_activation[e] = 0;
        }

        // Top-k selection: find k experts with highest logit values
        bool used[256] = {false};
        for (uint32_t k = 0; k < top_k; k++) {
            float best = -1e30f;
            uint32_t best_idx = 0;
            for (uint32_t e = 0; e < num_experts; e++) {
                if (!used[e] && logits[e] > best) {
                    best = logits[e];
                    best_idx = e;
                }
            }
            // Level 1: Mark the CERF group of this expert as active (HW skip)
            cerf_write_mask |= (1 << cerf_group_ids[best_idx]);
            // Level 2: Mark this specific expert as active (SW guard)
            if (cond_activation)
                cond_activation[best_idx] = 1;
            used[best_idx] = true;
        }
        BINGO_PRINTF(1, "Chip(%x, %x): [Host] CERF Gating top_k: n=%d, k=%d, ctrl=0x%04x, write=0x%04x\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(),
               num_experts, top_k, cerf_controlled_mask, cerf_write_mask);

    } else if (mode == BINGO_GATING_MODE_THRESHOLD) {
        union { uint32_t u; float f; } thresh_conv;
        thresh_conv.u = (uint32_t)top_k_or_threshold;
        float threshold = thresh_conv.f;
        float *confidence = (float *)(uintptr_t)pred_sp->return_value;

        if (*confidence < threshold) {
            cerf_write_mask = cerf_controlled_mask;  // activate all → continue
        }

    } else if (mode == BINGO_GATING_MODE_STATIC) {
        cerf_write_mask = (uint32_t)top_k_or_threshold;
    }

    // Write CERF registers: single bitmask update (read-modify-write)
    bingo_cerf_update(cerf_controlled_mask, cerf_write_mask);

    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    // Store per-expert activation array pointer in scratchpad so expert kernels
    // can find it via the gating node's scratchpad
    sp->return_value = (uint32_t)(uintptr_t)cond_activation;
    sp->num_return_values = 0;
    return 0;
}

