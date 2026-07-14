#pragma once

#include <stdint.h>
#include "uart.h"
#include "heterogeneous_runtime.h"
#include "perf_tracing.h"
#include "libbingo/bingo_utils.h"
#include "libbingo/host_kernel_args.h"  // BINGO_CHECK_TYPE_*, BINGO_GATING_MODE_*
#include "libbingo/bingo_api.h"         // BINGO_PRINTF, bingo_cerf_update(), BINGO_RET_*

// Host Bingo Kernel Implementations
// Return codes are the unified BINGO_RET_* defined in bingo_api.h:
//   BINGO_RET_SUCC (0): normal completion, scheduler continues
//   BINGO_RET_EXIT (1): graceful termination, scheduler exits (only the exit kernel)
//   BINGO_RET_FAIL (2): failure, scheduler errors out
// Normally the functions ret with BINGO_RET_SUCC.
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
    return BINGO_RET_SUCC;
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
    sp->return_value = BINGO_RET_EXIT;
    sp->num_return_values = 0;
    return BINGO_RET_EXIT;
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
    return BINGO_RET_SUCC;
}
// Union-based type punning for fp32 ↔ uint32 (no strict-aliasing UB, portable, zero-cost).
typedef union { float f; uint32_t u; } __bingo_f32_u32_t;

// Convert IEEE 754 binary16 (half) to binary32 (single) in software.
// Handles zero, subnormal, normal, inf, NaN. No libm / no _Float16 dependency.
static inline float __bingo_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 0x1;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1F;
    uint32_t mant = (uint32_t)(h & 0x3FF);
    uint32_t f_bits;
    if (exp == 0) {
        if (mant == 0) {
            // +/- zero
            f_bits = sign << 31;
        } else {
            // Subnormal: normalize by shifting mantissa left until bit 10 is 1
            int shift = 0;
            while ((mant & 0x400) == 0) { mant <<= 1; shift++; }
            mant &= 0x3FF;  // clear the implicit bit after normalization
            uint32_t e = (uint32_t)(127 - 15 - shift);
            f_bits = (sign << 31) | (e << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        // Inf or NaN
        f_bits = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        // Normal
        uint32_t e = exp - 15 + 127;
        f_bits = (sign << 31) | (e << 23) | (mant << 13);
    }
    __bingo_f32_u32_t u;
    u.u = f_bits;
    return u.f;
}

static inline uint64_t __host_bingo_kernel_check_result(void *arg){
    // Arg0-5: golden, output, size, name, check_type, tolerance_bits; Arg6: scratchpad_ptr
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint8_t* golden_data_addr = (uint8_t*)(((uint64_t *)arg)[0]);
    uint8_t* output_data_addr = (uint8_t*)(((uint64_t *)arg)[1]);
    uint64_t data_size        = ((uint64_t *)arg)[2];
    const char* name          = (const char*)(((uint64_t *)arg)[3]);
    uint64_t check_type       = ((uint64_t *)arg)[4];
    uint32_t tolerance_bits   = (uint32_t)((uint64_t *)arg)[5];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[6];
    if (!name) name = "?";
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);

    // Reinterpret tolerance uint32 bits as fp32 via union.
    __bingo_f32_u32_t tol_u;
    tol_u.u = tolerance_bits;
    const float tolerance = tol_u.f;

    // add a fence here before check to get the latest data in the main mem
    asm volatile("fence" ::: "memory");
    uint32_t err = 0;

    if (check_type == BINGO_CHECK_TYPE_BYTE_EXACT) {
        // Byte-exact comparison over the whole buffer.
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
            return BINGO_RET_SUCC;
        } else {
            printf_safe("[Host] Check [%s]: FAIL (%d / %d bytes)\r\n", name, err, data_size);
            return BINGO_RET_FAIL;
        }
    } else if (check_type == BINGO_CHECK_TYPE_FP32_TOL) {
        // FP32 absolute-tolerance mode
        const float* out_f    = (const float*)output_data_addr;
        const float* golden_f = (const float*)golden_data_addr;
        uint64_t num_elements = data_size / 4;
        for (uint64_t i = 0; i < num_elements; i++) {
            float o = out_f[i];
            float g = golden_f[i];
            float diff = o - g;
            if (diff < 0.0f) diff = -diff;
            if (diff > tolerance) {
                err++;
                // Hex-bits printing to avoid reliance on libc %f support
                __bingo_f32_u32_t uo, ug, ud;
                uo.f = o; ug.f = g; ud.f = diff;
                printf_safe("[%s] idx=%d out=0x%08x golden=0x%08x diff=0x%08x tol=0x%08x\n",
                       name, i, uo.u, ug.u, ud.u, tolerance_bits);
            }
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
        sp->return_value = err;
        sp->num_return_values = 0;
        if (err == 0) {
            printf_safe("[Host] Check [%s]: PASS (%d fp32 elems, tol_bits=0x%08x)\r\n",
                   name, num_elements, tolerance_bits);
            return BINGO_RET_SUCC;
        } else {
            printf_safe("[Host] Check [%s]: FAIL (%d / %d fp32 elems, tol_bits=0x%08x)\r\n",
                   name, err, num_elements, tolerance_bits);
            return BINGO_RET_FAIL;
        }
    } else if (check_type == BINGO_CHECK_TYPE_FP16_TOL) {
        // FP16 absolute-tolerance mode — promote each half to fp32, compare against fp32 tolerance
        const uint16_t* out_h    = (const uint16_t*)output_data_addr;
        const uint16_t* golden_h = (const uint16_t*)golden_data_addr;
        uint64_t num_elements = data_size / 2;
        for (uint64_t i = 0; i < num_elements; i++) {
            uint16_t oh = out_h[i];
            uint16_t gh = golden_h[i];
            float o = __bingo_fp16_to_fp32(oh);
            float g = __bingo_fp16_to_fp32(gh);
            float diff = o - g;
            if (diff < 0.0f) diff = -diff;
            if (diff > tolerance) {
                err++;
                __bingo_f32_u32_t uo, ug, ud;
                uo.f = o; ug.f = g; ud.f = diff;
                printf_safe("[%s] idx=%d out_h=0x%04x golden_h=0x%04x out_f=0x%08x golden_f=0x%08x diff=0x%08x tol=0x%08x\n",
                       name, i, oh, gh, uo.u, ug.u, ud.u, tolerance_bits);
            }
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
        sp->return_value = err;
        sp->num_return_values = 0;
        if (err == 0) {
            printf_safe("[Host] Check [%s]: PASS (%d fp16 elems, tol_bits=0x%08x)\r\n",
                   name, num_elements, tolerance_bits);
            return BINGO_RET_SUCC;
        } else {
            printf_safe("[Host] Check [%s]: FAIL (%d / %d fp16 elems, tol_bits=0x%08x)\r\n",
                   name, err, num_elements, tolerance_bits);
            return BINGO_RET_FAIL;
        }
    } else if (check_type == BINGO_CHECK_TYPE_FP16_RELTOL) {
        // FP16 combined-tolerance mode (numpy.allclose style for a quantized datapath):
        //   |out-golden| <= rtol*|golden| + atol,  atol = max(0.05, 0.01 * max|golden_tensor|).
        // `tolerance` holds rtol. The dynamic-range-relative abs floor (1% of the tensor's scale)
        // is essential where out = a - b CANCELS (e.g. h + down, |a|,|b|>>|out|): down carries an
        // inherent ~1-int8-LSB absolute error that is tiny vs down but huge vs the near-zero out,
        // so a purely per-element relative bound is the wrong yardstick there. A real bug gives
        // errors O(|g|) >> this floor, so it is still caught.
        const uint16_t* out_h    = (const uint16_t*)output_data_addr;
        const uint16_t* golden_h = (const uint16_t*)golden_data_addr;
        uint64_t num_elements = data_size / 2;
        float max_ag = 0.0f;                       // pass 1: dynamic range of the golden tensor
        for (uint64_t i = 0; i < num_elements; i++) {
            float g = __bingo_fp16_to_fp32(golden_h[i]);
            float ag = g < 0.0f ? -g : g;
            if (ag > max_ag) max_ag = ag;
        }
        float abs_floor = 0.01f * max_ag;
        if (abs_floor < 0.05f) abs_floor = 0.05f;
        for (uint64_t i = 0; i < num_elements; i++) {
            uint16_t oh = out_h[i];
            uint16_t gh = golden_h[i];
            float o = __bingo_fp16_to_fp32(oh);
            float g = __bingo_fp16_to_fp32(gh);
            float diff = o - g;
            if (diff < 0.0f) diff = -diff;
            float ag = g < 0.0f ? -g : g;
            float thresh = tolerance * ag + abs_floor;
            if (diff > thresh) {
                err++;
                __bingo_f32_u32_t uo, ug, ud, ut;
                uo.f = o; ug.f = g; ud.f = diff; ut.f = thresh;
                printf_safe("[%s] idx=%d out_h=0x%04x golden_h=0x%04x out_f=0x%08x golden_f=0x%08x diff=0x%08x thresh=0x%08x\n",
                       name, i, oh, gh, uo.u, ug.u, ud.u, ut.u);
            }
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
        sp->return_value = err;
        sp->num_return_values = 0;
        if (err == 0) {
            printf_safe("[Host] Check [%s]: PASS (%d fp16 elems, rtol_bits=0x%08x)\r\n",
                   name, num_elements, tolerance_bits);
            return BINGO_RET_SUCC;
        } else {
            printf_safe("[Host] Check [%s]: FAIL (%d / %d fp16 elems, rtol_bits=0x%08x)\r\n",
                   name, err, num_elements, tolerance_bits);
            // NON-FATAL during bring-up: return SUCC so a failing intermediate check does not
            // abort the run -- we want EVERY stage's verdict (incl. the final `out`) in one sim.
            // The failure is still printed and recorded in sp->return_value.
            return BINGO_RET_SUCC;
        }
    } else {
        // Unknown check_type — fail loudly
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
        printf_safe("[Host] Check [%s]: FAIL (unknown check_type=%d)\r\n", name, (int)check_type);
        sp->return_value = 1;
        sp->num_return_values = 0;
        return BINGO_RET_FAIL;
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
    return BINGO_RET_SUCC;
}

static inline uint64_t __host_bingo_kernel_xdma_1d_copy(void *arg){
    // Arg0-2: src, dst, size; Arg3: scratchpad_ptr
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint64_t src_addr = ((uint64_t *)arg)[0];
    uint64_t dst_addr = ((uint64_t *)arg)[1];
    uint64_t size = ((uint64_t *)arg)[2];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[3];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    if (size > UINT32_MAX) {
        printf_safe("[Host] xDMA 1D copy size too large: 0x%lx bytes\r\n", size);
        sp->return_value = 1;
        sp->num_return_values = 0;
        return BINGO_RET_FAIL;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
    int32_t cfg_ret = hemaia_xdma_memcpy_1d((const void*)(uintptr_t)src_addr,
                                            (void*)(uintptr_t)dst_addr,
                                            (uint32_t)size);
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);
    if (cfg_ret != 0) {
        printf_safe("[Host] xDMA 1D copy config failed: %d\r\n", cfg_ret);
        sp->return_value = (uint32_t)cfg_ret;
        sp->num_return_values = 0;
        return BINGO_RET_FAIL;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
    uint32_t task_id = hemaia_xdma_start();
    // !!! Note: only work when the src and dst memory is not the same!!!
    hemaia_xdma_remote_wait(task_id);
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
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
//
// Defined for BOTH LMUL=1 and LMUL=2 from one body. The fp32 kernels compute at f32m1;
// the fp16 kernels widen f16m1 -> f32m2 so an iteration covers VLEN/16 elements instead
// of VLEN/32 (see __BINGO_UNARY_F16_VIA_F32). Keeping one source for the polynomial is
// the point -- the two instantiations cannot drift.
//   L = LMUL suffix (m1 / m2); B = the mask type's SEW/LMUL ratio (32 for m1, 16 for m2).
#define __BINGO_DEF_EXP_F32(L, B)                                                            \
static inline vfloat32##L##_t __bingo_exp_f32_##L(vfloat32##L##_t x, size_t vl) {            \
    /* Clamp input to avoid overflow/underflow */                                            \
    vfloat32##L##_t exp_hi = __riscv_vfmv_v_f_f32##L(88.3762626647949f, vl);                 \
    vfloat32##L##_t exp_lo = __riscv_vfmv_v_f_f32##L(-88.3762626647949f, vl);                \
    x = __riscv_vfmin_vv_f32##L(x, exp_hi, vl);                                              \
    x = __riscv_vfmax_vv_f32##L(x, exp_lo, vl);                                              \
    /* Express exp(x) = exp(g + n*log(2)) */                                                 \
    vfloat32##L##_t cephes_LOG2EF = __riscv_vfmv_v_f_f32##L(1.44269504088896341f, vl);       \
    vfloat32##L##_t cephes_exp_C1 = __riscv_vfmv_v_f_f32##L(0.693359375f, vl);               \
    vfloat32##L##_t cephes_exp_C2 = __riscv_vfmv_v_f_f32##L(-2.12194440e-4f, vl);            \
    vfloat32##L##_t half = __riscv_vfmv_v_f_f32##L(0.5f, vl);                                \
    vfloat32##L##_t fx = __riscv_vfmacc_vv_f32##L(half, x, cephes_LOG2EF, vl);               \
    vint32##L##_t tmp = __riscv_vfcvt_x_f_v_i32##L(fx, vl);                                  \
    vfloat32##L##_t ftmp = __riscv_vfcvt_f_x_v_f32##L(tmp, vl);                              \
    /* Correct for floor behavior */                                                         \
    vbool##B##_t mask = __riscv_vmflt_vv_f32##L##_b##B(fx, ftmp, vl);                        \
    vfloat32##L##_t one = __riscv_vfmv_v_f_f32##L(1.0f, vl);                                 \
    ftmp = __riscv_vfsub_vv_f32##L(ftmp, __riscv_vmerge_vvm_f32##L(                          \
        __riscv_vfmv_v_f_f32##L(0.0f, vl), one, mask, vl), vl);                              \
    tmp = __riscv_vfcvt_x_f_v_i32##L(ftmp, vl);                                              \
    x = __riscv_vfsub_vv_f32##L(x, __riscv_vfmul_vv_f32##L(ftmp, cephes_exp_C1, vl), vl);    \
    x = __riscv_vfsub_vv_f32##L(x, __riscv_vfmul_vv_f32##L(ftmp, cephes_exp_C2, vl), vl);    \
    vfloat32##L##_t z = __riscv_vfmul_vv_f32##L(x, x, vl);                                   \
    /* Polynomial approx */                                                                  \
    vfloat32##L##_t y = __riscv_vfmv_v_f_f32##L(1.9875691500E-4f, vl);                       \
    y = __riscv_vfmadd_vv_f32##L(y, x, __riscv_vfmv_v_f_f32##L(1.3981999507E-3f, vl), vl);   \
    y = __riscv_vfmadd_vv_f32##L(y, x, __riscv_vfmv_v_f_f32##L(8.3334519073E-3f, vl), vl);   \
    y = __riscv_vfmadd_vv_f32##L(y, x, __riscv_vfmv_v_f_f32##L(4.1665795894E-2f, vl), vl);   \
    y = __riscv_vfmadd_vv_f32##L(y, x, __riscv_vfmv_v_f_f32##L(1.6666665459E-1f, vl), vl);   \
    y = __riscv_vfmadd_vv_f32##L(y, x, __riscv_vfmv_v_f_f32##L(5.0000001201E-1f, vl), vl);   \
    y = __riscv_vfmacc_vv_f32##L(__riscv_vfadd_vv_f32##L(x, one, vl), y, z, vl);             \
    /* Scale by 2^n via integer addition to exponent bits */                                 \
    vint32##L##_t shift = __riscv_vadd_vv_i32##L(tmp, __riscv_vmv_v_x_i32##L(127, vl), vl);  \
    shift = __riscv_vsll_vx_i32##L(shift, 23, vl);                                           \
    y = __riscv_vfmul_vv_f32##L(y, __riscv_vreinterpret_v_i32##L##_f32##L(shift), vl);       \
    return y;                                                                                \
}
__BINGO_DEF_EXP_F32(m1, 32)
__BINGO_DEF_EXP_F32(m2, 16)

// The fp32 kernels call the poly at LMUL=1; keep their existing spelling.
#define __bingo_exp_f32(x, vl) __bingo_exp_f32_m1((x), (vl))

static inline uint64_t __host_bingo_kernel_rmsnorm_f32(void *arg){
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

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
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
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    return BINGO_RET_SUCC;
}

static inline uint64_t __host_bingo_kernel_softmax_f32(void *arg){
    // Arg0-3: input, output, num_rows, row_length; Arg4: scratchpad_ptr
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float* input  = (float*)(((uint64_t *)arg)[0]);
    float* output = (float*)(((uint64_t *)arg)[1]);
    uint64_t num_rows      = ((uint64_t *)arg)[2];
    uint64_t row_length    = ((uint64_t *)arg)[3];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[4];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
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
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    // Write output info to scratchpad for successor gating kernels
    sp->return_value = (uint32_t)(uintptr_t)output;
    sp->num_return_values = num_rows * row_length;
    return BINGO_RET_SUCC;
}

static inline uint64_t __host_bingo_kernel_silu_mul_f32(void *arg){
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

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
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
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    return BINGO_RET_SUCC;
}

// ---- Generic binary elementwise: out[i] = op(a[i], b[i]) ----
// All share the same arg layout: a_addr, b_addr, output_addr, num_elements

#define DEFINE_FP32_BINARY_KERNEL(name, vec_op)                                 \
static inline uint64_t __host_bingo_kernel_##name##_f32(void *arg){              \
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);                     \
    float* a      = (float*)(((uint64_t *)arg)[0]);                             \
    float* b      = (float*)(((uint64_t *)arg)[1]);                             \
    float* output = (float*)(((uint64_t *)arg)[2]);                             \
    uint64_t num_elements  = ((uint64_t *)arg)[3];                              \
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);                       \
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);                         \
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
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);                           \
    return BINGO_RET_SUCC;                                                                   \
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
static inline uint64_t __host_bingo_kernel_add_i32(void *arg){
    // Arg0-3: a, b, output, num_elements; Arg4: precision (ignored); Arg5: scratchpad_ptr
    // (reads the unified ara_binary_args layout; precision is a no-op for int32 add)
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    int32_t* a      = (int32_t*)(((uint64_t *)arg)[0]);
    int32_t* b      = (int32_t*)(((uint64_t *)arg)[1]);
    int32_t* output = (int32_t*)(((uint64_t *)arg)[2]);
    uint64_t num_elements = ((uint64_t *)arg)[3];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[5];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
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
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    sp->return_value = (uint32_t)(uintptr_t)output;
    sp->num_return_values = num_elements;
    return BINGO_RET_SUCC;
}

// ---- Generic unary elementwise: out[i] = op(x[i]) ----
// Arg layout: input_addr, output_addr, num_elements

#define DEFINE_FP32_UNARY_KERNEL(name, body)                                    \
static inline uint64_t __host_bingo_kernel_##name##_f32(void *arg){              \
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);                     \
    float* input  = (float*)(((uint64_t *)arg)[0]);                             \
    float* output = (float*)(((uint64_t *)arg)[1]);                             \
    uint64_t num_elements = ((uint64_t *)arg)[2];                               \
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);                       \
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);                         \
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
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);                           \
    return BINGO_RET_SUCC;                                                                   \
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

// fp16 <-> fp32 bit conversions (integer-only, so no scalar zfh needed).
static inline float __bingo_host_f16bits_to_f32(uint16_t h){
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)(h & 0x3FFu);
    uint32_t bits;
    if (exp == 0u) {
        if (mant == 0u) { bits = sign << 31; }
        else { int s = 0; while (!(mant & 0x400u)) { mant <<= 1; s++; }
               mant &= 0x3FFu; bits = (sign << 31) | ((uint32_t)(127 - 15 - s) << 23) | (mant << 13); }
    } else if (exp == 0x1Fu) {
        bits = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float f; __builtin_memcpy(&f, &bits, sizeof f); return f;
}

// f32 -> fp16 bits, round-to-nearest-even. Integer-only (no scalar zfh needed).
static inline uint16_t __bingo_host_f32_to_f16bits(float f){
    uint32_t x; __builtin_memcpy(&x, &f, sizeof x);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t e    = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;
    if (e == 0xFFu) return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));  // Inf/NaN
    int32_t exp = (int32_t)e - 127 + 15;
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);                        // overflow -> Inf
    if (exp <= 0) {                                                            // subnormal/underflow
        if (exp < -10) return (uint16_t)sign;                                  // too small -> 0
        mant |= 0x800000u;
        uint32_t shift   = (uint32_t)(14 - exp);
        uint32_t half    = mant >> shift;
        uint32_t rem     = mant & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1u))) half++;          // RNE
        return (uint16_t)(sign | half);
    }
    uint16_t half = (uint16_t)((exp << 10) | (mant >> 13));
    uint32_t rem  = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half++;              // RNE (carries up)
    return (uint16_t)(sign | half);
}

// True if `addr` is in a host-cacheable main-memory region (WideSPM / NarrowSPM),
// where the device wrote through the global map and a `fence` (WT-dcache
// invalidate; DcacheFlushOnFence=1) gives the host a coherent read. False => the
// address is in cluster TCDM / other non-cacheable PMA (L1): so it must be staged
// into main memory via the system iDMA first. Chip-id lives in addr bits >= 40
// (get_chip_baseaddress = chip_id << 40); strip it before the range compare.
static inline int __bingo_host_addr_is_mainmem(uint64_t addr){
    uint64_t local = addr & ((1ULL << 40) - 1);
    return ((local >= (uint64_t)SPM_WIDE_BASE_ADDR) &&
            (local <  (uint64_t)SPM_WIDE_BASE_ADDR   + WIDE_SPM_SIZE)) ||
           ((local >= (uint64_t)SPM_NARROW_BASE_ADDR) &&
            (local <  (uint64_t)SPM_NARROW_BASE_ADDR + NARROW_SPM_SIZE));
}
// Per-tensor fp16->int8 REQUANT SCALE. The xDMA computes max|x| = max(MAX(x), MAX(-x)) via two
// StreamReduce(MAX) passes writing one splatted fp16 scalar each (lane 0 of a 64-B beat) to cluster L1;
// this tiny host kernel reads those two scalars, computes scale = max|x|/127 (the fp32 dequant qsc a
// downstream op multiplies by) and inv_scale = 127/max|x| (the fp32 the xDMA fp16_to_int8 narrow reads
// as a runtime CSR), and writes them out. It reads/writes 2+2 scalars, NOT the 4096-element tensor the
// host quantize_f16i8 streamed from L3 -- so it is ~free and moves the whole requant onto the xDMA.
// Uses the usual cluster-L1 bounce: read/write in place if host-cacheable main memory,
// else stage through the system iDMA + a fence.
static inline uint64_t __host_bingo_kernel_requant_scale(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint64_t xmax_addr  = ((uint64_t *)arg)[0];   // L1: lane 0 of the MAX(x) reduce beat  (fp16)
    uint64_t nmax_addr  = ((uint64_t *)arg)[1];   // L1: lane 0 of the MAX(-x) reduce beat (fp16)
    uint64_t scale_addr = ((uint64_t *)arg)[2];   // out: fp32 scale = max|x|/127 (downstream qsc)
    uint64_t inv_addr   = ((uint64_t *)arg)[3];   // out: fp32 inv_scale = 127/max|x| (fp16_to_int8 CSR)
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[4];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_QUANT_RUN_START);
    uint8_t chip = get_current_chip_id();
    static volatile uint32_t rq_bounce;            // host .bss -> WideSPM (holds one u16 or f32)
    // read one cluster-L1 fp16 scalar (lane 0), bouncing through main memory if the addr is cluster TCDM
    #define RQ_READ_F16(addr) ({ \
        uint64_t _a = (addr); const volatile uint16_t* _p; \
        if (__bingo_host_addr_is_mainmem(_a)) { _p = (const volatile uint16_t*)(uintptr_t)_a; } \
        else { uint32_t _tf = (uint32_t)sys_dma_memcpy(chip, (uint64_t)(uintptr_t)&rq_bounce, _a, \
                   sizeof(uint16_t)); while (*(sys_dma_done_ptr(chip)) != _tf) asm volatile("nop"); \
               _p = (const volatile uint16_t*)&rq_bounce; } \
        asm volatile("fence" ::: "memory"); __bingo_host_f16bits_to_f32(*_p); })
    float xmax = RQ_READ_F16(xmax_addr);
    float nmax = RQ_READ_F16(nmax_addr);
    #undef RQ_READ_F16
    float maxabs = xmax > nmax ? xmax : nmax;      // both are max of a non-negated / negated stream
    if (maxabs < 1e-10f) maxabs = 1e-10f;          // match _requant_f16i8's floor (avoid div-by-zero)
    float scale = maxabs / 127.0f, inv_scale = 127.0f / maxabs;
    // write one fp32 out (in place if main memory, else stage + system-iDMA to cluster L1)
    #define RQ_WRITE_F32(addr, val) do { \
        uint64_t _a = (addr); float _v = (val); \
        if (__bingo_host_addr_is_mainmem(_a)) { __builtin_memcpy((void*)(uintptr_t)_a, &_v, 4); \
            asm volatile("fence" ::: "memory"); } \
        else { __builtin_memcpy((void*)&rq_bounce, &_v, 4); asm volatile("fence" ::: "memory"); \
            uint32_t _tf = (uint32_t)sys_dma_memcpy(chip, _a, (uint64_t)(uintptr_t)&rq_bounce, 4); \
            while (*(sys_dma_done_ptr(chip)) != _tf) asm volatile("nop"); } } while (0)
    RQ_WRITE_F32(scale_addr, scale);
    RQ_WRITE_F32(inv_addr, inv_scale);
    #undef RQ_WRITE_F32
    BINGO_TRACE_MARKER(BINGO_TRACE_QUANT_RUN_END);
    sp->return_value = (uint32_t)(uintptr_t)inv_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// ---- Reduction kernels ----
// out[0] = reduce_op(x[0..n])
// Arg layout: input_addr, output_addr, num_elements

static inline uint64_t __host_bingo_kernel_reduce_sum_f32(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float* input  = (float*)(((uint64_t *)arg)[0]);
    float* output = (float*)(((uint64_t *)arg)[1]);
    uint64_t num_elements = ((uint64_t *)arg)[2];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
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
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    return BINGO_RET_SUCC;
}

static inline uint64_t __host_bingo_kernel_reduce_max_f32(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float* input  = (float*)(((uint64_t *)arg)[0]);
    float* output = (float*)(((uint64_t *)arg)[1]);
    uint64_t num_elements = ((uint64_t *)arg)[2];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
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
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    return BINGO_RET_SUCC;
}

static inline uint64_t __host_bingo_kernel_reduce_mean_f32(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float* input  = (float*)(((uint64_t *)arg)[0]);
    float* output = (float*)(((uint64_t *)arg)[1]);
    uint64_t num_elements = ((uint64_t *)arg)[2];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
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
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    return BINGO_RET_SUCC;
}

// ============================================================
// Data type conversion kernels for mixed-precision inference
// FP32 <-> INT8 at the boundary between CVA6 and VersaCore
// (FP16 -> INT8 quantize lives further down, after the BINGO_HAVE_FP16_VEC
//  block: see __host_bingo_kernel_quantize_f16i8.)
// ============================================================

// Narrow i32 -> i16 -> i8, one width step each, with a DISTINCT destination register.
//
// Ara HW limit (the same one __bingo_narrow_f32m1_f16mf2 works around): a narrowing op
// whose destination register overlaps the low part of the wide source register group --
// spec-legal, and exactly what the compiler emits for __riscv_vncvt_x_x_w_* ("vnsrl.wi
// v1,v1,0") -- returns WRONG data on Ara. The early-clobber ("=&vr") forces the
// destination clear of the source, which is why the narrow is written as inline asm here
// instead of the plain intrinsic.
// The vsetvli lives inside the asm because the asm is opaque to the compiler's vtype
// tracking; the caller's next vector op re-establishes vtype as needed.
static inline vint16mf2_t __bingo_narrow_i32m1_i16mf2(vint32m1_t w, size_t vl) {
    vint16mf2_t d;
    asm volatile("vsetvli zero, %2, e16, mf2, ta, ma\n\t"
                 "vnsrl.wi %0, %1, 0"
                 : "=&vr"(d) : "vr"(w), "r"(vl));
    return d;
}
static inline vint8mf4_t __bingo_narrow_i16mf2_i8mf4(vint16mf2_t w, size_t vl) {
    vint8mf4_t d;
    asm volatile("vsetvli zero, %2, e8, mf4, ta, ma\n\t"
                 "vnsrl.wi %0, %1, 0"
                 : "=&vr"(d) : "vr"(w), "r"(vl));
    return d;
}

static inline uint64_t __host_bingo_kernel_quantize_f32i8(void *arg){
    // Arg0-3: input, output, scale_out, num_elements; Arg4: precision (ignored); Arg5: scratchpad_ptr
    // (reads the unified ara_convert_args layout; precision is a no-op for the conversion)
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float*   input      = (float*)(((uint64_t *)arg)[0]);
    int8_t*  output     = (int8_t*)(((uint64_t *)arg)[1]);
    float*   scale_out  = (float*)(((uint64_t *)arg)[2]);
    uint64_t num_elements = ((uint64_t *)arg)[3];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[5];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_QUANT_RUN_START);

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
        // Narrow int32 -> int8 with two plain narrowing shifts (vncvt == vnsrl by 0).
        //
        // NOT vnclip: the saturating narrow would fold the clamp above into itself, but
        // it is a FIXED-POINT op, so the intrinsic first writes the vxrm CSR -- and Ara
        // does not implement vxrm. The csrwi traps (Illegal Instruction, tval=0x00a05073)
        // in the prologue the compiler hoists it into. Keep the explicit clamp; the
        // narrow truncates.
        //
        // The helpers (not the bare __riscv_vncvt_x_x_w_* intrinsics) because the plain
        // intrinsic lets the compiler pick an in-place destination, which Ara miscomputes.
        vint16mf2_t n16 = __bingo_narrow_i32m1_i16mf2(rounded, vl);
        vint8mf4_t  n8  = __bingo_narrow_i16mf2_i8mf4(n16, vl);
        __riscv_vse8_v_i8mf4(o_ptr, n8, vl);
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_QUANT_RUN_END);
    sp->return_value = (uint32_t)(uintptr_t)output;
    sp->num_return_values = num_elements;
    return BINGO_RET_SUCC;
}

static inline uint64_t __host_bingo_kernel_dequantize_i32f32(void *arg){
    // Dequantize INT32 GEMM accumulator to FP32
    // y[i] = int32_input[i] * combined_scale
    // where combined_scale = scale_a * scale_b (pre-computed, stored at scale_addr)
    // Arg0-3: input, output, scale, num_elements; Arg4: precision (ignored); Arg5: scratchpad_ptr
    // (reads the unified ara_convert_args layout; precision is a no-op for the conversion)
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    int32_t* input     = (int32_t*)(((uint64_t *)arg)[0]);
    float*   output    = (float*)(((uint64_t *)arg)[1]);
    float*   scale_ptr = (float*)(((uint64_t *)arg)[2]);
    uint64_t num_elements = ((uint64_t *)arg)[3];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[5];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
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

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    sp->return_value = (uint32_t)(uintptr_t)output;
    sp->num_return_values = num_elements;
    return BINGO_RET_SUCC;
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
    return BINGO_RET_SUCC;
}

// ============================================================
// Multi-precision Ara kernels (precision passed as a runtime arg)
// ============================================================
// The typed __host_bingo_kernel_<op>_f32 / _i32 kernels (and quantize_f32i8 /
// dequantize_i32f32) above are the FP32/INT32 entry points used by ci_ara and the
// bingo graph. The dispatchers below add fp16 / int8 / int16 variants for the
// precision sweep. Each dispatcher reads a BINGO_PREC_* word from a fixed arg
// slot; for BINGO_PREC_FP32 it delegates to the matching typed kernel so the
// FP32 path is byte-for-byte identical.
//
// Applicability (see ara_sweep.h for the per-op precision lists the apps use):
//   - fp16: all float ops. Elementwise/native-unary/reduce_max use native f16
//           vectors; transcendentals and the compound ops (softmax/rmsnorm/
//           silu_mul) widen f16->f32 (vfwcvt), reuse the fp32 math, then narrow
//           f32->f16 (vfncvt) so the fp32 polynomial constants are reused.
//   - int8/int16: only integer-meaningful ops (add/sub/mul/max/min, relu/neg/abs,
//           reduce_sum/reduce_max). reduce_sum widens to an int32 accumulator
//           (overflow-safe for any VLEN); reduce_* write their scalar as int32.
// Unsupported (op, precision) combos return BINGO_RET_FAIL (defensive; the sweep
// apps never iterate them).

#if defined(__riscv_zvfh) && !defined(BINGO_HAVE_FP16_VEC)
#define BINGO_HAVE_FP16_VEC 1
#endif
#ifndef BINGO_HAVE_FP16_VEC
#define BINGO_HAVE_FP16_VEC 0
#endif

#if BINGO_HAVE_FP16_VEC
// Narrow one f32m1 vector to f16mf2. WORKAROUND for an Ara HW limit: the in-place
// narrowing fp conversion the compiler emits for __riscv_vfncvt_f_f_w_f16mf2()
// (e.g. "vfncvt.f.f.w v4,v4", dest reg overlapping the lowest part of the wide
// source reg group -- spec-legal) produces a wrong f32->f16 result on Ara. Every
// transcendental/compound fp16 kernel (exp/sigmoid/sqrt/tanh/reciprocal/gelu/silu/
// silu_mul/softmax/rmsnorm) goes through this narrow, so all of them depend on it.
// Forcing the destination into a register distinct from the source (early-clobber
// "=&vr") computes correctly.
// The vsetvli is inside the asm because the asm is opaque to the compiler's vtype
// tracking; the caller's next vector op re-establishes vtype as needed.
static inline vfloat16mf2_t __bingo_narrow_f32m1_f16mf2(vfloat32m1_t w, size_t vl) {
    vfloat16mf2_t d;
    asm volatile("vsetvli zero, %2, e16, mf2, ta, ma\n\t"
                 "vfncvt.f.f.w %0, %1"
                 : "=&vr"(d) : "vr"(w), "r"(vl));
    return d;
}
// LMUL=2 sibling: narrow f32m2 -> f16m1, for the fp16 paths that widen at e16m1 (VLEN/16
// elements per iteration) instead of e16mf2 (VLEN/32). Same Ara in-place-narrow HW bug,
// same early-clobber workaround -- only the vtype and the register groups differ.
static inline vfloat16m1_t __bingo_narrow_f32m2_f16m1(vfloat32m2_t w, size_t vl) {
    vfloat16m1_t d;
    asm volatile("vsetvli zero, %2, e16, m1, ta, ma\n\t"
                 "vfncvt.f.f.w %0, %1"
                 : "=&vr"(d) : "vr"(w), "r"(vl));
    return d;
}
#endif

// ============================================================
// FP16 -> INT8 quantize. Sibling of __host_bingo_kernel_quantize_f32i8
// (see the FP32<->INT8 conversion section above); lives here because it
// uses the BINGO_HAVE_FP16_VEC widen path defined just above. Identical
// per-tensor symmetric scheme (scale = max(|x|)/127), only the input load
// differs: FP16 is widened to FP32 (vfwcvt) before reusing the f32 math.
// ============================================================
static inline uint64_t __host_bingo_kernel_quantize_f16i8(void *arg){
    // Arg0-3: input(fp16), output(int8), scale_out, num_elements; Arg4: precision (ignored); Arg5: scratchpad_ptr
    // (reads the unified ara_convert_args layout; precision is a no-op for the conversion)
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint16_t* input      = (uint16_t*)(((uint64_t *)arg)[0]);
    int8_t*   output     = (int8_t*)(((uint64_t *)arg)[1]);
    float*    scale_out  = (float*)(((uint64_t *)arg)[2]);
    uint64_t  num_elements = ((uint64_t *)arg)[3];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[5];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_QUANT_RUN_START);

#if BINGO_HAVE_FP16_VEC
    // NOTE ON LMUL: this kernel stays at e16mf2 -> f32m1, even though that retires only
    // VLEN/32 elements per iteration (the same as the fp32 sibling) and so gets no
    // width benefit from the fp16 input. The natural fix -- load e16m1 and widen to
    // f32m2 -- is what the fp16 UNARY kernels do and it works there, but pass 1 here is
    // a REDUCTION, and Ara does not complete a reduction with an LMUL>1 source: it
    // accepts vfredmax.vs with an f32m2 operand and then silently never retires it
    // (a hang, with no illegal-instruction trap). See __bingo_reduce_sum_f16.

    // Pass 1: widen fp16->fp32, find max(|x|) using RVV
    float abs_max = 0.0f;
    uint64_t avl = num_elements;
    _Float16 *ptr = (_Float16*)input;
    for (size_t vl = __riscv_vsetvl_e16mf2(avl); avl > 0; avl -= vl, ptr += vl) {
        vl = __riscv_vsetvl_e16mf2(avl);
        vfloat32m1_t v = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(ptr, vl), vl);
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

    // Pass 2: widen fp16->fp32, scale, round, clamp, narrow to int8
    avl = num_elements;
    ptr = (_Float16*)input;
    int8_t *o_ptr = output;
    for (size_t vl = __riscv_vsetvl_e16mf2(avl); avl > 0;
         avl -= vl, ptr += vl, o_ptr += vl) {
        vl = __riscv_vsetvl_e16mf2(avl);
        vfloat32m1_t v = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(ptr, vl), vl);
        vfloat32m1_t scaled = __riscv_vfmul_vf_f32m1(v, inv_scale, vl);
        // Round to nearest integer
        vint32m1_t rounded = __riscv_vfcvt_x_f_v_i32m1(scaled, vl);
        // Clamp to [-128, 127]
        vint32m1_t lo = __riscv_vmv_v_x_i32m1(-128, vl);
        vint32m1_t hi = __riscv_vmv_v_x_i32m1(127, vl);
        rounded = __riscv_vmax_vv_i32m1(rounded, lo, vl);
        rounded = __riscv_vmin_vv_i32m1(rounded, hi, vl);
        // Two-step narrow i32 -> i16 -> i8. NOT vnclip (needs the vxrm CSR, which Ara
        // traps on), and via the early-clobber helpers, not the bare intrinsics (Ara
        // miscomputes an in-place narrow). See quantize_f32i8 for the full note.
        vint16mf2_t n16 = __bingo_narrow_i32m1_i16mf2(rounded, vl);
        vint8mf4_t  n8  = __bingo_narrow_i16mf2_i8mf4(n16, vl);
        __riscv_vse8_v_i8mf4(o_ptr, n8, vl);
    }
#else
    // Scalar fallback (no fp16 vector support): convert each half to fp32 in SW.
    // Pass 1: find max(|x|)
    float abs_max = 0.0f;
    for (uint64_t i = 0; i < num_elements; i++) {
        float x = __bingo_fp16_to_fp32(input[i]);
        float ax = x < 0.0f ? -x : x;
        if (ax > abs_max) abs_max = ax;
    }
    // Compute scale
    float scale = abs_max / 127.0f;
    if (scale < 1e-10f) scale = 1e-10f;
    *scale_out = scale;
    float inv_scale = 1.0f / scale;
    // Pass 2: scale, round, clamp to int8
    for (uint64_t i = 0; i < num_elements; i++) {
        float scaled = __bingo_fp16_to_fp32(input[i]) * inv_scale;
        int32_t r = (int32_t)(scaled < 0.0f ? scaled - 0.5f : scaled + 0.5f);
        if (r < -128) r = -128;
        if (r > 127)  r = 127;
        output[i] = (int8_t)r;
    }
#endif

    BINGO_TRACE_MARKER(BINGO_TRACE_QUANT_RUN_END);
    sp->return_value = (uint32_t)(uintptr_t)output;
    sp->num_return_values = num_elements;
    return BINGO_RET_SUCC;
}

// ---- typed native binary impls: out[i] = vop(a[i], b[i]) ----
#define __BINGO_BINARY_IMPL(op, P, T, ESET, ELD, EST, VT, VOP)                 \
static inline void __bingo_##op##_##P(const T* a, const T* b, T* o, uint64_t n){\
    uint64_t avl = n; const T *ap=a,*bp=b; T *op_=o;                           \
    for (size_t vl = ESET(avl); avl>0; avl-=vl, ap+=vl, bp+=vl, op_+=vl){      \
        vl = ESET(avl);                                                        \
        VT va = ELD(ap, vl), vb = ELD(bp, vl);                                 \
        EST(op_, VOP(va, vb, vl), vl);                                         \
    }                                                                          \
}
#define __BINGO_BINARY_F16(op, VOP)                                            \
    __BINGO_BINARY_IMPL(op, f16, _Float16, __riscv_vsetvl_e16m1,              \
        __riscv_vle16_v_f16m1, __riscv_vse16_v_f16m1, vfloat16m1_t, VOP)
#define __BINGO_BINARY_I8(op, VOP)                                             \
    __BINGO_BINARY_IMPL(op, i8, int8_t, __riscv_vsetvl_e8m1,                 \
        __riscv_vle8_v_i8m1, __riscv_vse8_v_i8m1, vint8m1_t, VOP)
#define __BINGO_BINARY_I16(op, VOP)                                            \
    __BINGO_BINARY_IMPL(op, i16, int16_t, __riscv_vsetvl_e16m1,              \
        __riscv_vle16_v_i16m1, __riscv_vse16_v_i16m1, vint16m1_t, VOP)
#define __BINGO_BINARY_I32(op, VOP)                                            \
    __BINGO_BINARY_IMPL(op, i32, int32_t, __riscv_vsetvl_e32m1,              \
        __riscv_vle32_v_i32m1, __riscv_vse32_v_i32m1, vint32m1_t, VOP)

#if BINGO_HAVE_FP16_VEC
#define __BINGO_BINARY_CASE_FP16(op)                                           \
    case BINGO_PREC_FP16:                                                      \
        __bingo_##op##_f16((const _Float16*)A[0], (const _Float16*)A[1],      \
                            (_Float16*)A[2], n); break;
#else
#define __BINGO_BINARY_CASE_FP16(op) case BINGO_PREC_FP16: ret=BINGO_RET_FAIL; break;
#endif
#define __BINGO_BINARY_CASE_INT_BOTH(op)                                       \
    case BINGO_PREC_INT8:                                                      \
        __bingo_##op##_i8((const int8_t*)A[0], (const int8_t*)A[1],          \
                            (int8_t*)A[2], n); break;                          \
    case BINGO_PREC_INT16:                                                     \
        __bingo_##op##_i16((const int16_t*)A[0], (const int16_t*)A[1],       \
                             (int16_t*)A[2], n); break;
// int8 + int16 + int32 (for ops with a native int32 vector impl)
#define __BINGO_BINARY_CASE_INT_ALL(op)                                        \
    __BINGO_BINARY_CASE_INT_BOTH(op)                                           \
    case BINGO_PREC_INT32:                                                     \
        __bingo_##op##_i32((const int32_t*)A[0], (const int32_t*)A[1],       \
                             (int32_t*)A[2], n); break;
#define __BINGO_BINARY_CASE_INT_NONE(op)

// Binary dispatcher: arg = {a, b, out, n, precision}.
#define __BINGO_BINARY_DISPATCH(op, INTCASES)                                  \
static inline uint64_t __host_bingo_kernel_##op(void *arg){                    \
    uint64_t *A = (uint64_t*)arg; uint64_t prec = A[4];                        \
    if (prec == BINGO_PREC_FP32) return __host_bingo_kernel_##op##_f32(arg);    \
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);                            \
    uint64_t n = A[3]; uint64_t ret = BINGO_RET_SUCC;                          \
    switch (prec) {                                                            \
        __BINGO_BINARY_CASE_FP16(op)                                           \
        INTCASES(op)                                                           \
        default: ret = BINGO_RET_FAIL;                                         \
    }                                                                          \
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);                             \
    return ret;                                                                \
}

#if BINGO_HAVE_FP16_VEC
__BINGO_BINARY_F16(add, __riscv_vfadd_vv_f16m1)
__BINGO_BINARY_F16(sub, __riscv_vfsub_vv_f16m1)
__BINGO_BINARY_F16(mul, __riscv_vfmul_vv_f16m1)
__BINGO_BINARY_F16(div, __riscv_vfdiv_vv_f16m1)
__BINGO_BINARY_F16(max, __riscv_vfmax_vv_f16m1)
__BINGO_BINARY_F16(min, __riscv_vfmin_vv_f16m1)
#endif
__BINGO_BINARY_I8(add, __riscv_vadd_vv_i8m1)   __BINGO_BINARY_I16(add, __riscv_vadd_vv_i16m1)   __BINGO_BINARY_I32(add, __riscv_vadd_vv_i32m1)
__BINGO_BINARY_I8(sub, __riscv_vsub_vv_i8m1)   __BINGO_BINARY_I16(sub, __riscv_vsub_vv_i16m1)   __BINGO_BINARY_I32(sub, __riscv_vsub_vv_i32m1)
__BINGO_BINARY_I8(mul, __riscv_vmul_vv_i8m1)   __BINGO_BINARY_I16(mul, __riscv_vmul_vv_i16m1)   __BINGO_BINARY_I32(mul, __riscv_vmul_vv_i32m1)
__BINGO_BINARY_I8(max, __riscv_vmax_vv_i8m1)   __BINGO_BINARY_I16(max, __riscv_vmax_vv_i16m1)   __BINGO_BINARY_I32(max, __riscv_vmax_vv_i32m1)
__BINGO_BINARY_I8(min, __riscv_vmin_vv_i8m1)   __BINGO_BINARY_I16(min, __riscv_vmin_vv_i16m1)   __BINGO_BINARY_I32(min, __riscv_vmin_vv_i32m1)
// add takes int32 through the dispatcher like every other int-capable op. The
// separate __host_bingo_kernel_add_i32 (K-split partial-D accumulation) exists
// alongside it: it runs the SAME vadd but also publishes the result via the
// scratchpad at arg[5], which the plain {a,b,out,n,prec} callers (e.g. the ara
// sweep) do not pass.
__BINGO_BINARY_DISPATCH(add, __BINGO_BINARY_CASE_INT_ALL)
__BINGO_BINARY_DISPATCH(sub, __BINGO_BINARY_CASE_INT_ALL)
__BINGO_BINARY_DISPATCH(mul, __BINGO_BINARY_CASE_INT_ALL)
__BINGO_BINARY_DISPATCH(max, __BINGO_BINARY_CASE_INT_ALL)
__BINGO_BINARY_DISPATCH(min, __BINGO_BINARY_CASE_INT_ALL)
__BINGO_BINARY_DISPATCH(div, __BINGO_BINARY_CASE_INT_NONE)

// ---- typed native unary impls: out[i] = body(in[i]) ----
#define __BINGO_UNARY_IMPL(op, P, T, ESET, ELD, EST, VT, BODY)                 \
static inline void __bingo_##op##_##P(const T* in, T* o, uint64_t n){          \
    uint64_t avl = n; const T *ip=in; T *op_=o;                               \
    for (size_t vl = ESET(avl); avl>0; avl-=vl, ip+=vl, op_+=vl){              \
        vl = ESET(avl); VT v = ELD(ip, vl); VT result; BODY;                   \
        EST(op_, result, vl);                                                  \
    }                                                                          \
}
#define __BINGO_UNARY_F16_NATIVE(op, BODY)                                     \
    __BINGO_UNARY_IMPL(op, f16, _Float16, __riscv_vsetvl_e16m1,              \
        __riscv_vle16_v_f16m1, __riscv_vse16_v_f16m1, vfloat16m1_t, BODY)
#define __BINGO_UNARY_I8(op, BODY)                                             \
    __BINGO_UNARY_IMPL(op, i8, int8_t, __riscv_vsetvl_e8m1,                 \
        __riscv_vle8_v_i8m1, __riscv_vse8_v_i8m1, vint8m1_t, BODY)
#define __BINGO_UNARY_I16(op, BODY)                                            \
    __BINGO_UNARY_IMPL(op, i16, int16_t, __riscv_vsetvl_e16m1,              \
        __riscv_vle16_v_i16m1, __riscv_vse16_v_i16m1, vint16m1_t, BODY)
#define __BINGO_UNARY_I32(op, BODY)                                            \
    __BINGO_UNARY_IMPL(op, i32, int32_t, __riscv_vsetvl_e32m1,              \
        __riscv_vle32_v_i32m1, __riscv_vse32_v_i32m1, vint32m1_t, BODY)

// fp16 transcendental: widen f16->f32, reuse the fp32 math, narrow f32->f16.
// BODY32 reads vfloat32m2_t v -> sets result.
//
// The load is e16m1 and the widened value lands in f32m2, so one iteration covers
// VLEN/16 elements -- twice the fp32 kernel's VLEN/32. The LMUL matters: a fractional
// LMUL exactly cancels the narrower SEW (VLEN/16 x 1/2 == VLEN/32), which would retire
// no more elements per iteration than fp32 while still paying the widen+narrow pair.
#if BINGO_HAVE_FP16_VEC
#define __BINGO_UNARY_F16_VIA_F32(op, BODY32)                                  \
static inline void __bingo_##op##_f16(const _Float16* in, _Float16* o, uint64_t n){ \
    uint64_t avl = n; const _Float16 *ip=in; _Float16 *op_=o;                  \
    for (size_t vl = __riscv_vsetvl_e16m1(avl); avl>0; avl-=vl, ip+=vl, op_+=vl){ \
        vl = __riscv_vsetvl_e16m1(avl);                                        \
        vfloat32m2_t v = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ip,vl), vl); \
        vfloat32m2_t result; BODY32;                                          \
        __riscv_vse16_v_f16m1(op_, __bingo_narrow_f32m2_f16m1(result, vl), vl); \
    }                                                                          \
}
#else
#define __BINGO_UNARY_F16_VIA_F32(op, BODY32)
#endif

#if BINGO_HAVE_FP16_VEC
#define __BINGO_UNARY_CASE_FP16(op)                                            \
    case BINGO_PREC_FP16:                                                      \
        __bingo_##op##_f16((const _Float16*)A[0], (_Float16*)A[1], n); break;
#else
#define __BINGO_UNARY_CASE_FP16(op) case BINGO_PREC_FP16: ret=BINGO_RET_FAIL; break;
#endif
#define __BINGO_UNARY_CASE_INT_BOTH(op)                                        \
    case BINGO_PREC_INT8:                                                      \
        __bingo_##op##_i8((const int8_t*)A[0], (int8_t*)A[1], n); break;     \
    case BINGO_PREC_INT16:                                                     \
        __bingo_##op##_i16((const int16_t*)A[0], (int16_t*)A[1], n); break;
// int8 + int16 + int32 (for ops with a native int32 vector impl)
#define __BINGO_UNARY_CASE_INT_ALL(op)                                         \
    __BINGO_UNARY_CASE_INT_BOTH(op)                                            \
    case BINGO_PREC_INT32:                                                     \
        __bingo_##op##_i32((const int32_t*)A[0], (int32_t*)A[1], n); break;
#define __BINGO_UNARY_CASE_INT_NONE(op)

// Unary dispatcher: arg = {in, out, n, precision}.
#define __BINGO_UNARY_DISPATCH(op, INTCASES)                                   \
static inline uint64_t __host_bingo_kernel_##op(void *arg){                    \
    uint64_t *A = (uint64_t*)arg; uint64_t prec = A[3];                        \
    if (prec == BINGO_PREC_FP32) return __host_bingo_kernel_##op##_f32(arg);    \
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);                            \
    uint64_t n = A[2]; uint64_t ret = BINGO_RET_SUCC;                          \
    switch (prec) {                                                            \
        __BINGO_UNARY_CASE_FP16(op)                                            \
        INTCASES(op)                                                           \
        default: ret = BINGO_RET_FAIL;                                         \
    }                                                                          \
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);                             \
    return ret;                                                                \
}

// int-capable unary ops (native fp16 + int8/int16).
#if BINGO_HAVE_FP16_VEC
__BINGO_UNARY_F16_NATIVE(relu, { result = __riscv_vfmax_vv_f16m1(v, __riscv_vfmv_v_f_f16m1((_Float16)0.0f, vl), vl); })
__BINGO_UNARY_F16_NATIVE(neg,  { result = __riscv_vfneg_v_f16m1(v, vl); })
__BINGO_UNARY_F16_NATIVE(abs,  { result = __riscv_vfmax_vv_f16m1(v, __riscv_vfneg_v_f16m1(v, vl), vl); })
#endif
__BINGO_UNARY_I8 (relu, { result = __riscv_vmax_vx_i8m1(v, 0, vl); })
__BINGO_UNARY_I16(relu, { result = __riscv_vmax_vx_i16m1(v, 0, vl); })
__BINGO_UNARY_I32(relu, { result = __riscv_vmax_vx_i32m1(v, 0, vl); })
__BINGO_UNARY_I8 (neg,  { result = __riscv_vneg_v_i8m1(v, vl); })
__BINGO_UNARY_I16(neg,  { result = __riscv_vneg_v_i16m1(v, vl); })
__BINGO_UNARY_I32(neg,  { result = __riscv_vneg_v_i32m1(v, vl); })
__BINGO_UNARY_I8 (abs,  { result = __riscv_vmax_vv_i8m1(v, __riscv_vneg_v_i8m1(v, vl), vl); })
__BINGO_UNARY_I16(abs,  { result = __riscv_vmax_vv_i16m1(v, __riscv_vneg_v_i16m1(v, vl), vl); })
__BINGO_UNARY_I32(abs,  { result = __riscv_vmax_vv_i32m1(v, __riscv_vneg_v_i32m1(v, vl), vl); })
__BINGO_UNARY_DISPATCH(relu, __BINGO_UNARY_CASE_INT_ALL)
__BINGO_UNARY_DISPATCH(neg,  __BINGO_UNARY_CASE_INT_ALL)
__BINGO_UNARY_DISPATCH(abs,  __BINGO_UNARY_CASE_INT_ALL)

// float-only unary ops (fp16 via widen/narrow, reusing the fp32 math). The bodies
// compute at f32m2 -- see __BINGO_UNARY_F16_VIA_F32.
__BINGO_UNARY_F16_VIA_F32(exp, { result = __bingo_exp_f32_m2(v, vl); })
// sqrt and reciprocal have NATIVE fp16 instructions (zvfh), so they need no fp32
// detour: computing them in f16m1 doubles the elements per iteration and drops the
// widen+narrow pair. Accuracy is unaffected in any way that survives the fp16 store --
// the result is rounded to fp16 regardless, and vfsqrt/vfdiv are correctly rounded, so
// this is at most 1/2 ulp of fp16 from the widened path.
__BINGO_UNARY_F16_NATIVE(sqrt, { result = __riscv_vfsqrt_v_f16m1(v, vl); })
__BINGO_UNARY_F16_NATIVE(reciprocal, {
    result = __riscv_vfrdiv_vf_f16m1(v, (_Float16)1.0f, vl);
})
__BINGO_UNARY_F16_VIA_F32(sigmoid, {
    vfloat32m2_t one = __riscv_vfmv_v_f_f32m2(1.0f, vl);
    vfloat32m2_t en  = __bingo_exp_f32_m2(__riscv_vfneg_v_f32m2(v, vl), vl);
    result = __riscv_vfdiv_vv_f32m2(one, __riscv_vfadd_vv_f32m2(one, en, vl), vl);
})
__BINGO_UNARY_F16_VIA_F32(tanh, {
    vfloat32m2_t one = __riscv_vfmv_v_f_f32m2(1.0f, vl);
    vfloat32m2_t e2  = __bingo_exp_f32_m2(__riscv_vfmul_vf_f32m2(v, 2.0f, vl), vl);
    result = __riscv_vfdiv_vv_f32m2(__riscv_vfsub_vv_f32m2(e2, one, vl),
                                    __riscv_vfadd_vv_f32m2(e2, one, vl), vl);
})
__BINGO_UNARY_F16_VIA_F32(silu, {
    vfloat32m2_t one = __riscv_vfmv_v_f_f32m2(1.0f, vl);
    vfloat32m2_t en  = __bingo_exp_f32_m2(__riscv_vfneg_v_f32m2(v, vl), vl);
    vfloat32m2_t sig = __riscv_vfdiv_vv_f32m2(one, __riscv_vfadd_vv_f32m2(one, en, vl), vl);
    result = __riscv_vfmul_vv_f32m2(v, sig, vl);
})
__BINGO_UNARY_F16_VIA_F32(gelu, {
    vfloat32m2_t one    = __riscv_vfmv_v_f_f32m2(1.0f, vl);
    vfloat32m2_t scaled = __riscv_vfmul_vf_f32m2(v, 1.702f, vl);
    vfloat32m2_t en     = __bingo_exp_f32_m2(__riscv_vfneg_v_f32m2(scaled, vl), vl);
    vfloat32m2_t sig    = __riscv_vfdiv_vv_f32m2(one, __riscv_vfadd_vv_f32m2(one, en, vl), vl);
    result = __riscv_vfmul_vv_f32m2(v, sig, vl);
})
__BINGO_UNARY_DISPATCH(exp,        __BINGO_UNARY_CASE_INT_NONE)
__BINGO_UNARY_DISPATCH(sqrt,       __BINGO_UNARY_CASE_INT_NONE)
__BINGO_UNARY_DISPATCH(reciprocal, __BINGO_UNARY_CASE_INT_NONE)
__BINGO_UNARY_DISPATCH(sigmoid,    __BINGO_UNARY_CASE_INT_NONE)
__BINGO_UNARY_DISPATCH(tanh,       __BINGO_UNARY_CASE_INT_NONE)
__BINGO_UNARY_DISPATCH(silu,       __BINGO_UNARY_CASE_INT_NONE)
__BINGO_UNARY_DISPATCH(gelu,       __BINGO_UNARY_CASE_INT_NONE)

// ---- reduction impls (scalar return; reduce_* dispatchers below) ----
#if BINGO_HAVE_FP16_VEC
// ARA REDUCTION LIMITS -- why the fp16/int reduce_sum paths keep their fractional LMUL
// even though it costs them the width benefit of the narrow input:
//
//   1. WIDENING reductions are not implemented. vfwredosum.vs (f16 elements -> f32
//      accumulator) would be exactly right here -- it consumes f16m1 directly, so it
//      doubles the elements per iteration AND removes the explicit vfwcvt, at identical
//      accuracy. Ara traps it: Illegal Instruction, tval=0xce1110d7 (OPFVV funct6=110011).
//   2. A reduction with an LMUL>1 SOURCE never retires. Ara accepts vfredosum/vfredmax
//      with an f32m2 operand and then silently hangs -- no trap, just a stalled vector
//      unit, which is worse to debug than an illegal instruction.
//
// So a reduction must consume LMUL=1, which forces the widened value into f32m1, which
// forces the fp16 load down to f16mf2 -- VLEN/32 elements, the same as fp32.
// (__bingo_reduce_max_f16 below needs no widening at all, so it does run at f16m1.)
static inline float __bingo_reduce_sum_f16(const _Float16* in, uint64_t n){
    float acc = 0.0f; uint64_t avl = n; const _Float16 *p = in;
    for (size_t vl = __riscv_vsetvl_e16mf2(avl); avl>0; avl-=vl, p+=vl){
        vl = __riscv_vsetvl_e16mf2(avl);
        vfloat32m1_t v = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(p, vl), vl);
        vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        acc += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredosum_vs_f32m1_f32m1(v, z, vl));
    }
    return acc;
}
// max needs no wider accumulator: the max of a set of fp16 values IS an fp16 value, so
// reducing natively in fp16 is exact -- no widening, no accuracy change.
static inline float __bingo_reduce_max_f16(const _Float16* in, uint64_t n){
    _Float16 mx = in[0]; uint64_t avl = n; const _Float16 *p = in;
    for (size_t vl = __riscv_vsetvl_e16m1(avl); avl>0; avl-=vl, p+=vl){
        vl = __riscv_vsetvl_e16m1(avl);
        vfloat16m1_t v = __riscv_vle16_v_f16m1(p, vl);
        vfloat16m1_t i = __riscv_vfmv_v_f_f16m1(mx, vl);
        mx = __riscv_vfmv_f_s_f16m1_f16(__riscv_vfredmax_vs_f16m1_f16m1(v, i, vl));
    }
    return (float)mx;
}
#endif
// int reduce_sum: widen i8/i16 to i32 per chunk (overflow-safe at any VLEN).
//
// Same Ara constraint as __bingo_reduce_sum_f16: the reduction source must be LMUL=1, so
// the i32 accumulator has to land in i32m1, which drags the i8 load down to i8mf4 --
// VLEN/32 elements, no better than int32 itself. That is why reduce_sum is the one op
// where the NARROWEST type is the SLOWEST (int8 77900 cc vs int32 58458 cc): int8 pays
// two widening converts and retires no extra elements for them. Loading i8m1 and widening
// up to i32m4 (4x the elements) is not available: Ara silently hangs on vredsum.vs with
// an m4 source.
static inline int32_t __bingo_reduce_sum_i8(const int8_t* in, uint64_t n){
    int32_t acc = 0; uint64_t avl = n; const int8_t *p = in;
    for (size_t vl = __riscv_vsetvl_e8mf4(avl); avl>0; avl-=vl, p+=vl){
        vl = __riscv_vsetvl_e8mf4(avl);
        vint16mf2_t v16 = __riscv_vwcvt_x_x_v_i16mf2(__riscv_vle8_v_i8mf4(p, vl), vl);
        vint32m1_t  v32 = __riscv_vwcvt_x_x_v_i32m1(v16, vl);
        vint32m1_t  z   = __riscv_vmv_v_x_i32m1(0, vl);
        acc += __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m1_i32m1(v32, z, vl));
    }
    return acc;
}
static inline int32_t __bingo_reduce_sum_i16(const int16_t* in, uint64_t n){
    int32_t acc = 0; uint64_t avl = n; const int16_t *p = in;
    for (size_t vl = __riscv_vsetvl_e16mf2(avl); avl>0; avl-=vl, p+=vl){
        vl = __riscv_vsetvl_e16mf2(avl);
        vint32m1_t v32 = __riscv_vwcvt_x_x_v_i32m1(__riscv_vle16_v_i16mf2(p, vl), vl);
        vint32m1_t z   = __riscv_vmv_v_x_i32m1(0, vl);
        acc += __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m1_i32m1(v32, z, vl));
    }
    return acc;
}
static inline int32_t __bingo_reduce_max_i8(const int8_t* in, uint64_t n){
    int8_t mv = in[0]; uint64_t avl = n; const int8_t *p = in;
    for (size_t vl = __riscv_vsetvl_e8m1(avl); avl>0; avl-=vl, p+=vl){
        vl = __riscv_vsetvl_e8m1(avl);
        vint8m1_t v = __riscv_vle8_v_i8m1(p, vl);
        vint8m1_t i = __riscv_vmv_v_x_i8m1(mv, vl);
        mv = __riscv_vmv_x_s_i8m1_i8(__riscv_vredmax_vs_i8m1_i8m1(v, i, vl));
    }
    return (int32_t)mv;
}
static inline int32_t __bingo_reduce_max_i16(const int16_t* in, uint64_t n){
    int16_t mv = in[0]; uint64_t avl = n; const int16_t *p = in;
    for (size_t vl = __riscv_vsetvl_e16m1(avl); avl>0; avl-=vl, p+=vl){
        vl = __riscv_vsetvl_e16m1(avl);
        vint16m1_t v = __riscv_vle16_v_i16m1(p, vl);
        vint16m1_t i = __riscv_vmv_v_x_i16m1(mv, vl);
        mv = __riscv_vmv_x_s_i16m1_i16(__riscv_vredmax_vs_i16m1_i16m1(v, i, vl));
    }
    return (int32_t)mv;
}
// int32 reduce: int32 in -> int32 scalar out (same out type as the i8/i16 reduce).
// The sum accumulates in int32 and can overflow (same caveat as a chained int32 add).
static inline int32_t __bingo_reduce_sum_i32(const int32_t* in, uint64_t n){
    int32_t acc = 0; uint64_t avl = n; const int32_t *p = in;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl>0; avl-=vl, p+=vl){
        vl = __riscv_vsetvl_e32m1(avl);
        vint32m1_t v = __riscv_vle32_v_i32m1(p, vl);
        vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
        acc += __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m1_i32m1(v, z, vl));
    }
    return acc;
}
static inline int32_t __bingo_reduce_max_i32(const int32_t* in, uint64_t n){
    int32_t mv = in[0]; uint64_t avl = n; const int32_t *p = in;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl>0; avl-=vl, p+=vl){
        vl = __riscv_vsetvl_e32m1(avl);
        vint32m1_t v = __riscv_vle32_v_i32m1(p, vl);
        vint32m1_t i = __riscv_vmv_v_x_i32m1(mv, vl);
        mv = __riscv_vmv_x_s_i32m1_i32(__riscv_vredmax_vs_i32m1_i32m1(v, i, vl));
    }
    return mv;
}

// reduce dispatchers: arg = {in, out, n, precision}.
// FP32 -> float out (legacy). FP16 -> float out. INT8/INT16/INT32 -> int32 out.
static inline uint64_t __host_bingo_kernel_reduce_sum(void *arg){
    uint64_t *A = (uint64_t*)arg; uint64_t prec = A[3];
    if (prec == BINGO_PREC_FP32) return __host_bingo_kernel_reduce_sum_f32(arg);
    uint64_t n = A[2]; BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint64_t ret = BINGO_RET_SUCC;
    switch (prec) {
#if BINGO_HAVE_FP16_VEC
        case BINGO_PREC_FP16: *(float*)A[1]   = __bingo_reduce_sum_f16((const _Float16*)A[0], n); break;
#endif
        case BINGO_PREC_INT8:  *(int32_t*)A[1] = __bingo_reduce_sum_i8((const int8_t*)A[0], n); break;
        case BINGO_PREC_INT16: *(int32_t*)A[1] = __bingo_reduce_sum_i16((const int16_t*)A[0], n); break;
        case BINGO_PREC_INT32: *(int32_t*)A[1] = __bingo_reduce_sum_i32((const int32_t*)A[0], n); break;
        default: ret = BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    return ret;
}
static inline uint64_t __host_bingo_kernel_reduce_max(void *arg){
    uint64_t *A = (uint64_t*)arg; uint64_t prec = A[3];
    if (prec == BINGO_PREC_FP32) return __host_bingo_kernel_reduce_max_f32(arg);
    uint64_t n = A[2]; BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint64_t ret = BINGO_RET_SUCC;
    switch (prec) {
#if BINGO_HAVE_FP16_VEC
        case BINGO_PREC_FP16: *(float*)A[1]   = __bingo_reduce_max_f16((const _Float16*)A[0], n); break;
#endif
        case BINGO_PREC_INT8:  *(int32_t*)A[1] = __bingo_reduce_max_i8((const int8_t*)A[0], n); break;
        case BINGO_PREC_INT16: *(int32_t*)A[1] = __bingo_reduce_max_i16((const int16_t*)A[0], n); break;
        case BINGO_PREC_INT32: *(int32_t*)A[1] = __bingo_reduce_max_i32((const int32_t*)A[0], n); break;
        default: ret = BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    return ret;
}
static inline uint64_t __host_bingo_kernel_reduce_mean(void *arg){
    uint64_t *A = (uint64_t*)arg; uint64_t prec = A[3];
    if (prec == BINGO_PREC_FP32) return __host_bingo_kernel_reduce_mean_f32(arg);
    uint64_t n = A[2]; BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint64_t ret = BINGO_RET_SUCC;
#if BINGO_HAVE_FP16_VEC
    if (prec == BINGO_PREC_FP16) *(float*)A[1] = __bingo_reduce_sum_f16((const _Float16*)A[0], n) / (float)n;
    else ret = BINGO_RET_FAIL;
#else
    ret = BINGO_RET_FAIL;
#endif
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    return ret;
}

// ---- compound fp16 ops (widen f16->f32, compute, narrow f32->f16) ----
#if BINGO_HAVE_FP16_VEC
static inline void __bingo_silu_mul_f16(const _Float16* g, const _Float16* u,
                                         _Float16* o, uint64_t n){
    // Stays at f32m1. Unlike the single-input fp16 transcendentals (which run at f32m2
    // for a clean ~1.8x), silu_mul carries TWO widened m2 inputs live across the exp
    // polynomial's own m2 temporaries. At LMUL=2 that exceeds the vector register file
    // and the allocator SPILLS -- and a vector spill makes the compiler emit `csrr vlenb`
    // to size the spill slot. Ara implements no vlenb CSR, so the spill traps (Illegal
    // Instruction, tval=0xc22022f3) in the prologue, before main() prints anything.
    uint64_t avl = n; const _Float16 *gp=g,*up=u; _Float16 *op_=o;
    for (size_t vl = __riscv_vsetvl_e16mf2(avl); avl>0; avl-=vl, gp+=vl, up+=vl, op_+=vl){
        vl = __riscv_vsetvl_e16mf2(avl);
        vfloat32m1_t gv = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(gp, vl), vl);
        vfloat32m1_t uv = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(up, vl), vl);
        vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
        vfloat32m1_t en  = __bingo_exp_f32_m1(__riscv_vfneg_v_f32m1(gv, vl), vl);
        vfloat32m1_t sig = __riscv_vfdiv_vv_f32m1(one, __riscv_vfadd_vv_f32m1(one, en, vl), vl);
        vfloat32m1_t r   = __riscv_vfmul_vv_f32m1(__riscv_vfmul_vv_f32m1(gv, sig, vl), uv, vl);
        __riscv_vse16_v_f16mf2(op_, __bingo_narrow_f32m1_f16mf2(r, vl), vl);
    }
}
// softmax / rmsnorm fp16 stay at f32m1: every pass here is built around a REDUCTION
// (row max, sum(exp), sum(x^2)), and Ara silently hangs on a reduction whose source is
// LMUL>1 -- see __bingo_reduce_sum_f16 for the full note. The reduction pins the widened
// value to f32m1, which pins the fp16 load to f16mf2.
static inline void __bingo_softmax_row_f16(const _Float16* in, _Float16* out, uint64_t len){
    float maxv = (float)in[0];
    { uint64_t rem = len; const _Float16 *p = in;
      for (size_t vl = __riscv_vsetvl_e16mf2(rem); rem>0; rem-=vl, p+=vl){
          vl = __riscv_vsetvl_e16mf2(rem);
          vfloat32m1_t v = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(p, vl), vl);
          vfloat32m1_t i = __riscv_vfmv_v_f_f32m1(maxv, vl);
          maxv = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(v, i, vl)); } }
    float sum = 0.0f;
    { uint64_t rem = len; const _Float16 *p = in; _Float16 *op_ = out;
      for (size_t vl = __riscv_vsetvl_e16mf2(rem); rem>0; rem-=vl, p+=vl, op_+=vl){
          vl = __riscv_vsetvl_e16mf2(rem);
          vfloat32m1_t v  = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(p, vl), vl);
          vfloat32m1_t ev = __bingo_exp_f32_m1(__riscv_vfsub_vf_f32m1(v, maxv, vl), vl);
          __riscv_vse16_v_f16mf2(op_, __bingo_narrow_f32m1_f16mf2(ev, vl), vl);
          vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, vl);
          sum += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredosum_vs_f32m1_f32m1(ev, z, vl)); } }
    float inv = 1.0f / sum;
    { uint64_t rem = len; _Float16 *op_ = out;
      for (size_t vl = __riscv_vsetvl_e16mf2(rem); rem>0; rem-=vl, op_+=vl){
          vl = __riscv_vsetvl_e16mf2(rem);
          vfloat32m1_t v = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(op_, vl), vl);
          vfloat32m1_t r = __riscv_vfmul_vf_f32m1(v, inv, vl);
          __riscv_vse16_v_f16mf2(op_, __bingo_narrow_f32m1_f16mf2(r, vl), vl); } }
}
static inline void __bingo_rmsnorm_row_f16(const _Float16* in, const _Float16* w,
                                            _Float16* out, uint64_t hidden){
    float ss = 0.0f;
    { uint64_t rem = hidden; const _Float16 *p = in;
      for (size_t vl = __riscv_vsetvl_e16mf2(rem); rem>0; rem-=vl, p+=vl){
          vl = __riscv_vsetvl_e16mf2(rem);
          vfloat32m1_t v  = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(p, vl), vl);
          vfloat32m1_t sq = __riscv_vfmul_vv_f32m1(v, v, vl);
          vfloat32m1_t z  = __riscv_vfmv_v_f_f32m1(0.0f, vl);
          ss += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredosum_vs_f32m1_f32m1(sq, z, vl)); } }
    float rms = 1.0f / __builtin_sqrtf(ss / (float)hidden + 1e-6f);
    { uint64_t rem = hidden; const _Float16 *ip=in,*wp=w; _Float16 *op_=out;
      for (size_t vl = __riscv_vsetvl_e16mf2(rem); rem>0; rem-=vl, ip+=vl, wp+=vl, op_+=vl){
          vl = __riscv_vsetvl_e16mf2(rem);
          vfloat32m1_t v  = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(ip, vl), vl);
          vfloat32m1_t wv = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(wp, vl), vl);
          vfloat32m1_t r  = __riscv_vfmul_vv_f32m1(__riscv_vfmul_vf_f32m1(v, rms, vl), wv, vl);
          __riscv_vse16_v_f16mf2(op_, __bingo_narrow_f32m1_f16mf2(r, vl), vl); } }
}
#endif // BINGO_HAVE_FP16_VEC

// compound dispatchers.
// silu_mul: arg = {gate, up, out, n, precision}.
static inline uint64_t __host_bingo_kernel_silu_mul(void *arg){
    uint64_t *A = (uint64_t*)arg; uint64_t prec = A[4];
    if (prec == BINGO_PREC_FP32) return __host_bingo_kernel_silu_mul_f32(arg);
#if BINGO_HAVE_FP16_VEC
    if (prec == BINGO_PREC_FP16) {
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
        __bingo_silu_mul_f16((const _Float16*)A[0], (const _Float16*)A[1], (_Float16*)A[2], A[3]);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
        return BINGO_RET_SUCC;
    }
#endif
    return BINGO_RET_FAIL;
}
// softmax: arg = {in, out, num_rows, row_length, precision, scratchpad}.
static inline uint64_t __host_bingo_kernel_softmax(void *arg){
    uint64_t *A = (uint64_t*)arg; uint64_t prec = A[4];
    if (prec == BINGO_PREC_FP32) {
        uint64_t a2[5] = { A[0], A[1], A[2], A[3], A[5] };  // drop precision, keep scratchpad last
        return __host_bingo_kernel_softmax_f32(a2);
    }
#if BINGO_HAVE_FP16_VEC
    if (prec == BINGO_PREC_FP16) {
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
        uint64_t rows = A[2], len = A[3];
        for (uint64_t r = 0; r < rows; r++)
            __bingo_softmax_row_f16((const _Float16*)A[0] + r*len, (_Float16*)A[1] + r*len, len);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
        return BINGO_RET_SUCC;
    }
#endif
    return BINGO_RET_FAIL;
}
// rmsnorm: arg = {in, weight, out, hidden_dim, num_tokens, precision}.
static inline uint64_t __host_bingo_kernel_rmsnorm(void *arg){
    uint64_t *A = (uint64_t*)arg; uint64_t prec = A[5];
    if (prec == BINGO_PREC_FP32) return __host_bingo_kernel_rmsnorm_f32(arg);
#if BINGO_HAVE_FP16_VEC
    if (prec == BINGO_PREC_FP16) {
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
        uint64_t hidden = A[3], tokens = A[4];
        for (uint64_t t = 0; t < tokens; t++)
            __bingo_rmsnorm_row_f16((const _Float16*)A[0] + t*hidden,
                                     (const _Float16*)A[1], (_Float16*)A[2] + t*hidden, hidden);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
        return BINGO_RET_SUCC;
    }
#endif
    return BINGO_RET_FAIL;
}
