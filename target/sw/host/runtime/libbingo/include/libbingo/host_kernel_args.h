// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include <stdint.h>
#define __HOST_BINGO_KERNEL_ARGS_DEFINE typedef struct __attribute__((packed, aligned(8)))

// NOTE: Every host kernel args struct has `scratchpad_ptr` as its LAST field.
// The kernel reads it to access its own bingo_kernel_scratchpad_t.

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_dummy_args {
    uint64_t dummy_input;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_dummy_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_exit_args {
    uint64_t exit_code;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_exit_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_entry_args {
    uint64_t start_cc_reg_addr;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_entry_args_t;


__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_check_result_args {
    uint64_t golden_data_addr;
    uint64_t output_data_addr;
    uint64_t data_size;        // in Bytes
    uint64_t name_addr;        // const char* label for printf (0 = no label)
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_check_result_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_idma_args {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint64_t size;        // in Bytes
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_idma_args_t;

// FP32 -> INT8 per-tensor symmetric quantize
// scale = max(|x|) / 127; q[i] = clamp(round(x[i] / scale), -128, 127)
__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_fp32_quantize_args {
    uint64_t input_addr;       // float* FP32 input array
    uint64_t output_addr;      // int8_t* INT8 output array
    uint64_t scale_out_addr;   // float* write computed scale here
    uint64_t num_elements;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_fp32_quantize_args_t;

// INT32 -> FP32 dequantize (after VersaCore GEMM accumulator)
// y[i] = int32_input[i] * combined_scale
__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_int32_dequantize_args {
    uint64_t input_addr;       // int32_t* INT32 input (GEMM accumulator)
    uint64_t output_addr;      // float* FP32 output
    uint64_t scale_addr;       // float* read combined_scale = scale_a * scale_b
    uint64_t num_elements;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_int32_dequantize_args_t;

// INT32 elementwise add: out[i] = a[i] + b[i]
// Used for inter-cluster partial-D accumulation in K-split GEMM schemes.
__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_int32_add_args {
    uint64_t input_a_addr;     // int32_t* operand A
    uint64_t input_b_addr;     // int32_t* operand B
    uint64_t output_addr;      // int32_t* output (may alias A or B for in-place)
    uint64_t num_elements;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_int32_add_args_t;

// FP32 softmax along last dimension (Ara RVV).
// out[r,c] = exp(in[r,c] - max(in[r,:])) / sum(exp(in[r,:] - max(in[r,:])))
__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_fp32_softmax_args {
    uint64_t input_addr;       // float* FP32 input
    uint64_t output_addr;      // float* FP32 output
    uint64_t num_rows;         // outer dim (batch of softmaxes)
    uint64_t row_length;       // inner dim (length of each softmax row)
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_fp32_softmax_args_t;

// DARTS Tier 1: Unified CERF Gating kernel
// Supports multiple activation modes via the `mode` field.
// Mode 0 (top_k):    Read logits from predecessor scratchpad, select top-k experts
// Mode 1 (threshold): Read confidence from predecessor scratchpad, activate if below threshold
// Mode 2 (static):   Use compile-time cerf_write_mask directly (no predecessor read)
//
// Two-level gating for >32 experts (CERF group sharing):
//   Level 1 (HW): CERF groups (max 32) — inactive groups skip at zero cost
//   Level 2 (SW): Per-expert activation array — experts in active groups check
//                 their slot in cond_activation_addr before computing.
//                 Written by this gating kernel, read by expert kernels.
#define BINGO_GATING_MODE_TOP_K      0
#define BINGO_GATING_MODE_THRESHOLD  1
#define BINGO_GATING_MODE_STATIC     2
__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_cerf_gating_args {
    uint64_t mode;                    // BINGO_GATING_MODE_*
    uint64_t pred_scratchpad_addr;    // predecessor's scratchpad (unused for static)
    uint64_t cerf_controlled_mask;    // 32-bit: groups this gating node owns
    uint64_t top_k_or_threshold;      // top_k: k; threshold: FP32 bits; static: cerf_write_mask
    uint64_t cerf_group_ids_addr;    // top_k: &cerf_group_ids[]; others: unused (0)
    uint64_t cond_activation_addr;  // Per-expert activation array (uint8_t[num_experts])
                                      // Gating kernel writes 1 for selected experts, 0 for others.
                                      // Expert kernels read their slot before computing.
                                      // 0 = no activation array (all experts in active groups run).
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_cerf_gating_args_t;
