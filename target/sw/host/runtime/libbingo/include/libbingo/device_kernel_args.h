// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include <stdint.h>

#define __SNAX_KERNEL_ARGS_DEFINE typedef struct __attribute__((packed, aligned(4)))



// Define the argument structures for the device kernels
// Each structure is packed and aligned to 4 bytes
// The definition should match the kernel function argument parsing in snax_kernel_lib.h


///////////////////////// Cluster-level Kernels ////////////////////////
// Dummy kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_dummy_args {
  uint32_t dummy_input;    
} __snax_kernel_dummy_args_t;

// CSR kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_csr_args {
  uint32_t csr_addr;            
  uint32_t csr_value;            
} __snax_kernel_csr_args_t;

// Check Results kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_check_results_args {
  uint32_t golden_data_addr;            
  uint32_t output_data_addr;            
  uint32_t data_size;        // in Bytes
} __snax_kernel_check_results_args_t;

// Check Results Full kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_check_results_full_args {
  uint32_t golden_data_addr_hi;            
  uint32_t golden_data_addr_lo;            
  uint32_t output_data_addr_hi;            
  uint32_t output_data_addr_lo;            
  uint32_t data_size;        // in Bytes
} __snax_kernel_check_results_full_args_t;

// Load-Compute-Store kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_load_compute_store_args {
  uint32_t input_data_addr;            
  uint32_t input_data_size;        // in Bytes
  uint32_t output_data_addr;            
  uint32_t output_data_size;       // in Bytes
} __snax_kernel_load_compute_store_args_t;

// Double Buffer kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_double_buffer_args {
  uint32_t input_data_addr;            
  uint32_t output_data_addr;            
  uint32_t data_size;       // in Bytes
  uint32_t num_tiles;      // Number of tiles >=3
} __snax_kernel_double_buffer_args_t;

// XDMA 1D Copy kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_xdma_1d_copy_args {
  uint32_t src_addr_hi;    
  uint32_t src_addr_lo;            
  uint32_t dst_addr_hi;            
  uint32_t dst_addr_lo;            
  uint32_t size;        // in Bytes
} __snax_kernel_xdma_1d_copy_args_t;

// IDMA 1D Copy kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_idma_1d_copy_args_t {
  uint32_t src_addr_hi;    
  uint32_t src_addr_lo;            
  uint32_t dst_addr_hi;            
  uint32_t dst_addr_lo;            
  uint32_t size;        // in Bytes
} __snax_kernel_idma_1d_copy_args_t;

// ---------------------------------------------------------
// ---------------------VERSACORE---------------------------
// ---------------------------------------------------------

__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_versacore_load_compute_store_args {
  // Compute D = A*B + C using versacore
  // D will at the same address space as C
  // A: int8
  // B: int8
  // C: int32
  // D: int32
  // Inputs
  // 0
  uint32_t input_A_addr_hi;
  // 1
  uint32_t input_A_addr_lo;
  // 2
  uint32_t input_A_size;        // in Bytes
  // 3
  uint32_t input_B_addr_hi;
  // 4
  uint32_t input_B_addr_lo;
  // 5
  uint32_t input_B_size;        // in Bytes
  // 6
  uint32_t input_C_addr_hi;
  // 7
  uint32_t input_C_addr_lo;
  // 8
  uint32_t input_C_size;        // in Bytes
  // Outputs
  // 9
  uint32_t output_addr_hi;
  // 10
  uint32_t output_addr_lo;

  // Streamer Arguments
  // 11
  uint32_t streamer_cfg_addr_hi;
  // 12
  uint32_t streamer_cfg_addr_lo;
  // 13
  uint32_t streamer_cfg_size;        // in Bytes

  // Versacore Arguments
  // 14
  uint32_t versacore_cfg_addr_hi;
  // 15
  uint32_t versacore_cfg_addr_lo;
  // 16
  uint32_t versacore_cfg_size;        // in Bytes
  // Total arg length
  // 17
  uint32_t total_arg_length;        // in Bytes

} __snax_kernel_versacore_load_compute_store_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_gemm_intra_chiplet_args{
  uint32_t input_A_addr_hi;
  uint32_t input_A_addr_lo;
  uint32_t input_B_addr_hi;
  uint32_t input_B_addr_lo;
  uint32_t input_C_addr_hi;
  uint32_t input_C_addr_lo;
  uint32_t output_D_addr_hi;
  uint32_t output_D_addr_lo;
  uint32_t M;
  uint32_t K;
  uint32_t N;
  uint32_t array_shape;
  uint32_t transpose_A;
  uint32_t transpose_B;
  uint32_t accumPrevC;
} __snax_kernel_gemm_intra_chiplet_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_minimal_cfg_start_gemm_and_wait_args{
  uint32_t input_A_addr_lo;
  uint32_t input_B_addr_lo;
  uint32_t input_C_addr_lo;
  uint32_t output_D_addr_lo;
} __snax_kernel_minimal_cfg_start_gemm_and_wait_args_t;

//////////////////////// BINGO Core-level Kernels ////////////////////////
// BINGO Dummy kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_dummy_args {
  uint32_t dummy_input;            
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_dummy_args_t;
// BINGO Entry kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_entry_args {
  uint32_t start_cc_reg_addr;            
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_entry_args_t;
// BINGO Exit kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_exit_args {
  uint32_t exit_code;            
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_exit_args_t;

// BINGO IDMA 1D Copy kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_idma_1d_copy_args {
  uint32_t src_addr_hi;    
  uint32_t src_addr_lo;            
  uint32_t dst_addr_hi;            
  uint32_t dst_addr_lo;            
  uint32_t size;        // in Bytes
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_idma_1d_copy_args_t;

// BINGO IDMA Broadcast Kernel Args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_idma_broadcast_args {
  uint32_t src_addr_hi;    
  uint32_t src_addr_lo;            
  uint32_t dst_addr_hi;            
  uint32_t dst_addr_lo;            
  uint32_t size;        // in Bytes
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_idma_broadcast_args_t;

// BINGO GEMM Full kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_gemm_full_args {
  uint32_t input_A_addr;            
  uint32_t input_B_addr;            
  uint32_t input_C_addr;            
  uint32_t output_D_addr;            
  uint32_t M;            
  uint32_t K;            
  uint32_t N;            
  uint32_t array_shape_idx;            
  uint32_t transpose_A;            
  uint32_t transpose_B;            
  uint32_t accumPrevC;            
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_gemm_full_args_t;

// BINGO XDMA 1D Copy kernel args (same layout as cluster-level)
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_1d_copy_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t size;        // in Bytes
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_xdma_1d_copy_args_t;

// BINGO XDMA 6D kernel args (fixed-size, max 5 temporal dims = 6 total dims)
// Exposes full AGU strides/bounds to the user. Unused dims: stride=0, bound=1.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_6d_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t spatial_stride_src;
  uint32_t spatial_stride_dst;
  uint32_t num_temporal_dims;        // 1..5
  uint32_t temporal_strides_src[5];  // unused dims = 0
  uint32_t temporal_bounds_src[5];   // unused dims = 1
  uint32_t temporal_strides_dst[5];  // unused dims = 0
  uint32_t temporal_bounds_dst[5];   // unused dims = 1
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_xdma_6d_args_t;

// BINGO XDMA Transpose 2D (high-level: user provides shape, kernel computes strides)
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_transpose_2d_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t M;              // source rows
  uint32_t N;              // source cols
  uint32_t elem_bytes;     // element size (1=int8, 2=int16, 4=int32)
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_xdma_transpose_2d_args_t;

// BINGO XDMA Submatrix 2D (high-level: user provides shape + slice range)
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_submatrix_2d_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t src_rows;       // source matrix rows
  uint32_t src_cols;       // source matrix cols
  uint32_t row_start;      // slice start row (inclusive)
  uint32_t row_end;        // slice end row (exclusive)
  uint32_t col_start;      // slice start col (inclusive)
  uint32_t col_end;        // slice end col (exclusive)
  uint32_t elem_bytes;
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_xdma_submatrix_2d_args_t;

// BINGO XDMA Expand 2D (high-level: broadcast [1, N] -> [M, N])
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_expand_2d_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t M;              // number of output rows (broadcast factor)
  uint32_t N;              // row width (shared by src and dst)
  uint32_t elem_bytes;
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_xdma_expand_2d_args_t;

// BINGO XDMA Concat 2D (high-level: copy one chunk to offset in larger output)
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_concat_2d_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t src_rows;       // rows of THIS input chunk
  uint32_t src_cols;       // cols of THIS input chunk
  uint32_t dst_rows;       // rows of FULL output tensor
  uint32_t dst_cols;       // cols of FULL output tensor
  uint32_t axis;           // 0 = row-concat, 1 = col-concat
  uint32_t offset;         // element offset along concat axis
  uint32_t elem_bytes;
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_xdma_concat_2d_args_t;

// BINGO XDMA Pad 2D (high-level: zero-fill + strided copy into padded output)
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_pad_2d_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t src_rows;
  uint32_t src_cols;
  uint32_t pad_top;        // padding rows before
  uint32_t pad_bottom;     // padding rows after
  uint32_t pad_left;       // padding cols before
  uint32_t pad_right;      // padding cols after
  uint32_t elem_bytes;
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_xdma_pad_2d_args_t;

// BINGO XDMA Gather 2D (high-level: select rows by arithmetic stride)
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_gather_2d_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t src_rows;       // total rows in source tensor
  uint32_t src_cols;       // cols per row
  uint32_t num_indices;    // number of rows to gather
  uint32_t index_start;    // first row index to gather
  uint32_t index_stride;   // stride between indices (1 = contiguous)
  uint32_t elem_bytes;
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_xdma_gather_2d_args_t;

// ──────────────────────────────────────────────────────────────────────
// VersaCore blocked-layout conversion kernels
//
// All six kernels convert between row-major (logical 2D) and the three
// VersaCore blocked layouts {A, B, D}. They share a uniform arg layout
// that is parameterized by the scheduler's tile dimensions (M_T, K_T,
// N_T) and the array-shape (meshRow, tileSize, meshCol) so the same
// kernels work for any DSE-chosen tiling.
//
// Layout definitions:
//   A-layout [M_T, K_T, meshRow, tileSize]:
//     A_stored[m, k, r, s] = R_logical[m*meshRow + r, k*tileSize + s]
//   B-layout [N_T, K_T, meshCol, tileSize]:
//     B_stored[n, k, c, s] = R_logical[k*tileSize + s, n*meshCol + c]
//   D-layout [M_T, N_T, meshRow, meshCol]:
//     D_stored[m, n, r, c] = R_logical[m*meshRow + r, n*meshCol + c]
//
// The Python reference implementation lives at
// HeMAiA/util/sim/layout_convert.py — kernels must produce byte-identical
// output to the reference functions.
// ──────────────────────────────────────────────────────────────────────

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_d_to_row_major_args {
  uint32_t src_addr_hi;   // D-layout source
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;   // row-major destination
  uint32_t dst_addr_lo;
  uint32_t M_T;           // VersaCore M-tile count
  uint32_t N_T;           // VersaCore N-tile count
  uint32_t meshRow;
  uint32_t meshCol;
  uint32_t elem_bytes;    // 1 for INT8, 4 for INT32/FP32
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_xdma_d_to_row_major_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_row_major_to_a_args {
  uint32_t src_addr_hi;   // row-major source
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;   // A-layout destination
  uint32_t dst_addr_lo;
  uint32_t M_T;
  uint32_t K_T;
  uint32_t meshRow;
  uint32_t tileSize;
  uint32_t elem_bytes;
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_xdma_row_major_to_a_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_row_major_to_b_args {
  uint32_t src_addr_hi;   // row-major source
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;   // B-layout destination
  uint32_t dst_addr_lo;
  uint32_t K_T;
  uint32_t N_T;
  uint32_t tileSize;
  uint32_t meshCol;
  uint32_t elem_bytes;
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_xdma_row_major_to_b_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_a_to_row_major_args {
  uint32_t src_addr_hi;   // A-layout source
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;   // row-major destination
  uint32_t dst_addr_lo;
  uint32_t M_T;
  uint32_t K_T;
  uint32_t meshRow;
  uint32_t tileSize;
  uint32_t elem_bytes;
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_xdma_a_to_row_major_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_b_to_row_major_args {
  uint32_t src_addr_hi;   // B-layout source
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;   // row-major destination
  uint32_t dst_addr_lo;
  uint32_t K_T;
  uint32_t N_T;
  uint32_t tileSize;
  uint32_t meshCol;
  uint32_t elem_bytes;
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_xdma_b_to_row_major_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_row_major_to_d_args {
  uint32_t src_addr_hi;   // row-major source
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;   // D-layout destination
  uint32_t dst_addr_lo;
  uint32_t M_T;
  uint32_t N_T;
  uint32_t meshRow;
  uint32_t meshCol;
  uint32_t elem_bytes;
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_xdma_row_major_to_d_args_t;

// BINGO GEMM Minimal kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_gemm_minimal_args {
  uint32_t input_A_addr;            
  uint32_t input_B_addr;            
  uint32_t input_C_addr;            
  uint32_t output_D_addr;            
  uint32_t gating_sp_addr;    // SW guard: gating kernel scratchpad (0 = no guard)
  uint32_t cond_node_index;    // SW guard: this node's index in the activation array
  uint32_t scratchpad_ptr;  // pointer to this kernel's scratchpad
} __snax_bingo_kernel_gemm_minimal_args_t;

// DARTS Tier 1: MoE Gating kernel (device-side dynamic routing)
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_moe_gating_args {
    uint32_t logits_addr;
    uint32_t num_experts;
    uint32_t top_k;
    uint32_t cerf_controlled_mask;
    uint32_t cerf_group_ids_addr;
} __snax_kernel_moe_gating_args_t;