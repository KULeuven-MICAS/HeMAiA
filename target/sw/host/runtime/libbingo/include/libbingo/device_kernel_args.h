// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include <stdint.h>

#define __SNAX_KERNEL_ARGS_DEFINE typedef struct __attribute__((packed, aligned(4)))

// Every BINGO core-level kernel args struct ends with this 3-field trailer:
//   - gating_sp_addr  : SW guard / CERF group sharing (0 = no guard)
//   - cond_node_index : this node's index in the activation array
//   - scratchpad_ptr  : pointer to this kernel's bingo_kernel_scratchpad_t
//
// The trailer is consumed by BINGO_SW_GUARD_CHECK / BINGO_GET_SP on the
// device side. Append it to every BINGO args struct as the last entry —
// the user's `;` after the macro invocation supplies the `;` for the
// last field (standard preprocessor-list idiom).
//
// gating_sp_addr + cond_node_index in detail
// ------------------------------------------
// These two fields implement the per-task SW-side gate that pairs with
// the HW CERF gating (Tier 1) for fine-grained conditional execution
// inside a fired CERF group: if `gating_sp_addr` is non-zero the device
// kernel reads the upstream gating task's scratchpad to find a uint8_t
// activation[] array, indexes activation[cond_node_index], and
// early-returns BINGO_RET_SUCC when that slot is 0. The full two-tier
// (HW CERF + SW guard) protocol, the activation-array contract, and a
// worked routing example all live next to the macros that consume
// these fields:
//   target/sw/device/apps/snax/snax-bingo-offload/libsnaxkernel/macros.h
// (search for "SW Guard"). Set `gating_sp_addr = 0` on a kernel arg
// struct to disable the guard for that task; the device-side check then
// short-circuits to a single load + branch-not-taken.
//
// scratchpad_ptr in detail
// ------------------------
// Each task is given a 16-word (64-byte) per-task scratchpad allocated by
// the host runtime before dispatch; this field is the low 32 bits of its
// TCDM-local address (kernel runs on 32-bit snitch). The struct layout,
// the BINGO_GET_SCRATCHPAD accessor, and BINGO_SP_PROFILE live in
// shared/runtime/heterogeneous_runtime.h — see that header for the
// canonical definition.
//
// Three roles the scratchpad plays at runtime:
//   1. Result publication: the kernel writes return_value /
//      num_return_values before returning BINGO_RET_SUCC; downstream
//      tasks and the host post-process read them directly.
//   2. SW-guard activation hand-off: a gating task stashes the address
//      of its uint8_t activation[] array into its own return_value;
//      guarded downstream tasks reach it via gating_sp_addr (see SW
//      guard description in libsnaxkernel/macros.h).
//   3. Per-task profiling: BINGO_SP_PROFILE(sp, field, mcycle) is a
//      no-op unless -DBINGO_SCRATCHPAD_PROFILING is set.
#define BINGO_KERNEL_ARGS_TRAILER \
    uint32_t gating_sp_addr;   \
    uint32_t cond_node_index;  \
    uint32_t scratchpad_ptr


// Define the argument structures for the device kernels
// Each structure is packed and aligned to 4 bytes
// The definition should match the kernel function argument parsing in snax_kernel_lib.h


////////////////////////////////////////////////////////////////////////
///////////////////////// Cluster-level Kernels ////////////////////////
////////////////////////////////////////////////////////////////////////

// Note: name start with __snax_kernel_ 

// Dummy kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_dummy_args {
  uint32_t dummy_input;    
} __snax_kernel_dummy_args_t;

// Sync-probe kernel args (cross-chip synchronization latency measurement).
// The kernel does NO work: it exists only so a dependency edge has something to
// terminate on. It stamps the DM core's mcycle into `stamp_addr` so the host can read
// back when the task ran; pass 0 to skip stamping.
// ⚠️ stamp_addr must live on the SAME chiplet the task is assigned to -- it is written
// with a plain local store, and mcycle is not comparable across chiplets anyway.
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_sync_probe_args {
  uint32_t stamp_addr;
} __snax_kernel_sync_probe_args_t;

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

// Cluster-level GEMM kernel args. Layout MUST match the parsing in
// offload_sw_kernels/gemm.h (__snax_kernel_versacore_load_compute_store):
// arg0..14, all uint32_t, packed/aligned(4). Mesh dims are intentionally
// absent — the device looks them up from
// runtime/snax/versacore/gemm_shapes.h via array_shape.
//
// Compute: D = A*B + C
//   A: int8, B: int8, C: int32, D: int32
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_versacore_load_compute_store_args {
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
} __snax_kernel_versacore_load_compute_store_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_minimal_cfg_start_gemm_and_wait_args{
  uint32_t input_A_addr_lo;
  uint32_t input_B_addr_lo;
  uint32_t input_C_addr_lo;
  uint32_t output_D_addr_lo;
} __snax_kernel_minimal_cfg_start_gemm_and_wait_args_t;

////////////////////////////////////////////////////////////////////////
//////////////////////// BINGO Core-level Kernels ////////////////////////
////////////////////////////////////////////////////////////////////////

// Note: name start with __snax_bingo_kernel_

// BINGO Dummy kernel args
// Sync-probe kernel args (cross-chip synchronization latency, arm D).
// Core-level twin of __snax_kernel_sync_probe_args. The kernel does NO work: it exists
// only so a dependency edge has something to terminate on, and stamps mcycle into
// stamp_buf[slot] so the latency between two probes can be read back afterwards.
// ⚠️ stamp_buf must live on the SAME chiplet the task runs on -- it is written with a
// plain local store, and mcycle is not comparable across chiplets anyway.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_sync_probe_args {
  uint32_t stamp_buf;   // base of a uint32_t array, or 0 to skip stamping
  uint32_t slot;        // index into that array
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_sync_probe_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_dummy_args {
  uint32_t dummy_input;            
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_dummy_args_t;
// BINGO Entry kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_entry_args {
  uint32_t start_cc_reg_addr;            
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_entry_args_t;
// BINGO Exit kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_exit_args {
  uint32_t exit_code;            
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_exit_args_t;

// BINGO IDMA 1D Copy kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_idma_1d_copy_args {
  uint32_t src_addr_hi;    
  uint32_t src_addr_lo;            
  uint32_t dst_addr_hi;            
  uint32_t dst_addr_lo;            
  uint32_t size;        // in Bytes
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_idma_1d_copy_args_t;

// BINGO IDMA Broadcast Kernel Args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_idma_broadcast_args {
  uint32_t src_addr_hi;    
  uint32_t src_addr_lo;            
  uint32_t dst_addr_hi;            
  uint32_t dst_addr_lo;            
  uint32_t size;        // in Bytes
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_idma_broadcast_args_t;

// BINGO IDMA Pairwise Swap (flat adjacent-element-pair swap: dst[i] = src[i^1]).
// The data half of RoPE rotate_half; produced on-device via two strided iDMA copies.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_idma_pairwise_swap_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t num_elems;      // total element count (must be even)
  uint32_t elem_bytes;     // element size (1=int8, 2=int16/fp16, 4=int32)
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_idma_pairwise_swap_args_t;

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
  uint32_t quantization_enable;
  uint32_t shift_i;
  uint32_t multiplier_i;
  int32_t input_zp_i;
  int32_t output_zp_i;
  int32_t int32tofp16_enable;
  int32_t int4_a_enable;
  int32_t int4_b_enable;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_gemm_full_args_t;

// ---------------------------------------------------------
// Clean per-precision GEMM wrappers (over __snax_bingo_kernel_gemm_full)
// ---------------------------------------------------------
// These two structs back the precision-named GEMM wrapper kernels in
// offload_hw_kernels/gemm.h, which hide gemm_full's 8 precision/quant flags.
// Each wrapper picks the precision; the user only fills the GEMM operands/shape.
//   plain  -> __snax_bingo_kernel_gemm_i8i8_i32 / _i8i4_i32 / _i4i4_i32 / _i8i4_f16 / _i8i8_f16
//   quant  -> __snax_bingo_kernel_gemm_i8i8_i8  (adds requantization params)
// Compute: D = A*B + C. A/B/C/D are 32-bit cluster-local (L1) addresses.

// Shared "plain" GEMM args (int32 / int4-packed / fp16 outputs — no requant).
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_gemm_args {
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
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_gemm_args_t;

// Quantized GEMM args (int8 output): plain fields + requantization params.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_gemm_quant_args {
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
  uint32_t shift_i;
  uint32_t multiplier_i;
  int32_t  input_zp_i;
  int32_t  output_zp_i;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_gemm_quant_args_t;

// BINGO XDMA 1D Copy kernel args (same layout as cluster-level)
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_1d_copy_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t size;        // in Bytes
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_1d_copy_args_t;

// BINGO XDMA MULTICAST: one read from main memory, N writes into N different clusters' L1.
//
// WHY THIS EXISTS. A quadrant's four clusters mux onto ONE 512-bit path to the SoC, so
// four clusters each pulling the same bytes costs four times the port time for one byte of
// information. FlashAttention parallelised over the query heads of a GQA group is exactly
// that case: the group shares one KV head, so K and V are IDENTICAL on every cluster and
// only Q differs. Reading each KV tile once and fanning it out in the writer takes the
// workload's arithmetic intensity from Br to Br * NCL, which is what puts a 4-cluster
// quadrant back on the compute-bound side of its balance point. See
// docs/fa_decomposition_hierarchy.md.
//
// The fan-out is HARDWARE: the xDMA writer holds XDMA_MAX_DST_COUNT destination address
// slots and commits one transfer to all of them, so this is one task and one finish, not N
// sequential copies. That distinction is the whole point -- a software loop over N unicast
// copies was measured (as the V-push arm) and lost, because its single issuer serialises.
//
//   src      : the source, a FULL 64-bit address -- typically L3 or the memory chiplet.
//   dst[]    : dst_num destinations, each a full (chip|cluster|offset) address, which is
//              what bingo_l1_alloc returns for a handle on ANY cluster. No arithmetic on
//              the Python side and no assumption that the four heaps agree on offsets.
//   size     : bytes, the same to every destination; must be a multiple of the datapath.
//
// BINGO_XDMA_MCAST_MAX bounds the list in the ARG STRUCT only; the hardware bound is
// XDMA_MAX_DST_COUNT (generated and cfg-dependent, not visible in this shared header), so
// the kernel checks against it at call time rather than failing to compile.
#define BINGO_XDMA_MCAST_MAX 8
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_multicast_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_hi[BINGO_XDMA_MCAST_MAX];
  uint32_t dst_lo[BINGO_XDMA_MCAST_MAX];
  uint32_t dst_num;     // live entries in dst[], 1..BINGO_XDMA_MCAST_MAX
  uint32_t size;        // in Bytes
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_multicast_args_t;

// Fill a local L1 region with a repeating 32-bit PATTERN, using the xDMA's writer-side
// Memset extension.
//
// Why this exists rather than a zero-load over the iDMA: FlashAttention seeds m to -inf
// and l to 0, which is 128 bytes of CONSTANTS. Fetching constants from main memory wastes
// the load engine's critical head -- and on a KV-sharded run that is NQ separate loads
// standing in front of K(0). The xDMA core is otherwise idle (on HeMAiA it owns nothing
// but its exit node), so it generates them in place instead: no main-memory read at all.
//
// A 32-bit pattern, not a byte, because FP16 -inf is 0xFBFF and no single byte repeats
// into it. One mechanism then covers INT8, FP16, BF16, FP32 and INT32 constants.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_memset_args {
  // Field ORDER matches every other kernel here: hi before lo. The Python side emits
  // both through _process_addr, which writes <name>_hi and <name>_lo, so a struct that
  // declares them the other way round silently swaps the halves of the address.
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t size;          // bytes, a multiple of the 64-byte beat
  uint32_t pattern;       // the 32-bit word tiled across every beat
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_memset_args_t;

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
  BINGO_KERNEL_ARGS_TRAILER;
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
  BINGO_KERNEL_ARGS_TRAILER;
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
  BINGO_KERNEL_ARGS_TRAILER;
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
  BINGO_KERNEL_ARGS_TRAILER;
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
  BINGO_KERNEL_ARGS_TRAILER;
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
  BINGO_KERNEL_ARGS_TRAILER;
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
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_gather_2d_args_t;

// BINGO XDMA ElementwiseAdd (writer ext: accumulate N int32 operands -> 1).
// Fuses the GEMM K-split partial-sum adds into one streaming xDMA pass.
// num_int32_elem_per_operand must be a multiple of 16 (512b bus / 32b element).
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_elementwise_add_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t num_int32_elem_per_operand;
  uint32_t num_operands;
  uint32_t operand_stride;   // bytes between consecutive operand buffers
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_elementwise_add_args_t;

// BINGO pack of a FlashAttention (m, l) partial into the monoid junction's lane
// geometry, so a chain gather can fold it. See __snax_bingo_kernel_pack_fa_partial.
//
// The softmax arena keeps the running max and the running sum as two FP16 beats, one
// lane per query row. The monoid wants them interleaved as FP32 in ONE stream, at
// lane = field*S + slot: field 0 is the key m, field 1 is the exp-twisted value l, and
// S = 1 << sigma is how many query rows share a beat. A 512b beat holds 16 FP32 lanes,
// so S = 8 is the largest legal choice (F*S = 16) and Br = 32 rows pack into 4 beats.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_pack_fa_partial_args {
  uint32_t src_m_addr_hi;
  uint32_t src_m_addr_lo;
  uint32_t src_l_addr_hi;
  uint32_t src_l_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t n_rows;   // query rows (Br); must be a multiple of `slots`
  uint32_t slots;    // S = 1 << sigma; rows per beat. S * 2 <= 16, so S <= 8.
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_pack_fa_partial_args_t;

// BINGO XDMA ChainGather -- an in-fabric collective fold.
//
// The collector arms a writer JUNCTION and walks a chain of remote partials; each
// crossing folds the arriving stream with that node's local read, so the answer is
// reduced ON THE WAY and only the folded beat lands in the collector's buffer. Arming
// the junction is what turns a chained write into a gather (the frontend sets
// collectiveMode := junctionEnabled), which is why `junction` is not optional.
//
//   local_src : this collector's OWN partial (the reader pointer).
//   chain[]   : the gather path in DATA order, ending at the collector's dst buffer --
//               [S1, S2, ..., dst_local]. Every entry is that node's real partial
//               address; the routers decode only the cluster tag. These are FULL
//               (chip|cluster|offset) addresses, which is exactly what bingo_l1_alloc
//               returns for a handle on another cluster -- no address arithmetic needed
//               on the Python side.
//   junction  : WRITER_JCT_ELEMENTWISEJUNCTION (per-element ADD/MUL/MAX/MIN) or
//               WRITER_JCT_MONOIDJUNCTION (the softmax/attn/norm merge).
//   jct_csr0  : the junction's CSR(0). See snax_xdma_lib.h for both encodings -- the
//               monoid word is a GEOMETRY, not an operator id.
//
// BINGO_XDMA_CHAIN_MAX bounds the chain in the ARG STRUCT only. The hardware bound is
// XDMA_MAX_DST_COUNT (a generated, cfg-dependent define, not visible in this shared
// host/device header); the kernel checks against it at call time.
#define BINGO_XDMA_CHAIN_MAX 8
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_chain_gather_args {
  uint32_t local_src_hi;
  uint32_t local_src_lo;
  uint32_t chain_hi[BINGO_XDMA_CHAIN_MAX];
  uint32_t chain_lo[BINGO_XDMA_CHAIN_MAX];
  uint32_t chain_num;   // live entries in chain[], 2..BINGO_XDMA_CHAIN_MAX
  uint32_t size;        // bytes per partial
  uint32_t junction;    // WRITER_JCT_*
  uint32_t jct_csr0;    // junction CSR(0)
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_chain_gather_args_t;

// BINGO XDMA ElementwiseAdd AB (two-operand) (convenience: dst = a + b, int32).
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_elementwise_add_ab_args {
  uint32_t src_a_addr_hi;
  uint32_t src_a_addr_lo;
  uint32_t src_b_addr_hi;
  uint32_t src_b_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t num_int32_elements;        // multiple of 16
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_elementwise_add_ab_args_t;

// ──────────────────────────────────────────────────────────────────────
// xDMA FP16 streaming-SIMD primitives (reader extensions): the 3 generic ops
// the LLM layers (softmax/rmsnorm/silu/swiglu/rope) decompose into. One row =
// `beats` x 64-byte beats (64 B = 32 FP16). The AGU shape is identical across
// a same-shape group, so csr_mode lets a chain reuse it: 0=FULL (full AGU
// config), 1=STICKY (retask only — addresses + dst bound). dst_bound0 is the
// WRITER beat count the caller computes (1 / beats+1 for reduce; beats /
// beats/2 for map/elementwise when fused-quant packs FP16->INT8).
// ──────────────────────────────────────────────────────────────────────

// StreamReduce: per-row reduction (row -> scalar, op over each row). Runs `rows`
// independent reductions in one dispatch, emitting one splatted scalar beat per
// row (dst_bound0 = rows).
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_simd_stream_reduce_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t beats;            // beats per row (D FP16 elems = beats*32)
  uint32_t op;               // 0=MAX 1=ADD 2=SUMSQ; |0x100=TAP; |0x200=OUT_FP32
                             // (OUT_FP32: scalar emitted in FP32, no FP16 narrow ->
                             //  host reads it as float at stride 16, not u16 at stride 32)
  uint32_t rows;             // independent per-row reductions (1 = single row)
  uint32_t csr_mode;         // 0=FULL, 1=STICKY
  uint32_t dst_bound0;       // writer beats (= rows)
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_simd_stream_reduce_args_t;

// StreamMap: out = func(a*x + b) per element, over rows*beats flat beats. Optional
// fused FP16->INT8 quant.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_simd_stream_map_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t beats;            // beats per row
  uint32_t a_f32bits;        // StreamMap operand a (FP32 bit pattern)
  uint32_t b_f32bits;        // StreamMap operand b (FP32 bit pattern)
  uint32_t func;             // 0=LINEAR_FP16, 1=EXP_FP16, 2=SILU_FP16
  uint32_t rows;             // independent rows (flat outer bound = rows*beats)
  uint32_t csr_mode;         // 0=FULL, 1=STICKY
  uint32_t dst_bound0;       // writer beats (rows*beats; rows*beats/2 if quantizing)
  uint32_t out_dtype;        // 0=FP16, 1=INT8 (chain Fp16ToInt8)
  uint32_t inv_scale_f32bits;// Fp16ToInt8 inv_scale (FP32 bits), read only if out_dtype==1
  // Optional runtime scale: if a_addr (hi|lo) != 0, the DM core reads one FP32 from
  // that L1 word and uses it as the StreamMap 'a' operand (the GEMM->swiglu dequant
  // qsc) instead of the compile-time a_f32bits. One scalar applies to the whole tile.
  uint32_t a_addr_hi;
  uint32_t a_addr_lo;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_simd_stream_map_args_t;

// MERGED StreamMap -||> StreamReduce: map AND reduce in ONE xDMA task, i.e. per row
// out = reduce(reduce_op, map(func, a*x + b)). Both reader extensions are enabled for
// a single task, so the map feeds the reduce inside the datapath and the mapped row is
// never written out and re-read (this is the softmax exp+Sexp fusion; it replaces a
// map task followed by a reduce task that re-reads the whole mapped tensor).
//
// Output layout depends on the TAP bit (0x100) in reduce_op:
//   TAP set   -> the mapped row is passed through 1:1 AND the row scalar is appended as
//                a trailing beat: a PADDED [rows, beats+1] beat tensor. Downstream
//                readers of the mapped data must respect the (beats+1)*64-byte row
//                stride (rows=1 needs nothing: a flat `beats`-beat read stops short of
//                the scalar). dst_bound0 = rows*(beats+1).
//   TAP clear -> only the per-row scalar beats are written (a stream_reduce over the
//                MAPPED values, no passthrough).                dst_bound0 = rows.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_simd_stream_map_reduce_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t beats;            // beats per row; also the StreamReduce operandCount
  uint32_t a_f32bits;        // StreamMap operand a (FP32 bit pattern)
  uint32_t b_f32bits;        // StreamMap operand b (FP32 bit pattern)
  uint32_t func;             // StreamMap: 0=LINEAR_FP16, 1=EXP_FP16, 2=SILU_FP16
  uint32_t reduce_op;        // StreamReduce: 0=MAX 1=ADD 2=SUMSQ; |0x100=TAP; |0x200=OUT_FP32
  uint32_t rows;             // independent rows (flat reader bound = rows*beats)
  uint32_t csr_mode;         // 0=FULL, 1=STICKY
  uint32_t dst_bound0;       // writer beats: rows*(beats+1) with TAP, rows without
  // Optional runtime scale, exactly as in stream_map: if a_addr (hi|lo) != 0 the DM core
  // reads one FP32 from that L1 word and uses it as the StreamMap 'a' operand (the
  // GEMM->rmsnorm dequant qsc) instead of the compile-time a_f32bits.
  uint32_t a_addr_hi;
  uint32_t a_addr_lo;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_simd_stream_map_reduce_args_t;

// Fp16ToInt8: out = clamp(round(x * inv_scale), -128, 127) over rows*beats flat beats, on the
// HasFp16ToInt8 xDMA datapath (dedicated activation fp16 -> int8 GEMM-operand requant, replacing
// the host quantize_f16i8). inv_scale_f32bits = FP32 bits of 127/max|x|. dst_bound0 = rows*beats/2.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_simd_fp16_to_int8 {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t beats;             // beats per row
  uint32_t rows;              // independent rows (flat outer bound = rows*beats)
  uint32_t inv_scale_f32bits; // Fp16ToInt8 inv_scale (FP32 bits) = 127/max|x| (compile-time immediate)
  uint32_t csr_mode;          // 0=FULL, 1=STICKY
  uint32_t dst_bound0;        // int8 writer beats (rows*beats/2)
  uint32_t inv_scale_addr_hi; // optional: L1 addr of a runtime inv_scale FP32 word (requant_scale
  uint32_t inv_scale_addr_lo; // wrote it); read instead of the immediate when (hi|lo) != 0
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_simd_fp16_to_int8_args_t;

// StreamElementwise: out = op(operand_0, operand_1, ...) over `operand_count`
// interleaved streams (stride operand_stride bytes), across rows*beats flat beats.
// Optional fused quant.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_simd_stream_elementwise_args {
  uint32_t src_addr_hi;      // base of operand 0
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t beats;            // beats per row per operand
  uint32_t operand_stride;   // bytes between operands (inner AGU stride)
  uint32_t operand_count;    // number of interleaved operands (e.g. 2)
  uint32_t op;               // 0=MUL_FP16, 1=ADD_FP16
  uint32_t rows;             // independent rows (flat outer bound = rows*beats)
  uint32_t csr_mode;         // 0=FULL, 1=STICKY
  uint32_t dst_bound0;       // writer beats (rows*beats; rows*beats/2 if quantizing)
  uint32_t out_dtype;        // 0=FP16, 1=INT8 (chain Fp16ToInt8)
  uint32_t inv_scale_f32bits;// Fp16ToInt8 inv_scale (FP32 bits), read only if out_dtype==1
  // Optional 2nd operand address: if src_b_addr (hi|lo) != 0, operand_stride is
  // computed at run time as src_b - src_addr (lets operand 0 and 1 be SEPARATE
  // L1 buffers, like __snax_bingo_kernel_xdma_elementwise_add_ab). src_b must be
  // at a HIGHER address than src_addr (the reader only strides forward).
  uint32_t src_b_addr_hi;
  uint32_t src_b_addr_lo;
  // Row stride (BYTES) of the operand tensors, for reading PADDED rows.
  //   0 (default) = the operands are flat/packed [rows,D] -> 2D reader {operand, beat}.
  //   != 0        = each row occupies src_row_stride bytes of which only the first
  //                 `beats` beats are data -> 3D reader {operand, beat, row}, skipping
  //                 the rest. This is how an operand produced by the merged map+reduce
  //                 in TAP mode (rows (beats+1)*64 B apart, trailing scalar beat) is
  //                 consumed. BOTH operands must share this stride, so the broadcast
  //                 operand is written at the same padded stride by the xDMA broadcast
  //                 pass that builds it. The writer is always packed, never padded.
  uint32_t src_row_stride;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_simd_stream_elementwise_args_t;

// Fused FP16 RoPE: iDMA adjacent-pair swap of x + 3 StreamElementwise passes
// (x*cos, xswap*sin, +) -> out. cos_full/sin_signed are precomputed tables; the
// kernel allocates its xswap/tmp1/tmp2 scratch from L1. D = beats*32 fp16 per row.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_simd_rope_args {
  uint32_t x_addr_hi;        // input x (runtime Q/K), fp16 [rows, D]
  uint32_t x_addr_lo;
  uint32_t cos_addr_hi;      // cos_full table, same shape
  uint32_t cos_addr_lo;
  uint32_t sin_addr_hi;      // sin_signed table, same shape
  uint32_t sin_addr_lo;
  uint32_t out_addr_hi;      // output, same shape
  uint32_t out_addr_lo;
  uint32_t cols;             // per-row fp16 length D (a multiple of 32)
  uint32_t rows;             // rows / token positions
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_simd_rope_args_t;

// Whole FP16 softmax in one DM-core kernel: reduce-MAX, device negate (fp16 sign
// flip), broadcast, sub-max, merged EXP+Sexp, integer reciprocal (rv32iM divu),
// broadcast, normalize-MUL, and an optional fused FP16->INT8 quant leaf. The host
// is left with only Load / Store / Check. The user gives just the tensors and shape;
// the kernel derives everything else (beats = cols/32, and the int8 quant scale --
// softmax output is in [0,1], so a fixed 127.0 needs no user input).
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_simd_softmax_args {
  uint32_t input_addr_hi;        // input x, fp16 [rows, cols], packed
  uint32_t input_addr_lo;
  uint32_t output_addr_hi;       // fp16 softmax(x), [rows, cols], packed
  uint32_t output_addr_lo;
  uint32_t rows;                 // independent softmax rows
  uint32_t cols;                 // per-row fp16 length D (a multiple of 32)
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_simd_softmax_args_t;

// Whole FP16 rmsnorm in one DM-core kernel: reduce-SUMSQ, integer 1/sqrt(Sxx/N) (no FPU),
// broadcast, normalize-MUL, and an optional fused FP16->INT8 quant leaf. Same shape/args as
// softmax; the kernel derives beats = cols/32 and the int8 scale (a fixed 64.0). N = cols is
// taken to be a power of two.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_simd_rmsnorm_args {
  uint32_t input_addr_hi;        // input x, fp16 [rows, cols], packed
  uint32_t input_addr_lo;
  uint32_t output_addr_hi;       // fp16 rmsnorm(x), [rows, cols], packed
  uint32_t output_addr_lo;
  uint32_t rows;                 // independent rmsnorm rows
  uint32_t cols;                 // per-row fp16 length D (a power-of-two multiple of 32)
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_simd_rmsnorm_args_t;

// Whole FP16 SiLU (x*sigmoid(x)) in one DM-core kernel: a single StreamMap pass, plus an optional
// fused FP16->INT8 quant leaf. Elementwise; the kernel derives beats = cols/32 and the int8 scale
// (a fixed 16.0). The user gives just the tensors and shape -- no StreamMap / beat / quant detail.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_simd_silu_args {
  uint32_t input_addr_hi;        // input x, fp16 [rows, cols], packed
  uint32_t input_addr_lo;
  uint32_t output_addr_hi;       // fp16 silu(x), [rows, cols], packed
  uint32_t output_addr_lo;
  uint32_t rows;                 // independent silu rows
  uint32_t cols;                 // per-row fp16 length D (a multiple of 32)
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_simd_silu_args_t;

// Whole FP16 SwiGLU (silu(gate) * up) in one DM-core kernel: StreamMap(SiLU) gate->sg (L1 scratch
// the kernel allocates), then StreamElementwise(MUL) sg*up->out, plus an optional fused FP16->INT8
// quant leaf. Elementwise; the kernel derives beats = cols/32 and the int8 scale (a fixed 16.0).
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_simd_swiglu_args {
  uint32_t gate_addr_hi;         // gate input, fp16 [rows, cols], packed
  uint32_t gate_addr_lo;
  uint32_t up_addr_hi;           // up input, fp16 [rows, cols], packed
  uint32_t up_addr_lo;
  uint32_t output_addr_hi;       // fp16 swiglu, [rows, cols], packed
  uint32_t output_addr_lo;
  uint32_t rows;                 // independent swiglu rows
  uint32_t cols;                 // per-row fp16 length D (a multiple of 32)
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_simd_swiglu_args_t;

// FlashAttention online-softmax epilogue -- the whole per-tile SIMD half in one kernel.
// See offload_hw_kernels/simd.h for the recurrence, the arena layout and why the
// adjacencies in it are load-bearing.
//   s16_src   the producing GEMM's D32 output for this tile, fp16 [bc, 32]
//   p8_dst    int8 P^T for the consuming GEMM, bc/2 beats
//   arena     contiguous state + scratch, SIMD_FA_ARENA_BYTES(bc, dhead)
//   tile_idx  0 seeds m, l and O; later tiles carry them forward
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_simd_fa_softmax {
  uint32_t s16_src_addr_hi;
  uint32_t s16_src_addr_lo;
  uint32_t p8_dst_addr_hi;
  uint32_t p8_dst_addr_lo;
  uint32_t arena_addr_hi;
  uint32_t arena_addr_lo;
  uint32_t bc;
  uint32_t dhead;
  uint32_t tile_idx;
  // 1: this kernel seeds m = -inf and l = 0 itself on tile 0 (the historical behaviour).
  // 0: the WORKLOAD has already seeded them -- e.g. with an xDMA memset on the otherwise
  // idle xDMA core, which keeps a pair of constants off the SIMD's critical path in the
  // same way O's zero already is. A workload that passes 0 without arranging the fill
  // reads a stale m and l on its first KV tile, exactly as the O note above warns.
  uint32_t seed_state;
  // Geometry mode; must match SIMD_FA_GEOM_* in offload_hw_kernels/simd.h.
  //   0 SELF      rebuild the task geometry at tile 0 unconditionally. The only safe value
  //               for a graph with no prologue node, because a cold arena's memo word is
  //               whatever L1 happened to hold.
  //   1 PROLOGUE  build the geometry into `arena` and return. Moves no data, queues no
  //               accelerator task and does not touch m, l or O, so the node is pure setup.
  //   2 PRIMED    trust the memo even on tile 0. Legal only when a PROLOGUE node for THIS
  //               arena is an ancestor in the graph.
  // The build is 222 scalar TCDM stores that cost 9-14 cc each once the iDMA is streaming
  // through the same ports -- 4,470 cc measured on decode's first tile. PROLOGUE+PRIMED
  // moves that off the critical path; tile 0 then costs ~150 cc like every other tile.
  uint32_t geom_mode;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_simd_fa_softmax_args_t;


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
// HeMAiA/util/sim/xdma/layout_convert.py — kernels must produce byte-identical
// output to the reference functions.
// ──────────────────────────────────────────────────────────────────────

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_d_to_row_major_args {
  uint32_t src_addr_hi;   // D-layout source
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;   // row-major destination
  uint32_t dst_addr_lo;
  uint32_t M_T;           // VersaCore M-tile count
  uint32_t N_T;           // VersaCore N-tile count
  // NOTE: meshRow/meshCol/elem_bytes are NOT fields. They are part of the KERNEL:
  // __snax_bingo_kernel_xdma_d_to_row_major_e<elem_bytes>_<mesh> binds them as constants.
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_d_to_row_major_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_row_major_to_a_args {
  uint32_t src_addr_hi;   // row-major source
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;   // A-layout destination
  uint32_t dst_addr_lo;
  uint32_t M_T;
  uint32_t K_T;
  // NOTE: meshRow/tileSize/elem_bytes are NOT fields. They are part of the KERNEL:
  // __snax_bingo_kernel_xdma_row_major_to_a_e<elem_bytes>_<mesh> binds them as constants.
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_row_major_to_a_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_row_major_to_b_args {
  uint32_t src_addr_hi;   // row-major source
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;   // B-layout destination
  uint32_t dst_addr_lo;
  uint32_t K_T;
  uint32_t N_T;
  // NOTE: tileSize/meshCol/elem_bytes are NOT fields. They are part of the KERNEL:
  // __snax_bingo_kernel_xdma_row_major_to_b_e<elem_bytes>_<mesh> binds them as constants.
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_row_major_to_b_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_a_to_row_major_args {
  uint32_t src_addr_hi;   // A-layout source
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;   // row-major destination
  uint32_t dst_addr_lo;
  uint32_t M_T;
  uint32_t K_T;
  // NOTE: meshRow/tileSize/elem_bytes are NOT fields. They are part of the KERNEL:
  // __snax_bingo_kernel_xdma_a_to_row_major_e<elem_bytes>_<mesh> binds them as constants.
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_a_to_row_major_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_b_to_row_major_args {
  uint32_t src_addr_hi;   // B-layout source
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;   // row-major destination
  uint32_t dst_addr_lo;
  uint32_t K_T;
  uint32_t N_T;
  // NOTE: tileSize/meshCol/elem_bytes are NOT fields. They are part of the KERNEL:
  // __snax_bingo_kernel_xdma_b_to_row_major_e<elem_bytes>_<mesh> binds them as constants.
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_b_to_row_major_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_row_major_to_d_args {
  uint32_t src_addr_hi;   // row-major source
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;   // D-layout destination
  uint32_t dst_addr_lo;
  uint32_t M_T;
  uint32_t N_T;
  // NOTE: meshRow/meshCol/elem_bytes are NOT fields. They are part of the KERNEL:
  // __snax_bingo_kernel_xdma_row_major_to_d_e<elem_bytes>_<mesh> binds them as constants.
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_row_major_to_d_args_t;

// Reports the accumulated VersaCore hardware counters for one cluster and clears them.
// Runs on the GEMM core AFTER the last matmul, so its printf is outside the measured
// window and cannot perturb what it is reporting.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_gemm_perf_report_args {
  uint32_t perf_addr;         // the five-word L1 accumulator the FA matmuls wrote
  uint32_t ideal_cc;          // cycles this cluster's matmuls would take at 1 pass/cycle
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_gemm_perf_report_args_t;

// BINGO GEMM Minimal kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_gemm_minimal_args {
  uint32_t input_A_addr;            
  uint32_t input_B_addr;            
  uint32_t input_C_addr;            
  uint32_t output_D_addr;            
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_gemm_minimal_args_t;

// The two FlashAttention matmuls (offload_hw_kernels/gemm_fa.h). One struct, two
// kernels: __snax_bingo_kernel_gemm_fa_qk emits the score tile as FP16 for the SIMD
// block, __snax_bingo_kernel_gemm_fa_pv accumulates O += P.V in INT32 with C and D
// pointing at the SAME buffer.
//
// There is no array_shape / transpose / quant field. The cluster has one array shape and
// one data type; attention transposes ALGEBRAICALLY (S^T = K.Q^T is the same GEMM with
// its operands swapped), so no transposer is involved; and neither matmul requantises.
// What these kernels do carry that gemm_full cannot is the INTERLEAVED D layout the
// LANEWISE softmax needs -- see the header comment in gemm_fa.h.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_gemm_fa_args {
  uint32_t input_A_addr;
  uint32_t input_B_addr;
  uint32_t input_C_addr;
  uint32_t output_D_addr;
  uint32_t M;                 // qk: Bc/meshRow      pv: d/meshRow
  uint32_t K;                 // qk: d/tileSize      pv: Bc/tileSize
  uint32_t N;                 // Br/meshCol, the same for both
  // Optional L1 accumulator for the array's OWN hardware counters; 0 disables it and the
  // kernel does not even read the CSRs. Five words:
  //
  //   [0] VERSACORE_PERFORMANCE_COUNTER   cycles the array was in sBUSY
  //   [1] VERSACORE_STALL_A               operand A not delivered
  //   [2] VERSACORE_STALL_B               operand B not delivered
  //   [3] VERSACORE_STALL_D               D drain / accumulator backpressure
  //   [4] dispatches accumulated
  //
  // All four counters RESET ON CONFIG WRITE (VersaCore.scala: `.elsewhen(config_fire)`),
  // so they are per-dispatch and software must accumulate them -- which is also what the
  // snax reference does (snax-flashattn-decode.c, `gemm_cycles += c`). Reading them here
  // rather than timing spans in the trace is what makes this number comparable to the
  // reference's `GEMM core busy %`: a dispatch-to-retire SPAN also contains the CSR
  // writes and the retire poll, and so overstates how busy the array actually was.
  //
  // The array's invariant is stall_a + stall_b + stall_d + passes == performance_counter,
  // so the accumulated set decomposes array time into useful passes and the three reasons
  // it was not making one. A and B stalls are the MEMORY SYSTEM's tax measured inside the
  // array, which is exactly what changes as the machine scales out.
  uint32_t perf_addr;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_gemm_fa_args_t;
