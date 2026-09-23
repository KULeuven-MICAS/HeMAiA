// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
//
// Top-level aggregator for the snax-bingo-offload kernel library. Every
// kernel lives in a partial header under this directory; this file just
// includes them in the right order and defines the exported symbol table
// the host looks up at offload time.
//
// Layout (mirrors the host-side offload_bingo_sw / offload_bingo_hw split):
//   macros.h                             — SNAX_LIB_DEFINE, SNAX_EXPORT_FUNC,
//                                          BINGO_SW_GUARD_CHECK, debug prints.
//   offload_sw_kernels/basic.h           — cluster-level dummy/csr/check_results.
//   offload_sw_kernels/idma.h            — cluster-level iDMA copies + iDMA-backed
//                                          compute-pattern demos.
//   offload_sw_kernels/xdma.h            — cluster-level xDMA kernels.
//   offload_sw_kernels/gemm.h            — cluster-level GEMM kernels (hand-maintained).
//   offload_hw_kernels/basic.h           — core-level dummy/entry_point/exit.
//   offload_hw_kernels/idma.h            — core-level iDMA copies.
//   offload_hw_kernels/xdma.h            — core-level xDMA kernels (data movement).
//   offload_hw_kernels/simd.h            — core-level SIMD kernels (hart 1): the FP16
//                                          stream operators and the fused whole-ops
//                                          (softmax/rmsnorm/silu/swiglu/rope).
//   offload_hw_kernels/gemm.h            — core-level GEMM kernels (hand-maintained).
//   validate_shapes.py                   — lives at runtime/snax/versacore/;
//                                          cross-checks gemm_shapes.h vs hwcfg.

#pragma once

#include "macros.h"
#include "offload_sw_kernels/basic.h"
#include "offload_sw_kernels/idma.h"
#include "offload_sw_kernels/xdma.h"
#include "offload_sw_kernels/gemm.h"
#include "offload_hw_kernels/basic.h"
#include "offload_hw_kernels/idma.h"
#include "offload_hw_kernels/xdma.h"
#include "offload_hw_kernels/simd.h"
#include "offload_hw_kernels/gemm.h"
#include "offload_hw_kernels/gemm_fa.h"

//////////////////////// SYMBOL TABLE ////////////////////////
// The host offload runtime looks up kernels by name through this table.
// Exports must be listed in both branches so the .snax_symtab section
// contains every kernel the device may be asked to run.
SNAX_SYMTAB_SECTION const snax_symbol_t __snax_symtab[] = {
     /// Cluster-level Kernels ///
     /// Used for bingo sw     ///
    SNAX_EXPORT_FUNC(__snax_kernel_dummy),
    SNAX_EXPORT_FUNC(__snax_kernel_sync_probe),
    SNAX_EXPORT_FUNC(__snax_kernel_check_results),
    SNAX_EXPORT_FUNC(__snax_kernel_check_results_full),
    SNAX_EXPORT_FUNC(__snax_kernel_csr),
    SNAX_EXPORT_FUNC(__snax_kernel_load_compute_store),
    SNAX_EXPORT_FUNC(__snax_kernel_double_buffer),
    SNAX_EXPORT_FUNC(__snax_kernel_xdma_1d_copy),
    SNAX_EXPORT_FUNC(__snax_kernel_idma_1d_copy),
    SNAX_EXPORT_FUNC(__snax_kernel_versacore_load_compute_store),
    SNAX_EXPORT_FUNC(__snax_kernel_minimal_cfg_start_gemm_and_wait),
    /// Core-level Kernels ///
    /// Used for bingo hw  ///
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_dummy),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_sync_probe),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_exit),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_idma_1d_copy),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_idma_broadcast),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_idma_pairwise_swap),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_gemm_full),
    // Clean per-precision GEMM wrappers (over gemm_full) — see offload_hw_kernels/gemm.h.
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_gemm_i8i8_i32),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_gemm_i8i4_i32),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_gemm_i4i4_i32),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_gemm_i8i8_i8),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_gemm_i8i4_f16),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_gemm_i8i8_f16),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_gemm_minimal),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_gemm_fa_qk),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_gemm_fa_pv),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_gemm_perf_report),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_1d_copy),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_multicast),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_memset),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_6d),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_transpose_2d),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_submatrix_2d),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_expand_2d),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_concat_2d),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_pad_2d),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_gather_2d),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_pack_fa_partial),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_chain_gather),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_elementwise_add),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_elementwise_add_ab),
    /// Core-level SIMD Kernels  (offload_hw_kernels/simd.h)          ///
    /// All of these run on the SIMD core, hart 1.                     ///
    // The FP16 stream operators the LLM layers decompose into.
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_simd_stream_reduce),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_simd_stream_map),
    // Merged map+reduce: both operators in ONE task (softmax exp + Sexp).
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_simd_stream_map_reduce),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_simd_stream_elementwise),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_simd_fp16_to_int8),
    // RoPE, as ONE task: both products on the pre-map elementwise, their sum on the
    // post-map one, over a 4-row operand block. Pair it with
    // __snax_bingo_kernel_idma_pairwise_swap on the DM core, which produces the xswap
    // slot -- an adjacent-halfword permutation is below the reader AGU's granularity, so
    // no stride expresses it and it has to be a real byte-addressed DMA.
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_simd_rope),
    // Whole FP16 softmax in one kernel: the sub-max/exp/rowsum fused into ONE pass by
    // the pre-map EW0, and the reciprocal done as rsqrt(Sexp*Sexp) on the datapath --
    // the core's integer divide survives only at rows == 1 (cheaper there) and past
    // cols == 255 (where the FP16 square would overflow).
    // Precision picked by name: fp16 out, or int8 out (fused Fp16ToInt8, baked 127.0).
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_simd_softmax_f16_f16),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_simd_softmax_f16_i8),
    // The same distribution over x^T: one token per lane, so BOTH per-row scalars come
    // out of the accumulators with no fold and ride back as sticky seeds -- and the
    // reciprocal becomes rsqrt(s*s), so nothing leaves the datapath.
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_simd_softmax_t_f16_f16),
    // Whole FP16 rmsnorm in one kernel (StreamMap RSQRT), routed by the layout pair:
    // row_major -> row_major | A, col_major -> col_major | B. A and B are the GEMM's
    // operand layouts, fp16 or int8 (caller's scale, or a baked 64.0).
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_simd_rmsnorm),
    // The same normalisation over x^T: one token per lane, so the reduce is LANEWISE and
    // the scale is sticky -- no cross-lane fold, no broadcast plane, ~3x cheaper.
    // Whole FP16 silu / swiglu in one kernel (StreamMap / StreamMap+Elementwise;
    // baked 16.0 int8 scale).
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_simd_silu_f16_f16),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_simd_silu_f16_i8),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_simd_swiglu_f16_f16),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_simd_swiglu_f16_i8),
    // The reconvergence of a conditional fork: out = SUM w[e] * y[e] over the
    // experts the gating kernel selected. The weights are FP32 bits straight into
    // a CSR, so no float arithmetic runs on this FPU-less hart.
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_simd_moe_combine_f16),
    // FlashAttention online-softmax epilogue: the whole per-tile SIMD half (eleven
    // engine tasks -- rowmax, the m/corr/l recurrence, the fused exp+rowsum, the
    // quantise and the O rescale) in ONE kernel. The producing and consuming GEMM
    // nodes hand off through BINGO edges, so it carries no sync counters.
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_simd_fa_softmax),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_layout_convert),
    SNAX_SYMTAB_END
};

// __snax_symtab_start / __snax_symtab_end are provided by the device linker
// script (base.template.ld) as the boundaries of the .snax_symtab section;
// runtime/src/bingo.h declares them extern. No C-side definitions here.
