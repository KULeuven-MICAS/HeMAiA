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
//   offload_hw_kernels/xdma.h            — core-level xDMA kernels.
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
#include "offload_hw_kernels/gemm.h"

//////////////////////// SYMBOL TABLE ////////////////////////
// The host offload runtime looks up kernels by name through this table.
// Exports must be listed in both branches so the .snax_symtab section
// contains every kernel the device may be asked to run.
SNAX_SYMTAB_SECTION const snax_symbol_t __snax_symtab[] = {
     /// Cluster-level Kernels ///
     /// Used for bingo sw     ///
    SNAX_EXPORT_FUNC(__snax_kernel_dummy),
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
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_exit),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_idma_1d_copy),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_idma_broadcast),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_gemm_full),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_gemm_minimal),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_1d_copy),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_6d),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_transpose_2d),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_submatrix_2d),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_expand_2d),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_concat_2d),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_pad_2d),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_gather_2d),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_d_to_row_major),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_row_major_to_a),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_row_major_to_b),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_a_to_row_major),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_b_to_row_major),
    SNAX_EXPORT_FUNC(__snax_bingo_kernel_xdma_row_major_to_d),
    SNAX_SYMTAB_END
};

// __snax_symtab_start / __snax_symtab_end are provided by the device linker
// script (base.template.ld) as the boundaries of the .snax_symtab section;
// runtime/src/bingo.h declares them extern. No C-side definitions here.
