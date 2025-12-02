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

// Dummy kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_dummy_args {
  uint32_t dummy_arg_0;    
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