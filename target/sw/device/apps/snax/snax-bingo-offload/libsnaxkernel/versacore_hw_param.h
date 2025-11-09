// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>

#pragma once

// #define VERSACORE_DEBUG
#ifdef VERSACORE_DEBUG
#define VERSACORE_DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define VERSACORE_DEBUG_PRINT(...)
#endif

// useful c macro for Versacore
// highly coupled with the hardware parameters
#define meshRow_0 32
#define tileSize_0 4
#define meshCol_0 32

#define meshRow_1 1
#define tileSize_1 32
#define meshCol_1 16

#define A_elem_len 8
#define B_elem_len 8
#define C_elem_len 32
#define D32_elem_len 32

#define versacore_serial_c_d_width 1024

#define bankWidth 64

// GEMM mode 16 channels are all used
#define chanelEnA_0 65535
// GEMV mode 4 channels are used, but safely pad to 8 for wide access
#define chanelEnA_1 255

// GEMM mode 16 channels used
#define chanelEnB_0_0 0
// use the first 16 highest channels
#define chanelEnB_0_1 65535
// GEMV mode 64 channels used
#define chanelEnB_1_0 4294967295
#define chanelEnB_1_1 4294967295

// serial output for C, need 32 cycles to output a data
#define Ctlbound_0_0 (meshRow_0 * meshCol_0 * C_elem_len / versacore_serial_c_d_width)
// one cycle output
#define Ctlbound_1_0 1

// 16 channels are all used
#define c_spatial_bound_0_0 (versacore_serial_c_d_width / bankWidth)
// 8 channels are used
#define c_spatial_bound_0_1 (meshCol_1 * meshRow_1 * C_elem_len / bankWidth)

// 16 channels are all used
#define chanelEnC_0 65535
// 8 channels used 
#define chanelEnC_1 255
// C null, streamer C disabled
#define chanelEnC_C_null 0

// serial output for D32, need 32 cycles to output a data
#define D32tlbound_0_0 (meshRow_0 * meshCol_0 * C_elem_len / versacore_serial_c_d_width)
// one cycle output
#define D32tlbound_1_0 1

// 16 channels are all used
#define d32_spatial_bound_0_0 (versacore_serial_c_d_width / bankWidth)
#define d32_spatial_bound_0_1 (meshCol_1 * meshRow_1 * D32_elem_len / bankWidth)

// 16 channels are all used
#define chanelEnD32_0 65535
// 8 channels used
#define chanelEnD32_1 255
// D32 null, streamer D32 disabled
#define chanelEnD32_C_null 0
