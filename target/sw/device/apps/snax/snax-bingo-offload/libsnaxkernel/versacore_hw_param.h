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

#define chanelEnA_0 65535
#define chanelEnA_1 255

#define chanelEnB_0_0 255
#define chanelEnB_0_1 0
#define chanelEnB_1_0 4294967295
#define chanelEnB_1_1 4294967295

#define Ctlbound_0_0 (meshRow_0 * tileSize_0 * C_elem_len / versacore_serial_c_d_width)
#define Ctlbound_1_0 1

#define c_spatial_bound_0_0 (versacore_serial_c_d_width / bankWidth)
#define c_spatial_bound_0_1 (meshCol_1 * meshRow_1 * C_elem_len / bankWidth)

#define chanelEnC_0 65535   
#define chanelEnC_1 255
#define chanelEnC_C_null 0

#define D32tlbound_0_0 (meshRow_0 * tileSize_0 * C_elem_len / versacore_serial_c_d_width)
#define D32tlbound_1_0 1

#define d32_spatial_bound_0_0 (versacore_serial_c_d_width / bankWidth)
#define d32_spatial_bound_0_1 (meshCol_1 * meshRow_1 * D32_elem_len / bankWidth)

#define chanelEnD32_0 65535   
#define chanelEnD32_1 255
#define chanelEnD32_C_null 0
