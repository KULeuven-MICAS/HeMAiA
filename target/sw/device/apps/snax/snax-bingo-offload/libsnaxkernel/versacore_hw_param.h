// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>

// Generated at 2025-11-27 14:29:23

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

#define tileSize_1 8

#define meshCol_1 64

#define meshRow_2 4

#define tileSize_2 8

#define meshCol_2 64

#define meshRow_3 8

#define tileSize_3 8

#define meshCol_3 64

#define meshRow_4 4

#define tileSize_4 64

#define meshCol_4 4

#define A_elem_len 8

#define B_elem_len 8

#define C_elem_len 32

#define D32_elem_len 32

#define versacore_serial_c_d_width 1024

#define bankWidth 64

#define channel_en_A_0_0 65535

#define channel_en_A_1_0 255

#define channel_en_A_2_0 255

#define channel_en_A_3_0 255

#define channel_en_A_4_0 4294967295

#define channel_en_B_0_0 0

#define channel_en_B_0_1 65535

#define channel_en_B_1_0 4294967295

#define channel_en_B_1_1 4294967295

#define channel_en_B_2_0 4294967295

#define channel_en_B_2_1 4294967295

#define channel_en_B_3_0 4294967295

#define channel_en_B_3_1 4294967295

#define channel_en_B_4_0 0

#define channel_en_B_4_1 4294967295

#define Ctlstride0_0 128

#define Ctlbound0_0 32

#define channel_en_C_0_0 4294967295

#define channel_en_C_null_0_0 0

#define Ctlstride0_1 128

#define Ctlbound0_1 2

#define channel_en_C_1_0 4294967295

#define channel_en_C_null_1_0 0

#define Ctlstride0_2 128

#define Ctlbound0_2 8

#define channel_en_C_2_0 4294967295

#define channel_en_C_null_2_0 0

#define Ctlstride0_3 128

#define Ctlbound0_3 16

#define channel_en_C_3_0 4294967295

#define channel_en_C_null_3_0 0

#define Ctlstride0_4 64

#define Ctlbound0_4 1

#define channel_en_C_4_0 255

#define channel_en_C_null_4_0 0

#define D32tlstride0_0 128

#define D32tlbound0_0 32

#define channel_en_D32_0_0 4294967295

#define D32tlstride0_1 128

#define D32tlbound0_1 2

#define channel_en_D32_1_0 4294967295

#define D32tlstride0_2 128

#define D32tlbound0_2 8

#define channel_en_D32_2_0 4294967295

#define D32tlstride0_3 128

#define D32tlbound0_3 16

#define channel_en_D32_3_0 4294967295

#define D32tlstride0_4 64

#define D32tlbound0_4 1

#define channel_en_D32_4_0 255
