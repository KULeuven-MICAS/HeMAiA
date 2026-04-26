// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
//
// Hand-maintained per-shape parameter table for the core-level bingo-hw GEMM
// kernel. This header pairs with offload_hw_kernels/gemm.h (also committed)
// and is validated against the hwcfg at build time by
// libsnaxkernel/validate_shapes.py — any mismatch fails `make sw`.
//
// Derivation (bw := BINGO_BANK_WIDTH, ceil_8(x) := ((x + 7) / 8) * 8):
//
//   Ctlbound0    = max(1, meshRow * meshCol * BINGO_C_ELEM_LEN / BINGO_SERIAL_C_D_WIDTH)
//   Ctlstride0   = min(BINGO_SERIAL_C_D_WIDTH, meshRow*meshCol*BINGO_C_ELEM_LEN) / bw * (bw / 8)
//   D32tlbound0  = Ctlbound0
//   D32tlstride0 = Ctlstride0
//
//   channel_en_A: bits = max(8, ceil_8(meshRow*tileSize*BINGO_A_ELEM_LEN / bw))
//   channel_en_B: bits = max(8, ceil_8(meshCol*tileSize*BINGO_B_ELEM_LEN / bw))
//   channel_en_C: bits =        ceil_8(meshRow*meshCol *BINGO_C_ELEM_LEN / bw)
//   channel_en_D32: same bits as channel_en_C
//
// Once you have `bits`, pack the low `bits` bits as a bitmask across
// BINGO_*_CSR_NUM uint32 words, with the most-significant word first in the
// initializer:
//   csr_num == 1:  { low_bits_mask }
//   csr_num == 2:  { high_32_bits_mask, low_32_bits_mask }
// where low_bits_mask  = (bits >= 32) ? 0xFFFFFFFFu : ((1u << bits) - 1)
//       high_bits_mask = (bits >  32) ? ((1u << (bits - 32)) - 1) : 0u
//
// To add a shape: bump BINGO_NUM_ARRAY_SHAPES, append a [N] = { … } entry
// below (writing meshRow/tileSize/meshCol and computing the derived fields
// with the formulas above), update the hwcfg to match, and rerun `make sw`.
// validate_shapes.py will confirm the numbers.

#pragma once

#include <stdint.h>

// ----------------------------------------------------------------------
// Shape-invariant widths — must match the active hwcfg
// (snax_versacore_to_cluster.hjson). Checked by validate_shapes.py.
// ----------------------------------------------------------------------
#define BINGO_BANK_WIDTH          64
#define BINGO_A_ELEM_LEN          8
#define BINGO_B_ELEM_LEN          8
#define BINGO_C_ELEM_LEN          32
#define BINGO_D32_ELEM_LEN        32
#define BINGO_SERIAL_C_D_WIDTH    1024

// Per-stream CSR counts (how many uint32 words each channel-enable mask
// spans). Derived from the hwcfg's array-width fields.
#define BINGO_A_CSR_NUM           1
#define BINGO_B_CSR_NUM           2
#define BINGO_C_CSR_NUM           1
#define BINGO_D32_CSR_NUM         1

#define BINGO_NUM_ARRAY_SHAPES 3

typedef struct {
    uint32_t meshRow;
    uint32_t tileSize;
    uint32_t meshCol;
    uint32_t Ctlbound0;
    uint32_t Ctlstride0;
    uint32_t D32tlbound0;
    uint32_t D32tlstride0;
    uint32_t channel_en_A  [BINGO_A_CSR_NUM];
    uint32_t channel_en_B  [BINGO_B_CSR_NUM];
    uint32_t channel_en_C  [BINGO_C_CSR_NUM];
    uint32_t channel_en_D32[BINGO_D32_CSR_NUM];
} bingo_gemm_shape_params_t;

static const bingo_gemm_shape_params_t
    bingo_gemm_shape_params[BINGO_NUM_ARRAY_SHAPES] = {
    [0] = {
        .meshRow      = 32u,
        .tileSize     = 2u,
        .meshCol      = 32u,
        .Ctlbound0    = 32u,
        .Ctlstride0   = 128u,
        .D32tlbound0  = 32u,
        .D32tlstride0 = 128u,
        .channel_en_A   = { 0xffu },
        .channel_en_B   = { 0x0u, 0xffu },
        .channel_en_C   = { 0xffffffffu },
        .channel_en_D32 = { 0xffffffffu },
    },
    [1] = {
        .meshRow      = 1u,
        .tileSize     = 16u,
        .meshCol      = 32u,
        .Ctlbound0    = 1u,
        .Ctlstride0   = 128u,
        .D32tlbound0  = 1u,
        .D32tlstride0 = 128u,
        .channel_en_A   = { 0xffu },
        .channel_en_B   = { 0xffffffffu, 0xffffffffu },
        .channel_en_C   = { 0xffffu },
        .channel_en_D32 = { 0xffffu },
    },
    [2] = {
        .meshRow      = 16u,
        .tileSize     = 8u,
        .meshCol      = 16u,
        .Ctlbound0    = 8u,
        .Ctlstride0   = 128u,
        .D32tlbound0  = 8u,
        .D32tlstride0 = 128u,
        .channel_en_A   = { 0xffffu },
        .channel_en_B   = { 0x0u, 0xffffu },
        .channel_en_C   = { 0xffffffffu },
        .channel_en_D32 = { 0xffffffffu },
    },
};

// Shape-invariant zero mask for C (used when accumPrevC==1 or addNonZeroC==0).
static const uint32_t bingo_channel_en_C_null[BINGO_C_CSR_NUM] = { 0u };
