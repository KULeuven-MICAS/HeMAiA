// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
//
// Core-level GEMM kernels (bingo-hw flow). Hand-maintained plain C header.
// Paired with offload_hw_kernels/gemm_shapes.h, which holds the per-shape
// parameter table and shape-invariant widths. Both headers are validated
// against the hwcfg at build time by libsnaxkernel/validate_shapes.py —
// any drift fails `make sw`.
//
// Kernels exposed:
//   - __snax_bingo_kernel_gemm_full       (configures streamer+versacore and runs)
//   - __snax_bingo_kernel_gemm_minimal    (reuses prior config, just starts/waits)
// Both return uint32_t (BINGO_RET_SUCC / BINGO_RET_FAIL). Shape-dependent
// parameters are read via `bingo_gemm_shape_params[array_shape_idx]` —
// a typed const struct from gemm_shapes.h — rather than dispatched through
// `switch (array_shape_idx)` blocks. A single bounds check at the top of
// __snax_bingo_kernel_gemm_full handles out-of-range indices.

#pragma once

#include "../macros.h"
#include <snax_versacore_lib.h>
#include <gemm_shapes.h>

// =============================================================
// Core-level GEMM kernels (bingo-hw)
// =============================================================

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_gemm_full(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_gemm_full_args_t);
    // Assume the matrix data are all in L1
    // So all the addr are 32bit local addr
    // This kernel will configure the versacore and streamer and start the computation
    // There is another __snax_bingo_kernel_gemm_minimal kernel that only starts the versacore and streamer with pre-configured CSRs
    if (snrt_cluster_core_idx() != 0)
    {
        printf_safe("[Cluster %d Core %d]: Error! Bingo GEMM full should be called from core 0!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_gemm_full_args_t *args =
        (const __snax_bingo_kernel_gemm_full_args_t *)arg;
    uint32_t A_addr = args->input_A_addr;
    uint32_t B_addr = args->input_B_addr;
    uint32_t C_addr = args->input_C_addr;
    uint32_t D_addr = args->output_D_addr;
    VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Bingo GEMM Full called with A_addr=0x%08x, B_addr=0x%08x, C_addr=0x%08x, D_addr=0x%08x\r\n",
                          snrt_cluster_idx(), snrt_cluster_core_idx(),
                          A_addr, B_addr, C_addr, D_addr);
    uint32_t M = args->M;
    uint32_t K = args->K;
    uint32_t N = args->N;
    uint32_t array_shape_idx = args->array_shape_idx;
    uint32_t transpose_A = args->transpose_A;
    uint32_t transpose_B = args->transpose_B;
    uint32_t accumPrevC = args->accumPrevC;
    uint32_t quantization_enable = args->quantization_enable;
    uint32_t shift_i = args->shift_i;
    uint32_t multiplier_i = args->multiplier_i;
    int32_t input_zp_i = args->input_zp_i;
    int32_t output_zp_i = args->output_zp_i;
    int32_t int32tofp16_enable = args->int32tofp16_enable;
    int32_t int4_a_enable = args->int4_a_enable;
    int32_t int4_b_enable = args->int4_b_enable;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_gemm_full_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    // Bounds-check array_shape_idx against the hwcfg, then grab the per-shape
    // parameter block. All shape-dependent values are read from `shape` below
    // instead of dispatching through `switch (array_shape_idx)`.
    if (array_shape_idx >= BINGO_NUM_ARRAY_SHAPES)
    {
        VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Error! array_shape_idx=%d invalid (only %d shapes in hwcfg)\r\n",
                              snrt_cluster_idx(), snrt_cluster_core_idx(),
                              array_shape_idx, BINGO_NUM_ARRAY_SHAPES);
        return BINGO_RET_FAIL;
    }
    const bingo_gemm_shape_params_t *shape = &bingo_gemm_shape_params[array_shape_idx];
    const uint32_t meshRow = shape->meshRow;
    const uint32_t tileSize = shape->tileSize;
    const uint32_t meshCol = shape->meshCol;

    if (quantization_enable && int32tofp16_enable)
    {
        VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Error! quantization and int32tofp16 cannot both be enabled\r\n",
                              snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    const uint32_t a_elem_len = int4_a_enable ? 4u : BINGO_A_ELEM_LEN;
    const uint32_t b_elem_len = int4_b_enable ? 4u : BINGO_B_ELEM_LEN;
    const uint32_t d_elem_len = quantization_enable ? 8u
                                                    : (int32tofp16_enable ? 16u : BINGO_D32_ELEM_LEN);
    const uint32_t has_d_extension = quantization_enable || int32tofp16_enable;
    uint32_t channel_en_A_dyn[BINGO_A_CSR_NUM] = {0};
    uint32_t channel_en_B_dyn[BINGO_B_CSR_NUM] = {0};
    uint32_t channel_en_D32_dyn[BINGO_D32_CSR_NUM] = {0};

    // dynamically construct channel_en masks
    const uint32_t *channel_en_A_ptr = shape->channel_en_A;
    if (int4_a_enable)
    {
        uint32_t bits = (meshRow * tileSize * a_elem_len + BINGO_BANK_WIDTH - 1) /
                        BINGO_BANK_WIDTH;
        if (bits == 0)
        {
            bits = 1;
        }
        for (uint32_t i = 0; i < bits; i++)
        {
            uint32_t idx = i / 32;
            if (idx < BINGO_A_CSR_NUM)
            {
                // high location of channel_en_A_dyn relates to low channel index
                channel_en_A_dyn[BINGO_A_CSR_NUM - 1 - idx] |=
                    (uint32_t)(1u << (i % 32));
            }
        }
        channel_en_A_ptr = channel_en_A_dyn;
    }

    const uint32_t *channel_en_B_ptr = shape->channel_en_B;
    if (int4_b_enable)
    {
        uint32_t bits = (meshCol * tileSize * b_elem_len + BINGO_BANK_WIDTH - 1) /
                        BINGO_BANK_WIDTH;
        if (bits == 0)
        {
            bits = 1;
        }
        for (uint32_t i = 0; i < bits; i++)
        {
            uint32_t idx = i / 32;
            if (idx < BINGO_B_CSR_NUM)
            {
                channel_en_B_dyn[BINGO_B_CSR_NUM - 1 - idx] |=
                    (uint32_t)(1u << (i % 32));
            }
        }
        channel_en_B_ptr = channel_en_B_dyn;
    }

    const uint32_t *channel_en_D32_ptr = shape->channel_en_D32;
    if (has_d_extension)
    {
        uint32_t bits = (BINGO_SERIAL_C_D_WIDTH + BINGO_BANK_WIDTH - 1) /
                        BINGO_BANK_WIDTH;
        for (uint32_t i = 0; i < bits; i++)
        {
            uint32_t idx = i / 32;
            if (idx < BINGO_D32_CSR_NUM)
            {
                channel_en_D32_dyn[BINGO_D32_CSR_NUM - 1 - idx] |=
                    (uint32_t)(1u << (i % 32));
            }
        }
        channel_en_D32_ptr = channel_en_D32_dyn;
    }

    // some inferenced args
    uint32_t addNonZeroC;
    if (accumPrevC)
    {
        addNonZeroC = 0;
    }
    else if (C_addr != 0)
    {
        addNonZeroC = 1;
    }
    else
    {
        addNonZeroC = 0;
    }
    // Configuration the Steamer and Versacore
    //////////////////////////////////////////////////////////////
    // Streamer cfg for A
    //////////////////////////////////////////////////////////////

    // Aslstride0
    uint32_t Aslstride0 = BINGO_BANK_WIDTH / 8;
    // Atlbound0~5
    uint32_t Atlbound[6];
    // Atlbound0
    Atlbound[0] = K;
    // Atlbound1
    Atlbound[1] = N;
    // Atlbound2
    Atlbound[2] = M;
    // Atlbound3
    Atlbound[3] = 1;
    // Atlbound4
    Atlbound[4] = 1;
    // Atlbound5
    Atlbound[5] = 1;
    uint32_t Atlstride[6];
    // Atlstride0
    Atlstride[0] = a_elem_len * meshRow * tileSize / 8;
    // Atlstride1
    Atlstride[1] = 0;
    // Atlstride2
    Atlstride[2] = a_elem_len * meshRow * tileSize * K / 8;
    // Atlstride3
    Atlstride[3] = 0;
    // Atlstride4
    Atlstride[4] = 0;
    // Atlstride5
    Atlstride[5] = 0;
    uint32_t set_addr_remap_index_A = 0;
    //////////////////////////////////////////////////////////////
    // Streamer cfg for B
    //////////////////////////////////////////////////////////////
    // Bslstride0
    uint32_t Bslstride0 = BINGO_BANK_WIDTH / 8;
    // Btlbound0~2
    uint32_t Btlbound[3];
    // Btlbound0
    Btlbound[0] = K;
    // Btlbound1
    Btlbound[1] = N;
    // Btlbound2
    Btlbound[2] = M;
    uint32_t Btlstride[3];
    // Btlstride0
    Btlstride[0] = b_elem_len * tileSize * meshCol / 8;
    // Btlstride1
    Btlstride[1] = b_elem_len * tileSize * meshCol * K / 8;
    // Btlstride2
    Btlstride[2] = 0;
    uint32_t set_addr_remap_index_B = 0;
    //////////////////////////////////////////////////////////////
    // Streamer cfg for C
    //////////////////////////////////////////////////////////////
    // Cslstride0
    uint32_t Cslstride0 = BINGO_BANK_WIDTH / 8;
    // Ctlbound0~3
    uint32_t Ctlbound[4];
    Ctlbound[0] = (accumPrevC == 1) ? 0 : shape->Ctlbound0;
    // Ctlbound1
    Ctlbound[1] = N;
    // Ctlbound2
    Ctlbound[2] = M;
    // Ctlbound3
    Ctlbound[3] = 1;
    uint32_t Ctlstride[4];
    // Ctlstride0
    Ctlstride[0] = shape->Ctlstride0;
    // Ctlstride1
    Ctlstride[1] = BINGO_C_ELEM_LEN * meshRow *
                   meshCol / 8;
    // Ctlstride2
    Ctlstride[2] = N * BINGO_C_ELEM_LEN *
                   meshRow *
                   meshCol / 8;
    // Ctlstride3
    Ctlstride[3] = 0;
    uint32_t set_addr_remap_index_C = 0;
    // Pick the shape's channel_en_C mask, or the shape-invariant zero mask when
    // accumPrevC or !addNonZeroC.
    const uint32_t *channel_en_C_ptr = (accumPrevC == 1 || addNonZeroC == 0)
                                           ? bingo_channel_en_C_null
                                           : shape->channel_en_C;
    //////////////////////////////////////////////////////////////
    // Streamer cfg for D
    //////////////////////////////////////////////////////////////
    // D32slstride0
    uint32_t D32slstride0 = BINGO_BANK_WIDTH / 8;
    // D32tlbound0~3
    uint32_t D32tlbound[4];
    uint32_t D32tlstride[4];
    if (has_d_extension)
    {
        const uint32_t output_bits = meshRow * meshCol * d_elem_len;
        if (output_bits > BINGO_SERIAL_C_D_WIDTH)
        {
            // one output tile is larger than the streamer width, need to split into multiple stores
            if ((output_bits % BINGO_SERIAL_C_D_WIDTH) != 0)
            {
                VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Error! D extension output width is not streamer-aligned\r\n",
                                      snrt_cluster_idx(), snrt_cluster_core_idx());
                return BINGO_RET_FAIL;
            }
            D32tlbound[0] = output_bits / BINGO_SERIAL_C_D_WIDTH;
            D32tlbound[1] = N * M;
            D32tlbound[2] = 1;
            D32tlbound[3] = 1;
            D32tlstride[0] = BINGO_SERIAL_C_D_WIDTH / 8;
            // D32tlbound[1] times 1 tile store
            D32tlstride[1] = d_elem_len * meshRow *
                             meshCol / 8;
            D32tlstride[2] = 0;
            D32tlstride[3] = 0;
        }
        else
        {
            // one output tile fits within the streamer width, will pack multiple tiles into one store
            D32tlbound[0] = 1;
            uint32_t output_matrix_per_store = BINGO_SERIAL_C_D_WIDTH / output_bits;
            if (((M * N * output_bits) % BINGO_SERIAL_C_D_WIDTH) != 0)
            {
                VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Error! D extension output does not fill streamer stores cleanly\r\n",
                                      snrt_cluster_idx(), snrt_cluster_core_idx());
                return BINGO_RET_FAIL;
            }
            D32tlbound[1] = N * M / output_matrix_per_store;
            D32tlbound[2] = 1;
            D32tlbound[3] = 1;
            // bound0 is 0
            D32tlstride[0] = 0;
            // each is BINGO_SERIAL_C_D_WIDTH width
            D32tlstride[1] = BINGO_SERIAL_C_D_WIDTH / 8;
            D32tlstride[2] = 0;
            D32tlstride[3] = 0;
        }
    }
    else
    {
        D32tlbound[0] = shape->D32tlbound0;
        // D32tlbound1
        D32tlbound[1] = N;
        // D32tlbound2
        D32tlbound[2] = M;
        // D32tlbound3
        D32tlbound[3] = 1;
        // D32tlstride0
        D32tlstride[0] = shape->D32tlstride0;
        // D32tlstride1
        // /8 for bit to byte
        D32tlstride[1] = BINGO_D32_ELEM_LEN * meshRow *
                         meshCol / 8;
        // D32tlstride2
        D32tlstride[2] = N * BINGO_D32_ELEM_LEN *
                         meshRow *
                         meshCol / 8;
        // D32tlstride3
        D32tlstride[3] = 0;
    }
    uint32_t set_addr_remap_index_D32 = 0;
    //////////////////////////////////////////////////////////////
    // Configuration the Steamer and Versacore
    //////////////////////////////////////////////////////////////
    VERSACORE_DEBUG_PRINT(
        "Bingo GEMM Full Kernel Compute Streamer Cfg Start!\r\n");
    set_versacore_streamer_csr(
        A_addr,                         // A_addr
        &Aslstride0,                    // Aslstride[] base
        Atlbound,                       // Atlbound[] base
        Atlstride,                      // Atlstride[] base
        set_addr_remap_index_A,         // set_addr_remap_index_A
        transpose_A,                    // transpose_A
        (uint32_t *)channel_en_A_ptr,   // channel_en_A []
        B_addr,                         // B_addr
        &Bslstride0,                    // Bslstride[] base
        Btlbound,                       // Btlbound[] base
        Btlstride,                      // Btlstride[] base
        set_addr_remap_index_B,         // set_addr_remap_index_B
        transpose_B,                    // transpose_B
        (uint32_t *)channel_en_B_ptr,   // channel_en_B []
        C_addr,                         // C_addr
        &Cslstride0,                    // Cslstride[] base
        Ctlbound,                       // Ctlbound[] base
        Ctlstride,                      // Ctlstride[] base
        set_addr_remap_index_C,         // set_addr_remap_index_C
        (uint32_t *)channel_en_C_ptr,   // channel_en_C []
        D_addr,                         // D_addr
        &D32slstride0,                  // D32slstride[] base
        D32tlbound,                     // D32tlbound[] base
        D32tlstride,                    // D32tlstride[] base
        set_addr_remap_index_D32,       // set_addr_remap_index_D32
        (uint32_t *)channel_en_D32_ptr, // channel_en_D32 []
        array_shape_idx,
        quantization_enable,
        shift_i,
        multiplier_i,
        input_zp_i,
        output_zp_i,
        int32tofp16_enable,
        int4_a_enable,
        int4_b_enable);
    set_versacore_csr(
        // accPrevC means takes new C
        accumPrevC == 0,
        K,
        N * M,
        0,
        array_shape_idx,
        0);
    VERSACORE_DEBUG_PRINT(
        "Bingo GEMM Full Kernel Streamer Configuration Done!\r\n");
    // Set CSR to start Streamer
    start_versacore_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    // Poll until Streamer and GEMM accelerator finish
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    wait_versacore_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
    VERSACORE_DEBUG_PRINT("Bingo GEMM Full Kernel Compute Done!\r\n");
    sp->return_value = D_addr;
    sp->num_return_values = M * N;
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_gemm_minimal(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_gemm_minimal_args_t);
    // This kernel will only start the versacore and streamer with pre-configured CSRs
    if (snrt_cluster_core_idx() != 0)
    {
        printf_safe("[Cluster %d Core %d]: Error! Bingo GEMM minimal should be called from core 0!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t A_addr = ((uint32_t *)arg)[0];
    uint32_t B_addr = ((uint32_t *)arg)[1];
    uint32_t C_addr = ((uint32_t *)arg)[2];
    uint32_t D_addr = ((uint32_t *)arg)[3];
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_gemm_minimal_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_CFG_START);
    set_minimal_streamer_cfg(
        A_addr,
        B_addr,
        C_addr,
        D_addr);
    start_versacore_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_RUN_START);
    wait_versacore_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_RUN_END);
    VERSACORE_DEBUG_PRINT("Bingo GEMM Minimal Kernel Compute Done!\r\n");
    sp->return_value = D_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}
