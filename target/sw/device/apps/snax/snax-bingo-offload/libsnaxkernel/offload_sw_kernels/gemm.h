// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
//
// Cluster-level GEMM kernels (bingo-sw flow). Hand-maintained plain C header.
// Pairs with runtime/snax/versacore/gemm_shapes.h (the shared per-shape
// parameter table keyed by array_shape_idx) and with offload_hw_kernels/gemm.h
// on the core-level path. validate_shapes.py cross-checks gemm_shapes.h
// against the hwcfg at build time.
//
// Kernels exposed:
//   - __snax_kernel_versacore_load_compute_store
//       (full load/compute/store; matches __snax_kernel_versacore_load_compute_store_args_t)
//   - __snax_kernel_minimal_cfg_start_gemm_and_wait
// Both return `void`. Shape-dependent parameters are read from
// bingo_gemm_shape_params[array_shape_idx] directly

#pragma once

#include "../macros.h"
#include <snax_versacore_lib.h>
#include <gemm_shapes.h>

// Compute D = A*B + C. For each operand the kernel decides per-call whether
// to DMA it in from L3 or use the host-staged L1 copy directly (see the
// load_A/load_B/load_C/store_D logic below).
SNAX_LIB_DEFINE void __snax_kernel_versacore_load_compute_store(void *arg)
{
    if (snrt_is_dm_core())
    {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        VERSACORE_DEBUG_PRINT("GEMM Intra-Chiplet Kernel Start!\r\n");

        // Allocate the cluster-shared scratchpad and DMA the 15 packed uint32
        // args from L3 into it. The scratchpad layout is documented below.
        get_cls_shared_ptrs()[0] = (uint32_t *)snrt_l1_malloc(128);
        if (!get_cls_shared_ptrs()[0]) {
            VERSACORE_DEBUG_PRINT("ERROR: arg alloc failed!\r\n");
            return;
        }
        VERSACORE_DEBUG_PRINT("Allocated arg at %p\r\n",
                              (void *)get_cls_shared_ptrs()[0]);
        snrt_dma_start_1d(
            (void *)get_cls_shared_ptrs()[0], (void *)arg,
            sizeof(uint32_t) * 15);
        snrt_dma_wait_all();

        uint32_t *arg_ptr = get_cls_shared_ptrs()[0];
        uint32_t A_addr_hi = arg_ptr[0];
        uint32_t A_addr_lo = arg_ptr[1];
        uint32_t B_addr_hi = arg_ptr[2];
        uint32_t B_addr_lo = arg_ptr[3];
        uint32_t C_addr_hi = arg_ptr[4];
        uint32_t C_addr_lo = arg_ptr[5];
        uint32_t D_addr_hi = arg_ptr[6];
        uint32_t D_addr_lo = arg_ptr[7];
        uint32_t M = arg_ptr[8];
        uint32_t K = arg_ptr[9];
        uint32_t N = arg_ptr[10];
        uint32_t array_shape_idx = arg_ptr[11];
        uint32_t transpose_A = arg_ptr[12];
        uint32_t transpose_B = arg_ptr[13];
        uint32_t accumPrevC = arg_ptr[14];
        // VERSACORE_DEBUG_PRINT("chip id: %d\r\n", get_current_chip_id());
        VERSACORE_DEBUG_PRINT("A addr: 0x%08x%08x\r\n", A_addr_hi, A_addr_lo);
        VERSACORE_DEBUG_PRINT("B addr: 0x%08x%08x\r\n", B_addr_hi, B_addr_lo);
        VERSACORE_DEBUG_PRINT("C addr: 0x%08x%08x\r\n", C_addr_hi, C_addr_lo);
        VERSACORE_DEBUG_PRINT("D addr: 0x%08x%08x\r\n", D_addr_hi, D_addr_lo);

        // Shared cluster scratchpad layout (get_cls_shared_ptrs()[0][i]).
        // Slots 0..14 mirror the host-side __snax_kernel_versacore_load_compute_store_args_t.
        // Slots 15..23 are populated by the dm core below and read by every
        // cluster core after the barrier. Mesh dimensions are read directly
        // from bingo_gemm_shape_params[array_shape_idx] (no stashing needed).
        //
        //   slot   field            source
        //   ----   -----            ------
        //    0     A_addr_hi        kernel arg 0
        //    1     A_addr_lo        kernel arg 1
        //    2     B_addr_hi        kernel arg 2
        //    3     B_addr_lo        kernel arg 3
        //    4     C_addr_hi        kernel arg 4
        //    5     C_addr_lo        kernel arg 5
        //    6     D_addr_hi        kernel arg 6
        //    7     D_addr_lo        kernel arg 7
        //    8     M                kernel arg 8
        //    9     K                kernel arg 9
        //   10     N                kernel arg 10
        //   11     array_shape_idx  kernel arg 11
        //   12     transpose_A      kernel arg 12
        //   13     transpose_B      kernel arg 13
        //   14     accumPrevC       kernel arg 14
        //   15     addNonZeroC      derived: !accumPrevC && C_addr != 0
        //   16     load_A           true if A is in L3 (DMA needed)
        //   17     load_B           true if B is in L3
        //   18     load_C           true if C is in L3 and addNonZeroC
        //   19     store_D          true if D is in L3 (DMA store back)
        //   20     local_A_addr     L1 address of A after optional load
        //   21     local_B_addr     L1 address of B after optional load
        //   22     local_C_addr     L1 address of C after optional load
        //   23     local_D_addr     L1 address of D (allocated if store_D)

        // addNonZeroC: skip C entirely when accumulating onto a previous D,
        // otherwise pull C in iff the host gave us a non-null pointer.
        const uint32_t addNonZeroC =
            (!accumPrevC && make_u64(C_addr_hi, C_addr_lo) != 0) ? 1u : 0u;
        arg_ptr[15] = addNonZeroC;

        // Bounds-check array_shape_idx and grab the shape params block.
        if (array_shape_idx >= BINGO_NUM_ARRAY_SHAPES) {
            VERSACORE_DEBUG_PRINT("ERROR: array_shape_idx=%d invalid (only %d shapes)\r\n",
                                  array_shape_idx, BINGO_NUM_ARRAY_SHAPES);
            return;
        }
        const bingo_gemm_shape_params_t *shape = &bingo_gemm_shape_params[array_shape_idx];
        const uint32_t meshRow  = shape->meshRow;
        const uint32_t tileSize = shape->tileSize;
        const uint32_t meshCol  = shape->meshCol;

        // For each operand: if its host pointer falls outside the local L1
        // range it lives in L3 and we need to DMA it in (or, for D, allocate
        // L1 and DMA-store it back at the end). Otherwise the host has
        // already staged it in L1.
        bool load_A  = !((A_addr_lo > snrt_l1_start_addr()) && (A_addr_lo < snrt_l1_end_addr()));
        bool load_B  = !((B_addr_lo > snrt_l1_start_addr()) && (B_addr_lo < snrt_l1_end_addr()));
        bool load_C  = !((C_addr_lo > snrt_l1_start_addr()) && (C_addr_lo < snrt_l1_end_addr()))
                       && (addNonZeroC == 1);
        bool store_D = !((D_addr_lo > snrt_l1_start_addr()) && (D_addr_lo < snrt_l1_end_addr()));
        arg_ptr[16] = load_A ? 1 : 0;
        arg_ptr[17] = load_B ? 1 : 0;
        arg_ptr[18] = load_C ? 1 : 0;
        arg_ptr[19] = store_D ? 1 : 0;
        VERSACORE_DEBUG_PRINT("load_A: %d, load_B: %d, load_C: %d, store_D: %d\r\n", load_A, load_B, load_C, store_D);
        if (load_A)
        {
            uint64_t src_A_addr = make_u64(A_addr_hi, A_addr_lo);
            uint32_t A_size = M * K * meshRow * tileSize * sizeof(uint8_t);
            get_cls_shared_ptrs()[1] = (uint32_t *)snrt_l1_malloc(A_size);
            if (!get_cls_shared_ptrs()[1]) {
                VERSACORE_DEBUG_PRINT("ERROR: A alloc failed!\r\n");
                return;
            }
            VERSACORE_DEBUG_PRINT("DMA Load A: src 0x%08x%08x dst 0x%lx size %d\r\n",
                                  A_addr_hi, A_addr_lo,
                                  chiplet_addr_transform((uint64_t)get_cls_shared_ptrs()[1]),
                                  A_size);
            snrt_dma_start_1d_wideptr(chiplet_addr_transform((uint64_t)get_cls_shared_ptrs()[1]),
                                      src_A_addr, A_size);
            snrt_dma_wait_all();
        }
        if (load_B)
        {
            uint64_t src_B_addr = make_u64(B_addr_hi, B_addr_lo);
            uint32_t B_size = K * N * tileSize * meshCol * sizeof(uint8_t);
            get_cls_shared_ptrs()[2] = (uint32_t *)snrt_l1_malloc(B_size);
            if (!get_cls_shared_ptrs()[2]) {
                VERSACORE_DEBUG_PRINT("ERROR: B alloc failed!\r\n");
                return;
            }
            VERSACORE_DEBUG_PRINT("DMA Load B: src 0x%08x%08x dst 0x%lx size %d\r\n",
                                  B_addr_hi, B_addr_lo,
                                  chiplet_addr_transform((uint64_t)get_cls_shared_ptrs()[2]),
                                  B_size);
            snrt_dma_start_1d_wideptr(chiplet_addr_transform((uint64_t)get_cls_shared_ptrs()[2]),
                                      src_B_addr, B_size);
            snrt_dma_wait_all();
        }
        if (load_C)
        {
            uint64_t src_C_addr = make_u64(C_addr_hi, C_addr_lo);
            uint32_t C_size = M * N * meshRow * meshCol * sizeof(uint32_t);
            get_cls_shared_ptrs()[3] = (uint32_t *)snrt_l1_malloc(C_size);
            if (!get_cls_shared_ptrs()[3]) {
                VERSACORE_DEBUG_PRINT("ERROR: C alloc failed!\r\n");
                return;
            }
            VERSACORE_DEBUG_PRINT("DMA Load C: src 0x%08x%08x dst 0x%lx size %d\r\n",
                                  C_addr_hi, C_addr_lo,
                                  chiplet_addr_transform((uint64_t)get_cls_shared_ptrs()[3]),
                                  C_size);
            snrt_dma_start_1d_wideptr(chiplet_addr_transform((uint64_t)get_cls_shared_ptrs()[3]),
                                      src_C_addr, C_size);
            snrt_dma_wait_all();
        }
        // Allocate L1 for D iff the host's D pointer is in L3 (we'll
        // DMA-store back at the end). Otherwise the host's L1 buffer is used
        // in place.
        if (store_D)
        {
            uint32_t D_size = M * N * meshRow * meshCol * sizeof(uint32_t);
            get_cls_shared_ptrs()[4] = (uint32_t *)snrt_l1_malloc(D_size);
            if (!get_cls_shared_ptrs()[4]) {
                VERSACORE_DEBUG_PRINT("ERROR: D alloc failed!\r\n");
                return;
            }
            VERSACORE_DEBUG_PRINT("Allocated D at %p\r\n",
                                  (void *)get_cls_shared_ptrs()[4]);
        }

        // Stash the local L1 addresses for each operand (slots 20..23) so
        // every core can read them after the barrier without redoing the
        // L1-vs-L3 range checks.
        get_cls_shared_ptrs()[0][20] = load_A ? (uint32_t)get_cls_shared_ptrs()[1] : A_addr_lo;
        get_cls_shared_ptrs()[0][21] = load_B ? (uint32_t)get_cls_shared_ptrs()[2] : B_addr_lo;
        get_cls_shared_ptrs()[0][22] = load_C ? (uint32_t)get_cls_shared_ptrs()[3] : C_addr_lo;
        get_cls_shared_ptrs()[0][23] = store_D ? (uint32_t)get_cls_shared_ptrs()[4] : D_addr_lo;
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    }
    snrt_cluster_hw_barrier();

    // Past the barrier every core can read the L1 scratchpad. Re-extract the
    // args we need (non-dm cores skipped the block above) and re-run the
    // bounds check + shape-table lookup. bingo_gemm_shape_params is static
    // const in .rodata so every core sees the same values without any
    // cluster-level stashing.
    uint32_t *arg_ptr = get_cls_shared_ptrs()[0];
    uint32_t M               = arg_ptr[8];
    uint32_t K               = arg_ptr[9];
    uint32_t N               = arg_ptr[10];
    uint32_t array_shape_idx = arg_ptr[11];
    uint32_t transpose_A     = arg_ptr[12];
    uint32_t transpose_B     = arg_ptr[13];
    uint32_t accumPrevC      = arg_ptr[14];
    uint32_t addNonZeroC     = arg_ptr[15];

    if (array_shape_idx >= BINGO_NUM_ARRAY_SHAPES) {
        VERSACORE_DEBUG_PRINT("ERROR: array_shape_idx=%d invalid (only %d shapes)\r\n",
                              array_shape_idx, BINGO_NUM_ARRAY_SHAPES);
        return;
    }
    const bingo_gemm_shape_params_t *shape = &bingo_gemm_shape_params[array_shape_idx];
    const uint32_t meshRow  = shape->meshRow;
    const uint32_t tileSize = shape->tileSize;
    const uint32_t meshCol  = shape->meshCol;

    uint32_t load_A       = arg_ptr[16];
    uint32_t load_B       = arg_ptr[17];
    uint32_t load_C       = arg_ptr[18];
    uint32_t store_D      = arg_ptr[19];
    uint32_t local_A_addr = arg_ptr[20];
    uint32_t local_B_addr = arg_ptr[21];
    uint32_t local_C_addr = arg_ptr[22];
    uint32_t local_D_addr = arg_ptr[23];
    // Core 0 configures and starts the versacore + streamer.
    if (snrt_cluster_core_idx() == 0)
    {
        VERSACORE_DEBUG_PRINT("GEMM Intra-Chiplet Kernel Compute Streamer Cfg Start!\r\n");
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);

        // Streamer cfg buffer in L1: 22 uint32 slots (A:6, B:6, C:5, D:5).
        get_cls_shared_ptrs()[5] = (uint32_t *)snrt_l1_malloc(sizeof(uint32_t) * 22);
        if (!get_cls_shared_ptrs()[5]) {
            VERSACORE_DEBUG_PRINT("ERROR: streamer cfg alloc failed!\r\n");
            return;
        }
        VERSACORE_DEBUG_PRINT("Allocated streamer cfg at %p\r\n",
                              (void *)get_cls_shared_ptrs()[5]);
        //////////////////////////////////////////////////////////////
        // streamer cfg for A
        //////////////////////////////////////////////////////////////

        // Aslstride0
        get_cls_shared_ptrs()[5][0] = BINGO_BANK_WIDTH / 8;

        // Atlbound0~5
        BINGO_L1_ALLOC_OR_RETURN(Atlbound, sizeof(uint32_t) * 6, "Atlbound");
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
        get_cls_shared_ptrs()[5][1] = (uint32_t)(uintptr_t)Atlbound;

        // Atlstride0~5
        BINGO_L1_ALLOC_OR_RETURN(Atlstride, sizeof(uint32_t) * 6, "Atlstride");
        // Atlstride0
        Atlstride[0] = meshRow * tileSize;
        // Atlstride1
        Atlstride[1] = 0;
        // Atlstride2
        Atlstride[2] = meshRow * tileSize * K;
        // Atlstride3
        Atlstride[3] = 0;
        // Atlstride4
        Atlstride[4] = 0;
        Atlstride[5] = 0;
        get_cls_shared_ptrs()[5][2] = (uint32_t)(uintptr_t)Atlstride;

        // set_addr_remap_index_A
        get_cls_shared_ptrs()[5][3] = 0;
        // transpose_A
        get_cls_shared_ptrs()[5][4] = transpose_A;
        // channel_en_A []
        get_cls_shared_ptrs()[5][5] = shape->channel_en_A[0];
        VERSACORE_DEBUG_PRINT(
            "GEMM Intra-Chiplet Kernel Compute Streamer Cfg A Done!\r\n");

        //////////////////////////////////////////////////////////////
        // streamer cfg for B
        //////////////////////////////////////////////////////////////

        // Bslstride0
        get_cls_shared_ptrs()[5][6] = BINGO_BANK_WIDTH / 8;

        // Btlbound0~2
        BINGO_L1_ALLOC_OR_RETURN(Btlbound, sizeof(uint32_t) * 3, "Btlbound");
        get_cls_shared_ptrs()[5][7] = (uint32_t)(uintptr_t)Btlbound;
        // Btlbound0
        Btlbound[0] = K;
        // Btlbound1
        Btlbound[1] = N;
        // Btlbound2
        Btlbound[2] = M;

        // Btlstride0~2
        BINGO_L1_ALLOC_OR_RETURN(Btlstride, sizeof(uint32_t) * 3, "Btlstride");
        get_cls_shared_ptrs()[5][8] = (uint32_t)(uintptr_t)Btlstride;

        // Btlstride0
        Btlstride[0] = tileSize * meshCol;
        // Btlstride1
        Btlstride[1] = tileSize * meshCol * K;

        // Btlstride2
        Btlstride[2] = 0;

        // set_addr_remap_index_B
        get_cls_shared_ptrs()[5][9] = 0;
        // transpose_B
        get_cls_shared_ptrs()[5][10] = transpose_B;
        // channel_en_B []
        BINGO_L1_ALLOC_OR_RETURN(channel_en_B, sizeof(uint32_t) * 2, "channel_en_B");
        channel_en_B[0] = shape->channel_en_B[0];
        channel_en_B[1] = shape->channel_en_B[1];
        get_cls_shared_ptrs()[5][11] = (uint32_t)(uintptr_t)channel_en_B;
        VERSACORE_DEBUG_PRINT(
            "GEMM Intra-Chiplet Kernel Compute Streamer Cfg B Done!\r\n");

        //////////////////////////////////////////////////////////////
        // streamer cfg for C
        //////////////////////////////////////////////////////////////
        // Cslstride0
        get_cls_shared_ptrs()[5][12] = BINGO_BANK_WIDTH / 8;
        // Ctlbound0~3
        BINGO_L1_ALLOC_OR_RETURN(Ctlbound, sizeof(uint32_t) * 4, "Ctlbound");
        get_cls_shared_ptrs()[5][13] = (uint32_t)(uintptr_t)Ctlbound;
        Ctlbound[0] = (accumPrevC == 1) ? 0 : shape->Ctlbound0;
        Ctlbound[1] = N;
        Ctlbound[2] = M;
        Ctlbound[3] = 1;
        // Ctlstride0~3
        BINGO_L1_ALLOC_OR_RETURN(Ctlstride, sizeof(uint32_t) * 4, "Ctlstride");
        get_cls_shared_ptrs()[5][14] = (uint32_t)(uintptr_t)Ctlstride;

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
        // set_addr_remap_index_C
        get_cls_shared_ptrs()[5][15] = 0;
        // channel_en_C []: zero mask when accumPrevC or !addNonZeroC, else
        // the shape's channel_en_C.
        get_cls_shared_ptrs()[5][16] = (accumPrevC == 1 || addNonZeroC == 0)
            ? bingo_channel_en_C_null[0]
            : shape->channel_en_C[0];
        VERSACORE_DEBUG_PRINT(
            "GEMM Intra-Chiplet Kernel Compute Streamer Cfg C Done!\r\n");

        //////////////////////////////////////////////////////////////
        // streamer cfg for D
        //////////////////////////////////////////////////////////////
        // D32slstride0
        get_cls_shared_ptrs()[5][17] = BINGO_BANK_WIDTH / 8;
        // D32tlbound0~3
        BINGO_L1_ALLOC_OR_RETURN(D32tlbound, sizeof(uint32_t) * 4, "D32tlbound");
        get_cls_shared_ptrs()[5][18] = (uint32_t)(uintptr_t)D32tlbound;
        D32tlbound[0] = shape->D32tlbound0;
        D32tlbound[1] = N;
        D32tlbound[2] = M;
        D32tlbound[3] = 1;
        // D32tlstride0~3
        BINGO_L1_ALLOC_OR_RETURN(D32tlstride, sizeof(uint32_t) * 4, "D32tlstride");
        get_cls_shared_ptrs()[5][19] = (uint32_t)(uintptr_t)D32tlstride;
        // D32tlstride0
        D32tlstride[0] = shape->D32tlstride0;
        // D32tlstride1
        D32tlstride[1] = BINGO_D32_ELEM_LEN * meshRow *
                         meshCol / 8;
        // D32tlstride2
        D32tlstride[2] = N * BINGO_D32_ELEM_LEN *
                         meshRow *
                         meshCol / 8;
        // D32tlstride3
        D32tlstride[3] = 0;
        // set_addr_remap_index_D32
        get_cls_shared_ptrs()[5][20] = 0;
        // channel_en_D32 []
        get_cls_shared_ptrs()[5][21] = shape->channel_en_D32[0];
        VERSACORE_DEBUG_PRINT(
            "GEMM Intra-Chiplet Kernel Compute Streamer Cfg D Done!\r\n");

        //////////////////////////////////////////////////////////////
        // Configuration the Steamer and Versacore
        //////////////////////////////////////////////////////////////

        set_versacore_streamer_csr(
            local_A_addr,                               // A_addr
            (uint32_t *)(&get_cls_shared_ptrs()[5][0]), // Aslstride
            (uint32_t *)(get_cls_shared_ptrs()[5][1]),  // Atlbound[] base
            (uint32_t *)(get_cls_shared_ptrs()[5][2]),  // Atlstride[] base
            get_cls_shared_ptrs()[5][3],                // set_addr_remap_index_A
            get_cls_shared_ptrs()[5][4],                // transpose_A
            (uint32_t *)(&get_cls_shared_ptrs()[5][5]), // channel_en_A []

            local_B_addr,                               // B_addr
            (uint32_t *)(&get_cls_shared_ptrs()[5][6]), // Bslstride[] base
            (uint32_t *)(get_cls_shared_ptrs()[5][7]),  // Btlbound[] base
            (uint32_t *)(get_cls_shared_ptrs()[5][8]),  // Btlstride[] base
            get_cls_shared_ptrs()[5][9],                // set_addr_remap_index_B
            get_cls_shared_ptrs()[5][10],               // transpose_B
            (uint32_t *)(get_cls_shared_ptrs()[5][11]), // channel_en_B []

            local_C_addr,                                // C_addr
            (uint32_t *)(&get_cls_shared_ptrs()[5][12]), // Cslstride[] base
            (uint32_t *)(get_cls_shared_ptrs()[5][13]),  // Ctlbound[] base
            (uint32_t *)(get_cls_shared_ptrs()[5][14]),  // Ctlstride[] base
            get_cls_shared_ptrs()[5][15],                // set_addr_remap_index_C
            (uint32_t *)(&get_cls_shared_ptrs()[5][16]), // channel_en_C []

            local_D_addr,                                // D_addr
            (uint32_t *)(&get_cls_shared_ptrs()[5][17]), // D32slstride[] base
            (uint32_t *)(get_cls_shared_ptrs()[5][18]),  // D32tlbound[] base
            (uint32_t *)(get_cls_shared_ptrs()[5][19]),  // D32tlstride[] base
            get_cls_shared_ptrs()[5][20],                // set_addr_remap_index_D32
            (uint32_t *)(&get_cls_shared_ptrs()[5][21]), // channel_en_D32 []
            array_shape_idx, // array shape index
            0, 0, 0, 0, 0, 0, 0, 0);

        set_versacore_csr(
            // accPrevC means takes new C
            accumPrevC == 0,
            K,
            N * M,
            0,
            array_shape_idx,
            0);

        VERSACORE_DEBUG_PRINT(
            "GEMM Intra-Chiplet Kernel Streamer Configuration Done!\r\n");

        // Set CSR to start Streamer
        start_versacore_and_streamer();
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
        // Poll until Streamer and GEMM accelerator finish
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
        wait_versacore_and_streamer();
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);

        VERSACORE_DEBUG_PRINT("GEMM Intra-Chiplet Kernel Compute Done!\r\n");

        for (int i = 0; i < 2; i++)
        {
            VERSACORE_DEBUG_PRINT("D[%d] = %d\r\n", i,
                                  *((uint32_t *)local_D_addr + i));
        }
    }

    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core())
    {
        // If D was allocated in L1 by this kernel, DMA-store it back to its
        // host-supplied L3 destination. The mesh dims used to size D come from
        // the same shared shape table the rest of the kernel reads.
        if (store_D)
        {
            uint32_t D_addr_hi = get_cls_shared_ptrs()[0][6];
            uint32_t D_addr_lo = get_cls_shared_ptrs()[0][7];
            uint64_t dst_D_addr = make_u64(D_addr_hi, D_addr_lo);
            uint32_t local_D_addr = get_cls_shared_ptrs()[0][23];
            uint32_t D_size = M * N * meshRow * meshCol * sizeof(uint32_t);

            VERSACORE_DEBUG_PRINT("DMA Store D Dst addr: 0x%08x%08x\r\n", D_addr_hi, D_addr_lo);
            VERSACORE_DEBUG_PRINT("DMA Store D Src: 0x%lx\r\n", chiplet_addr_transform((uint64_t)local_D_addr));
            VERSACORE_DEBUG_PRINT("DMA Store D Size %d\r\n", D_size);
            BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
            snrt_dma_start_1d_wideptr(dst_D_addr,
                                      chiplet_addr_transform((uint64_t)local_D_addr),
                                      D_size);
            snrt_dma_wait_all();
            BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
        }

        // Free everything this kernel allocated. Scratchpad slot first, then
        // each per-operand buffer that was alloc'd, then the streamer-cfg
        // buffer and its embedded sub-allocations.
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[0]);
        if (load_A)
        {
            snrt_l1_free((uint32_t)get_cls_shared_ptrs()[1]);
        }
        if (load_B)
        {
            snrt_l1_free((uint32_t)get_cls_shared_ptrs()[2]);
        }
        if (load_C)
        {
            snrt_l1_free((uint32_t)get_cls_shared_ptrs()[3]);
        }
        if (store_D)
        {
            snrt_l1_free((uint32_t)get_cls_shared_ptrs()[4]);
        }
        // Free Atlbound
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5][1]);
        // Free Atlstride
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5][2]);
        // Free Btlbound
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5][7]);
        // Free Btlstride
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5][8]);
        // Free channel_en_B
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5][11]);
        // Free Ctlbound
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5][13]);
        // Free Ctlstride
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5][14]);
        // Free D32tlbound
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5][18]);
        // Free D32tlstride
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5][19]);
        // Free streamer cfg
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5]);
        VERSACORE_DEBUG_PRINT("GEMM Intra-Chiplet Kernel L1 Free Done!\r\n");
    }
}

// Lightweight variant: assumes the streamer/versacore CSRs are already
// configured (e.g. by a prior __snax_kernel_versacore_load_compute_store
// call) and only re-arms the four base addresses, then starts and waits.
// Args layout: 4 uint32 base addresses in L1 (A, B, C, D).
SNAX_LIB_DEFINE void __snax_kernel_minimal_cfg_start_gemm_and_wait(void *arg)
{
    if (snrt_is_dm_core())
    {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        get_cls_shared_ptrs()[0] = (uint32_t *)snrt_l1_malloc(128);
        if (!get_cls_shared_ptrs()[0])
        {
            VERSACORE_DEBUG_PRINT("ERROR: arg alloc failed!\r\n");
            return;
        }
        VERSACORE_DEBUG_PRINT("Allocated arg at %p\r\n",
                              (void *)get_cls_shared_ptrs()[0]);

        // DMA the 4 packed uint32 args from L3 into the cluster's L1.
        snrt_dma_start_1d(
            (void *)get_cls_shared_ptrs()[0], (void *)arg,
            sizeof(uint32_t) * 4);
        snrt_dma_wait_all();
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    }

    snrt_cluster_hw_barrier();

    uint32_t *arg_ptr = get_cls_shared_ptrs()[0];
    uint32_t A_addr_lo = arg_ptr[0];
    uint32_t B_addr_lo = arg_ptr[1];
    uint32_t C_addr_lo = arg_ptr[2];
    uint32_t D_addr_lo = arg_ptr[3];

    if (snrt_cluster_core_idx() == 0)
    {
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_CFG_START);
        set_minimal_streamer_cfg(A_addr_lo, B_addr_lo, C_addr_lo, D_addr_lo);
        start_versacore_and_streamer();
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_CFG_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_RUN_START);
        wait_versacore_and_streamer();
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_RUN_END);
    }
}


