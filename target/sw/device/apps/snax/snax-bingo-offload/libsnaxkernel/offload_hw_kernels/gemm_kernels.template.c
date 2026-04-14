// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
//
// Core-level GEMM kernel template (bingo-hw flow). Contains the two
// single-core GEMM kernels the bingo HW scheduler invokes:
//   - __snax_bingo_kernel_gemm_full       (configures streamer+versacore and runs)
//   - __snax_bingo_kernel_gemm_minimal    (reuses prior config, just starts/waits)
// Both return uint32_t (BINGO_RET_SUCC / BINGO_RET_FAIL) and dispatch per
// array_shape_idx via `switch` blocks, each of which has a `default:` arm
// returning BINGO_RET_FAIL.
//
// READ by libsnaxkernel/gen_bingo_gemm_kernels.py which substitutes every
// HW-parameter macro (meshRow_N, channel_en_A_N_M, bankWidth, …) with the
// literal value computed from the hwcfg and strips `case N:` arms for
// shapes the hwcfg does not define. The emitted file is
// offload_hw_kernels/gemm.h (gitignored).
//
// Edit THIS file when you want to change core-level GEMM kernel logic.
// Do not edit the generated offload_hw_kernels/gemm.h — any changes there
// will be overwritten on the next `make sw`.

// =============================================================
// Core-level GEMM kernels (bingo-hw)
// =============================================================

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_gemm_full(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, 14);
    // Assume the matrix data are all in L1
    // So all the addr are 32bit local addr
    // This kernel will configure the versacore and streamer and start the computation
    // There is another __snax_bingo_kernel_gemm_minimal kernel that only starts the versacore and streamer with pre-configured CSRs
    if (snrt_cluster_core_idx() != 0){
        printf_safe("[Cluster %d Core %d]: Error! Bingo GEMM full should be called from core 0!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t A_addr = ((uint32_t *)arg)[0];
    uint32_t B_addr = ((uint32_t *)arg)[1];
    uint32_t C_addr = ((uint32_t *)arg)[2];
    uint32_t D_addr = ((uint32_t *)arg)[3];
    VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Bingo GEMM Full called with A_addr=0x%08x, B_addr=0x%08x, C_addr=0x%08x, D_addr=0x%08x\r\n",
                          snrt_cluster_idx(), snrt_cluster_core_idx(),
                          A_addr, B_addr, C_addr, D_addr);
    uint32_t M = ((uint32_t *)arg)[4];
    uint32_t K = ((uint32_t *)arg)[5];
    uint32_t N = ((uint32_t *)arg)[6];
    uint32_t array_shape_idx = ((uint32_t *)arg)[7];
    uint32_t transpose_A = ((uint32_t *)arg)[8];
    uint32_t transpose_B = ((uint32_t *)arg)[9];
    uint32_t accumPrevC = ((uint32_t *)arg)[10];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, 14);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    // some inferenced args
    uint32_t addNonZeroC;
    if (accumPrevC)
    {
        addNonZeroC = 0;
    }
    else if (C_addr!= 0)
    {
        addNonZeroC = 1;
    }
    else
    {
        addNonZeroC = 0;
    }
    uint32_t meshRow;
    uint32_t tileSize;
    uint32_t meshCol;
    switch (array_shape_idx)
    {
    case 0:
        meshRow = meshRow_0;
        tileSize = tileSize_0;
        meshCol = meshCol_0;
        break;
    case 1:
        meshRow = meshRow_1;
        tileSize = tileSize_1;
        meshCol = meshCol_1;
        break;
    case 2:
        meshRow = meshRow_2;
        tileSize = tileSize_2;
        meshCol = meshCol_2;
        break;
    case 3:
        meshRow = meshRow_3;
        tileSize = tileSize_3;
        meshCol = meshCol_3;
        break;
    case 4:
        meshRow = meshRow_4;
        tileSize = tileSize_4;
        meshCol = meshCol_4;
        break;
    default:
        VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Error! array_shape_idx invalid!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    // Configuration the Steamer and Versacore
    //////////////////////////////////////////////////////////////
    // Streamer cfg for A
    //////////////////////////////////////////////////////////////
    
    // Aslstride0
    uint32_t Aslstride0 = bankWidth / 8;
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
    Atlstride[0] = meshRow * tileSize;
    // Atlstride1
    Atlstride[1] = 0;
    // Atlstride2
    Atlstride[2] = meshRow * tileSize * K;
    // Atlstride3
    Atlstride[3] = 0;
    // Atlstride4
    Atlstride[4] = 0;
    // Atlstride5
    Atlstride[5] = 0;
    uint32_t set_addr_remap_index_A = 0;
    uint32_t channel_en_A;
    switch (array_shape_idx)
    {
    case 0:
        channel_en_A = channel_en_A_0_0;
        break;
    case 1:
        channel_en_A = channel_en_A_1_0;
        break;
    case 2:
        channel_en_A = channel_en_A_2_0;
        break;
    case 3:
        channel_en_A = channel_en_A_3_0;
        break;
    case 4:
        channel_en_A = channel_en_A_4_0;
        break;
    default:
        VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Error! array_shape_idx invalid!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    //////////////////////////////////////////////////////////////
    // Streamer cfg for B
    //////////////////////////////////////////////////////////////
    // Bslstride0
    uint32_t Bslstride0 = bankWidth / 8;
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
    Btlstride[0] = tileSize * meshCol;
    // Btlstride1
    Btlstride[1] = tileSize * meshCol * K;
    // Btlstride2
    Btlstride[2] = 0;
    uint32_t set_addr_remap_index_B = 0;
    uint32_t channel_en_B[2];
    switch (array_shape_idx)
    {
    case 0:
        channel_en_B[0] = channel_en_B_0_0;
        channel_en_B[1] = channel_en_B_0_1;
        break;
    case 1:
        channel_en_B[0] = channel_en_B_1_0;
        channel_en_B[1] = channel_en_B_1_1;
        break;
    case 2:
        channel_en_B[0] = channel_en_B_2_0;
        channel_en_B[1] = channel_en_B_2_1;
        break;
    case 3:
        channel_en_B[0] = channel_en_B_3_0;
        channel_en_B[1] = channel_en_B_3_1;
        break;
    case 4:
        channel_en_B[0] = channel_en_B_4_0;
        channel_en_B[1] = channel_en_B_4_1;
        break;
    default:
        VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Error! array_shape_idx invalid!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    //////////////////////////////////////////////////////////////
    // Streamer cfg for C
    //////////////////////////////////////////////////////////////
    // Cslstride0
    uint32_t Cslstride0 = bankWidth / 8;
    // Ctlbound0~3
    uint32_t Ctlbound[4];
    if (accumPrevC == 1)
    {
        // accumPrevC is true
        Ctlbound[0] = 0;
    }
    else
    {
        switch (array_shape_idx)
        {
        case 0:
            Ctlbound[0] = Ctlbound0_0;
            break;
        case 1:
            Ctlbound[0] = Ctlbound0_1;
            break;
        case 2:
            Ctlbound[0] = Ctlbound0_2;
            break;
        case 3:
            Ctlbound[0] = Ctlbound0_3;
            break;
        case 4:
            Ctlbound[0] = Ctlbound0_4;
            break;
        default:
            VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Error! array_shape_idx invalid!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }
    }
    // Ctlbound1
    Ctlbound[1] = N;
    // Ctlbound2
    Ctlbound[2] = M;
    // Ctlbound3
    Ctlbound[3] = 1;
    uint32_t Ctlstride[4];
    switch (array_shape_idx)
    {
    case 0:
        // Ctlstride0
        Ctlstride[0] = Ctlstride0_0;
        break;
    case 1:
        // Ctlstride0
        Ctlstride[0] = Ctlstride0_1;
        break;
    case 2:
        // Ctlstride0
        Ctlstride[0] = Ctlstride0_2;
        break;
    case 3:
        // Ctlstride0
        Ctlstride[0] = Ctlstride0_3;
        break;
    case 4:
        // Ctlstride0
        Ctlstride[0] = Ctlstride0_4;
        break;
    default:
        VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Error! array_shape_idx invalid!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    // Ctlstride1
    Ctlstride[1] = C_elem_len * meshRow *
                   meshCol / 8;
    // Ctlstride2
    Ctlstride[2] = N * C_elem_len *
                   meshRow *
                   meshCol / 8;
    // Ctlstride3
    Ctlstride[3] = 0;
    uint32_t set_addr_remap_index_C = 0;
    uint32_t channel_en_C;
    // set channel_en_C to zero if accumPrevC or addNonZeroC
    if (accumPrevC == 1 || addNonZeroC == 0)
    {
        channel_en_C = channel_en_C_null_0_0;;
    }
    else
    {
        switch (array_shape_idx)
        {
        case 0:
            channel_en_C = channel_en_C_0_0;
            break;
        case 1:
            channel_en_C = channel_en_C_1_0;
            break;
        case 2:
            channel_en_C = channel_en_C_2_0;
            break;
        case 3:
            channel_en_C = channel_en_C_3_0;
            break;
        case 4:
            channel_en_C = channel_en_C_4_0;
            break;
        default:
            VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Error! array_shape_idx invalid!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }
    }
    //////////////////////////////////////////////////////////////
    // Streamer cfg for D
    //////////////////////////////////////////////////////////////
    // D32slstride0
    uint32_t D32slstride0 = bankWidth / 8;
    // D32tlbound0~3
    uint32_t D32tlbound[4];
    switch (array_shape_idx)
    {
    case 0:
        D32tlbound[0] = D32tlbound0_0;
        break;
    case 1:
        D32tlbound[0] = D32tlbound0_1;
        break;
    case 2:
        D32tlbound[0] = D32tlbound0_2;
        break;
    case 3:
        D32tlbound[0] = D32tlbound0_3;
        break;
    case 4:
        D32tlbound[0] = D32tlbound0_4;
        break;
    default:
        VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Error! array_shape_idx invalid!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    // D32tlbound1
    D32tlbound[1] = N;
    // D32tlbound2
    D32tlbound[2] = M;
    // D32tlbound3
    D32tlbound[3] = 1;
    uint32_t D32tlstride[4];
    switch (array_shape_idx)
    {
    case 0:
        // D32tlstride0
        D32tlstride[0] = D32tlstride0_0;
        break;
    case 1:
        // D32tlstride0
        D32tlstride[0] = D32tlstride0_1;
        break;
    case 2:
        // D32tlstride0
        D32tlstride[0] = D32tlstride0_2;
        break;
    case 3:
        // D32tlstride0
        D32tlstride[0] = D32tlstride0_3;
        break;
    case 4:
        // D32tlstride0
        D32tlstride[0] = D32tlstride0_4;
        break;
    default:
        VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Error! array_shape_idx invalid!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    // D32tlstride1
    D32tlstride[1] = D32_elem_len * meshRow *
                     meshCol / 8;
    // D32tlstride2
    D32tlstride[2] = N * D32_elem_len *
                     meshRow *
                     meshCol / 8;
    // D32tlstride3
    D32tlstride[3] = 0;
    uint32_t set_addr_remap_index_D32 = 0;
    uint32_t channel_en_D32;
    switch (array_shape_idx)
    {
    case 0:
        channel_en_D32 = channel_en_D32_0_0;
        break;
    case 1:
        channel_en_D32 = channel_en_D32_1_0;
        break;
    case 2:
        channel_en_D32 = channel_en_D32_2_0;
        break;
    case 3:
        channel_en_D32 = channel_en_D32_3_0;
        break;
    case 4:
        channel_en_D32 = channel_en_D32_4_0;
        break;
    default:
        VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Error! array_shape_idx invalid!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    //////////////////////////////////////////////////////////////
    // Configuration the Steamer and Versacore
    //////////////////////////////////////////////////////////////
    VERSACORE_DEBUG_PRINT(
        "Bingo GEMM Full Kernel Compute Streamer Cfg Start!\r\n");
    set_versacore_streamer_csr(
        A_addr,               // A_addr
        &Aslstride0,         // Aslstride[] base
        Atlbound,            // Atlbound[] base
        Atlstride,           // Atlstride[] base
        set_addr_remap_index_A, // set_addr_remap_index_A
        transpose_A,         // transpose_A
        &channel_en_A,       // channel_en_A []
        B_addr,              // B_addr
        &Bslstride0,         // Bslstride[] base
        Btlbound,            // Btlbound[] base
        Btlstride,           // Btlstride[] base
        set_addr_remap_index_B, // set_addr_remap_index_B
        transpose_B,         // transpose_B
        channel_en_B,        // channel_en_B []
        C_addr,               // C_addr
        &Cslstride0,         // Cslstride[] base
        Ctlbound,            // Ctlbound[] base
        Ctlstride,           // Ctlstride[] base
        set_addr_remap_index_C, // set_addr_remap_index_C
        &channel_en_C,       // channel_en_C []
        D_addr,               // D_addr
        &D32slstride0,         // D32slstride[] base
        D32tlbound,          // D32tlbound[] base
        D32tlstride,         // D32tlstride[] base
        set_addr_remap_index_D32, // set_addr_remap_index_D32
        &channel_en_D32,     // channel_en_D32 []
        0, 0, 0, 0, 0, 0, 0, 0, 0);
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
    BINGO_SW_GUARD_CHECK(arg, 7);
    // This kernel will only start the versacore and streamer with pre-configured CSRs
    if (snrt_cluster_core_idx() != 0){
        printf_safe("[Cluster %d Core %d]: Error! Bingo GEMM minimal should be called from core 0!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t A_addr = ((uint32_t *)arg)[0];
    uint32_t B_addr = ((uint32_t *)arg)[1];
    uint32_t C_addr = ((uint32_t *)arg)[2];
    uint32_t D_addr = ((uint32_t *)arg)[3];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, 7);
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
