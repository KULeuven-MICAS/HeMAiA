// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
//
// Cluster-level GEMM kernel template (bingo-sw flow). Contains the two
// cluster-coordinated GEMM kernels:
//   - __snax_kernel_gemm                       (full load/compute/store)
//   - __snax_kernel_minimal_cfg_start_gemm_and_wait
// Both return `void` and dispatch per array_shape_idx via
// `switch (array_shape_idx)` blocks so the generator can trim unused cases.
//
// READ by libsnaxkernel/gen_bingo_gemm_kernels.py which substitutes every
// HW-parameter macro (meshRow_N, channel_en_A_N_M, bankWidth, …) with the
// literal value computed from the hwcfg and strips `case N:` arms for
// shapes the hwcfg does not define.
// emitted file is offload_sw_kernels/gemm.h (gitignored).
//
// Edit THIS file when you want to change cluster-level GEMM kernel logic.
// Do not edit the generated offload_sw_kernels/gemm.h — any changes there
// will be overwritten on the next `make sw`.

SNAX_LIB_DEFINE void __snax_kernel_gemm(void *arg)
{
    // Compute GeMM D = A*B + C
    // This kernel will check the A,B,C matrix address to decide whether to load inputs or
    // the inputs are already in the local TCDM
    // Usage 1: All the matrix are not in L1, so the kernel will load all inputs using DMA
    // Usage 2: Some matrix are already in L1, so the kernel will skip loading those matrix and assume they are in the local TCDM already

    // move the args to local variables
    if (snrt_is_dm_core())
    {
        // allocate L1 memory for global variables across the cluster
        ///////////////////////////////////////
        // Allocate the L1 memory //
        ///////////////////////////////////////
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        get_cls_shared_ptrs()[0] = (uint32_t *)snrt_l1_malloc(128);
        if (!get_cls_shared_ptrs()[0])
        {
            VERSACORE_DEBUG_PRINT("ERROR: arg alloc failed!\r\n");
            return;
        }
        else
        {
            VERSACORE_DEBUG_PRINT("Allocated arg at %p\r\n",
                                  (void *)get_cls_shared_ptrs()[0]);
        }

        ////////////////////////////////////
        // Call the idma to load the args //
        ////////////////////////////////////
        // First we need to use the DMA to load all the arguments from L3 into
        // L1 Here we assume the arguments are packed in a contiguous way in L3
        VERSACORE_DEBUG_PRINT("GEMM Intra-Chiplet Kernel Start!\r\n");

        snrt_dma_start_1d(
            (void *)get_cls_shared_ptrs()[0], (void *)arg,
            sizeof(uint32_t) * 15); // 15 uint32_t passed args in total
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

        // we use the following args, they are shared across the cluster
        // arg0: uint32_t A_addr_hi get_cls_shared_ptrs()[0][0]
        // arg1: uint32_t A_addr_lo get_cls_shared_ptrs()[0][1]
        // arg2: uint32_t B_addr_hi get_cls_shared_ptrs()[0][2]
        // arg3: uint32_t B_addr_lo get_cls_shared_ptrs()[0][3]
        // arg4: uint32_t C_addr_hi get_cls_shared_ptrs()[0][4]
        // arg5: uint32_t C_addr_lo get_cls_shared_ptrs()[0][5]
        // arg6: uint32_t D_addr_hi get_cls_shared_ptrs()[0][6]
        // arg7: uint32_t D_addr_lo get_cls_shared_ptrs()[0][7]
        // these address are all in L1! It's the TCDM local address.

        // arg8: uint32_t M get_cls_shared_ptrs()[0][8]
        // arg9: uint32_t K get_cls_shared_ptrs()[0][9]
        // arg10: uint32_t N get_cls_shared_ptrs()[0][10]
        // arg11: uint32_t array_shape_idx get_cls_shared_ptrs()[0][11]
        // arg12: transpose A get_cls_shared_ptrs()[0][12]
        // arg13: transpose B get_cls_shared_ptrs()[0][13]
        // arg14: get_cls_shared_ptrs()[0][14] is used for accumPrevC

        // get_cls_shared_ptrs()[0][15] is used for addNonZeroC
        // get_cls_shared_ptrs()[0][16] is used for meshRow
        // get_cls_shared_ptrs()[0][17] is used for tileSize
        // get_cls_shared_ptrs()[0][18] is used for meshCol
        // get_cls_shared_ptrs()[0][19] is used for load_A
        // get_cls_shared_ptrs()[0][20] is used for load_B
        // get_cls_shared_ptrs()[0][21] is used for load_C
        // get_cls_shared_ptrs()[0][22] is used for store_D
        // get_cls_shared_ptrs()[0][23] is used for local_A_addr
        // get_cls_shared_ptrs()[0][24] is used for local_B_addr
        // get_cls_shared_ptrs()[0][25] is used for local_C_addr
        // get_cls_shared_ptrs()[0][26] is used for local_D_addr

        if (accumPrevC)
        {
            arg_ptr[15] = 0;
        }
        else if (make_u64(C_addr_hi, C_addr_lo) != 0)
        {
            arg_ptr[15] = 1;
        }
        else
        {
            arg_ptr[15] = 0;
        }

        // get_cls_shared_ptrs()[0][16] is used for meshRow
        // get_cls_shared_ptrs()[0][17] is used for tileSize
        // get_cls_shared_ptrs()[0][18] is used for meshCol
        switch (array_shape_idx)
        {
        case 0:
            arg_ptr[16] = meshRow_0;
            arg_ptr[17] = tileSize_0;
            arg_ptr[18] = meshCol_0;
            break;
        case 1:
            arg_ptr[16] = meshRow_1;
            arg_ptr[17] = tileSize_1;
            arg_ptr[18] = meshCol_1;
            break;
        case 2:
            arg_ptr[16] = meshRow_2;
            arg_ptr[17] = tileSize_2;
            arg_ptr[18] = meshCol_2;
            break;
        case 3:
            arg_ptr[16] = meshRow_3;
            arg_ptr[17] = tileSize_3;
            arg_ptr[18] = meshCol_3;
            break;
        case 4:
            arg_ptr[16] = meshRow_4;
            arg_ptr[17] = tileSize_4;
            arg_ptr[18] = meshCol_4;
            break;
        default:
            VERSACORE_DEBUG_PRINT("ERROR: array_shape_idx invalid!\r\n");
            return;
        }
        uint32_t addNonZeroC = arg_ptr[15];
        uint32_t meshRow = arg_ptr[16];
        uint32_t tileSize = arg_ptr[17];
        uint32_t meshCol = arg_ptr[18];

        // We need to determine whether we need to use the dma to load the data from the other cores or
        // the data has already been loaded into L1 by the other cores
        // If the addr is in L1 range, we assume the data is already in L1
        // Otherwise we need to use the DMA to load the data from the specified addr
        bool load_A = !((A_addr_lo > snrt_l1_start_addr()) && (A_addr_lo < snrt_l1_end_addr()));
        bool load_B = !((B_addr_lo > snrt_l1_start_addr()) && (B_addr_lo < snrt_l1_end_addr()));
        bool load_C = (!((C_addr_lo > snrt_l1_start_addr()) && (C_addr_lo < snrt_l1_end_addr()))) && (addNonZeroC == 1);
        // If D addr is in L3, we need to allocate the L1 memory for D and then store it back using DMA
        // If D addr is in L1, it means the host is allocated the L1 already, we do not need L1 alloc and the DMA
        bool store_D = !((D_addr_lo > snrt_l1_start_addr()) && (D_addr_lo < snrt_l1_end_addr()));
        arg_ptr[19] = load_A ? 1 : 0;
        arg_ptr[20] = load_B ? 1 : 0;
        arg_ptr[21] = load_C ? 1 : 0;
        arg_ptr[22] = store_D ? 1 : 0;
        VERSACORE_DEBUG_PRINT("load_A: %d, load_B: %d, load_C: %d, store_D: %d\r\n", load_A, load_B, load_C, store_D);
        if (load_A)
        {
            // Load A to the local L1
            uint64_t src_A_addr = make_u64(A_addr_hi, A_addr_lo);
            // Size A = M * K * meshRow * tileSize * sizeof(uint8_t)
            uint32_t A_size = M * K * meshRow * tileSize * sizeof(uint8_t);
            // Allocate L1 memory for A
            get_cls_shared_ptrs()[1] = (uint32_t *)snrt_l1_malloc(A_size);
            if (!get_cls_shared_ptrs()[1])
            {
                VERSACORE_DEBUG_PRINT("ERROR: A alloc failed!\r\n");
                return;
            }
            else
            {
                VERSACORE_DEBUG_PRINT("Allocated A at %p\r\n",
                                      (void *)get_cls_shared_ptrs()[1]);
            }
            VERSACORE_DEBUG_PRINT("DMA Load A Src  0x%08x%08x\r\n", A_addr_hi, A_addr_lo);
            VERSACORE_DEBUG_PRINT("DMA Load A Dst  0x%lx\r\n", chiplet_addr_transform((uint64_t)get_cls_shared_ptrs()[1]));
            VERSACORE_DEBUG_PRINT("DMA Load A Size %d\r\n", A_size);
            // Dst is L1 addr, SRC is specified from the args
            snrt_dma_start_1d_wideptr(chiplet_addr_transform((uint64_t)get_cls_shared_ptrs()[1]),
                                      src_A_addr,
                                      A_size);
            snrt_dma_wait_all();
        }
        if (load_B)
        {
            // Load B to the local L1
            uint64_t src_B_addr = make_u64(B_addr_hi, B_addr_lo);
            // Size B = K * N * tileSize * meshCol * sizeof(uint8_t)
            uint32_t B_size = K * N * tileSize * meshCol * sizeof(uint8_t);
            // Allocate L1 memory for B
            get_cls_shared_ptrs()[2] = (uint32_t *)snrt_l1_malloc(B_size);
            if (!get_cls_shared_ptrs()[2])
            {
                VERSACORE_DEBUG_PRINT("ERROR: B alloc failed!\r\n");
                return;
            }
            else
            {
                VERSACORE_DEBUG_PRINT("Allocated B at %p\r\n",
                                      (void *)get_cls_shared_ptrs()[2]);
            }
            // Dst is L1 addr, SRC is specified from the args
            VERSACORE_DEBUG_PRINT("DMA Load B Src  0x%08x%08x\r\n", B_addr_hi, B_addr_lo);
            VERSACORE_DEBUG_PRINT("DMA Load B Dst  0x%lx\r\n", chiplet_addr_transform((uint64_t)get_cls_shared_ptrs()[2]));
            VERSACORE_DEBUG_PRINT("DMA Load B Size %d\r\n", B_size);
            snrt_dma_start_1d_wideptr(chiplet_addr_transform((uint64_t)get_cls_shared_ptrs()[2]),
                                      src_B_addr,
                                      B_size);
            snrt_dma_wait_all();
        }
        if (load_C)
        {
            // Load C to the local L1
            uint64_t src_C_addr = make_u64(C_addr_hi, C_addr_lo);
            // Size C = M * N * meshRow * meshCol * sizeof(uint32_t)
            uint32_t C_size = M * N * meshRow * meshCol * sizeof(uint32_t);
            // Allocate L1 memory for C
            get_cls_shared_ptrs()[3] = (uint32_t *)snrt_l1_malloc(C_size);
            if (!get_cls_shared_ptrs()[3])
            {
                VERSACORE_DEBUG_PRINT("ERROR: C alloc failed!\r\n");
                return;
            }
            else
            {
                VERSACORE_DEBUG_PRINT("Allocated C at %p\r\n",
                                      (void *)get_cls_shared_ptrs()[3]);
            }
            VERSACORE_DEBUG_PRINT("DMA Load C Src  0x%08x%08x\r\n", C_addr_hi, C_addr_lo);
            VERSACORE_DEBUG_PRINT("DMA Load C Dst  0x%lx\r\n", chiplet_addr_transform((uint64_t)get_cls_shared_ptrs()[3]));
            VERSACORE_DEBUG_PRINT("DMA Load C Size %d\r\n", C_size);
            // Dst is L1 addr, SRC is specified from the args
            snrt_dma_start_1d_wideptr(chiplet_addr_transform((uint64_t)get_cls_shared_ptrs()[3]),
                                      src_C_addr,
                                      C_size);
            snrt_dma_wait_all();
        }
        // Allocate the L1 memory for output D
        // If the D addr is not in L1, we need to allocate the L1 memory for D and then use the DMA to store according to the inputs
        // Otherwise the D addr is directly used
        if (store_D)
        {
            // Allocate L1 memory for D
            uint32_t D_size = M * N * meshRow * meshCol * sizeof(uint32_t);
            get_cls_shared_ptrs()[4] = (uint32_t *)snrt_l1_malloc(D_size);
            if (!get_cls_shared_ptrs()[4])
            {
                VERSACORE_DEBUG_PRINT("ERROR: D alloc failed!\r\n");
                return;
            }
            else
            {
                VERSACORE_DEBUG_PRINT("Allocated D at %p\r\n",
                                      (void *)get_cls_shared_ptrs()[4]);
            }
        }

        // Update the local A,B,C,D addr pointers in the shared ptrs
        get_cls_shared_ptrs()[0][23] = load_A ? (uint32_t)get_cls_shared_ptrs()[1] : A_addr_lo;
        get_cls_shared_ptrs()[0][24] = load_B ? (uint32_t)get_cls_shared_ptrs()[2] : B_addr_lo;
        get_cls_shared_ptrs()[0][25] = load_C ? (uint32_t)get_cls_shared_ptrs()[3] : C_addr_lo;
        get_cls_shared_ptrs()[0][26] = store_D ? (uint32_t)get_cls_shared_ptrs()[4] : D_addr_lo;
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    }
    snrt_cluster_hw_barrier();

    // Here all the cores can access the args from L1
    // Specifically the compute core will configure the streamer and versacore accoring to the following shared ptrs
    uint32_t *arg_ptr = get_cls_shared_ptrs()[0];
    uint32_t M = arg_ptr[8];
    uint32_t K = arg_ptr[9];
    uint32_t N = arg_ptr[10];
    uint32_t array_shape_idx = arg_ptr[11];
    uint32_t transpose_A = arg_ptr[12];
    uint32_t transpose_B = arg_ptr[13];
    uint32_t accumPrevC = arg_ptr[14];

    // some inferenced args
    uint32_t addNonZeroC = arg_ptr[15];

    uint32_t meshRow = arg_ptr[16];
    uint32_t tileSize = arg_ptr[17];
    uint32_t meshCol = arg_ptr[18];

    uint32_t load_A = arg_ptr[19];
    uint32_t load_B = arg_ptr[20];
    uint32_t load_C = arg_ptr[21];
    uint32_t store_D = arg_ptr[22];
    uint32_t local_A_addr = arg_ptr[23];
    uint32_t local_B_addr = arg_ptr[24];
    uint32_t local_C_addr = arg_ptr[25];
    uint32_t local_D_addr = arg_ptr[26];
    // The core 0 will be responsible for configuring and starting the versacore
    if (snrt_cluster_core_idx() == 0)
    {
        // compute the streamer cfg first
        //////////////////////////////////////////////////////////////
        // Allocated the L1 memory for Streamer CSRs    //
        //////////////////////////////////////////////////////////////

        VERSACORE_DEBUG_PRINT(
            "GEMM Intra-Chiplet Kernel Compute Streamer Cfg Start!\r\n");
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
        // the snicth needs to generate these streamer cfgs
        // 22 args in total
        // A 6, B 6, C 5, D 5
        get_cls_shared_ptrs()[5] = (uint32_t *)snrt_l1_malloc(sizeof(uint32_t) * 22);
        if (!get_cls_shared_ptrs()[5])
        {
            VERSACORE_DEBUG_PRINT("ERROR: streamer cfg alloc failed!\r\n");
            return;
        }
        else
        {
            VERSACORE_DEBUG_PRINT("Allocated streamer cfg at %p\r\n",
                                  (void *)get_cls_shared_ptrs()[5]);
        }
        //////////////////////////////////////////////////////////////
        // streamer cfg for A
        //////////////////////////////////////////////////////////////

        // Aslstride0
        get_cls_shared_ptrs()[5][0] = bankWidth / 8;

        // Atlbound0~5
        uint32_t *Atlbound = (uint32_t *)snrt_l1_malloc(sizeof(uint32_t) * 6);
        if (!Atlbound)
        {
            VERSACORE_DEBUG_PRINT("ERROR: Atlbound alloc failed!\r\n");
            return;
        }
        else
        {
            VERSACORE_DEBUG_PRINT("Allocated Atlbound at %p\r\n",
                                  (void *)Atlbound);
        }
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
        uint32_t *Atlstride = (uint32_t *)snrt_l1_malloc(sizeof(uint32_t) * 6);
        if (!Atlstride)
        {
            VERSACORE_DEBUG_PRINT("ERROR: Atlstride alloc failed!\r\n");
            return;
        }
        else
        {
            VERSACORE_DEBUG_PRINT("Allocated Atlstride at %p\r\n",
                                  (void *)Atlstride);
        }
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
        switch (array_shape_idx)
        {
        case 0:
            get_cls_shared_ptrs()[5][5] = channel_en_A_0_0;
            break;
        case 1:
            get_cls_shared_ptrs()[5][5] = channel_en_A_1_0;
            break;
        case 2:
            get_cls_shared_ptrs()[5][5] = channel_en_A_2_0;
            break;
        case 3:
            get_cls_shared_ptrs()[5][5] = channel_en_A_3_0;
            break;
        case 4:
            get_cls_shared_ptrs()[5][5] = channel_en_A_4_0;
            break;
        default:
            break;
        }
        VERSACORE_DEBUG_PRINT(
            "GEMM Intra-Chiplet Kernel Compute Streamer Cfg A Done!\r\n");

        //////////////////////////////////////////////////////////////
        // streamer cfg for B
        //////////////////////////////////////////////////////////////

        // Bslstride0
        get_cls_shared_ptrs()[5][6] = bankWidth / 8;

        // Btlbound0~2
        uint32_t *Btlbound = (uint32_t *)snrt_l1_malloc(sizeof(uint32_t) * 3);
        if (!Btlbound)
        {
            VERSACORE_DEBUG_PRINT("ERROR: Btlbound alloc failed!\r\n");
            return;
        }
        else
        {
            VERSACORE_DEBUG_PRINT("Allocated Btlbound at %p\r\n",
                                  (void *)Btlbound);
        }
        get_cls_shared_ptrs()[5][7] = (uint32_t)(uintptr_t)Btlbound;
        // Btlbound0
        Btlbound[0] = K;
        // Btlbound1
        Btlbound[1] = N;
        // Btlbound2
        Btlbound[2] = M;

        // Btlstride0~2
        uint32_t *Btlstride = (uint32_t *)snrt_l1_malloc(sizeof(uint32_t) * 3);
        if (!Btlstride)
        {
            VERSACORE_DEBUG_PRINT("ERROR: Btlstride alloc failed!\r\n");
            return;
        }
        else
        {
            VERSACORE_DEBUG_PRINT("Allocated Btlstride at %p\r\n",
                                  (void *)Btlstride);
        }
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
        uint32_t *channel_en_B = (uint32_t *)snrt_l1_malloc(sizeof(uint32_t) * 2);
        if (!channel_en_B)
        {
            VERSACORE_DEBUG_PRINT("ERROR: channel_en_B alloc failed!\r\n");
            return;
        }
        else
        {
            VERSACORE_DEBUG_PRINT("Allocated channel_en_B at %p\r\n",
                                  (void *)channel_en_B);
        }
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
            break;
        }
        get_cls_shared_ptrs()[5][11] = (uint32_t)(uintptr_t)channel_en_B;
        VERSACORE_DEBUG_PRINT(
            "GEMM Intra-Chiplet Kernel Compute Streamer Cfg B Done!\r\n");

        //////////////////////////////////////////////////////////////
        // streamer cfg for C
        //////////////////////////////////////////////////////////////
        // Cslstride0
        get_cls_shared_ptrs()[5][12] = bankWidth / 8;
        // Ctlbound0~3
        uint32_t *Ctlbound = (uint32_t *)snrt_l1_malloc(sizeof(uint32_t) * 4);
        if (!Ctlbound)
        {
            VERSACORE_DEBUG_PRINT("ERROR: Ctlbound alloc failed!\r\n");
            return;
        }
        else
        {
            VERSACORE_DEBUG_PRINT("Allocated Ctlbound at %p\r\n",
                                  (void *)Ctlbound);
        }
        get_cls_shared_ptrs()[5][13] = (uint32_t)(uintptr_t)Ctlbound;
        if (accumPrevC == 1)
        {
            // accumPrevC is true
            Ctlbound[0] = 0;
        }
        else
        { // else bound is normal
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
                break;
            }
        }
        Ctlbound[1] = N;
        Ctlbound[2] = M;
        Ctlbound[3] = 1;
        // Ctlstride0~3
        uint32_t *Ctlstride = (uint32_t *)snrt_l1_malloc(sizeof(uint32_t) * 4);
        if (!Ctlstride)
        {
            VERSACORE_DEBUG_PRINT("ERROR: Ctlstride alloc failed!\r\n");
            return;
        }
        else
        {
            VERSACORE_DEBUG_PRINT("Allocated Ctlstride at %p\r\n",
                                  (void *)Ctlstride);
        }
        get_cls_shared_ptrs()[5][14] = (uint32_t)(uintptr_t)Ctlstride;

        // Ctlstride0
        switch (array_shape_idx)
        {
        case 0:
            Ctlstride[0] = Ctlstride0_0;
            break;
        case 1:
            Ctlstride[0] = Ctlstride0_1;
            break;
        case 2:
            Ctlstride[0] = Ctlstride0_2;
            break;
        case 3:
            Ctlstride[0] = Ctlstride0_3;
            break;
        case 4:
            Ctlstride[0] = Ctlstride0_4;
            break;
        default:
            break;
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
        // set_addr_remap_index_C
        get_cls_shared_ptrs()[5][15] = 0;
        // channel_en_C []
        // set channel_en_C to zero if accumPrevC or addNonZeroC
        if (accumPrevC == 1 || addNonZeroC == 0)
        {
            get_cls_shared_ptrs()[5][16] = channel_en_C_null_0_0;
        }
        else
        {
            switch (array_shape_idx)
            {
            case 0:
                get_cls_shared_ptrs()[5][16] = channel_en_C_0_0;
                break;
            case 1:
                get_cls_shared_ptrs()[5][16] = channel_en_C_1_0;
                break;
            case 2:
                get_cls_shared_ptrs()[5][16] = channel_en_C_2_0;
                break;
            case 3:
                get_cls_shared_ptrs()[5][16] = channel_en_C_3_0;
                break;
            case 4:
                get_cls_shared_ptrs()[5][16] = channel_en_C_4_0;
                break;
            default:
                break;
            }
        }
        VERSACORE_DEBUG_PRINT(
            "GEMM Intra-Chiplet Kernel Compute Streamer Cfg C Done!\r\n");

        //////////////////////////////////////////////////////////////
        // streamer cfg for D
        //////////////////////////////////////////////////////////////
        // D32slstride0
        get_cls_shared_ptrs()[5][17] = bankWidth / 8;
        // D32tlbound0~3
        uint32_t *D32tlbound = (uint32_t *)snrt_l1_malloc(sizeof(uint32_t) * 4);
        if (!D32tlbound)
        {
            VERSACORE_DEBUG_PRINT("ERROR: D32tlbound alloc failed!\r\n");
            return;
        }
        else
        {
            VERSACORE_DEBUG_PRINT("Allocated D32tlbound at %p\r\n",
                                  (void *)D32tlbound);
        }
        get_cls_shared_ptrs()[5][18] = (uint32_t)(uintptr_t)D32tlbound;
        // D32tlbound0~3
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
            break;
        }
        D32tlbound[1] = N;
        D32tlbound[2] = M;
        D32tlbound[3] = 1;
        // D32tlstride0~3
        uint32_t *D32tlstride = (uint32_t *)snrt_l1_malloc(sizeof(uint32_t) * 4);
        if (!D32tlstride)
        {
            VERSACORE_DEBUG_PRINT("ERROR: D32tlstride alloc failed!\r\n");
            return;
        }
        else
        {
            VERSACORE_DEBUG_PRINT("Allocated D32tlstride at %p\r\n",
                                  (void *)D32tlstride);
        }
        get_cls_shared_ptrs()[5][19] = (uint32_t)(uintptr_t)D32tlstride;
        // D32tlstride0
        switch (array_shape_idx)
        {
        case 0:
            D32tlstride[0] = D32tlstride0_0;
            break;
        case 1:
            D32tlstride[0] = D32tlstride0_1;
            break;
        case 2:
            D32tlstride[0] = D32tlstride0_2;
            break;
        case 3:
            D32tlstride[0] = D32tlstride0_3;
            break;
        case 4:
            D32tlstride[0] = D32tlstride0_4;
            break;
        default:
            break;
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
        // set_addr_remap_index_D32
        get_cls_shared_ptrs()[5][20] = 0;
        // channel_en_D32 []
        switch (array_shape_idx)
        {
        case 0:
            get_cls_shared_ptrs()[5][21] = channel_en_D32_0_0;
            break;
        case 1:
            get_cls_shared_ptrs()[5][21] = channel_en_D32_1_0;
            break;
        case 2:
            get_cls_shared_ptrs()[5][21] = channel_en_D32_2_0;
            break;
        case 3:
            get_cls_shared_ptrs()[5][21] = channel_en_D32_3_0;
            break;
        case 4:
            get_cls_shared_ptrs()[5][21] = channel_en_D32_4_0;
            break;
        default:
            break;
        }
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
        // If the D addr is not in L1, we need to use the DMA to store the data back according to the input args
        if (store_D)
        {

            uint32_t D_addr_hi = get_cls_shared_ptrs()[0][6];
            uint32_t D_addr_lo = get_cls_shared_ptrs()[0][7];
            uint64_t dst_D_addr = make_u64(D_addr_hi, D_addr_lo);
            uint32_t local_D_addr = get_cls_shared_ptrs()[0][26];

            uint32_t M = get_cls_shared_ptrs()[0][8];
            uint32_t N = get_cls_shared_ptrs()[0][10];
            uint32_t meshRow = get_cls_shared_ptrs()[0][16];
            uint32_t meshCol = get_cls_shared_ptrs()[0][18];
            // Size D = M * N * meshRow * meshCol * sizeof(uint32_t)
            uint32_t D_size = M *
                              N *
                              meshRow *
                              meshCol * sizeof(uint32_t);
            VERSACORE_DEBUG_PRINT("DMA Store D Dst addr: 0x%08x%08x\r\n", D_addr_hi, D_addr_lo);
            VERSACORE_DEBUG_PRINT("DMA Store D Src: 0x%lx\r\n", chiplet_addr_transform((uint64_t)local_D_addr));
            VERSACORE_DEBUG_PRINT("DMA Store D Size %d\r\n", D_size);
            // Dst is specified from the args, SRC is L1 addr
            BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
            snrt_dma_start_1d_wideptr(dst_D_addr,
                                      chiplet_addr_transform((uint64_t)local_D_addr),
                                      D_size);
            snrt_dma_wait_all();
            BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
        }
        ///////////////////////////////////////
        // Free the L1 memory //
        ///////////////////////////////////////
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

SNAX_LIB_DEFINE void __snax_kernel_minimal_cfg_start_gemm_and_wait(void *arg)
{
    if (snrt_is_dm_core())
    {
        // allocate L1 memory for global variables across the cluster
        ///////////////////////////////////////
        // Allocate the L1 memory //
        ///////////////////////////////////////
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        get_cls_shared_ptrs()[0] = (uint32_t *)snrt_l1_malloc(128);
        if (!get_cls_shared_ptrs()[0])
        {
            VERSACORE_DEBUG_PRINT("ERROR: arg alloc failed!\r\n");
            return;
        }
        else
        {
            VERSACORE_DEBUG_PRINT("Allocated arg at %p\r\n",
                                  (void *)get_cls_shared_ptrs()[0]);
        }

        ////////////////////////////////////
        // Call the idma to load the args //
        ////////////////////////////////////
        // First we need to use the DMA to load all the arguments from L3 into
        // L1 Here we assume the arguments are packed in a contiguous way in L3

        snrt_dma_start_1d(
            (void *)get_cls_shared_ptrs()[0], (void *)arg,
            sizeof(uint32_t) * 4); // 4 uint32_t passed args in total
        snrt_dma_wait_all();
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    }

    snrt_cluster_hw_barrier();

    uint32_t *arg_ptr = get_cls_shared_ptrs()[0];
    // Extract the args
    uint32_t A_addr_lo = arg_ptr[0];
    uint32_t B_addr_lo = arg_ptr[1];
    uint32_t C_addr_lo = arg_ptr[2];
    uint32_t D_addr_lo = arg_ptr[3];

    // new pointers for the streamers
    if (snrt_cluster_core_idx() == 0)
    {
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_CFG_START);
        set_minimal_streamer_cfg(
            A_addr_lo,
            B_addr_lo,
            C_addr_lo,
            D_addr_lo);
        start_versacore_and_streamer();
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_CFG_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_RUN_START);
        wait_versacore_and_streamer();
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_RUN_END);
    }
}


