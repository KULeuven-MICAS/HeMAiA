// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
//
// Cluster-level SNAX kernels (used by the pure bingo-sw flow). These kernels
// run on a whole cluster and coordinate across the DMA core and compute
// cores. Nothing here depends on the versacore hwcfg — the HW-parameterized
// GEMM kernels live in offload_hw_kernels/gemm.h and are regenerated from
// the hwcfg.

#pragma once

#include "../macros.h"

//////////////////////// Cluster-level Kernel functions ////////////////////////
// Template for user defined kernels
// Each kernel function takes a single void* argument

// SNAX_LIB_DEFINE void __snax_kernel_template(void *arg){
//     // Some description of the kernel
//     // The arguments are all uin32_t type
//     // Arg0: ...
//     // Arg1: ...
// }

SNAX_LIB_DEFINE void __snax_kernel_dummy(void *arg)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t arg0 = ((uint32_t *)arg)[0];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    if (snrt_is_dm_core())
    {
        printf_safe(
            "Chip(%x, %x): [Cluster %d] Core(%d) Running Dummy Kernel with "
            "arg[0] = %d\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(),
            snrt_cluster_idx(), snrt_cluster_core_idx(), arg0);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
}

SNAX_LIB_DEFINE void __snax_kernel_csr(void *arg)
{
    // Arg0: csr_addr
    // Arg1: csr_value

    // shared_ptrs[0] arg_ptr
    // shared_ptrs[1] csr_read_value
    if (snrt_is_dm_core())
    {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        get_cls_shared_ptrs()[0] = (uint32_t *)snrt_l1_malloc(256);
        get_cls_shared_ptrs()[1] = (uint32_t *)snrt_l1_malloc(4);
        get_cls_shared_ptrs()[0][0] = ((uint32_t *)arg)[0];
        get_cls_shared_ptrs()[0][1] = ((uint32_t *)arg)[1];
        uint32_t csr_addr = get_cls_shared_ptrs()[0][0];
        uint32_t csr_write_value = get_cls_shared_ptrs()[0][1];
        uint32_t csr_read_value = get_cls_shared_ptrs()[1][0];
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
        printf_safe(
            "Chip(%x, %x): Running CSR Kernel with arg[0](csr_addr) = %d, "
            "arg[1](csr_value) = %x\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(), csr_addr,
            csr_write_value);
        csrw_ss(csr_addr, csr_write_value);
        csr_read_value = csrr_ss(csr_addr);
        if (csr_read_value != csr_write_value)
        {
            printf_safe("Chip(%x, %x): Error! R/W CSR values are not the same!\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y());
        }
        else
        {
            printf_safe("Chip(%x, %x): Pass!\n", get_current_chip_loc_x(),
                   get_current_chip_loc_y());
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    }
}

SNAX_LIB_DEFINE void __snax_kernel_check_results(void *arg)
{
    // This function is used to check the results of the computation
    // We assume the data type is uint32_t
    // Arg0: golden_data_addr
    // Arg1: output_data_addr
    // Arg2: data_size in Byte
    if (snrt_is_dm_core())
    {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t golden_data_addr = ((uint32_t *)arg)[0];
        uint32_t output_data_addr = ((uint32_t *)arg)[1];
        uint32_t data_size = ((uint32_t *)arg)[2];
        uint32_t num_errors = 0;
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
        for (uint32_t i = 0; i < data_size / sizeof(uint32_t); i++)
        {
            if (((uint32_t *)golden_data_addr)[i] !=
                ((uint32_t *)output_data_addr)[i])
            {
                num_errors++;
                if (num_errors < 10)
                {
                    printf_safe("Error at index %d: golden = %x, output = %x\n", i,
                           ((uint32_t *)golden_data_addr)[i],
                           ((uint32_t *)output_data_addr)[i]);
                }
            }
        }
        if (num_errors == 0)
        {
            printf_safe("Check Results: PASS! No errors found.\n");
        }
        else
        {
            printf_safe("Check Results: FAIL! %d errors found.\n", num_errors);
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    }
}

SNAX_LIB_DEFINE void __snax_kernel_check_results_full(void *arg)
{
    // This function is used to check the results of the computation
    // Different from the previous check_results kernel, this function will use the DMA core to load data from L3 to L1 first
    // Arg0: golden_data_addr_hi
    // Arg1: golden_data_addr_lo
    // Arg2: output_data_addr_hi
    // Arg3: output_data_addr_lo
    // Arg4: data_size in Byte
    if (snrt_is_dm_core())
    {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t golden_data_addr_hi = ((uint32_t *)arg)[0];
        uint32_t golden_data_addr_lo = ((uint32_t *)arg)[1];
        uint32_t output_data_addr_hi = ((uint32_t *)arg)[2];
        uint32_t output_data_addr_lo = ((uint32_t *)arg)[3];
        uint32_t data_size = ((uint32_t *)arg)[4];
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
        uint32_t golden_data_l1 =
            snrt_l1_malloc(data_size); // allocate L1 memory for golden data
        uint32_t output_data_l1 =
            snrt_l1_malloc(data_size); // allocate L1 memory for output data

        snrt_dma_start_1d((void *)golden_data_l1,
                          (void *)make_u64(golden_data_addr_hi, golden_data_addr_lo),
                          data_size);
        snrt_dma_start_1d((void *)output_data_l1,
                          (void *)make_u64(output_data_addr_hi, output_data_addr_lo),
                          data_size);
        snrt_dma_wait_all();

        uint32_t num_errors = 0;
        for (uint32_t i = 0; i < data_size / sizeof(uint32_t); i++)
        {
            if (((uint32_t *)golden_data_l1)[i] != ((uint32_t *)output_data_l1)[i])
            {
                num_errors++;
                if (num_errors < 10)
                {
                    printf("Error at index %d: golden = %d, output = %d\n", i,
                           ((uint32_t *)golden_data_l1)[i], ((uint32_t *)output_data_l1)[i]);
                }
            }
        }
        if (num_errors == 0)
        {
            printf("Check Results: PASS! No errors found.\n");
        }
        else
        {
            printf("Check Results: FAIL! %d errors found.\n", num_errors);
        }

        snrt_l1_free(golden_data_l1);
        snrt_l1_free(output_data_l1);
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    }
}

SNAX_LIB_DEFINE void __snax_kernel_load_compute_store(void *arg)
{
    // Arg0: uint32_t input_data_addr
    // Arg1: uint32_t input_data_size in Byte
    // Arg2: uint32_t output_data_addr
    // Arg3: uint32_t output_data_size in Byte

    // shared_ptrs[0] arg_ptr
    // shared_ptrs[1] l1_input_data_ptr
    // shared_ptrs[2] l1_output_data_ptr
    if (snrt_is_dm_core())
    {
        // We allocate 256 bytes of heap memory for the args
        // Each arg is 4 bytes, so we can store 64 args
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        get_cls_shared_ptrs()[0] = (uint32_t *)snrt_l1_malloc(256);
        get_cls_shared_ptrs()[0][0] = ((uint32_t *)arg)[0];
        get_cls_shared_ptrs()[0][1] = ((uint32_t *)arg)[1];
        get_cls_shared_ptrs()[0][2] = ((uint32_t *)arg)[2];
        get_cls_shared_ptrs()[0][3] = ((uint32_t *)arg)[3];
        // We then allocate the space for input and output data
        get_cls_shared_ptrs()[1] = (uint32_t *)snrt_l1_malloc(get_cls_shared_ptrs()[0][1]);
        get_cls_shared_ptrs()[2] = (uint32_t *)snrt_l1_malloc(get_cls_shared_ptrs()[0][3]);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    }
    snrt_cluster_hw_barrier();
    
    if (snrt_is_dm_core())
    {
        // load the input data
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
        snrt_dma_start_1d((void *)get_cls_shared_ptrs()[1],
                          (void *)get_cls_shared_ptrs()[0][0],
                          get_cls_shared_ptrs()[0][1]);
        snrt_dma_wait_all();
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
    }
    snrt_cluster_hw_barrier();
    if (snrt_cluster_core_idx() == 0)
    {
        // We choose the first core to do the dummy +1 computation
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
        printf("[Cluster %d] Core(%d) Compute...\n", snrt_cluster_idx(),
               snrt_cluster_core_idx());
        for (uint32_t i = 0; i < get_cls_shared_ptrs()[0][1] / sizeof(uint32_t);
             i++)
        {
            get_cls_shared_ptrs()[1][i] += 1;
        }
        // printf("[Cluster %d] Core(%d) Validate...\n", snrt_cluster_idx(),
        // snrt_cluster_core_idx()); for(uint32_t
        // i=0;i<get_cls_shared_ptrs()[0][1]/sizeof(uint32_t);i++){
        //     printf("[Cluster %d] Core(%d) Output[%d] = %d\n",
        //     snrt_cluster_idx(), snrt_cluster_core_idx(), i,
        //     get_cls_shared_ptrs()[1][i]);
        // }
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    }
    snrt_cluster_hw_barrier();
    if (snrt_is_dm_core())
    {
        // We store the output data
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
        snrt_dma_start_1d((void *)get_cls_shared_ptrs()[0][2],
                          (void *)get_cls_shared_ptrs()[1],
                          get_cls_shared_ptrs()[0][3]);
        snrt_dma_wait_all();
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
    }
    snrt_cluster_hw_barrier();
    if (snrt_is_dm_core())
    {
        // Free the allocated memory
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[0]);
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[1]);
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[2]);
        printf("[Cluster %d] Core(%d) Free allocated heap memory\n",
               snrt_cluster_idx(), snrt_cluster_core_idx());
        snrt_memset(get_cls_shared_ptrs(), 0,
                    sizeof(uint32_t *) * NUM_CLS_SHARED_PTRS);
        printf("[Cluster %d] Core(%d) Free shared pointers\n",
               snrt_cluster_idx(), snrt_cluster_core_idx());
    }
}

SNAX_LIB_DEFINE void __snax_kernel_double_buffer(void *arg)
{
    // Example kernel for double buffering

    // Input data is assumed to be array of uint32_t
    // Arg0: uint32_t input_data_addr
    // Arg1: uint32_t output_data_addr
    // Arg2: uint32_t data_size in Byte
    // Arg3: uint32_t num_tiles (>3)
    // This kernel demonstrates how to do the double buffering
    // ----------------------
    // Double buffering idea
    // ----------------------
    // One standard way to to load_compute_store is as follows:
    // * - Bubbles indicate idle time
    // Core_DMA  Load0 |     *    | Store0
    // Core_Comp   *   | Compute0 |    *
    // The double buffering idea is to overlap the loading of the next tile
    // If we have 6 tiles, the computation flow will be as follows:
    //              ____________Sync points_______________________
    //              |    |       |       |       |        |       |
    //              v    v       v       v       v        v       v
    // Core_DMA  L0 | L1 | S0,L2 | S1,L3 | S2,L4 | S3,L5  | S4  | S5
    // Core_Comp *  | C0 | C1    | C2    | C3    | C4     | C5  | *
    // Time slot s0   s1   s2      s3      s4      s5       s6    s7
    // Memory layout:
    // Buffer 0: t0                t3                       t6
    // Buffer 1:      t1                   t4
    // Buffer 2:           t2                     t5
    // So at the same time at most three tiles are in the system
    // Memory mappings:
    // T     load tile | compute tile | store tile
    // s=0   Buf0      | ---          | ---
    // s=1   Buf1      | Buf0         | ---
    // s=2   Buf2      | Buf1         | Buf0
    // s=3   Buf0      | Buf2         | Buf1
    // s=4   Buf1      | Buf0         | Buf2
    // s=5   Buf2      | Buf1         | Buf0
    // s=x   x%3       | (x+2)%3      | (x+1)%3
    // ....
    // s=S-1 ----      | (S+1)%3      | (S)  %3
    // s=S   ----      | ----         | (S+1)%3
    // ----------- Main code -------------------

    // Notice the following vars are allocated in the Thread-Local Storage (TLS)
    // which means each core has its own copy
    // Since those are just arguments and will not be modified during the kernel
    // execution, it is safe to put here
    uint32_t input_data_addr = ((uint32_t *)arg)[0];
    uint32_t output_data_addr = ((uint32_t *)arg)[1];
    uint32_t input_data_size = ((uint32_t *)arg)[2];
    uint32_t num_tiles = ((uint32_t *)arg)[3];        // assume num_tiles >= 3
    uint32_t tile_size = input_data_size / num_tiles; // in Byte
    // if(snrt_is_dm_core()){
    //     printf("input_data_addr=0x%x, output_data_addr=0x%x,
    //     input_data_size=%d, num_tiles=%d, tile_size=%d bytes\n",
    //            input_data_addr, output_data_addr, input_data_size, num_tiles,
    //            tile_size);
    // }
    snrt_cluster_hw_barrier();

    // Use one core to allocate the shared storage
    if (snrt_is_dm_core())
    {
        if (num_tiles < 3)
        {
            printf(
                "Error: num_tiles should be >= 3 for double buffering example "
                "kernel\n");
            return;
        }
        get_cls_shared_ptrs()[0] =
            (uint32_t *)snrt_l1_malloc(tile_size); // ptr for temporal tile storage 1
        get_cls_shared_ptrs()[1] =
            (uint32_t *)snrt_l1_malloc(tile_size); // ptr for temporal tile storage 2
        get_cls_shared_ptrs()[2] =
            (uint32_t *)snrt_l1_malloc(tile_size); // ptr for temporal tile storage 3
        // printf("Allocated shared buffers, buf1=0x%x, buf2=0x%x, buf3=0x%x\n",
        //        get_cls_shared_ptrs()[0],
        //        get_cls_shared_ptrs()[1],
        //        get_cls_shared_ptrs()[2]);
    }
    snrt_cluster_hw_barrier();
    // Here all the cores can access the shared tile buffers
    uint32_t *tile_buf_0 = get_cls_shared_ptrs()[0];
    uint32_t *tile_buf_1 = get_cls_shared_ptrs()[1];
    uint32_t *tile_buf_2 = get_cls_shared_ptrs()[2];

    for (uint32_t s = 0; s < num_tiles + 2; s++)
    {
        // Here all the varialbes are derived from the tile_bufs, hence they are
        // shared among all cores
        uint32_t *load_buf =
            s % 3 == 0 ? tile_buf_0 : (s % 3 == 1 ? tile_buf_1 : tile_buf_2);
        uint32_t *compute_buf =
            (s + 2) % 3 == 0 ? tile_buf_0
                             : ((s + 2) % 3 == 1 ? tile_buf_1 : tile_buf_2);
        uint32_t *store_buf =
            (s + 1) % 3 == 0 ? tile_buf_0
                             : ((s + 1) % 3 == 1 ? tile_buf_1 : tile_buf_2);
        if (s == 0)
        {
            
            // T0: Load tile 0
            if (snrt_is_dm_core())
            {
                BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
                snrt_dma_start_1d((void *)load_buf, (void *)input_data_addr,
                                  tile_size);
                snrt_dma_wait_all();
                BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
            }
            
        }

        if (s == 1)
        {
            // T1: Load tile 1 + Compute tile 0
            if (snrt_is_dm_core())
            {
                BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
                snrt_dma_start_1d((void *)load_buf,
                                  (void *)(input_data_addr + tile_size),
                                  tile_size);
                snrt_dma_wait_all();
                BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
            }
            if (snrt_cluster_core_idx() == 0)
            {   
                BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
                for (uint32_t i = 0; i < tile_size / sizeof(uint32_t); i++)
                {
                    compute_buf[i] += 1;
                }
                BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
            }
        }

        if (s >= 2 && s <= num_tiles - 1)
        {
            // T2 to TS-2: Load tile s + Compute tile s-1 + Store tile s-2
            if (snrt_is_dm_core())
            {
                // Load tile s
                BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
                snrt_dma_start_1d((void *)load_buf,
                                  (void *)(input_data_addr + s * tile_size),
                                  tile_size);
                // Store tile s-2
                snrt_dma_start_1d(
                    (void *)(output_data_addr + (s - 2) * tile_size),
                    (void *)store_buf, tile_size);
                snrt_dma_wait_all();
                BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
            }
            // Compute tile s-1
            if (snrt_cluster_core_idx() == 0)
            {
                BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
                for (uint32_t i = 0; i < tile_size / sizeof(uint32_t); i++)
                {
                    compute_buf[i] += 1;
                }
                BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
            }
        }

        if (s == num_tiles)
        {
            // TS-1: Compute tile s-1 + Store tile s-2
            if (snrt_cluster_core_idx() == 0)
            {
                BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
                for (uint32_t i = 0; i < tile_size / sizeof(uint32_t); i++)
                {
                    compute_buf[i] += 1; // Example computation
                }
                BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
            }
            if (snrt_is_dm_core())
            {
                BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
                snrt_dma_start_1d(
                    (void *)(output_data_addr + (num_tiles - 2) * tile_size),
                    (void *)store_buf, tile_size);
                snrt_dma_wait_all();
                BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
            }
        }

        if (s == num_tiles + 1)
        {
            // TS: Store tile s-2
            if (snrt_is_dm_core())
            {
                BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
                snrt_dma_start_1d(
                    (void *)(output_data_addr + (num_tiles - 1) * tile_size),
                    (void *)store_buf, tile_size);
                snrt_dma_wait_all();
                BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
            }
        }

        snrt_cluster_hw_barrier();
    }
}

SNAX_LIB_DEFINE void __snax_kernel_xdma_1d_copy(void *arg)
{
    // Copy 1d data from src to dst using xdma
    // Arg0: uint32_t src_addr_hi
    // Arg1: uint32_t src_addr_lo
    // Arg2: uint32_t dst_addr_hi
    // Arg3: uint32_t dst_addr_lo
    // Arg4: uint32_t size in Byte

    if (snrt_is_dm_core())
    {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint64_t src_addr = make_u64(((uint32_t *)arg)[0], ((uint32_t *)arg)[1]);
        uint64_t dst_addr = make_u64(((uint32_t *)arg)[2], ((uint32_t *)arg)[3]);
        uint32_t data_size = ((uint32_t *)arg)[4];
        XDMA_DEBUG_PRINT("XDMA copy: src_addr=0x%lx, dst_addr=0x%lx, size=%d bytes\n",
               src_addr, dst_addr, data_size);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        xdma_memcpy_1d_full_addr(src_addr, dst_addr, data_size);
        int task_id = xdma_start();
        xdma_wait_task(src_addr, dst_addr, task_id);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
    }
}

SNAX_LIB_DEFINE void __snax_kernel_idma_1d_copy(void *arg)
{
    // Copy 1d data from src to dst using idma
    // Arg0: uint32_t src_addr_hi
    // Arg1: uint32_t src_addr_lo
    // Arg2: uint32_t dst_addr_hi
    // Arg3: uint32_t dst_addr_lo
    // Arg4: uint32_t size in Byte

    if (snrt_is_dm_core())
    {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint64_t src_addr = make_u64(((uint32_t *)arg)[0], ((uint32_t *)arg)[1]);
        uint64_t dst_addr = make_u64(((uint32_t *)arg)[2], ((uint32_t *)arg)[3]);
        uint32_t data_size = ((uint32_t *)arg)[4];
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
        snrt_dma_start_1d_wideptr(dst_addr, src_addr, data_size);
        snrt_dma_wait_all();
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
        IDMA_DEBUG_PRINT("IDMA copy completed\r\n");
        IDMA_DEBUG_PRINT("SRC ADDR = %lx\r\n", src_addr);
        IDMA_DEBUG_PRINT("DST ADDR = %lx\r\n", dst_addr);
    }
}
