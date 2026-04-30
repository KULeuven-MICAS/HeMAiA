// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
//
// Cluster-level basic kernels (bingo-sw flow): minimal utilities that don't
// program the iDMA or xDMA — dummy, CSR R/W, in-place L1 result check.

#pragma once

#include "../macros.h"

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
    // Compare uint32 buffers already resident in L1.
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
    // Variant of check_results that DMA-loads both buffers from L3 first.
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
        uint32_t golden_data_l1 = snrt_l1_malloc(data_size);
        uint32_t output_data_l1 = snrt_l1_malloc(data_size);

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
                    printf_safe("Error at index %d: golden = %d, output = %d\n", i,
                           ((uint32_t *)golden_data_l1)[i], ((uint32_t *)output_data_l1)[i]);
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

        snrt_l1_free(golden_data_l1);
        snrt_l1_free(output_data_l1);
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    }
}

SNAX_LIB_DEFINE void __snax_kernel_load_compute_store(void *arg)
{
    // Demo: DMA-load input → compute (+1 per element) → DMA-store output.
    // Arg0: uint32_t input_data_addr
    // Arg1: uint32_t input_data_size in Byte
    // Arg2: uint32_t output_data_addr
    // Arg3: uint32_t output_data_size in Byte

    // shared_ptrs[0] arg_ptr
    // shared_ptrs[1] l1_input_data_ptr
    // shared_ptrs[2] l1_output_data_ptr
    if (snrt_is_dm_core())
    {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        // 256-byte arg scratchpad gives room for up to 64 uint32 args.
        get_cls_shared_ptrs()[0] = (uint32_t *)snrt_l1_malloc(256);
        get_cls_shared_ptrs()[0][0] = ((uint32_t *)arg)[0];
        get_cls_shared_ptrs()[0][1] = ((uint32_t *)arg)[1];
        get_cls_shared_ptrs()[0][2] = ((uint32_t *)arg)[2];
        get_cls_shared_ptrs()[0][3] = ((uint32_t *)arg)[3];
        get_cls_shared_ptrs()[1] = (uint32_t *)snrt_l1_malloc(get_cls_shared_ptrs()[0][1]);
        get_cls_shared_ptrs()[2] = (uint32_t *)snrt_l1_malloc(get_cls_shared_ptrs()[0][3]);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    }
    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core())
    {
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
        // Single compute core does an in-place +1 over the input buffer.
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
        printf_safe("[Cluster %d] Core(%d) Compute...\n", snrt_cluster_idx(),
               snrt_cluster_core_idx());
        for (uint32_t i = 0; i < get_cls_shared_ptrs()[0][1] / sizeof(uint32_t);
             i++)
        {
            get_cls_shared_ptrs()[1][i] += 1;
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    }
    snrt_cluster_hw_barrier();
    if (snrt_is_dm_core())
    {
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
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[0]);
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[1]);
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[2]);
        printf_safe("[Cluster %d] Core(%d) Free allocated heap memory\n",
               snrt_cluster_idx(), snrt_cluster_core_idx());
        snrt_memset(get_cls_shared_ptrs(), 0,
                    sizeof(uint32_t *) * NUM_CLS_SHARED_PTRS);
        printf_safe("[Cluster %d] Core(%d) Free shared pointers\n",
               snrt_cluster_idx(), snrt_cluster_core_idx());
    }
}

SNAX_LIB_DEFINE void __snax_kernel_double_buffer(void *arg)
{
    // Double-buffered load_compute_store demo.
    //
    // Arg0: uint32_t input_data_addr
    // Arg1: uint32_t output_data_addr
    // Arg2: uint32_t data_size in Byte
    // Arg3: uint32_t num_tiles (>=3)
    //
    // Naive load_compute_store has the DMA core idle during compute and the
    // compute core idle during DMA:
    //   Core_DMA  Load0 |     *    | Store0
    //   Core_Comp   *   | Compute0 |    *
    //
    // Double buffering overlaps the next-tile load with the current compute
    // and the previous-tile store, with three buffers in rotation:
    //                ____________Sync points_______________________
    //                |    |       |       |       |        |       |
    //                v    v       v       v       v        v       v
    //   Core_DMA  L0 | L1 | S0,L2 | S1,L3 | S2,L4 | S3,L5  | S4  | S5
    //   Core_Comp *  | C0 | C1    | C2    | C3    | C4     | C5  | *
    //   Time slot s0   s1   s2      s3      s4      s5       s6    s7
    // Buffer rotation: at slot s the load tile is s%3, compute is (s+2)%3,
    // store is (s+1)%3 — at most three tiles in flight at any time.

    // TLS-allocated; safe because args are read-only for the kernel run.
    uint32_t input_data_addr  = ((uint32_t *)arg)[0];
    uint32_t output_data_addr = ((uint32_t *)arg)[1];
    uint32_t input_data_size  = ((uint32_t *)arg)[2];
    uint32_t num_tiles        = ((uint32_t *)arg)[3];
    uint32_t tile_size        = input_data_size / num_tiles;
    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core())
    {
        if (num_tiles < 3)
        {
            printf_safe("Error: num_tiles should be >= 3 for double buffering example "
                   "kernel\n");
            return;
        }
        get_cls_shared_ptrs()[0] = (uint32_t *)snrt_l1_malloc(tile_size);
        get_cls_shared_ptrs()[1] = (uint32_t *)snrt_l1_malloc(tile_size);
        get_cls_shared_ptrs()[2] = (uint32_t *)snrt_l1_malloc(tile_size);
    }
    snrt_cluster_hw_barrier();

    uint32_t *tile_buf_0 = get_cls_shared_ptrs()[0];
    uint32_t *tile_buf_1 = get_cls_shared_ptrs()[1];
    uint32_t *tile_buf_2 = get_cls_shared_ptrs()[2];

    for (uint32_t s = 0; s < num_tiles + 2; s++)
    {
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
            // T2..TS-2: Load tile s + Compute tile s-1 + Store tile s-2
            if (snrt_is_dm_core())
            {
                BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
                snrt_dma_start_1d((void *)load_buf,
                                  (void *)(input_data_addr + s * tile_size),
                                  tile_size);
                snrt_dma_start_1d(
                    (void *)(output_data_addr + (s - 2) * tile_size),
                    (void *)store_buf, tile_size);
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

        if (s == num_tiles)
        {
            // TS-1: Compute tile s-1 + Store tile s-2
            if (snrt_cluster_core_idx() == 0)
            {
                BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
                for (uint32_t i = 0; i < tile_size / sizeof(uint32_t); i++)
                {
                    compute_buf[i] += 1;
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
