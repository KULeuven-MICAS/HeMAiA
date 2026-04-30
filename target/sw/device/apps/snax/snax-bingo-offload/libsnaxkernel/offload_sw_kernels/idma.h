// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
//
// Cluster-level iDMA kernels (bingo-sw flow). Programs the snitch iDMA to
// move data between L1 and L3, plus a couple of compute-pattern demos
// (load_compute_store, double_buffer) and the load-then-compare check kernel.

#pragma once

#include "../macros.h"

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

