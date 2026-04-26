// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
//
// Cluster-level xDMA kernels (bingo-sw flow). Programs the snax xDMA engine.

#pragma once

#include "../macros.h"

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
