// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Core-level bingo IDMA kernels: 1D copy and broadcast. These program the
// local snitch iDMA to move data between L1 and L3 / other clusters without
// touching the versacore streamer.

#pragma once

#include "../macros.h"

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_idma_1d_copy(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_idma_1d_copy_args_t);
    if (snrt_is_dm_core()){
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint64_t src_addr = make_u64(((uint32_t *)arg)[0], ((uint32_t *)arg)[1]);
        uint64_t dst_addr = make_u64(((uint32_t *)arg)[2], ((uint32_t *)arg)[3]);
        uint32_t data_size = ((uint32_t *)arg)[4];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_idma_1d_copy_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
        snrt_dma_start_1d_wideptr(dst_addr, src_addr, data_size);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
        snrt_dma_wait_all();
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
        IDMA_DEBUG_PRINT("IDMA copy completed\r\n");
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else{
        printf_safe("[Cluster %d Core %d]: Error! IDMA 1D copy should be called from a DM core!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_idma_broadcast(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_idma_broadcast_args_t);

    // Copy 1d data from src to dst using idma
    // Arg0: uint32_t src_addr_hi
    // Arg1: uint32_t src_addr_lo
    // Arg2: uint32_t dst_addr_hi
    // Arg3: uint32_t dst_addr_lo
    // Arg4: uint32_t size in Byte
    if (snrt_is_dm_core()){
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint64_t src_addr = make_u64(((uint32_t *)arg)[0], ((uint32_t *)arg)[1]);
        uint64_t dst_addr = make_u64(((uint32_t *)arg)[2], ((uint32_t *)arg)[3]);
        uint64_t dst_addr_broadcast = chiplet_addr_transform_loc(0xF, 0xF, dst_addr);
        uint32_t data_size = ((uint32_t *)arg)[4];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_idma_broadcast_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
        snrt_dma_start_1d_wideptr(dst_addr_broadcast, src_addr, data_size);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
        snrt_dma_wait_all();
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
        IDMA_DEBUG_PRINT("IDMA copy completed\r\n");
        IDMA_DEBUG_PRINT("SRC ADDR = %lx\r\n", src_addr);
        IDMA_DEBUG_PRINT("DST ADDR = %lx\r\n", dst_addr_broadcast);
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else{
        printf_safe("[Cluster %d Core %d]: Error! IDMA 1D copy should be called from a DM core!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

