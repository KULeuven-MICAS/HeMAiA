// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Core-level bingo kernels: dummy, entry_point, exit. These are the minimal
// control kernels the bingo-hw scheduler uses to mark task boundaries.

#pragma once

#include "../macros.h"

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_dummy(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t dummy_input = ((uint32_t *)arg)[0];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, 4);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    printf_safe("[Cluster %d Core %d]: Bingo Dummy Kernel: %d\r\n", snrt_cluster_idx(), snrt_cluster_core_idx(), dummy_input);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    sp->return_value = 0;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_entry_point(void *arg){
    // This is a special kernel to indicate the bingo hw manager loop has started
    // In the future we can add some content here
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, 4);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    printf_safe("[Cluster %d Core %d]: Start: \r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    sp->return_value = 0;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_exit(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t exit_code = ((uint32_t *)arg)[0];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, 4);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    printf_safe("[Cluster %d Core %d]: Exiting with code %d\r\n", snrt_cluster_idx(), snrt_cluster_core_idx(), exit_code);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    sp->return_value = BINGO_RET_EXIT;
    sp->num_return_values = 0;
    return BINGO_RET_EXIT;
}
