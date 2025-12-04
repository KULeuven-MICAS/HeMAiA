// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>

#pragma once
#include "gemm_multi_chiplet_data.h"
#include "host.h"
#include "libbingo/bingo_api.h"

// #define HOST_DEBUG
// #define HOST_DEBUG
#ifdef HOST_DEBUG
#define HOST_DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define HOST_DEBUG_PRINT(...)
#endif

// This workload tests the 

uint32_t __workload_versacore_stacked_gemm_intra_chiplet(bingo_task_t** task_list) {
    // In a multichiplet scenario, the task_list should be created in each chiplet
    // The important thing is that the dependency is created for all local task_list var
    // But the task arguments should be set only in the corresponding chiplet
    ///////////////////
    // Main function //
    ///////////////////
    // 1. Get the kernel function address by the kernel name
    // 2. Prepare the args
    // 3. Register the tasks
    // 4. Set the task dependency
    // 5. Set the assigned chiplet id and cluster id

    // 1 Get the needed kernel function address by the kernel name
    check_kernel_tab_ready();

    uint32_t __snax_kernel_xdma_1d_copy_func_addr = get_device_function("__snax_kernel_xdma_1d_copy");
    uint32_t __snax_kernel_idma_1d_copy_func_addr = get_device_function("__snax_kernel_idma_1d_copy");
    uint32_t __snax_kernel_gemm = get_device_function("__snax_kernel_gemm");
    uint32_t __snax_kernel_check_results_func_addr = get_device_function("__snax_kernel_check_results");
    if (__snax_kernel_xdma_1d_copy_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        __snax_kernel_idma_1d_copy_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        __snax_kernel_gemm == SNAX_SYMTAB_END_FN_ADDR ||
        __snax_kernel_check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        HOST_DEBUG_PRINT("Error: Kernel symbol lookup failed!\r\n");
    }

    asm volatile("fence" ::: "memory");

    uint32_t num_tasks = 26;

    return num_tasks;
}
