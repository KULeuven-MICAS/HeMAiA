// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>

#pragma once
#include "gemm_data.h"
#include "host.h"
#include "libbingo/bingo_api.h"

uint32_t __workload_versacore(bingo_task_t** task_list,
                              uintptr_t* output_data_ptr) {
    ///////////////////
    // Main function //
    ///////////////////
    // 1. Get the kernel function address by the kernel name and prepare the
    // args
    // 2. Register the tasks
    // 3. Set the task dependency
    // 4. Set the assigned chiplet id and cluster id

    uint8_t current_chip_id = get_current_chip_id();
    uint8_t task_chip_id = current_chip_id;
    uint8_t cluster_id = 0;  // versacore is located at cluster

    // 1.1 Get the kernel function address by the kernel name
    check_kernel_tab_ready();
    uint32_t check_results_func_addr =
        get_device_function("__snax_kernel_check_results");
    uint32_t __snax_kernel_gemm_intra_chiplet_func_addr =
        get_device_function("__snax_kernel_gemm_intra_chiplet");
    if (check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        __snax_kernel_gemm_intra_chiplet_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf("Error: Kernel symbol lookup failed!\r\n");
    }

    // 1.2 Prepare the args
    // versacore args
    uint32_t gemm_args[15];
    // A matrix
    gemm_args[0] = HIGH32(&A[0]);
    gemm_args[1] = LOW32(&A[0]);
    // B matrix
    gemm_args[2] = HIGH32(&B[0]);
    gemm_args[3] = LOW32(&B[0]);
    // C matrix
    if (accumPrevC || addZeroC) {
        // When accumPrevC is true, we use D as the previous C matrix
        gemm_args[4] = 0;
        gemm_args[5] = 0;
    } else {
        gemm_args[4] = HIGH32(&C[0]);
        gemm_args[5] = LOW32(&C[0]);
    }

    // D matrix (output)
    O1HeapInstance* local_l3_heap_manager =
        bingo_get_l3_heap_manager(current_chip_id);
    uintptr_t output_data_addr =
        (uintptr_t)o1heapAllocate(local_l3_heap_manager, ARRAY_SIZE_BYTES(D));
    gemm_args[6] = HIGH32(output_data_addr);
    gemm_args[7] = LOW32(output_data_addr);
    // Matrix dimensions
    gemm_args[8] = M;   // M
    gemm_args[9] = K;   // K
    gemm_args[10] = N;  // N
    // SUs
    gemm_args[11] = array_shape;
    // transpose A
    gemm_args[12] = transposed_A;
    // transpose B
    gemm_args[13] = transposed_B;
    // accumPrevC
    gemm_args[14] = accumPrevC;

    // checkresults args
    __snax_kernel_check_results_args_t task_check_results_args;
    task_check_results_args.golden_data_addr = (uint32_t)(uintptr_t)(&D);
    task_check_results_args.output_data_addr =
        (uint32_t)(uintptr_t)(output_data_addr);
    task_check_results_args.data_size = ARRAY_SIZE_BYTES(D);

    // 2. Register the tasks
    bingo_task_t* task_versacore = bingo_task_create(
        __snax_kernel_gemm_intra_chiplet_func_addr,
        (uint32_t)(uintptr_t)(&gemm_args), task_chip_id, cluster_id);
    if (task_versacore == NULL) {
        printf("Error: Task versacore creation failed!\r\n");
    }
    bingo_task_t* task_check_results =
        bingo_task_create(check_results_func_addr,
                          (uint32_t)(uintptr_t)(&task_check_results_args),
                          task_chip_id, cluster_id);
    if (task_check_results == NULL) {
        printf("Error: Task check results creation failed!\r\n");
    }

    // 3. Set the task dependency
    // Here we have only two tasks and simple dependency
    // versacore -> check results
    bingo_task_add_depend(task_check_results, task_versacore);

    // 4. Flush the caches to make sure the device can see the latest arguments
    asm volatile("fence" ::: "memory");
    //////////////////////
    // End main function //
    //////////////////////
    // Handle the output parameters

    task_list[0] = task_versacore;
    task_list[1] = task_check_results;
    uint32_t num_tasks = 2;
    output_data_ptr = (uintptr_t*)output_data_addr;
    return num_tasks;
}
