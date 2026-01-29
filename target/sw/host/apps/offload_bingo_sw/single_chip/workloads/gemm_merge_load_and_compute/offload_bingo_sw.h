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

uint32_t __workload_gemm_merge_load_and_compute(bingo_task_t** task_list) {
    ///////////////////
    // Main function //
    ///////////////////
    // 1. Get the kernel function address by the kernel name and prepare the
    // args
    // 2. Register the tasks
    // 3. Set the task dependency
    // 4. Set the assigned chiplet id and cluster id

    // only test chip 0 for intra-chiplet gemm
    uint8_t assigned_chip_id = 0;
    uint8_t assigned_cluster_id = 0;

    // 1.1 Get the kernel function address by the kernel name
    check_kernel_tab_ready();
    printf_safe("Chip(%x, %x): [Host] Preparing GEMM Merge Load and Compute Workload\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());

    uint32_t check_results_func_addr =
        get_device_function("__snax_kernel_check_results");
    uint32_t __snax_kernel_gemm_func_addr =
        get_device_function("__snax_kernel_gemm");
    if (check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        __snax_kernel_gemm_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf("Error: Kernel symbol lookup failed!\r\n");
    }
    uint64_t output_data_ptr = bingo_l3_alloc(assigned_chip_id, ARRAY_SIZE_BYTES(D));
    // 1.2 Prepare the args
    // versacore args
    __snax_kernel_gemm_intra_chiplet_args_t* gemm_args = (__snax_kernel_gemm_intra_chiplet_args_t*)bingo_l3_alloc(
        assigned_chip_id,
        sizeof(__snax_kernel_gemm_intra_chiplet_args_t));
    // A matrix
    gemm_args->input_A_addr_hi = HIGH32(&A[0]);
    gemm_args->input_A_addr_lo = LOW32(&A[0]);
    // B matrix
    gemm_args->input_B_addr_hi = HIGH32(&B[0]);
    gemm_args->input_B_addr_lo = LOW32(&B[0]);
    // C matrix
    if (accumPrevC || addZeroC) {
        // When accumPrevC is true, we use D as the previous C matrix
        gemm_args->input_C_addr_hi = 0;
        gemm_args->input_C_addr_lo = 0;
    } else {
        gemm_args->input_C_addr_hi = HIGH32(&C[0]);
        gemm_args->input_C_addr_lo = LOW32(&C[0]);
    }

    // D matrix (output)
   
    gemm_args->output_D_addr_hi = HIGH32(output_data_ptr);
    gemm_args->output_D_addr_lo = LOW32(output_data_ptr);
    // Matrix dimensions
    gemm_args->M = M;   // M
    gemm_args->K = K;   // K
    gemm_args->N = N;  // N
    // SUs
    gemm_args->array_shape = array_shape;
    // transpose A
    gemm_args->transpose_A = transposed_A;
    // transpose B
    gemm_args->transpose_B = transposed_B;
    // accumPrevC
    gemm_args->accumPrevC = accumPrevC;

    // checkresults args
    __snax_kernel_check_results_args_t task_check_results_args;
    task_check_results_args.golden_data_addr = (uint32_t)(uintptr_t)(&D);
    task_check_results_args.output_data_addr = (uint32_t)(output_data_ptr);
    task_check_results_args.data_size = ARRAY_SIZE_BYTES(D);

    // 2. Register the tasks
    bingo_task_t* task_versacore = bingo_task_create(
        __snax_kernel_gemm_func_addr,
        (uint32_t)(uintptr_t)(gemm_args), assigned_chip_id, assigned_cluster_id);
    if (task_versacore == NULL) {
        printf("Error: Task versacore creation failed!\r\n");
    }
    bingo_task_t* task_check_results =
        bingo_task_create(check_results_func_addr,
                          (uint32_t)(uintptr_t)(&task_check_results_args),
                          assigned_chip_id, assigned_cluster_id);
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
    return num_tasks;
}

// Kernel Execution
int kernel_execution(){
    // Set up the tasks list
    // We set a maximum number of tasks to 64
    // Can be changed if needed
    bingo_task_t *task_list[64] = {0};
    uint32_t num_tasks = 0;

    num_tasks = __workload_gemm_merge_load_and_compute(task_list);
    ////////////////////////////
    // End user defined workload
    ////////////////////////////

    // Call bingo runtime
    bingo_runtime_schedule(
        task_list, 
        num_tasks
    );
    printf("Chip(%x, %x): [Host] All tasks done.\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    // Close all the clusters
    bingo_close_all_clusters(task_list, num_tasks);
    // Free the output data

    return 0;
}