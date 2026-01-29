// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include "libbingo/bingo_api.h"
#include "host.h"
#include "double_buffer_data.h"

uint32_t __workload_double_buffer(bingo_task_t **task_list) {

    // This is a double buffering workload for testing
    // It creates two tasks on two different clusters

    ///////////////////
    // Main function //
    ///////////////////
    // 1. Get the kernel function address by the kernel name and prepare the args
    // 2. Register the tasks
    // 3. Set the task dependency

    // 1.Get the kernel function address by the kernel name
    uint8_t assigned_chip_id = 0;
    uint8_t assigned_cluster_id = 0;
    check_kernel_tab_ready();
    printf_safe("Chip(%x, %x): [Host] Preparing Double Buffer Workload\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    uint32_t double_buffer_func_addr = get_device_function("__snax_kernel_double_buffer");
    uint32_t check_results_func_addr = get_device_function("__snax_kernel_check_results");
    if (double_buffer_func_addr == SNAX_SYMTAB_END_FN_ADDR || check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf("Error: Kernel symbol lookup failed! Exiting\r\n");
        return 0;
    }
    uint64_t output_data_ptr = bingo_l3_alloc(assigned_chip_id, ARRAY_SIZE_BYTES(input_array));
    // 2. Register the tasks
    ///////////////////////////////////
    /// Double buffer args
    ///////////////////////////////////    
    __snax_kernel_double_buffer_args_t* double_buffer_args = (__snax_kernel_double_buffer_args_t*)bingo_l3_alloc(
        assigned_chip_id,
        sizeof(__snax_kernel_double_buffer_args_t)
    );
    double_buffer_args->input_data_addr = (uint32_t)(uintptr_t)(&input_array);
    double_buffer_args->output_data_addr = (uint32_t)(uintptr_t)(output_data_ptr);
    double_buffer_args->data_size = ARRAY_SIZE_BYTES(input_array);
    double_buffer_args->num_tiles = 4;
    bingo_task_t *task_double_buffer = bingo_task_create(double_buffer_func_addr,                   // function ptr
                                                        (uint32_t)(uintptr_t)(double_buffer_args),  // args ptr
                                                        assigned_chip_id,                           // assigned chip id
                                                        assigned_cluster_id                         // assigned cluster id
                                                    );
    ///////////////////////////////////
    /// Check results args
    ///////////////////////////////////
    __snax_kernel_check_results_args_t* task_check_results_args = (__snax_kernel_check_results_args_t*)bingo_l3_alloc(
        assigned_chip_id,
        sizeof(__snax_kernel_check_results_args_t)
    );
    task_check_results_args->golden_data_addr = (uint32_t)(uintptr_t)(&golden_output_array);
    task_check_results_args->output_data_addr = (uint32_t)(uintptr_t)(output_data_ptr);
    task_check_results_args->data_size = ARRAY_SIZE_BYTES(input_array);
    // Adjust cluster id for check results task
    // If there are more than 1 cluster, we assign it to cluster 1 for better test coverage
    uint8_t check_result_cluster_id;
    if (N_CLUSTERS_PER_CHIPLET >= 2) {
        check_result_cluster_id = 1;
    } else {
        check_result_cluster_id = 0;
    }
    bingo_task_t *task_check_results = bingo_task_create(check_results_func_addr,
                                                        (uint32_t)(uintptr_t)(task_check_results_args),
                                                        assigned_chip_id,
                                                        check_result_cluster_id
                                                    );

    // 3. Set the task dependency
    //      T0 (cluster 0)
    //       |
    //       v
    //      T1 
    bingo_task_add_depend(task_check_results, task_double_buffer);

    //////////////////////
    // End main function //
    //////////////////////
    // Handle the output parameters
    // 1. Set up the task list to outside
    task_list[0] = task_double_buffer;
    task_list[1] = task_check_results;
    asm volatile("fence" ::: "memory");
    // User need to define the number of tasks

    int num_tasks = 2;
    return num_tasks;
}

// Kernel Execution
int kernel_execution(){
    // Set up the tasks list
    // We set a maximum number of tasks to 64
    // Can be changed if needed
    bingo_task_t *task_list[64] = {0};
    uint32_t num_tasks = 0;

    num_tasks = __workload_double_buffer(task_list);
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