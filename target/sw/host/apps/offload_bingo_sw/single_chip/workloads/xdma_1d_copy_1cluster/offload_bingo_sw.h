// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include "libbingo/bingo_api.h"
#include "host.h"
#include "xdma_data.h"
uint32_t __workload_xdma_1d_copy(bingo_task_t **task_list) {
    ///////////////////
    // Main function //
    ///////////////////
    // 1. Get the kernel function address by the kernel name and prepare the args
    // 2. Register the tasks
    // 3. Set the task dependency
    // 4. Set the assigned chiplet id and cluster id
    uint8_t assigned_chip_id = 0;
    uint8_t assigned_cluster_id = 0;
    // 1.1 Get the kernel function address by the kernel name
    check_kernel_tab_ready();
    printf_safe("Chip(%x, %x): [Host] Preparing XDMA 1D Copy Workload\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    uint32_t xdma_1d_copy_func_addr = get_device_function("__snax_kernel_xdma_1d_copy");
    uint32_t check_results_func_addr = get_device_function("__snax_kernel_check_results");
    if (xdma_1d_copy_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf("Error: Kernel symbol lookup failed! Exiting\r\n");
        return 0;
    }
    uint64_t ptr_l1_data = bingo_l1_alloc(assigned_chip_id, assigned_cluster_id, ARRAY_SIZE_BYTES(data));
    uint64_t ptr_output_data = bingo_l3_alloc(assigned_chip_id, ARRAY_SIZE_BYTES(data));
    uint64_t data_addr_l3 = chiplet_addr_transform((uint64_t)(uintptr_t)(data));
    // 1.2 Prepare the args
    // Args for the xdma copy tasks
    // Arg0: uint32_t src_addr_hi
    // Arg1: uint32_t src_addr_lo
    // Arg2: uint32_t dst_addr_hi
    // Arg3: uint32_t dst_addr_lo
    // Arg4: uint32_t size in Byte
    // Task 0: L3 -> Cluster0
    __snax_kernel_xdma_1d_copy_args_t *task_l3_to_cluster0_args = ( __snax_kernel_xdma_1d_copy_args_t *)bingo_l3_alloc(
        assigned_chip_id,
        sizeof(__snax_kernel_xdma_1d_copy_args_t));
    task_l3_to_cluster0_args->src_addr_hi = HIGH32(data_addr_l3);
    task_l3_to_cluster0_args->src_addr_lo = LOW32(data_addr_l3);
    task_l3_to_cluster0_args->dst_addr_hi = 0; // in single chiplet system, the cluster high addr is always 0
    task_l3_to_cluster0_args->dst_addr_lo = (uint32_t)ptr_l1_data;
    task_l3_to_cluster0_args->size = ARRAY_SIZE_BYTES(data);
    // Task 1: Cluster0 -> L3
    __snax_kernel_xdma_1d_copy_args_t *task_cluster0_to_l3_args = ( __snax_kernel_xdma_1d_copy_args_t *)bingo_l3_alloc(
        assigned_chip_id,
        sizeof(__snax_kernel_xdma_1d_copy_args_t));
    // Here we write back to a different location for easy checking
    // We use allocated memory at main
    
    task_cluster0_to_l3_args->src_addr_hi = 0;
    task_cluster0_to_l3_args->src_addr_lo = (uint32_t)ptr_l1_data;
    task_cluster0_to_l3_args->dst_addr_hi = HIGH32(ptr_output_data);
    task_cluster0_to_l3_args->dst_addr_lo = LOW32(ptr_output_data);
    task_cluster0_to_l3_args->size = ARRAY_SIZE_BYTES(data);

    // Task 2: Check result
    __snax_kernel_check_results_args_t *task_check_results_args = ( __snax_kernel_check_results_args_t *)bingo_l3_alloc(
        assigned_chip_id,
        sizeof(__snax_kernel_check_results_args_t));
    task_check_results_args->golden_data_addr = (uint32_t)data_addr_l3;
    task_check_results_args->output_data_addr = (uint32_t)ptr_output_data;
    task_check_results_args->data_size = 64;

    // 2. Register the tasks
    bingo_task_t *task_l3_to_cluster0 = bingo_task_create(xdma_1d_copy_func_addr,
                                                         (uint32_t)(uintptr_t)(task_l3_to_cluster0_args),
                                                         assigned_chip_id, 
                                                         assigned_cluster_id);
    bingo_task_t *task_cluster0_to_l3 = bingo_task_create(xdma_1d_copy_func_addr,
                                                         (uint32_t)(uintptr_t)(task_cluster0_to_l3_args),
                                                         assigned_chip_id,
                                                         assigned_cluster_id);
    bingo_task_t *task_check_results = bingo_task_create(check_results_func_addr,
                                                        (uint32_t)(uintptr_t)(task_check_results_args),
                                                        0,
                                                        0);
    // 3. Set the task dependency
    //      task_l3_to_cluster0
    //       |
    //       v
    //      task_cluster0_to_l3
    //       |
    //       v
    //    task_check_results
    bingo_task_add_depend(task_check_results, task_cluster0_to_l3);
    bingo_task_add_depend(task_cluster0_to_l3, task_l3_to_cluster0);

    //////////////////////
    // End main function //
    //////////////////////
    // Handle the output parameters
    // 1. Set up the task list to outside
    task_list[0] = task_l3_to_cluster0;
    task_list[1] = task_cluster0_to_l3;
    task_list[2] = task_check_results;
    asm volatile("fence" ::: "memory");
    // User need to define the number of tasks
    int num_tasks = 3;
    return num_tasks;
}


// Kernel Execution
int kernel_execution(){
    // Set up the tasks list
    // We set a maximum number of tasks to 64
    // Can be changed if needed
    bingo_task_t *task_list[64] = {0};
    uint32_t num_tasks = 0;
    if (get_current_chip_id() == 0) {
        ////////////////////////////
        // User defined workload
        ///////////////////////////
        num_tasks = __workload_xdma_1d_copy(task_list);
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
    else
    {
        // other chiplets do nothing
        return 0;
    }
}