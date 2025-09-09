// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include "data.h"
#include "libbingo/bingo_api.h"
#include "host.h"
#include "device_kernel_args.h"
void __workload_xdma_1d_copy(bingo_task_t **task_list, uint32_t *num_tasks_ptr, uintptr_t *output_data_ptr) {

    // User need to define the number of tasks
    int num_tasks = 3;


    ///////////////////
    // Main function //
    ///////////////////
    // 1. Get the kernel function address by the kernel name and prepare the args
    // 2. Register the tasks
    // 3. Set the task dependency
    // 4. Set the assigned chiplet id and cluster id

    // 1.1 Get the kernel function address by the kernel name
    check_kernel_tab_ready();
    uint32_t xdma_1d_copy_func_addr = get_device_function("__snax_kernel_xdma_1d_copy");
    uint32_t check_results_func_addr = get_device_function("__snax_kernel_check_results");
    if (xdma_1d_copy_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf("Error: Kernel symbol lookup failed!\r\n");
    }
    // 1.2 Prepare the args
    // Args for the xdma copy tasks
    // Arg0: uint32_t src_addr_hi
    // Arg1: uint32_t src_addr_lo
    // Arg2: uint32_t dst_addr_hi
    // Arg3: uint32_t dst_addr_lo
    // Arg4: uint32_t size in Byte
    // Task 0: L3 -> Cluster0
    __snax_kernel_xdma_1d_copy_args_t task_l3_to_cluster0_args;
    task_l3_to_cluster0_args.src_addr_hi = (uint32_t)(((uint64_t)(&data[0]) >> 32) & 0xFFFFFFFF);
    task_l3_to_cluster0_args.src_addr_lo = (uint32_t)(((uint64_t)(&data[0]) >> 0) & 0xFFFFFFFF);
    task_l3_to_cluster0_args.dst_addr_hi = 0; // in single chiplet system, the cluster high addr is always 0
    task_l3_to_cluster0_args.dst_addr_lo = (uint32_t)cluster_tcdm_start_addr(0);
    task_l3_to_cluster0_args.size = data_size * sizeof(data[0]);
    // Task 1: Cluster0 -> Cluster1
    __snax_kernel_xdma_1d_copy_args_t task_cluster0_to_cluster1_args;
    task_cluster0_to_cluster1_args.src_addr_hi = 0;
    task_cluster0_to_cluster1_args.src_addr_lo = (uint32_t)cluster_tcdm_start_addr(0);
    task_cluster0_to_cluster1_args.dst_addr_hi = 0;
    task_cluster0_to_cluster1_args.dst_addr_lo = (uint32_t)cluster_tcdm_start_addr(1);
    task_cluster0_to_cluster1_args.size = data_size * sizeof(data[0]);
    // Task 2: Cluster1 -> L3
    __snax_kernel_xdma_1d_copy_args_t task_cluster1_to_l3_args;
    // Here we write back to a different location for easy checking
    // We use allocated memory at main
    uintptr_t output_data_addr = hero_host_l3_malloc(data_size * sizeof(data[0]), output_data_ptr);
    task_cluster1_to_l3_args.src_addr_hi = 0;
    task_cluster1_to_l3_args.src_addr_lo = (uint32_t)cluster_tcdm_start_addr(1);
    task_cluster1_to_l3_args.dst_addr_hi = (uint32_t)(((uint64_t)(output_data_addr) >> 32) & 0xFFFFFFFF);
    task_cluster1_to_l3_args.dst_addr_lo = (uint32_t)(((uint64_t)(output_data_addr) >> 0) & 0xFFFFFFFF);
    task_cluster1_to_l3_args.size = data_size * sizeof(data[0]);

    // Task 3: Check result
    __snax_kernel_check_results_args_t task_check_results_args;
    task_check_results_args.golden_data_addr = (uint32_t)(uintptr_t)(&data[0]);
    task_check_results_args.output_data_addr = (uint32_t)(uintptr_t)(*output_data_ptr);
    task_check_results_args.data_size = data_size * sizeof(data[0]);

    // 2. Register the tasks
    bingo_task_t *task_l3_to_cluster0 = bingo_task_create(xdma_1d_copy_func_addr, (uint32_t)(uintptr_t)(&task_l3_to_cluster0_args));
    bingo_task_t *task_cluster0_to_cluster1 = bingo_task_create(xdma_1d_copy_func_addr, (uint32_t)(uintptr_t)(&task_cluster0_to_cluster1_args));
    bingo_task_t *task_cluster1_to_l3 = bingo_task_create(xdma_1d_copy_func_addr, (uint32_t)(uintptr_t)(&task_cluster1_to_l3_args));
    bingo_task_t *task_check_results = bingo_task_create(check_results_func_addr, (uint32_t)(uintptr_t)(&task_check_results_args));
    // 3. Set the task dependency
    //      task_l3_to_cluster0
    //       |
    //       v
    //      task_cluster0_to_cluster1
    //       |
    //       v
    //      task_cluster1_to_l3
    //       |
    //       v
    //    task_check_results
    bingo_task_add_depend(task_check_results, task_cluster1_to_l3);
    bingo_task_add_depend(task_cluster1_to_l3, task_cluster0_to_cluster1);
    bingo_task_add_depend(task_cluster0_to_cluster1, task_l3_to_cluster0);
    // 4. Set the assigned cluster id
    task_l3_to_cluster0->assigned_cluster_id = 0;
    task_cluster0_to_cluster1->assigned_cluster_id = 1;
    task_cluster1_to_l3->assigned_cluster_id = 1;
    task_check_results->assigned_cluster_id = 0; // We set the check results task to cluster 0
    //////////////////////
    // End main function //
    //////////////////////
    // Handle the output parameters
    // 1. Set up the task list to outside
    task_list[0] = task_l3_to_cluster0;
    task_list[1] = task_cluster0_to_cluster1;
    task_list[2] = task_cluster1_to_l3;
    task_list[3] = task_check_results;
    // Since this is a single chiplet workload
    // We set the assigned chiplet id to 0 for all tasks
    for (int i = 0; i < num_tasks; i++)
    {
        task_list[i]->assigned_chip_id = 0;
    }
    // 2. Set the number of tasks to outside
    *num_tasks_ptr = num_tasks;    
}