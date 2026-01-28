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

    // 1.1 Get the kernel function address by the kernel name
    check_kernel_tab_ready();
    uint32_t xdma_1d_copy_func_addr = get_device_function("__snax_kernel_xdma_1d_copy");
    uint32_t check_results_func_addr = get_device_function("__snax_kernel_check_results");
    if (xdma_1d_copy_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf("Error: Kernel symbol lookup failed! Exiting\r\n");
        return 0;
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
    task_l3_to_cluster0_args.src_addr_hi = HIGH32(&data[0]);
    task_l3_to_cluster0_args.src_addr_lo = LOW32(&data[0]);
    task_l3_to_cluster0_args.dst_addr_hi = 0; // in single chiplet system, the cluster high addr is always 0
    task_l3_to_cluster0_args.dst_addr_lo = (uint32_t)cluster_tcdm_start_addr(0);
    task_l3_to_cluster0_args.size = ARRAY_SIZE_BYTES(data);
    // Task 1: Cluster0 -> L3
    __snax_kernel_xdma_1d_copy_args_t task_cluster0_to_l3_args;
    // Here we write back to a different location for easy checking
    // We use allocated memory at main
    uint64_t output_data_ptr = bingo_l3_alloc(get_current_chip_id(), ARRAY_SIZE_BYTES(data));
    if (!output_data_ptr) {
        printf("Error: output buffer allocation failed!\r\n");
        return 0;
    }
    task_cluster0_to_l3_args.src_addr_hi = 0;
    task_cluster0_to_l3_args.src_addr_lo = (uint32_t)cluster_tcdm_start_addr(0);
    task_cluster0_to_l3_args.dst_addr_hi = HIGH32(output_data_ptr);
    task_cluster0_to_l3_args.dst_addr_lo = LOW32(output_data_ptr);
    task_cluster0_to_l3_args.size = ARRAY_SIZE_BYTES(data);

    // Task 2: Check result
    __snax_kernel_check_results_args_t task_check_results_args;
    task_check_results_args.golden_data_addr = (uint32_t)(uintptr_t)(&data[0]);
    task_check_results_args.output_data_addr = (uint32_t)output_data_ptr;
    task_check_results_args.data_size = ARRAY_SIZE_BYTES(data);

    // 2. Register the tasks
    bingo_task_t *task_l3_to_cluster0 = bingo_task_create(xdma_1d_copy_func_addr,
                                                         (uint32_t)(uintptr_t)(&task_l3_to_cluster0_args),
                                                         get_current_chip_id(), 
                                                         0);
    bingo_task_t *task_cluster0_to_l3 = bingo_task_create(xdma_1d_copy_func_addr, (uint32_t)(uintptr_t)(&task_cluster0_to_l3_args),get_current_chip_id(), 1);
    bingo_task_t *task_check_results = bingo_task_create(check_results_func_addr, (uint32_t)(uintptr_t)(&task_check_results_args),get_current_chip_id(), 0);
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
    return 0;
}