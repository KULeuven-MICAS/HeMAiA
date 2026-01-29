// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include "libbingo/bingo_api.h"
#include "host.h"
#include "dummy_data.h"

uint32_t __workload_dummy(bingo_task_t **task_list) {

    check_kernel_tab_ready();
    uint32_t dummy_func_addr = get_device_function("__snax_kernel_dummy");
    if (dummy_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf_safe("Error: Kernel symbol lookup failed! Exiting\r\n");
        return 0;
    }

    // 2. Register the tasks
    bingo_task_t *task_dummy_0 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_0),0,0);
    bingo_task_t *task_dummy_1 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_1),0,0);
    bingo_task_t *task_dummy_2 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_2),0,0);

    // 3. Set the task dependency
    //      T0 (cluster 0)
    //       |
    //       v
    //      T1 (cluster 0)
    //       |
    //       v
    //      T2 (cluster 0)
    bingo_task_add_depend(task_dummy_2, task_dummy_1);
    bingo_task_add_depend(task_dummy_1, task_dummy_0);
    
    //////////////////////
    // End main function //
    //////////////////////
    // Handle the output parameters
    // 1. Set up the task list
    task_list[0] = task_dummy_0;
    task_list[1] = task_dummy_1;
    task_list[2] = task_dummy_2;
    asm volatile("fence" ::: "memory");
    // 2. Return the number of tasks
    return 3;

}

// Kernel Execution
int kernel_execution(){
    // Set up the tasks list
    // We set a maximum number of tasks to 64
    // Can be changed if needed
    bingo_task_t *task_list[64] = {0};
    uint32_t num_tasks = 0;

    num_tasks = __workload_dummy(task_list);
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