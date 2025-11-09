// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include "dummy_data.h"
#include "libbingo/bingo_api.h"
#include "host.h"
uint32_t __workload_dummy(bingo_task_t **task_list) {

    // This is a dummy workload for testing
    // It creates two tasks on two different clusters

    ///////////////////
    // Main function //
    ///////////////////
    // 1. Get the kernel function address by the kernel name and prepare the args
    // 2. Register the tasks
    // 3. Set the task dependency

    // 1.Get the kernel function address by the kernel name
    check_kernel_tab_ready();
    uint32_t dummy_func_addr = get_device_function("__snax_kernel_dummy");
    if (dummy_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf("Error: Kernel symbol lookup failed!\r\n");
    }

    // 2. Register the tasks
    bingo_task_t *task_dummy_0 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_0),0,0);
    bingo_task_t *task_dummy_1 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_1),0,1);
    bingo_task_t *task_dummy_2 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_0),0,0);

    // 3. Set the task dependency
    //      T0 (cluster 0)
    //       |
    //       v
    //      T1 (cluster 1)
    //       |
    //       v
    //      T2 (cluster 0)
    bingo_task_add_depend(task_dummy_2, task_dummy_1);
    bingo_task_add_depend(task_dummy_1, task_dummy_0);
    
    //////////////////////
    // End main function //
    //////////////////////
    // Handle the output parameters
    // 1. Set up the task list to outside
    task_list[0] = task_dummy_0;
    task_list[1] = task_dummy_1;
    task_list[2] = task_dummy_2;
    asm volatile("fence" ::: "memory");
    // User need to define the number of tasks
    int num_tasks = 3;
    return num_tasks;
}

