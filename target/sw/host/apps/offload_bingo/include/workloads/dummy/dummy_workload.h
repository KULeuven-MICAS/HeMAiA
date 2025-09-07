// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#include "dummy_data.h"
#include "libbingo/bingo_api.h"
#include "host.h"

void __workload_dummy(bingo_task_t **task_list, uint32_t *num_tasks_ptr) {
    // User need to define the number of tasks
    int num_tasks = 2;
    // This is a dummy workload for testing
    // It creates two tasks on two different clusters

    ///////////////////
    // Main function //
    ///////////////////
    // 1. Get the kernel function address by the kernel name
    // 2. Register the tasks
    // 3. Set the task dependency
    // 4. Set the assigned chiplet id and cluster id

    // 1. Get the kernel function address by the kernel name
    check_kernel_tab_ready();
    uint32_t dummy_func_addr = get_device_function("__snax_kernel_dummy");
    if (dummy_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf("Error: Kernel symbol lookup failed!\r\n");
    }

    // 2. Register the tasks
    bingo_task_t *task_dummy_cluster0 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_cluster0));
    bingo_task_t *task_dummy_cluster1 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_cluster1));

    // 3. Set the task dependency
    //      cl0
    //       |
    //       v
    //      cl1
    // So the cl0 will start to run first, then cl1
    bingo_task_add_depend(task_dummy_cluster1, task_dummy_cluster0);

    // 4. Set the assigned cluster id
    task_dummy_cluster0->assigned_cluster_id = 0;
    task_dummy_cluster1->assigned_cluster_id = 1;

    //////////////////////
    // End main function //
    //////////////////////
    // Handle the output parameters
    // 1. Set up the task list to outside
    task_list[0] = task_dummy_cluster0;
    task_list[1] = task_dummy_cluster1;
    // Since this is a single chiplet workload
    // We set the assigned chiplet id to 0 for all tasks
    for (int i = 0; i < num_tasks; i++)
    {
        task_list[i]->assigned_chip_id = 0;
    }
    // 2. Set the number of tasks to outside
    *num_tasks_ptr = num_tasks;
}

