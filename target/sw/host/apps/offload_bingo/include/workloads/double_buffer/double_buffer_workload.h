// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include "double_buffer_data.h"
#include "libbingo/bingo_api.h"
#include "host.h"
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
    check_kernel_tab_ready();
    uint32_t double_buffer_func_addr = get_device_function("__snax_kernel_double_buffer_example");
    uint32_t check_results_func_addr = get_device_function("__snax_kernel_check_results");
    if (double_buffer_func_addr == SNAX_SYMTAB_END_FN_ADDR || check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf("Error: Kernel symbol lookup failed!\r\n");
    }

    // 2. Register the tasks
    ///////////////////////////////////
    /// Double buffer args
    ///////////////////////////////////    
    __snax_kernel_double_buffer_args_t double_buffer_args;
    double_buffer_args.input_data_addr = (uint32_t)(uintptr_t)(&input_array);
    uint32_t* output_data_addr = o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), ARRAY_SIZE_BYTES(input_array));
    if (!output_data_addr) {
        printf("Error: output buffer allocation failed!\r\n");
        return 0;
    }
    double_buffer_args.output_data_addr = (uint32_t)(uintptr_t)(output_data_addr);
    double_buffer_args.data_size = ARRAY_SIZE_BYTES(input_array);
    double_buffer_args.num_tiles = 4;

    bingo_task_t *task_double_buffer = bingo_task_create(double_buffer_func_addr,                   // function ptr
                                                        (uint32_t)(uintptr_t)(&double_buffer_args), // args ptr
                                                        0,                                          // assigned chip id
                                                        0                                           // assigned cluster id
                                                    );
    ///////////////////////////////////
    /// Check results args
    ///////////////////////////////////
    __snax_kernel_check_results_args_t task_check_results_args;
    task_check_results_args.golden_data_addr = (uint32_t)(uintptr_t)(&golden_output_array);
    task_check_results_args.output_data_addr = (uint32_t)(uintptr_t)(output_data_addr);
    task_check_results_args.data_size = ARRAY_SIZE_BYTES(input_array);
    bingo_task_t *task_check_results = bingo_task_create(check_results_func_addr,
                                                        (uint32_t)(uintptr_t)(&task_check_results_args),
                                                        0,
                                                        1
                                                    );

    // 3. Set the task dependency
    //      T0 (cluster 0)
    //       |
    //       v
    //      T1 (cluster 0)
    bingo_task_add_depend(task_check_results, task_double_buffer);

    //////////////////////
    // End main function //
    //////////////////////
    // Handle the output parameters
    // 1. Set up the task list to outside
    task_list[0] = task_double_buffer;
    task_list[1] = task_check_results;
    asm volatile("fence" ::: "memory");
    // task_list[1] = task_dummy_1;
    // task_list[2] = task_dummy_2;
    // User need to define the number of tasks

    int num_tasks = 2;
    return num_tasks;
}

