// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include "libbingo/bingo_api.h"
#include "host.h"
#include "bingo_workload.h"
// Kernel Execution
int kernel_execution(){
    // In bingo_workload.h we have all the task descriptors and the kernel names
    // We need to prepare the following arguments
    // 1. device arg list: list of pointers to the arguments for device kernels
    // 2. device kernel list: list of function pointers for each task
    // 3. host arg list: list of pointers to the arguments for host kernels
    // 4. host kernel list: list of function pointers for each host task
    check_kernel_tab_ready();
    // 1. Prepare device arg list
    uint64_t device_arg_list[num_dev_tasks];
    __snax_bingo_kernel_dummy_args_t* dev_task_dummy;
    dev_task_dummy = (__snax_bingo_kernel_dummy_args_t*)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(__snax_bingo_kernel_dummy_args_t));
    dev_task_dummy->csr_addr = 960;
    dev_task_dummy->csr_value = 99;
    __snax_bingo_kernel_exit_args_t* dev_task_exit;
    dev_task_exit = (__snax_bingo_kernel_exit_args_t*)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(__snax_bingo_kernel_exit_args_t));
    dev_task_exit->exit_code = 0;
    // See the arg reuse here!
    device_arg_list[0] = (uint64_t)(uintptr_t)dev_task_dummy;
    device_arg_list[1] = (uint64_t)(uintptr_t)dev_task_dummy;
    device_arg_list[2] = (uint64_t)(uintptr_t)dev_task_exit;
    device_arg_list[3] = (uint64_t)(uintptr_t)dev_task_exit;
    // 2. Prepare device kernel function list
    uint64_t device_kernel_list[num_dev_tasks];
    uint32_t __snax_bingo_kernel_dummy_func_addr = get_device_function("__snax_bingo_kernel_dummy");
    uint32_t __snax_bingo_kernel_exit_func_addr = get_device_function("__snax_bingo_kernel_exit");
    device_kernel_list[0] = (uint64_t)(uintptr_t)__snax_bingo_kernel_dummy_func_addr;
    device_kernel_list[1] = (uint64_t)(uintptr_t)__snax_bingo_kernel_dummy_func_addr;
    device_kernel_list[2] = (uint64_t)(uintptr_t)__snax_bingo_kernel_exit_func_addr;
    device_kernel_list[3] = (uint64_t)(uintptr_t)__snax_bingo_kernel_exit_func_addr;
    // 3. Prepare host kernel args
    // Prepare the 1st host kernel args: __host_bingo_kernel_dummy
    __host_bingo_kernel_dummy_args_t* host_task_dummy;
    host_task_dummy = (__host_bingo_kernel_dummy_args_t*)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(__host_bingo_kernel_dummy_args_t));
    host_task_dummy->dummy_arg_0 = 42;
    // The other host kernel is the exit kernel
    __host_bingo_kernel_exit_args_t* host_task_exit;
    host_task_exit = (__host_bingo_kernel_exit_args_t*)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(__host_bingo_kernel_exit_args_t));
    host_task_exit->exit_code = 0;
    // In total we have 2 host kernels
    uint64_t host_arg_list[num_host_tasks];
    host_arg_list[0] = (uint64_t)(uintptr_t)host_task_dummy;
    host_arg_list[1] = (uint64_t)(uintptr_t)host_task_exit;
    // 4. Prepare host kernel function list
    uint64_t host_kernel_list[num_host_tasks];
    host_kernel_list[0] = (uint64_t)(uintptr_t)&__host_bingo_kernel_dummy;
    host_kernel_list[1] = (uint64_t)(uintptr_t)&__host_bingo_kernel_exit;

    return 0;
}

