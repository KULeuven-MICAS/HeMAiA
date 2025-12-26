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
int kernel_execution(volatile comm_buffer_t* comm_buffer_ptr){
    volatile uint32_t* shared_lock = get_shared_lock(comm_buffer_ptr);
    // In bingo_workload.h we have all the task descriptors and the kernel names
    // We need to prepare the following arguments
    // 1. device arg list: list of pointers to the arguments for device kernels
    // 2. device kernel list: list of function pointers for each task
    // 3. host arg list: list of pointers to the arguments for host kernels
    // 4. host kernel list: list of function pointers for each host task
    check_kernel_tab_ready();
    // 1. Prepare device arg list
    uint32_t device_arg_list[num_dev_tasks];
    __snax_bingo_kernel_dummy_args_t* dev_task_dummy;
    dev_task_dummy = (__snax_bingo_kernel_dummy_args_t*)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(__snax_bingo_kernel_dummy_args_t));
    dev_task_dummy->dummy_input = 99;
    __snax_bingo_kernel_exit_args_t* dev_task_exit;
    dev_task_exit = (__snax_bingo_kernel_exit_args_t*)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(__snax_bingo_kernel_exit_args_t));
    dev_task_exit->exit_code = 0;
    // See the arg reuse here!
    device_arg_list[0] = (uint32_t)(uintptr_t)dev_task_dummy;
    device_arg_list[1] = (uint32_t)(uintptr_t)dev_task_dummy;
    device_arg_list[2] = (uint32_t)(uintptr_t)dev_task_exit;
    device_arg_list[3] = (uint32_t)(uintptr_t)dev_task_exit;
    // 2. Prepare device kernel function list
    uint32_t device_kernel_list[num_dev_tasks];
    uint32_t __snax_bingo_kernel_dummy_func_addr = get_device_function("__snax_bingo_kernel_dummy");
    uint32_t __snax_bingo_kernel_exit_func_addr = get_device_function("__snax_bingo_kernel_exit");
    device_kernel_list[0] = (uint32_t)(uintptr_t)__snax_bingo_kernel_dummy_func_addr;
    device_kernel_list[1] = (uint32_t)(uintptr_t)__snax_bingo_kernel_dummy_func_addr;
    device_kernel_list[2] = (uint32_t)(uintptr_t)__snax_bingo_kernel_exit_func_addr;
    device_kernel_list[3] = (uint32_t)(uintptr_t)__snax_bingo_kernel_exit_func_addr;
    // 3. Prepare host kernel args
    // Prepare the 1st host kernel args: __host_bingo_kernel_dummy
    __host_bingo_kernel_dummy_args_t* host_task_dummy;
    host_task_dummy = (__host_bingo_kernel_dummy_args_t*)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(__host_bingo_kernel_dummy_args_t));
    host_task_dummy->dummy_arg_0 = 42;
    host_task_dummy->global_mutex_ptr = (uint64_t)(uintptr_t)shared_lock;
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
    mutex_tas_acquire(shared_lock);
    printf("Chip(%x, %x): [Host] Init HW Bingo Scheduler\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y());
    mutex_release(shared_lock);
    // Notice here we pass the device arg/kernel lists and other info to initialize the HW scheduler
    bingo_hw_scheduler_init((uint32_t)(uintptr_t)&device_arg_list[0],
                            (uint32_t)(uintptr_t)&device_kernel_list[0],
                            (uint32_t)(uintptr_t)&global_task_id_to_dev_task_id[0],
                            (uint64_t)(uintptr_t)&task_desc_list[0],
                            num_tasks,
                            comm_buffer_ptr);
    // Start the HW scheduler
    uint32_t err = bingo_hw_scheduler(&host_arg_list[0],
                                      &host_kernel_list[0],
                                      &global_task_id_to_host_task_id[0],
                                      comm_buffer_ptr);
    // Debug to make the host wfi
    wfi();
    return 0;
}

