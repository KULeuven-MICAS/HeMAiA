// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>

#pragma once
#include "versacore_data.h"
#include "libbingo/bingo_api.h"
#include "host.h"

uint32_t __workload_versacore(bingo_task_t **task_list, uintptr_t *output_data_ptr) {
    ///////////////////
    // Main function //
    ///////////////////
    // 1. Get the kernel function address by the kernel name and prepare the args
    // 2. Register the tasks
    // 3. Set the task dependency
    // 4. Set the assigned chiplet id and cluster id
    
    // 1.1 Get the kernel function address by the kernel name
    check_kernel_tab_ready(); 
    uint32_t check_results_func_addr = get_device_function("__snax_kernel_check_results");
    uint32_t versacore_load_compute_store_func_addr = get_device_function("__snax_kernel_versacore_load_compute_store");
    if (check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        versacore_load_compute_store_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf("Error: Kernel symbol lookup failed!\r\n");
    }
    // 1.2 Prepare the args
    ///////////////////////////////////
    /// versacore args
    ///////////////////////////////////
    __snax_kernel_versacore_load_compute_store_args_t versacore_load_compute_store_args;
    // Inputs
    versacore_load_compute_store_args.input_A_addr_hi = HIGH32(&A[0]);
    versacore_load_compute_store_args.input_A_addr_lo = LOW32(&A[0]);
    versacore_load_compute_store_args.input_A_size = ARRAY_SIZE_BYTES(A);
    versacore_load_compute_store_args.input_B_addr_hi = HIGH32(&B[0]);
    versacore_load_compute_store_args.input_B_addr_lo = LOW32(&B[0]);
    versacore_load_compute_store_args.input_B_size = ARRAY_SIZE_BYTES(B);
    versacore_load_compute_store_args.input_C_addr_hi = HIGH32(&C[0]);
    versacore_load_compute_store_args.input_C_addr_lo = LOW32(&C[0]);
    versacore_load_compute_store_args.input_C_size = ARRAY_SIZE_BYTES(C);
    // Outputs
    void *output_data_addr = o1heapAllocate(
        bingo_get_l3_heap_manager(get_current_chip_id()),
        ARRAY_SIZE_BYTES(D));     // total bytes of D
    if (!output_data_addr) {
        printf("Error: output buffer allocation failed!\r\n");
        return 0;
    }
    *output_data_ptr = (uintptr_t)output_data_addr;
    versacore_load_compute_store_args.output_addr_hi = HIGH32(output_data_addr);
    versacore_load_compute_store_args.output_addr_lo = LOW32(output_data_addr);
    // Streamer arguments
    int32_t Aslstride[] = {Aslstride0};
    int32_t Atlbound[] = {Atlbound0, Atlbound1, Atlbound2,
                          Atlbound3, Atlbound4, Atlbound5};
    int32_t Atlstride[] = {Atlstride0, Atlstride1, Atlstride2,
                           Atlstride3, Atlstride4, Atlstride5};
    int32_t Bslstride[] = {Bslstride0};
    int32_t Btlbound[] = {Btlbound0, Btlbound1, Btlbound2};
    int32_t Btlstride[] = {Btlstride0, Btlstride1, Btlstride2};

    int32_t Cslstride[] = {Cslstride0};
    int32_t Ctlbound[] = {Ctlbound0, Ctlbound1, Ctlbound2, Ctlbound3};
    int32_t Ctlstride[] = {Ctlstride0, Ctlstride1, Ctlstride2, Ctlstride3};

    int32_t D32slstride[] = {D32slstride0};
    int32_t D32tlbound[] = {D32tlbound0, D32tlbound1, D32tlbound2, D32tlbound3};
    int32_t D32tlstride[] = {D32tlstride0, D32tlstride1, D32tlstride2,
                             D32tlstride3};    
    uint32_t streamer_cfg[] = {
        // A
        (uint32_t)(uintptr_t)&Aslstride[0],
        (uint32_t)(uintptr_t)&Atlbound[0],
        (uint32_t)(uintptr_t)&Atlstride[0],
        (uint32_t)set_addr_remap_index_A,
        (uint32_t)transposed_A,
        (uint32_t)(uintptr_t)&channel_en_A[0],
        // B
        (uint32_t)(uintptr_t)&Bslstride[0],
        (uint32_t)(uintptr_t)&Btlbound[0],
        (uint32_t)(uintptr_t)&Btlstride[0],
        (uint32_t)set_addr_remap_index_B,
        (uint32_t)transposed_B,
        (uint32_t)(uintptr_t)&channel_en_B[0],
        // C
        (uint32_t)(uintptr_t)&Cslstride[0],
        (uint32_t)(uintptr_t)&Ctlbound[0],
        (uint32_t)(uintptr_t)&Ctlstride[0],
        (uint32_t)set_addr_remap_index_C,
        (uint32_t)(uintptr_t)&channel_en_C[0],
        // D
        (uint32_t)(uintptr_t)&D32slstride[0],
        (uint32_t)(uintptr_t)&D32tlbound[0],
        (uint32_t)(uintptr_t)&D32tlstride[0],
        (uint32_t)set_addr_remap_index_D32,
        (uint32_t)(uintptr_t)&channel_en_D[0]
    };
    versacore_load_compute_store_args.streamer_cfg_addr_hi = HIGH32(&streamer_cfg[0]);
    versacore_load_compute_store_args.streamer_cfg_addr_lo = LOW32(&streamer_cfg[0]);
    versacore_load_compute_store_args.streamer_cfg_size = ARRAY_SIZE_BYTES(streamer_cfg);
    // Versacore configuration arguments
    uint32_t versacore_cfg[] = {
        K,
        N,
        M,
        subtraction_a,
        subtraction_b,
        array_shape,
        data_type
    };
    versacore_load_compute_store_args.versacore_cfg_addr_hi = HIGH32(&versacore_cfg[0]);
    versacore_load_compute_store_args.versacore_cfg_addr_lo = LOW32(&versacore_cfg[0]);
    versacore_load_compute_store_args.versacore_cfg_size = ARRAY_SIZE_BYTES(versacore_cfg);
    // Total arg length
    versacore_load_compute_store_args.total_arg_length = sizeof(__snax_kernel_versacore_load_compute_store_args_t);

    ///////////////////////////////////
    /// Check results args
    ///////////////////////////////////
    __snax_kernel_check_results_args_t task_check_results_args;
    task_check_results_args.golden_data_addr = (uint32_t)(uintptr_t)(&D);
    task_check_results_args.output_data_addr = (uint32_t)(uintptr_t)(output_data_addr);
    task_check_results_args.data_size = ARRAY_SIZE_BYTES(D);

    // 2. Register the tasks
    bingo_task_t *task_versacore = bingo_task_create(versacore_load_compute_store_func_addr, (uint32_t)(uintptr_t)(&versacore_load_compute_store_args), chip_loc_to_chip_id(0,0), 0);
    if (task_versacore == NULL) {
        printf("Error: Task versacore creation failed!\r\n");
    }
    bingo_task_t *task_check_results = bingo_task_create(check_results_func_addr, (uint32_t)(uintptr_t)(&task_check_results_args), chip_loc_to_chip_id(0,0), 0);
    if (task_check_results == NULL) {
        printf("Error: Task check results creation failed!\r\n");
    }
    // 3. Set the task dependency
    // Here we have only two tasks and simple dependency
    // versacore -> check results
    bingo_task_add_depend(task_check_results, task_versacore);
    //////////////////////
    // End main function //
    //////////////////////
    // Handle the output parameters
    // 1. Set up the task list to outside

    task_list[0] = task_versacore;
    task_list[1] = task_check_results;


    // User need to define the number of tasks
    int num_tasks = 2;

    // 2. Set the number of tasks to outside
    return num_tasks;
}