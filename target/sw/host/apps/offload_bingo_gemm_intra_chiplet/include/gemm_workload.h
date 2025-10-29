// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>

#pragma once
#include "gemm_data.h"
#include "libbingo/bingo_api.h"
#include "host.h"

void __workload_versacore(bingo_task_t **task_list, uint32_t *num_tasks_ptr){
    ///////////////////
    // Main function //
    ///////////////////
    // 1. Get the kernel function address by the kernel name and prepare the args
    // 2. Register the tasks
    // 3. Set the task dependency
    // 4. Set the assigned chiplet id and cluster id

    uint8_t current_chip_id = get_current_chip_id();
    uint8_t cluster_id = 0; // versacore is located at cluster

    // 1.1 Get the kernel function address by the kernel name
    check_kernel_tab_ready(); 
    uint32_t check_results_func_addr = get_device_function("__snax_kernel_check_results");
    uint32_t __snax_kernel_gemm_intra_chiplet_func_addr = get_device_function("__snax_kernel_gemm_intra_chiplet");
    if (check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        __snax_kernel_gemm_intra_chiplet_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf("Error: Kernel symbol lookup failed!\r\n");
    }

    // 1.2 Prepare the args
    // versacore args
    uint32_t gemm_args[14];
    // A matrix
    gemm_args[0] = (uint32_t)(((uint64_t)(&A[0]) >> 32) & 0xFFFFFFFF);
    gemm_args[1] = (uint32_t)(((uint64_t)(&A[0]) >> 0) & 0xFFFFFFFF);
    // B matrix
    gemm_args[2] = (uint32_t)(((uint64_t)(&B[0]) >> 32) & 0xFFFFFFFF);
    gemm_args[3] = (uint32_t)(((uint64_t)(&B[0]) >> 0) & 0xFFFFFFFF);
    // C matrix
    gemm_args[4] = (uint32_t)(((uint64_t)(&C[0]) >> 32) & 0xFFFFFFFF);
    gemm_args[5] = (uint32_t)(((uint64_t)(&C[0]) >> 0) & 0xFFFFFFFF);
    // D matrix (output)
    O1HeapInstance *local_l3_heap_manager = bingo_get_l3_heap_manager(current_chip_id);
    uintptr_t output_data_addr = (uintptr_t)o1heapAllocate(local_l3_heap_manager, sizeof(D) * sizeof(D[0]));
    gemm_args[6] = (uint32_t)(((uint64_t)(output_data_addr) >> 32) & 0xFFFFFFFF);
    gemm_args[7] = (uint32_t)(((uint64_t)(output_data_addr) >> 0) & 0xFFFFFFFF);
    // Matrix dimensions
    gemm_args[8]  = M; // M
    gemm_args[9]  = K; // K
    gemm_args[10] = N; // N
    // SUs
    gemm_args[11] = array_shape;
    // transpose A
    gemm_args[12] = transposed_A;
    // transpose B
    gemm_args[13] = transposed_B;
    
    __snax_kernel_gemm_intra_chiplet_args_t gemm_intra_chiplet_args;
    gemm_intra_chiplet_args.args_ptr = (uint32_t)(uintptr_t)&gemm_args;

    // checkresults args
    __snax_kernel_check_results_args_t task_check_results_args;
    task_check_results_args.golden_data_addr = (uint32_t)(uintptr_t)(&D);
    task_check_results_args.output_data_addr = (uint32_t)(uintptr_t)(output_data_addr);
    task_check_results_args.data_size = ARRAY_SIZE_BYTES(D);

    // 2. Register the tasks
    bingo_task_t *task_versacore = bingo_task_create(__snax_kernel_gemm_intra_chiplet_func_addr, (uint32_t)(uintptr_t)(&gemm_intra_chiplet_args), current_chip_id, cluster_id);
    if (task_versacore == NULL) {
        printf("Error: Task versacore creation failed!\r\n");
    }
    bingo_task_t *task_check_results = bingo_task_create(check_results_func_addr, (uint32_t)(uintptr_t)(&task_check_results_args), current_chip_id, cluster_id);
    if (task_check_results == NULL) {
        printf("Error: Task check results creation failed!\r\n");
    }

    // 3. Set the task dependency
    // Here we have only two tasks and simple dependency
    // versacore -> check results
    bingo_task_add_depend(task_check_results, task_versacore);

    // 4. Set the assigned cluster id and chip id
    task_versacore->assigned_cluster_id = 0;
    task_check_results->assigned_cluster_id = 0;

    //////////////////////
    // End main function //
    //////////////////////
    // Handle the output parameters
    // 1. Set up the task list to outside

    task_list[0] = task_versacore;
    task_list[1] = task_check_results;


    // User need to define the number of tasks
    int num_tasks = 2;

    // Since this is a single chiplet workload
    // We set the assigned chiplet id to 0 for all tasks
    for (int i = 0; i < num_tasks; i++)
    {
        task_list[i]->assigned_chip_id = 0;
    }

    // 2. Set the number of tasks to outside
    *num_tasks_ptr = num_tasks;
}

// void __workload_versacore_w_streamer_args(bingo_task_t **task_list, uint32_t *num_tasks_ptr) {
//     ///////////////////
//     // Main function //
//     ///////////////////
//     // 1. Get the kernel function address by the kernel name and prepare the args
//     // 2. Register the tasks
//     // 3. Set the task dependency
//     // 4. Set the assigned chiplet id and cluster id
    
//     uint8_t current_chip_id = get_current_chip_id();
//     uint8_t cluster_id = 0; // versacore is located at cluster 0

//     // 1.1 Get the kernel function address by the kernel name
//     check_kernel_tab_ready(); 
//     uint32_t check_results_func_addr = get_device_function("__snax_kernel_check_results");
//     uint32_t versacore_load_compute_store_func_addr = get_device_function("__snax_kernel_versacore_load_compute_store_w_streamer_args");
//     if (check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
//         versacore_load_compute_store_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
//         printf("Error: Kernel symbol lookup failed!\r\n");
//     }

//     // 1.2 Prepare the args
//     ///////////////////////////////////
//     /// versacore args
//     ///////////////////////////////////
//     __snax_kernel_versacore_load_compute_store_args_t versacore_load_compute_store_args;
//     // Inputs
//     versacore_load_compute_store_args.input_A_addr_hi = (uint32_t)(((uint64_t)(&A[0]) >> 32) & 0xFFFFFFFF);
//     versacore_load_compute_store_args.input_A_addr_lo = (uint32_t)(((uint64_t)(&A[0]) >> 0) & 0xFFFFFFFF);
//     versacore_load_compute_store_args.input_A_size = ARRAY_SIZE_BYTES(A);
//     versacore_load_compute_store_args.input_B_addr_hi = (uint32_t)(((uint64_t)(&B[0]) >> 32) & 0xFFFFFFFF);
//     versacore_load_compute_store_args.input_B_addr_lo = (uint32_t)(((uint64_t)(&B[0]) >> 0) & 0xFFFFFFFF);
//     versacore_load_compute_store_args.input_B_size = ARRAY_SIZE_BYTES(B);
//     versacore_load_compute_store_args.input_C_addr_hi = (uint32_t)(((uint64_t)(&C[0]) >> 32) & 0xFFFFFFFF);
//     versacore_load_compute_store_args.input_C_addr_lo = (uint32_t)(((uint64_t)(&C[0]) >> 0) & 0xFFFFFFFF);
//     versacore_load_compute_store_args.input_C_size = ARRAY_SIZE_BYTES(C);
//     // Outputs
//     O1HeapInstance *local_l3_heap_manager = bingo_get_l3_heap_manager(current_chip_id);

//     uintptr_t output_data_addr = (uintptr_t)o1heapAllocate(local_l3_heap_manager, sizeof(D) * sizeof(D[0]));
//     versacore_load_compute_store_args.output_addr_hi = (uint32_t)(((uint64_t)(output_data_addr) >> 32) & 0xFFFFFFFF);
//     versacore_load_compute_store_args.output_addr_lo = (uint32_t)(((uint64_t)(output_data_addr) >> 0) & 0xFFFFFFFF);
//     // Streamer arguments
//     int32_t Aslstride[] = {Aslstride0};
//     int32_t Atlbound[] = {Atlbound0, Atlbound1, Atlbound2,
//                           Atlbound3, Atlbound4, Atlbound5};
//     int32_t Atlstride[] = {Atlstride0, Atlstride1, Atlstride2,
//                            Atlstride3, Atlstride4, Atlstride5};
//     int32_t Bslstride[] = {Bslstride0};
//     int32_t Btlbound[] = {Btlbound0, Btlbound1, Btlbound2};
//     int32_t Btlstride[] = {Btlstride0, Btlstride1, Btlstride2};

//     int32_t Cslstride[] = {Cslstride0};
//     int32_t Ctlbound[] = {Ctlbound0, Ctlbound1, Ctlbound2, Ctlbound3};
//     int32_t Ctlstride[] = {Ctlstride0, Ctlstride1, Ctlstride2, Ctlstride3};

//     int32_t D32slstride[] = {D32slstride0};
//     int32_t D32tlbound[] = {D32tlbound0, D32tlbound1, D32tlbound2, D32tlbound3};
//     int32_t D32tlstride[] = {D32tlstride0, D32tlstride1, D32tlstride2,
//                              D32tlstride3};    
//     uint32_t streamer_cfg[] = {
//         // A
//         (uint32_t)(uintptr_t)&Aslstride[0],
//         (uint32_t)(uintptr_t)&Atlbound[0],
//         (uint32_t)(uintptr_t)&Atlstride[0],
//         (uint32_t)set_addr_remap_index_A,
//         (uint32_t)transposed_A,
//         (uint32_t)(uintptr_t)&channel_en_A[0],
//         // B
//         (uint32_t)(uintptr_t)&Bslstride[0],
//         (uint32_t)(uintptr_t)&Btlbound[0],
//         (uint32_t)(uintptr_t)&Btlstride[0],
//         (uint32_t)set_addr_remap_index_B,
//         (uint32_t)transposed_B,
//         (uint32_t)(uintptr_t)&channel_en_B[0],
//         // C
//         (uint32_t)(uintptr_t)&Cslstride[0],
//         (uint32_t)(uintptr_t)&Ctlbound[0],
//         (uint32_t)(uintptr_t)&Ctlstride[0],
//         (uint32_t)set_addr_remap_index_C,
//         (uint32_t)(uintptr_t)&channel_en_C[0],
//         // D
//         (uint32_t)(uintptr_t)&D32slstride[0],
//         (uint32_t)(uintptr_t)&D32tlbound[0],
//         (uint32_t)(uintptr_t)&D32tlstride[0],
//         (uint32_t)set_addr_remap_index_D32,
//         (uint32_t)(uintptr_t)&channel_en_D[0]
//     };
//     versacore_load_compute_store_args.streamer_cfg_addr_hi = (uint32_t)(((uint64_t)(&streamer_cfg[0]) >> 32) & 0xFFFFFFFF);
//     versacore_load_compute_store_args.streamer_cfg_addr_lo = (uint32_t)(((uint64_t)(&streamer_cfg[0]) >> 0) & 0xFFFFFFFF);
//     versacore_load_compute_store_args.streamer_cfg_size = ARRAY_SIZE_BYTES(streamer_cfg);
//     // Versacore configuration arguments
//     uint32_t versacore_cfg[] = {
//         K,
//         N,
//         M,
//         subtraction_a,
//         subtraction_b,
//         array_shape,
//         data_type
//     };
//     versacore_load_compute_store_args.versacore_cfg_addr_hi = (uint32_t)(((uint64_t)(&versacore_cfg[0]) >> 32) & 0xFFFFFFFF);
//     versacore_load_compute_store_args.versacore_cfg_addr_lo = (uint32_t)(((uint64_t)(&versacore_cfg[0]) >> 0) & 0xFFFFFFFF);
//     versacore_load_compute_store_args.versacore_cfg_size = ARRAY_SIZE_BYTES(versacore_cfg);
//     // Total arg length
//     versacore_load_compute_store_args.total_arg_length = sizeof(__snax_kernel_versacore_load_compute_store_args_t);

//     ///////////////////////////////////
//     /// Check results args
//     ///////////////////////////////////
//     __snax_kernel_check_results_args_t task_check_results_args;
//     task_check_results_args.golden_data_addr = (uint32_t)(uintptr_t)(&D);
//     task_check_results_args.output_data_addr = (uint32_t)(uintptr_t)(output_data_addr);
//     task_check_results_args.data_size = ARRAY_SIZE_BYTES(D);

//     // 2. Register the tasks
//     bingo_task_t *task_versacore = bingo_task_create(versacore_load_compute_store_func_addr, (uint32_t)(uintptr_t)(&versacore_load_compute_store_args), current_chip_id, cluster_id);
//     if (task_versacore == NULL) {
//         printf("Error: Task versacore creation failed!\r\n");
//     }
//     bingo_task_t *task_check_results = bingo_task_create(check_results_func_addr, (uint32_t)(uintptr_t)(&task_check_results_args), current_chip_id, cluster_id);
//     if (task_check_results == NULL) {
//         printf("Error: Task check results creation failed!\r\n");
//     }

//     // 3. Set the task dependency
//     // Here we have only two tasks and simple dependency
//     // versacore -> check results
//     bingo_task_add_depend(task_check_results, task_versacore);

//     // 4. Set the assigned cluster id and chip id
//     task_versacore->assigned_cluster_id = 0;
//     task_check_results->assigned_cluster_id = 0;

//     //////////////////////
//     // End main function //
//     //////////////////////
//     // Handle the output parameters
//     // 1. Set up the task list to outside

//     task_list[0] = task_versacore;
//     task_list[1] = task_check_results;


//     // User need to define the number of tasks
//     int num_tasks = 2;

//     // Since this is a single chiplet workload
//     // We set the assigned chiplet id to 0 for all tasks
//     for (int i = 0; i < num_tasks; i++)
//     {
//         task_list[i]->assigned_chip_id = 0;
//     }

//     // 2. Set the number of tasks to outside
//     *num_tasks_ptr = num_tasks;
// }
