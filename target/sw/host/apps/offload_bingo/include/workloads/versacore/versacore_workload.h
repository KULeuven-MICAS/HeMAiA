// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>

#pragma once
#include "data.h"
#include "libbingo/bingo_api.h"
#include "host.h"
#include "device_kernel_args.h"

// void __workload_versacore(bingo_task_t **task_list, uint32_t *num_tasks_ptr, uintptr_t *output_data_ptr) {

//     // task 1
//     // move data A from L3 to cluster 0
//     // task 2
//     // move data B from L3 to cluster 0
//     // task 3
//     // move data C from L3 to cluster 0
//     // task 4
//     // configure the versacore streamer and luanch
//     // task 5
//     // configure the versacore and luanch
//     // task 6
//     // wait the streamer/versacore to finish
//     // task 7
//     // move data D from cluster 0 to L3
//     // task 8
//     // check the results

//     ///////////////////
//     // Main function //
//     ///////////////////
//     // 1. Get the kernel function address by the kernel name and prepare the args
//     // 2. Register the tasks
//     // 3. Set the task dependency
//     // 4. Set the assigned chiplet id and cluster id

//     // 1.1 Get the kernel function address by the kernel name
//     check_kernel_tab_ready();
//     uint32_t xdma_1d_copy_func_addr = get_device_function("__snax_kernel_xdma_1d_copy");
//     uint32_t check_results_func_addr = get_device_function("__snax_kernel_check_results");
//     if (xdma_1d_copy_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
//         check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
//         printf("Error: Kernel symbol lookup failed!\r\n");
//     }
//     uint32_t versacore_streamer_cfg_func_addr = get_device_function("__snax_kernel_versacore_streamer_cfg");
//     if (versacore_streamer_cfg_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
//         printf("Error: Kernel symbol lookup failed!\r\n");
//     }
//     uint32_t versacore_cfg_func_addr = get_device_function("__snax_kernel_versacore_cfg");
//     if (versacore_cfg_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
//         printf("Error: Kernel symbol lookup failed!\r\n");
//     }
//     uint32_t streamer_and_versacore_wait_func_addr = get_device_function("__snax_kernel_streamer_and_versacore_wait");
//     if (streamer_and_versacore_wait_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
//         printf("Error: Kernel symbol lookup failed!\r\n");
//     }

//     // 1.2 Prepare the args
//     ///////////////////////////////////
//     /// versacore args
//     ///////////////////////////////////
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

//     // Args for the versacore task
//     // Arg0: int32_t delta_local_a
//     // Arg1: int32_t* Aslstride
//     // Arg2: int32_t* Atlbound
//     // Arg3: int32_t* Atlstride
//     // Arg4: int32_t set_addr_remap_index_A
//     // Arg5: int32_t transposed_A
//     // Arg6: int32_t* channel_en_A

//     uint32_t a_streamer_cfg_addr[] = {
//         delta_local_a, (uint32_t)(uintptr_t)&Aslstride[0], (uint32_t)(uintptr_t)&Atlbound[0], (uint32_t)(uintptr_t)&Atlstride[0], (uint32_t)set_addr_remap_index_A, (uint32_t)transposed_A, (uint32_t)(uintptr_t)&channel_en_A[0]
//     };

//     // Arg7: int32_t delta_local_b
//     // Arg8: int32_t* Bslstride
//     // Arg9: int32_t* Btlbound
//     // Arg10: int32_t* Btlstride
//     // Arg11: int32_t set_addr_remap_index_B
//     // Arg12: int32_t transposed_B
//     // Arg13: int32_t* channel_en_B

//     uint32_t b_streamer_cfg_addr[] = {
//         delta_local_b, (uint32_t)(uintptr_t)&Bslstride[0], (uint32_t)(uintptr_t)&Btlbound[0], (uint32_t)(uintptr_t)&Btlstride[0], (uint32_t)set_addr_remap_index_B, (uint32_t)transposed_B, (uint32_t)(uintptr_t)&channel_en_B[0]
//     };

//     // Arg14: int32_t delta_local_c
//     // Arg15: int32_t* Cslstride
//     // Arg16: int32_t* Ctlbound
//     // Arg17: int32_t* Ctlstride
//     // Arg18: int32_t set_addr_remap_index_C
//     // Arg19: int32_t* channel_en_C

//     uint32_t c_streamer_cfg_addr[] = {
//         delta_local_c, (uint32_t)(uintptr_t)&Cslstride[0], (uint32_t)(uintptr_t)&Ctlbound[0], (uint32_t)(uintptr_t)&Ctlstride[0], (uint32_t)set_addr_remap_index_C, (uint32_t)(uintptr_t)&channel_en_C[0]
//     };

//     // Arg20: int32_t delta_local_d32
//     // Arg21: int32_t* D32slstride
//     // Arg22: int32_t* D32tlbound
//     // Arg23: int32_t* D32tlstride
//     // Arg24: int32_t set_addr_remap_index_D32
//     // Arg25: int32_t* channel_en_D

//     uint32_t d32_streamer_cfg_addr[] = {
//         delta_local_d, (uint32_t)(uintptr_t)&D32slstride[0], (uint32_t)(uintptr_t)&D32tlbound[0], (uint32_t)(uintptr_t)&D32tlstride[0], (uint32_t)set_addr_remap_index_D32, (uint32_t)(uintptr_t)&channel_en_D[0]
//     };

//     __snax_kernel_versacore_streamer_cfg_args_t task_versacore_args;

//     task_versacore_args.a_streamer_cfg_addr = (uint32_t)(uintptr_t)a_streamer_cfg_addr;
//     task_versacore_args.b_streamer_cfg_addr = (uint32_t)(uintptr_t)b_streamer_cfg_addr;
//     task_versacore_args.c_streamer_cfg_addr = (uint32_t)(uintptr_t)c_streamer_cfg_addr;
//     task_versacore_args.d_streamer_cfg_addr = (uint32_t)(uintptr_t)d32_streamer_cfg_addr;

//     __snax_kernel_versacore_cfg_args_t task_versacore_cfg_args;
//     task_versacore_cfg_args.tempLoop0 = K;
//     task_versacore_cfg_args.tempLoop1 = N;
//     task_versacore_cfg_args.tempLoop2 = M;
//     task_versacore_cfg_args.subtraction_a = subtraction_a;
//     task_versacore_cfg_args.subtraction_b = subtraction_b;
//     task_versacore_cfg_args.array_shape = array_shape;
//     task_versacore_cfg_args.data_type = data_type;

//     ///////////////////////////////////
//     ///// xdma ////
//     ///////////////////////////////////
//     // Args for the xdma task

//     // XDMA task to move A from L3 to cluster 0
//     __snax_kernel_xdma_1d_copy_args_t task_xdma_args_a;
//     task_xdma_args_a.src_addr_hi = (uint32_t)(((uint64_t)(&A[0]) >> 32) & 0xFFFFFFFF);
//     task_xdma_args_a.src_addr_lo = (uint32_t)(((uint64_t)(&A[0]) >> 0) & 0xFFFFFFFF);
//     task_xdma_args_a.dst_addr_hi = 0; // L1 address, so high part is 0
//     task_xdma_args_a.dst_addr_lo = (uint32_t)(cluster_tcdm_start_addr(0) + delta_local_a);
//     task_xdma_args_a.size = sizeof(A) * sizeof(A[0]);

//     // XDMA task to move B from L3 to cluster 0
//     __snax_kernel_xdma_1d_copy_args_t task_xdma_args_b;
//     task_xdma_args_b.src_addr_hi = (uint32_t)(((uint64_t)(&B[0]) >> 32) & 0xFFFFFFFF);
//     task_xdma_args_b.src_addr_lo = (uint32_t)(((uint64_t)(&B[0]) >> 0) & 0xFFFFFFFF);
//     task_xdma_args_b.dst_addr_hi = 0; // L1 address, so high part is 0
//     task_xdma_args_b.dst_addr_lo = (uint32_t)(cluster_tcdm_start_addr(0) + delta_local_b);
//     task_xdma_args_b.size = sizeof(B) * sizeof(B[0]);

//     // XDMA task to move C from L3 to cluster 0
//     __snax_kernel_xdma_1d_copy_args_t task_xdma_args_c;
//     task_xdma_args_c.src_addr_hi = (uint32_t)(((uint64_t)(&C[0]) >> 32) & 0xFFFFFFFF);
//     task_xdma_args_c.src_addr_lo = (uint32_t)(((uint64_t)(&C[0]) >> 0) & 0xFFFFFFFF);
//     task_xdma_args_c.dst_addr_hi = 0; // L1 address, so high part is 0
//     task_xdma_args_c.dst_addr_lo = (uint32_t)(cluster_tcdm_start_addr(0) + delta_local_c);
//     task_xdma_args_c.size = sizeof(C) * sizeof(C[0]);

//     // XDMA task to move D from cluster 0 to L3
//     __snax_kernel_xdma_1d_copy_args_t task_xdma_args_d;
//     task_xdma_args_d.src_addr_hi = 0; // L1 address, so high part is 0
//     task_xdma_args_d.src_addr_lo = (uint32_t)(cluster_tcdm_start_addr(0) + delta_local_d);
//     uintptr_t output_data_addr = hero_host_l3_malloc(sizeof(D) * sizeof(D[0]), output_data_ptr);
//     task_xdma_args_d.dst_addr_hi = (uint32_t)(((uint64_t)(output_data_addr) >> 32) & 0xFFFFFFFF);
//     task_xdma_args_d.dst_addr_lo = (uint32_t)(((uint64_t)(output_data_addr) >> 0) & 0xFFFFFFFF);
//     task_xdma_args_d.size = sizeof(D) * sizeof(D[0]);

//     // Args for the check results task
//     __snax_kernel_check_results_args_t task_check_results_args;
//     task_check_results_args.golden_data_addr = (uint32_t)(uintptr_t)(&D);
//     task_check_results_args.output_data_addr = (uint32_t)(uintptr_t)(output_data_addr);
//     task_check_results_args.data_size = sizeof(D) * sizeof(D[0]);

//     // 2. Register the tasks
//     bingo_task_t *task_move_a = bingo_task_create(xdma_1d_copy_func_addr, (uint32_t)(uintptr_t)(&task_xdma_args_a));
//     if (task_move_a == NULL) {
//         printf("Error: Task move A creation failed!\r\n");
//     }
//     bingo_task_t *task_move_b = bingo_task_create(xdma_1d_copy_func_addr, (uint32_t)(uintptr_t)(&task_xdma_args_b));
//     if (task_move_b == NULL) {
//         printf("Error: Task move B creation failed!\r\n");
//     }
//     bingo_task_t *task_move_c = bingo_task_create(xdma_1d_copy_func_addr, (uint32_t)(uintptr_t)(&task_xdma_args_c));
//     if (task_move_c == NULL) {
//         printf("Error: Task move C creation failed!\r\n");
//     }

//     bingo_task_t *task_versacore_streamer_cfg = bingo_task_create(versacore_streamer_cfg_func_addr, (uint32_t)(uintptr_t)(&task_versacore_args));
//     if (task_versacore_streamer_cfg == NULL) {
//         printf("Error: Task versacore streamer cfg creation failed!\r\n");
//     }
//     bingo_task_t *task_versacore_cfg = bingo_task_create(versacore_cfg_func_addr, (uint32_t)(uintptr_t)(&task_versacore_cfg_args));
//     if (task_versacore_cfg == NULL) {
//         printf("Error: Task versacore cfg creation failed!\r\n");
//     }

//     bingo_task_t *task_streamer_and_versacore_wait = bingo_task_create(streamer_and_versacore_wait_func_addr, 0);
//     if (task_streamer_and_versacore_wait == NULL) {
//         printf("Error: Task streamer and versacore wait creation failed!\r\n");
//     }

//     bingo_task_t *task_move_d = bingo_task_create(xdma_1d_copy_func_addr, (uint32_t)(uintptr_t)(&task_xdma_args_d));
//     if (task_move_d == NULL) {
//         printf("Error: Task move D creation failed!\r\n");
//     }

//     bingo_task_t *task_check_results = bingo_task_create(check_results_func_addr, (uint32_t)(uintptr_t)(&task_check_results_args));
//     if (task_check_results == NULL) {
//         printf("Error: Task check results creation failed!\r\n");
//     }

//     // 3. Set the task dependency
//     // move A, B, C -> versacore streamer cfg -> 
//     //                          versacore cfg -> wait -> move D -> check results
//     bingo_task_add_depend(task_check_results, task_move_d);
//     bingo_task_add_depend(task_move_d, task_streamer_and_versacore_wait);
//     bingo_task_add_depend(task_streamer_and_versacore_wait, task_versacore_cfg);
//     bingo_task_add_depend(task_streamer_and_versacore_wait, task_versacore_streamer_cfg);
//     bingo_task_add_depend(task_versacore_streamer_cfg, task_move_c);
//     bingo_task_add_depend(task_move_c, task_move_b);
//     bingo_task_add_depend(task_move_b, task_move_a);


//     // 4. Set the assigned cluster id and chip id
//     task_move_a->assigned_cluster_id = 0;
//     task_move_b->assigned_cluster_id = 0;
//     task_move_c->assigned_cluster_id = 0;
//     task_versacore_streamer_cfg->assigned_cluster_id = 0;
//     task_versacore_cfg->assigned_cluster_id = 0;
//     task_streamer_and_versacore_wait->assigned_cluster_id = 0;
//     task_move_d->assigned_cluster_id = 0;
//     task_check_results->assigned_cluster_id = 0;

//     //////////////////////
//     // End main function //
//     //////////////////////
//     // Handle the output parameters
//     // 1. Set up the task list to outside

//     task_list[0] = task_move_a;
//     task_list[1] = task_move_b;
//     task_list[2] = task_move_c;
//     task_list[3] = task_versacore_streamer_cfg;
//     task_list[4] = task_versacore_cfg;
//     task_list[5] = task_streamer_and_versacore_wait;
//     task_list[6] = task_move_d;
//     task_list[7] = task_check_results;

//     // User need to define the number of tasks
//     int num_tasks = 8;

//     // Since this is a single chiplet workload
//     // We set the assigned chiplet id to 0 for all tasks
//     for (int i = 0; i < num_tasks; i++)
//     {
//         task_list[i]->assigned_chip_id = 0;
//     }

//     // 2. Set the number of tasks to outside
//     *num_tasks_ptr = num_tasks;
// }


void __workload_versacore(bingo_task_t **task_list, uint32_t *num_tasks_ptr, uintptr_t *output_data_ptr) {
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
    versacore_load_compute_store_args.input_A_addr_hi = (uint32_t)(((uint64_t)(&A[0]) >> 32) & 0xFFFFFFFF);
    versacore_load_compute_store_args.input_A_addr_lo = (uint32_t)(((uint64_t)(&A[0]) >> 0) & 0xFFFFFFFF);
    versacore_load_compute_store_args.input_A_size = ARRAY_SIZE_BYTES(A);
    versacore_load_compute_store_args.input_B_addr_hi = (uint32_t)(((uint64_t)(&B[0]) >> 32) & 0xFFFFFFFF);
    versacore_load_compute_store_args.input_B_addr_lo = (uint32_t)(((uint64_t)(&B[0]) >> 0) & 0xFFFFFFFF);
    versacore_load_compute_store_args.input_B_size = ARRAY_SIZE_BYTES(B);
    versacore_load_compute_store_args.input_C_addr_hi = (uint32_t)(((uint64_t)(&C[0]) >> 32) & 0xFFFFFFFF);
    versacore_load_compute_store_args.input_C_addr_lo = (uint32_t)(((uint64_t)(&C[0]) >> 0) & 0xFFFFFFFF);
    versacore_load_compute_store_args.input_C_size = ARRAY_SIZE_BYTES(C);
    // Outputs
    uintptr_t output_data_addr = hero_host_l3_malloc(sizeof(D) * sizeof(D[0]), output_data_ptr);
    versacore_load_compute_store_args.output_addr_hi = (uint32_t)(((uint64_t)(output_data_addr) >> 32) & 0xFFFFFFFF);
    versacore_load_compute_store_args.output_addr_lo = (uint32_t)(((uint64_t)(output_data_addr) >> 0) & 0xFFFFFFFF);
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
    versacore_load_compute_store_args.streamer_cfg_addr_hi = (uint32_t)(((uint64_t)(&streamer_cfg[0]) >> 32) & 0xFFFFFFFF);
    versacore_load_compute_store_args.streamer_cfg_addr_lo = (uint32_t)(((uint64_t)(&streamer_cfg[0]) >> 0) & 0xFFFFFFFF);
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
    versacore_load_compute_store_args.versacore_cfg_addr_hi = (uint32_t)(((uint64_t)(&versacore_cfg[0]) >> 32) & 0xFFFFFFFF);
    versacore_load_compute_store_args.versacore_cfg_addr_lo = (uint32_t)(((uint64_t)(&versacore_cfg[0]) >> 0) & 0xFFFFFFFF);
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
    bingo_task_t *task_versacore = bingo_task_create(versacore_load_compute_store_func_addr, (uint32_t)(uintptr_t)(&versacore_load_compute_store_args));
    if (task_versacore == NULL) {
        printf("Error: Task versacore creation failed!\r\n");
    }
    bingo_task_t *task_check_results = bingo_task_create(check_results_func_addr, (uint32_t)(uintptr_t)(&task_check_results_args));
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

    asm volatile("fence" ::: "memory");
    // User need to define the number of tasks
    int num_tasks = 2;

    // 2. Set the number of tasks to outside
    return num_tasks;
}