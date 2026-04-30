// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>

#pragma once
#include "host.h"
#include "libbingo/bingo_api.h"
#include "xdma_data.h"
#include <gemm_shapes.h>

// This workload
// 1. test xdma copy from mempool to cluster L1
// 2. test xdma copy from mempool to cluster L3

uint32_t __workload_xdma_mempool(bingo_task_t** task_list) {

    // 1 Get the needed kernel function address by the kernel name
    check_kernel_tab_ready();
    uint32_t __snax_kernel_xdma_1d_copy_func_addr =
        get_device_function("__snax_kernel_xdma_1d_copy");
    uint32_t __snax_kernel_idma_1d_copy_func_addr =
        get_device_function("__snax_kernel_idma_1d_copy");
    uint32_t __check_results_func_addr =
        get_device_function("__snax_kernel_check_results");
    if (__snax_kernel_xdma_1d_copy_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        __snax_kernel_idma_1d_copy_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        __check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf("Error: Kernel symbol lookup failed!\r\n");
    }

    // 2 Prepare the args
    ///////////////////
    ///////////////////
    // args for xdma cp for loading A matrices from mempool to L1 at each
    // chiplet.
    // src the mempool address for A1, A2, A3, A4. dst the L1 address
    // in each cluster size. the size in bytes
    ///////////////////
    ///////////////////
    // Below are data in mempool chiplet
    // 0x8000_0000     A1_mp
    volatile uintptr_t A_mp = chiplet_addr_transform_loc(
        2, 0, SPM_WIDE_BASE_ADDR);  // L4 mempool base address
    // Notice the M,N,Ks, which are defined in a sperate header file
    // Those variables are compiled with the address known at compile time,
    // which are all stored in 0x8000_0000 range If we read them directly, it
    // will incur the cross-chiplet traffic By forcing the read by
    // BINGO_CHIPLET_READW marco, we can ensure those variables are read from
    // local chiplet SPM
    const bingo_gemm_shape_params_t *shape =
        &bingo_gemm_shape_params[BINGO_CHIPLET_READW(array_shape)];
    uint32_t AdataTileSize = BINGO_CHIPLET_READW(M) * BINGO_CHIPLET_READW(K) *
                             shape->meshRow * shape->tileSize * sizeof(uint8_t);
    uintptr_t A1_mp = A_mp + 0 * AdataTileSize;
    uintptr_t A2_mp = A_mp + 1 * AdataTileSize;
    uintptr_t A3_mp = A_mp + 2 * AdataTileSize;
    uintptr_t A4_mp = A_mp + 3 * AdataTileSize;
    // Below are the golden data in local chiplet's L3
    uint64_t A_golden_l3 =
        chiplet_addr_transform((uint64_t)(uintptr_t)(A_data_L3));
    uint64_t A1_golden_l3 = A_golden_l3 + 0 * AdataTileSize;
    uint64_t A2_golden_l3 = A_golden_l3 + 1 * AdataTileSize;
    uint64_t A3_golden_l3 = A_golden_l3 + 2 * AdataTileSize;
    uint64_t A4_golden_l3 = A_golden_l3 + 3 * AdataTileSize;
    // Load A tiles from mempool to L1
    __snax_kernel_xdma_1d_copy_args_t *task_load_A1_mempool_to_chip00_L1_args = NULL;
    // check A loaded correctly
    __snax_kernel_check_results_args_t *task_check_A1_chip00_L1_args = NULL;
    // Assign args for the chiplet 0x00
    if (get_current_chip_id() == 0x00) {
        // Allocate TCDM space for A1 in L1
        uint64_t A1_l1 = bingo_l1_alloc(get_current_chip_id(), 0, BINGO_CHIPLET_READW(AdataTileSize));

        // Prepare args for loading A1 tile from mempool to L1 at chiplet 0x00
        task_load_A1_mempool_to_chip00_L1_args = (__snax_kernel_xdma_1d_copy_args_t *)bingo_l3_alloc(get_current_chip_id(), sizeof(__snax_kernel_xdma_1d_copy_args_t));
        task_load_A1_mempool_to_chip00_L1_args->src_addr_hi = HIGH32(A1_mp);
        task_load_A1_mempool_to_chip00_L1_args->src_addr_lo = LOW32(A1_mp);
        task_load_A1_mempool_to_chip00_L1_args->dst_addr_hi = HIGH32(A1_l1);
        task_load_A1_mempool_to_chip00_L1_args->dst_addr_lo = LOW32(A1_l1);
        task_load_A1_mempool_to_chip00_L1_args->size = BINGO_CHIPLET_READW(AdataTileSize);
        printf_safe(
            "Chip(%x, %x): load A1 from mempool to L1 Args: src_addr_hi=%x, "
            "src_addr_lo=%x, dst_addr_hi=%x, dst_addr_lo=%x, size=%x\r\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(),
            task_load_A1_mempool_to_chip00_L1_args->src_addr_hi,
            task_load_A1_mempool_to_chip00_L1_args->src_addr_lo,
            task_load_A1_mempool_to_chip00_L1_args->dst_addr_hi,
            task_load_A1_mempool_to_chip00_L1_args->dst_addr_lo,
            task_load_A1_mempool_to_chip00_L1_args->size
        );

        // Prepare args for checking A1 loaded in L1 at chiplet 0x00
        task_check_A1_chip00_L1_args = (__snax_kernel_check_results_args_t *)bingo_l3_alloc(get_current_chip_id(), sizeof(__snax_kernel_check_results_args_t));
        task_check_A1_chip00_L1_args->golden_data_addr = LOW32(A1_golden_l3);  // golden A1 at cluster L1
        task_check_A1_chip00_L1_args->output_data_addr = LOW32(A1_l1);         // loaded A1 at L1
        task_check_A1_chip00_L1_args->data_size = BINGO_CHIPLET_READW(AdataTileSize);
        printf_safe(
            "Chip(%x, %x): check A1 loaded in L1 Args: golden_data_addr=%x, output_data_addr=%x, data_size=%x\r\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(),
            task_check_A1_chip00_L1_args->golden_data_addr,
            task_check_A1_chip00_L1_args->output_data_addr,
            task_check_A1_chip00_L1_args->data_size
        );
    }
    /////////////////////
    // 2. Register the tasks
    /////////////////////
    /////////////////
    bingo_task_t* task_load_A1_mempool_to_chip00_L1 = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)(task_load_A1_mempool_to_chip00_L1_args), 0x00,
        0);
    if (task_load_A1_mempool_to_chip00_L1 == NULL) {
        printf("Error: Task mempool to cluster A creation failed!\r\n");
    }
    bingo_task_t* task_check_A1_chip_0x00_L1 = bingo_task_create(
        __check_results_func_addr,
        (uint32_t)(uintptr_t)(task_check_A1_chip00_L1_args), 0x00, 0);
    if (task_check_A1_chip_0x00_L1 == NULL) {
        printf("Error: Task check A creation failed!\r\n");
    }

    // Dependencies for chip00 A1 load and check
    bingo_task_add_depend(task_check_A1_chip_0x00_L1,
                          task_load_A1_mempool_to_chip00_L1);
    //////////////////////
    // End main function //
    //////////////////////

    // Prepare the task list to return
    task_list[0] = task_load_A1_mempool_to_chip00_L1;
    task_list[1] = task_check_A1_chip_0x00_L1;

    asm volatile("fence" ::: "memory");
    uint32_t num_tasks = 2;
    return num_tasks;
}

// Kernel Execution
int kernel_execution(){
    // Set up the tasks list
    // We set a maximum number of tasks to 64
    // Can be changed if needed
    bingo_task_t *task_list[64] = {0};
    uint32_t num_tasks = 0;

    num_tasks = __workload_xdma_mempool(task_list);
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
