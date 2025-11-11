// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>

#pragma once
#include "host.h"
#include "libbingo/bingo_api.h"
#include "mempool_xdma_data.h"

// This workload
// 1. test xdma copy from mempool to cluster L1
// 2. test xdma copy from mempool to cluster L3

uint32_t __workload_mempool_xdma(bingo_task_t** task_list) {

    // 1 Get the needed kernel function address by the kernel name
    check_kernel_tab_ready();
    uint32_t __snax_kernel_xdma_1d_copy_func_addr =
        get_device_function("__snax_kernel_xdma_1d_copy");
    uint32_t __snax_kernel_idma_1d_copy_func_addr =
        get_device_function("__snax_kernel_idma_1d_copy");
    uint32_t check_results_func_addr =
        get_device_function("__snax_kernel_check_results");
    if (__snax_kernel_xdma_1d_copy_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        __snax_kernel_idma_1d_copy_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
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
    // + 1 * AtileSize A2_mp
    // + 2 * AtileSize A3_mp
    // + 3 * AtileSize A4_mp
    // + 4 * AtileSize B_mp
    volatile uintptr_t A_mp = chiplet_addr_transform_loc(
        2, 0, SPM_WIDE_BASE_ADDR);  // L4 mempool base address
    // Notice the M,N,Ks, which are defined in a sperate header file
    // Those variables are compiled with the address known at compile time,
    // which are all stored in 0x8000_0000 range If we read them directly, it
    // will incur the cross-chiplet traffic By forcing the read by
    // BINGO_CHIPLET_READW marco, we can ensure those variables are read from
    // local chiplet SPM
    uint32_t AdataTileSize = BINGO_CHIPLET_READW(M) * BINGO_CHIPLET_READW(K) *
                             BINGO_CHIPLET_READW(meshRow) *
                             BINGO_CHIPLET_READW(tileSize) * sizeof(uint8_t);
    uintptr_t A1_mp = A_mp + 0 * AdataTileSize;
    uintptr_t A2_mp = A_mp + 1 * AdataTileSize;
    uintptr_t A3_mp = A_mp + 2 * AdataTileSize;
    uintptr_t A4_mp = A_mp + 3 * AdataTileSize;
    uintptr_t B_mp = A4_mp + AdataTileSize;
    // Below are the golden data in local chiplet's L3
    uint64_t A_golden_l3 =
        chiplet_addr_transform((uint64_t)(uintptr_t)(A_data_L3));
    uint64_t A1_golden_l3 = A_golden_l3 + 0 * AdataTileSize;
    uint64_t A2_golden_l3 = A_golden_l3 + 1 * AdataTileSize;
    uint64_t A3_golden_l3 = A_golden_l3 + 2 * AdataTileSize;
    uint64_t A4_golden_l3 = A_golden_l3 + 3 * AdataTileSize;

    O1HeapInstance32* local_l1_heap_manager =
        bingo_get_l1_heap_manager(get_current_chip_id(), 0);
    // Allocate TCDM space for A, B, C, D in L1
    uintptr_t A_l1 =
        (uintptr_t)o1heapAllocate32(local_l1_heap_manager, AdataTileSize);

    // Load A tiles from mempool to L1 at each chiplet
    __snax_kernel_xdma_1d_copy_args_t task_mempool_to_cluster_args_A1_chip_0x00_L1;
    // check A loaded correctly at each chiplet
    __snax_kernel_check_results_args_t task_check_A1_args_chip_0x00_L1;

    // Assign args for the chiplet 0x00
    if (get_current_chip_id() == 0x00) {
        // Load A1 from mempool to L1 at chiplet 0x00
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L1)
            ->src_addr_hi = HIGH32(A1_mp);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L1)
            ->src_addr_lo = LOW32(A1_mp);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L1)
            ->dst_addr_hi = HIGH32(A_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L1)
            ->dst_addr_lo = LOW32(A_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L1)
            ->size = BINGO_CHIPLET_READW(AdataTileSize);
        printf(
            "Chip(%x, %x): load A1 from mempool to L1 Args: src_addr_hi=%x, "
            "src_addr_lo=%x, dst_addr_hi=%x, dst_addr_lo=%x, size=%x\r\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(),
            BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L1)
                ->src_addr_hi,
            BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L1)
                ->src_addr_lo,
            BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L1)
                ->dst_addr_hi,
            BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L1)
                ->dst_addr_lo,
            BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L1)
                ->size);

        // check A1 loaded correctly at chiplet 0x00
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A1_args_chip_0x00_L1)
            ->golden_data_addr =
            LOW32(A1_golden_l3);  // golden A1 at cluster L3
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A1_args_chip_0x00_L1)
            ->output_data_addr = LOW32(A_l1);  // loaded A1 at L1
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A1_args_chip_0x00_L1)->data_size =
            BINGO_CHIPLET_READW(AdataTileSize);
        printf(
            "Chip(%x, %x): check A1 loaded correctly Args: "
            "golden_data_addr=%x, output_data_addr=%x, data_size=%x\r\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(),
            BINGO_CHIPLET_LOCAL_AUTO(task_check_A1_args_chip_0x00_L1)
                ->golden_data_addr,
            BINGO_CHIPLET_LOCAL_AUTO(task_check_A1_args_chip_0x00_L1)
                ->output_data_addr,
            BINGO_CHIPLET_LOCAL_AUTO(task_check_A1_args_chip_0x00_L1)->data_size);
    }

    
    // L3 in chip00
    uintptr_t A_l3 = (uintptr_t)o1heapAllocate(
            bingo_get_l3_heap_manager(get_current_chip_id()),
            AdataTileSize);
    // Load A tiles from mempool to L1 at each chiplet
    __snax_kernel_xdma_1d_copy_args_t task_mempool_to_cluster_args_A1_chip_0x00_L3;
    // check A loaded correctly at each chiplet
    __snax_kernel_check_results_args_t task_check_A1_args_chip_0x00_L3;

    // Assign args for the chiplet 0x00
    if (get_current_chip_id() == 0x00) {
        // Load A1 from mempool to L1 at chiplet 0x00
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L3)
            ->src_addr_hi = HIGH32(A1_mp);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L3)
            ->src_addr_lo = LOW32(A1_mp);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L3)
            ->dst_addr_hi = HIGH32(A_l3);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L3)
            ->dst_addr_lo = LOW32(A_l3);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L3)
            ->size = BINGO_CHIPLET_READW(AdataTileSize);
        printf(
            "Chip(%x, %x): load A1 from mempool to L3 Args: src_addr_hi=%x, "
            "src_addr_lo=%x, dst_addr_hi=%x, dst_addr_lo=%x, size=%x\r\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(),
            BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L3)
                ->src_addr_hi,
            BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L3)
                ->src_addr_lo,
            BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L3)
                ->dst_addr_hi,
            BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L3)
                ->dst_addr_lo,
            BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00_L3)
                ->size);

        // check A1 loaded correctly at chiplet 0x00
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A1_args_chip_0x00_L3)
            ->golden_data_addr =
            LOW32(A1_golden_l3);  // golden A1 at cluster L3
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A1_args_chip_0x00_L3)
            ->output_data_addr = LOW32(A_l3);  // loaded A1 at L3
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A1_args_chip_0x00_L3)->data_size =
            BINGO_CHIPLET_READW(AdataTileSize);
        printf(
            "Chip(%x, %x): check A1 loaded correctly Args: "
            "golden_data_addr=%x, output_data_addr=%x, data_size=%x\r\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(),
            BINGO_CHIPLET_LOCAL_AUTO(task_check_A1_args_chip_0x00_L3)
                ->golden_data_addr,
            BINGO_CHIPLET_LOCAL_AUTO(task_check_A1_args_chip_0x00_L3)
                ->output_data_addr,
            BINGO_CHIPLET_LOCAL_AUTO(task_check_A1_args_chip_0x00_L3)->data_size);
    }

    /////////////////////
    // 2. Register the tasks
    /////////////////////
    /////////////////////
    // xdma cp tasks for loading A matrices from mempool to L1 at each chiplet
    // chiplet 0x00
    ///////////////////
    /////////////////
    bingo_task_t* task_mempool_to_cluster_A1_chip_0x00_L1 = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)(&task_mempool_to_cluster_args_A1_chip_0x00_L1), 0x00,
        0);
    if (task_mempool_to_cluster_A1_chip_0x00_L1 == NULL) {
        printf("Error: Task mempool to cluster A creation failed!\r\n");
    }
    bingo_task_t* task_check_A1_chip_0x00_L1 = bingo_task_create(
        check_results_func_addr,
        (uint32_t)(uintptr_t)(&task_check_A1_args_chip_0x00_L1), 0x00, 0);
    if (task_check_A1_chip_0x00_L1 == NULL) {
        printf("Error: Task check A creation failed!\r\n");
    }

    bingo_task_t* task_mempool_to_cluster_A1_chip_0x00_L3 = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)(&task_mempool_to_cluster_args_A1_chip_0x00_L3), 0x00,
        0);
    if (task_mempool_to_cluster_A1_chip_0x00_L3 == NULL) {
        printf("Error: Task mempool to cluster A creation failed!\r\n");
    }
    bingo_task_t* task_check_A1_chip_0x00_L3 = bingo_task_create(
        check_results_func_addr,
        (uint32_t)(uintptr_t)(&task_check_A1_args_chip_0x00_L3), 0x00, 0);
    if (task_check_A1_chip_0x00_L3 == NULL) {
        printf("Error: Task check A creation failed!\r\n");
    }

    // Dependencies for chip00
    // L1 first, then L3
    // bingo_task_add_depend(task_check_A1_chip_0x00_L1,
    //                       task_mempool_to_cluster_A1_chip_0x00_L1);
    // bingo_task_add_depend(task_mempool_to_cluster_A1_chip_0x00_L3,
    //                       task_check_A1_chip_0x00_L1);
    // bingo_task_add_depend(task_check_A1_chip_0x00_L3,
    //                       task_mempool_to_cluster_A1_chip_0x00_L3);
    // L3 first, then L1
    bingo_task_add_depend(task_check_A1_chip_0x00_L3,
                          task_mempool_to_cluster_A1_chip_0x00_L3);
    bingo_task_add_depend(task_mempool_to_cluster_A1_chip_0x00_L1,
                          task_check_A1_chip_0x00_L3);
    bingo_task_add_depend(task_check_A1_chip_0x00_L1,
                          task_mempool_to_cluster_A1_chip_0x00_L1);
    //////////////////////
    // End main function //
    //////////////////////

    // Prepare the task list to return
    task_list[0] = task_mempool_to_cluster_A1_chip_0x00_L1;
    task_list[1] = task_check_A1_chip_0x00_L1;
    task_list[2] = task_mempool_to_cluster_A1_chip_0x00_L3;
    task_list[3] = task_check_A1_chip_0x00_L3;

    asm volatile("fence" ::: "memory");
    uint32_t num_tasks = 4;
    return num_tasks;
}
