// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>

#pragma once
#include "gemm_intra_chiplet_data.h"
#include "host.h"
#include "libbingo/bingo_api.h"

// This workload tests the single-chiplet
// 1. load A and B matrices from L3 to L1
// 2. perform GEMM on versacore, pass correct args to versacore
// 3. store the result D matrix back to L3
// 4. check the result D matrix with the golden data

uint32_t __workload_versacore_intra_chiplet(bingo_task_t** task_list
                                             ) {
    ///////////////////
    // Main function //
    ///////////////////
    // 1. Get the kernel function address by the kernel name and prepare the
    // args
    // 2. Register the tasks
    // 3. Set the task dependency
    // 4. Set the assigned chiplet id and cluster id

    uint8_t current_chip_id = get_current_chip_id();
    uint8_t cluster_id = 0;  // versacore is located at cluster

    // 1.1 Get the kernel function address by the kernel name
    check_kernel_tab_ready();
    uint32_t __snax_kernel_xdma_1d_copy_func_addr =
        get_device_function("__snax_kernel_xdma_1d_copy");
    uint32_t __snax_kernel_gemm_func_addr =
        get_device_function("__snax_kernel_gemm");
    uint32_t check_results_func_addr =
        get_device_function("__snax_kernel_check_results");

    if (__snax_kernel_xdma_1d_copy_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        __snax_kernel_gemm_func_addr ==
            SNAX_SYMTAB_END_FN_ADDR ||
        check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf("Error: Kernel symbol lookup failed!\r\n");
    }

    // args for xdma cp for loading A and B matrices from L3 to L1
    // Copy 1d data from src to dst using xdma
    // Arg0: uint32_t src_addr_hi
    // Arg1: uint32_t src_addr_lo
    // Arg2: uint32_t dst_addr_hi
    // Arg3: uint32_t dst_addr_lo
    // Arg4: uint32_t size in Byte

    uint32_t AdataTileSize = BINGO_CHIPLET_READW(M) * BINGO_CHIPLET_READW(K) * BINGO_CHIPLET_READW(meshRow) * BINGO_CHIPLET_READW(tileSize) * sizeof(uint8_t);
    uint32_t BdataSize     = BINGO_CHIPLET_READW(K) * BINGO_CHIPLET_READW(N) * BINGO_CHIPLET_READW(meshCol) * BINGO_CHIPLET_READW(tileSize) * sizeof(uint8_t);
    uint32_t CdataSize     = BINGO_CHIPLET_READW(M) * BINGO_CHIPLET_READW(N) * BINGO_CHIPLET_READW(meshRow) * BINGO_CHIPLET_READW(meshCol)  * sizeof(uint8_t);
    uint32_t DdataSize     = BINGO_CHIPLET_READW(M) * BINGO_CHIPLET_READW(N) * BINGO_CHIPLET_READW(meshRow) * BINGO_CHIPLET_READW(meshCol)  * sizeof(uint32_t);

    __snax_kernel_xdma_1d_copy_args_t task_l3_to_cluster_args_A;
    task_l3_to_cluster_args_A.src_addr_hi = HIGH32(&A1[0]);
    task_l3_to_cluster_args_A.src_addr_lo = LOW32(&A1[0]);
    // destination address in cluster L1 for A matrix
    uintptr_t cluster_l1_addr_A = bingo_l1_alloc(get_current_chip_id(), 0, BINGO_CHIPLET_READW(AdataTileSize));
    task_l3_to_cluster_args_A.dst_addr_hi = HIGH32(cluster_l1_addr_A);
    task_l3_to_cluster_args_A.dst_addr_lo = LOW32(cluster_l1_addr_A);
    task_l3_to_cluster_args_A.size = ARRAY_SIZE_BYTES(A1);

    __snax_kernel_xdma_1d_copy_args_t task_l3_to_cluster_args_B;
    task_l3_to_cluster_args_B.src_addr_hi = HIGH32(&B[0]);
    task_l3_to_cluster_args_B.src_addr_lo = LOW32(&B[0]);
    // destination address in cluster L1 for B matrix
    uintptr_t cluster_l1_addr_B = bingo_l1_alloc(get_current_chip_id(), 0, BINGO_CHIPLET_READW(BdataSize));
    task_l3_to_cluster_args_B.dst_addr_hi = HIGH32(cluster_l1_addr_B);
    task_l3_to_cluster_args_B.dst_addr_lo = LOW32(cluster_l1_addr_B);
    task_l3_to_cluster_args_B.size = ARRAY_SIZE_BYTES(B);

    uint32_t* gemm_args_chip_0x00 = (uint32_t*)NULL;

    uintptr_t cluster_l1_addr_C =
        bingo_l1_alloc(get_current_chip_id(), 0, BINGO_CHIPLET_READW(CdataSize));
    uintptr_t cluster_l1_addr_D =
        bingo_l1_alloc(get_current_chip_id(), 0, BINGO_CHIPLET_READW(DdataSize));

    // Prepare the args for Chiplet 0
    if (current_chip_id == 0x00) {
        gemm_args_chip_0x00 = (uint32_t*)o1heapAllocate(
            bingo_get_l3_heap_manager(get_current_chip_id()),
            sizeof(uint32_t) * 15);
        // versacore args
        // A matrix
        gemm_args_chip_0x00[0] = HIGH32(cluster_l1_addr_A);
        gemm_args_chip_0x00[1] = LOW32(cluster_l1_addr_A);
        // B matrix
        gemm_args_chip_0x00[2] = HIGH32(cluster_l1_addr_B);
        gemm_args_chip_0x00[3] = LOW32(cluster_l1_addr_B);
        // C matrix
        if (accumPrevC || addZeroC) {
            // When accumPrevC is true, we use D as the previous C matrix
            gemm_args_chip_0x00[4] = 0;
            gemm_args_chip_0x00[5] = 0;
        } else {
            gemm_args_chip_0x00[4] = HIGH32(cluster_l1_addr_C);
            gemm_args_chip_0x00[5] = LOW32(cluster_l1_addr_C);
        }

        gemm_args_chip_0x00[6] = HIGH32(cluster_l1_addr_D);
        gemm_args_chip_0x00[7] = LOW32(cluster_l1_addr_D);
        // Matrix dimensions
        gemm_args_chip_0x00[8] = M;   // M
        gemm_args_chip_0x00[9] = K;   // K
        gemm_args_chip_0x00[10] = N;  // N
        // SUs
        gemm_args_chip_0x00[11] = array_shape;
        // transpose A
        gemm_args_chip_0x00[12] = transposed_A;
        // transpose B
        gemm_args_chip_0x00[13] = transposed_B;
        // accumPrevC
        gemm_args_chip_0x00[14] = accumPrevC;
    }

    // move the data from cluster L1 to L3 for D matrix
    __snax_kernel_xdma_1d_copy_args_t task_cluster_to_l3_args_D;
    // D matrix (output)
    uintptr_t output_data_addr_chip_0x00 = (uintptr_t)NULL;
    output_data_addr_chip_0x00 =
        (uintptr_t)o1heapAllocate(bingo_get_l3_heap_manager(current_chip_id), ARRAY_SIZE_BYTES(D1));
    task_cluster_to_l3_args_D.src_addr_hi = HIGH32(cluster_l1_addr_D);
    task_cluster_to_l3_args_D.src_addr_lo = LOW32(cluster_l1_addr_D);
    task_cluster_to_l3_args_D.dst_addr_hi = HIGH32(output_data_addr_chip_0x00);
    task_cluster_to_l3_args_D.dst_addr_lo = LOW32(output_data_addr_chip_0x00);
    task_cluster_to_l3_args_D.size = ARRAY_SIZE_BYTES(D1);

    // checkresults args
    __snax_kernel_check_results_args_t task_check_results_args_chip_0x00;
    if (current_chip_id == 0x00) {
        task_check_results_args_chip_0x00.golden_data_addr =
            (uint32_t)(uintptr_t)(&D1[0]);
        task_check_results_args_chip_0x00.output_data_addr =
            (uint32_t)(uintptr_t)(output_data_addr_chip_0x00);
        task_check_results_args_chip_0x00.data_size = ARRAY_SIZE_BYTES(D1);
    }

    // 2. Register the tasks, only for chiplet 0x00
    // two xdma move
    bingo_task_t* task_xdma_l3_to_cluster_A = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)(&task_l3_to_cluster_args_A), 0x00, cluster_id);
    bingo_task_t* task_xdma_l3_to_cluster_B = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)(&task_l3_to_cluster_args_B), 0x00, cluster_id);
    // versacore gemm compute
    bingo_task_t* task_versacore_chip_0x00 = bingo_task_create(
        __snax_kernel_gemm_func_addr,
        (uint32_t)(uintptr_t)(gemm_args_chip_0x00), 0x00, cluster_id);
    if (task_versacore_chip_0x00 == NULL) {
        printf("Error: Task versacore creation failed!\r\n");
    }
    // xdma move D back to L3
    bingo_task_t* task_xdma_cluster_to_l3_D = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)(&task_cluster_to_l3_args_D), 0x00, cluster_id);
    // check results at L3
    bingo_task_t* task_check_results_chip_0x00 = bingo_task_create(
        check_results_func_addr,
        (uint32_t)(uintptr_t)(&task_check_results_args_chip_0x00), 0x00,
        cluster_id);
    if (task_check_results_chip_0x00 == NULL) {
        printf("Error: Task check results creation failed!\r\n");
    }

    // 3. Set the task dependency
    // Here we have only two tasks and simple dependency
    // ldA, ld B -> versacore compute -> stD -> check results
    bingo_task_add_depend(task_versacore_chip_0x00, task_xdma_l3_to_cluster_A);
    bingo_task_add_depend(task_versacore_chip_0x00, task_xdma_l3_to_cluster_B);
    bingo_task_add_depend(task_xdma_cluster_to_l3_D, task_versacore_chip_0x00);
    bingo_task_add_depend(task_check_results_chip_0x00,
                          task_xdma_cluster_to_l3_D);

    // 4. Flush the caches to make sure the device can see the latest arguments
    asm volatile("fence" ::: "memory");
    //////////////////////
    // End main function //
    //////////////////////

    // Prepare the task list to return
    task_list[0] = task_xdma_l3_to_cluster_A;
    task_list[1] = task_xdma_l3_to_cluster_B;
    task_list[2] = task_versacore_chip_0x00;
    task_list[3] = task_xdma_cluster_to_l3_D;
    task_list[4] = task_check_results_chip_0x00;
    uint32_t num_tasks = 5;


    return num_tasks;
}
