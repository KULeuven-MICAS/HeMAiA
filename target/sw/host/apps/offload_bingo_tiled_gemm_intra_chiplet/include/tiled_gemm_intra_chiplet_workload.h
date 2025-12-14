// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>

#pragma once
#include "tiled_gemm_intra_chiplet_data.h"
#include "host.h"
#include "libbingo/bingo_api.h"

// This workload tests the single-chiplet
// 1. load A and B matrices from L3 to L1
// 2. perform GEMM on versacore, pass correct args to versacore
// 3. store the result D matrix back to L3
// 4. check the result D matrix with the golden data
#ifdef HOST_DEBUG
#define HOST_DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define HOST_DEBUG_PRINT(...)
#endif
uint32_t __workload_versacore_intra_chiplet(bingo_task_t** task_list) {
    ///////////////////
    // Main function //
    ///////////////////
    // 1. Get the kernel function address by the kernel name and prepare the
    // args
    // 2. Register the tasks
    // 3. Set the task dependency
    // 4. Set the assigned chiplet id and cluster id

    uint8_t current_chip_id = get_current_chip_id();
    // only test chip 0 for intra-chiplet gemm
    uint8_t task_chip_id = 0;
    uint8_t cluster_id = 0;  // versacore is located at cluster

    // 1.1 Get the kernel function address by the kernel name
    check_kernel_tab_ready();
    uint32_t __snax_kernel_idma_1d_copy_func_addr =
        get_device_function("__snax_kernel_idma_1d_copy");
    uint32_t __snax_kernel_gemm_func_addr =
        get_device_function("__snax_kernel_gemm");
    uint32_t check_results_func_addr =
        get_device_function("__snax_kernel_check_results");

    if (__snax_kernel_idma_1d_copy_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        __snax_kernel_gemm_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf("Error: Kernel symbol lookup failed!\r\n");
    }

    // args for idma cp for loading A and B matrices from L3 to L1
    // Copy 1d data from src to dst using idma
    // Arg0: uint32_t src_addr_hi
    // Arg1: uint32_t src_addr_lo
    // Arg2: uint32_t dst_addr_hi
    // Arg3: uint32_t dst_addr_lo
    // Arg4: uint32_t size in Byte

    uint32_t AdataTileSize = BINGO_CHIPLET_READW(M) * BINGO_CHIPLET_READW(K) *
                             BINGO_CHIPLET_READW(meshRow) *
                             BINGO_CHIPLET_READW(tileSize) * sizeof(uint8_t);
    uint32_t BdataSize = BINGO_CHIPLET_READW(K) * BINGO_CHIPLET_READW(N) *
                         BINGO_CHIPLET_READW(meshCol) *
                         BINGO_CHIPLET_READW(tileSize) * sizeof(uint8_t);
    uint32_t CdataSize = BINGO_CHIPLET_READW(M) * BINGO_CHIPLET_READW(N) *
                         BINGO_CHIPLET_READW(meshRow) *
                         BINGO_CHIPLET_READW(meshCol) * sizeof(uint8_t);
    uint32_t DdataSize = BINGO_CHIPLET_READW(M) * BINGO_CHIPLET_READW(N) *
                         BINGO_CHIPLET_READW(meshRow) *
                         BINGO_CHIPLET_READW(meshCol) * sizeof(uint32_t);
    uint64_t A1_addr_l3 = chiplet_addr_transform((uint64_t)(uintptr_t)(A1));
    HOST_DEBUG_PRINT("Chip(%x, %x): A1 matrix L3 address: 0x%lx\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), A1_addr_l3);
    uint64_t B_addr_l3 = chiplet_addr_transform((uint64_t)(uintptr_t)(B));
    HOST_DEBUG_PRINT("Chip(%x, %x): B matrix L3 address: 0x%lx\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), B_addr_l3);
    uint64_t D_addr_golden = chiplet_addr_transform((uint64_t)(uintptr_t)(D1));
    uint64_t cluster_l1_addr_A = bingo_l1_alloc(current_chip_id, 0, BINGO_CHIPLET_READW(AdataTileSize));
    uint64_t cluster_l1_addr_B = bingo_l1_alloc(current_chip_id, 0, BINGO_CHIPLET_READW(BdataSize));
    uint64_t cluster_l1_addr_C = bingo_l1_alloc(current_chip_id, 0, BINGO_CHIPLET_READW(CdataSize));
    uint64_t cluster_l1_addr_D = bingo_l1_alloc(current_chip_id, 0, BINGO_CHIPLET_READW(D1));
    // D matrix (output)
    uint64_t output_data_addr_chip_0x00 = o1heapAllocate(bingo_get_l3_heap_manager(current_chip_id), ARRAY_SIZE_BYTES(D1));
    __snax_kernel_idma_1d_copy_args_t *task_l3_to_cluster_args_A = 
        (__snax_kernel_idma_1d_copy_args_t *)o1heapAllocate(
            bingo_get_l3_heap_manager(current_chip_id),
            sizeof(__snax_kernel_idma_1d_copy_args_t));
    task_l3_to_cluster_args_A->src_addr_hi = HIGH32(BINGO_CHIPLET_READD(A1_addr_l3));
    task_l3_to_cluster_args_A->src_addr_lo = LOW32(BINGO_CHIPLET_READD(A1_addr_l3));
    task_l3_to_cluster_args_A->dst_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_A));
    task_l3_to_cluster_args_A->dst_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_A));
    task_l3_to_cluster_args_A->size = ARRAY_SIZE_BYTES(A1);
    HOST_DEBUG_PRINT("Chip(%x, %x): Load A matrix from L3 to L1 Args: src_addr_hi=%x, src_addr_lo=%x, dst_addr_hi=%x, dst_addr_lo=%x, size=%x\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(),
           task_l3_to_cluster_args_A->src_addr_hi,
           task_l3_to_cluster_args_A->src_addr_lo,
           task_l3_to_cluster_args_A->dst_addr_hi,
           task_l3_to_cluster_args_A->dst_addr_lo,
           task_l3_to_cluster_args_A->size);
    __snax_kernel_idma_1d_copy_args_t *task_l3_to_cluster_args_B = 
        (__snax_kernel_idma_1d_copy_args_t *)o1heapAllocate(
            bingo_get_l3_heap_manager(current_chip_id),
            sizeof(__snax_kernel_idma_1d_copy_args_t));
    task_l3_to_cluster_args_B->src_addr_hi = HIGH32(BINGO_CHIPLET_READD(B_addr_l3));
    task_l3_to_cluster_args_B->src_addr_lo = LOW32(BINGO_CHIPLET_READD(B_addr_l3));
    task_l3_to_cluster_args_B->dst_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_B));
    task_l3_to_cluster_args_B->dst_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_B));
    task_l3_to_cluster_args_B->size = ARRAY_SIZE_BYTES(B);
    HOST_DEBUG_PRINT("Chip(%x, %x): Load B matrix from L3 to L1 Args: src_addr_hi=%x, src_addr_lo=%x, dst_addr_hi=%x, dst_addr_lo=%x, size=%x\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(),
           task_l3_to_cluster_args_B->src_addr_hi,
           task_l3_to_cluster_args_B->src_addr_lo,
           task_l3_to_cluster_args_B->dst_addr_hi,
           task_l3_to_cluster_args_B->dst_addr_lo,
           task_l3_to_cluster_args_B->size);
    // Prepare the args for Chiplet 0
    uint32_t* gemm_args_chip_0x00 = (uint32_t*)o1heapAllocate(bingo_get_l3_heap_manager(current_chip_id), sizeof(uint32_t) * 15);
    // versacore args
    // A matrix
    gemm_args_chip_0x00[0] = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_A));
    gemm_args_chip_0x00[1] = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_A));
    // B matrix
    gemm_args_chip_0x00[2] = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_B));
    gemm_args_chip_0x00[3] = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_B));
    // C matrix
    if (accumPrevC || addZeroC) {
        // When accumPrevC is true, we use D as the previous C matrix
        gemm_args_chip_0x00[4] = 0;
        gemm_args_chip_0x00[5] = 0;
    } else {
        gemm_args_chip_0x00[4] = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_C));
        gemm_args_chip_0x00[5] = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_C));
    }

    gemm_args_chip_0x00[6] = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_D));
    gemm_args_chip_0x00[7] = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_D));
    // Matrix dimensions
    gemm_args_chip_0x00[8] = BINGO_CHIPLET_READW(M);   // M
    gemm_args_chip_0x00[9] = BINGO_CHIPLET_READW(K);   // K
    gemm_args_chip_0x00[10] = BINGO_CHIPLET_READW(N);  // N
    // SUs
    gemm_args_chip_0x00[11] = BINGO_CHIPLET_READW(array_shape);
    // transpose A
    gemm_args_chip_0x00[12] = BINGO_CHIPLET_READW(transposed_A);
    // transpose B
    gemm_args_chip_0x00[13] = BINGO_CHIPLET_READW(transposed_B);
    // accumPrevC
    gemm_args_chip_0x00[14] = BINGO_CHIPLET_READW(accumPrevC);
    HOST_DEBUG_PRINT("Chip(%x, %x): gemm compute Args: A_addr_hi=%x, A_addr_lo=%x, B_addr_hi=%x, B_addr_lo=%x, C_addr_hi=%x, C_addr_lo=%x, D_addr_hi=%x, D_addr_lo=%x, M=%x, K=%x, N=%x\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(),
           gemm_args_chip_0x00[0], gemm_args_chip_0x00[1],
           gemm_args_chip_0x00[2], gemm_args_chip_0x00[3],
           gemm_args_chip_0x00[4], gemm_args_chip_0x00[5],
           gemm_args_chip_0x00[6], gemm_args_chip_0x00[7],
           gemm_args_chip_0x00[8], gemm_args_chip_0x00[9],
           gemm_args_chip_0x00[10]);
    // move the data from cluster L1 to L3 for D matrix
    __snax_kernel_idma_1d_copy_args_t *task_cluster_to_l3_args_D = 
        (__snax_kernel_idma_1d_copy_args_t *)o1heapAllocate(
            bingo_get_l3_heap_manager(current_chip_id),
            sizeof(__snax_kernel_idma_1d_copy_args_t));
    task_cluster_to_l3_args_D->src_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_D));
    task_cluster_to_l3_args_D->src_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_D));
    task_cluster_to_l3_args_D->dst_addr_hi = HIGH32(BINGO_CHIPLET_READD(output_data_addr_chip_0x00));
    task_cluster_to_l3_args_D->dst_addr_lo = LOW32(BINGO_CHIPLET_READD(output_data_addr_chip_0x00));
    task_cluster_to_l3_args_D->size = ARRAY_SIZE_BYTES(D1);
    HOST_DEBUG_PRINT("Chip(%x, %x): Store D matrix from L1 to L3 Args: src_addr_hi=%x, src_addr_lo=%x, dst_addr_hi=%x, dst_addr_lo=%x, size=%x\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(),
           task_cluster_to_l3_args_D->src_addr_hi,
           task_cluster_to_l3_args_D->src_addr_lo,
           task_cluster_to_l3_args_D->dst_addr_hi,
           task_cluster_to_l3_args_D->dst_addr_lo,
           task_cluster_to_l3_args_D->size);
    // args for checking results
    // checkresults args
    __snax_kernel_check_results_args_t *task_check_results_args_chip_0x00 = 
        (__snax_kernel_check_results_args_t *)o1heapAllocate(
            bingo_get_l3_heap_manager(current_chip_id),
            sizeof(__snax_kernel_check_results_args_t));
    task_check_results_args_chip_0x00->golden_data_addr = LOW32(BINGO_CHIPLET_READD(D_addr_golden));
    task_check_results_args_chip_0x00->output_data_addr = LOW32(BINGO_CHIPLET_READD(output_data_addr_chip_0x00));
    task_check_results_args_chip_0x00->data_size = ARRAY_SIZE_BYTES(D1);
    printf("Chip(%x, %x): Check results Args: golden_data_addr=%x, output_data_addr=%x, data_size=%x\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(),
           task_check_results_args_chip_0x00->golden_data_addr,
           task_check_results_args_chip_0x00->output_data_addr,
           task_check_results_args_chip_0x00->data_size);

    // 2. Register the tasks, only for chiplet 0x00
    // two idma move
    bingo_task_t* task_idma_l3_to_cluster_A =
        bingo_task_create(__snax_kernel_idma_1d_copy_func_addr,
                          (uint32_t)(uintptr_t)(task_l3_to_cluster_args_A),
                          task_chip_id, cluster_id);
    bingo_task_t* task_idma_l3_to_cluster_B =
        bingo_task_create(__snax_kernel_idma_1d_copy_func_addr,
                          (uint32_t)(uintptr_t)(task_l3_to_cluster_args_B),
                          task_chip_id, cluster_id);
    // versacore gemm compute
    bingo_task_t* task_versacore_chip_0x00 = bingo_task_create(
        __snax_kernel_gemm_func_addr,
        (uint32_t)(uintptr_t)(gemm_args_chip_0x00), task_chip_id, cluster_id);
    if (task_versacore_chip_0x00 == NULL) {
        HOST_DEBUG_PRINT("Error: Task versacore creation failed!\r\n");
    }
    // idma move D back to L3
    bingo_task_t* task_idma_cluster_to_l3_D =
        bingo_task_create(__snax_kernel_idma_1d_copy_func_addr,
                          (uint32_t)(uintptr_t)(task_cluster_to_l3_args_D),
                          task_chip_id, cluster_id);
    // check results at L3
    bingo_task_t* task_check_results_chip_0x00 = bingo_task_create(
        check_results_func_addr,
        (uint32_t)(uintptr_t)(task_check_results_args_chip_0x00), task_chip_id,
        cluster_id);
    if (task_check_results_chip_0x00 == NULL) {
        HOST_DEBUG_PRINT("Error: Task check results creation failed!\r\n");
    }

    // 3. Set the task dependency
    // Here we have only two tasks and simple dependency
    // ldA -> ld B -> versacore compute -> stD -> check results
    bingo_task_add_depend(task_idma_l3_to_cluster_B, task_idma_l3_to_cluster_A);
    bingo_task_add_depend(task_versacore_chip_0x00, task_idma_l3_to_cluster_B);
    bingo_task_add_depend(task_idma_cluster_to_l3_D, task_versacore_chip_0x00);
    bingo_task_add_depend(task_check_results_chip_0x00,
                          task_idma_cluster_to_l3_D);

    // 4. Flush the caches to make sure the device can see the latest arguments
    asm volatile("fence" ::: "memory");
    //////////////////////
    // End main function //
    //////////////////////

    // Prepare the task list to return
    task_list[0] = task_idma_l3_to_cluster_A;
    task_list[1] = task_idma_l3_to_cluster_B;
    task_list[2] = task_versacore_chip_0x00;
    task_list[3] = task_idma_cluster_to_l3_D;
    task_list[4] = task_check_results_chip_0x00;
    uint32_t num_tasks = 5;

    return num_tasks;
}
