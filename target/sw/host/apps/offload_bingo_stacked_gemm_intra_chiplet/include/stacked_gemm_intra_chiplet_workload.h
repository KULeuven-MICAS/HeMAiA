// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>

#pragma once
#include "stacked_gemm_intra_chiplet_data.h"
#include "host.h"
#include "libbingo/bingo_api.h"

#define HOST_DEBUG
#ifdef HOST_DEBUG
#define HOST_DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define HOST_DEBUG_PRINT(...)
#endif

// This workload tests the stacked GEMM within a chiplet
// firstly, we define the workload function:
// D=A1*B1
// E=D*B2

// task1: LOAD A1, --> task2: LOAD B1, --> task3: GEMM1
// task4: LOAD B2, task3 --> task5: GEMM2
// task6: store E
// task7: check results

uint32_t __workload_versacore_stacked_gemm_intra_chiplet(bingo_task_t **task_list)
{
    // In a multichiplet scenario, the task_list should be created in each chiplet
    // The important thing is that the dependency is created for all local task_list var
    // But the task arguments should be set only in the corresponding chiplet
    ///////////////////
    // Main function //
    ///////////////////
    // 1. Get the kernel function address by the kernel name
    // 2. Prepare the args
    // 3. Register the tasks
    // 4. Set the task dependency
    // 5. Set the assigned chiplet id and cluster id
    uint8_t current_chip_id = get_current_chip_id();
    // only test chip 0 for intra-chiplet gemm
    uint8_t task_chip_id = 0;

    // 1 Get the needed kernel function address by the kernel name
    check_kernel_tab_ready();

    uint32_t __snax_kernel_xdma_1d_copy_func_addr = get_device_function("__snax_kernel_xdma_1d_copy");
    uint32_t __snax_kernel_idma_1d_copy_func_addr = get_device_function("__snax_kernel_idma_1d_copy");
    uint32_t __snax_kernel_gemm = get_device_function("__snax_kernel_gemm");
    uint32_t __snax_kernel_minimal_cfg_start_gemm_and_wait_func_addr = get_device_function("__snax_kernel_minimal_cfg_start_gemm_and_wait");
    uint32_t __snax_kernel_check_results_func_addr = get_device_function("__snax_kernel_check_results");
    if (__snax_kernel_xdma_1d_copy_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        __snax_kernel_idma_1d_copy_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        __snax_kernel_gemm == SNAX_SYMTAB_END_FN_ADDR || __snax_kernel_minimal_cfg_start_gemm_and_wait_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        __snax_kernel_check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR)
    {
        HOST_DEBUG_PRINT("Error: Kernel symbol lookup failed!\r\n");
    }

    // 2 Prepare the args
    ///////////////////

    uint32_t A1dataSize = BINGO_CHIPLET_READW(M1) * BINGO_CHIPLET_READW(K1) *
                              BINGO_CHIPLET_READW(meshRow) *
                              BINGO_CHIPLET_READW(tileSize) * sizeof(uint8_t);
    uint32_t B1dataSize = BINGO_CHIPLET_READW(K1) * BINGO_CHIPLET_READW(N1) *
                          BINGO_CHIPLET_READW(meshCol) *
                          BINGO_CHIPLET_READW(tileSize) * sizeof(uint8_t);
    uint32_t D1dataSize = BINGO_CHIPLET_READW(M1) * BINGO_CHIPLET_READW(N1) *
                          BINGO_CHIPLET_READW(meshRow) *
                          BINGO_CHIPLET_READW(meshCol) * sizeof(uint32_t);
    uint32_t B2dataSize = BINGO_CHIPLET_READW(K2) * BINGO_CHIPLET_READW(N2) *
                          BINGO_CHIPLET_READW(meshCol) *
                          BINGO_CHIPLET_READW(tileSize) * sizeof(uint8_t);
    uint32_t D2datasize = BINGO_CHIPLET_READW(M2) * BINGO_CHIPLET_READW(N2) *
                          BINGO_CHIPLET_READW(meshRow) *
                          BINGO_CHIPLET_READW(meshCol) * sizeof(uint32_t);

    uint64_t A1_addr_l3 = chiplet_addr_transform((uint64_t)(uintptr_t)(A1));
    // HOST_DEBUG_PRINT("Chip(%x, %x): A1 matrix L3 address: 0x%lx\r\n",
                    //  get_current_chip_loc_x(), get_current_chip_loc_y(), A1_addr_l3);
    uint64_t B1_addr_l3 = chiplet_addr_transform((uint64_t)(uintptr_t)(B1));
    // HOST_DEBUG_PRINT("Chip(%x, %x): B1 matrix L3 address: 0x%lx\r\n",
    //                  get_current_chip_loc_x(), get_current_chip_loc_y(), B1_addr_l3);
    uint64_t D1_addr_golden = chiplet_addr_transform((uint64_t)(uintptr_t)(D1));
    // HOST_DEBUG_PRINT("Chip(%x, %x): D1 golden matrix L3 address: 0x%lx\r\n",
    //                  get_current_chip_loc_x(), get_current_chip_loc_y(), D1_addr_golden);
    uint64_t B2_addr_l3 = chiplet_addr_transform((uint64_t)(uintptr_t)(B2));
    // HOST_DEBUG_PRINT("Chip(%x, %x): B2 matrix L3 address: 0x%lx\r\n",
    //                  get_current_chip_loc_x(), get_current_chip_loc_y(), B2_addr_l3);
    uint64_t D2_addr_golden = chiplet_addr_transform((uint64_t)(uintptr_t)(D2));
    // HOST_DEBUG_PRINT("Chip(%x, %x): D2 golden matrix L3 address: 0x%lx\r\n",
    //                  get_current_chip_loc_x(), get_current_chip_loc_y(), D2_addr_golden);

    uint64_t cluster_l1_addr_A1 = bingo_l1_alloc(current_chip_id, 0, BINGO_CHIPLET_READW(A1dataSize));

    uint64_t cluster_l1_addr_B1 = bingo_l1_alloc(current_chip_id, 0, BINGO_CHIPLET_READW(B1dataSize));

    uint64_t cluster_l1_addr_D1 = bingo_l1_alloc(current_chip_id, 0, BINGO_CHIPLET_READW(D1dataSize));

    uint64_t cluster_l1_addr_B2 = bingo_l1_alloc(current_chip_id, 0, BINGO_CHIPLET_READW(B2dataSize));

    uint64_t cluster_l1_addr_D2 = bingo_l1_alloc(current_chip_id, 0, BINGO_CHIPLET_READW(D2datasize));

    // Args for loading A1
    __snax_kernel_xdma_1d_copy_args_t *task_l3_to_cluster_args_A = (__snax_kernel_xdma_1d_copy_args_t *)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(__snax_kernel_xdma_1d_copy_args_t));
    task_l3_to_cluster_args_A->src_addr_hi = HIGH32(BINGO_CHIPLET_READD(A1_addr_l3));
    task_l3_to_cluster_args_A->src_addr_lo = LOW32(BINGO_CHIPLET_READD(A1_addr_l3));
    task_l3_to_cluster_args_A->dst_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_A1));
    task_l3_to_cluster_args_A->dst_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_A1));
    task_l3_to_cluster_args_A->size = ARRAY_SIZE_BYTES(A1);

    // Args for loading B1
    __snax_kernel_xdma_1d_copy_args_t *task_l3_to_cluster_args_B1 =
        (__snax_kernel_xdma_1d_copy_args_t *)o1heapAllocate(
            bingo_get_l3_heap_manager(current_chip_id),
            sizeof(__snax_kernel_xdma_1d_copy_args_t));
    task_l3_to_cluster_args_B1->src_addr_hi = HIGH32(BINGO_CHIPLET_READD(B1_addr_l3));
    task_l3_to_cluster_args_B1->src_addr_lo = LOW32(BINGO_CHIPLET_READD(B1_addr_l3));
    task_l3_to_cluster_args_B1->dst_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_B1));
    task_l3_to_cluster_args_B1->dst_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_B1));
    task_l3_to_cluster_args_B1->size = ARRAY_SIZE_BYTES(B1);

    // args for gemm1
    __snax_kernel_gemm_intra_chiplet_args_t *gemm1_args = (__snax_kernel_gemm_intra_chiplet_args_t *)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(__snax_kernel_gemm_intra_chiplet_args_t));
    gemm1_args->input_A_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_A1));
    gemm1_args->input_A_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_A1));
    gemm1_args->input_B_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_B1));
    gemm1_args->input_B_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_B1));
    gemm1_args->input_C_addr_hi = 0;
    gemm1_args->input_C_addr_lo = 0; // no C matrix
    gemm1_args->output_D_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_D1));
    gemm1_args->output_D_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_D1));
    gemm1_args->M = BINGO_CHIPLET_READW(M1);
    gemm1_args->K = BINGO_CHIPLET_READW(K1);
    gemm1_args->N = BINGO_CHIPLET_READW(N1);
    gemm1_args->array_shape = BINGO_CHIPLET_READW(array_shape);
    gemm1_args->transpose_A = BINGO_CHIPLET_READW(transposed_A);
    gemm1_args->transpose_B = BINGO_CHIPLET_READW(transposed_B);
    gemm1_args->accumPrevC = BINGO_CHIPLET_READW(accumPrevC); // false

    // args for loading B2
    __snax_kernel_xdma_1d_copy_args_t *task_l3_to_cluster_args_B2 =
        (__snax_kernel_xdma_1d_copy_args_t *)o1heapAllocate(
            bingo_get_l3_heap_manager(current_chip_id),
            sizeof(__snax_kernel_xdma_1d_copy_args_t));
    task_l3_to_cluster_args_B2->src_addr_hi = HIGH32(BINGO_CHIPLET_READD(B2_addr_l3));
    task_l3_to_cluster_args_B2->src_addr_lo = LOW32(BINGO_CHIPLET_READD(B2_addr_l3));
    task_l3_to_cluster_args_B2->dst_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_B2));
    task_l3_to_cluster_args_B2->dst_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_B2));
    task_l3_to_cluster_args_B2->size = ARRAY_SIZE_BYTES(B2);

    // args for gemm2
    __snax_kernel_minimal_cfg_start_gemm_and_wait_args_t *gemm2_args = (__snax_kernel_minimal_cfg_start_gemm_and_wait_args_t *)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(__snax_kernel_minimal_cfg_start_gemm_and_wait_args_t));
    gemm2_args->input_A_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_D1));
    gemm2_args->input_B_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_B2));
    gemm2_args->input_C_addr_lo = 0; // no C matrix
    gemm2_args->output_D_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_D2));

    // args for storing E
    uint64_t D2_addr_l3 = o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), ARRAY_SIZE_BYTES(D2));

    __snax_kernel_xdma_1d_copy_args_t *task_cluster_to_l3_args_D2 =
        (__snax_kernel_xdma_1d_copy_args_t *)o1heapAllocate(
            bingo_get_l3_heap_manager(current_chip_id),
            sizeof(__snax_kernel_xdma_1d_copy_args_t));
    task_cluster_to_l3_args_D2->src_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_D2));
    task_cluster_to_l3_args_D2->src_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_D2));
    task_cluster_to_l3_args_D2->dst_addr_hi = HIGH32(BINGO_CHIPLET_READD(D2_addr_l3));
    task_cluster_to_l3_args_D2->dst_addr_lo = LOW32(BINGO_CHIPLET_READD(D2_addr_l3));
    task_cluster_to_l3_args_D2->size = ARRAY_SIZE_BYTES(D2);

    // args for checking results
    __snax_kernel_check_results_args_t *task_check_results_args_D2 =
        (__snax_kernel_check_results_args_t *)o1heapAllocate(
            bingo_get_l3_heap_manager(current_chip_id),
            sizeof(__snax_kernel_check_results_args_t));
    task_check_results_args_D2->golden_data_addr = LOW32(BINGO_CHIPLET_READD(D2_addr_golden));
    task_check_results_args_D2->output_data_addr = LOW32(BINGO_CHIPLET_READD(D2_addr_l3));
    task_check_results_args_D2->data_size = ARRAY_SIZE_BYTES(D2);

    // register the tasks
    // 1. Load A1
    bingo_task_t *task_load_A1 = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)task_l3_to_cluster_args_A,
        task_chip_id,
        0);
    bingo_task_t *task_load_B1 = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)task_l3_to_cluster_args_B1,
        task_chip_id,
        0);
    bingo_task_t *task_gemm1 = bingo_task_create(
        __snax_kernel_gemm,
        (uint32_t)(uintptr_t)gemm1_args,
        task_chip_id,
        0);
    bingo_task_t *task_load_B2 = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)task_l3_to_cluster_args_B2,
        task_chip_id,
        0);
    bingo_task_t *task_gemm2 = bingo_task_create(
        __snax_kernel_gemm,
        (uint32_t)(uintptr_t)gemm2_args,
        task_chip_id,
        0);
    bingo_task_t *task_store_D2 = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)task_cluster_to_l3_args_D2,
        task_chip_id,
        0);
    bingo_task_t *task_check_D2 = bingo_task_create(
        __snax_kernel_check_results_func_addr,
        (uint32_t)(uintptr_t)task_check_results_args_D2,
        task_chip_id,
        0);

    // Set the task dependency
    // task1: LOAD A1, --> task2: LOAD B1, --> task3: GEMM1
    bingo_task_add_depend(task_load_B1, task_load_A1);
    bingo_task_add_depend(task_gemm1, task_load_B1);
    // task4: LOAD B2, task3 --> task5: GEMM2
    bingo_task_add_depend(task_load_B2, task_load_B1);
    bingo_task_add_depend(task_gemm2, task_load_B2);
    bingo_task_add_depend(task_gemm2, task_gemm1);
    // task6: store E
    bingo_task_add_depend(task_store_D2, task_gemm2);
    // task7: check results
    bingo_task_add_depend(task_check_D2, task_store_D2);

    asm volatile("fence" ::: "memory");

    // Prepare the task list to return
    task_list[0] = task_load_A1;
    task_list[1] = task_load_B1;
    task_list[2] = task_gemm1;
    task_list[3] = task_load_B2;
    task_list[4] = task_gemm2;
    task_list[5] = task_store_D2;
    task_list[6] = task_check_D2;

    uint32_t num_tasks = 7;

    return num_tasks;
}
