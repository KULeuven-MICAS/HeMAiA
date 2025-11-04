// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>

#pragma once
#include "gemm_multi_chiplet_data.h"
#include "host.h"
#include "libbingo/bingo_api.h"

// This workload tests the multi-chiplet gemm offloading with versacore
// Each chiplet works on a sub-matrix of A and the full matrix of B
// independently, without any communication between chiplets.

// A=[A1;A2;A3;A4] split into 4 sub-matrices along the row dimension
// A1*B on Chiplet 0x00
// A2*B on Chiplet 0x01
// A3*B on Chiplet 0x10
// A4*B on Chiplet 0x11

// the agruments for gemm kernel for each chiplet are the same except the
// A1/A2/A3/A4 pointers, C1/C2/C3/C4 and the output D1/D2/D3/D4 pointers

uint32_t __workload_versacore_multi_chiplet(bingo_task_t** task_list,
                                            uintptr_t* output_data_ptr) {
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
    uint32_t check_results_func_addr =
        get_device_function("__snax_kernel_check_results");
    uint32_t __snax_kernel_gemm_intra_chiplet_func_addr =
        get_device_function("__snax_kernel_gemm_intra_chiplet");
    if (check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        __snax_kernel_gemm_intra_chiplet_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf("Error: Kernel symbol lookup failed!\r\n");
    }

    uint32_t* gemm_args_chip_0x00 = (uint32_t*)NULL;
    uint32_t* gemm_args_chip_0x01 = (uint32_t*)NULL;
    uint32_t* gemm_args_chip_0x10 = (uint32_t*)NULL;
    uint32_t* gemm_args_chip_0x11 = (uint32_t*)NULL;
    uintptr_t output_data_addr_chip_0x00 = (uintptr_t)NULL;
    uintptr_t output_data_addr_chip_0x01 = (uintptr_t)NULL;
    uintptr_t output_data_addr_chip_0x10 = (uintptr_t)NULL;
    uintptr_t output_data_addr_chip_0x11 = (uintptr_t)NULL;

    // 1.2 Prepare the args for Chiplet 0
    if (current_chip_id == 0x00) {
        printf("L3 Heap Manager is at 0x%lx\r\n",
               bingo_get_l3_heap_manager(current_chip_id));
        printf("gemm_args_chip_0x00 is located at: %x\r\n",
               &gemm_args_chip_0x00);
        gemm_args_chip_0x00 = (uint32_t*)o1heapAllocate(
            bingo_get_l3_heap_manager(get_current_chip_id()),
            sizeof(uint32_t) * 15);
        // versacore args
        // A matrix
        gemm_args_chip_0x00[0] = HIGH32(&A1[0]);
        gemm_args_chip_0x00[1] = LOW32(&A1[0]);
        // B matrix
        gemm_args_chip_0x00[2] = HIGH32(&B[0]);
        gemm_args_chip_0x00[3] = LOW32(&B[0]);
        // C matrix
        if (accumPrevC || addZeroC) {
            // When accumPrevC is true, we use D as the previous C matrix
            gemm_args_chip_0x00[4] = 0;
            gemm_args_chip_0x00[5] = 0;
        } else {
            gemm_args_chip_0x00[4] = HIGH32(&C1[0]);
            gemm_args_chip_0x00[5] = LOW32(&C1[0]);
        }

        // D matrix (output)
        O1HeapInstance* local_l3_heap_manager =
            bingo_get_l3_heap_manager(current_chip_id);
        output_data_addr_chip_0x00 = (uintptr_t)o1heapAllocate(
            local_l3_heap_manager, ARRAY_SIZE_BYTES(D1));
        gemm_args_chip_0x00[6] = HIGH32(output_data_addr_chip_0x00);
        gemm_args_chip_0x00[7] = LOW32(output_data_addr_chip_0x00);
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

    if (current_chip_id == 0x01) {
        printf("L3 Heap Manager is at 0x%lx\r\n",
               bingo_get_l3_heap_manager(current_chip_id));
        printf("gemm_args_chip_0x01 is located at: %x\r\n",
               &gemm_args_chip_0x01);
        gemm_args_chip_0x01 = (uint32_t*)o1heapAllocate(
            bingo_get_l3_heap_manager(get_current_chip_id()),
            sizeof(uint32_t) * 15);
        // versacore args
        // A matrix
        gemm_args_chip_0x01[0] = HIGH32(&A2[0]);
        gemm_args_chip_0x01[1] = LOW32(&A2[0]);
        // B matrix
        gemm_args_chip_0x01[2] = HIGH32(&B[0]);
        gemm_args_chip_0x01[3] = LOW32(&B[0]);
        // C matrix
        if (accumPrevC || addZeroC) {
            // When accumPrevC is true, we use D as the previous C matrix
            gemm_args_chip_0x01[4] = 0;
            gemm_args_chip_0x01[5] = 0;
        } else {
            gemm_args_chip_0x01[4] = HIGH32(&C2[0]);
            gemm_args_chip_0x01[5] = LOW32(&C2[0]);
        }

        // D matrix (output)
        O1HeapInstance* local_l3_heap_manager =
            bingo_get_l3_heap_manager(current_chip_id);
        output_data_addr_chip_0x01 = (uintptr_t)o1heapAllocate(
            local_l3_heap_manager, ARRAY_SIZE_BYTES(D2));
        gemm_args_chip_0x01[6] = HIGH32(output_data_addr_chip_0x01);
        gemm_args_chip_0x01[7] = LOW32(output_data_addr_chip_0x01);
        // Matrix dimensions
        gemm_args_chip_0x01[8] = M;   // M
        gemm_args_chip_0x01[9] = K;   // K
        gemm_args_chip_0x01[10] = N;  // N
        // SUs
        gemm_args_chip_0x01[11] = array_shape;
        // transpose A
        gemm_args_chip_0x01[12] = transposed_A;
        // transpose B
        gemm_args_chip_0x01[13] = transposed_B;
        // accumPrevC
        gemm_args_chip_0x01[14] = accumPrevC;
    }

    if (current_chip_id == 0x10) {
        printf("L3 Heap Manager is at 0x%lx\r\n",
               bingo_get_l3_heap_manager(current_chip_id));
        printf("gemm_args_chip_0x10 is located at: %x\r\n",
               &gemm_args_chip_0x10);
        gemm_args_chip_0x10 = (uint32_t*)o1heapAllocate(
            bingo_get_l3_heap_manager(get_current_chip_id()),
            sizeof(uint32_t) * 15);
        // versacore args
        // A matrix
        gemm_args_chip_0x10[0] = HIGH32(&A3[0]);
        gemm_args_chip_0x10[1] = LOW32(&A3[0]);
        // B matrix
        gemm_args_chip_0x10[2] = HIGH32(&B[0]);
        gemm_args_chip_0x10[3] = LOW32(&B[0]);
        // C matrix
        if (accumPrevC || addZeroC) {
            // When accumPrevC is true, we use D as the previous C matrix
            gemm_args_chip_0x10[4] = 0;
            gemm_args_chip_0x10[5] = 0;
        } else {
            gemm_args_chip_0x10[4] = HIGH32(&C3[0]);
            gemm_args_chip_0x10[5] = LOW32(&C3[0]);
        }

        // D matrix (output)
        O1HeapInstance* local_l3_heap_manager =
            bingo_get_l3_heap_manager(current_chip_id);
        output_data_addr_chip_0x10 = (uintptr_t)o1heapAllocate(
            local_l3_heap_manager, ARRAY_SIZE_BYTES(D3));
        gemm_args_chip_0x10[6] = HIGH32(output_data_addr_chip_0x10);
        gemm_args_chip_0x10[7] = LOW32(output_data_addr_chip_0x10);
        // Matrix dimensions
        gemm_args_chip_0x10[8] = M;   // M
        gemm_args_chip_0x10[9] = K;   // K
        gemm_args_chip_0x10[10] = N;  // N
        // SUs
        gemm_args_chip_0x10[11] = array_shape;
        // transpose A
        gemm_args_chip_0x10[12] = transposed_A;
        // transpose B
        gemm_args_chip_0x10[13] = transposed_B;
        // accumPrevC
        gemm_args_chip_0x10[14] = accumPrevC;
    }

    if (current_chip_id == 0x11) {
        printf("L3 Heap Manager is at 0x%lx\r\n",
               bingo_get_l3_heap_manager(current_chip_id));
        printf("gemm_args_chip_0x11 is located at: %x\r\n",
               &gemm_args_chip_0x11);
        gemm_args_chip_0x11 = (uint32_t*)o1heapAllocate(
            bingo_get_l3_heap_manager(get_current_chip_id()),
            sizeof(uint32_t) * 15);
        // versacore args
        // A matrix
        gemm_args_chip_0x11[0] = HIGH32(&A4[0]);
        gemm_args_chip_0x11[1] = LOW32(&A4[0]);
        // B matrix
        gemm_args_chip_0x11[2] = HIGH32(&B[0]);
        gemm_args_chip_0x11[3] = LOW32(&B[0]);
        // C matrix
        if (accumPrevC || addZeroC) {
            // When accumPrevC is true, we use D as the previous C matrix
            gemm_args_chip_0x11[4] = 0;
            gemm_args_chip_0x11[5] = 0;
        } else {
            gemm_args_chip_0x11[4] = HIGH32(&C4[0]);
            gemm_args_chip_0x11[5] = LOW32(&C4[0]);
        }

        // D matrix (output)
        O1HeapInstance* local_l3_heap_manager =
            bingo_get_l3_heap_manager(current_chip_id);
        output_data_addr_chip_0x11 = (uintptr_t)o1heapAllocate(
            local_l3_heap_manager, ARRAY_SIZE_BYTES(D4));
        gemm_args_chip_0x11[6] = HIGH32(output_data_addr_chip_0x11);
        gemm_args_chip_0x11[7] = LOW32(output_data_addr_chip_0x11);
        // Matrix dimensions
        gemm_args_chip_0x11[8] = M;   // M
        gemm_args_chip_0x11[9] = K;   // K
        gemm_args_chip_0x11[10] = N;  // N
        // SUs
        gemm_args_chip_0x11[11] = array_shape;
        // transpose A
        gemm_args_chip_0x11[12] = transposed_A;
        // transpose B
        gemm_args_chip_0x11[13] = transposed_B;
        // accumPrevC
        gemm_args_chip_0x11[14] = accumPrevC;
    }

    if (current_chip_id == 0x00) {
        printf(
            "Chip(%x, %x): [Host] Preparing gemm args for chiplet 0x%02x\r\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(),
            current_chip_id);
        printf("gemm_args_chip_0x00 addr high: 0x%x\r\n",
               HIGH32((uintptr_t)gemm_args_chip_0x00));
        printf("gemm_args_chip_0x00 addr low : 0x%x\r\n",
               LOW32((uintptr_t)gemm_args_chip_0x00));
    }
    if (current_chip_id == 0x01) {
        printf(
            "Chip(%x, %x): [Host] Preparing gemm args for chiplet 0x%02x\r\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(),
            current_chip_id);
        printf("gemm_args_chip_0x01 addr high: 0x%x\r\n",
               HIGH32((uintptr_t)gemm_args_chip_0x01));
        printf("gemm_args_chip_0x01 addr low : 0x%x\r\n",
               LOW32((uintptr_t)gemm_args_chip_0x01));
    }

    if (current_chip_id == 0x10) {
        printf(
            "Chip(%x, %x): [Host] Preparing gemm args for chiplet 0x%02x\r\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(),
            current_chip_id);
        printf("gemm_args_chip_0x10 addr high: 0x%x\r\n",
               HIGH32((uintptr_t)gemm_args_chip_0x10));
        printf("gemm_args_chip_0x10 addr low : 0x%x\r\n",
               LOW32((uintptr_t)gemm_args_chip_0x10));
    }

    if (current_chip_id == 0x11) {
        printf(
            "Chip(%x, %x): [Host] Preparing gemm args for chiplet 0x%02x\r\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(),
            current_chip_id);
        printf("gemm_args_chip_0x11 addr high: 0x%x\r\n",
               HIGH32((uintptr_t)gemm_args_chip_0x11));
        printf("gemm_args_chip_0x11 addr low : 0x%x\r\n",
               LOW32((uintptr_t)gemm_args_chip_0x11));
    }

    __snax_kernel_check_results_args_t task_check_results_args_chip_0x00;
    __snax_kernel_check_results_args_t task_check_results_args_chip_0x01;
    __snax_kernel_check_results_args_t task_check_results_args_chip_0x10;
    __snax_kernel_check_results_args_t task_check_results_args_chip_0x11;

    // checkresults args
    if (current_chip_id == 0x00) {
        task_check_results_args_chip_0x00.golden_data_addr =
            (uint32_t)(uintptr_t)(&D1[0]);
        task_check_results_args_chip_0x00.output_data_addr =
            (uint32_t)(uintptr_t)(output_data_addr_chip_0x00);
        task_check_results_args_chip_0x00.data_size = ARRAY_SIZE_BYTES(D1);
    }

    if (current_chip_id == 0x01) {
        task_check_results_args_chip_0x01.golden_data_addr =
            (uint32_t)(uintptr_t)(&D2[0]);
        task_check_results_args_chip_0x01.output_data_addr =
            (uint32_t)(uintptr_t)(output_data_addr_chip_0x01);
        task_check_results_args_chip_0x01.data_size = ARRAY_SIZE_BYTES(D2);
    }

    if (current_chip_id == 0x10) {
        task_check_results_args_chip_0x10.golden_data_addr =
            (uint32_t)(uintptr_t)(&D3[0]);
        task_check_results_args_chip_0x10.output_data_addr =
            (uint32_t)(uintptr_t)(output_data_addr_chip_0x10);
        task_check_results_args_chip_0x10.data_size = ARRAY_SIZE_BYTES(D3);
    }

    if (current_chip_id == 0x11) {
        task_check_results_args_chip_0x11.golden_data_addr =
            (uint32_t)(uintptr_t)(&D4[0]);
        task_check_results_args_chip_0x11.output_data_addr =
            (uint32_t)(uintptr_t)(output_data_addr_chip_0x11);
        task_check_results_args_chip_0x11.data_size = ARRAY_SIZE_BYTES(D4);
    }

    // 2. Register the tasks
    bingo_task_t* task_versacore_chip_0x00 = bingo_task_create(
        __snax_kernel_gemm_intra_chiplet_func_addr,
        (uint32_t)(uintptr_t)(gemm_args_chip_0x00), 0x00, cluster_id);
    if (task_versacore_chip_0x00 == NULL) {
        printf("Error: Task versacore creation failed!\r\n");
    }
    bingo_task_t* task_check_results_chip_0x00 = bingo_task_create(
        check_results_func_addr,
        (uint32_t)(uintptr_t)(&task_check_results_args_chip_0x00), 0x00,
        cluster_id);
    if (task_check_results_chip_0x00 == NULL) {
        printf("Error: Task check results creation failed!\r\n");
    }

    bingo_task_t* task_versacore_chip_0x01 = bingo_task_create(
        __snax_kernel_gemm_intra_chiplet_func_addr,
        (uint32_t)(uintptr_t)(gemm_args_chip_0x01), 0x01, cluster_id);
    if (task_versacore_chip_0x01 == NULL) {
        printf("Error: Task versacore creation failed!\r\n");
    }
    bingo_task_t* task_check_results_chip_0x01 = bingo_task_create(
        check_results_func_addr,
        (uint32_t)(uintptr_t)(&task_check_results_args_chip_0x01), 0x01,
        cluster_id);
    if (task_check_results_chip_0x01 == NULL) {
        printf("Error: Task check results creation failed!\r\n");
    }

    bingo_task_t* task_versacore_chip_0x10 = bingo_task_create(
        __snax_kernel_gemm_intra_chiplet_func_addr,
        (uint32_t)(uintptr_t)(gemm_args_chip_0x10), 0x10, cluster_id);
    if (task_versacore_chip_0x10 == NULL) {
        printf("Error: Task versacore creation failed!\r\n");
    }
    bingo_task_t* task_check_results_chip_0x10 = bingo_task_create(
        check_results_func_addr,
        (uint32_t)(uintptr_t)(&task_check_results_args_chip_0x10), 0x10,
        cluster_id);
    if (task_check_results_chip_0x10 == NULL) {
        printf("Error: Task check results creation failed!\r\n");
    }

    bingo_task_t* task_versacore_chip_0x11 = bingo_task_create(
        __snax_kernel_gemm_intra_chiplet_func_addr,
        (uint32_t)(uintptr_t)(gemm_args_chip_0x11), 0x11, cluster_id);
    if (task_versacore_chip_0x11 == NULL) {
        printf("Error: Task versacore creation failed!\r\n");
    }
    bingo_task_t* task_check_results_chip_0x11 = bingo_task_create(
        check_results_func_addr,
        (uint32_t)(uintptr_t)(&task_check_results_args_chip_0x11), 0x11,
        cluster_id);
    if (task_check_results_chip_0x11 == NULL) {
        printf("Error: Task check results creation failed!\r\n");
    }

    // 3. Set the task dependency
    // Here we have only two tasks and simple dependency
    // versacore -> check results
    bingo_task_add_depend(task_check_results_chip_0x00,
                          task_versacore_chip_0x00);
    bingo_task_add_depend(task_check_results_chip_0x01,
                          task_versacore_chip_0x01);
    bingo_task_add_depend(task_check_results_chip_0x10,
                          task_versacore_chip_0x10);
    bingo_task_add_depend(task_check_results_chip_0x11,
                          task_versacore_chip_0x11);

    // 4. Flush the caches to make sure the device can see the latest arguments
    asm volatile("fence" ::: "memory");
    //////////////////////
    // End main function //
    //////////////////////

    // Prepare the task list to return
    task_list[0] = task_versacore_chip_0x00;
    task_list[1] = task_check_results_chip_0x00;
    task_list[2] = task_versacore_chip_0x01;
    task_list[3] = task_check_results_chip_0x01;
    task_list[4] = task_versacore_chip_0x10;
    task_list[5] = task_check_results_chip_0x10;
    task_list[6] = task_versacore_chip_0x11;
    task_list[7] = task_check_results_chip_0x11;
    uint32_t num_tasks = 8;

    output_data_ptr = (uintptr_t*)output_data_addr_chip_0x00;
    return num_tasks;
}
