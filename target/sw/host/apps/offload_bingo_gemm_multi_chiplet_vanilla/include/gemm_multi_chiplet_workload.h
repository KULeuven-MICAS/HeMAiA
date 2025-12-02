// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>

#pragma once
#include "gemm_multi_chiplet_data.h"
#include "host.h"
#include "libbingo/bingo_api.h"


#ifdef HOST_DEBUG
#define HOST_DEBUG_PRINT(...) HOST_DEBUG_PRINT(__VA_ARGS__)
#else
#define HOST_DEBUG_PRINT(...)
#endif

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

uint32_t __workload_versacore_multi_chiplet(bingo_task_t** task_list) {
    ///////////////////
    // Main function //
    ///////////////////
    // 1. Get the kernel function address by the kernel name and prepare the
    // args
    // 2. Register the tasks
    // 3. Set the task dependency
    // 4. Set the assigned chiplet id and cluster id

    uint8_t cluster_id = 0;  // versacore is located at cluster

    // 1.1 Get the kernel function address by the kernel name
    check_kernel_tab_ready();
    uint32_t check_results_func_addr =
        get_device_function("__snax_kernel_check_results");
    uint32_t __snax_kernel_gemm_func_addr =
        get_device_function("__snax_kernel_gemm");
    if (check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        __snax_kernel_gemm_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        HOST_DEBUG_PRINT("Error: Kernel symbol lookup failed!\r\n");
    }
    __snax_kernel_check_results_args_t* task_check_results_args_chip_0x00 = NULL ;
    __snax_kernel_check_results_args_t* task_check_results_args_chip_0x01 = NULL ;
    __snax_kernel_check_results_args_t* task_check_results_args_chip_0x10 = NULL ;
    __snax_kernel_check_results_args_t* task_check_results_args_chip_0x11 = NULL ;

    uint32_t* gemm_args_chip_0x00 = NULL;
    uint32_t* gemm_args_chip_0x01 = NULL;
    uint32_t* gemm_args_chip_0x10 = NULL;
    uint32_t* gemm_args_chip_0x11 = NULL;
    // 1.2 Prepare the args for Chiplet 0
    if (get_current_chip_id() == 0x00) {
        gemm_args_chip_0x00 = (uint32_t*)o1heapAllocate(
            bingo_get_l3_heap_manager(get_current_chip_id()),
            sizeof(uint32_t) * 15);
        // versacore args
        // A matrix
        gemm_args_chip_0x00[0] = HIGH32(BINGO_CHIPLET_LOCAL_PTR_AUTO(A1[0]));
        gemm_args_chip_0x00[1] = LOW32(BINGO_CHIPLET_LOCAL_PTR_AUTO(A1[0]));
        // B matrix
        gemm_args_chip_0x00[2] = HIGH32(BINGO_CHIPLET_LOCAL_PTR_AUTO(B[0]));
        gemm_args_chip_0x00[3] = LOW32(BINGO_CHIPLET_LOCAL_PTR_AUTO(B[0]));
        // C matrix
        if (BINGO_CHIPLET_READW(accumPrevC) || BINGO_CHIPLET_READW(addZeroC)) {
            // When accumPrevC is true, we use D as the previous C matrix
            gemm_args_chip_0x00[4] = 0;
            gemm_args_chip_0x00[5] = 0;
        } else {
            gemm_args_chip_0x00[4] = HIGH32(BINGO_CHIPLET_LOCAL_PTR_AUTO(C1[0]));
            gemm_args_chip_0x00[5] = LOW32(BINGO_CHIPLET_LOCAL_PTR_AUTO(C1[0]));
        }

        // D matrix (output)
        uint64_t output_data_addr_chip_0x00 = o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), ARRAY_SIZE_BYTES(D1));
        gemm_args_chip_0x00[6] = HIGH32(BINGO_CHIPLET_READD(output_data_addr_chip_0x00));
        gemm_args_chip_0x00[7] = LOW32(BINGO_CHIPLET_READD(output_data_addr_chip_0x00));
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

        HOST_DEBUG_PRINT(
            "Chip(%x, %x): [Host] Preparing gemm args for chiplet 0x%x\r\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(),
            get_current_chip_id());
        HOST_DEBUG_PRINT("gemm_args_chip_0x00 A addr high: 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x00[0]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x00 A addr low : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x00[1]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x00 B addr high: 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x00[2]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x00 B addr low : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x00[3]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x00 C addr high: 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x00[4]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x00 C addr low : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x00[5]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x00 D addr high: 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x00[6]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x00 D addr low : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x00[7]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x00 M          : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x00[8]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x00 K          : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x00[9]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x00 N          : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x00[10]));
        task_check_results_args_chip_0x00 = (__snax_kernel_check_results_args_t*)o1heapAllocate(
            bingo_get_l3_heap_manager(get_current_chip_id()),
            sizeof(__snax_kernel_check_results_args_t));
        task_check_results_args_chip_0x00->golden_data_addr = LOW32(BINGO_CHIPLET_LOCAL_PTR_AUTO(D1[0]));
        task_check_results_args_chip_0x00->output_data_addr = LOW32(BINGO_CHIPLET_READD(output_data_addr_chip_0x00));
        task_check_results_args_chip_0x00->data_size = ARRAY_SIZE_BYTES(D1);
        HOST_DEBUG_PRINT("golden_data_addr: 0x%x\r\n",
               task_check_results_args_chip_0x00->golden_data_addr);
        HOST_DEBUG_PRINT("output_data_addr: 0x%x\r\n",
               task_check_results_args_chip_0x00->output_data_addr);
        HOST_DEBUG_PRINT("data_size      : 0x%x\r\n",
               task_check_results_args_chip_0x00->data_size);
    }

    if (get_current_chip_id() == 0x01) {
        gemm_args_chip_0x01 = (uint32_t*)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(uint32_t) * 15);
        // versacore args
        // A matrix
        gemm_args_chip_0x01[0] = HIGH32(BINGO_CHIPLET_LOCAL_PTR_AUTO(A2[0]));
        gemm_args_chip_0x01[1] = LOW32(BINGO_CHIPLET_LOCAL_PTR_AUTO(A2[0]));
        // B matrix
        gemm_args_chip_0x01[2] = HIGH32(BINGO_CHIPLET_LOCAL_PTR_AUTO(B[0]));
        gemm_args_chip_0x01[3] = LOW32(BINGO_CHIPLET_LOCAL_PTR_AUTO(B[0]));
        // C matrix
        if (BINGO_CHIPLET_READW(accumPrevC) || BINGO_CHIPLET_READW(addZeroC)) {
            // When accumPrevC is true, we use D as the previous C matrix
            gemm_args_chip_0x01[4] = 0;
            gemm_args_chip_0x01[5] = 0;
        } else {
            gemm_args_chip_0x01[4] = HIGH32(BINGO_CHIPLET_LOCAL_PTR_AUTO(C2[0]));
            gemm_args_chip_0x01[5] = LOW32(BINGO_CHIPLET_LOCAL_PTR_AUTO(C2[0]));
        }

        // D matrix (output)
        uint64_t output_data_addr_chip_0x01 = o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), ARRAY_SIZE_BYTES(D2));
        gemm_args_chip_0x01[6] = HIGH32(output_data_addr_chip_0x01);
        gemm_args_chip_0x01[7] = LOW32(output_data_addr_chip_0x01);
        // Matrix dimensions
        gemm_args_chip_0x01[8] = BINGO_CHIPLET_READW(M);   // M
        gemm_args_chip_0x01[9] = BINGO_CHIPLET_READW(K);   // K
        gemm_args_chip_0x01[10] = BINGO_CHIPLET_READW(N);  // N
        // SUs
        gemm_args_chip_0x01[11] = BINGO_CHIPLET_READW(array_shape);
        // transpose A
        gemm_args_chip_0x01[12] = BINGO_CHIPLET_READW(transposed_A);
        // transpose B
        gemm_args_chip_0x01[13] = BINGO_CHIPLET_READW(transposed_B);
        // accumPrevC
        gemm_args_chip_0x01[14] = BINGO_CHIPLET_READW(accumPrevC);
        HOST_DEBUG_PRINT(
            "Chip(%x, %x): [Host] Preparing gemm args for chiplet 0x%x\r\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(),
            get_current_chip_id());
        HOST_DEBUG_PRINT("gemm_args_chip_0x01 A addr high: 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x01[0]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x01 A addr low : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x01[1]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x01 B addr high: 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x01[2]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x01 B addr low : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x01[3]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x01 C addr high: 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x01[4]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x01 C addr low : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x01[5]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x01 D addr high: 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x01[6]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x01 D addr low : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x01[7]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x01 M          : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x01[8]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x01 K          : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x01[9]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x01 N          : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x01[10]));
    
        task_check_results_args_chip_0x01 = ( __snax_kernel_check_results_args_t*)o1heapAllocate(
            bingo_get_l3_heap_manager(get_current_chip_id()),
            sizeof(__snax_kernel_check_results_args_t));
        task_check_results_args_chip_0x01->golden_data_addr = LOW32(BINGO_CHIPLET_LOCAL_PTR_AUTO(D2[0]));
        task_check_results_args_chip_0x01->output_data_addr = LOW32(BINGO_CHIPLET_READD(output_data_addr_chip_0x01));
        task_check_results_args_chip_0x01->data_size = ARRAY_SIZE_BYTES(D2);
        HOST_DEBUG_PRINT("golden_data_addr: 0x%x\r\n",
               task_check_results_args_chip_0x01->golden_data_addr);
        HOST_DEBUG_PRINT("output_data_addr: 0x%x\r\n",
               task_check_results_args_chip_0x01->output_data_addr);
        HOST_DEBUG_PRINT("data_size      : 0x%x\r\n",
               task_check_results_args_chip_0x01->data_size);
    }

    if (get_current_chip_id() == 0x10) {
        gemm_args_chip_0x10 = (uint32_t*)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(uint32_t) * 15);
        // versacore args
        // A matrix
        gemm_args_chip_0x10[0] = HIGH32(BINGO_CHIPLET_LOCAL_PTR_AUTO(A3[0]));
        gemm_args_chip_0x10[1] = LOW32(BINGO_CHIPLET_LOCAL_PTR_AUTO(A3[0]));
        // B matrix
        gemm_args_chip_0x10[2] = HIGH32(BINGO_CHIPLET_LOCAL_PTR_AUTO(B[0]));
        gemm_args_chip_0x10[3] = LOW32(BINGO_CHIPLET_LOCAL_PTR_AUTO(B[0]));
        // C matrix
        if (BINGO_CHIPLET_READW(accumPrevC) || BINGO_CHIPLET_READW(addZeroC)) {
            // When accumPrevC is true, we use D as the previous C matrix
            gemm_args_chip_0x10[4] = 0;
            gemm_args_chip_0x10[5] = 0;
        } else {
            gemm_args_chip_0x10[4] = HIGH32(BINGO_CHIPLET_LOCAL_PTR_AUTO(C3[0]));
            gemm_args_chip_0x10[5] = LOW32(BINGO_CHIPLET_LOCAL_PTR_AUTO(C3[0]));
        }

        // D matrix (output)
        uint64_t output_data_addr_chip_0x10 = o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), ARRAY_SIZE_BYTES(D3));
        gemm_args_chip_0x10[6] = HIGH32(output_data_addr_chip_0x10);
        gemm_args_chip_0x10[7] = LOW32(output_data_addr_chip_0x10);
        // Matrix dimensions
        gemm_args_chip_0x10[8] = BINGO_CHIPLET_READW(M);   // M
        gemm_args_chip_0x10[9] = BINGO_CHIPLET_READW(K);   // K
        gemm_args_chip_0x10[10] = BINGO_CHIPLET_READW(N);  // N
        // SUs
        gemm_args_chip_0x10[11] = BINGO_CHIPLET_READW(array_shape);
        // transpose A
        gemm_args_chip_0x10[12] = BINGO_CHIPLET_READW(transposed_A);
        // transpose B
        gemm_args_chip_0x10[13] = BINGO_CHIPLET_READW(transposed_B);
        // accumPrevC
        gemm_args_chip_0x10[14] = BINGO_CHIPLET_READW(accumPrevC);
        HOST_DEBUG_PRINT(
            "Chip(%x, %x): [Host] Preparing gemm args for chiplet 0x%x\r\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(),
            get_current_chip_id());
        HOST_DEBUG_PRINT("gemm_args_chip_0x10 A addr high: 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x10[0]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x10 A addr low : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x10[1]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x00 B addr high: 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x10[2]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x10 B addr low : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x10[3]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x10 C addr high: 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x10[4]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x10 C addr low : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x10[5]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x10 D addr high: 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x10[6]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x10 D addr low : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x10[7]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x10 M          : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x10[8]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x10 K          : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x10[9]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x10 N          : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x10[10]));
        task_check_results_args_chip_0x10 = ( __snax_kernel_check_results_args_t*)o1heapAllocate(
            bingo_get_l3_heap_manager(get_current_chip_id()),
            sizeof(__snax_kernel_check_results_args_t));
        task_check_results_args_chip_0x10->golden_data_addr = LOW32(BINGO_CHIPLET_LOCAL_PTR_AUTO(D3[0]));
        task_check_results_args_chip_0x10->output_data_addr = LOW32(BINGO_CHIPLET_READD(output_data_addr_chip_0x10));
        task_check_results_args_chip_0x10->data_size = ARRAY_SIZE_BYTES(D3);
        HOST_DEBUG_PRINT("golden_data_addr: 0x%x\r\n",
               task_check_results_args_chip_0x10->golden_data_addr);
        HOST_DEBUG_PRINT("output_data_addr: 0x%x\r\n",
               task_check_results_args_chip_0x10->output_data_addr);
        HOST_DEBUG_PRINT("data_size      : 0x%x\r\n",
               task_check_results_args_chip_0x10->data_size);
    }

    if (get_current_chip_id() == 0x11) {
        gemm_args_chip_0x11 = (uint32_t*)o1heapAllocate(
            bingo_get_l3_heap_manager(get_current_chip_id()),
            sizeof(uint32_t) * 15);
        // versacore args
        // A matrix
        gemm_args_chip_0x11[0] = HIGH32(BINGO_CHIPLET_LOCAL_PTR_AUTO(A4[0]));
        gemm_args_chip_0x11[1] = LOW32(BINGO_CHIPLET_LOCAL_PTR_AUTO(A4[0]));
        // B matrix
        gemm_args_chip_0x11[2] = HIGH32(BINGO_CHIPLET_LOCAL_PTR_AUTO(B[0]));
        gemm_args_chip_0x11[3] = LOW32(BINGO_CHIPLET_LOCAL_PTR_AUTO(B[0]));
        // C matrix
        if (BINGO_CHIPLET_READW(accumPrevC) || BINGO_CHIPLET_READW(addZeroC)) {
            // When accumPrevC is true, we use D as the previous C matrix
            gemm_args_chip_0x11[4] = 0;
            gemm_args_chip_0x11[5] = 0;
        } else {
            gemm_args_chip_0x11[4] = HIGH32(BINGO_CHIPLET_LOCAL_PTR_AUTO(C4[0]));
            gemm_args_chip_0x11[5] = LOW32(BINGO_CHIPLET_LOCAL_PTR_AUTO(C4[0]));
        }

        // D matrix (output)
        uint64_t output_data_addr_chip_0x11 = o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), ARRAY_SIZE_BYTES(D4));
        gemm_args_chip_0x11[6] = HIGH32(output_data_addr_chip_0x11);
        gemm_args_chip_0x11[7] = LOW32(output_data_addr_chip_0x11);
        // Matrix dimensions
        gemm_args_chip_0x11[8] = BINGO_CHIPLET_READW(M);   // M
        gemm_args_chip_0x11[9] = BINGO_CHIPLET_READW(K);   // K
        gemm_args_chip_0x11[10] = BINGO_CHIPLET_READW(N);  // N
        // SUs
        gemm_args_chip_0x11[11] = BINGO_CHIPLET_READW(array_shape);
        // transpose A
        gemm_args_chip_0x11[12] = BINGO_CHIPLET_READW(transposed_A);
        // transpose B
        gemm_args_chip_0x11[13] = BINGO_CHIPLET_READW(transposed_B);
        // accumPrevC
        gemm_args_chip_0x11[14] = BINGO_CHIPLET_READW(accumPrevC);
        HOST_DEBUG_PRINT(
            "Chip(%x, %x): [Host] Preparing gemm args for chiplet 0x%x\r\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(),
            get_current_chip_id());
        HOST_DEBUG_PRINT("gemm_args_chip_0x11 A addr high: 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x11[0]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x11 A addr low : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x11[1]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x11 B addr high: 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x11[2]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x11 B addr low : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x11[3]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x11 C addr high: 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x11[4]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x11 C addr low : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x11[5]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x11 D addr high: 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x11[6]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x11 D addr low : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x11[7]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x11 M          : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x11[8]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x11 K          : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x11[9]));
        HOST_DEBUG_PRINT("gemm_args_chip_0x11 N          : 0x%x\r\n",
               BINGO_CHIPLET_READW(gemm_args_chip_0x11[10]));
        task_check_results_args_chip_0x11 = ( __snax_kernel_check_results_args_t*)o1heapAllocate(
            bingo_get_l3_heap_manager(get_current_chip_id()),
            sizeof(__snax_kernel_check_results_args_t));
        task_check_results_args_chip_0x11->golden_data_addr = LOW32(BINGO_CHIPLET_LOCAL_PTR_AUTO(D4[0]));
        task_check_results_args_chip_0x11->output_data_addr = LOW32(BINGO_CHIPLET_READD(output_data_addr_chip_0x11));
        task_check_results_args_chip_0x11->data_size = ARRAY_SIZE_BYTES(D4);
        HOST_DEBUG_PRINT("golden_data_addr: 0x%x\r\n",
               task_check_results_args_chip_0x11->golden_data_addr);
        HOST_DEBUG_PRINT("output_data_addr: 0x%x\r\n",
               task_check_results_args_chip_0x11->output_data_addr);
        HOST_DEBUG_PRINT("data_size      : 0x%x\r\n",
               task_check_results_args_chip_0x11->data_size);
    }


    // 2. Register the tasks
    // Chiplet 00
    bingo_task_t* task_versacore_chip_0x00 = bingo_task_create(
        __snax_kernel_gemm_func_addr,
        (uint32_t)(uintptr_t)(gemm_args_chip_0x00), 0x00, cluster_id);
    if (task_versacore_chip_0x00 == NULL) {
        printf("Error: Task versacore creation failed!\r\n");
    }

    bingo_task_t* task_check_results_chip_0x00 = bingo_task_create(
        check_results_func_addr,
        (uint32_t)(uintptr_t)task_check_results_args_chip_0x00,
        0x00,
        cluster_id);
    if (task_check_results_chip_0x00 == NULL) {
        printf("Error: Task check results creation failed!\r\n");
    }

    // Chiplet 01
    bingo_task_t* task_versacore_chip_0x01 = bingo_task_create(
        __snax_kernel_gemm_func_addr,
        (uint32_t)(uintptr_t)(gemm_args_chip_0x01), 0x01, cluster_id);
    if (task_versacore_chip_0x01 == NULL) {
        printf("Error: Task versacore creation failed!\r\n");
    }

    bingo_task_t* task_check_results_chip_0x01 = bingo_task_create(
        check_results_func_addr,
        (uint32_t)(uintptr_t)task_check_results_args_chip_0x01,
        0x01,
        cluster_id);
    if (task_check_results_chip_0x01 == NULL) {
        printf("Error: Task check results creation failed!\r\n");
    }

    // Chiplet 10
    bingo_task_t* task_versacore_chip_0x10 = bingo_task_create(
        __snax_kernel_gemm_func_addr,
        (uint32_t)(uintptr_t)(gemm_args_chip_0x10),
        0x10,
        cluster_id);
    if (task_versacore_chip_0x10 == NULL) {
        printf("Error: Task versacore creation failed!\r\n");
    }

    bingo_task_t* task_check_results_chip_0x10 = bingo_task_create(
        check_results_func_addr,
        (uint32_t)(uintptr_t)task_check_results_args_chip_0x10,
        0x10,
        cluster_id);
    if (task_check_results_chip_0x10 == NULL) {
        printf("Error: Task check results creation failed!\r\n");
    }
    
    // Chiplet 11
    bingo_task_t* task_versacore_chip_0x11 = bingo_task_create(
        __snax_kernel_gemm_func_addr,
        (uint32_t)(uintptr_t)(gemm_args_chip_0x11),
        0x11,
        cluster_id);
    if (task_versacore_chip_0x11 == NULL) {
        printf("Error: Task versacore creation failed!\r\n");
    }
    bingo_task_t* task_check_results_chip_0x11 = bingo_task_create(
        check_results_func_addr,
        (uint32_t)(uintptr_t)task_check_results_args_chip_0x11,
        0x11,
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
    for (uint32_t i = 0; i < num_tasks; i++) {
        HOST_DEBUG_PRINT("Chip(%x, %x): [Host] Registered task %d with "
               "task_id=0x%x on chiplet 0x%x\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), i,
               (uint32_t)task_list[i]->task_id,
               (uint32_t)task_list[i]->assigned_chip_id);
    }
    return num_tasks;
}
