// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>

#pragma once
#include "gemm_multi_chiplet_data.h"
#include "host.h"
#include "libbingo/bingo_api.h"

// This workload tests the multi-chiplet co-work for a large GEMM task
// 1. each chiplet's snitch xdma copy A1  A2, A3, A4 to chiplet 0x00, 0x01,
// 0x10, 0x11 respectively
// 2. chiplet 0x00's snitch xdma copy B to chiplet 0x00
// 3. chiplet 0x00's snitch idma broadcast B to all remaining chiplets
// 3. each chiplet does gemm for its own A and the broadcasted B
// 4. each chiplet's snitch xdma copy the corresponding result D back to host
// memory Note: C matrix is not used in this workload, we set accumPrevC to
// false and addZeroC to true The correctness is checked by comparing D with the
// golden result at each chiplet

uint32_t __workload_versacore_multi_chiplet_broadcast(
    bingo_task_t** task_list, uintptr_t* output_data_ptr) {
    ///////////////////
    // Main function //
    ///////////////////
    // 1. Get the kernel function address by the kernel name
    // 2. Prepare the args
    // 3. Register the tasks
    // 4. Set the task dependency
    // 5. Set the assigned chiplet id and cluster id

    uint8_t current_chip_id = get_current_chip_id();
    uint8_t cluster_id = 0;  // versacore is located at cluster

    // 1 Get the needed kernel function address by the kernel name
    check_kernel_tab_ready();
    uint32_t __snax_kernel_xdma_1d_copy_func_addr =
        get_device_function("__snax_kernel_xdma_1d_copy");
    uint32_t __snax_kernel_idma_1d_copy_func_addr =
        get_device_function("__snax_kernel_idma_1d_copy");
    uint32_t __snax_kernel_gemm_compute_only_intra_chiplet_func_addr =
        get_device_function("__snax_kernel_gemm_compute_only_intra_chiplet");
    uint32_t check_results_func_addr =
        get_device_function("__snax_kernel_check_results");
    if (__snax_kernel_xdma_1d_copy_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        __snax_kernel_idma_1d_copy_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        __snax_kernel_gemm_compute_only_intra_chiplet_func_addr ==
            SNAX_SYMTAB_END_FN_ADDR ||
        check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf("Error: Kernel symbol lookup failed!\r\n");
    }

    // 2 Prepare the args
    ///////////////////
    ///////////////////
    // args for xdma cp for loading A matrices from mempool to L1 at each
    // chiplet// src the mempool address for A1, A2, A3, A4 dst the L1 address
    // in each cluster size the size in bytes
    ///////////////////
    ///////////////////
    __snax_kernel_xdma_1d_copy_args_t task_mempool_to_cluster_args_A1_chip_0x00;
    __snax_kernel_xdma_1d_copy_args_t task_mempool_to_cluster_args_A2_chip_0x01;
    __snax_kernel_xdma_1d_copy_args_t task_mempool_to_cluster_args_A3_chip_0x10;
    __snax_kernel_xdma_1d_copy_args_t task_mempool_to_cluster_args_A4_chip_0x11;

    volatile uintptr_t A = chiplet_addr_transform_loc(
        2, 0, 0x80000000);          // L3 mempool base address
    volatile uintptr_t A1 = A + 0;  // A1 at chiplet 0x00
    volatile uintptr_t A2 = A1 + M * K * meshRow * tileSize *
                                     sizeof(uint8_t);  // A2 at chiplet 0x01
    volatile uintptr_t A3 = A2 + M * K * meshRow * tileSize *
                                     sizeof(uint8_t);  // A3 at chiplet 0x10
    volatile uintptr_t A4 = A3 + M * K * meshRow * tileSize *
                                     sizeof(uint8_t);  // A4 at chiplet 0x11

    for (int i = 0; i < 5; i++) {
        printf("Chip(%x): A1[%d] = %d, A2[%d] = %d, A3[%d] = %d, A4[%d] = %d\r\n",
               current_chip_id, i,
               *((uint8_t*)(uintptr_t)(A1 + i * sizeof(uint8_t))), i,
               *((uint8_t*)(uintptr_t)(A2 + i * sizeof(uint8_t))), i,
               *((uint8_t*)(uintptr_t)(A3 + i * sizeof(uint8_t))), i,
               *((uint8_t*)(uintptr_t)(A4 + i * sizeof(uint8_t))) );
    }

    // the A matrix load destination address in cluster L1 for each chiplet
    uintptr_t cluster_l1_addr_A =
        chiplet_addr_transform(0x10001000);  // predefined

    // Prepare the args for loading A1 at chiplet 0x00
    if (current_chip_id == 0x00) {
        task_mempool_to_cluster_args_A1_chip_0x00.src_addr_hi = HIGH32(A1);
        task_mempool_to_cluster_args_A1_chip_0x00.src_addr_lo = LOW32(A1);
        task_mempool_to_cluster_args_A1_chip_0x00.dst_addr_hi =
            HIGH32(cluster_l1_addr_A);
        task_mempool_to_cluster_args_A1_chip_0x00.dst_addr_lo =
            LOW32(cluster_l1_addr_A);
        task_mempool_to_cluster_args_A1_chip_0x00.size =
            M * K * meshRow * tileSize * sizeof(uint8_t);
    }

    // Prepare the args for loading A2 at chiplet 0x01
    if (current_chip_id == 0x01) {
        task_mempool_to_cluster_args_A2_chip_0x01.src_addr_hi = HIGH32(A2);
        task_mempool_to_cluster_args_A2_chip_0x01.src_addr_lo = LOW32(A2);
        task_mempool_to_cluster_args_A2_chip_0x01.dst_addr_hi =
            HIGH32(cluster_l1_addr_A);
        task_mempool_to_cluster_args_A2_chip_0x01.dst_addr_lo =
            LOW32(cluster_l1_addr_A);
        task_mempool_to_cluster_args_A2_chip_0x01.size =
            M * K * meshRow * tileSize * sizeof(uint8_t);
    }

    // Prepare the args for loading A3 at chiplet 0x10
    if (current_chip_id == 0x10) {
        task_mempool_to_cluster_args_A3_chip_0x10.src_addr_hi = HIGH32(A3);
        task_mempool_to_cluster_args_A3_chip_0x10.src_addr_lo = LOW32(A3);
        task_mempool_to_cluster_args_A3_chip_0x10.dst_addr_hi =
            HIGH32(cluster_l1_addr_A);
        task_mempool_to_cluster_args_A3_chip_0x10.dst_addr_lo =
            LOW32(cluster_l1_addr_A);
        task_mempool_to_cluster_args_A3_chip_0x10.size =
            M * K * meshRow * tileSize * sizeof(uint8_t);
    }

    // Prepare the args for loading A4 at chiplet 0x11
    if (current_chip_id == 0x11) {
        task_mempool_to_cluster_args_A4_chip_0x11.src_addr_hi = HIGH32(A4);
        task_mempool_to_cluster_args_A4_chip_0x11.src_addr_lo = LOW32(A4);
        task_mempool_to_cluster_args_A4_chip_0x11.dst_addr_hi =
            HIGH32(cluster_l1_addr_A);
        task_mempool_to_cluster_args_A4_chip_0x11.dst_addr_lo =
            LOW32(cluster_l1_addr_A);
        task_mempool_to_cluster_args_A4_chip_0x11.size =
            M * K * meshRow * tileSize * sizeof(uint8_t);
    }

    ///////////////////
    ///////////////////
    // args for xdma cp for loading B matrix from mempool to L1 at chiplet
    // 0x00// src the mempool address for B dst the L1 address in cluster 0 at
    // chiplet 0x00 size the size in bytes
    ///////////////////
    ///////////////////
    __snax_kernel_xdma_1d_copy_args_t task_mempool_to_cluster_args_B_chip_0x00;

    volatile uintptr_t B = A4 + M * K * meshRow * tileSize *
                                    sizeof(uint8_t);  // L3 mempool base address
    // the B matrix load destination address in cluster L1 for chiplet 0x00
    uintptr_t cluster_l1_addr_B = chiplet_addr_transform(0x10001100);  // predefined

    if (current_chip_id == 0x00) {
        task_mempool_to_cluster_args_B_chip_0x00.src_addr_hi = HIGH32(B);
        task_mempool_to_cluster_args_B_chip_0x00.src_addr_lo = LOW32(B);
        task_mempool_to_cluster_args_B_chip_0x00.dst_addr_hi =
            HIGH32(cluster_l1_addr_B);
        task_mempool_to_cluster_args_B_chip_0x00.dst_addr_lo =
            LOW32(cluster_l1_addr_B);
        task_mempool_to_cluster_args_B_chip_0x00.size =
            K * N * tileSize * meshCol * sizeof(uint8_t);
    }

    ///////////////////
    ///////////////////
    // args for idma broadcast B matrix from chiplet 0x00 to all other
    // chiplets// src the L1 address in cluster 0 at chiplet 0x00 dst the L1
    // address in each cluster at all chiplets size the size in bytes
    ///////////////////
    ///////////////////
    __snax_kernel_idma_1d_copy_args_t task_broadcast_args_B_chip_0x00;
    uintptr_t broadcast_dst_addr_B =
        chiplet_addr_transform_loc(0xF, 0xF, 0x10001100);  // predefined

    if (current_chip_id == 0x00) {
        task_broadcast_args_B_chip_0x00.src_addr_hi = HIGH32(cluster_l1_addr_B);
        task_broadcast_args_B_chip_0x00.src_addr_lo = LOW32(cluster_l1_addr_B);
        // destination address in cluster L1 for B matrix at each chiplet
        task_broadcast_args_B_chip_0x00.dst_addr_hi =
            HIGH32(broadcast_dst_addr_B);
        task_broadcast_args_B_chip_0x00.dst_addr_lo =
            LOW32(broadcast_dst_addr_B);
        task_broadcast_args_B_chip_0x00.size =
            K * N * tileSize * meshCol * sizeof(uint8_t);
    }

    ///////////////////
    ///////////////////
    // args for gemm at each chiplet//
    ///////////////////
    ///////////////////

    // args for gemm at each chiplet//
    uint32_t* gemm_args_chip_0x00 = (uint32_t*)NULL;
    uint32_t* gemm_args_chip_0x01 = (uint32_t*)NULL;
    uint32_t* gemm_args_chip_0x10 = (uint32_t*)NULL;
    uint32_t* gemm_args_chip_0x11 = (uint32_t*)NULL;
    // output data addr for outputed D matrix at each chiplet at L3
    uintptr_t cluster_l1_addr_C =
        chiplet_addr_transform(0x10001200);  // predefined
    uintptr_t cluster_l1_addr_D =
        chiplet_addr_transform(0x10002000);  // predefined

    // 1.2 Prepare the args for Chiplet 0
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

        // D matrix (output)
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

    if (current_chip_id == 0x01) {
        gemm_args_chip_0x01 = (uint32_t*)o1heapAllocate(
            bingo_get_l3_heap_manager(get_current_chip_id()),
            sizeof(uint32_t) * 15);
        // versacore args
        // A matrix
        gemm_args_chip_0x01[0] = HIGH32(cluster_l1_addr_A);
        gemm_args_chip_0x01[1] = LOW32(cluster_l1_addr_A);
        // B matrix
        gemm_args_chip_0x01[2] = HIGH32(cluster_l1_addr_B);
        gemm_args_chip_0x01[3] = LOW32(cluster_l1_addr_B);
        // C matrix
        if (accumPrevC || addZeroC) {
            // When accumPrevC is true, we use D as the previous C matrix
            gemm_args_chip_0x01[4] = 0;
            gemm_args_chip_0x01[5] = 0;
        } else {
            gemm_args_chip_0x01[4] = HIGH32(cluster_l1_addr_C);
            gemm_args_chip_0x01[5] = LOW32(cluster_l1_addr_C);
        }

        // D matrix (output)
        gemm_args_chip_0x01[6] = HIGH32(cluster_l1_addr_D);
        gemm_args_chip_0x01[7] = LOW32(cluster_l1_addr_D);
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
        gemm_args_chip_0x10 = (uint32_t*)o1heapAllocate(
            bingo_get_l3_heap_manager(get_current_chip_id()),
            sizeof(uint32_t) * 15);
        // versacore args
        // A matrix
        gemm_args_chip_0x10[0] = HIGH32(cluster_l1_addr_A);
        gemm_args_chip_0x10[1] = LOW32(cluster_l1_addr_A);
        // B matrix
        gemm_args_chip_0x10[2] = HIGH32(cluster_l1_addr_B);
        gemm_args_chip_0x10[3] = LOW32(cluster_l1_addr_B);
        // C matrix
        if (accumPrevC || addZeroC) {
            // When accumPrevC is true, we use D as the previous C matrix
            gemm_args_chip_0x10[4] = 0;
            gemm_args_chip_0x10[5] = 0;
        } else {
            gemm_args_chip_0x10[4] = HIGH32(cluster_l1_addr_C);
            gemm_args_chip_0x10[5] = LOW32(cluster_l1_addr_C);
        }

        // D matrix (output)
        gemm_args_chip_0x10[6] = HIGH32(cluster_l1_addr_D);
        gemm_args_chip_0x10[7] = LOW32(cluster_l1_addr_D);
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
        gemm_args_chip_0x11 = (uint32_t*)o1heapAllocate(
            bingo_get_l3_heap_manager(get_current_chip_id()),
            sizeof(uint32_t) * 15);
        // versacore args
        // A matrix
        gemm_args_chip_0x11[0] = HIGH32(cluster_l1_addr_A);
        gemm_args_chip_0x11[1] = LOW32(cluster_l1_addr_A);
        // B matrix
        gemm_args_chip_0x11[2] = HIGH32(cluster_l1_addr_B);
        gemm_args_chip_0x11[3] = LOW32(cluster_l1_addr_B);
        // C matrix
        if (accumPrevC || addZeroC) {
            // When accumPrevC is true, we use D as the previous C matrix
            gemm_args_chip_0x11[4] = 0;
            gemm_args_chip_0x11[5] = 0;
        } else {
            gemm_args_chip_0x11[4] = HIGH32(cluster_l1_addr_C);
            gemm_args_chip_0x11[5] = LOW32(cluster_l1_addr_C);
        }

        // D matrix (output)
        gemm_args_chip_0x11[6] = HIGH32(cluster_l1_addr_D);
        gemm_args_chip_0x11[7] = LOW32(cluster_l1_addr_D);
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

    ///////////////////
    ///////////////////
    // args for move output D from L1 to L3//
    ///////////////////
    ///////////////////
    __snax_kernel_xdma_1d_copy_args_t task_cluster_to_l3_args_D_chip_0x00;
    __snax_kernel_xdma_1d_copy_args_t task_cluster_to_l3_args_D_chip_0x01;
    __snax_kernel_xdma_1d_copy_args_t task_cluster_to_l3_args_D_chip_0x10;
    __snax_kernel_xdma_1d_copy_args_t task_cluster_to_l3_args_D_chip_0x11;

    volatile uintptr_t output_data_addr_chip_0x00 = (uintptr_t)NULL;

    if (current_chip_id == 0x00) {
        O1HeapInstance* local_l3_heap_manager =
            bingo_get_l3_heap_manager(current_chip_id);
        output_data_addr_chip_0x00 = (uintptr_t)o1heapAllocate(
            local_l3_heap_manager,
            M * N * meshRow * meshCol * sizeof(uint32_t));

        task_cluster_to_l3_args_D_chip_0x00.src_addr_hi =
            HIGH32(cluster_l1_addr_D);
        task_cluster_to_l3_args_D_chip_0x00.src_addr_lo =
            LOW32(cluster_l1_addr_D);
        task_cluster_to_l3_args_D_chip_0x00.dst_addr_hi =
            HIGH32((uintptr_t)(output_data_addr_chip_0x00));
        task_cluster_to_l3_args_D_chip_0x00.dst_addr_lo =
            LOW32((uintptr_t)(output_data_addr_chip_0x00));
        task_cluster_to_l3_args_D_chip_0x00.size =
            M * N * meshRow * meshCol * sizeof(uint32_t);
    }

    volatile uintptr_t output_data_addr_chip_0x01 = (uintptr_t)NULL;
    if (current_chip_id == 0x01) {
        O1HeapInstance* local_l3_heap_manager =
            bingo_get_l3_heap_manager(current_chip_id);
        output_data_addr_chip_0x01 = (uintptr_t)o1heapAllocate(
            local_l3_heap_manager,
            M * N * meshRow * meshCol * sizeof(uint32_t));

        task_cluster_to_l3_args_D_chip_0x01.src_addr_hi =
            HIGH32(cluster_l1_addr_D);
        task_cluster_to_l3_args_D_chip_0x01.src_addr_lo =
            LOW32(cluster_l1_addr_D);
        task_cluster_to_l3_args_D_chip_0x01.dst_addr_hi =
            HIGH32((uintptr_t)(output_data_addr_chip_0x01));
        task_cluster_to_l3_args_D_chip_0x01.dst_addr_lo =
            LOW32((uintptr_t)(output_data_addr_chip_0x01));
        task_cluster_to_l3_args_D_chip_0x01.size =
            M * N * meshRow * meshCol * sizeof(uint32_t);
    }

    volatile uintptr_t output_data_addr_chip_0x10 = (uintptr_t)NULL;
    if (current_chip_id == 0x10) {
        O1HeapInstance* local_l3_heap_manager =
            bingo_get_l3_heap_manager(current_chip_id);
        output_data_addr_chip_0x10 = (uintptr_t)o1heapAllocate(
            local_l3_heap_manager,
            M * N * meshRow * meshCol * sizeof(uint32_t));
        task_cluster_to_l3_args_D_chip_0x10.src_addr_hi =
            HIGH32(cluster_l1_addr_D);
        task_cluster_to_l3_args_D_chip_0x10.src_addr_lo =
            LOW32(cluster_l1_addr_D);
        task_cluster_to_l3_args_D_chip_0x10.dst_addr_hi =
            HIGH32((uintptr_t)(output_data_addr_chip_0x10));
        task_cluster_to_l3_args_D_chip_0x10.dst_addr_lo =
            LOW32((uintptr_t)(output_data_addr_chip_0x10));
        task_cluster_to_l3_args_D_chip_0x10.size =
            M * N * meshRow * meshCol * sizeof(uint32_t);
    }

    volatile uintptr_t output_data_addr_chip_0x11 = (uintptr_t)NULL;
    if (current_chip_id == 0x11) {
        O1HeapInstance* local_l3_heap_manager =
            bingo_get_l3_heap_manager(current_chip_id);
        output_data_addr_chip_0x11 = (uintptr_t)o1heapAllocate(
            local_l3_heap_manager,
            M * N * meshRow * meshCol * sizeof(uint32_t));
        task_cluster_to_l3_args_D_chip_0x11.src_addr_hi =
            HIGH32(cluster_l1_addr_D);
        task_cluster_to_l3_args_D_chip_0x11.src_addr_lo =
            LOW32(cluster_l1_addr_D);
        task_cluster_to_l3_args_D_chip_0x11.dst_addr_hi =
            HIGH32((uintptr_t)(output_data_addr_chip_0x11));
        task_cluster_to_l3_args_D_chip_0x11.dst_addr_lo =
            LOW32((uintptr_t)(output_data_addr_chip_0x11));
        task_cluster_to_l3_args_D_chip_0x11.size =
            M * N * meshRow * meshCol * sizeof(uint32_t);
    }

    ///////////////////
    ///////////////////
    // args for check results at each chiplet//
    ///////////////////
    ///////////////////
    __snax_kernel_check_results_args_t task_check_results_args_chip_0x00;
    __snax_kernel_check_results_args_t task_check_results_args_chip_0x01;
    __snax_kernel_check_results_args_t task_check_results_args_chip_0x10;
    __snax_kernel_check_results_args_t task_check_results_args_chip_0x11;

    volatile uintptr_t D = chiplet_addr_transform_loc(
        2, 0,
        B + (K * N * tileSize * meshCol *
             sizeof(uint8_t)));  // golden result for chiplet 0x00
    volatile uintptr_t D1 = D;    // D1 at chiplet 0x00
    volatile uintptr_t D2 = D1 + M * N * meshRow * meshCol *
                                     sizeof(uint32_t);  // D2 at chiplet 0x01
    volatile uintptr_t D3 = D2 + M * N * meshRow * meshCol *
                                     sizeof(uint32_t);  // D3 at chiplet 0x10
    volatile uintptr_t D4 = D3 + M * N * meshRow * meshCol *
                                     sizeof(uint32_t);  // D4 at chiplet 0x11

    // checkresults args
    if (current_chip_id == 0x00) {
        task_check_results_args_chip_0x00.golden_data_addr =
            (uint32_t)(uintptr_t)(D1);
        task_check_results_args_chip_0x00.output_data_addr =
            (uint32_t)(uintptr_t)(output_data_addr_chip_0x00);
        task_check_results_args_chip_0x00.data_size =
            M * N * meshRow * meshCol * sizeof(uint32_t);
    }

    if (current_chip_id == 0x01) {
        task_check_results_args_chip_0x01.golden_data_addr =
            (uint32_t)(uintptr_t)(D2);
        task_check_results_args_chip_0x01.output_data_addr =
            (uint32_t)(uintptr_t)(output_data_addr_chip_0x01);
        task_check_results_args_chip_0x01.data_size =
            M * N * meshRow * meshCol * sizeof(uint32_t);
    }

    if (current_chip_id == 0x10) {
        task_check_results_args_chip_0x10.golden_data_addr =
            (uint32_t)(uintptr_t)(D3);
        task_check_results_args_chip_0x10.output_data_addr =
            (uint32_t)(uintptr_t)(output_data_addr_chip_0x10);
        task_check_results_args_chip_0x10.data_size =
            M * N * meshRow * meshCol * sizeof(uint32_t);
    }

    if (current_chip_id == 0x11) {
        task_check_results_args_chip_0x11.golden_data_addr =
            (uint32_t)(uintptr_t)(D4);
        task_check_results_args_chip_0x11.output_data_addr =
            (uint32_t)(uintptr_t)(output_data_addr_chip_0x11);
        task_check_results_args_chip_0x11.data_size =
            M * N * meshRow * meshCol * sizeof(uint32_t);
    }

    // 2. Register the tasks
    /////////////////////
    /////////////////////
    // xdma cp tasks for loading A matrices from mempool to L1 at each chiplet
    // chiplet 0x00
    ///////////////////
    /////////////////
    bingo_task_t* task_mempool_to_cluster_A1_chip_0x00 = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)(&task_mempool_to_cluster_args_A1_chip_0x00), 0x00,
        cluster_id);
    if (task_mempool_to_cluster_A1_chip_0x00 == NULL) {
        printf("Error: Task mempool to cluster A1 creation failed!\r\n");
    }
    bingo_task_t* task_mempool_to_cluster_A2_chip_0x01 = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)(&task_mempool_to_cluster_args_A2_chip_0x01), 0x01,
        cluster_id);
    if (task_mempool_to_cluster_A2_chip_0x01 == NULL) {
        printf("Error: Task mempool to cluster A2 creation failed!\r\n");
    }
    bingo_task_t* task_mempool_to_cluster_A3_chip_0x10 = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)(&task_mempool_to_cluster_args_A3_chip_0x10), 0x10,
        cluster_id);
    if (task_mempool_to_cluster_A3_chip_0x10 == NULL) {
        printf("Error: Task mempool to cluster A3 creation failed!\r\n");
    }
    bingo_task_t* task_mempool_to_cluster_A4_chip_0x11 = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)(&task_mempool_to_cluster_args_A4_chip_0x11), 0x11,
        cluster_id);
    if (task_mempool_to_cluster_A4_chip_0x11 == NULL) {
        printf("Error: Task mempool to cluster A4 creation failed!\r\n");
    }
    ///////////////////
    ///////////////////
    // xdma cp task for loading B matrix from mempool to L1 at chiplet 0x00
    ///////////////////
    ///////////////////
    bingo_task_t* task_mempool_to_cluster_B_chip_0x00 = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)(&task_mempool_to_cluster_args_B_chip_0x00), 0x00,
        cluster_id);
    if (task_mempool_to_cluster_B_chip_0x00 == NULL) {
        printf("Error: Task mempool to cluster B creation failed!\r\n");
    }
    ///////////////////
    ///////////////////
    // idma broadcast task for B matrix from chiplet 0x00 to all other
    // chiplets//
    ///////////////////
    ///////////////////
    bingo_task_t* task_broadcast_B_chip_0x00 = bingo_task_create(
        __snax_kernel_idma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)(&task_broadcast_args_B_chip_0x00), 0x00,
        cluster_id);
    if (task_broadcast_B_chip_0x00 == NULL) {
        printf("Error: Task broadcast B creation failed!\r\n");
    }
    ///////////////////
    ///////////////////
    // versacore gemm compute tasks at each chiplet//
    // store output D from L1 to L3 at each chiplet//
    // check results tasks at each chiplet//
    ///////////////////
    ///////////////////
    bingo_task_t* task_versacore_chip_0x00 = bingo_task_create(
        __snax_kernel_gemm_compute_only_intra_chiplet_func_addr,
        (uint32_t)(uintptr_t)(gemm_args_chip_0x00), 0x00, cluster_id);
    if (task_versacore_chip_0x00 == NULL) {
        printf("Error: Task versacore creation failed!\r\n");
    }
    bingo_task_t* task_store_output_chip_0x00 = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)(&task_cluster_to_l3_args_D_chip_0x00), 0x00,
        cluster_id);
    if (task_store_output_chip_0x00 == NULL) {
        printf("Error: Task store output creation failed!\r\n");
    }
    bingo_task_t* task_check_results_chip_0x00 = bingo_task_create(
        check_results_func_addr,
        (uint32_t)(uintptr_t)(&task_check_results_args_chip_0x00), 0x00,
        cluster_id);
    if (task_check_results_chip_0x00 == NULL) {
        printf("Error: Task check results creation failed!\r\n");
    }

    bingo_task_t* task_versacore_chip_0x01 = bingo_task_create(
        __snax_kernel_gemm_compute_only_intra_chiplet_func_addr,
        (uint32_t)(uintptr_t)(gemm_args_chip_0x01), 0x01, cluster_id);
    if (task_versacore_chip_0x01 == NULL) {
        printf("Error: Task versacore creation failed!\r\n");
    }
    bingo_task_t* task_store_output_chip_0x01 = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)(&task_cluster_to_l3_args_D_chip_0x01), 0x01,
        cluster_id);
    if (task_store_output_chip_0x01 == NULL) {
        printf("Error: Task store output creation failed!\r\n");
    }
    bingo_task_t* task_check_results_chip_0x01 = bingo_task_create(
        check_results_func_addr,
        (uint32_t)(uintptr_t)(&task_check_results_args_chip_0x01), 0x01,
        cluster_id);
    if (task_check_results_chip_0x01 == NULL) {
        printf("Error: Task check results creation failed!\r\n");
    }

    bingo_task_t* task_versacore_chip_0x10 = bingo_task_create(
        __snax_kernel_gemm_compute_only_intra_chiplet_func_addr,
        (uint32_t)(uintptr_t)(gemm_args_chip_0x10), 0x10, cluster_id);
    if (task_versacore_chip_0x10 == NULL) {
        printf("Error: Task versacore creation failed!\r\n");
    }
    bingo_task_t* task_store_output_chip_0x10 = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)(&task_cluster_to_l3_args_D_chip_0x10), 0x10,
        cluster_id);
    if (task_store_output_chip_0x10 == NULL) {
        printf("Error: Task store output creation failed!\r\n");
    }
    bingo_task_t* task_check_results_chip_0x10 = bingo_task_create(
        check_results_func_addr,
        (uint32_t)(uintptr_t)(&task_check_results_args_chip_0x10), 0x10,
        cluster_id);
    if (task_check_results_chip_0x10 == NULL) {
        printf("Error: Task check results creation failed!\r\n");
    }

    bingo_task_t* task_versacore_chip_0x11 = bingo_task_create(
        __snax_kernel_gemm_compute_only_intra_chiplet_func_addr,
        (uint32_t)(uintptr_t)(gemm_args_chip_0x11), 0x11, cluster_id);
    if (task_versacore_chip_0x11 == NULL) {
        printf("Error: Task versacore creation failed!\r\n");
    }
    bingo_task_t* task_store_output_chip_0x11 = bingo_task_create(
        __snax_kernel_xdma_1d_copy_func_addr,
        (uint32_t)(uintptr_t)(&task_cluster_to_l3_args_D_chip_0x11), 0x11,
        cluster_id);
    if (task_store_output_chip_0x11 == NULL) {
        printf("Error: Task store output creation failed!\r\n");
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
    // firstly load A1, A2, A3, A4 and B
    // then broadcast B
    // then versacore gemm compute
    // then store output D
    // then check results
    // Load A1 → VersaCore 0x00 → Store Out 0x00 → Check Results 0x00
    // Load A2 → VersaCore 0x01 → Store Out 0x01 → Check Results 0x01
    // Load A3 → VersaCore 0x10 → Store Out 0x10 → Check Results 0x10
    // Load A4 → VersaCore 0x11 → Store Out 0x11 → Check Results 0x11
    // Load B  → VersaCores (0x00)
    // Load B  → Broadcast B → VersaCores (0x01, 0x10, 0x11)

    bingo_task_add_depend(task_broadcast_B_chip_0x00,
                          task_mempool_to_cluster_B_chip_0x00);
    bingo_task_add_depend(task_versacore_chip_0x00,
                          task_mempool_to_cluster_A1_chip_0x00);
    bingo_task_add_depend(task_versacore_chip_0x00,
                          task_mempool_to_cluster_B_chip_0x00);
    bingo_task_add_depend(task_versacore_chip_0x01,
                          task_mempool_to_cluster_A2_chip_0x01);
    bingo_task_add_depend(task_versacore_chip_0x01, task_broadcast_B_chip_0x00);
    bingo_task_add_depend(task_versacore_chip_0x10,
                          task_mempool_to_cluster_A3_chip_0x10);
    bingo_task_add_depend(task_versacore_chip_0x10, task_broadcast_B_chip_0x00);
    bingo_task_add_depend(task_versacore_chip_0x11,
                          task_mempool_to_cluster_A4_chip_0x11);
    bingo_task_add_depend(task_versacore_chip_0x11, task_broadcast_B_chip_0x00);
    bingo_task_add_depend(task_store_output_chip_0x00,
                          task_versacore_chip_0x00);
    bingo_task_add_depend(task_check_results_chip_0x00,
                          task_store_output_chip_0x00);
    bingo_task_add_depend(task_store_output_chip_0x01,
                          task_versacore_chip_0x01);
    bingo_task_add_depend(task_check_results_chip_0x01,
                          task_store_output_chip_0x01);
    bingo_task_add_depend(task_store_output_chip_0x10,
                          task_versacore_chip_0x10);
    bingo_task_add_depend(task_check_results_chip_0x10,
                          task_store_output_chip_0x10);
    bingo_task_add_depend(task_store_output_chip_0x11,
                          task_versacore_chip_0x11);
    bingo_task_add_depend(task_check_results_chip_0x11,
                          task_store_output_chip_0x11);

    // 4. Flush the caches to make sure the device can see the latest arguments
    asm volatile("fence" ::: "memory");
    //////////////////////
    // End main function //
    //////////////////////

    // Prepare the task list to return
    task_list[0] = task_mempool_to_cluster_A1_chip_0x00;
    task_list[1] = task_mempool_to_cluster_A2_chip_0x01;
    task_list[2] = task_mempool_to_cluster_A3_chip_0x10;
    task_list[3] = task_mempool_to_cluster_A4_chip_0x11;
    task_list[4] = task_mempool_to_cluster_B_chip_0x00;
    task_list[5] = task_broadcast_B_chip_0x00;
    task_list[6] = task_versacore_chip_0x00;
    task_list[7] = task_store_output_chip_0x00;
    task_list[8] = task_check_results_chip_0x00;
    task_list[9] = task_versacore_chip_0x01;
    task_list[10] = task_store_output_chip_0x01;
    task_list[11] = task_check_results_chip_0x01;
    task_list[12] = task_versacore_chip_0x10;
    task_list[13] = task_store_output_chip_0x10;
    task_list[14] = task_check_results_chip_0x10;
    task_list[15] = task_versacore_chip_0x11;
    task_list[16] = task_store_output_chip_0x11;
    task_list[17] = task_check_results_chip_0x11;
    uint32_t num_tasks = 18;

    output_data_ptr = (uintptr_t*)output_data_addr_chip_0x00;
    return num_tasks;
}
