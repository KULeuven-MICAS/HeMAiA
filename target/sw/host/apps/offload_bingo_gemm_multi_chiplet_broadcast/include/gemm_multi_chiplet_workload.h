// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>

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

uint32_t check_result(uint8_t* src, uint8_t* dst, uint32_t size) {
    int err_count = 0;
    for (uint32_t i = 0; i < size; i++) {
        if (*(src + i) != *(dst + i)) {
            err_count++;
            if (err_count <= 2) {
                printf("Error in loading A1 to chiplet 0x00 at L1 addr %x\r\n",
                       (uintptr_t)(src) + i);
            }
        }
        if (err_count == 0) {
            printf("A1 loaded correctly to chiplet 0x00\r\n");
        }
    }
}

uint32_t __workload_versacore_multi_chiplet_broadcast(bingo_task_t** task_list) {
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

    // 1 Get the needed kernel function address by the kernel name
    check_kernel_tab_ready();
    uint32_t __snax_kernel_xdma_1d_copy_func_addr = get_device_function("__snax_kernel_xdma_1d_copy");
    uint32_t __snax_kernel_idma_1d_copy_func_addr = get_device_function("__snax_kernel_idma_1d_copy");
    uint32_t __snax_kernel_gemm_compute_only_intra_chiplet_func_addr = get_device_function("__snax_kernel_gemm_compute_only_intra_chiplet");
    uint32_t check_results_func_addr = get_device_function("__snax_kernel_check_results");
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
    volatile uintptr_t A_mp = chiplet_addr_transform_loc(2, 0, SPM_WIDE_BASE_ADDR);  // L4 mempool base address
    // Notice the M,N,Ks, which are defined in a sperate header file
    // Those variables are compiled with the address known at compile time, which are all stored in 0x8000_0000 range
    // If we read them directly, it will incur the cross-chiplet traffic
    // By forcing the read by BINGO_CHIPLET_READW marco, we can ensure those variables are read from local chiplet SPM
    uint32_t AdataTileSize = BINGO_CHIPLET_READW(M) * BINGO_CHIPLET_READW(K) * BINGO_CHIPLET_READW(meshRow) * BINGO_CHIPLET_READW(tileSize) * sizeof(uint8_t);
    uint32_t BdataSize     = BINGO_CHIPLET_READW(K) * BINGO_CHIPLET_READW(N) * BINGO_CHIPLET_READW(meshCol) * BINGO_CHIPLET_READW(tileSize) * sizeof(uint8_t);
    uint32_t CdataSize     = BINGO_CHIPLET_READW(M) * BINGO_CHIPLET_READW(N) * BINGO_CHIPLET_READW(meshRow) * BINGO_CHIPLET_READW(meshCol)  * sizeof(uint8_t);
    uint32_t DdataSize     = BINGO_CHIPLET_READW(M) * BINGO_CHIPLET_READW(N) * BINGO_CHIPLET_READW(meshRow) * BINGO_CHIPLET_READW(meshCol)  * sizeof(uint32_t);
    uintptr_t A1_mp        = A_mp + 0 * AdataTileSize;
    uintptr_t A2_mp        = A_mp + 1 * AdataTileSize;
    uintptr_t A3_mp        = A_mp + 2 * AdataTileSize;
    uintptr_t A4_mp        = A_mp + 3 * AdataTileSize;
    uintptr_t B_mp         = A4_mp + AdataTileSize;
    // Below are the golden data in local chiplet's L3
    uint64_t A_golden_l3   = chiplet_addr_transform((uint64_t)(uintptr_t)(A_data_L3));
    uint64_t A1_golden_l3  = A_golden_l3 + 0 * AdataTileSize;
    uint64_t A2_golden_l3  = A_golden_l3 + 1 * AdataTileSize;
    uint64_t A3_golden_l3  = A_golden_l3 + 2 * AdataTileSize;
    uint64_t A4_golden_l3  = A_golden_l3 + 3 * AdataTileSize;
    // D L3
    uint64_t B_golden_l3   = chiplet_addr_transform((uint64_t)(uintptr_t)B_data_L3);
    uint64_t D_golden_l3   = chiplet_addr_transform((uint64_t)(uintptr_t)D_data_L3);
    uint64_t D1_golden_l3  = D_golden_l3 + 0 * DdataSize;
    uint64_t D2_golden_l3  = D_golden_l3 + 1 * DdataSize;
    uint64_t D3_golden_l3  = D_golden_l3 + 2 * DdataSize;
    uint64_t D4_golden_l3  = D_golden_l3 + 3 * DdataSize;

    O1HeapInstance32* local_l1_heap_manager = bingo_get_l1_heap_manager(get_current_chip_id(), 0);
    // Allocate TCDM space for A, B, C, D in L1
    uintptr_t A_l1 = (uintptr_t)o1heapAllocate32(local_l1_heap_manager, AdataTileSize);
    uintptr_t B_l1 = (uintptr_t)o1heapAllocate32(local_l1_heap_manager, BdataSize);
    uintptr_t C_l1 = (uintptr_t)o1heapAllocate32(local_l1_heap_manager, CdataSize);
    uintptr_t D_l1 = (uintptr_t)o1heapAllocate32(local_l1_heap_manager, DdataSize);

    // Load A tiles from mempool to L1 at each chiplet
    __snax_kernel_xdma_1d_copy_args_t task_mempool_to_cluster_args_A1_chip_0x00;
    __snax_kernel_xdma_1d_copy_args_t task_mempool_to_cluster_args_A2_chip_0x01;
    __snax_kernel_xdma_1d_copy_args_t task_mempool_to_cluster_args_A3_chip_0x10;
    __snax_kernel_xdma_1d_copy_args_t task_mempool_to_cluster_args_A4_chip_0x11;
    // check A loaded correctly at each chiplet
    __snax_kernel_check_results_args_t task_check_A1_args_chip_0x00;
    __snax_kernel_check_results_args_t task_check_A2_args_chip_0x01;
    __snax_kernel_check_results_args_t task_check_A3_args_chip_0x10;
    __snax_kernel_check_results_args_t task_check_A4_args_chip_0x11;
    // Then chiplet 0x00 loads B from mempool to l1
    __snax_kernel_xdma_1d_copy_args_t task_mempool_to_cluster_args_B_chip_0x00;
    // Then chiplet 0x00 check B loaded correctly at L1
    __snax_kernel_check_results_args_t task_check_B_args_chip_0x00;
    // Then chiplet 0x00 broadcast B to other chiplets
    __snax_kernel_idma_1d_copy_args_t task_idma_broadcast_B_args_chip_0x00;
    // After the broadcast, other chiplets check B loaded correctly at L1
    __snax_kernel_check_results_args_t task_check_B_args_chip_0x01;
    __snax_kernel_check_results_args_t task_check_B_args_chip_0x10;
    __snax_kernel_check_results_args_t task_check_B_args_chip_0x11;
    // Then each chiplet does gemm compute
    __snax_kernel_gemm_intra_chiplet_args_t task_gemm_args_chip_0x00;
    __snax_kernel_gemm_intra_chiplet_args_t task_gemm_args_chip_0x01;
    __snax_kernel_gemm_intra_chiplet_args_t task_gemm_args_chip_0x10;
    __snax_kernel_gemm_intra_chiplet_args_t task_gemm_args_chip_0x11;
    // Then each chiplet store the result in local L3
    __snax_kernel_idma_1d_copy_args_t task_store_output_args_chip_0x00;
    __snax_kernel_idma_1d_copy_args_t task_store_output_args_chip_0x01;
    __snax_kernel_idma_1d_copy_args_t task_store_output_args_chip_0x10;
    __snax_kernel_idma_1d_copy_args_t task_store_output_args_chip_0x11;
    // Finally, each chiplet checks the result D
    __snax_kernel_check_results_args_t task_check_D1_args_chip_0x00;
    __snax_kernel_check_results_args_t task_check_D2_args_chip_0x01;
    __snax_kernel_check_results_args_t task_check_D3_args_chip_0x10;
    __snax_kernel_check_results_args_t task_check_D4_args_chip_0x11;

    // Assign args for the chiplet 0x00
    if (get_current_chip_id() == 0x00) {
        printf("Setting args for chiplet 0x00\r\n");
        // Load A1 from mempool to L1 at chiplet 0x00
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00)->src_addr_hi = HIGH32(A1_mp);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00)->src_addr_lo = LOW32(A1_mp);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00)->dst_addr_hi = HIGH32(A_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00)->dst_addr_lo = LOW32(A_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00)->size        = BINGO_CHIPLET_READW(AdataTileSize);
        printf("Chip(%x, %x): load A1 from mempool to L1 Args: src_addr_hi=%x, src_addr_lo=%x, dst_addr_hi=%x, dst_addr_lo=%x, size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00)->src_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00)->src_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00)->dst_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00)->dst_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A1_chip_0x00)->size);

        // check A1 loaded correctly at chiplet 0x00
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A1_args_chip_0x00)->golden_data_addr = LOW32(A1_golden_l3);  // golden A1 at cluster L3
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A1_args_chip_0x00)->output_data_addr = LOW32(A_l1);       // loaded A1 at L1
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A1_args_chip_0x00)->data_size = BINGO_CHIPLET_READW(AdataTileSize);
        printf("Chip(%x, %x): check A1 loaded correctly Args: golden_data_addr=%x, output_data_addr=%x, data_size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A1_args_chip_0x00)->golden_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A1_args_chip_0x00)->output_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A1_args_chip_0x00)->data_size);
        // Load B from mempool to L1 at chiplet 0x00
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_B_chip_0x00)->src_addr_hi = HIGH32(B_mp);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_B_chip_0x00)->src_addr_lo = LOW32(B_mp);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_B_chip_0x00)->dst_addr_hi = HIGH32(B_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_B_chip_0x00)->dst_addr_lo = LOW32(B_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_B_chip_0x00)->size = BdataSize;
        printf("Chip(%x, %x): load B from mempool to L1 Args: src_addr_hi=%x, src_addr_lo=%x, dst_addr_hi=%x, dst_addr_lo=%x, size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_B_chip_0x00)->src_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_B_chip_0x00)->src_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_B_chip_0x00)->dst_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_B_chip_0x00)->dst_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_B_chip_0x00)->size);

        // Check B loaded correctly at chiplet 0x00
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x00)->golden_data_addr = LOW32(B_golden_l3);  // golden B at cluster L3
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x00)->output_data_addr = LOW32(B_l1);  // loaded B at L1
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x00)->data_size = BdataSize;
        printf("Chip(%x, %x): check B loaded correctly Args: golden_data_addr=%x, output_data_addr=%x, data_size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x00)->golden_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x00)->output_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x00)->data_size);

        // Args for broadcasting B from chiplet 0x00 to other chiplets
        uintptr_t broadcast_dst_addr_B = chiplet_addr_transform_loc(0xF, 0xF, (uint64_t)(uintptr_t)B_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_idma_broadcast_B_args_chip_0x00)->src_addr_hi = HIGH32(B_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_idma_broadcast_B_args_chip_0x00)->src_addr_lo = LOW32(B_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_idma_broadcast_B_args_chip_0x00)->dst_addr_hi = HIGH32(broadcast_dst_addr_B);
        BINGO_CHIPLET_LOCAL_AUTO(task_idma_broadcast_B_args_chip_0x00)->dst_addr_lo = LOW32(broadcast_dst_addr_B);
        BINGO_CHIPLET_LOCAL_AUTO(task_idma_broadcast_B_args_chip_0x00)->size = BdataSize;
        printf("Chip(%x, %x): broadcast B to other chiplets Args: src_addr_hi=%x, src_addr_lo=%x, dst_addr_hi=%x, dst_addr_lo=%x, size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_idma_broadcast_B_args_chip_0x00)->src_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_idma_broadcast_B_args_chip_0x00)->src_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_idma_broadcast_B_args_chip_0x00)->dst_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_idma_broadcast_B_args_chip_0x00)->dst_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_idma_broadcast_B_args_chip_0x00)->size);  

        // Args for gemm compute at chiplet 0x00
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->input_A_addr_hi = HIGH32(A_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->input_A_addr_lo = LOW32(A_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->input_B_addr_hi = HIGH32(B_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->input_B_addr_lo = LOW32(B_l1);
        // C matrix
        if (BINGO_CHIPLET_READW(accumPrevC) || BINGO_CHIPLET_READW(addZeroC)) {
            // When accumPrevC is true, we use D as the previous C matrix
            BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->input_C_addr_hi = 0;
            BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->input_C_addr_lo = 0;
        } else {
            BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->input_C_addr_hi = HIGH32(C_l1);
            BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->input_C_addr_lo = LOW32(C_l1);
        }
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->output_D_addr_hi = HIGH32(D_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->output_D_addr_lo = LOW32(D_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->M = BINGO_CHIPLET_READW(M);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->K = BINGO_CHIPLET_READW(K);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->N = BINGO_CHIPLET_READW(N);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->array_shape = BINGO_CHIPLET_READW(array_shape);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->transpose_A = BINGO_CHIPLET_READW(transposed_A);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->transpose_B = BINGO_CHIPLET_READW(transposed_B);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->accumPrevC = BINGO_CHIPLET_READW(accumPrevC);
        printf("Chip(%x, %x): gemm compute Args: A_addr_hi=%x, A_addr_lo=%x, B_addr_hi=%x, B_addr_lo=%x, C_addr_hi=%x, C_addr_lo=%x, D_addr_hi=%x, D_addr_lo=%x, M=%x, K=%x, N=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->input_A_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->input_A_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->input_B_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->input_B_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->input_C_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->input_C_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->output_D_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->output_D_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->M,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->K,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x00)->N);

        // Args for storing output D from L1 to L3 at chiplet 0x00
        uintptr_t D_dst_addr_chip_0x00 = (uintptr_t)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), DdataSize);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x00)->src_addr_hi = HIGH32(D_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x00)->src_addr_lo = LOW32(D_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x00)->dst_addr_hi = HIGH32(D_dst_addr_chip_0x00);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x00)->dst_addr_lo = LOW32(D_dst_addr_chip_0x00);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x00)->size = DdataSize;
        printf("Chip(%x, %x): store output D from L1 to L3 Args: src_addr_hi=%x, src_addr_lo=%x, dst_addr_hi=%x, dst_addr_lo=%x, size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x00)->src_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x00)->src_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x00)->dst_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x00)->dst_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x00)->size);

        // Args for checking output D1 at chiplet 0x00
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D1_args_chip_0x00)->golden_data_addr = LOW32(D1_golden_l3);  // golden D1 at cluster L3
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D1_args_chip_0x00)->output_data_addr = LOW32(D_dst_addr_chip_0x00);       // stored D1 at L3
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D1_args_chip_0x00)->data_size = DdataSize;
        printf("Chip(%x, %x): check output D1 Args: golden_data_addr=%x, output_data_addr=%x, data_size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D1_args_chip_0x00)->golden_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D1_args_chip_0x00)->output_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D1_args_chip_0x00)->data_size);
    }

    // Assign args for the chiplet 0x01
    if (get_current_chip_id() == 0x01) {
        printf("Setting args for chiplet 0x01\r\n");
        // Load A2 from mempool to L1 at chiplet 0x01
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A2_chip_0x01)->src_addr_hi = HIGH32(A2_mp);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A2_chip_0x01)->src_addr_lo = LOW32(A2_mp);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A2_chip_0x01)->dst_addr_hi = HIGH32(A_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A2_chip_0x01)->dst_addr_lo = LOW32(A_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A2_chip_0x01)->size        = BINGO_CHIPLET_READW(AdataTileSize);
        printf("Chip(%x, %x): load A2 from mempool to L1 Args: src_addr_hi=%x, src_addr_lo=%x, dst_addr_hi=%x, dst_addr_lo=%x, size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A2_chip_0x01)->src_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A2_chip_0x01)->src_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A2_chip_0x01)->dst_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A2_chip_0x01)->dst_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A2_chip_0x01)->size);

        // check A2 loaded correctly at chiplet 0x01
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A2_args_chip_0x01)->golden_data_addr = LOW32(A2_golden_l3);  // golden A2 at cluster L3
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A2_args_chip_0x01)->output_data_addr = LOW32(A_l1);       // loaded A2 at L1
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A2_args_chip_0x01)->data_size = BINGO_CHIPLET_READW(AdataTileSize);
        printf("Chip(%x, %x): check A2 loaded correctly Args: golden_data_addr=%x, output_data_addr=%x, data_size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A2_args_chip_0x01)->golden_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A2_args_chip_0x01)->output_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A2_args_chip_0x01)->data_size);
        // check B loaded correctly at chiplet 0x01
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x01)->golden_data_addr = LOW32(B_golden_l3);  // golden B at cluster L3
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x01)->output_data_addr = LOW32(B_l1);  // loaded B at L1
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x01)->data_size = BdataSize;
        printf("Chip(%x, %x): check B loaded correctly Args: golden_data_addr=%x, output_data_addr=%x, data_size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x01)->golden_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x01)->output_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x01)->data_size);

        // GeMM args for chiplet 0x01
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->input_A_addr_hi = HIGH32(A_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->input_A_addr_lo = LOW32(A_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->input_B_addr_hi = HIGH32(B_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->input_B_addr_lo = LOW32(B_l1);
        // C matrix
        if (BINGO_CHIPLET_READW(accumPrevC) || BINGO_CHIPLET_READW(addZeroC)) {
            // When accumPrevC is true, we use D as the previous C matrix
            BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->input_C_addr_hi = 0;
            BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->input_C_addr_lo = 0;
        } else {
            BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->input_C_addr_hi = HIGH32(C_l1);
            BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->input_C_addr_lo = LOW32(C_l1);
        }
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->output_D_addr_hi = HIGH32(D_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->output_D_addr_lo = LOW32(D_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->M = BINGO_CHIPLET_READW(M);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->K = BINGO_CHIPLET_READW(K);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->N = BINGO_CHIPLET_READW(N);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->array_shape = BINGO_CHIPLET_READW(array_shape);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->transpose_A = BINGO_CHIPLET_READW(transposed_A);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->transpose_B = BINGO_CHIPLET_READW(transposed_B);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->accumPrevC = BINGO_CHIPLET_READW(accumPrevC);
        printf("Chip(%x, %x): gemm compute Args: A_addr_hi=%x, A_addr_lo=%x, B_addr_hi=%x, B_addr_lo=%x, C_addr_hi=%x, C_addr_lo=%x, D_addr_hi=%x, D_addr_lo=%x, M=%x, K=%x, N=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->input_A_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->input_A_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->input_B_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->input_B_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->input_C_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->input_C_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->output_D_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->output_D_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->M,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->K,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x01)->N);
        // Args for storing output D from L1 to L3 at chiplet 0x01
        uintptr_t D_dst_addr_chip_0x01 = (uintptr_t)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), DdataSize);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x01)->src_addr_hi = HIGH32(D_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x01)->src_addr_lo = LOW32(D_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x01)->dst_addr_hi = HIGH32(D_dst_addr_chip_0x01);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x01)->dst_addr_lo = LOW32(D_dst_addr_chip_0x01);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x01)->size = DdataSize;
        printf("Chip(%x, %x): store output D from L1 to L3 Args: src_addr_hi=%x, src_addr_lo=%x, dst_addr_hi=%x, dst_addr_lo=%x, size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x01)->src_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x01)->src_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x01)->dst_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x01)->dst_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x01)->size);
        // Args for checking output D2 at chiplet 0x01
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D2_args_chip_0x01)->golden_data_addr = LOW32(D2_golden_l3);  // golden D2
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D2_args_chip_0x01)->output_data_addr = LOW32(D_dst_addr_chip_0x01);       // loaded D2 at L1
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D2_args_chip_0x01)->data_size = DdataSize;
        printf("Chip(%x, %x): check output D2 Args: golden_data_addr=%x, output_data_addr=%x, data_size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D2_args_chip_0x01)->golden_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D2_args_chip_0x01)->output_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D2_args_chip_0x01)->data_size);
    }

    if (get_current_chip_id() == 0x10) {
        printf("Setting args for chiplet 0x10\r\n");
        // Load A3 from mempool to L1 at chiplet 0x10
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A3_chip_0x10)->src_addr_hi = HIGH32(A3_mp);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A3_chip_0x10)->src_addr_lo = LOW32(A3_mp);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A3_chip_0x10)->dst_addr_hi = HIGH32(A_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A3_chip_0x10)->dst_addr_lo = LOW32(A_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A3_chip_0x10)->size        = BINGO_CHIPLET_READW(AdataTileSize);
        printf("Chip(%x, %x): load A3 from mempool to L1 Args: src_addr_hi=%x, src_addr_lo=%x, dst_addr_hi=%x, dst_addr_lo=%x, size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A3_chip_0x10)->src_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A3_chip_0x10)->src_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A3_chip_0x10)->dst_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A3_chip_0x10)->dst_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A3_chip_0x10)->size);

        // check A3 loaded correctly at chiplet 0x10
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A3_args_chip_0x10)->golden_data_addr = LOW32(A3_golden_l3);
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A3_args_chip_0x10)->output_data_addr = LOW32(A_l1);       // loaded A3 at L1
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A3_args_chip_0x10)->data_size = BINGO_CHIPLET_READW(AdataTileSize);
        printf("Chip(%x, %x): check A3 loaded correctly Args: golden_data_addr=%x, output_data_addr=%x, data_size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A3_args_chip_0x10)->golden_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A3_args_chip_0x10)->output_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A3_args_chip_0x10)->data_size); 

        // check B loaded correctly at chiplet 0x10
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x10)->golden_data_addr = LOW32(B_golden_l3);  // golden B at cluster L3
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x10)->output_data_addr = LOW32(B_l1);  // loaded B at L1
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x10)->data_size = BdataSize;
        printf("Chip(%x, %x): check B loaded correctly Args: golden_data_addr=%x, output_data_addr=%x, data_size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x10)->golden_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x10)->output_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x10)->data_size);  

        // GeMM args for chiplet 0x10
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->input_A_addr_hi = HIGH32(A_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->input_A_addr_lo = LOW32(A_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->input_B_addr_hi = HIGH32(B_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->input_B_addr_lo = LOW32(B_l1);
        // C matrix
        if (BINGO_CHIPLET_READW(accumPrevC) || BINGO_CHIPLET_READW(addZeroC)) {
            // When accumPrevC is true, we use D as the previous C matrix
            BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->input_C_addr_hi = 0;
            BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->input_C_addr_lo = 0;
        } else {
            BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->input_C_addr_hi = HIGH32(C_l1);
            BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->input_C_addr_lo = LOW32(C_l1);
        }
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->output_D_addr_hi = HIGH32(D_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->output_D_addr_lo = LOW32(D_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->M = BINGO_CHIPLET_READW(M);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->K = BINGO_CHIPLET_READW(K);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->N = BINGO_CHIPLET_READW(N);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->array_shape = BINGO_CHIPLET_READW(array_shape);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->transpose_A = BINGO_CHIPLET_READW(transposed_A);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->transpose_B = BINGO_CHIPLET_READW(transposed_B);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->accumPrevC = BINGO_CHIPLET_READW(accumPrevC);
        printf("Chip(%x, %x): gemm compute Args: A_addr_hi=%x, A_addr_lo=%x, B_addr_hi=%x, B_addr_lo=%x, C_addr_hi=%x, C_addr_lo=%x, D_addr_hi=%x, D_addr_lo=%x, M=%x, K=%x, N=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->input_A_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->input_A_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->input_B_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->input_B_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->input_C_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->input_C_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->output_D_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->output_D_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->M,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->K,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x10)->N);
        // Args for storing output D from L1 to L3 at chiplet 0x10
        uintptr_t D_dst_addr_chip_0x10 = (uintptr_t)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), DdataSize);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x10)->src_addr_hi = HIGH32(D_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x10)->src_addr_lo = LOW32(D_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x10)->dst_addr_hi = HIGH32(D_dst_addr_chip_0x10);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x10)->dst_addr_lo = LOW32(D_dst_addr_chip_0x10);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x10)->size = DdataSize;
        printf("Chip(%x, %x): store output D from L1 to L3 Args: src_addr_hi=%x, src_addr_lo=%x, dst_addr_hi=%x, dst_addr_lo=%x, size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x10)->src_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x10)->src_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x10)->dst_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x10)->dst_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x10)->size);
        // Args for checking output D3 at chiplet 0x10
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D3_args_chip_0x10)->golden_data_addr = LOW32(D3_golden_l3);
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D3_args_chip_0x10)->output_data_addr = LOW32(D_dst_addr_chip_0x10);
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D3_args_chip_0x10)->data_size = DdataSize;
        printf("Chip(%x, %x): check output D3 Args: golden_data_addr=%x, output_data_addr=%x, data_size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D3_args_chip_0x10)->golden_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D3_args_chip_0x10)->output_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D3_args_chip_0x10)->data_size);
    }

    if (get_current_chip_id() == 0x11){
        printf("Setting args for chiplet 0x11\r\n");
        // Load A4 from mempool to L1 at chiplet 0x11
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A4_chip_0x11)->src_addr_hi = HIGH32(A4_mp);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A4_chip_0x11)->src_addr_lo = LOW32(A4_mp);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A4_chip_0x11)->dst_addr_hi = HIGH32(A_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A4_chip_0x11)->dst_addr_lo = LOW32(A_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A4_chip_0x11)->size        = BINGO_CHIPLET_READW(AdataTileSize);
        printf("Chip(%x, %x): load A4 from mempool to L1 Args: src_addr_hi=%x, src_addr_lo=%x, dst_addr_hi=%x, dst_addr_lo=%x, size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A4_chip_0x11)->src_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A4_chip_0x11)->src_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A4_chip_0x11)->dst_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A4_chip_0x11)->dst_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_mempool_to_cluster_args_A4_chip_0x11)->size);
        // check A4 loaded correctly at chiplet 0x11
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A4_args_chip_0x11)->golden_data_addr = LOW32(A4_golden_l3);
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A4_args_chip_0x11)->output_data_addr = LOW32(A_l1);       // loaded A4 at L1
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A4_args_chip_0x11)->data_size = BINGO_CHIPLET_READW(AdataTileSize);
        printf("Chip(%x, %x): check A4 loaded correctly Args: golden_data_addr=%x, output_data_addr=%x, data_size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A4_args_chip_0x11)->golden_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A4_args_chip_0x11)->output_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_A4_args_chip_0x11)->data_size);
        // check B loaded correctly at chiplet 0x11
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x11)->golden_data_addr = LOW32(B_golden_l3);  // golden B at cluster L3
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x11)->output_data_addr = LOW32(B_l1);  // loaded B at L1
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x11)->data_size = BdataSize;
        printf("Chip(%x, %x): check B loaded correctly Args: golden_data_addr=%x, output_data_addr=%x, data_size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x11)->golden_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x11)->output_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_B_args_chip_0x11)->data_size);
        // GeMM args for chiplet 0x11
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->input_A_addr_hi = HIGH32(A_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->input_A_addr_lo = LOW32(A_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->input_B_addr_hi = HIGH32(B_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->input_B_addr_lo = LOW32(B_l1);
        // C matrix
        if (BINGO_CHIPLET_READW(accumPrevC) || BINGO_CHIPLET_READW(addZeroC)) {
            // When accumPrevC is true, we use D as the previous C matrix
            BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->input_C_addr_hi = 0;
            BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->input_C_addr_lo = 0;
        } else {
            BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->input_C_addr_hi = HIGH32(C_l1);
            BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->input_C_addr_lo = LOW32(C_l1);
        }
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->output_D_addr_hi = HIGH32(D_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->output_D_addr_lo = LOW32(D_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->M = BINGO_CHIPLET_READW(M);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->K = BINGO_CHIPLET_READW(K);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->N = BINGO_CHIPLET_READW(N);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->array_shape = BINGO_CHIPLET_READW(array_shape);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->transpose_A = BINGO_CHIPLET_READW(transposed_A);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->transpose_B = BINGO_CHIPLET_READW(transposed_B);
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->accumPrevC = BINGO_CHIPLET_READW(accumPrevC);
        printf("Chip(%x, %x): gemm compute Args: A_addr_hi=%x, A_addr_lo=%x, B_addr_hi=%x, B_addr_lo=%x, C_addr_hi=%x, C_addr_lo=%x, D_addr_hi=%x, D_addr_lo=%x, M=%x, K=%x, N=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->input_A_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->input_A_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->input_B_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->input_B_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->input_C_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->input_C_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->output_D_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->output_D_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->M,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->K,
        BINGO_CHIPLET_LOCAL_AUTO(task_gemm_args_chip_0x11)->N);
        // Args for storing output D from L1 to L3 at chiplet 0x11
        uintptr_t D_dst_addr_chip_0x11 = (uintptr_t)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), DdataSize);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x11)->src_addr_hi = HIGH32(D_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x11)->src_addr_lo = LOW32(D_l1);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x11)->dst_addr_hi = HIGH32(D_dst_addr_chip_0x11);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x11)->dst_addr_lo = LOW32(D_dst_addr_chip_0x11);
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x11)->size = DdataSize;
        printf("Chip(%x, %x): store output D from L1 to L3 Args: src_addr_hi=%x, src_addr_lo=%x, dst_addr_hi=%x, dst_addr_lo=%x, size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x11)->src_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x11)->src_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x11)->dst_addr_hi,
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x11)->dst_addr_lo,
        BINGO_CHIPLET_LOCAL_AUTO(task_store_output_args_chip_0x11)->size);
        // Args for checking output D4 at chiplet 0x11
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D4_args_chip_0x11)->golden_data_addr = LOW32(D4_golden_l3);
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D4_args_chip_0x11)->output_data_addr = LOW32(D_dst_addr_chip_0x11);
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D4_args_chip_0x11)->data_size = DdataSize;
        printf("Chip(%x, %x): check output D4 Args: golden_data_addr=%x, output_data_addr=%x, data_size=%x\r\n", get_current_chip_loc_x(),get_current_chip_loc_y(),
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D4_args_chip_0x11)->golden_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D4_args_chip_0x11)->output_data_addr,
        BINGO_CHIPLET_LOCAL_AUTO(task_check_D4_args_chip_0x11)->data_size);
    }

    /////////////////////
    // 2. Register the tasks
    /////////////////////
    /////////////////////
    // xdma cp tasks for loading A matrices from mempool to L1 at each chiplet
    // chiplet 0x00
    ///////////////////
    /////////////////
    bingo_task_t* task_mempool_to_cluster_A1_chip_0x00 = bingo_task_create(__snax_kernel_xdma_1d_copy_func_addr,
                                                                          (uint32_t)(uintptr_t)(&task_mempool_to_cluster_args_A1_chip_0x00), 
                                                                          0x00,
                                                                          0);
    bingo_task_t* task_mempool_to_cluster_A2_chip_0x01 = bingo_task_create(__snax_kernel_xdma_1d_copy_func_addr,
                                                                          (uint32_t)(uintptr_t)(&task_mempool_to_cluster_args_A2_chip_0x01),
                                                                          0x01,
                                                                          0);

    bingo_task_t* task_mempool_to_cluster_A3_chip_0x10 = bingo_task_create(__snax_kernel_xdma_1d_copy_func_addr,
                                                                          (uint32_t)(uintptr_t)(&task_mempool_to_cluster_args_A3_chip_0x10),
                                                                          0x10,
                                                                          0);
    bingo_task_t* task_mempool_to_cluster_A4_chip_0x11 = bingo_task_create(__snax_kernel_xdma_1d_copy_func_addr,
                                                                          (uint32_t)(uintptr_t)(&task_mempool_to_cluster_args_A4_chip_0x11),
                                                                          0x11,
                                                                          0);
    if (task_mempool_to_cluster_A1_chip_0x00 == NULL ||
        task_mempool_to_cluster_A2_chip_0x01 == NULL ||
        task_mempool_to_cluster_A3_chip_0x10 == NULL ||
        task_mempool_to_cluster_A4_chip_0x11 == NULL) {
        printf("Error: Task mempool to cluster A creation failed!\r\n");
    }
    bingo_task_t* task_check_A1_chip_0x00 = bingo_task_create(check_results_func_addr,
                                                             (uint32_t)(uintptr_t)(&task_check_A1_args_chip_0x00),
                                                             0x00,
                                                             0);

    bingo_task_t* task_check_A2_chip_0x01 = bingo_task_create(check_results_func_addr, 
                                                             (uint32_t)(uintptr_t)(&task_check_A2_args_chip_0x01),
                                                             0x01,
                                                             0);
    bingo_task_t* task_check_A3_chip_0x10 = bingo_task_create(check_results_func_addr,
                                                             (uint32_t)(uintptr_t)(&task_check_A3_args_chip_0x10),
                                                             0x10,
                                                             0);
    bingo_task_t* task_check_A4_chip_0x11 = bingo_task_create(check_results_func_addr,
                                                             (uint32_t)(uintptr_t)(&task_check_A4_args_chip_0x11),
                                                             0x11,
                                                             0);
    if (task_check_A1_chip_0x00 == NULL ||
        task_check_A2_chip_0x01 == NULL ||
        task_check_A3_chip_0x10 == NULL ||
        task_check_A4_chip_0x11 == NULL) {
        printf("Error: Task check A creation failed!\r\n");
    }

    bingo_task_t* task_mempool_to_cluster_B_chip_0x00 = bingo_task_create(__snax_kernel_xdma_1d_copy_func_addr,
                                                                         (uint32_t)(uintptr_t)(&task_mempool_to_cluster_args_B_chip_0x00),
                                                                         0x00,
                                                                         0);

    bingo_task_t* task_check_B_chip_0x00 = bingo_task_create(check_results_func_addr,
                                                            (uint32_t)(uintptr_t)(&task_check_B_args_chip_0x00),
                                                            0x00,
                                                            0);

    bingo_task_t* task_broadcast_B_chip_0x00 = bingo_task_create(__snax_kernel_idma_1d_copy_func_addr,
                                                                (uint32_t)(uintptr_t)(&task_idma_broadcast_B_args_chip_0x00),
                                                                0x00,
                                                                0);

    bingo_task_t* task_check_B_broadcasted_chip_0x01 = bingo_task_create(check_results_func_addr,
                                                                        (uint32_t)(uintptr_t)(&task_check_B_args_chip_0x01),
                                                                        0x01,
                                                                        0);
    bingo_task_t* task_check_B_broadcasted_chip_0x10 = bingo_task_create(check_results_func_addr,
                                                                        (uint32_t)(uintptr_t)(&task_check_B_args_chip_0x10),
                                                                        0x10,
                                                                        0);
    bingo_task_t* task_check_B_broadcasted_chip_0x11 = bingo_task_create(check_results_func_addr,
                                                                        (uint32_t)(uintptr_t)(&task_check_B_args_chip_0x11),
                                                                        0x11,
                                                                        0);
    if (task_mempool_to_cluster_B_chip_0x00 == NULL ||
        task_check_B_chip_0x00 == NULL ||
        task_broadcast_B_chip_0x00 == NULL ||
        task_check_B_broadcasted_chip_0x01 == NULL ||
        task_check_B_broadcasted_chip_0x10 == NULL ||
        task_check_B_broadcasted_chip_0x11 == NULL) {
        printf("Error: Task mempool to cluster B or check B creation failed!\r\n");
    }

    bingo_task_t* task_versacore_chip_0x00 = bingo_task_create(__snax_kernel_gemm_compute_only_intra_chiplet_func_addr,
                                                              (uint32_t)(uintptr_t)(&task_gemm_args_chip_0x00),
                                                              0x00,
                                                              0);
    bingo_task_t* task_store_output_chip_0x00 = bingo_task_create(__snax_kernel_xdma_1d_copy_func_addr,
                                                                 (uint32_t)(uintptr_t)(&task_store_output_args_chip_0x00),
                                                                 0x00,
                                                                 0);
    bingo_task_t* task_check_results_chip_0x00 = bingo_task_create(check_results_func_addr,
                                                                   (uint32_t)(uintptr_t)(&task_check_D1_args_chip_0x00),
                                                                   0x00,
                                                                   0);

    bingo_task_t* task_versacore_chip_0x01 = bingo_task_create(__snax_kernel_gemm_compute_only_intra_chiplet_func_addr,
                                                              (uint32_t)(uintptr_t)(&task_gemm_args_chip_0x01),
                                                              0x01,
                                                              0);
    bingo_task_t* task_store_output_chip_0x01 = bingo_task_create(__snax_kernel_xdma_1d_copy_func_addr,
                                                                 (uint32_t)(uintptr_t)(&task_store_output_args_chip_0x01),
                                                                 0x01,
                                                                 0);
    bingo_task_t* task_check_results_chip_0x01 = bingo_task_create(check_results_func_addr,
                                                                   (uint32_t)(uintptr_t)(&task_check_D2_args_chip_0x01),
                                                                   0x01,
                                                                   0);
    bingo_task_t* task_versacore_chip_0x10 = bingo_task_create(__snax_kernel_gemm_compute_only_intra_chiplet_func_addr,
                                                              (uint32_t)(uintptr_t)(&task_gemm_args_chip_0x10),
                                                              0x10,
                                                              0);
    bingo_task_t* task_store_output_chip_0x10 = bingo_task_create(__snax_kernel_xdma_1d_copy_func_addr,
                                                                 (uint32_t)(uintptr_t)(&task_store_output_args_chip_0x10),
                                                                 0x10,
                                                                 0);
    bingo_task_t* task_check_results_chip_0x10 = bingo_task_create(check_results_func_addr,
                                                                   (uint32_t)(uintptr_t)(&task_check_D3_args_chip_0x10),
                                                                   0x10,
                                                                   0);
    bingo_task_t* task_versacore_chip_0x11 = bingo_task_create(__snax_kernel_gemm_compute_only_intra_chiplet_func_addr,
                                                              (uint32_t)(uintptr_t)(&task_gemm_args_chip_0x11),
                                                              0x11,
                                                              0);
    bingo_task_t* task_store_output_chip_0x11 = bingo_task_create(__snax_kernel_xdma_1d_copy_func_addr,
                                                                 (uint32_t)(uintptr_t)(&task_store_output_args_chip_0x11),
                                                                 0x11,
                                                                 0);
    bingo_task_t* task_check_results_chip_0x11 = bingo_task_create(check_results_func_addr,
                                                                   (uint32_t)(uintptr_t)(&task_check_D4_args_chip_0x11),
                                                                   0x11,
                                                                   0);
    if (task_versacore_chip_0x00 == NULL ||
        task_store_output_chip_0x00 == NULL ||
        task_check_results_chip_0x00 == NULL ||
        task_versacore_chip_0x01 == NULL ||
        task_store_output_chip_0x01 == NULL ||
        task_check_results_chip_0x01 == NULL ||
        task_versacore_chip_0x10 == NULL ||
        task_store_output_chip_0x10 == NULL ||
        task_check_results_chip_0x10 == NULL ||
        task_versacore_chip_0x11 == NULL ||
        task_store_output_chip_0x11 == NULL ||
        task_check_results_chip_0x11 == NULL) {
        printf("Error: Task versacore gemm, store output or check results creation failed!\r\n");
    }

    // 3. Set the task dependency
    // load A1 (Chip00)      load A2 (Chip01)     load A3 (Chip10)    load A4 (Chip11)
    //   |                         |                      |                      |
    //   v                         v                      v                      v
    // check A1 (Chip00)     check A2 (Chip01)     check A3 (Chip10)     check A4 (Chip11)
    //  |                        
    //  v                        
    // load B (Chip00)
    //  |
    //  v
    // check B (Chip00)
    //  |
    //  v
    // broadcast B(Chip00)--------------------------------------------------------
    //   |                      |                          |                     |
    //   v                      v                          v                     v
    //  Gemm(Chip00)          check B (Chip01)     check B (Chip10)     check B (Chip11)
    //   |                      |                          |                     |
    //   v                      v                          v                     v
    // Store Out(Chip00)    Gemm(Chip01)               Gemm(Chip10)           Gemm(Chip11)
    //   |                      |                          |                     |
    //   v                      v                          v                     v
    // Check Results(Chip00) Store Out(Chip01)       Store Out(Chip10)       Store Out(Chip11)
    //                          |                          |                     |
    //                          v                          v                     v
    //                   Check Results(Chip01)     Check Results(Chip10)   Check Results(Chip11)


    // Dependencies for chip00
    bingo_task_add_depend(task_check_A1_chip_0x00,
                          task_mempool_to_cluster_A1_chip_0x00);

    bingo_task_add_depend(task_mempool_to_cluster_B_chip_0x00,
                          task_check_A1_chip_0x00);

    bingo_task_add_depend(task_check_B_chip_0x00,
                          task_mempool_to_cluster_B_chip_0x00);

    bingo_task_add_depend(task_broadcast_B_chip_0x00,
                          task_check_B_chip_0x00);

    bingo_task_add_depend(task_versacore_chip_0x00,
                          task_broadcast_B_chip_0x00);
    
    bingo_task_add_depend(task_store_output_chip_0x00,
                          task_versacore_chip_0x00);

    bingo_task_add_depend(task_check_results_chip_0x00,
                          task_store_output_chip_0x00);
    // Dependencies for chip01
    bingo_task_add_depend(task_check_A2_chip_0x01,
                          task_mempool_to_cluster_A2_chip_0x01);

    bingo_task_add_depend(task_check_B_broadcasted_chip_0x01,
                          task_broadcast_B_chip_0x00);

    bingo_task_add_depend(task_versacore_chip_0x01,
                          task_check_B_broadcasted_chip_0x01);

    bingo_task_add_depend(task_store_output_chip_0x01,
                          task_versacore_chip_0x01);

    bingo_task_add_depend(task_check_results_chip_0x01,
                          task_store_output_chip_0x01);

    // Dependencies for chip10
    bingo_task_add_depend(task_check_A3_chip_0x10,
                          task_mempool_to_cluster_A3_chip_0x10);

    bingo_task_add_depend(task_check_B_broadcasted_chip_0x10,
                          task_broadcast_B_chip_0x00);

    bingo_task_add_depend(task_versacore_chip_0x10,
                          task_check_B_broadcasted_chip_0x10);

    bingo_task_add_depend(task_store_output_chip_0x10,
                          task_versacore_chip_0x10);

    bingo_task_add_depend(task_check_results_chip_0x10,
                          task_store_output_chip_0x10);

    // Dependencies for chip11
    bingo_task_add_depend(task_check_A4_chip_0x11,
                          task_mempool_to_cluster_A4_chip_0x11);

    bingo_task_add_depend(task_check_B_broadcasted_chip_0x11,
                          task_broadcast_B_chip_0x00);

    bingo_task_add_depend(task_versacore_chip_0x11,
                          task_check_B_broadcasted_chip_0x11);

    bingo_task_add_depend(task_store_output_chip_0x11,
                          task_versacore_chip_0x11);

    bingo_task_add_depend(task_check_results_chip_0x11,
                          task_store_output_chip_0x11);

    //////////////////////
    // End main function //
    //////////////////////

    // Prepare the task list to return
    task_list[0] = task_mempool_to_cluster_A1_chip_0x00;
    task_list[1] = task_mempool_to_cluster_A2_chip_0x01;
    task_list[2] = task_mempool_to_cluster_A3_chip_0x10;
    task_list[3] = task_mempool_to_cluster_A4_chip_0x11;
    task_list[4] = task_check_A1_chip_0x00;
    task_list[5] = task_check_A2_chip_0x01;
    task_list[6] = task_check_A3_chip_0x10;
    task_list[7] = task_check_A4_chip_0x11;

    task_list[8] = task_mempool_to_cluster_B_chip_0x00;
    task_list[9] = task_check_B_chip_0x00;
    
    task_list[10] = task_broadcast_B_chip_0x00;
    task_list[11] = task_check_B_broadcasted_chip_0x01;
    task_list[12] = task_check_B_broadcasted_chip_0x10;
    task_list[13] = task_check_B_broadcasted_chip_0x11;

    task_list[14] = task_versacore_chip_0x00;
    task_list[15] = task_store_output_chip_0x00;
    task_list[16] = task_check_results_chip_0x00;
    task_list[17] = task_versacore_chip_0x01;
    task_list[18] = task_store_output_chip_0x01;
    task_list[19] = task_check_results_chip_0x01;
    task_list[20] = task_versacore_chip_0x10;
    task_list[21] = task_store_output_chip_0x10;
    task_list[22] = task_check_results_chip_0x10;
    task_list[23] = task_versacore_chip_0x11;
    task_list[24] = task_store_output_chip_0x11;
    task_list[25] = task_check_results_chip_0x11;
    asm volatile("fence" ::: "memory");
    uint32_t num_tasks = 26;
    return num_tasks;
}
