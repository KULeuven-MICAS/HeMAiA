// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>

// This program copies data from one chip to another chip. It utilizes all four
// cluster's DMA to apply pressure on the D2D communication. They copy the data
// to remote, then copy back to local, and finally check the data.
#include "data.h"

#include "snrt.h"

// The array to store the start address of TCDM for four clusters
uint64_t next_source_start_addr[4];
uint64_t next_destination_start_addr[4];
uint64_t test_data_start_addr;

int main() {
    int err = 0;
    // First set the addr of cluster 0
    // tcdm0_start_addr = (int8_t*)0x10000000;
    // tcdm1_start_addr = (int8_t*)0x10100000;
    if (snrt_is_dm_core() && snrt_cluster_idx() == 0) {
        test_data_start_addr = (uint64_t)test_data;
        test_data_start_addr += (uint64_t)snrt_cluster_base_addrh() << 32;
    }

    if (snrt_is_dm_core()) {
        uint8_t current_cluster_idx = snrt_cluster_idx();
        next_destination_start_addr[current_cluster_idx] =
            (uint64_t)snrt_cluster_base_addrl();
        next_destination_start_addr[current_cluster_idx] +=
            (uint64_t)snrt_cluster_base_addrh() << 32;
    }

    // Report the address of test data
    // if (snrt_global_core_idx() == 0) {
    //     printf("Test data start address: %p%p\r\n",
    //            (uint8_t*)(test_data_start_addr >> 32),
    //            (uint8_t*)test_data_start_addr);
    //     printf("TCDM0 start address: %p%p\r\n",
    //            (uint8_t*)(next_destination_start_addr[0] >> 32),
    //            (uint8_t*)next_destination_start_addr[0]);
    //     printf("TCDM1 start address: %p%p\r\n",
    //            (uint8_t*)(next_destination_start_addr[1] >> 32),
    //            (uint8_t*)next_destination_start_addr[1]);
    //     printf("TCDM2 start address: %p%p\r\n",
    //            (uint8_t*)(next_destination_start_addr[2] >> 32),
    //            (uint8_t*)next_destination_start_addr[2]);
    //     printf("TCDM3 start address: %p%p\r\n",
    //            (uint8_t*)(next_destination_start_addr[3] >> 32),
    //            (uint8_t*)next_destination_start_addr[3]);
    // }
    snrt_global_barrier();

    // Loading data to TCDM
    if (snrt_is_dm_core()) {
        uint8_t current_cluster_idx = snrt_cluster_idx();

        snrt_dma_start_1d_wideptr(
            next_destination_start_addr[current_cluster_idx],
            test_data_start_addr, length_data);
        snrt_dma_wait_all();
    }

    snrt_global_barrier();
    if (snrt_global_core_idx() == 0) {
        printf("Load data to TCDM done!\r\n");
    }
    snrt_global_barrier();

    // Copy data from this chip to remote chip
    if (snrt_is_dm_core()) {
        uint8_t current_cluster_idx = snrt_cluster_idx();
        uint8_t current_chip_idx = get_current_chip_id();
        uint8_t target_chip_idx;
        if (current_chip_idx == 0x00) {
            target_chip_idx = 0x10;
        } else {
            target_chip_idx = 0;
        }

        next_source_start_addr[current_cluster_idx] =
            next_destination_start_addr[current_cluster_idx];

        next_destination_start_addr[current_cluster_idx] =
            (uint64_t)snrt_cluster_base_addrl();
        next_destination_start_addr[current_cluster_idx] +=
            ((uint64_t)target_chip_idx) << 40;
        snrt_dma_start_1d_wideptr(
            next_destination_start_addr[current_cluster_idx],
            next_source_start_addr[current_cluster_idx], length_data);
        snrt_dma_wait_all();
    }
    snrt_global_barrier();
    if (snrt_global_core_idx() == 0) {
        printf("Copy to remote cluster done!\r\n");
    }
    snrt_global_barrier();

    // Then, copy the data back to the local chip
    if (snrt_is_dm_core()) {
        uint8_t current_cluster_idx = snrt_cluster_idx();
        snrt_dma_start_1d_wideptr(
            next_source_start_addr[current_cluster_idx],
            next_destination_start_addr[current_cluster_idx], length_data);
        snrt_dma_wait_all();
    }
    snrt_global_barrier();
    if (snrt_global_core_idx() == 0) {
        printf("Copy back from remote cluster done!\r\n");
    }
    snrt_global_barrier();

    // Start to check
    if (snrt_is_dm_core() && snrt_cluster_idx() == 0) {
        uint8_t* tcdm0_start_addr = (uint8_t*)snrt_cluster_base_addrl();
        ;
        printf("C0 Checking the results\r\n");
        for (int i = 0; i < length_data; i++) {
            if (tcdm0_start_addr[i] != test_data[i]) {
                err++;
                printf("C0 data is incorrect!\r\n");
                printf("tcdm0[%d]=%d, test_data[%d]=%d\r\n", i,
                       tcdm0_start_addr[i], i, test_data[i]);
                return -1;
            }
        }
    } else if (snrt_is_dm_core() && snrt_cluster_idx() == 1) {
        uint8_t* tcdm1_start_addr = (uint8_t*)snrt_cluster_base_addrl();
        printf("C1 Checking the results\r\n");
        for (int i = 0; i < length_data; i++) {
            if (tcdm1_start_addr[i] != test_data[i]) {
                err++;
                printf("C1 data is incorrect!\r\n");
                printf("tcdm1[%d]=%d, test_data[%d]=%d\r\n", i,
                       tcdm1_start_addr[i], i, test_data[i]);
                return -1;
            }
        }
    } else if (snrt_is_dm_core() && snrt_cluster_idx() == 2) {
        uint8_t* tcdm2_start_addr = (uint8_t*)snrt_cluster_base_addrl();
        printf("C2 Checking the results\r\n");
        for (int i = 0; i < length_data; i++) {
            if (tcdm2_start_addr[i] != test_data[i]) {
                err++;
                printf("C2 data is incorrect!\r\n");
                printf("tcdm2[%d]=%d, test_data[%d]=%d\r\n", i,
                       tcdm2_start_addr[i], i, test_data[i]);
                return -1;
            }
        }
    } else if (snrt_is_dm_core() && snrt_cluster_idx() == 3) {
        uint8_t* tcdm3_start_addr = (uint8_t*)snrt_cluster_base_addrl();
        printf("C3 Checking the results\r\n");
        for (int i = 0; i < length_data; i++) {
            if (tcdm3_start_addr[i] != test_data[i]) {
                err++;
                printf("C3 data is incorrect!\r\n");
                printf("tcdm3[%d]=%d, test_data[%d]=%d\r\n", i,
                       tcdm3_start_addr[i], i, test_data[i]);
                return -1;
            }
        }
    } else
        return 0;
}
