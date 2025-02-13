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
uint8_t current_chip_id;

int main() {
    int err = 0;

    if (snrt_is_dm_core() && snrt_cluster_idx() == 0) {
        current_chip_id = get_current_chip_id();
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
    snrt_global_barrier();

    // // Report the address of test data
    // if (snrt_global_core_idx() == 0) {
    //     printf("Current Chip ID is: %x\r\n", current_chip_id);
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
    // snrt_global_barrier();

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

    if (current_chip_id == 0x00) {
        // Utilize Cluster 0 and Cluster 1 to copy data forth and back
        if (snrt_is_dm_core() &&
            (snrt_cluster_idx() == 0 | snrt_cluster_idx() == 1)) {
            uint8_t current_cluster_idx = snrt_cluster_idx();
            next_source_start_addr[current_cluster_idx] =
                next_destination_start_addr[current_cluster_idx];
            next_destination_start_addr[current_cluster_idx] =
                (uint64_t)snrt_cluster_base_addrl();
            next_destination_start_addr[current_cluster_idx] += ((uint64_t)0x10)
                                                                << 40;

            // Start torturing the AXI
            for (uint32_t i = 0; i < torture_times; i++) {
                snrt_dma_start_1d_wideptr(
                    next_destination_start_addr[current_cluster_idx],
                    next_source_start_addr[current_cluster_idx], length_data);
                snrt_dma_wait_all();
                snrt_dma_start_1d_wideptr(
                    next_source_start_addr[current_cluster_idx],
                    next_destination_start_addr[current_cluster_idx],
                    length_data);
                snrt_dma_wait_all();
            }
        }
    }

    if (current_chip_id == 0x10) {
        // Utilize Cluster 2 and Cluster 3 to copy data forth and back
        if (snrt_is_dm_core() &&
            (snrt_cluster_idx() == 2 | snrt_cluster_idx() == 3)) {
            uint8_t current_cluster_idx = snrt_cluster_idx();
            next_source_start_addr[current_cluster_idx] =
                next_destination_start_addr[current_cluster_idx];
            next_destination_start_addr[current_cluster_idx] =
                (uint64_t)snrt_cluster_base_addrl();
            next_destination_start_addr[current_cluster_idx] += ((uint64_t)0x00)
                                                                << 40;

            // Start torturing the AXI
            for (uint32_t i = 0; i < torture_times; i++) {
                snrt_dma_start_1d_wideptr(
                    next_destination_start_addr[current_cluster_idx],
                    next_source_start_addr[current_cluster_idx], length_data);
                snrt_dma_wait_all();
                snrt_dma_start_1d_wideptr(
                    next_source_start_addr[current_cluster_idx],
                    next_destination_start_addr[current_cluster_idx],
                    length_data);
                snrt_dma_wait_all();
            }
        }
    }
    snrt_global_barrier();

    // Start to check: Ensure all cores go to one of the branch
    if (current_chip_id == 0x00 && snrt_is_dm_core() &&
        (snrt_cluster_idx() == 0 | snrt_cluster_idx() == 1)) {
        uint32_t* tcdm_start_addr = (uint32_t*)snrt_cluster_base_addrl();
        printf("Cluster %d Checking the results\r\n", snrt_cluster_idx());
        for (int i = 0; i < length_data; i++) {
            if (tcdm_start_addr[i] != test_data[i]) {
                err++;
                printf("Cluster %d data is incorrect!\r\n", snrt_cluster_idx());
                printf("tcdm[%d]=%d, test_data[%d]=%d\r\n", i,
                       tcdm_start_addr[i], i, test_data[i]);
                return -1;
            }
        }
    } else if (current_chip_id == 0x10 && snrt_is_dm_core() &&
               (snrt_cluster_idx() == 2 | snrt_cluster_idx() == 3)) {
        uint32_t* tcdm_start_addr = (uint32_t*)snrt_cluster_base_addrl();
        printf("Cluster %d Checking the results\r\n", snrt_cluster_idx());
        for (int i = 0; i < length_data; i++) {
            if (tcdm_start_addr[i] != test_data[i]) {
                err++;
                printf("Cluster %d data is incorrect!\r\n", snrt_cluster_idx());
                printf("tcdm[%d]=%d, test_data[%d]=%d\r\n", i,
                       tcdm_start_addr[i], i, test_data[i]);
                return -1;
            }
        }
    } else
        return 0;
}
