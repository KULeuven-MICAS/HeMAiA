// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

// The purpose of this program is to test the integration of the system
// We initialize two clusters
// cluster 0 fetch data from the l3 -> tcdm0
// cluster 1 fetch data from tcdm0 -> tcdm1
// Then each cluster check data is correct or not
#include "data.h"

#include "snrt.h"

uint64_t tcdm0_start_addr;
uint64_t tcdm1_start_addr;
uint64_t test_data_start_addr;
#define TCDM_OFFSET 0x1000
int main() {
    int err = 0;
    // First set the addr of cluster 0
    // tcdm0_start_addr = (int8_t*)0x10000000;
    // tcdm1_start_addr = (int8_t*)0x10100000;
    if (snrt_cluster_idx() == 0) {
        if (snrt_is_dm_core()) {
            tcdm0_start_addr = (uint64_t)snrt_cluster_base_addrl();
            tcdm0_start_addr += (uint64_t)snrt_cluster_base_addrh() << 32;
            printf("The C0 TCDM ADDR is %p%p \r\n",
                   (uint8_t*)(tcdm0_start_addr >> 32),
                   (uint8_t*)tcdm0_start_addr);
        }
    }
    snrt_global_barrier();

    if (snrt_cluster_idx() == 1) {
        if (snrt_is_dm_core()) {
            tcdm1_start_addr = (uint64_t)snrt_cluster_base_addrl();
            tcdm1_start_addr += (uint64_t)snrt_cluster_base_addrh() << 32;
            printf("The C1 TCDM ADDR is %p%p \r\n",
                   (uint8_t*)(tcdm1_start_addr >> 32),
                   (uint8_t*)tcdm1_start_addr);
        }
    }
    snrt_global_barrier();
    // C0 Load the data from l3 -> l1
    if (snrt_cluster_idx() == 0) {
        if (snrt_is_dm_core()) {
            test_data_start_addr = (uint64_t)test_data;
            test_data_start_addr += (uint64_t)snrt_cluster_base_addrh() << 32;
            printf("[C0] Start to load data from %lx \r\n", test_data_start_addr+TCDM_OFFSET);
            snrt_dma_start_1d_wideptr((tcdm0_start_addr+TCDM_OFFSET), test_data_start_addr,
                                      length_data);
            snrt_dma_wait_all();
        }
    }
    // wait the C0 is done
    snrt_global_barrier();

    // Thenc C1 fetches data from C0
    if (snrt_cluster_idx() == 1) {
        if (snrt_is_dm_core()) {
            printf("[C1] Start to load data from %lx \r\n", tcdm0_start_addr+TCDM_OFFSET);
            snrt_dma_start_1d_wideptr((tcdm1_start_addr+TCDM_OFFSET), (tcdm0_start_addr+TCDM_OFFSET),
                                      length_data);
            snrt_dma_wait_all();
        }
    }
    // wait the C1 is done
    snrt_global_barrier();

    // Start to check
    if (snrt_cluster_idx() == 0 && snrt_is_dm_core()) {
        printf_safe("C0 Checking the results\r\n");
        for (int i = 0; i < length_data; i++) {
            if (((int8_t*)(tcdm0_start_addr+TCDM_OFFSET))[i] != test_data[i]) {
                err++;
                printf_safe("C0 data is incorrect!\r\n");
                printf_safe("tcdm0[%d]=%d, test_data[%d]=%d\r\n", i,
                       ((int8_t*)(tcdm0_start_addr+TCDM_OFFSET))[i], i, test_data[i]);
                return -1;
            }
        }
    } else if (snrt_cluster_idx() == 1 && snrt_is_dm_core()) {
        printf_safe("C1 Checking the results\r\n");
        for (int i = 0; i < length_data; i++) {
            if (((int8_t*)(tcdm1_start_addr+TCDM_OFFSET))[i] != test_data[i]) {
                err++;
                printf_safe("C1 data is incorrect!\r\n");
                printf_safe("tcdm1[%d]=%d, test_data[%d]=%d\r\n", i,
                       ((int8_t*)(tcdm1_start_addr+TCDM_OFFSET))[i], i, test_data[i]);
                return -1;
            }
        }
    } else
        return 0;
}
