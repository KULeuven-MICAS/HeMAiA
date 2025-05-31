// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

int main() {
    // Set err value for checking
    int err = 0;
    // Obtain the start address of the TCDM memory
    uint32_t dma_load_input_start;
    uint32_t dma_load_input_end;
    uint32_t tcdm_baseaddress = snrt_cluster_base_addrl();

    if (snrt_cluster_idx() == 0 && snrt_is_dm_core()) {
        // First we need to transfer the input data from L3->TCDM
        snrt_dma_start_1d((void *)tcdm_baseaddress, data,
                          data_size * sizeof(data[0]));
        snrt_dma_wait_all();

        // Configure the extension
        if (xdma_disable_dst_ext(0) != 0) {
            printf("Error in disabling xdma writer extension 0\n");
            err++;
        }
        if (xdma_disable_dst_ext(1) != 0) {
            printf("Error in disabling xdma writer extension 1\n");
            err++;
        }
        if (xdma_disable_src_ext(0) != 0) {
            printf("Error in disabling xdma reader extension 0\n");
            err++;
        }

        uint64_t dest[3];
        register uint32_t start_time;
        register uint32_t end_time;

        // Multicast evaluation to three destinations. The first test is
        // multicast to the clusters in the same chiplet.
        for (int i = 0; i < 3; i++) {
            dest[i] = (uint32_t)tcdm_baseaddress + (i + 1) * cluster_offset;
        }

        xdma_multicast_1d_full_address((uint64_t)tcdm_baseaddress,
                                       (uint64_t *)dest, 3,
                                       data_size * sizeof(data[0]));
        __asm__ volatile("csrr %0, mcycle;" : "=r"(start_time));
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        __asm__ volatile("csrr %0, mcycle;" : "=r"(end_time));
        printf("The XDMA copy is finished in %d cycles\r\n",
               end_time - start_time);

        // The seoncd test is multicast to the clusters in Chiplet (1, 0).
        for (int i = 0; i < 3; i++) {
            dest[i] = dest[i] + get_chip_baseaddress_value(1);
        }

        xdma_multicast_1d_full_address((uint64_t)tcdm_baseaddress,
                                       (uint64_t *)dest, 3,
                                       data_size * sizeof(data[0]));
        __asm__ volatile("csrr %0, mcycle;" : "=r"(start_time));
        task_id = xdma_start();
        xdma_remote_wait(task_id);
        __asm__ volatile("csrr %0, mcycle;" : "=r"(end_time));
        printf("The XDMA copy is finished in %d cycles\r\n",
               end_time - start_time);
    }
    return 0;
}
