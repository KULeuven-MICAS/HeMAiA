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

        // Prepare the broadcast destination address
        uint32_t dest[4];
        for (int i = 0; i < 4; i++) {
            dest[i] = (uint32_t)tcdm_baseaddress + i * cluster_offset;
        }

        // Configure the extension
        if (xdma_disable_dst_ext(0) != 0) {
            printf("Error in disabling xdma extension 0\n");
            err++;
        }
        if (xdma_disable_dst_ext(1) != 0) {
            printf("Error in disabling xdma extension 1\n");
            err++;
        }
        if (xdma_disable_dst_ext(2) != 0) {
            printf("Error in disabling xdma extension 1\n");
            err++;
        }

        // Broacast to j Destination
        for (int j = 1; j <= 4; j++) {
            // Reference group:
            register uint32_t start_time;
            register uint32_t end_time;
            __asm__ volatile("csrr %0, mcycle;" : "=r"(start_time));
            for (int i = 0; i < j; i++) {
                snrt_dma_start_1d((void *)dest[i], (void *)tcdm_baseaddress,
                                  data_size * sizeof(data[0]));
            }
            snrt_dma_wait_all();
            __asm__ volatile("csrr %0, mcycle;" : "=r"(end_time));
            printf("The IDMA copy to %d dest is finished in %d cycles\r\n", j,
                   end_time - start_time);
            // Experiment group:
            xdma_multicast_1d((void *)tcdm_baseaddress, (void **)dest, j,
                              data_size * sizeof(data[0]));
            __asm__ volatile("csrr %0, mcycle;" : "=r"(start_time));
            int task_id = xdma_start();
            xdma_remote_wait(task_id);
            __asm__ volatile("csrr %0, mcycle;" : "=r"(end_time));
            printf("The XDMA copy to %d dest is finished in %d cycles\r\n", j,
                   end_time - start_time);
        }
    }

    return 0;
}
