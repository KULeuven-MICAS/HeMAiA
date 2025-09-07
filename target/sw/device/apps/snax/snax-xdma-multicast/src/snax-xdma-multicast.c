// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include "data.h"
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
    }

    snrt_global_barrier();

    if (snrt_cluster_idx() == 1 && snrt_is_dm_core()) {
        // Normal copy evaluation
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

        xdma_memcpy_1d((void *)tcdm_baseaddress - cluster_offset,
                       (void *)(tcdm_baseaddress), data_size * sizeof(data[0]));
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf("The XDMA copy is finished in %d cycles\r\n",
               xdma_last_task_cycle());
    }

    snrt_global_barrier();

    if (snrt_cluster_idx() == 0 && snrt_is_dm_core()) {
        // Multicast evaluation
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

        // Broacast to j Destination
        void* dest[MULTICAST_NUM];
        for (int i = 0; i < MULTICAST_NUM; i++) {
            dest[i] = (void *)(tcdm_baseaddress + (multicast_pointers[i]) * cluster_offset);
        }

        for (int j = 1; j <= MULTICAST_NUM; j++) {
            // Reference group:
            snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY,
                                    snrt_hartid());
            for (int i = 0; i < j; i++) {
                snrt_dma_start_1d(dest[i], (void *)tcdm_baseaddress,
                                  data_size * sizeof(data[0]));
            }
            snrt_dma_wait_all();
            printf("The IDMA copy to %d dest is finished in %d cycles\r\n", j,
                   snrt_get_perf_counter(SNRT_PERF_CNT0));
            snrt_reset_perf_counter(SNRT_PERF_CNT0);

            // Experiment group:
            xdma_multicast_1d((void *)tcdm_baseaddress, dest, j,
                              data_size * sizeof(data[0]));
            int task_id = xdma_start();
            xdma_remote_wait(task_id);
            printf("The XDMA copy to %d dest is finished in %d cycles\r\n", j,
                   xdma_last_task_cycle());
        }
    }

    return 0;
}
