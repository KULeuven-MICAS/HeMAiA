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

        // Broacast to j Destination - IDMA and XDMA - Unoptimized
        void *dest[MULTICAST_NUM];

        for (int i = 0; i < MULTICAST_NUM; i++) {
            dest[i] = (void *)(tcdm_baseaddress +
                               (multicast_pointers[i]) * cluster_offset);
        }

        // Reference group:
        snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY,
                                snrt_hartid());
        for (int i = 0; i < MULTICAST_NUM; i++) {
            snrt_dma_start_1d(dest[i], (void *)tcdm_baseaddress,
                              data_size * sizeof(data[0]));
        }
        snrt_dma_wait_all();
        printf("The IDMA copy to %d dest is finished in %d cycles\r\n",
               MULTICAST_NUM, snrt_get_perf_counter(SNRT_PERF_CNT0));
        snrt_reset_perf_counter(SNRT_PERF_CNT0);

        // Experiment group 1:
        xdma_multicast_1d((void *)tcdm_baseaddress, dest, MULTICAST_NUM,
                          data_size * sizeof(data[0]));
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf(
            "The XDMA normal copy to %d dest is finished in %d cycles with "
            "hops of %d\r\n",
            MULTICAST_NUM, xdma_last_task_cycle(), multicast_total_hops);

        // Experiment group 2:
        for (int i = 0; i < MULTICAST_NUM; i++) {
            dest[i] =
                (void *)(tcdm_baseaddress +
                         (multicast_pointers_optimized[i]) * cluster_offset);
        }
        xdma_multicast_1d((void *)tcdm_baseaddress, dest, MULTICAST_NUM,
                          data_size * sizeof(data[0]));
        task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf(
            "The XDMA optimal copy to %d dest is finished in %d cycles with "
            "hops of %d\r\n",
            MULTICAST_NUM, xdma_last_task_cycle(),
            multicast_pointers_optimized);
    }

    return 0;
}
