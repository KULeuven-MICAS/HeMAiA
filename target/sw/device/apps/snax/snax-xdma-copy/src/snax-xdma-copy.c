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
        xdma_memcpy_1d(data, (void *)tcdm_baseaddress,
                       data_size * sizeof(data[0]));
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf("XDMA copy from L3 to TCDM C0 is done in %d cycles.\r\n",
               xdma_last_task_cycle());
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
                       (void *)tcdm_baseaddress, data_size * sizeof(data[0]));
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf("XDMA copy from TCDM C0 to TCDM C1 is done in %d cycles.\r\n",
               xdma_last_task_cycle());

#ifdef XDMA_CHECK_RESULT
        // Check the result
        uint32_t *golden_result = (uint32_t *)data;
        uint32_t *tcdm_result = (uint32_t *)tcdm_baseaddress;

        for (int i = 0; i < data_size * sizeof(data[0]) / 4; i++) {
            if (tcdm_result[i] != golden_result[i]) {
                printf("The data copy is incorrect at byte %d! \n", i << 2);
            }
        }
        printf("Checking is done. All values are right\n");
#endif
        xdma_memcpy_1d((void *)tcdm_baseaddress, (void *)data,
                       data_size * sizeof(data[0]));
        task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf("XDMA copy from TCDM C1 to L3 is done in %d cycles.\r\n",
               xdma_last_task_cycle());
#ifdef XDMA_CHECK_RESULT
        // Check the result
        uint32_t *golden_result = (uint32_t *)tcdm_baseaddress;
        uint32_t *tcdm_result = (uint32_t *)data;

        for (int i = 0; i < data_size * sizeof(data[0]) / 4; i++) {
            if (tcdm_result[i] != golden_result[i]) {
                printf("The data copy is incorrect at byte %d! \n", i << 2);
            }
        }
        printf("Checking is done. All values are right\n");
#endif
    }

    return 0;
}
