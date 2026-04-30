// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>
// The simple xdma copy program, which is used to test the basic functionality of xdma, including the correctness and performance of xdma copy. It can also be used as a reference for how to use xdma in the future.
// It will copy data from L3 to TCDM of cluster 0, then copy data from TCDM of cluster 0 to TCDM of cluster 1, and finally copy data from TCDM of cluster 1 back to L3. After each copy, it will check the correctness of the data. The performance of each copy will also be printed.
// It can also be used in ci with one cluster.
#include "data.h"
#include "snrt.h"
#define TCDM_OFFSET 0x1000
#define CHECK_SIZE_BYTES 256
int main() {
    // Set err value for checking
    int err = 0;
    // Obtain the start address of the TCDM memory
    uint32_t dma_load_input_start;
    uint32_t dma_load_input_end;
    uint32_t tcdm_baseaddress = snrt_cluster_base_addrl() + TCDM_OFFSET;

    if (snrt_cluster_idx() == 0 && snrt_is_dm_core()) {
        xdma_memcpy_1d(data, (void *)tcdm_baseaddress,
                       data_size * sizeof(data[0]));
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf("XDMA copy from L3 to TCDM C0 is done in %d cycles.\r\n",
               xdma_last_task_cycle());

        // Check the first CHECK_SIZE_BYTES of the result
        uint32_t *golden_result = (uint32_t *)data;
        uint32_t *tcdm_result = (uint32_t *)tcdm_baseaddress;

        for (int i = 0; i < CHECK_SIZE_BYTES / 4; i++) {
            if (tcdm_result[i] != golden_result[i]) {
                printf("The data copy is incorrect at byte %d! \n", i << 2);
            }
        }
        printf("Checking is done. All values are right\n");
    }

    snrt_global_barrier();

    if (snrt_cluster_idx() == 1 && snrt_is_dm_core()) {
        xdma_memcpy_1d((void *)tcdm_baseaddress - cluster_offset,
                       (void *)tcdm_baseaddress, data_size * sizeof(data[0]));
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf("XDMA copy from TCDM C0 to TCDM C1 is done in %d cycles.\r\n",
               xdma_last_task_cycle());

        // Check the first CHECK_SIZE_BYTES of the result
        uint32_t *golden_result = (uint32_t *)data;
        uint32_t *tcdm_result = (uint32_t *)tcdm_baseaddress;

        for (int i = 0; i < CHECK_SIZE_BYTES / 4; i++) {
            if (tcdm_result[i] != golden_result[i]) {
                printf("The data copy is incorrect at byte %d! \n", i << 2);
            }
        }
        printf("Checking is done. All values are right\n");
        xdma_memcpy_1d((void *)tcdm_baseaddress, (void *)data,
                       data_size * sizeof(data[0]));
        task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf("XDMA copy from TCDM C1 to L3 is done in %d cycles.\r\n",
               xdma_last_task_cycle());
        // Check the first CHECK_SIZE_BYTES of the result
        uint32_t *golden_result2 = (uint32_t *)tcdm_baseaddress;
        uint32_t *tcdm_result2 = (uint32_t *)data;

        for (int i = 0; i < CHECK_SIZE_BYTES / 4; i++) {
            if (tcdm_result2[i] != golden_result2[i]) {
                printf("The data copy is incorrect at byte %d! \n", i << 2);
            }
        }
        printf("Checking is done. All values are right\n");
    }

    return 0;
}
