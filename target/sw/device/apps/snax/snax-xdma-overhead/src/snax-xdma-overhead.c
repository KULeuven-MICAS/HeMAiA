// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

uint8_t *tcdm_c0 = 0;
uint8_t *tcdm_c1 = 0;
uint8_t *tcdm_c2 = 0;

int main() {
    // Set err value for checking
    int err = 0;
    // Obtain the start address of the TCDM memory
    uint32_t dma_load_input_start;
    uint32_t dma_load_input_end;
    uint32_t tcdm_baseaddress = snrt_cluster_base_addrl();

    if (snrt_cluster_idx() == 1 && snrt_is_dm_core()) {
        tcdm_c1 = (void *)tcdm_baseaddress;
        printf("Size: %d\r\n", matrix_size * sizeof(uint8_t));
        printf("1: %d\r\n",
               (matrix_size * sizeof(uint8_t) + (1 << 6) - 1) >> 6);
        snrt_dma_start_1d(tcdm_c1, input_matrix,
                          matrix_size * sizeof(input_matrix[0]));
        snrt_dma_wait_all();
    }

    snrt_global_barrier();

    if (snrt_cluster_idx() == 0 && snrt_is_dm_core()) {
        tcdm_c0 = (void *)(tcdm_baseaddress);
        // Experiment Group 2: IDMA copy
        snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY,
                                snrt_hartid());

        snrt_dma_start_1d(tcdm_c0, tcdm_c1,
                          matrix_size * sizeof(input_matrix[0]));
        snrt_dma_wait_all();
        printf("2: %d\r\n", snrt_get_perf_counter(SNRT_PERF_CNT0));

        // Experiment Group 3: XDMA remote read copy
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
        xdma_memcpy_1d(tcdm_c1, tcdm_c0, matrix_size * sizeof(input_matrix[0]));
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf("3: %d\r\n", xdma_last_task_cycle());
        // Experiment Group 4: XDMA remote write copy
        xdma_memcpy_1d(tcdm_c0, tcdm_c1, matrix_size * sizeof(input_matrix[0]));
        task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf("4: %d\r\n", xdma_last_task_cycle());
        // Experiment Group 5: XDMA chain write to 2 dests
        void *dest[2];
        dest[0] = tcdm_c1;
        tcdm_c2 = tcdm_c1 + cluster_offset;
        dest[1] = tcdm_c2;
        xdma_multicast_1d(tcdm_c0, dest, 2,
                          matrix_size * sizeof(input_matrix[0]));
        task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf("5: %d\r\n", xdma_last_task_cycle());
        // Experiment Group 6: XDMA nD chain write to 2 dests
        xdma_multicast_nd(tcdm_c0, dest, 2, spatial_stride_src_xdma,
                          spatial_stride_dst_xdma, temporal_dimension_src_xdma,
                          temporal_strides_src_xdma, temporal_bounds_src_xdma,
                          temporal_dimension_dst_xdma,
                          temporal_strides_dst_xdma, temporal_bounds_dst_xdma,
                          0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf("6: %d\r\n", xdma_last_task_cycle());

        // --------------------- Checking the Results --------------------- //
    }

    snrt_global_barrier();
#ifdef XDMA_CHECK_RESULT
    if ((snrt_cluster_idx() == 1 || snrt_cluster_idx() == 2) &&
        snrt_is_dm_core()) {
        uint32_t *golden_result = (uint32_t *)golden_output_matrix;
        uint32_t *tcdm_result = (uint32_t *)tcdm_c0;

        for (int i = 0; i < matrix_size * sizeof(input_matrix[0]) / 4; i++) {
            if (tcdm_result[i] != golden_result[i]) {
                printf("The reshape is incorrect at byte %d! \n", i << 2);
            }
        }
        printf("Checking is done. All values are right\n");
    }
#endif

    return 0;
}
