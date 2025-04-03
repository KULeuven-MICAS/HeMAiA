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
    // Put the input at the starting of tcdm
    uint8_t *tcdm_in = (uint8_t *)tcdm_baseaddress;
    // Put the output at the middle of tcdm
    uint8_t *tcdm_out =
        (uint8_t *)(tcdm_baseaddress +
                    (matrix_size * sizeof(uint8_t) * 8 + 7) / 8);

    if (snrt_is_dm_core() && snrt_cluster_idx() == 0) {
        // First we need to transfer the input data from L3->TCDM
        snrt_dma_start_1d(tcdm_in, input_matrix, matrix_size * sizeof(uint8_t));
        snrt_dma_wait_all();

        // --------------------- Configure the Ext --------------------- //

        if (xdma_disable_dst_ext(0) != 0) {
            printf("Error in disabling xdma extension 0\r\n");
            err++;
        }

        if (xdma_disable_dst_ext(1) != 0) {
            printf("Error in disabling xdma extension 1\r\n");
            err++;
        }

        if (enable_transpose) {
            if (xdma_enable_dst_ext(2, (uint32_t *)NULL) != 0) {
                printf("Error in enabling xdma extension 2\r\n");
                err++;
            }
        } else {
            if (xdma_disable_dst_ext(2) != 0) {
                printf("Error in disabling xdma extension 1\r\n");
                err++;
            }
        }

        // --------------------- Configure the AGU --------------------- //
        xdma_memcpy_nd(tcdm_in, tcdm_out, spatial_stride_src,
                       spatial_stride_dst, temporal_dimension_src,
                       temporal_strides_src, temporal_bounds_src,
                       temporal_dimension_dst, temporal_strides_dst,
                       temporal_bounds_dst, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        uint32_t start_time;
        uint32_t end_time;
        __asm__ volatile("fence" ::: "memory");
        __asm__ volatile("csrr %0, mcycle;" : "=r"(start_time));
        int task_id = xdma_start();
        xdma_local_wait(task_id);
        __asm__ volatile("csrr %0, mcycle;" : "=r"(end_time));
        printf("The xdma copy is finished in %d cycles\r\n",
               end_time - start_time);
        // --------------------- Checking the Results --------------------- //
        for (int i = 0; i < matrix_size; i++) {
            if (tcdm_out[i] != golden_output_matrix[i]) {
                printf("The transpose is incorrect!\r\n");
                printf("tcdm_out[%d]=%d, golden_output_matrix[%d]=%d\r\n", i,
                       tcdm_out[i], i, golden_output_matrix[i]);
            }
        }
        printf("Checking is done. All values are right\r\n");
    }

    return 0;
}
