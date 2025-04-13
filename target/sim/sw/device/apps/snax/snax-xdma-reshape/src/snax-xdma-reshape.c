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
    uint8_t *tcdm_in = 0;
    // Put the output at the middle of tcdm
    uint8_t *tcdm_out = 0;
    // (uint8_t *)(tcdm_baseaddress +
    //             (matrix_size * sizeof(uint8_t) * 8 + 7) / 8);

    if (snrt_cluster_idx() == 1 && snrt_is_dm_core()) {
        tcdm_in = (void *)tcdm_baseaddress;
        // First we need to transfer the input data from L3->TCDM
        snrt_dma_start_1d(tcdm_in, input_matrix,
                          matrix_size * sizeof(input_matrix[0]));
        snrt_dma_wait_all();
    }

    snrt_global_barrier();

    if (snrt_cluster_idx() == 0 && snrt_is_dm_core()) {
        tcdm_in = (void *)tcdm_baseaddress + cluster_offset;
        tcdm_out = (void *)tcdm_baseaddress;
        // The baseline group: Copy by IDMA
        char *src_addr[TOTAL_ITERATIONS_IDMA];
        char *dst_addr[TOTAL_ITERATIONS_IDMA];
        for (uint32_t i = 0; i < TOTAL_ITERATIONS_IDMA; i++) {
            src_addr[i] =
                (char *)((uint32_t)tcdm_in +
                         (i % sw_src_bound_idma[0]) * sw_src_stride_idma[0] +
                         (i / sw_src_bound_idma[0] % sw_src_bound_idma[1]) *
                             sw_src_stride_idma[1] +
                         (i / sw_src_bound_idma[0] / sw_src_bound_idma[1] %
                          sw_src_bound_idma[2]) *
                             sw_src_stride_idma[2] +
                         (i / sw_src_bound_idma[0] / sw_src_bound_idma[1] /
                          sw_src_bound_idma[2] % sw_src_bound_idma[3]) *
                             sw_src_stride_idma[3]);
            dst_addr[i] =
                (char *)((uint32_t)tcdm_out +
                         (i % sw_dst_bound_idma[0]) * sw_dst_stride_idma[0] +
                         (i / sw_dst_bound_idma[0] % sw_dst_bound_idma[1]) *
                             sw_dst_stride_idma[1] +
                         (i / sw_dst_bound_idma[0] / sw_dst_bound_idma[1] %
                          sw_dst_bound_idma[2]) *
                             sw_dst_stride_idma[2] +
                         (i / sw_dst_bound_idma[0] / sw_dst_bound_idma[1] /
                          sw_dst_bound_idma[2] % sw_dst_bound_idma[3]) *
                             sw_dst_stride_idma[3]);
        }

        uint32_t start_time;
        uint32_t end_time;
        __asm__ volatile("csrr %0, mcycle;" : "=r"(start_time));
        for (int i = 0; i < TOTAL_ITERATIONS_IDMA; i++) {
            snrt_dma_start_2d(dst_addr[i], src_addr[i], size_idma,
                              dst_stride_idma, src_stride_idma, repeat_idma);
        }
        snrt_dma_wait_all();
        __asm__ volatile("csrr %0, mcycle;" : "=r"(end_time));
        printf("The IDMA copy is finished in %d cycles\r\n",
               end_time - start_time);

        // The XDMA group: Copy by XDMA
        // --------------------- Configure the Ext --------------------- //

        if (xdma_disable_dst_ext(0) != 0) {
            printf("Error in disabling xdma extension 0\r\n");
            err++;
        }

        if (xdma_disable_dst_ext(1) != 0) {
            printf("Error in disabling xdma extension 1\r\n");
            err++;
        }
        if (xdma_disable_dst_ext(2) != 0) {
            printf("Error in disabling xdma extension 2\r\n");
            err++;
        }

        // --------------------- Configure the AGU --------------------- //
        xdma_memcpy_nd(tcdm_in, tcdm_out, spatial_stride_src_xdma,
                       spatial_stride_dst_xdma, temporal_dimension_src_xdma,
                       temporal_strides_src_xdma, temporal_bounds_src_xdma,
                       temporal_dimension_dst_xdma, temporal_strides_dst_xdma,
                       temporal_bounds_dst_xdma, 0xFFFFFFFF, 0xFFFFFFFF,
                       0xFFFFFFFF);

        __asm__ volatile("csrr %0, mcycle;" : "=r"(start_time));
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        __asm__ volatile("csrr %0, mcycle;" : "=r"(end_time));
        printf("The XDMA copy is finished in %d cycles\r\n",
               end_time - start_time);

        // --------------------- Checking the Results --------------------- //
        #ifdef XDMA_CHECK_RESULT
        uint32_t *golden_result = (uint32_t *)golden_output_matrix;
        uint32_t *tcdm_result = (uint32_t *)tcdm_out;

        for (int i = 0; i < matrix_size * sizeof(input_matrix[0]) / 4; i++) {
            if (tcdm_result[i] != golden_result[i]) {
                printf("The reshape is incorrect at byte %d! \n", i << 2);
            }
        }
        printf("Checking is done. All values are right\n");
        #endif
    }

    return 0;
}
