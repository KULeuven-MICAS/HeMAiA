// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

// Put the input at the starting of tcdm
void *tcdm_in = 0;
// Put the output at the middle of tcdm
void *tcdm_out[MULTICAST_NUM] = {0};

int main() {
    // Set err value for checking
    int err = 0;
    // Obtain the start address of the TCDM memory
    uint32_t tcdm_baseaddress = snrt_cluster_base_addrl();
    // (uint8_t *)(tcdm_baseaddress +
    //             (matrix_size * sizeof(uint8_t) * 8 + 7) / 8);

    if (snrt_cluster_idx() == 0 && snrt_is_dm_core()) {
        tcdm_in = (void *)tcdm_baseaddress;
        // First we need to transfer the input data from L3->TCDM
        snrt_dma_start_1d(tcdm_in, input_matrix,
                          matrix_size * sizeof(input_matrix[0]));
        snrt_dma_wait_all();

        // The baseline 1 group: Copy by IDMA to cluster 1, then copy cluster
        // 1's data to 2 -> n (If broadcast to more than one destination)
        for (uint32_t i = 0; i < MULTICAST_NUM; i++) {
            tcdm_out[i] = tcdm_in + (multicast_pointers[i]) * cluster_offset;
        }
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
                (char *)((uint32_t)tcdm_out[0] +
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

        snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY,
                                snrt_hartid());
        for (int i = 0; i < TOTAL_ITERATIONS_IDMA; i++) {
            snrt_dma_start_2d(dst_addr[i], src_addr[i], size_idma,
                              dst_stride_idma, src_stride_idma, repeat_idma);
        }
        for (int i = 1; i < MULTICAST_NUM; i++) {
            snrt_dma_start_1d(tcdm_out[i], tcdm_out[0], matrix_size);
        }
        snrt_dma_wait_all();
        printf("The IDMA copy to %d dest is finished in %d cycles\r\n",
               MULTICAST_NUM, snrt_get_perf_counter(SNRT_PERF_CNT0));

        // The baseline 2 group: Copy by XDMA1
        uint32_t elapsed_time = 0;
        int task_id = 0;
        // --------------------- Configure the Ext --------------------- //

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

        for (uint32_t i = 0; i < MULTICAST_NUM; i++) {
            xdma_memcpy_nd(tcdm_in, tcdm_out[i], spatial_stride_src_xdma,
                           spatial_stride_dst_xdma, temporal_dimension_src_xdma,
                           temporal_strides_src_xdma, temporal_bounds_src_xdma,
                           temporal_dimension_dst_xdma,
                           temporal_strides_dst_xdma, temporal_bounds_dst_xdma,
                           0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
            task_id = xdma_start();
            xdma_remote_wait(task_id);
            elapsed_time += xdma_last_task_cycle();
        }
        printf("The XDMA copy to %d dest is finished in %d cycles\r\n",
               MULTICAST_NUM, elapsed_time);

        // The experiment group: Copy / multicast by XDMA2
        xdma_multicast_nd(
            tcdm_in, tcdm_out, MULTICAST_NUM, spatial_stride_src_xdma,
            spatial_stride_dst_xdma, temporal_dimension_src_xdma,
            temporal_strides_src_xdma, temporal_bounds_src_xdma,
            temporal_dimension_dst_xdma, temporal_strides_dst_xdma,
            temporal_bounds_dst_xdma, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf("The XDMA multicast to %d dest is finished in %d cycles\r\n",
               MULTICAST_NUM, xdma_last_task_cycle());
    }
    snrt_global_barrier();

// --------------------- Checking the Results --------------------- //
#ifdef XDMA_CHECK_RESULT
    if (snrt_is_dm_core() && snrt_cluster_idx() <= MULTICAST_NUM &&
        snrt_cluster_idx() > 0) {
        uint32_t *golden_result = (uint32_t *)golden_output_matrix;
        uint32_t *tcdm_result = (uint32_t *)tcdm_out[snrt_cluster_idx() - 1];

        for (int i = 0; i < matrix_size * sizeof(input_matrix[0]) / 4; i++) {
            if (tcdm_result[i] != golden_result[i]) {
                printf("Cluster %d: The reshape is incorrect at byte %d! \n",
                       snrt_cluster_idx(), i << 2);
            }
        }
    }
#endif

    return 0;
}
