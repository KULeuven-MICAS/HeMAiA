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
    void *tcdm_in = 0;
    void *tcdm_out = 0;

    if (snrt_cluster_idx() == 1 && snrt_is_dm_core()) {
        tcdm_in = (void *)tcdm_baseaddress;
        // First we need to transfer the input data from L3->TCDM
        snrt_dma_start_1d(tcdm_in, input_matrix,
                          matrix_size * sizeof(input_matrix[0]));
        snrt_dma_wait_all();
    }

    snrt_global_barrier();

    if (snrt_cluster_idx() == 0 && snrt_is_dm_core()) {
        // The baseline group: Copy by IDMA + transposer to transpose locally
        tcdm_in = (void *)tcdm_baseaddress + cluster_offset;
        tcdm_out = (void *)tcdm_baseaddress;
        void *tcdm_out_transposed =
            (void *)tcdm_baseaddress +
            (matrix_size * sizeof(input_matrix[0]) + 7) / 8 * 8;

        // --------------------- Configure the Ext --------------------- //

        if (xdma_disable_src_ext(0) != 0) {
            printf("Error in disabling reader xdma extension 0\n");
            err++;
        }

        if (xdma_disable_dst_ext(0) != 0) {
            printf("Error in disabling writer xdma extension 1\n");
            err++;
        }

        if (enable_transpose) {
            if (xdma_enable_dst_ext(1, (uint32_t *)transposer_param) != 0) {
                printf("Error in enabling xdma writer extension 1\n");
                err++;
            }
        } else {
            if (xdma_disable_dst_ext(1) != 0) {
                printf("Error in disabling xdma writer extension 1\n");
                err++;
            }
        }

        xdma_memcpy_nd(tcdm_out, tcdm_out_transposed, spatial_stride_src,
                       spatial_stride_dst, temporal_dimension_src,
                       temporal_strides_src, temporal_bounds_src,
                       temporal_dimension_dst, temporal_strides_dst,
                       temporal_bounds_dst, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

        register uint32_t start_time;
        register uint32_t end_time;

        __asm__ volatile("fence" ::: "memory");
        __asm__ volatile("csrr %0, mcycle;" : "=r"(start_time));
        snrt_dma_start_1d(tcdm_in, input_matrix,
                          matrix_size * sizeof(input_matrix[0]));
        snrt_dma_wait_all();
        int task_id = xdma_start();
        xdma_local_wait(task_id);
        __asm__ volatile("csrr %0, mcycle;" : "=r"(end_time));
        printf("The IDMA + Transposer copy is finished in %d cycles\r\n",
               end_time - start_time);

        // The experiment group: XDMA to do transpose during data copy
        tcdm_in = (void *)tcdm_baseaddress + cluster_offset;
        tcdm_out = (void *)tcdm_baseaddress;

        // --------------------- Configure the AGU --------------------- //
        xdma_memcpy_nd(tcdm_in, tcdm_out, spatial_stride_src,
                       spatial_stride_dst, temporal_dimension_src,
                       temporal_strides_src, temporal_bounds_src,
                       temporal_dimension_dst, temporal_strides_dst,
                       temporal_bounds_dst, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

        __asm__ volatile("fence" ::: "memory");
        __asm__ volatile("csrr %0, mcycle;" : "=r"(start_time));
        task_id = xdma_start();
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
                printf("The transpose is incorrect at byte %d! \n", i << 2);
            }
        }
        printf("Checking is done. All values are right\n");
#endif
    }

    return 0;
}
