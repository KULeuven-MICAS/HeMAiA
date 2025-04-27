// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

int main() {
    // Set err value for checking
    // We first load the MNM8N8 layout matrix to C0
    int err = 0;
    uint32_t tcdm_baseaddress = snrt_cluster_base_addrl();
    uint8_t *tcdm_dst = 0;
    uint8_t *tcdm_src = 0;
    uint8_t *golden_result = (uint8_t *)golden_output_matrix;
    if (snrt_cluster_idx() == 0 && snrt_is_dm_core()) {
        tcdm_dst = (void *)tcdm_baseaddress;
        // First we need to transfer the input data from L3->TCDM
        snrt_dma_start_1d(tcdm_dst, input_matrix,
                          matrix_size * sizeof(input_matrix[0]));
        snrt_dma_wait_all();
    }

    snrt_global_barrier();
    if (snrt_cluster_idx() == 0 && snrt_is_dm_core()) {
        // The baseline group: Write by IDMA
        uint32_t start_time;
        uint32_t end_time;
        tcdm_dst = (void *)tcdm_baseaddress + cluster_offset;
        tcdm_src = (void *)tcdm_baseaddress;
        // Then we use the idma 2d to store data to C1
        // dst = C1
        // src = C0
        // size = 8B
        // dst_stride = 8
        // src_stride = 64
        // repeat = 64
        __asm__ volatile("nop;nop;nop;nop;nop;nop;nop;");
        __asm__ volatile("csrr %0, mcycle;" : "=r"(start_time));
        //snrt_dma_start_2d(void *dst, const void *src, size_t size, size_t dst_stride, size_t src_stride, size_t repeat)
        snrt_dma_start_2d(tcdm_dst, //dst
                          tcdm_src, //src
                          8,  //size
                          8,  //dst_stride
                          64, //src_stride
                          64  //repeat
                        );
        snrt_dma_wait_all();
        __asm__ volatile("csrr %0, mcycle;" : "=r"(end_time));
        printf("[XDMA-kvcache-store]The IDMA store is finished in %d cycles\r\n",
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
        if (xdma_disable_src_ext(0) != 0) {
            printf("Error in disabling xdma extension 0\r\n");
            err++;
        }

        // --------------------- Configure the AGU --------------------- //
        // SRC = C0
        // DST = C1
        // Put the xdma data into another place
        tcdm_dst = (void *)tcdm_baseaddress + cluster_offset + 4096*1024;
        uint32_t man_spatial_stride_src_xdma = 64;
        uint32_t man_spatial_stride_dst_xdma = 8;
        uint32_t man_temporal_dimension_src_xdma = 1;
        uint32_t man_temporal_strides_src_xdma[1]  = {
            512,
        };
        uint32_t man_temporal_bounds_src_xdma[1]  = {
            8,
        };
        uint32_t man_temporal_dimension_dst_xdma = 1;
        uint32_t man_temporal_strides_dst_xdma[1]  = {
            64,
        };       
        uint32_t man_temporal_bounds_dst_xdma[1]  = {
            8,
        };   

        xdma_memcpy_nd(tcdm_src, tcdm_dst, man_spatial_stride_src_xdma,
                       man_spatial_stride_dst_xdma, man_temporal_dimension_src_xdma,
                       man_temporal_strides_src_xdma, man_temporal_bounds_src_xdma,
                       man_temporal_dimension_dst_xdma, man_temporal_strides_dst_xdma,
                       man_temporal_bounds_dst_xdma, 0xFFFFFFFF, 0xFFFFFFFF,
                       0xFFFFFFFF);
        __asm__ volatile("nop;nop;nop;nop;nop;nop;nop;");
        __asm__ volatile("csrr %0, mcycle;" : "=r"(start_time));
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        __asm__ volatile("csrr %0, mcycle;" : "=r"(end_time));
        printf("[XDMA-kvcache-store]The XDMA store is finished in %d cycles\r\n",
               end_time - start_time);
    }
    snrt_global_barrier();
    // --------------------- Checking the Results --------------------- // 
    // First load the golden matrix to the local tcdm
    if (snrt_cluster_idx() == 1 && snrt_is_dm_core()) {
        // Check the IDMA result
        tcdm_dst = (void *)tcdm_baseaddress;
        for (int i = 0; i < 64; i++) {
            printf("Check IDMA %d\r\n", i);
            if ((uint32_t *)tcdm_dst[i] != (uint32_t *)golden_result[i]) {
                printf("tcdm[%d]=%d, golden[%d]=%d\n", i, (uint32_t *)tcdm_dst[i], i, (uint32_t *)golden_result[i]);
            }
        }
        printf("[XDMA-kvcache-store]Checking IDMA is done. All values are right\r\n");        
        // Check the XDMA result
        tcdm_dst = (void *)tcdm_baseaddress + 4096*1024;
        for (int i = 0; i < 64; i++) {
            printf("Check XDMA %d\r\n", i);
            if ((uint32_t *)tcdm_dst[i] != (uint32_t *)golden_result[i]) {
                printf("tcdm[%d]=%d, golden[%d]=%d\n", i, (uint32_t *)tcdm_dst[i], i, (uint32_t *)golden_result[i]);
            }
        }
        printf("[XDMA-kvcache-store]Checking XDMA is done. All values are right\r\n");  
    }

    return 0;
}
