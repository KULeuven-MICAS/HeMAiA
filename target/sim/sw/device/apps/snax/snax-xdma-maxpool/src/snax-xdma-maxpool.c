// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

uint32_t tstride_src[5] = {0};
uint32_t tbound_src[5] = {0};
uint32_t tstride_dst[3] = {0};
uint32_t tbound_dst[3] = {0};

int main() {
    // Set err value for checking
    int err = 0;
    // Obtain the start address of the TCDM memory
    uint32_t dma_load_input_start;
    uint32_t dma_load_input_end;
    uint32_t tcdm_baseaddress = snrt_cluster_base_addrl();
    // Currently tcdm_in is at the start of the TCDM
    uint8_t *tcdm_in = (uint8_t *)tcdm_baseaddress;
    // tcdm_out is also at the start of the TCDM
    uint8_t *tcdm_out = (uint8_t *)(tcdm_baseaddress);

    if (snrt_is_dm_core() && snrt_cluster_idx() == 0) {
        // The xdma core is the last compute core in the cluster

        // Load the CFG from data.h
        tstride_src[0] = tempStride0_in;
        tstride_src[1] = tempStride1_in;
        tstride_src[2] = tempStride2_in;
        tstride_src[3] = tempStride3_in;
        tstride_src[4] = tempStride4_in;
        tbound_src[0] = tempLoop0_in;
        tbound_src[1] = tempLoop1_in;
        tbound_src[2] = tempLoop2_in;
        tbound_src[3] = tempLoop3_in;
        tbound_src[4] = tempLoop4_in;
        tstride_dst[0] = tempStride0_out;
        tstride_dst[1] = tempStride1_out;
        tstride_dst[2] = tempStride2_out;
        tbound_dst[0] = tempLoop0_out;
        tbound_dst[1] = tempLoop1_out;
        tbound_dst[2] = tempLoop2_out;

        tcdm_out = tcdm_out + delta_local_out * sizeof(int8_t);

        // Reference Group: Copy by IDMA + MaxPool to compute locally

        // --------------------- Configure the Ext --------------------- //

        if (xdma_disable_dst_ext(0) != 0) {
            printf("Error in disabling xdma extension 0\r\n");
            err++;
        }

        uint32_t ext_param_maxpool_size[1] = {reduceLen};
        if (xdma_enable_dst_ext(1, ext_param_maxpool_size) != 0) {
            printf("Error in enabling xdma extension 1\r\n");
            err++;
        }

        if (xdma_disable_dst_ext(2) != 0) {
            printf("Error in disabling xdma extension 2\r\n");
            err++;
        }

        // --------------------- Configure the AGU --------------------- //
        xdma_memcpy_nd(tcdm_in, tcdm_out, spatialStride1_in, spatialStride1_out,
                       5, tstride_src, tbound_src, 3, tstride_dst, tbound_dst,
                       0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

        register uint32_t start_time;
        register uint32_t end_time;
        __asm__ volatile("csrr %0, mcycle;" : "=r"(start_time));
        // First we need to transfer the input data from L3->TCDM
        snrt_dma_start_1d(tcdm_in, DataIn, input_data_len * sizeof(int8_t));
        snrt_dma_wait_all();
        // Then start XDMA to do the maxpool
        int task_id = xdma_start();
        xdma_local_wait(task_id);
        __asm__ volatile("csrr %0, mcycle;" : "=r"(end_time));
        printf("The IDMA + MaxPool copy is finished in %d cycles\r\n",
               end_time - start_time);

        // --------------------- Checking the Results --------------------- //
        #ifdef XDMA_CHECK_RESULT
        for (int i = 0; i < output_data_len; i++) {
            if ((int8_t)tcdm_out[i] != C_golden[i]) {
                printf("The maxpool is incorrect!\r\n");
                printf("tcdm_out[%d]=%d, C_golden[%d]=%d", i,
                       (int8_t)tcdm_out[i], i, C_golden[i]);
            }
        }
        printf("Checking is done. All values are right\r\n");
        #endif
    }

    snrt_global_barrier();

    if (snrt_is_dm_core() && snrt_cluster_idx() == 1) {
        tcdm_in = tcdm_in - cluster_offset;
        // --------------------- Configure the AGU --------------------- //
        xdma_memcpy_nd(tcdm_in, tcdm_out, spatialStride1_in, spatialStride1_out,
                       5, tstride_src, tbound_src, 3, tstride_dst, tbound_dst,
                       0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

        register uint32_t start_time;
        register uint32_t end_time;
        __asm__ volatile("csrr %0, mcycle;" : "=r"(start_time));
        // Start XDMA to do the maxpool
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        __asm__ volatile("csrr %0, mcycle;" : "=r"(end_time));
        printf("The XDMA copy is finished in %d cycles\r\n",
               end_time - start_time);

        // --------------------- Checking the Results --------------------- //
        #ifdef XDMA_CHECK_RESULT
        for (int i = 0; i < output_data_len; i++) {
            if ((int8_t)tcdm_out[i] != C_golden[i]) {
                printf("The maxpool is incorrect!\r\n");
                printf("tcdm_out[%d]=%d, C_golden[%d]=%d", i,
                       (int8_t)tcdm_out[i], i, C_golden[i]);
            }
        }
        printf("Checking is done. All values are right\r\n");
        #endif
    }

    return 0;
}
