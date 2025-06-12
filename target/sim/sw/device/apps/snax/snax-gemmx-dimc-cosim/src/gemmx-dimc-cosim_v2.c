// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// smoke test for DIMC cluster & gemmx cluster

// Version 1 function: DIMC loads data to TCDM, GEMMX reads that chunk of data

// Version 2 function: DIMC computes one head, GEMMX reads the result

// Version 3 function: DIMC computes one head, GEMMX reads the result and compute MM
#include "snax-dimc-csr.h"
#include "snax-dimc-lib.h"
#include "snrt.h"
#include "data_mha.h"

#include "data.h"
#include "snax-gemmx-params.h"
#include "snax-gemmx-lib.h"

uint64_t * tcdm_1_start_addr; // starting address for GEMMX cluster
uint64_t * tcdm_3_start_addr; // starting address for DIMC cluster

int main() {

    // set error value for checking
    int err = 0;
    
    if (snrt_cluster_idx() == 3){

        // allocate 32+32+32KB in TCDM for activation and weight pair
        uint64_t *tcdm_ptr_0, *tcdm_ptr_1, *tcdm_ptr_2, *tcdm_ptr_danger;

        tcdm_ptr_0 = (uint64_t *)snrt_cluster_base_addrl();
        tcdm_ptr_1     = tcdm_ptr_0 + Q_LENGTH;
        tcdm_ptr_2     = tcdm_ptr_1 + Q_LENGTH;
        // must not fully use the 32KB space
        tcdm_ptr_danger     = tcdm_ptr_2 + Q_LENGTH;

        tcdm_3_start_addr = tcdm_ptr_0;

        if (snrt_is_dm_core()) {
            // read weight WK and ativation K from data.h
            size_t vector_size = Q_LENGTH * sizeof(uint64_t);
            snrt_dma_start_1d(tcdm_ptr_0, K,  vector_size);
            snrt_dma_start_1d(tcdm_ptr_1, Q,  vector_size);
            snrt_dma_start_1d(tcdm_ptr_2, WK, vector_size);

            snrt_dma_wait_all();
        }

        snrt_cluster_hw_barrier();

        if (snrt_cluster_core_idx() == 0) {
            uint32_t busy = dimc_query_busy();
            printf("%d: busy\n", busy);
            printf("QUERYING BUSY SUCCEEDED\n");

            configure_accelerator();

            // send WK
            dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
            dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_2));
            dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_2 + 8));
            dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_2 + 16));
            dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_2 + 24));
            
            dimc_start_mha();

            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
        }
        
        snrt_cluster_hw_barrier();

        // load weight WQ from data.h to tcdm
        if (snrt_is_dm_core()) {
            size_t vector_size = Q_LENGTH * sizeof(uint64_t);

            snrt_dma_start_1d(tcdm_ptr_2, WQ, vector_size);

            snrt_dma_wait_all();
        }

        if (snrt_cluster_core_idx() == 0) {
            // send K
            dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
            dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_0));
            dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_0 + 8));
            dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_0 + 16));
            dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_0 + 24));

            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
        }
        
        snrt_cluster_hw_barrier();
        
        if (snrt_cluster_core_idx == 0 ) {
            // printf("CONFIGURING STREAMERS for WQ\n");
            dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
            dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_2));
            dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_2 + 8));
            dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_2 + 16));
            dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_2 + 24));
            // printf("STREAMER CONFIGURED for WQ\n");

            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
        }

        snrt_cluster_hw_barrier();

        if (snrt_is_dm_core()) {
            
            // load WV and V from data.h to tcdm
            size_t vector_size = Q_LENGTH * sizeof(uint64_t);

            snrt_dma_start_1d(tcdm_ptr_0, V,  vector_size);
            snrt_dma_start_1d(tcdm_ptr_2, WV, vector_size);

            snrt_dma_wait_all();
        }
        
        snrt_cluster_hw_barrier();

        if (snrt_cluster_core_idx() == 0) {
            // send Q
            // printf("CONFIGURING STREAMERS for Q\n");
            dimc_set_streamer_dim_w(64, 1, 64, 0, 8, (uint32_t)(tcdm_ptr_danger));
            dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_1));
            dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_1 + 8));
            dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_1 + 16));
            dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_1 + 24));
            // printf("STREAMER CONFIGURED for Q\n");

            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
        }
    }

    snrt_global_barrier();

    if (snrt_cluster_idx() == 3){
        if (snrt_cluster_core_idx() == 0) {
            // send V
            // printf("CONFIGURING STREAMERS for V\n");
            dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
            dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_0));
            dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_0 + 8));
            dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_0 + 16));
            dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_0 + 24));
            // printf("STREAMER CONFIGURED for V\n");

            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
        }
    }

    snrt_global_barrier();

    if (snrt_cluster_idx() == 3){
        if (snrt_cluster_core_idx() == 0) {
            // send Q1K1T
            printf("CONFIGURING STREAMERS for Q1K1T\n");
            dimc_set_streamer_dim_w(64, 1, 64, 0, 8, (uint32_t)(tcdm_ptr_0));
            dimc_set_streamer_dim_r0(16, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_danger));
            dimc_set_streamer_dim_r1(16, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_danger + 8));
            dimc_set_streamer_dim_r2(16, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_danger + 16));
            dimc_set_streamer_dim_r3(16, 1, 256, 0, 8, (uint32_t)(tcdm_ptr_danger + 24));
            printf("STREAMER CONFIGURED for Q1K1T\n");

            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
        }
    }

    snrt_global_barrier();

    if (snrt_cluster_idx() == 1){
        
        tcdm_1_start_addr = (uint64_t *)snrt_cluster_base_addrl();

        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(tcdm_1_start_addr, tcdm_3_start_addr, 512 * sizeof(uint64_t));
            snrt_dma_wait_all();
        }
    }

    snrt_global_barrier();

    if (snrt_cluster_idx() == 1) {
        if (snrt_cluster_core_idx() == 0) {
            printf("GEMMX cluster starts to check copied data]\n");
            // check the final result
            
            for (int i = 0; i < 512; ++i) {
                uint64_t value = tcdm_ptr_0[i];

                uint64_t index = i * 8;

                // Split each uint64_t element into 8 uint8_t elements
                for (int j = 0; j < 8; ++j) {
                    uint8_t tmp_res = (uint8_t)((value >> (j * 8)) & 0xFF);
                    // printf("%d ", tmp_res);
                    if(tmp_res != gold[index + j]) {
                        printf("MISMATCH at %d, res:%d, gold:%d\n", (index + j), tmp_res, gold[index + j]);
                        err += 1;
                    }
                }
                printf("RESULTS MATCH WITH GOLDEN MODEL\n");
            }
        }
    }

    snrt_global_barrier();
    
    return err;
}
