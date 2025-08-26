// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// this version of the code will only allocate 64KB of TCDM for the matrix Q/WQ/K/WK/V/WV
// since the constraints of a limited TCDM (128-KB) and the existence of a memory stack

#include "snax-dimc-csr.h"
#include "snax-dimc-lib.h"
#include "snrt.h"
#include "data_mha.h"

int main() {

    // set error value for checking
    int err = 0;
    if (snrt_cluster_idx() == 3){

        // printf("Starting the DIMC test\r\n");

        // allocate 32+32+32KB in TCDM for activation and weight pair
        uint64_t *activation_ptr, *weight_ptr, *output_ptr;

        activation_ptr = (uint64_t *)snrt_l1_next();
        weight_ptr     = activation_ptr + Q_LENGTH;
        output_ptr     = weight_ptr + Q_LENGTH;

        // allocate 4KB in TCDM for Q1K1T and final result
        uint64_t *buffer_ptr = output_ptr + Q_LENGTH;

        // alias for output_ptr, also holding Q
        uint64_t *activation_ptr_i = output_ptr; 

        // stage 1:
        // load WK, K, Q to TCDM

        if (snrt_is_dm_core()) {
            // printf("DMA core is configured for K and WK\r\n");

            // measure the start of cycle count for preloading data to TCDM
            uint32_t start_dma_load = snrt_mcycle();

            // initialize TCDM with matrix K by DMA
            // printf("INITIALIZING TCDM\r\n");

            // read weight WK and ativation K from data.h
            size_t vector_size = Q_LENGTH * sizeof(uint64_t);

            snrt_dma_start_1d(activation_ptr,   K,  vector_size);
            snrt_dma_start_1d(activation_ptr_i, Q,  vector_size);
            snrt_dma_start_1d(weight_ptr,       WK, vector_size);

            snrt_dma_wait_all();

            // measures the end of the DMA transfer process
            // printf("DMA core exits after loading K and WK\r\n"); 
            uint32_t end_dma_load = snrt_mcycle();
            printf("DMTI %d\n", end_dma_load - start_dma_load);
        }

        /**************************************************************************/
        // wait for the DMA to finish loading WK and K to TCDM
        snrt_cluster_hw_barrier();
        /**************************************************************************/
        // stage 2: all three regions are occupied with WK, K, and Q
        // send WK from TCDM to DIMC
        /**************************************************************************/

        if (snrt_is_compute_core()){
            uint32_t core_config_start = snrt_mcycle();
            // printf("COMPUTE CORE is configured\r\n");

            // // configure the accelerator
            // printf("ENTERING MHA MODE\r\n");

            // uint32_t busy = dimc_query_busy();
            // printf("%d: busy\r\n", busy);
            // printf("QUERYING BUSY SUCCEEDED\r\n");

            configure_accelerator();

            // printf("CONFIGURING ACCELERATOR SUCCEEDED\r\n");

            // uint32_t read_zp_qkv = read_zp();
            // printf("%d: read_zp_qkv\r\n", read_zp_qkv);
            // printf("READING ZP SUCCEEDED\r\n");

            // send WK
            // printf("CONFIGURING STREAMERS for WK\r\n");
            dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
            dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(weight_ptr));
            dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(weight_ptr + 8));
            dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(weight_ptr + 16));
            dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(weight_ptr + 24));
            // printf("STREAMER CONFIGURED FOR WK\r\n");
            uint32_t core_config_end = snrt_mcycle();
            printf("CGTP %d\n", core_config_end - core_config_start);

            uint32_t core_start = snrt_mcycle();
            // configure the accelerator to start MHA computation
            dimc_start_mha();

            // start streamer data transfer
            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
            uint32_t core_end = snrt_mcycle();
            printf("CRTP %d\n", core_end - core_start);
        }

        /**************************************************************************/
        // wait for streamer to finish sending WK to DIMC
        snrt_cluster_hw_barrier();
        /**************************************************************************/
        // stage 3: weight_ptr is free, has K and Q in TCDM
        // load WQ to TCDM;
        // send K from TCDM to DIMC; kick start K1 generation;
        /**************************************************************************/

        if (snrt_is_dm_core()) {
            // printf("DMA core is configured for WQ\r\n");

            // read weight WQ from data.h
            uint32_t start_dma_load = snrt_mcycle();
            size_t vector_size = Q_LENGTH * sizeof(uint64_t);

            snrt_dma_start_1d(weight_ptr, WQ, vector_size);

            snrt_dma_wait_all();
            uint32_t end_dma_load = snrt_mcycle();
            // printf("DMTP %d\n", end_dma_load - start_dma_load);
        }

        if (snrt_is_compute_core()){
            // send K
            // printf("CONFIGURING STREAMERS for K\r\n");
            uint32_t core_config_start = snrt_mcycle();
            dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
            dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(activation_ptr));
            dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(activation_ptr + 8));
            dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(activation_ptr + 16));
            dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(activation_ptr + 24));
            // printf("STREAMER CONFIGURED FOR K\r\n");
            uint32_t core_config_end = snrt_mcycle();
            printf("CGTP %d\n", core_config_end - core_config_start);

            uint32_t core_start = snrt_mcycle();
            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }

            uint32_t core_end = snrt_mcycle();
            printf("CRTP %d\n", core_end - core_start);
        }

        /**************************************************************************/
        // wait for the DMA to finish loading WQ to TCDM and K1 generation
        snrt_cluster_hw_barrier();
        /**************************************************************************/
        // stage 4: activation_ptr is free, has WQ and Q in TCDM
        // streamer sends WQ to DIMC
        /**************************************************************************/

        if(snrt_is_compute_core()) {
            // send WQ
            // printf("CONFIGURING STREAMERS for WQ\r\n");
            uint32_t core_config_start = snrt_mcycle();
            dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
            dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(weight_ptr));
            dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(weight_ptr + 8));
            dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(weight_ptr + 16));
            dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(weight_ptr + 24));
            // printf("STREAMER CONFIGURED for WQ\r\n");
            uint32_t core_config_end = snrt_mcycle();
            printf("CGTP %d\n", core_config_end - core_config_start);

            uint32_t core_start = snrt_mcycle();
            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
            uint32_t core_end = snrt_mcycle();
            printf("CRTP %d\n", core_end - core_start);
        }

        /**************************************************************************/
        // wait for the streamer to finish sending WQ to DIMC
        snrt_cluster_hw_barrier();
        /**************************************************************************/
        // stage 5: weight_ptr and actication_ptr are free, has Q in TCDM
        // load WV and V to TCDM;
        // streamer sends Q to DIMC & kick start Q1K1T generation;
        /**************************************************************************/

        if (snrt_is_dm_core()) {
            // printf("DMA core is configured for WQ\r\n");

            // read weight WK and ativation K from data.h
            uint32_t start_dma_load = snrt_mcycle();
            size_t vector_size = Q_LENGTH * sizeof(uint64_t);

            snrt_dma_start_1d(activation_ptr, V, vector_size);
            snrt_dma_start_1d(weight_ptr, WV, vector_size);

            snrt_dma_wait_all();
            uint32_t end_dma_load = snrt_mcycle();
            // printf("DMTP %d\n", end_dma_load - start_dma_load);
        }

        if (snrt_is_compute_core()){
            // send Q
            // printf("CONFIGURING STREAMERS for Q\r\n");
            uint32_t core_config_start = snrt_mcycle();
            dimc_set_streamer_dim_w(64, 1, 64, 0, 8, (uint32_t)(buffer_ptr));
            dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(activation_ptr_i));
            dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(activation_ptr_i + 8));
            dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(activation_ptr_i + 16));
            dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(activation_ptr_i + 24));
            // printf("STREAMER CONFIGURED for Q\r\n");
            uint32_t core_config_end = snrt_mcycle();
            printf("CGTP %d\n", core_config_end - core_config_start);

            uint32_t core_start = snrt_mcycle();
            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
            uint32_t core_end = snrt_mcycle();
            printf("CRTP %d\n", core_end - core_start);
        }

        /**************************************************************************/
        // wait for the streamer to finish receiving Q1K1T
        snrt_cluster_hw_barrier();
        /**************************************************************************/
        // stage 6: activation_ptr_i is free, has V and WV in TCDM
        // send V to DIMC
        /**************************************************************************/

        if (snrt_is_compute_core()){
            // send V
            // printf("CONFIGURING STREAMERS for V\r\n");
            uint32_t core_config_start = snrt_mcycle();
            dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
            dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(activation_ptr));
            dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(activation_ptr + 8));
            dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(activation_ptr + 16));
            dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(activation_ptr + 24));
            // printf("STREAMER CONFIGURED for V\r\n");
            uint32_t core_config_end = snrt_mcycle();
            printf("CGTP %d\n", core_config_end - core_config_start);

            uint32_t core_start = snrt_mcycle();
            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
            uint32_t core_end = snrt_mcycle();
            printf("CRTP %d\n", core_end - core_start);
        }

        /**************************************************************************/
        // wait for the streamer to finish sending V to DIMC
        snrt_cluster_hw_barrier();
        /**************************************************************************/
        // stage 7: activation_ptr, activation_ptr_i are free, has WV in TCDM
        // send WV to DIMC
        /**************************************************************************/

        if (snrt_is_compute_core()) {
            // send WV
            // printf("CONFIGURING STREAMERS for WV\r\n");
            uint32_t core_config_start = snrt_mcycle();
            dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
            dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(weight_ptr));
            dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(weight_ptr + 8));
            dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(weight_ptr + 16));
            dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(weight_ptr + 24));
            // printf("STREAMER CONFIGURED for WV\r\n");
            uint32_t core_config_end = snrt_mcycle();
            printf("CGTP %d\n", core_config_end - core_config_start);

            uint32_t core_start = snrt_mcycle();
            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
            uint32_t core_end = snrt_mcycle();
            printf("CRTP %d\n", core_end - core_start);
        }

        /**************************************************************************/
        // wait for the streamer to finish sending WV to DIMC and V1 generation
        snrt_cluster_hw_barrier();
        /**************************************************************************/
        // stage 8: activation_ptr, activation_ptr_i, weight_ptr are free
        // has Q1K1T in TCDM
        // send Q1K1T to DIMC, saving final result in 
        /**************************************************************************/

        if (snrt_is_compute_core()) {
            // send Q1K1T
            // printf("CONFIGURING STREAMERS for Q1K1T\r\n");
            uint32_t core_config_start = snrt_mcycle();
            dimc_set_streamer_dim_w(64, 1, 64, 0, 8, (uint32_t)(output_ptr));
            dimc_set_streamer_dim_r0(16, 1, 256, 0, 8, (uint32_t)(buffer_ptr));
            dimc_set_streamer_dim_r1(16, 1, 256, 0, 8, (uint32_t)(buffer_ptr + 8));
            dimc_set_streamer_dim_r2(16, 1, 256, 0, 8, (uint32_t)(buffer_ptr + 16));
            dimc_set_streamer_dim_r3(16, 1, 256, 0, 8, (uint32_t)(buffer_ptr + 24));
            // printf("STREAMER CONFIGURED for Q1K1T\r\n");
            uint32_t core_config_end = snrt_mcycle();
            printf("CGTP %d\n", core_config_end - core_config_start);

            uint32_t core_start = snrt_mcycle();
            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
            uint32_t core_end = snrt_mcycle();
            printf("CRTP %d\n", core_end - core_start);

            printf("CHECK FINAL RESULT\r\n");

            // check the final result
            for (int i = 0; i < 512; ++i) {
                uint64_t value = output_ptr[i];

                uint64_t index = i * 8;

                // Split each uint64_t element into 8 uint8_t elements
                for (int j = 0; j < 8; ++j) {
                    uint8_t tmp_res = (uint8_t)((value >> (j * 8)) & 0xFF);
                    // printf("%d ", tmp_res);
                    if(tmp_res != gold[index + j]) {
                        printf("MISMATCH at %d, res:%d, gold:%d\r\n", (index + j), tmp_res, gold[index + j]);
                        err += 1;
                    }
                }
                // printf("RESULTS MATCH WITH GOLDEN MODEL\r\n");
            }
            
            // printf("MIMATCH COUNT %d\r\n", err);
        }
    }
    return err;
}   
