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
// #include "data_mha.h"

#include "data.h"
#include "snax-gemmx-params.h"
#include "snax-gemmx-lib.h"

#include "data_gemmx.h"

uint64_t * tcdm_1_start_addr; // starting address for GEMMX cluster
uint64_t * tcdm_3_start_addr; // starting address for DIMC cluster

int main() {

    // set error value for checking
    int err = 0;

    // allocate 32+32+32KB in TCDM for activation and weight pair
    uint64_t *dimc_tcdm_ptr_0, *dimc_tcdm_ptr_1, *dimc_tcdm_ptr_2, *dimc_tcdm_ptr_danger;
    
    if (snrt_cluster_idx() == 1){
        // allocate memory for GEMMX cluster
        tcdm_1_start_addr = (uint64_t *)snrt_cluster_base_addrl();
    }
    
    snrt_global_barrier();

    if (snrt_cluster_idx() == 3){

        dimc_tcdm_ptr_0 = (uint64_t *)snrt_cluster_base_addrl();
        dimc_tcdm_ptr_1     = dimc_tcdm_ptr_0 + Q_LENGTH;
        dimc_tcdm_ptr_2     = dimc_tcdm_ptr_1 + Q_LENGTH;
        // must not fully use the 32KB space
        dimc_tcdm_ptr_danger     = dimc_tcdm_ptr_2 + Q_LENGTH;

        tcdm_3_start_addr = dimc_tcdm_ptr_0;

        if (snrt_is_dm_core()) {
            // read weight WK and ativation K from data.h
            size_t vector_size = Q_LENGTH * sizeof(uint64_t);
            snrt_dma_start_1d(dimc_tcdm_ptr_0, K,  vector_size);
            snrt_dma_start_1d(dimc_tcdm_ptr_1, Q,  vector_size);
            snrt_dma_start_1d(dimc_tcdm_ptr_2, WK, vector_size);

            snrt_dma_wait_all();
        }

        snrt_cluster_hw_barrier();

        if (snrt_is_compute_core()) {
            uint32_t busy = dimc_query_busy();
            printf("%d: busy\n", busy);
            printf("QUERYING BUSY SUCCEEDED\n");

            configure_accelerator();

            // send WK
            printf("CONFIGURING STREAMERS for WK\n");
            dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
            dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_2));
            dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_2 + 8));
            dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_2 + 16));
            dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_2 + 24));
            
            dimc_start_mha();

            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
            printf("STREAMER Finished for WK\n");
        }
        
        snrt_cluster_hw_barrier();

        // load weight WQ from data.h to tcdm
        if (snrt_is_dm_core()) {
            size_t vector_size = Q_LENGTH * sizeof(uint64_t);

            snrt_dma_start_1d(dimc_tcdm_ptr_2, WQ, vector_size);

            snrt_dma_wait_all();
        }

        if (snrt_is_compute_core()) {
            // send K
            printf("CONFIGURING STREAMERS for K\n");
            dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
            dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_0));
            dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_0 + 8));
            dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_0 + 16));
            dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_0 + 24));

            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
            printf("STREAMER Finished for K\n");
        }
        
        snrt_cluster_hw_barrier();
        
        if (snrt_cluster_core_idx == 0 ) {
            printf("CONFIGURING STREAMERS for WQ\n");
            dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
            dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_2));
            dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_2 + 8));
            dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_2 + 16));
            dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_2 + 24));

            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
            printf("STREAMER Finished for WQ\n");
        }

        snrt_cluster_hw_barrier();

        if (snrt_is_dm_core()) {
            
            // load WV and V from data.h to tcdm
            size_t vector_size = Q_LENGTH * sizeof(uint64_t);

            snrt_dma_start_1d(dimc_tcdm_ptr_0, V,  vector_size);
            snrt_dma_start_1d(dimc_tcdm_ptr_2, WV, vector_size);

            snrt_dma_wait_all();
        }
        
        snrt_cluster_hw_barrier();

        if (snrt_is_compute_core()) {
            // send Q
            printf("CONFIGURING STREAMERS for Q\n");
            dimc_set_streamer_dim_w(64, 1, 64, 0, 8, (uint32_t)(dimc_tcdm_ptr_danger));
            dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_1));
            dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_1 + 8));
            dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_1 + 16));
            dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_1 + 24));

            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
            printf("STREAMER Finished for Q\n");
        }
        
        snrt_cluster_hw_barrier();

        if (snrt_is_compute_core()) {
            // send V
            printf("CONFIGURING STREAMERS for V\n");
            dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
            dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_0));
            dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_0 + 8));
            dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_0 + 16));
            dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_0 + 24));

            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
            printf("STREAMER Finished for V\n");
        }

        snrt_cluster_hw_barrier();

        if (snrt_is_compute_core()) {
            // send WV
            printf("CONFIGURING STREAMERS for WV\r\n");
            dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
            dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_2));
            dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_2 + 8));
            dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_2 + 16));
            dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_2 + 24));

            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
            printf("STREAMER sended WV\r\n");
        }

        snrt_cluster_hw_barrier();

        if (snrt_is_compute_core()) {
            // send Q1K1T
            printf("CONFIGURING STREAMERS for Q1K1T\n");
            dimc_set_streamer_dim_w(64, 1, 64, 0, 8, (uint32_t)(tcdm_1_start_addr));
            dimc_set_streamer_dim_r0(16, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_danger));
            dimc_set_streamer_dim_r1(16, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_danger + 8));
            dimc_set_streamer_dim_r2(16, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_danger + 16));
            dimc_set_streamer_dim_r3(16, 1, 256, 0, 8, (uint32_t)(dimc_tcdm_ptr_danger + 24));

            dimc_start_streamer();

            while (dimc_is_streamer_busy()) { }
            printf("STREAMER Finished for Q1K1T\n");
        }
    }

    snrt_global_barrier();

    if (snrt_cluster_idx() == 1){

        if (snrt_is_compute_core()) {
            printf("GEMMX cluster starts to configure GEMMX\n");
            // Set Streamer configuration CSR for conv2d
            set_gemmx_streamer_csr(
                Aslstride0, Aslstride1, Atlbound0, Atlstride0, Atlbound1,
                Atlstride1, Atlbound2, Atlstride2, Atlbound3, Atlstride3,
                Atlbound4, Atlstride4, Atlbound5, Atlstride5,
                set_addr_remap_index_A,

                Bslstride0, Bslstride1, Btlbound0, Btlstride0, Btlbound1,
                Btlstride1, Btlbound2, Btlstride2, set_addr_remap_index_B,

                D8slstride0, D8slstride1, D8tlbound0, D8tlstride0, D8tlbound1,
                D8tlstride1, D8tlbound2, D8tlstride2, set_addr_remap_index_D8,

                Cslstride0, Cslstride1, Ctlbound0, Ctlstride0, Ctlbound1,
                Ctlstride1, Ctlbound2, Ctlstride2, set_addr_remap_index_C,

                D32slstride0, D32slstride1, D32tlbound0, D32tlstride0,
                D32tlbound1, D32tlstride1, D32tlbound2, D32tlstride2,
                set_addr_remap_index_D32,

                delta_local_a, delta_local_b, delta_local_d8, delta_local_c,
                delta_local_d32, bypassSIMD, transposed_A, transposed_B,
                channel_en_C, broadcast_C);

            // Set GEMMX configuration CSR
            uint32_t subtraction_setting =
                gen_subtraction_config(subtraction_a, subtraction_b);

            uint32_t csr0 =
                gen_csr0_config(input_zp_i, output_zp_i, max_int_i, min_int_i);
            uint32_t csr1 = gen_csr1_config(double_round_i);

            set_gemmx_csr(
                gemmx_K, gemmx_N, gemmx_M, subtraction_setting, csr0, csr1,
                shared_bitpacked_shift0, shared_bitpacked_shift1,
                shared_multiplier0, shared_multiplier1, shared_multiplier2,
                shared_multiplier3, shared_multiplier4, shared_multiplier5,
                shared_multiplier6, shared_multiplier7, gemmx_M * gemmx_N, bypassSIMD);

            // Set CSR to start Streamer for conv2d
            set_gemmx_streamer_start();

            // Set CSR to start GEMM
            set_gemmx_start();

            // Poll until Streamer and GEMM accelerator finish
            wait_gemmx_and_streamer();
            printf("GEMMX cluster finished the computation\n");

            // check the final result
            printf("GEMMX cluster starts to check copied data\n");

            for (int i = 0; i < 512; ++i) {
                uint64_t value = dimc_tcdm_ptr_0[i];

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
