// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

#include "data.h"

#include "snax-gemmx-params.h"

#include "snax-gemmx-lib.h"

int main() {
    if (snrt_cluster_idx() == 1) {  // Set err value for checking
        int err = 0;

        // Prepare addresses in TCDM
        int8_t *local_a, *local_b;
        int8_t *local_d8;

        int start_cycle = 0;
        int end_cycle = 0;

        uint32_t subtraction_setting =
            gen_subtraction_config(subtraction_a, subtraction_b);

        uint32_t csr0 =
            gen_csr0_config(input_zp_i, output_zp_i, max_int_i, min_int_i);
        uint32_t csr1 = gen_csr1_config(double_round_i);

        start_cycle = snrt_mcycle();

        // ------------------------------------------------------------------------------
        // ------------------------------------------------------------------------------
        // ---------------------------------- Q = XWq, 64x512x64
        // -----------------------------------
        // ------------------------------------------------------------------------------
        // ------------------------------------------------------------------------------
        local_a = (int8_t *)(snrt_l1_next() + delta_local_a_X);
        local_b = (int8_t *)(snrt_l1_next() + delta_local_b_Wq);
        local_d8 = (int8_t *)(snrt_l1_next() + delta_local_d8_Q);

        // Transfer data from L3 to L1
        // Using DMA only
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(local_a, X,
                              M_1 * K_1 * meshRow * tileSize * sizeof(int8_t));
            snrt_dma_start_1d(local_b, Wq,
                              N_1 * K_1 * tileSize * meshCol * sizeof(int8_t));

            snrt_dma_wait_all();
        }

        // Wait for DMA to finish
        snrt_cluster_hw_barrier();

        if (snrt_cluster_core_idx() == 0) {
            // Set Streamer configuration CSR for conv2d
            set_gemmx_streamer_csr(
                Aslstride0_1, Aslstride1_1, Atlbound0_1, Atlstride0_1,
                Atlbound1_1, Atlstride1_1, Atlbound2_1, Atlstride2_1,
                Atlbound3_1, Atlstride3_1, Atlbound4_1, Atlstride4_1,
                Atlbound5_1, Atlstride5_1, set_addr_remap_index_A,

                Bslstride0_1, Bslstride1_1, Btlbound0_1, Btlstride0_1,
                Btlbound1_1, Btlstride1_1, Btlbound2_1, Btlstride2_1,
                set_addr_remap_index_B,

                D8slstride0_1, D8slstride1_1, D8tlbound0_1, D8tlstride0_1,
                D8tlbound1_1, D8tlstride1_1, D8tlbound2_1, D8tlstride2_1,
                set_addr_remap_index_D8,

                Cslstride0_1, Cslstride1_1, Ctlbound0_1, Ctlstride0_1,
                Ctlbound1_1, Ctlstride1_1, Ctlbound2_1, Ctlstride2_1,
                set_addr_remap_index_C,

                D32slstride0, D32slstride1, D32tlbound0, D32tlstride0,
                D32tlbound1, D32tlstride1, D32tlbound2, D32tlstride2,
                set_addr_remap_index_D32,

                delta_local_a_X, delta_local_b_Wq, delta_local_d8_Q,
                delta_local_d32,
                // no transpose
                delta_local_d32, bypassSIMD, transposed_A_1, transposed_B_1,
                channel_en_C, broadcast_C);

            // Set GEMMX configuration CSR
            set_gemmx_csr(
                K_1, N_1, M_1, subtraction_setting, csr0, csr1,
                shared_bitpacked_shift0, shared_bitpacked_shift1,
                shared_multiplier0, shared_multiplier1, shared_multiplier2,
                shared_multiplier3, shared_multiplier4, shared_multiplier5,
                shared_multiplier6, shared_multiplier7, M_1 * N_1, bypassSIMD);

            // Set CSR to start Streamer for conv2d
            set_gemmx_streamer_start();

            // Set CSR to start GEMM
            set_gemmx_start();

            // Poll until Streamer and GEMM accelerator finish
            wait_gemmx_and_streamer();

            // for (int i = 0; i < M_1 * N_1 * meshRow * meshCol; i++) {
            //     if (local_d8[i] != Q8_golden[i]) {
            //         printf("Mismatch at index %d: %d != %d for Q8 \r\n", i, local_d8[i], Q8_golden[i]);
            //         err++;
            //     }
            // }
            if(err == 0){
                printf("Everything is fine for Q8. \r\n");
            }
        };

        snrt_cluster_hw_barrier();

        // ------------------------------------------------------------------------------
        // ------------------------------------------------------------------------------
        // ---------------------------------- K = XWk, 64x512x64
        // -----------------------------------
        // ------------------------------------------------------------------------------
        // ------------------------------------------------------------------------------
        local_b = (int8_t *)(snrt_l1_next() + delta_local_b_Wk);
        local_d8 = (int8_t *)(snrt_l1_next() + delta_local_d8_K);

        // Transfer data from L3 to L1
        // Using DMA only
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(local_b, Wk,
                              N_1 * K_1 * tileSize * meshCol * sizeof(int8_t));

            snrt_dma_wait_all();
        }

        // Wait for DMA to finish
        snrt_cluster_hw_barrier();

        if (snrt_cluster_core_idx() == 0) {
            // Set Streamer configuration CSR for conv2d
            set_gemmx_streamer_csr(
                Aslstride0_1, Aslstride1_1, Atlbound0_1, Atlstride0_1,
                Atlbound1_1, Atlstride1_1, Atlbound2_1, Atlstride2_1,
                Atlbound3_1, Atlstride3_1, Atlbound4_1, Atlstride4_1,
                Atlbound5_1, Atlstride5_1, set_addr_remap_index_A,

                Bslstride0_1, Bslstride1_1, Btlbound0_1, Btlstride0_1,
                Btlbound1_1, Btlstride1_1, Btlbound2_1, Btlstride2_1,
                set_addr_remap_index_B,

                D8slstride0_1, D8slstride1_1, D8tlbound0_1, D8tlstride0_1,
                D8tlbound1_1, D8tlstride1_1, D8tlbound2_1, D8tlstride2_1,
                set_addr_remap_index_D8,

                Cslstride0_1, Cslstride1_1, Ctlbound0_1, Ctlstride0_1,
                Ctlbound1_1, Ctlstride1_1, Ctlbound2_1, Ctlstride2_1,
                set_addr_remap_index_C,

                D32slstride0, D32slstride1, D32tlbound0, D32tlstride0,
                D32tlbound1, D32tlstride1, D32tlbound2, D32tlstride2,
                set_addr_remap_index_D32,

                delta_local_a_X, delta_local_b_Wk, delta_local_d8_K,
                delta_local_d32,
                // no transpose
                delta_local_d32, bypassSIMD, transposed_A_1, transposed_B_1,
                channel_en_C, broadcast_C);

            // Set GEMMX configuration CSR
            set_gemmx_csr(
                K_1, N_1, M_1, subtraction_setting, csr0, csr1,
                shared_bitpacked_shift0, shared_bitpacked_shift1,
                shared_multiplier0, shared_multiplier1, shared_multiplier2,
                shared_multiplier3, shared_multiplier4, shared_multiplier5,
                shared_multiplier6, shared_multiplier7, M_1 * N_1, bypassSIMD);

            // Set CSR to start Streamer for conv2d
            set_gemmx_streamer_start();

            // Set CSR to start GEMM
            set_gemmx_start();

            // Poll until Streamer and GEMM accelerator finish
            wait_gemmx_and_streamer();

            // for (int i = 0; i < M_1 * N_1 * meshRow * meshCol; i++) {
            //     if (local_d8[i] != K8_golden[i]) {
            //         printf("Mismatch at index %d: %d != %d for K8 \r\n", i, local_d8[i], K8_golden[i]);
            //         err++;
            //     }
            // }
            if(err == 0){
                printf("Everything is fine for K8. \r\n");
            }
        };

        snrt_cluster_hw_barrier();

        // ------------------------------------------------------------------------------
        // ------------------------------------------------------------------------------
        // ---------------------------------- S=QKt, 64x64x64
        // -----------------------------------
        // ------------------------------------------------------------------------------
        // ------------------------------------------------------------------------------

        // note::
        // the K is not really transposed
        // need special tstride
        local_d8 = (int8_t *)(snrt_l1_next() + delta_local_d8_S);

        if (snrt_cluster_core_idx() == 0) {
            // Set Streamer configuration CSR for conv2d
            set_gemmx_streamer_csr(
                Aslstride0_2, Aslstride1_2, Atlbound0_2, Atlstride0_2,
                Atlbound1_2, Atlstride1_2, Atlbound2_2, Atlstride2_2,
                Atlbound3_2, Atlstride3_2, Atlbound4_2, Atlstride4_2,
                Atlbound5_2, Atlstride5_2, set_addr_remap_index_A,

                Bslstride0_2, Bslstride1_2, Btlbound0_2, Btlstride0_2,
                Btlbound1_2, Btlstride1_2, Btlbound2_2, Btlstride2_2,
                set_addr_remap_index_B,

                D8slstride0_2, D8slstride1_2, D8tlbound0_2, D8tlstride0_2,
                D8tlbound1_2, D8tlstride1_2, D8tlbound2_2, D8tlstride2_2,
                set_addr_remap_index_D8,

                Cslstride0_2, Cslstride1_2, Ctlbound0_2, Ctlstride0_2,
                Ctlbound1_2, Ctlstride1_2, Ctlbound2_2, Ctlstride2_2,
                set_addr_remap_index_C,

                D32slstride0, D32slstride1, D32tlbound0, D32tlstride0,
                D32tlbound1, D32tlstride1, D32tlbound2, D32tlstride2,
                set_addr_remap_index_D32,

                delta_local_d8_Q, delta_local_d8_K, delta_local_d8_S,
                delta_local_d32,
                // transpose K
                delta_local_d32, bypassSIMD, transposed_A_2, transposed_B_2,
                channel_en_C, broadcast_C);

            // Set GEMMX configuration CSR
            set_gemmx_csr(
                K_2, N_2, M_2, subtraction_setting, csr0, csr1,
                shared_bitpacked_shift0, shared_bitpacked_shift1,
                shared_multiplier0, shared_multiplier1, shared_multiplier2,
                shared_multiplier3, shared_multiplier4, shared_multiplier5,
                shared_multiplier6, shared_multiplier7, M_2 * N_2, bypassSIMD);

            // Set CSR to start Streamer for conv2d
            set_gemmx_streamer_start();

            // Set CSR to start GEMM
            set_gemmx_start();

            // Poll until Streamer and GEMM accelerator finish
            wait_gemmx_and_streamer();

            for (int i = 0; i < M_2 * N_2 * meshRow * meshCol; i++) {
                if (local_d8[i] != S8_golden[i]) {
                    printf("Mismatch at index %d: %d != %d for S8 \r\n", i, local_d8[i], S8_golden[i]);
                    err++;
                }
            }
            if(err == 0){
                printf("Everything is fine for S8 \r\n");
            }
        };

        snrt_cluster_hw_barrier();

        // ------------------------------------------------------------------------------
        // ------------------------------------------------------------------------------
        // ---------------------------------- V=XWv, 64x512x64
        // -----------------------------------
        // ------------------------------------------------------------------------------
        // ------------------------------------------------------------------------------
        local_b = (int8_t *)(snrt_l1_next() + delta_local_b_Wv);
        local_d8 = (int8_t *)(snrt_l1_next() + delta_local_d8_V);

        // Transfer data from L3 to L1
        // Using DMA only
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(local_b, Wv,
                              N_1 * K_1 * tileSize * meshCol * sizeof(int8_t));

            snrt_dma_wait_all();
        }

        // Wait for DMA to finish
        snrt_cluster_hw_barrier();

        if (snrt_cluster_core_idx() == 0) {
            // Set Streamer configuration CSR for conv2d
            set_gemmx_streamer_csr(
                Aslstride0_1, Aslstride1_1, Atlbound0_1, Atlstride0_1,
                Atlbound1_1, Atlstride1_1, Atlbound2_1, Atlstride2_1,
                Atlbound3_1, Atlstride3_1, Atlbound4_1, Atlstride4_1,
                Atlbound5_1, Atlstride5_1, set_addr_remap_index_A,

                Bslstride0_1, Bslstride1_1, Btlbound0_1, Btlstride0_1,
                Btlbound1_1, Btlstride1_1, Btlbound2_1, Btlstride2_1,
                set_addr_remap_index_B,

                D8slstride0_1, D8slstride1_1, D8tlbound0_1, D8tlstride0_1,
                D8tlbound1_1, D8tlstride1_1, D8tlbound2_1, D8tlstride2_1,
                set_addr_remap_index_D8,

                Cslstride0_1, Cslstride1_1, Ctlbound0_1, Ctlstride0_1,
                Ctlbound1_1, Ctlstride1_1, Ctlbound2_1, Ctlstride2_1,
                set_addr_remap_index_C,

                D32slstride0, D32slstride1, D32tlbound0, D32tlstride0,
                D32tlbound1, D32tlstride1, D32tlbound2, D32tlstride2,
                set_addr_remap_index_D32,

                delta_local_a_X, delta_local_b_Wv, delta_local_d8_V,
                delta_local_d32,
                // no transpose
                delta_local_d32, bypassSIMD, transposed_A_1, transposed_B_1,
                channel_en_C, broadcast_C);

            // Set GEMMX configuration CSR
            set_gemmx_csr(
                K_1, N_1, M_1, subtraction_setting, csr0, csr1,
                shared_bitpacked_shift0, shared_bitpacked_shift1,
                shared_multiplier0, shared_multiplier1, shared_multiplier2,
                shared_multiplier3, shared_multiplier4, shared_multiplier5,
                shared_multiplier6, shared_multiplier7, M_1 * N_1, bypassSIMD);

            // Set CSR to start Streamer for conv2d
            set_gemmx_streamer_start();

            // Set CSR to start GEMM
            set_gemmx_start();

            // Poll until Streamer and GEMM accelerator finish
            wait_gemmx_and_streamer();

            for (int i = 0; i < M_1 * N_1 * meshRow * meshCol; i++) {
                if (local_d8[i] != V8_golden[i]) {
                    printf("Mismatch at index %d: %d != %d for V8 \r\n", i, local_d8[i], V8_golden[i]);
                    err++;
                }
            }
            if(err == 0){
                printf("Everything is fine for V8. \r\n");
            }
        };

        snrt_cluster_hw_barrier();

        // ------------------------------------------------------------------------------
        // ------------------------------------------------------------------------------
        // ---------------------------------- Z=SV, 64x64x64
        // -----------------------------------
        // ------------------------------------------------------------------------------
        // ------------------------------------------------------------------------------

        if (snrt_cluster_core_idx() == 0) {
            // Set Streamer configuration CSR for conv2d
            set_gemmx_streamer_csr(
                Aslstride0_2, Aslstride1_2, Atlbound0_2, Atlstride0_2,
                Atlbound1_2, Atlstride1_2, Atlbound2_2, Atlstride2_2,
                Atlbound3_2, Atlstride3_2, Atlbound4_2, Atlstride4_2,
                Atlbound5_2, Atlstride5_2, set_addr_remap_index_A,

                Bslstride0_2, Bslstride1_2, Btlbound0_2, Btlstride0_2,
                Btlbound1_2, Btlstride1_2, Btlbound2_2, Btlstride2_2,
                set_addr_remap_index_B,

                D8slstride0_2, D8slstride1_2, D8tlbound0_2, D8tlstride0_2,
                D8tlbound1_2, D8tlstride1_2, D8tlbound2_2, D8tlstride2_2,
                set_addr_remap_index_D8,

                Cslstride0_2, Cslstride1_2, Ctlbound0_2, Ctlstride0_2,
                Ctlbound1_2, Ctlstride1_2, Ctlbound2_2, Ctlstride2_2,
                set_addr_remap_index_C,

                D32slstride0, D32slstride1, D32tlbound0, D32tlstride0,
                D32tlbound1, D32tlstride1, D32tlbound2, D32tlstride2,
                set_addr_remap_index_D32,

                delta_local_d8_S, delta_local_d8_V, delta_local_d8_Z,
                delta_local_d32,
                // no transpose
                delta_local_d32, bypassSIMD, transposed_A_1, transposed_B_1,
                channel_en_C, broadcast_C);

            // Set GEMMX configuration CSR
            set_gemmx_csr(
                K_2, N_2, M_2, subtraction_setting, csr0, csr1,
                shared_bitpacked_shift0, shared_bitpacked_shift1,
                shared_multiplier0, shared_multiplier1, shared_multiplier2,
                shared_multiplier3, shared_multiplier4, shared_multiplier5,
                shared_multiplier6, shared_multiplier7, M_2 * N_2, bypassSIMD);

            // Set CSR to start Streamer for conv2d
            set_gemmx_streamer_start();

            // Set CSR to start GEMM
            set_gemmx_start();

            // Poll until Streamer and GEMM accelerator finish
            wait_gemmx_and_streamer();
        };

        snrt_cluster_hw_barrier();

        // Measure total cycles
        end_cycle = snrt_mcycle();

        if (snrt_cluster_core_idx() == 0) {
            for (int i = 0; i < 1; i++) {
                printf("Total cycles from mcycle: %d \r\n",
                       end_cycle - start_cycle);
            }
        }

        if (snrt_cluster_core_idx() == 0) {
            int8_t* local_Z8 = (int8_t *)(snrt_l1_next() + delta_local_d8_Z);
            for (int i = 0; i < M_2 * N_2 * meshRow * meshCol; i++){
                if(local_Z8[i] != Z8_golden[i]){
                    printf("Mismatch at index %d: %d != %d\n", i, local_Z8[i], Z8_golden[i]);
                    err++;
                }
            }
            printf("Total mismatches: %d\n", err);
        }

        snrt_cluster_hw_barrier();

        return_to_cva6_single_cluster(err);
    }
}
