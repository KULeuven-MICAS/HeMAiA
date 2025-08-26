// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

#include "data.h"

#include "snax-gemmx-params.h"

#include "snax-gemmx-lib.h"

int main() {
    if (snrt_cluster_idx() == 1) {
        // Set err value for checking
        int err = 0;

        // Prepare addresses in TCDM
        int8_t *local_a, *local_b;
        int32_t *local_c, *local_d32;
        int8_t *local_d8;

        // Allocate space in TCDM
        local_a = (int8_t *)(snrt_l1_next() + delta_local_a);
        local_b = (int8_t *)(snrt_l1_next() + delta_local_b);
        local_c = (int32_t *)(snrt_l1_next() + delta_local_c);
        local_d32 = (int32_t *)(snrt_l1_next() + delta_local_d32);
        local_d8 = (int8_t *)(snrt_l1_next() + delta_local_d8);
        int start_cycle = 0;
        int end_cycle = 0;
        int performance_counter = 0;
        //------------------------------------------------------------------------
        //------------------------------------------------------------------------
        // first matmul
        // 8x26x50
        //------------------------------------------------------------------------
        //------------------------------------------------------------------------
        if (snrt_is_dm_core()) {
                        snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY,
                                    snrt_hartid());
            snrt_dma_start_1d(local_a, A,
                              M * K * meshRow * tileSize * sizeof(int8_t));
            snrt_dma_start_1d(local_b, B,
                              N * K * tileSize * meshCol * sizeof(int8_t));

            snrt_dma_wait_all();
            printf(
                "[MATMUL-1]-DMA transfer cycle from DMA hardware counter %d  for A "
                "and B\r\n",
                snrt_get_perf_counter(SNRT_PERF_CNT0));
            snrt_reset_perf_counter(SNRT_PERF_CNT0);
        }

        snrt_cluster_hw_barrier();

        if (snrt_cluster_core_idx() == 0) {
            // Set GEMMX configuration CSR
            uint32_t subtraction_setting =
                gen_subtraction_config(subtraction_a, subtraction_b);

            uint32_t csr0 =
                gen_csr0_config(input_zp_i, output_zp_i, max_int_i, min_int_i);
            uint32_t csr1 = gen_csr1_config(double_round_i);
start_cycle = snrt_mcycle();
            set_gemmx_csr(
                K, N, M, subtraction_setting, csr0, csr1,
                shared_bitpacked_shift0, shared_bitpacked_shift1,
                shared_multiplier0, shared_multiplier1, shared_multiplier2,
                shared_multiplier3, shared_multiplier4, shared_multiplier5,
                shared_multiplier6, shared_multiplier7, M * N, bypassSIMD);

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
            end_cycle = snrt_mcycle();
            printf("[MATMUL-1]-Configuration cycles: %d \r\n",
                   end_cycle - start_cycle);
            write_csr_obs(0x00b);

            // printf("CSR configuration finished \r\n");

            // Set CSR to start GEMM
            set_gemmx_start();

            // Set CSR to start Streamer for conv2d
            set_gemmx_streamer_start();
            // printf("Streamer and GeMM started \r\n");
            write_csr_obs(0x00d);

            // Poll until Streamer and GEMM accelerator finish
            wait_gemmx_and_streamer();
            write_csr_obs(0x00c);

            performance_counter = read_gemmx_streamer_perf_counter();

            printf("[MATMUL-1]-GeMM performance counter: %d \r\n",
                   performance_counter);
        };

        snrt_cluster_hw_barrier();

        //------------------------------------------------------------------------
        //------------------------------------------------------------------------
        // first matmul
        // 8x26x50
        //------------------------------------------------------------------------
        //------------------------------------------------------------------------

        // Prepare addresses in TCDM
        int8_t *local_a_2nd_layer, *local_b_2nd_layer;
        int32_t *local_c_2nd_layer, *local_d32_2nd_layer;
        int8_t *local_d8_2nd_layer;

        // Allocate space in TCDM
        local_a_2nd_layer = (int8_t *)(snrt_l1_next() + delta_local_a_2nd_layer);
        local_b_2nd_layer = (int8_t *)(snrt_l1_next() + delta_local_b_2nd_layer);
        local_c_2nd_layer = (int32_t *)(snrt_l1_next() + delta_local_c_2nd_layer);
        local_d32_2nd_layer = (int32_t *)(snrt_l1_next() + delta_local_d32_2nd_layer);
        local_d8_2nd_layer = (int8_t *)(snrt_l1_next() + delta_local_d8_2nd_layer);

        // Transfer data from L3 to L1
        // Using DMA only
        if (snrt_is_dm_core()) {
            snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY,
                                    snrt_hartid());
            snrt_dma_start_1d(
                local_b_2nd_layer, B_2nd_layer,
                N_2nd_layer * K_2nd_layer * tileSize * meshCol * sizeof(int8_t));

            snrt_dma_wait_all();
            printf(
                "[FC]-DMA transfer cycle from DMA hardware counter %d  for "
                "B\r\n",
                snrt_get_perf_counter(SNRT_PERF_CNT0));
            snrt_reset_perf_counter(SNRT_PERF_CNT0);
        }

        // Wait for DMA to finish
        snrt_cluster_hw_barrier();

        if (snrt_cluster_core_idx() == 0) {
            // Set GEMMX configuration CSR
            uint32_t subtraction_setting_2nd_layer =
                gen_subtraction_config(subtraction_a_2nd_layer, subtraction_b_2nd_layer);

            uint32_t csr0_2nd_layer = gen_csr0_config(input_zp_i_2nd_layer, output_zp_i_2nd_layer,
                                               max_int_i_2nd_layer, min_int_i_2nd_layer);
            uint32_t csr1_2nd_layer = gen_csr1_config(double_round_i_2nd_layer);

            start_cycle = snrt_mcycle();
            set_gemmx_csr(K_2nd_layer, N_2nd_layer, M_2nd_layer, subtraction_setting_2nd_layer, csr0_2nd_layer,
                          csr1_2nd_layer, shared_bitpacked_shift0_2nd_layer,
                          shared_bitpacked_shift1_2nd_layer, shared_multiplier0_2nd_layer,
                          shared_multiplier1_2nd_layer, shared_multiplier2_2nd_layer,
                          shared_multiplier3_2nd_layer, shared_multiplier4_2nd_layer,
                          shared_multiplier5_2nd_layer, shared_multiplier6_2nd_layer,
                          shared_multiplier7_2nd_layer, M_2nd_layer * N_2nd_layer, bypassSIMD_2nd_layer);

            // Set Streamer configuration CSR for conv2d
            set_gemmx_streamer_csr(
                Aslstride0_2nd_layer, Aslstride1_2nd_layer, Atlbound0_2nd_layer, Atlstride0_2nd_layer,
                Atlbound1_2nd_layer, Atlstride1_2nd_layer, Atlbound2_2nd_layer, Atlstride2_2nd_layer,
                Atlbound3_2nd_layer, Atlstride3_2nd_layer, Atlbound4_2nd_layer, Atlstride4_2nd_layer,
                Atlbound5_2nd_layer, Atlstride5_2nd_layer, set_addr_remap_index_A_2nd_layer,

                Bslstride0_2nd_layer, Bslstride1_2nd_layer, Btlbound0_2nd_layer, Btlstride0_2nd_layer,
                Btlbound1_2nd_layer, Btlstride1_2nd_layer, Btlbound2_2nd_layer, Btlstride2_2nd_layer,
                set_addr_remap_index_B_2nd_layer,

                D8slstride0_2nd_layer, D8slstride1_2nd_layer, D8tlbound0_2nd_layer, D8tlstride0_2nd_layer,
                D8tlbound1_2nd_layer, D8tlstride1_2nd_layer, D8tlbound2_2nd_layer, D8tlstride2_2nd_layer,
                set_addr_remap_index_D8_2nd_layer,

                Cslstride0_2nd_layer, Cslstride1_2nd_layer, Ctlbound0_2nd_layer, Ctlstride0_2nd_layer,
                Ctlbound1_2nd_layer, Ctlstride1_2nd_layer, Ctlbound2_2nd_layer, Ctlstride2_2nd_layer,
                set_addr_remap_index_C_2nd_layer,

                D32slstride0_2nd_layer, D32slstride1_2nd_layer, D32tlbound0_2nd_layer,
                D32tlstride0_2nd_layer, D32tlbound1_2nd_layer, D32tlstride1_2nd_layer,
                D32tlbound2_2nd_layer, D32tlstride2_2nd_layer, set_addr_remap_index_D32_2nd_layer,

                delta_local_a_2nd_layer, delta_local_b_2nd_layer, delta_local_d8_2nd_layer,
                delta_local_c_2nd_layer, delta_local_d32_2nd_layer, bypassSIMD_2nd_layer,
                transposed_A_2nd_layer, transposed_B_2nd_layer, channel_en_C_2nd_layer,
                broadcast_C_2nd_layer);

            end_cycle = snrt_mcycle();

            printf("[FC]-CSR configuration cycle from mcycle: %d \r\n",
                   end_cycle - start_cycle);

            write_csr_obs(0x00b);

            // printf("CSR configuration finished \r\n");
            // Set CSR to start GEMM
            set_gemmx_start();

            // Set CSR to start Streamer for conv2d
            set_gemmx_streamer_start();
            // printf("Streamer and GeMM started \r\n");
            write_csr_obs(0x00d);

            // Poll until Streamer and GEMM accelerator finish
            wait_gemmx_and_streamer();

            performance_counter = read_gemmx_streamer_perf_counter();
            printf("[FC]-Streamer performance counter: %d \r\n",
                   performance_counter);

            write_csr_obs(0x00c);

        };

        snrt_cluster_hw_barrier();

        return_to_cva6_single_cluster(err);

    }
}
