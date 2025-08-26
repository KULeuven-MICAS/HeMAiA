// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

#include "data.h"

#include "snax-gemmx-params.h"

#include "snax-gemmx-lib.h"

#include "snax-xdma-lib.h"

int main() {
    if (snrt_cluster_idx() == 1) {  // Set err value for checking
        int err = 0;

        // ---------------------------------------------------------------
        // ---------------------------------------------------------------
        // ---------------------------------------------------------------
        // Conv2D
        // ---------------------------------------------------------------
        // ---------------------------------------------------------------
        // ---------------------------------------------------------------
        // Prepare addresses pointers in TCDM for DMA
        int8_t *local_a_dma, *local_b_dma;
        int32_t *local_c_dma, *local_d32_dma;
        int8_t *local_d8_dma;

        // Allocate space in TCDM for DMA
        local_a_dma = (int8_t *)(snrt_l1_next() + delta_physical_a);
        local_b_dma = (int8_t *)(snrt_l1_next() + delta_physical_b);
        local_c_dma = (int32_t *)(snrt_l1_next() + delta_physical_c);
        local_d32_dma = (int32_t *)(snrt_l1_next() + delta_physical_d32);
        local_d8_dma = (int8_t *)(snrt_l1_next() + delta_physical_d8);

        // Prepare addresses pointers in TCDM for streamer
        int8_t *local_a, *local_b;
        int32_t *local_c, *local_d32;
        int8_t *local_d8;

        // Allocate space in TCDM for streamer
        local_a = (int8_t *)(snrt_l1_next() + delta_local_a);
        local_b = (int8_t *)(snrt_l1_next() + delta_local_b);
        local_c = (int32_t *)(snrt_l1_next() + delta_local_c);
        local_d32 = (int32_t *)(snrt_l1_next() + delta_local_d32);
        local_d8 = (int8_t *)(snrt_l1_next() + delta_local_d8);

        int start_cycle = 0;
        int end_cycle = 0;
        int performance_counter = 0;

        uint32_t subtraction_setting =
            gen_subtraction_config(subtraction_a, subtraction_b);

        uint32_t csr0 =
            gen_csr0_config(input_zp_i, output_zp_i, max_int_i, min_int_i);
        uint32_t csr1 = gen_csr1_config(double_round_i);

        // Transfer data from L3 to L1
        // Using DMA only
        if (snrt_is_dm_core()) {
            snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY,
                                    snrt_hartid());
            if (interleaved_address == 1) {
                start_cycle = snrt_mcycle();
                snrt_dma_start_1d(local_a, A,
                                  Nbatch * (H + 2 * pad_h) * (W + 2 * pad_w) *
                                      Cin * sizeof(int8_t));
                snrt_dma_start_1d(local_b, B,
                                  Cout * Kh * Kw * Cin * sizeof(int8_t));
            } else {
                start_cycle = snrt_mcycle();
                snrt_dma_start_2d(
                    local_a_dma, A, 64 * sizeof(int8_t), 256, 64,
                    Nbatch * (H + 2 * pad_h) * (W + 2 * pad_w) * Cin / 64);
                snrt_dma_start_2d(local_b_dma, B, 64 * sizeof(int8_t), 256, 64,
                                  Cout * Kh * Kw * Cin / 64);
            }
            snrt_dma_wait_all();
            end_cycle = snrt_mcycle();
            printf(
                "[CONV]-DMA transfer cycle from DMA hardware counter %d  for A "
                "and B\r\n",
                snrt_get_perf_counter(SNRT_PERF_CNT0));
            snrt_reset_perf_counter(SNRT_PERF_CNT0);
            printf(
                "[CONV]-DMA transfer cycle from mcycle: %d cycles for A and B "
                "\r\n",
                end_cycle - start_cycle);
        }

        // Wait for DMA to finish
        snrt_cluster_hw_barrier();
        if (snrt_is_dm_core()) {
            snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY,
                                    snrt_hartid());
            if (interleaved_address == 1) {
                start_cycle = snrt_mcycle();
                snrt_dma_start_1d(local_c, C,
                                  M * N * meshRow * meshCol * sizeof(int32_t));
            } else {
                start_cycle = snrt_mcycle();
                snrt_dma_start_2d(local_c_dma, C, 16 * sizeof(int32_t), 256,
                                  16 * sizeof(int32_t),
                                  M * N * meshRow * meshCol / 16);
            }
            snrt_dma_wait_all();
            end_cycle = snrt_mcycle();
            printf(
                "[CONV]-DMA transfer cycle from DMA hardware counter %d \r\n",
                snrt_get_perf_counter(SNRT_PERF_CNT0));
            snrt_reset_perf_counter(SNRT_PERF_CNT0);
            printf(
                "[CONV]-DMA transfer cycle from mcycle: %d cycles for C \r\n",
                end_cycle - start_cycle);
        }

        snrt_cluster_hw_barrier();

        if (snrt_cluster_core_idx() == 0) {
            // Set Streamer configuration CSR for conv2d
            start_cycle = snrt_mcycle();
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
            set_gemmx_csr(
                K, N, M, subtraction_setting, csr0, csr1,
                shared_bitpacked_shift0, shared_bitpacked_shift1,
                shared_multiplier0, shared_multiplier1, shared_multiplier2,
                shared_multiplier3, shared_multiplier4, shared_multiplier5,
                shared_multiplier6, shared_multiplier7, M * N, bypassSIMD);
            end_cycle = snrt_mcycle();
            printf("[CONV]-Configuration cycles: %d \r\n",
                   end_cycle - start_cycle);

            // Set CSR to start Streamer for conv2d
            start_cycle = snrt_mcycle();
            set_gemmx_streamer_start();

            // Set CSR to start GEMM
            set_gemmx_start();

            // Poll until Streamer and GEMM accelerator finish
            wait_gemmx_and_streamer();
            end_cycle = snrt_mcycle();

            printf("[CONV]-Compute total cycles from mcycle: %d \r\n",
                   end_cycle - start_cycle);

            performance_counter = read_gemmx_streamer_perf_counter();

            printf("[CONV]-GeMM performance counter: %d \r\n",
                   performance_counter);

            int if_inifinit_loop = 0;
            while (0) {
                // Set CSR to start Streamer for conv2d
                set_gemmx_streamer_start();

                // Set CSR to start GEMM
                set_gemmx_start();

                // Poll until Streamer and GEMM accelerator finish
                wait_gemmx_and_streamer();
            }

            // check the result of the implicit im2col convolution
            // if (interleaved_address == 1) {
            //     if (!bypassSIMD) {
            //         err +=
            //             check_gemmx_result_D8(local_d8, D8, Batch, M, N,
            //             false);
            //     } else {
            //         err += check_gemmx_result_D32(local_d32, D32, Batch, M,
            //         N,
            //                                       false);
            //     }
            // } else {
            //     if (!bypassSIMD) {
            //         err += check_gemmx_result_D8(local_d8_dma, D8, Batch, M,
            //         N,
            //                                      true);
            //     } else {
            //         err += check_gemmx_result_D32(local_d32_dma, D32, Batch,
            //         M,
            //                                       N, true);
            //     }
            // }

            // printf("SNAX GEMM Conv2d: %s, Error: %d . bypassSIMD = %d .\r\n",
            //        err ? "FAIL" : "PASS", err, bypassSIMD);
            // if (err == 0) {
            //     write_csr_obs(0x00f);
            //     printf("P \r\n");
            // } else {
            //     write_csr_obs(0x00e);
            //     printf("F: %d \r\n", err);
            // }
        };
        snrt_cluster_hw_barrier();

        // ---------------------------------------------------------------
        // ---------------------------------------------------------------
        // ---------------------------------------------------------------
        // MAXpool
        // ---------------------------------------------------------------
        // ---------------------------------------------------------------
        // ---------------------------------------------------------------

        // Obtain the start address of the TCDM memory
        uint32_t dma_load_input_start;
        uint32_t dma_load_input_end;
        uint32_t tcdm_baseaddress = snrt_cluster_base_addrl();
        // Put the input at the starting of tcdm
        uint8_t *tcdm_in = (uint8_t *)tcdm_baseaddress;
        // Put the output at the middle of tcdm
        uint8_t *tcdm_out = (uint8_t *)(tcdm_baseaddress + delta_local_out);

        if (snrt_is_dm_core()) {
            // The xdma core is the last compute core in the cluster
            uint32_t sstride_src[1] = {0};
            uint32_t sstride_dst[1] = {0};
            uint32_t tstride_src[5] = {0};
            uint32_t tbound_src[5] = {0};
            uint32_t tstride_dst[3] = {0};
            uint32_t tbound_dst[3] = {0};

            // Load the CFG from data.h
            sstride_src[0] = spatialStride1_in;
            sstride_dst[0] = spatialStride1_out;
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

            // // First we need to transfer the input data from L3->TCDM
            // snrt_dma_start_1d(tcdm_in, DataIn, input_data_len *
            // sizeof(int8_t)); snrt_dma_wait_all();

            // --------------------- Configure the Ext --------------------- //

            if (xdma_disable_dst_ext(0) != 0) {
                printf("Error in disabling xdma extension 0\r\n");
                err++;
            } else {
                printf("The xdma extension 0 is disabled\r\n");
            }

            uint32_t ext_param_maxpool_size[1] = {reduceLen};
            if (xdma_enable_dst_ext(1, ext_param_maxpool_size) != 0) {
                printf("Error in enabling xdma extension 1\r\n");
                err++;
            } else {
                printf("The xdma extension 1 is enabled\r\n");
            }

            if (xdma_disable_dst_ext(2) != 0) {
                printf("Error in disabling xdma extension 2\r\n");
                err++;
            } else {
                printf("The xdma extension 2 is disabled\r\n");
            }

            // --------------------- Configure the AGU --------------------- //

            xdma_memcpy_nd(tcdm_in, tcdm_out, sstride_src, sstride_dst, 5,
                           tstride_src, tbound_src, 3, tstride_dst, tbound_dst,
                           0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
            start_cycle = snrt_mcycle();

            int task_id = xdma_start();
            xdma_wait(task_id);

            end_cycle = snrt_mcycle();

            printf("[MAXPOOL]-XDMA cycle from mcycle: %d \r\n",
                   end_cycle - start_cycle);

            // --------------------- Checking the Results ---------------------
            // //
            // for (int i = 0; i < output_data_len; i++) {
            //     if ((int8_t)tcdm_out[i] != C_golden[i]) {
            //         printf("The maxpool is incorrect!\r\n");
            //         printf("tcdm_out[%d]=%d, C_golden[%d]=%d", i,
            //                (int8_t)tcdm_out[i], i, C_golden[i]);
            //     }
            // }
            // printf("Checking is done. All values are right\r\n");
        }
        snrt_cluster_hw_barrier();

        // ---------------------------------------------------------------
        // ---------------------------------------------------------------
        // ---------------------------------------------------------------
        // FC
        // ---------------------------------------------------------------
        // ---------------------------------------------------------------
        // ---------------------------------------------------------------

        // Prepare addresses in TCDM
        int8_t *local_a_fc, *local_b_fc;
        int32_t *local_c_fc, *local_d32_fc;
        int8_t *local_d8_fc;

        // Allocate space in TCDM
        local_a_fc = (int8_t *)(snrt_l1_next() + delta_local_a_fc);
        local_b_fc = (int8_t *)(snrt_l1_next() + delta_local_b_fc);
        local_c_fc = (int32_t *)(snrt_l1_next() + delta_local_c_fc);
        local_d32_fc = (int32_t *)(snrt_l1_next() + delta_local_d32_fc);
        local_d8_fc = (int8_t *)(snrt_l1_next() + delta_local_d8_fc);

        // Transfer data from L3 to L1
        // Using DMA only
        if (snrt_is_dm_core()) {
            // snrt_dma_start_1d(local_a_fc, A_fc,
            //                   M_fc * K_fc * meshRow_fc * tileSize_fc *
            //                   sizeof(int8_t));
            snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY,
                                    snrt_hartid());
            snrt_dma_start_1d(
                local_b_fc, B_fc,
                N_fc * K_fc * tileSize * meshCol * sizeof(int8_t));

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
            uint32_t subtraction_setting_fc =
                gen_subtraction_config(subtraction_a_fc, subtraction_b_fc);

            uint32_t csr0_fc = gen_csr0_config(input_zp_i_fc, output_zp_i_fc,
                                               max_int_i_fc, min_int_i_fc);
            uint32_t csr1_fc = gen_csr1_config(double_round_i_fc);

            start_cycle = snrt_mcycle();
            set_gemmx_csr(K_fc, N_fc, M_fc, subtraction_setting_fc, csr0_fc,
                          csr1_fc, shared_bitpacked_shift0_fc,
                          shared_bitpacked_shift1_fc, shared_multiplier0_fc,
                          shared_multiplier1_fc, shared_multiplier2_fc,
                          shared_multiplier3_fc, shared_multiplier4_fc,
                          shared_multiplier5_fc, shared_multiplier6_fc,
                          shared_multiplier7_fc, M_fc * N_fc, bypassSIMD_fc);

            // Set Streamer configuration CSR for conv2d
            set_gemmx_streamer_csr(
                Aslstride0_fc, Aslstride1_fc, Atlbound0_fc, Atlstride0_fc,
                Atlbound1_fc, Atlstride1_fc, Atlbound2_fc, Atlstride2_fc,
                Atlbound3_fc, Atlstride3_fc, Atlbound4_fc, Atlstride4_fc,
                Atlbound5_fc, Atlstride5_fc, set_addr_remap_index_A_fc,

                Bslstride0_fc, Bslstride1_fc, Btlbound0_fc, Btlstride0_fc,
                Btlbound1_fc, Btlstride1_fc, Btlbound2_fc, Btlstride2_fc,
                set_addr_remap_index_B_fc,

                D8slstride0_fc, D8slstride1_fc, D8tlbound0_fc, D8tlstride0_fc,
                D8tlbound1_fc, D8tlstride1_fc, D8tlbound2_fc, D8tlstride2_fc,
                set_addr_remap_index_D8_fc,

                Cslstride0_fc, Cslstride1_fc, Ctlbound0_fc, Ctlstride0_fc,
                Ctlbound1_fc, Ctlstride1_fc, Ctlbound2_fc, Ctlstride2_fc,
                set_addr_remap_index_C_fc,

                D32slstride0_fc, D32slstride1_fc, D32tlbound0_fc,
                D32tlstride0_fc, D32tlbound1_fc, D32tlstride1_fc,
                D32tlbound2_fc, D32tlstride2_fc, set_addr_remap_index_D32_fc,

                delta_local_a_fc, delta_local_b_fc, delta_local_d8_fc,
                delta_local_c_fc, delta_local_d32_fc, bypassSIMD_fc,
                transposed_A_fc, transposed_B_fc, channel_en_C_fc,
                broadcast_C_fc);

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

            while (0) {
                // Set CSR to start Streamer for conv2d
                set_gemmx_streamer_start();

                // Set CSR to start GEMM
                set_gemmx_start();

                // Poll until Streamer and GEMM accelerator finish
                wait_gemmx_and_streamer();
            }

            // printf("Computation finished \r\n");
            // check the result of the implicit im2col convolution
            // if (!bypassSIMD_fc) {
            //     err += check_gemmx_result_D8(local_d8_fc, D8_fc, Batch_fc,
            //     M_fc, N_fc, false);
            // } else {
            //     err +=
            //         check_gemmx_result_D32(local_d32_fc, D32_fc, Batch_fc,
            //         M_fc, N_fc, false);
            // }

            // printf("SNAX GEMM Matmul: %s, Error: %d . bypassSIMD = %d .\r\n",
            //    err ? "FAIL" : "PASS", err, bypassSIMD);

            if (err == 0) {
                write_csr_obs(0x00f);
                printf("P \r\n");
            } else {
                write_csr_obs(0x00e);
                printf("F: %d \r\n", err);
            }
        };

        snrt_cluster_hw_barrier();

        // ---------------------------------------------------------------
        // ---------------------------------------------------------------
        // ---------------------------------------------------------------
        // Data Reshuffler
        // ---------------------------------------------------------------
        // ---------------------------------------------------------------
        // ---------------------------------------------------------------

        // Put the input at the starting of tcdm
        uint8_t *tcdm_in_dr = (uint8_t *)tcdm_baseaddress + delta_local_d8_fc;
        // Put the output at the middle of tcdm
        uint8_t *tcdm_out_dr = (uint8_t *)(tcdm_baseaddress + delta_local_out_dr);

        if (snrt_is_dm_core()) {
            // The xdma core is the last compute core in the cluster
            uint32_t sstride_src_dr[1] = {0};
            uint32_t sstride_dst_dr[1] = {0};
            uint32_t tstride_src_dr[5] = {0};
            uint32_t tbound_src_dr[5] = {0};
            uint32_t tstride_dst_dr[3] = {0};
            uint32_t tbound_dst_dr[3] = {0};

            // Load the CFG from data.h
            sstride_src_dr[0] = spatialStride1_in_dr;
            sstride_dst_dr[0] = spatialStride1_out_dr;

            tstride_src_dr[0] = tempStride0_in_dr;
            tstride_src_dr[1] = tempStride1_in_dr;
            tstride_src_dr[2] = tempStride2_in_dr;
            tstride_src_dr[3] = tempStride3_in_dr;
            tstride_src_dr[4] = tempStride4_in_dr;

            tbound_src_dr[0] = tempLoop0_in_dr;
            tbound_src_dr[1] = tempLoop1_in_dr;
            tbound_src_dr[2] = tempLoop2_in_dr;
            tbound_src_dr[3] = tempLoop3_in_dr;
            tbound_src_dr[4] = tempLoop4_in_dr;

            tstride_dst_dr[0] = tempStride0_out_dr;
            tstride_dst_dr[1] = tempStride1_out_dr;
            tstride_dst_dr[2] = tempStride2_out_dr;

            tbound_dst_dr[0] = tempLoop0_out_dr;
            tbound_dst_dr[1] = tempLoop1_out_dr;
            tbound_dst_dr[2] = tempLoop2_out_dr;

            // --------------------- Configure the Ext --------------------- //

            if (xdma_disable_dst_ext(0) != 0) {
                printf("Error in disabling xdma extension 0\r\n");
                err++;
            } else {
                printf("The xdma extension 0 is disabled\r\n");
            }

            if (xdma_disable_dst_ext(1) != 0) {
                printf("Error in disabling xdma extension 1\r\n");
                err++;
            } else {
                printf("The xdma extension 1 is disabled\r\n");
            }

            if (xdma_enable_dst_ext(2, NULL) != 0) {
                printf("Error in enabling xdma extension 2\r\n");
                err++;
            } else {
                printf("The xdma extension 2 is enabled\r\n");
            }

            // --------------------- Configure the AGU --------------------- //

            xdma_memcpy_nd(tcdm_in_dr, tcdm_out_dr, sstride_src_dr, sstride_dst_dr, 5,
                           tstride_src_dr, tbound_src_dr, 3, tstride_dst_dr, tbound_dst_dr,
                           0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
            start_cycle = snrt_mcycle();

            int task_id = xdma_start();
            xdma_wait(task_id);

            end_cycle = snrt_mcycle();

            printf("[Data Reshuffler]-XDMA cycle from mcycle: %d \r\n",
                   end_cycle - start_cycle);

        }

        snrt_cluster_hw_barrier();

        return_to_cva6_single_cluster(err);
    }
}
