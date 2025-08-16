// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

#include "snrt.h"
#include "data.h"
#include "snax-gemmx-lib.h"
#include "snax-gemmx-params.h"

int main() {

    if (snrt_cluster_idx() == 1) {
        int err = 0;

        int8_t *local_a, *local_b;
        int32_t *local_c, *local_d32;
        int8_t *local_d8;

        // Allocate space in TCDM
        local_a = (int8_t *)(snrt_l1_next() + delta_local_a);
        local_b = (int8_t *)(snrt_l1_next() + delta_local_b);
        local_c = (int32_t *)(snrt_l1_next() + delta_local_c);
        local_d32 = (int32_t *)(snrt_l1_next() + delta_local_d32);
        local_d8 = (int8_t *)(snrt_l1_next() + delta_local_d8);

        // Transfer data from L3 to L1
        // Using DMA only
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(local_a, A,
                              M * K * meshRow * tileSize * sizeof(int8_t));
            snrt_dma_start_1d(local_b, B,
                              N * K * tileSize * meshCol * sizeof(int8_t));

            snrt_dma_wait_all();
        }

        // Wait for DMA to finish
        snrt_cluster_hw_barrier();
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(local_c, C,
                              M * N * meshRow * meshCol * sizeof(int32_t));
            snrt_dma_wait_all();
        }

        snrt_cluster_hw_barrier();

        if (snrt_cluster_core_idx() == 0) {
            printf("Hello world from gemm core\r\n");
            // write streamer CSR manually
            csrw_ss(T_BOUND_READER_0_0, 1);
            csrw_ss(T_BOUND_READER_0_1, 1);
            csrw_ss(T_BOUND_READER_0_2, 1);
            csrw_ss(T_BOUND_READER_0_3, 1);
            csrw_ss(T_BOUND_READER_0_4, 1);
            csrw_ss(T_BOUND_READER_0_5, 1);

            int T_BOUND_READER_0_0_CSR = csrr_ss(T_BOUND_READER_0_0);
            int T_BOUND_READER_0_1_CSR = csrr_ss(T_BOUND_READER_0_1);
            int T_BOUND_READER_0_2_CSR = csrr_ss(T_BOUND_READER_0_2);
            int T_BOUND_READER_0_3_CSR = csrr_ss(T_BOUND_READER_0_3);
            int T_BOUND_READER_0_4_CSR = csrr_ss(T_BOUND_READER_0_4);
            int T_BOUND_READER_0_5_CSR = csrr_ss(T_BOUND_READER_0_5);

            if (T_BOUND_READER_0_0_CSR != 1){
                err = err + 1;
            }
            if (T_BOUND_READER_0_1_CSR != 1){
                err = err + 1;
            }
            if (T_BOUND_READER_0_2_CSR != 1){
                err = err + 1;
            }
            if (T_BOUND_READER_0_3_CSR != 1){
                err = err + 1;
            }
            if (T_BOUND_READER_0_4_CSR != 1){
                err = err + 1;
            }
            if (T_BOUND_READER_0_5_CSR != 1){
                err = err + 1;
            }

            csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)(delta_local_a + snrt_l1_next()));
            csrw_ss(S_STRIDE_READER_0_0, Aslstride1);

            printf("A err = %d\n", err);

            csrw_ss(T_BOUND_READER_1_0, 1);
            csrw_ss(T_BOUND_READER_1_1, 1);
            csrw_ss(T_BOUND_READER_1_2, 1);

            int T_BOUND_READER_1_0_CSR = csrr_ss(T_BOUND_READER_1_0);
            int T_BOUND_READER_1_1_CSR = csrr_ss(T_BOUND_READER_1_1);
            int T_BOUND_READER_1_2_CSR = csrr_ss(T_BOUND_READER_1_2);

            if (T_BOUND_READER_1_0_CSR != 1) {
                err = err + 1;
            }
            if (T_BOUND_READER_1_1_CSR != 1) {
                err = err + 1;
            }
            if (T_BOUND_READER_1_2_CSR != 1) {
                err = err + 1;
            }

            csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)(delta_local_b + snrt_l1_next()));
            csrw_ss(S_STRIDE_READER_1_0, Bslstride1);

            printf("B err = %d\n", err);

            csrw_ss(T_BOUND_WRITER_0_0, 0);
            csrw_ss(T_BOUND_WRITER_0_1, 0);
            csrw_ss(T_BOUND_WRITER_0_2, 0);

            int T_BOUND_WRITER_0_0_CSR = csrr_ss(T_BOUND_WRITER_0_0);
            int T_BOUND_WRITER_0_1_CSR = csrr_ss(T_BOUND_WRITER_0_1);
            int T_BOUND_WRITER_0_2_CSR = csrr_ss(T_BOUND_WRITER_0_2);

            if (T_BOUND_WRITER_0_0_CSR != 0) {
                err = err + 1;
            }
            if (T_BOUND_WRITER_0_1_CSR != 0) {
                err = err + 1;
            }
            if (T_BOUND_WRITER_0_2_CSR != 0) {
                err = err + 1;
            }
            printf("D8 err = %d\n", err);

            csrw_ss(T_BOUND_READER_WRITER_0_0, 1);
            csrw_ss(T_BOUND_READER_WRITER_0_1, 1);
            csrw_ss(T_BOUND_READER_WRITER_0_2, 1);

            int T_BOUND_READER_WRITER_0_0_CSR = csrr_ss(T_BOUND_READER_WRITER_0_0);
            int T_BOUND_READER_WRITER_0_1_CSR = csrr_ss(T_BOUND_READER_WRITER_0_1);
            int T_BOUND_READER_WRITER_0_2_CSR = csrr_ss(T_BOUND_READER_WRITER_0_2);

            if (T_BOUND_READER_WRITER_0_0_CSR != 1) {
                err = err + 1;
            }
            if (T_BOUND_READER_WRITER_0_1_CSR != 1) {
                err = err + 1;
            }
            if (T_BOUND_READER_WRITER_0_2_CSR != 1) {
                err = err + 1;
            }

            csrw_ss(BASE_PTR_READER_WRITER_0_LOW,
                    (uint32_t)(delta_local_c + snrt_l1_next()));

            printf("C err = %d\n", err);

            csrw_ss(T_BOUND_READER_WRITER_1_0, 1);
            csrw_ss(T_BOUND_READER_WRITER_1_1, 1);
            csrw_ss(T_BOUND_READER_WRITER_1_2, 1);

            int T_BOUND_READER_WRITER_1_0_CSR = csrr_ss(T_BOUND_READER_WRITER_1_0);
            int T_BOUND_READER_WRITER_1_1_CSR = csrr_ss(T_BOUND_READER_WRITER_1_1);
            int T_BOUND_READER_WRITER_1_2_CSR = csrr_ss(T_BOUND_READER_WRITER_1_2);

            if (T_BOUND_READER_WRITER_1_0_CSR != 1) {
                err = err + 1;
            }
            if (T_BOUND_READER_WRITER_1_1_CSR != 1) {
                err = err + 1;
            }
            if (T_BOUND_READER_WRITER_1_2_CSR != 1) {
                err = err + 1;
            }

            csrw_ss(BASE_PTR_READER_WRITER_1_LOW,
                    (uint32_t)(delta_local_d32 + snrt_l1_next()));

            csrw_ss(S_STRIDE_READER_WRITER_1_0, D32slstride0);
            csrw_ss(S_STRIDE_READER_WRITER_1_1, D32slstride1);

            printf("D32 err = %d\n", err);

            csrw_ss(TRANSPOSE_CSR_READER_0, 1);
            csrw_ss(TRANSPOSE_CSR_READER_1, 1);

            // GEMM CSR
            csrw_ss(T_BOUND_K, 1);
            csrw_ss(T_BOUND_N, 1);
            csrw_ss(T_BOUND_M, 1);

            int T_BOUND_K_CSR = csrr_ss(T_BOUND_K);
            int T_BOUND_N_CSR = csrr_ss(T_BOUND_N);
            int T_BOUND_M_CSR = csrr_ss(T_BOUND_M);

            if (T_BOUND_K_CSR != 1) {
                err = err + 1;
            }
            if (T_BOUND_N_CSR != 1) {
                err = err + 1;
            }
            if (T_BOUND_M_CSR != 1) {
                err = err + 1;
            }

            csrw_ss(BYPASS_SIMD, 1);
            int BYPASS_SIMD_CSR = csrr_ss(BYPASS_SIMD);

            if (BYPASS_SIMD_CSR != 1) {
                err = err + 1;
            }

            uint32_t subtraction_setting =
                gen_subtraction_config(subtraction_a, subtraction_b);

            csrw_ss(SUBTRACTIONS, subtraction_setting);

            int SUBTRACTIONS_CSR = csrr_ss(SUBTRACTIONS);

            if (SUBTRACTIONS_CSR != subtraction_setting) {
                err = err + 1;
            }

            printf("GeMM CFG err = %d\n", err);

            csrw_ss(GEMMX_START, 1);
            csrw_ss(STREAMER_START_CSR, 1);

            write_csr_obs(0x009);

            int GEMMX_BUSY_CSR = csrr_ss(GEMMX_BUSY);
            while (GEMMX_BUSY_CSR) {
                printf("GEMMX is busy\n");
                GEMMX_BUSY_CSR = csrr_ss(GEMMX_BUSY);
            }

            write_csr_obs(0x00a);

            while (csrr_ss(STREAMER_BUSY_CSR)) {
            }

            write_csr_obs(0x00b);

            for (int i = 0; i < Batch * M * N * meshRow * meshCol; i++) {
                if (local_d32[i] != D32[i]) {
                    printf("Mismatch at index %d: local_d32 = %d, D32 = %d \r\n", i, local_d32[i], D32[i]);
                    err++;
                }
            }

            if (err == 0) {
                write_csr_obs(0x00e);
                printf("P \r\n");
            } else {
                write_csr_obs(0x00f);
                printf("F: %d \r\n", err);
            }

            if (err == 0) {
                write_csr_obs(0x00e);
                printf("P \r\n");
            } else {
                write_csr_obs(0x00f);
                printf("F: %d \r\n", err);
            }

            if (err == 0) {
                write_csr_obs(0x00e);
                printf("P \r\n");
            } else {
                write_csr_obs(0x00f);
                printf("F: %d \r\n", err);
            }

            if (err == 0) {
                write_csr_obs(0x00e);
                printf("P \r\n");
            } else {
                write_csr_obs(0x00f);
                printf("F: %d \r\n", err);
            }
        };

        snrt_cluster_hw_barrier();

        return_to_cva6_single_cluster(err);
    }
}
