// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

#include "data.h"

#include "snax-gemmx-params.h"

#include "snax-gemmx-lib.h"

// This is the main function for the SNAX GEMM for Conv2d
// We use several nested loops to iterate over the input data and weights,
// achieving implicit im2col
int main() {
    if (snrt_cluster_idx() == 1) {  // Set err value for checking
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

        int8_t *cur_A, *cur_B;
        int32_t *cur_C;
        cur_A = A;
        cur_B = B;
        cur_C = C;

        for (int m = 0; m < M2;) {
            for (int n = 0; n < N2;) {
                for (int k = 0; k < K2;) {
                    // Transfer data from L3 to L1
                    // Using DMA only

                    if (snrt_is_dm_core()) {
                        cur_A = A + m * K2 * M * K * meshRow * tileSize +
                                k * M * K * meshRow * tileSize;
                        cur_B = B + n * K2 * N * K * tileSize * meshCol +
                                k * N * K * tileSize * meshCol;
                        snrt_dma_start_1d(
                            local_a, cur_A,
                            M * K * meshRow * tileSize * sizeof(int8_t));
                        snrt_dma_start_1d(
                            local_b, cur_B,
                            N * K * tileSize * meshCol * sizeof(int8_t));

                        snrt_dma_wait_all();
                    }

                    // Wait for DMA to finish
                    snrt_cluster_hw_barrier();

                    if (snrt_is_dm_core()) {
                        if (k == 0) {
                            cur_C = C + m * N2 * M * N * meshRow * meshCol +
                                    n * M * N * meshRow * meshCol;
                            snrt_dma_start_1d(
                                local_d32, cur_C,
                                M * N * meshRow * meshCol * sizeof(int32_t));
                            snrt_dma_wait_all();
                        }
                    }

                    snrt_cluster_hw_barrier();

                    if (snrt_cluster_core_idx() == 0) {
                        // printf("start to print the data \r\n");
                        // // print A
                        // for (int i = 0;
                        //      i < (M * K * meshRow * tileSize); i++) {
                        //     printf("A[%d] = %d \r\n", i, local_a[i]);
                        // }
                        // // print B
                        // for (int i = 0;
                        //      i < (K * N * tileSize * meshCol); i++) {
                        //     printf("B[%d] = %d \r\n", i, local_b[i]);
                        // }
                        // // print C
                        // for (int i = 0;
                        //      i < (M * N * meshRow * meshCol); i++) {
                        //     printf("C[%d] = %d \r\n", i, local_d32[i]);
                        // }

                        printf(
                            "delta_local_a = %d, delta_local_b = %d, "
                            "delta_local_c = %d, delta_local_d32 = %d \r\n",
                            delta_local_a, delta_local_b, delta_local_c,
                            delta_local_d32);
                        printf("start to configure the CSR \r\n");

                        // Set Streamer configuration CSR for conv2d
                        set_gemmx_streamer_csr(
                            Aslstride0, Aslstride1, Atlbound0, Atlstride0,
                            Atlbound1, Atlstride1, Atlbound2, Atlstride2,
                            Atlbound3, Atlstride3, Atlbound4, Atlstride4,
                            Atlbound5, Atlstride5, set_addr_remap_index_A,

                            Bslstride0, Bslstride1, Btlbound0, Btlstride0,
                            Btlbound1, Btlstride1, Btlbound2, Btlstride2,
                            set_addr_remap_index_B,

                            D8slstride0, D8slstride1, D8tlbound0, D8tlstride0,
                            D8tlbound1, D8tlstride1, D8tlbound2, D8tlstride2,
                            set_addr_remap_index_D8,

                            Cslstride0, Cslstride1, Ctlbound0, Ctlstride0,
                            Ctlbound1, Ctlstride1, Ctlbound2, Ctlstride2,
                            set_addr_remap_index_C,

                            D32slstride0, D32slstride1, D32tlbound0,
                            D32tlstride0, D32tlbound1, D32tlstride1,
                            D32tlbound2, D32tlstride2, set_addr_remap_index_D32,

                            delta_local_a, delta_local_b, delta_local_d8,
                            delta_local_d32, delta_local_d32, bypassSIMD,
                            transposed_A, transposed_B, channel_en_C,
                            broadcast_C);

                        // Set GEMMX configuration CSR
                        uint32_t subtraction_setting = gen_subtraction_config(
                            subtraction_a, subtraction_b);

                        uint32_t csr0 = gen_csr0_config(input_zp_i, output_zp_i,
                                                        max_int_i, min_int_i);
                        uint32_t csr1 = gen_csr1_config(double_round_i);

                        set_gemmx_csr(K, N, M, subtraction_setting, csr0, csr1,
                                      shared_bitpacked_shift0,
                                      shared_bitpacked_shift1,
                                      shared_multiplier0, shared_multiplier1,
                                      shared_multiplier2, shared_multiplier3,
                                      shared_multiplier4, shared_multiplier5,
                                      shared_multiplier6, shared_multiplier7,
                                      M * N, bypassSIMD);

                        printf("CSR configure finished and start now \r\n");
                        // Set CSR to start Streamer for conv2d
                        set_gemmx_streamer_start();

                        // Set CSR to start GEMM
                        set_gemmx_start();
                        int start_cycle;
                        asm volatile("csrr %0, mcycle" : "=r"(start_cycle));

                        // Poll until Streamer and GEMM accelerator finish
                        wait_gemmx_and_streamer();
                        int end_cycle;
                        asm volatile("csrr %0, mcycle" : "=r"(end_cycle));
                        printf("SNAX GEMM Matmul finished, cycle = %d\r\n", (end_cycle - start_cycle));
                        k++;
                    };
                    snrt_cluster_hw_barrier();
                }

                snrt_cluster_hw_barrier();

                // check the result of the implicit im2col convolution
                if (snrt_cluster_core_idx() == 0) {
                    for (int i = 0; i < M * N * meshRow * meshCol; i++) {
                        asm volatile("nop\n");
                    }
                    if (!bypassSIMD) {
                        for (int i = 0; i < M * N * meshRow * meshCol; i++) {
                            if (local_d8[i] !=
                                D8_golden[i +
                                          m * N2 * M * N * meshRow * meshCol +
                                          n * M * N * meshRow * meshCol]) {
                                err++;
                                // printf(
                                //     "SNAX GEMM Matmul failed to verify the "
                                //     "computed data at index %d, expected %d "
                                //     "but got %d \r\n",
                                //     i,
                                //     D8_golden[i +
                                //               m * N2 * M * N * meshRow *
                                //                   meshCol +
                                //               n * M * N * meshRow * meshCol],
                                //     local_d8[i]);
                            }
                        }
                    } else {
                        printf(
                            "start to verify the computed data for D32 \r\n");
                        // printf("m = %d, n = %d \r\n", m, n);
                        for (int i = 0; i < M * N * meshRow * meshCol; i++) {
                            if (local_d32[i] !=
                                D32_golden[i +
                                           m * N2 * M * N * meshRow * meshCol +
                                           n * M * N * meshRow * meshCol]) {
                                err++;
                                printf(
                                    "SNAX GEMM Matmul failed to verify the "
                                    "computed data at index %d, at address = "
                                    "%p, expected %d but got %d \r\n",
                                    i,
                                    &D32_golden[i +
                                                m * N2 * M * N * meshRow *
                                                    meshCol +
                                                n * M * N * meshRow * meshCol],
                                    D32_golden[i +
                                               m * N2 * M * N * meshRow *
                                                   meshCol +
                                               n * M * N * meshRow * meshCol],
                                    local_d32[i]);
                            }
                        }
                    }

                    printf(
                        "SNAX GEMM Matmul: %s, Error: %d . bypassSIMD = %d, m= "
                        "%d, n= %d \r\n",
                        err ? "FAIL" : "PASS", err, bypassSIMD, m, n);
                    n++;
                }

                snrt_cluster_hw_barrier();

                // if (snrt_is_dm_core()) {
                //     snrt_dma_start_1d(
                //         D32_generated + m * N2 * M * N * meshRow * meshCol +
                //             n * M * N * meshRow * meshCol,
                //         local_d32, M * N * meshRow * meshCol *
                //         sizeof(int32_t));
                //     snrt_dma_wait_all();
                // }

                // snrt_cluster_hw_barrier();
            }

            if (snrt_cluster_core_idx() == 0) {
                m++;
            }
            snrt_cluster_hw_barrier();
        }
        return err;
    } else
        return 0;
}
