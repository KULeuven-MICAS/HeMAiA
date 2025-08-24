// // Copyright 2024 KU Leuven.
// // Licensed under the Apache License, Version 2.0, see LICENSE for details.
// // SPDX-License-Identifier: Apache-2.0
// //
// // Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

// #include "data.h"

// #include "snax-gemmx-params.h"

// #include "snax-gemmx-lib.h"

// int main() {
//     if (snrt_cluster_idx() == 1) {
//         // Set err value for checking
//         int err = 0;

//         // Prepare addresses in TCDM
//         int8_t *local_a_ping, *local_b_ping;
//         int32_t *local_c_ping, *local_d32_ping;
//         int8_t *local_d8_ping;

//         int8_t *local_a_pong, *local_b_pong;
//         int32_t *local_c_pong, *local_d32_pong;
//         int8_t *local_d8_pong;

//         // Allocate space in TCDM
//         local_a_ping = (int8_t *)(snrt_l1_next() + delta_local_a_ping);
//         local_b_ping = (int8_t *)(snrt_l1_next() + delta_local_b_ping);
//         local_c_ping = (int32_t *)(snrt_l1_next() + delta_local_c_ping);
//         local_d32_ping = (int32_t *)(snrt_l1_next() + delta_local_d32_ping);
//         local_d8_ping = (int8_t *)(snrt_l1_next() + delta_local_d8_ping);

//         local_a_pong = (int8_t *)(snrt_l1_next() + delta_local_a_pong);
//         local_b_pong = (int8_t *)(snrt_l1_next() + delta_local_b_pong);
//         local_c_pong = (int32_t *)(snrt_l1_next() + delta_local_c_pong);
//         local_d32_pong = (int32_t *)(snrt_l1_next() + delta_local_d32_pong);
//         local_d8_pong = (int8_t *)(snrt_l1_next() + delta_local_d8_pong);

//         int8_t *cur_A, *cur_B;
//         int32_t *cur_C;
//         cur_A = A;
//         cur_B = B;
//         cur_C = C;

//         int current_compute_counter = 0;
//         int sw_loop_break = 1;

//         // first data load
//         // Transfer data from L3 to L1
//         // Using DMA only

//         // upload A and B in every software loop
//         if (snrt_is_dm_core()) {
//             cur_A = A;
//             cur_B = B;
//             snrt_dma_start_1d(local_a_ping, cur_A,
//                               M * K * meshRow * tileSize * sizeof(int8_t));
//             snrt_dma_start_1d(local_b_ping, cur_B,
//                               N * K * tileSize * meshCol * sizeof(int8_t));

//             snrt_dma_wait_all();
//         }

//         // Wait for DMA to finish
//         snrt_cluster_hw_barrier();

//         // only update the C when k == 0
//         // C initialization, otherwise use the partial sum
//         if (snrt_is_dm_core()) {
//             cur_C = C;
//             snrt_dma_start_1d(local_d32_ping, cur_C,
//                               M * N * meshRow * meshCol * sizeof(int32_t));
//             snrt_dma_wait_all();
//         }

//         snrt_cluster_hw_barrier();

//         uint32_t subtraction_setting =
//             gen_subtraction_config(subtraction_a, subtraction_b);

//         uint32_t csr0 =
//             gen_csr0_config(input_zp_i, output_zp_i, max_int_i, min_int_i);
//         uint32_t csr1 = gen_csr1_config(double_round_i);

//         for (int m = 0; m < M2; m++) {
//             for (int n = 0; n < N2; n++) {
//                 for (int k = 0; k < K2; k++) {
//                     // compute conduction
//                     if (snrt_cluster_core_idx() == 0) {
//                         // Set Streamer configuration CSR for conv2d
//                         if (current_compute_counter % 2 == 0) {
//                             set_gemmx_streamer_csr(
//                                 Aslstride0, Aslstride1, Atlbound0,
//                                 Atlstride0, Atlbound1, Atlstride1, Atlbound2,
//                                 Atlstride2, Atlbound3, Atlstride3, Atlbound4,
//                                 Atlstride4, Atlbound5, Atlstride5,
//                                 set_addr_remap_index_A,

//                                 Bslstride0, Bslstride1, Btlbound0,
//                                 Btlstride0, Btlbound1, Btlstride1, Btlbound2,
//                                 Btlstride2, set_addr_remap_index_B,

//                                 D8slstride0, D8slstride1, D8tlbound0,
//                                 D8tlstride0, D8tlbound1, D8tlstride1,
//                                 D8tlbound2, D8tlstride2,
//                                 set_addr_remap_index_D8,

//                                 Cslstride0, Cslstride1, Ctlbound0,
//                                 Ctlstride0, Ctlbound1, Ctlstride1, Ctlbound2,
//                                 Ctlstride2, set_addr_remap_index_C,

//                                 D32slstride0, D32slstride1, D32tlbound0,
//                                 D32tlstride0, D32tlbound1, D32tlstride1,
//                                 D32tlbound2, D32tlstride2,
//                                 set_addr_remap_index_D32,

//                                 delta_local_a_ping, delta_local_b_ping,
//                                 delta_local_d8_ping, delta_local_d32_ping,
//                                 delta_local_d32_ping, bypassSIMD,
//                                 transposed_A, transposed_B, channel_en_C,
//                                 broadcast_C);
//                         } else {
//                             set_gemmx_streamer_csr(
//                                 Aslstride0, Aslstride1, Atlbound0,
//                                 Atlstride0, Atlbound1, Atlstride1, Atlbound2,
//                                 Atlstride2, Atlbound3, Atlstride3, Atlbound4,
//                                 Atlstride4, Atlbound5, Atlstride5,
//                                 set_addr_remap_index_A,

//                                 Bslstride0, Bslstride1, Btlbound0,
//                                 Btlstride0, Btlbound1, Btlstride1, Btlbound2,
//                                 Btlstride2, set_addr_remap_index_B,

//                                 D8slstride0, D8slstride1, D8tlbound0,
//                                 D8tlstride0, D8tlbound1, D8tlstride1,
//                                 D8tlbound2, D8tlstride2,
//                                 set_addr_remap_index_D8,

//                                 Cslstride0, Cslstride1, Ctlbound0,
//                                 Ctlstride0, Ctlbound1, Ctlstride1, Ctlbound2,
//                                 Ctlstride2, set_addr_remap_index_C,

//                                 D32slstride0, D32slstride1, D32tlbound0,
//                                 D32tlstride0, D32tlbound1, D32tlstride1,
//                                 D32tlbound2, D32tlstride2,
//                                 set_addr_remap_index_D32,

//                                 delta_local_a_pong, delta_local_b_pong,
//                                 delta_local_d8_pong, delta_local_d32_pong,
//                                 delta_local_d32_pong, bypassSIMD,
//                                 transposed_A, transposed_B, channel_en_C,
//                                 broadcast_C);
//                         }

//                         // Set GEMMX configuration CSR

//                         set_gemmx_csr(K, N, M, subtraction_setting, csr0,
//                         csr1,
//                                       shared_bitpacked_shift0,
//                                       shared_bitpacked_shift1,
//                                       shared_multiplier0, shared_multiplier1,
//                                       shared_multiplier2, shared_multiplier3,
//                                       shared_multiplier4, shared_multiplier5,
//                                       shared_multiplier6, shared_multiplier7,
//                                       M * N, bypassSIMD);

//                         // Set CSR to start Streamer for conv2d
//                         set_gemmx_streamer_start();

//                         // Set CSR to start GEMM
//                         set_gemmx_start();

//                         // Poll until Streamer and GEMM accelerator finish
//                         wait_gemmx_and_streamer();

//                         if (current_compute_counter == (K2 * M2 * N2 - 1)) {
//                             sw_loop_break = 1;
//                             break;
//                         }
//                         current_compute_counter = current_compute_counter +
//                         1;
//                     };

//                     // snrt_cluster_hw_barrier();

//                     // load the the next data to be computed
//                     if (snrt_is_dm_core()) {
//                         // the index for fetching next data
//                         int next_k = current_compute_counter % K2;
//                         int next_n = current_compute_counter / K2 % N2;
//                         int next_m = current_compute_counter / (K2 * N2) %
//                         M2;

//                         // load the correct next piece of data
//                         cur_A = A + next_m * K2 * M * K * meshRow * tileSize
//                         +
//                                 next_k * M * K * meshRow * tileSize;
//                         // load the correct next piece of data
//                         cur_B = B + next_n * K2 * N * K * tileSize * meshCol
//                         +
//                                 next_k * N * K * tileSize * meshCol;
//                         if (current_compute_counter % 2 == 0) {
//                             snrt_dma_start_1d(
//                                 local_a_ping, cur_A,
//                                 M * K * meshRow * tileSize * sizeof(int8_t));
//                             snrt_dma_start_1d(
//                                 local_b_ping, cur_B,
//                                 N * K * tileSize * meshCol * sizeof(int8_t));
//                         } else {
//                             snrt_dma_start_1d(
//                                 local_a_pong, cur_A,
//                                 M * K * meshRow * tileSize * sizeof(int8_t));
//                             snrt_dma_start_1d(
//                                 local_b_pong, cur_B,
//                                 N * K * tileSize * meshCol * sizeof(int8_t));
//                         }

//                         snrt_dma_wait_all();
//                     }

//                     // Wait for DMA to finish
//                     snrt_cluster_hw_barrier();

//                     if (k == 0) {
//                         if (snrt_is_dm_core()) {
//                             // load the correct next piece of data
//                             cur_C = C +
//                                     next_m * N2 * M * N * meshRow * meshCol +
//                                     next_n * M * N * meshRow * meshCol;
//                             if (current_compute_counter % 2 == 0) {
//                                 snrt_dma_start_1d(local_d32_ping, cur_C,
//                                                   M * N * meshRow * meshCol *
//                                                       sizeof(int32_t));
//                             } else {
//                                 snrt_dma_start_1d(local_d32_pong, cur_C,
//                                                   M * N * meshRow * meshCol *
//                                                       sizeof(int32_t));
//                             }
//                             snrt_dma_wait_all();
//                         }
//                     }

//                     snrt_cluster_hw_barrier();
//                 }

//                 if (sw_loop_break == 1) {
//                     break;
//                 }

//                 snrt_cluster_hw_barrier();

//                 // check the result after K times computation and
//                 accumulation if (snrt_cluster_core_idx() == 0) {
//                     if (!bypassSIMD) {
//                         printf("start to verify the computed data for D8
//                         \r\n"); for (int i = 0; i < M * N * meshRow *
//                         meshCol; i++) {
//                             if (local_d8[i] !=
//                                 D8_golden[i +
//                                           m * N2 * M * N * meshRow * meshCol
//                                           + n * M * N * meshRow * meshCol]) {
//                                 err++;
//                                 // printf(
//                                 //     "SNAX GEMM Matmul failed to verify the
//                                 "
//                                 //     "computed data at index %d, expected
//                                 %d "
//                                 //     "but got %d \r\n",
//                                 //     i,
//                                 //     D8_golden[i +
//                                 //               m * N2 * M * N * meshRow *
//                                 //                   meshCol +
//                                 //               n * M * N * meshRow *
//                                 meshCol],
//                                 //     local_d8[i]);
//                             }
//                         }
//                     } else {
//                         printf(
//                             "start to verify the computed data for D32
//                             \r\n");
//                         for (int i = 0; i < M * N * meshRow * meshCol; i++) {
//                             if (local_d32[i] !=
//                                 D32_golden[i +
//                                            m * N2 * M * N * meshRow * meshCol
//                                            + n * M * N * meshRow * meshCol])
//                                            {
//                                 err++;
//                                 printf(
//                                     "SNAX GEMM Matmul failed to verify the "
//                                     "computed data at index %d, at address =
//                                     "
//                                     "%p, expected %d but got %d \r\n",
//                                     i,
//                                     &D32_golden[i +
//                                                 m * N2 * M * N * meshRow *
//                                                     meshCol +
//                                                 n * M * N * meshRow *
//                                                 meshCol],
//                                     D32_golden[i +
//                                                m * N2 * M * N * meshRow *
//                                                    meshCol +
//                                                n * M * N * meshRow *
//                                                meshCol],
//                                     local_d32[i]);
//                             }
//                         }
//                     }

//                     printf(
//                         "SNAX GEMM Matmul: %s, Error: %d . bypassSIMD = %d,
//                         m= "
//                         "%d, n= %d \n",
//                         err ? "FAIL" : "PASS", err, bypassSIMD, m, n);
//                 }

//                 snrt_cluster_hw_barrier();

//                 // load the final result back to L3
//                 if (snrt_is_dm_core()) {
//                     if (!bypassSIMD) {
//                         snrt_dma_start_1d(
//                             D8_generated + m * N2 * M * N * meshRow * meshCol
//                             +
//                                 n * M * N * meshRow * meshCol,
//                             local_d8,
//                             M * N * meshRow * meshCol * sizeof(int8_t));
//                         snrt_dma_wait_all();
//                     } else {
//                         snrt_dma_start_1d(
//                             D32_generated + m * N2 * M * N * meshRow *
//                             meshCol +
//                                 n * M * N * meshRow * meshCol,
//                             local_d32,
//                             M * N * meshRow * meshCol * sizeof(int32_t));
//                         snrt_dma_wait_all();
//                     }
//                 }
//             }

//             if (sw_loop_break == 1) {
//                 break;
//             }
//         }

//         // last computation and result check
//         if (snrt_cluster_core_idx() == 0) {
//             // Set Streamer configuration CSR for conv2d
//             if (current_compute_counter % 2 == 0) {
//                 set_gemmx_streamer_csr(
//                     Aslstride0, Aslstride1, Atlbound0, Atlstride0, Atlbound1,
//                     Atlstride1, Atlbound2, Atlstride2, Atlbound3, Atlstride3,
//                     Atlbound4, Atlstride4, Atlbound5, Atlstride5,
//                     set_addr_remap_index_A,

//                     Bslstride0, Bslstride1, Btlbound0, Btlstride0, Btlbound1,
//                     Btlstride1, Btlbound2, Btlstride2,
//                     set_addr_remap_index_B,

//                     D8slstride0, D8slstride1, D8tlbound0, D8tlstride0,
//                     D8tlbound1, D8tlstride1, D8tlbound2, D8tlstride2,
//                     set_addr_remap_index_D8,

//                     Cslstride0, Cslstride1, Ctlbound0, Ctlstride0, Ctlbound1,
//                     Ctlstride1, Ctlbound2, Ctlstride2,
//                     set_addr_remap_index_C,

//                     D32slstride0, D32slstride1, D32tlbound0, D32tlstride0,
//                     D32tlbound1, D32tlstride1, D32tlbound2, D32tlstride2,
//                     set_addr_remap_index_D32,

//                     delta_local_a_ping, delta_local_b_ping,
//                     delta_local_d8_ping, delta_local_d32_ping,
//                     delta_local_d32_ping, bypassSIMD, transposed_A,
//                     transposed_B, channel_en_C, broadcast_C);
//             } else {
//                 set_gemmx_streamer_csr(
//                     Aslstride0, Aslstride1, Atlbound0, Atlstride0, Atlbound1,
//                     Atlstride1, Atlbound2, Atlstride2, Atlbound3, Atlstride3,
//                     Atlbound4, Atlstride4, Atlbound5, Atlstride5,
//                     set_addr_remap_index_A,

//                     Bslstride0, Bslstride1, Btlbound0, Btlstride0, Btlbound1,
//                     Btlstride1, Btlbound2, Btlstride2,
//                     set_addr_remap_index_B,

//                     D8slstride0, D8slstride1, D8tlbound0, D8tlstride0,
//                     D8tlbound1, D8tlstride1, D8tlbound2, D8tlstride2,
//                     set_addr_remap_index_D8,

//                     Cslstride0, Cslstride1, Ctlbound0, Ctlstride0, Ctlbound1,
//                     Ctlstride1, Ctlbound2, Ctlstride2,
//                     set_addr_remap_index_C,

//                     D32slstride0, D32slstride1, D32tlbound0, D32tlstride0,
//                     D32tlbound1, D32tlstride1, D32tlbound2, D32tlstride2,
//                     set_addr_remap_index_D32,

//                     delta_local_a_pong, delta_local_b_pong,
//                     delta_local_d8_pong, delta_local_d32_pong,
//                     delta_local_d32_pong, bypassSIMD, transposed_A,
//                     transposed_B, channel_en_C, broadcast_C);
//             }

//             // Set GEMMX configuration CSR

//             set_gemmx_csr(
//                 K, N, M, subtraction_setting, csr0, csr1,
//                 shared_bitpacked_shift0, shared_bitpacked_shift1,
//                 shared_multiplier0, shared_multiplier1, shared_multiplier2,
//                 shared_multiplier3, shared_multiplier4, shared_multiplier5,
//                 shared_multiplier6, shared_multiplier7, M * N, bypassSIMD);

//             // Set CSR to start Streamer for conv2d
//             set_gemmx_streamer_start();

//             // Set CSR to start GEMM
//             set_gemmx_start();

//             // Poll until Streamer and GEMM accelerator finish
//             wait_gemmx_and_streamer();

//             if (!bypassSIMD) {
//                 printf("start to verify the computed data for D8 \r\n");
//                 for (int i = 0; i < M * N * meshRow * meshCol; i++) {
//                     if (local_d8[i] !=
//                         D8_golden[i + m * N2 * M * N * meshRow * meshCol +
//                                   n * M * N * meshRow * meshCol]) {
//                         err++;
//                         // printf(
//                         //     "SNAX GEMM Matmul failed to verify the "
//                         //     "computed data at index %d, expected %d "
//                         //     "but got %d \r\n",
//                         //     i,
//                         //     D8_golden[i +
//                         //               m * N2 * M * N * meshRow *
//                         //                   meshCol +
//                         //               n * M * N * meshRow * meshCol],
//                         //     local_d8[i]);
//                     }
//                 }
//             } else {
//                 printf("start to verify the computed data for D32 \r\n");
//                 for (int i = 0; i < M * N * meshRow * meshCol; i++) {
//                     if (local_d32[i] !=
//                         D32_golden[i + m * N2 * M * N * meshRow * meshCol +
//                                    n * M * N * meshRow * meshCol]) {
//                         err++;
//                         printf(
//                             "SNAX GEMM Matmul failed to verify the "
//                             "computed data at index %d, at address = "
//                             "%p, expected %d but got %d \r\n",
//                             i,
//                             &D32_golden[i + m * N2 * M * N * meshRow *
//                             meshCol +
//                                         n * M * N * meshRow * meshCol],
//                             D32_golden[i + m * N2 * M * N * meshRow * meshCol
//                             +
//                                        n * M * N * meshRow * meshCol],
//                             local_d32[i]);
//                     }
//                 }
//             }

//             printf(
//                 "SNAX GEMM Matmul: %s, Error: %d . bypassSIMD = %d, m= "
//                 "%d, n= %d \n",
//                 err ? "FAIL" : "PASS", err, bypassSIMD, m, n);
//         }

//         snrt_cluster_hw_barrier();
//         return_to_cva6_single_cluster(err);
//     }
// }

#include "snrt.h"

int main() {
    if (snrt_cluster_idx() == 1) {
        return_to_cva6_single_cluster(0);
    };
}
