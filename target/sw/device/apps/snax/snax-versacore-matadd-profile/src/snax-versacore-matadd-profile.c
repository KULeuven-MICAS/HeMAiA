// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

#include "data.h"

#include "snax_versacore_lib.h"

int main() {
    // Set err value for checking
    int err = 0;

    // Prepare addresses in TCDM
    int8_t *local_a, *local_b;
    int32_t *local_c, *local_d;

    // Allocate space in TCDM
    local_a = (int8_t *)(snrt_cluster_base_addrl() + delta_local_a);
    local_b = (int8_t *)(snrt_cluster_base_addrl() + delta_local_b);
    local_c = (int32_t *)(snrt_cluster_base_addrl() + delta_local_c);
    local_d = (int32_t *)(snrt_cluster_base_addrl() + delta_local_d);

    // Transfer data from L3 to L1
    // Using DMA only
    if (snrt_cluster_idx() == 0 && snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, A, a_data_length);
        snrt_dma_start_1d(local_b, B, b_data_length);
        snrt_dma_wait_all();
    }

    // Wait for DMA to finish
    snrt_cluster_hw_barrier();

    if (snrt_cluster_idx() == 0 && snrt_is_dm_core()) {
        snrt_dma_start_1d(local_c, C, c_data_length);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Set the CSR for the Streamer
    int32_t Aslstride[] = {Aslstride0};
    int32_t Atlbound[] = {Atlbound0, Atlbound1, Atlbound2,
                          Atlbound3, Atlbound4, Atlbound5};
    int32_t Atlstride[] = {Atlstride0, Atlstride1, Atlstride2,
                           Atlstride3, Atlstride4, Atlstride5};
    int32_t Bslstride[] = {Bslstride0};
    int32_t Btlbound[] = {Btlbound0, Btlbound1, Btlbound2};
    int32_t Btlstride[] = {Btlstride0, Btlstride1, Btlstride2};

    int32_t Cslstride[] = {Cslstride0};
    int32_t Ctlbound[] = {Ctlbound0, Ctlbound1, Ctlbound2, Ctlbound3};
    int32_t Ctlbound_accum[] = {0, 0, 0, 0};
    int32_t Ctlstride[] = {Ctlstride0, Ctlstride1, Ctlstride2, Ctlstride3};

    int32_t D32slstride[] = {D32slstride0};
    int32_t D32tlbound[] = {D32tlbound0, D32tlbound1, D32tlbound2, D32tlbound3};
    int32_t D32tlstride[] = {D32tlstride0, D32tlstride1, D32tlstride2,
                             D32tlstride3};

    // Call compute core
    if (snrt_cluster_idx() == 0 && snrt_global_core_idx() == 0) {
        uint32_t subtraction_setting =
            gen_subtraction_config(subtraction_a, subtraction_b);
        uint32_t output_tile_elements = d_tile_data_length / sizeof(int32_t);
        int32_t phase1_cycles = 0;
        int32_t phase1_streamer_cycles = 0;
        int32_t phase2_cycles = 0;
        int32_t phase2_streamer_cycles = 0;
        int phase1_err = 0;
        int phase2_err = 0;

        for (uint32_t m2 = 0; m2 < M2; m2++) {
            for (uint32_t n2 = 0; n2 < N2; n2++) {
                uint32_t tile_id = m2 * N2 + n2;
                uint32_t a_addr = delta_local_a + m2 * a_sw_tile_data_length;
                uint32_t b_addr = delta_local_b + n2 * b_k_tile_data_length;
                uint32_t c_addr = delta_local_c + tile_id * c_tile_data_length;
                uint32_t d_addr = delta_local_d + tile_id * d_tile_data_length;
                int32_t *local_tile_d =
                    (int32_t *)(snrt_cluster_base_addrl() + d_addr);

                // Phase 1: D1[m2,n2] = A[m2]*B[n2] + C[m2,n2].
                set_versacore_streamer_csr(
                    a_addr, Aslstride, Atlbound, Atlstride,
                    set_addr_remap_index_A, transposed_A, channel_en_A,

                    b_addr, Bslstride, Btlbound, Btlstride,
                    set_addr_remap_index_B, transposed_B, channel_en_B,

                    c_addr, Cslstride, Ctlbound, Ctlstride,
                    set_addr_remap_index_C, channel_en_C,

                    d_addr, D32slstride, D32tlbound, D32tlstride,
                    set_addr_remap_index_D32, channel_en_D, array_shape,

                    quantization_enable, shift_i, multiplier_i, input_zp_i,
                    output_zp_i, int32tofp16_enable, int4_a_enable,
                    int4_b_enable);

                if (stationary == 0) {
                    set_versacore_csr(1, K, N * M, subtraction_setting,
                                      array_shape, data_type);
                } else {
                    set_versacore_csr(1, 1, N * K * M, subtraction_setting,
                                      array_shape, data_type);
                }

                start_streamer();
                start_versacore();
                wait_versacore_and_streamer();

                int tile_phase1_err = check_versacore_result_D32(
                    (int8_t *)local_tile_d,
                    (int8_t *)&D1[tile_id * output_tile_elements],
                    d_tile_data_length, false);
                if (tile_phase1_err) {
                    printf("Phase 1 tile check failed: m2 = %d, n2 = %d, "
                           "tile_id = %d, Error: %d.\n",
                           m2, n2, tile_id, tile_phase1_err);
                }
                phase1_err += tile_phase1_err;
                err += tile_phase1_err;
                phase1_cycles += read_versacore_perf_counter();
                phase1_streamer_cycles +=
                    read_versacore_streamer_perf_counter();

                // Phase 2: D2[m2,n2] = A[m2]*B[n2] + D1[m2,n2].
                set_versacore_streamer_csr(
                    a_addr, Aslstride, Atlbound, Atlstride,
                    set_addr_remap_index_A, transposed_A, channel_en_A,

                    b_addr, Bslstride, Btlbound, Btlstride,
                    set_addr_remap_index_B, transposed_B, channel_en_B,

                    c_addr, Cslstride, Ctlbound_accum, Ctlstride,
                    set_addr_remap_index_C, channel_en_C_zero,

                    d_addr, D32slstride, D32tlbound, D32tlstride,
                    set_addr_remap_index_D32, channel_en_D, array_shape,

                    quantization_enable, shift_i, multiplier_i, input_zp_i,
                    output_zp_i, int32tofp16_enable, int4_a_enable,
                    int4_b_enable);

                if (stationary == 0) {
                    set_versacore_csr(0, K, N * M, subtraction_setting,
                                      array_shape, data_type);
                } else {
                    set_versacore_csr(0, 1, N * K * M, subtraction_setting,
                                      array_shape, data_type);
                }

                start_streamer();
                start_versacore();
                wait_versacore_and_streamer();

                int tile_phase2_err = check_versacore_result_D32(
                    (int8_t *)local_tile_d,
                    (int8_t *)&D2[tile_id * output_tile_elements],
                    d_tile_data_length, false);
                if (tile_phase2_err) {
                    printf("Phase 2 tile check failed: m2 = %d, n2 = %d, "
                           "tile_id = %d, Error: %d.\n",
                           m2, n2, tile_id, tile_phase2_err);
                }
                phase2_err += tile_phase2_err;
                err += tile_phase2_err;
                phase2_cycles += read_versacore_perf_counter();
                phase2_streamer_cycles +=
                    read_versacore_streamer_perf_counter();
            }
        }

        printf(
            "Array shape: %d, meshRow %d, tileSize %d, meshCol %d, stationary: "
            "%d, software M2: %d, N2: %d, SNAX GEMM Accum Phase 1: %s, "
            "Error: %d.\n",
            array_shape, meshRow, tileSize, meshCol, stationary, M2, N2,
            phase1_err ? "FAIL" : "PASS", phase1_err);

        printf(
            "Array shape: %d, meshRow %d, tileSize %d, meshCol %d, stationary: "
            "%d, software M2: %d, N2: %d, SNAX GEMM Accum Phase 2: %s, "
            "Error: %d.\n",
            array_shape, meshRow, tileSize, meshCol, stationary, M2, N2,
            phase2_err ? "FAIL" : "PASS", phase2_err);

        printf("Workload size per VersaCore call: M = %d, N = %d, K = %d\n", M,
               N, K);
        printf("Software tile loop: M2 = %d, N2 = %d\n", M2, N2);
        printf("SNAX GEMM Accum Ideal cycles per phase total: %d\n",
               M2 * N2 * M * K * N);
        printf("SNAX GEMM Accum Phase 1 total cycles: %d\n", phase1_cycles);
        printf("SNAX GEMM Accum Phase 1 Streamer cycles: %d\n",
               phase1_streamer_cycles);
        printf("SNAX GEMM Accum Phase 2 total cycles: %d\n", phase2_cycles);
        printf("SNAX GEMM Accum Phase 2 Streamer cycles: %d\n",
               phase2_streamer_cycles);
    };

    return err;
}
