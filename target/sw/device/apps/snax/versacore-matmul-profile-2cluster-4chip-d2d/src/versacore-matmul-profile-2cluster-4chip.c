// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>
//
// Memory-chip/D2D variant of the two-cluster VersaCore profile. Each active
// cluster first stages A, B, and C from the memory chip into its local TCDM,
// then runs the original profile workload.

#include "data.h"

#include "chip_id.h"
#include "snax_versacore_lib.h"

// Memory-chip coordinate in the 4-compute-chip tapeout configurations.
#define MEM_CHIP_LOC_X 0x2
#define MEM_CHIP_LOC_Y 0x0

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

    // Full 48-bit memory-chip addresses. The generated mempool.bin is loaded at
    // mem_chip_local_base and contains the pre-streamer representations of all
    // three inputs.
    uint64_t mem_a = chiplet_addr_transform_loc(
        MEM_CHIP_LOC_X, MEM_CHIP_LOC_Y,
        (uint64_t)mem_chip_local_base + mem_off_A);
    uint64_t mem_b = chiplet_addr_transform_loc(
        MEM_CHIP_LOC_X, MEM_CHIP_LOC_Y,
        (uint64_t)mem_chip_local_base + mem_off_B);
    uint64_t mem_c = chiplet_addr_transform_loc(
        MEM_CHIP_LOC_X, MEM_CHIP_LOC_Y,
        (uint64_t)mem_chip_local_base + mem_off_C);

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
    int32_t Ctlstride[] = {Ctlstride0, Ctlstride1, Ctlstride2, Ctlstride3};

    int32_t D32slstride[] = {D32slstride0};
    int32_t D32tlbound[] = {D32tlbound0, D32tlbound1, D32tlbound2, D32tlbound3};
    int32_t D32tlstride[] = {D32tlstride0, D32tlstride1, D32tlstride2,
                             D32tlstride3};

    // Run one cluster at a time: cluster 0 first, then cluster 1, etc.
    for (uint32_t active_cluster = 0; active_cluster < snrt_cluster_num();
         active_cluster++) {
        if (snrt_cluster_idx() == active_cluster) {
            uint32_t end_to_end_start = snrt_mcycle();

            // Transfer A and B from the memory chip to this cluster's L1/TCDM.
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d_wideptr(
                    chiplet_addr_transform((uint64_t)(uintptr_t)local_a), mem_a,
                    a_data_length);
                snrt_dma_start_1d_wideptr(
                    chiplet_addr_transform((uint64_t)(uintptr_t)local_b), mem_b,
                    b_data_length);
                snrt_dma_wait_all();
            }

            // Wait for DMA to finish.
            snrt_cluster_hw_barrier();

            // C shares its TCDM base with D, so load it after the first pair of
            // transfers and before the accelerator starts writing D.
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d_wideptr(
                    chiplet_addr_transform((uint64_t)(uintptr_t)local_c), mem_c,
                    c_data_length);
                snrt_dma_wait_all();
            }

            snrt_cluster_hw_barrier();

            // Launch compute on cluster-local core 0 only.
            if (snrt_cluster_core_idx() == 0) {
                uint32_t mem_to_l1_cycles =
                    snrt_mcycle() - end_to_end_start;

                // Set Streamer configuration CSR
                set_versacore_streamer_csr(
                    delta_local_a, Aslstride, Atlbound, Atlstride,
                    set_addr_remap_index_A, transposed_A, channel_en_A,

                    delta_local_b, Bslstride, Btlbound, Btlstride,
                    set_addr_remap_index_B, transposed_B, channel_en_B,

                    delta_local_c, Cslstride, Ctlbound, Ctlstride,
                    set_addr_remap_index_C, channel_en_C,

                    delta_local_d, D32slstride, D32tlbound, D32tlstride,
                    set_addr_remap_index_D32, channel_en_D, array_shape,

                    quantization_enable, shift_i, multiplier_i, input_zp_i,
                    output_zp_i, int32tofp16_enable, int4_a_enable,
                    int4_b_enable);

                // Set GEMMX configuration CSR
                uint32_t subtraction_setting =
                    gen_subtraction_config(subtraction_a, subtraction_b);

                if (stationary == 0) {
                    // Set CSR for output-stationary
                    set_versacore_csr(1, K, N * M, subtraction_setting,
                                      array_shape, data_type);
                } else {
                    // Set CSR for weight-stationary or input-stationary
                    set_versacore_csr(1, 1, N * K * M, subtraction_setting,
                                      array_shape, data_type);
                }

                // Set CSR to start Streamer
                start_streamer();

                // Set CSR to start GEMM
                start_versacore();

                // Poll until Streamer and GEMM accelerator finish
                wait_versacore_and_streamer();
                uint32_t end_to_end_cycles =
                    snrt_mcycle() - end_to_end_start;

                printf(
                    "Array shape: %d, meshRow %d, tileSize %d, meshCol %d, "
                    "stationary: %d, SNAX GEMM Matmul: memory-chip -> L1 -> "
                    "compute.\n",
                    array_shape, meshRow, tileSize, meshCol, stationary);

                // Result check
                if (quantization_enable == 0 && int32tofp16_enable == 0)
                    err += check_versacore_result_D32(
                        (int32_t *)local_d, (int32_t *)D, d_data_length, false);
                else if (quantization_enable == 1 &&
                         int32tofp16_enable == 0) {
                    err += check_versacore_result_D32(
                        (int8_t *)local_d, (int8_t *)D_quantized,
                        d_data_length, false);
                } else if (int32tofp16_enable == 1) {
                    err += check_versacore_result_D32(
                        (int8_t *)local_d, (int8_t *)D_int32tofp16,
                        d_data_length, false);
                }

                printf(
                    "Cluster %d Array shape: %d, meshRow %d, tileSize %d, "
                    "meshCol %d, stationary: %d, SNAX GEMM Matmul: %s, "
                    "Error: %d.\n",
                    snrt_cluster_idx(), array_shape, meshRow, tileSize,
                    meshCol, stationary, err ? "FAIL" : "PASS", err);

                int32_t gemmx_cycles = read_versacore_perf_counter();
                int32_t gemmx_streamer_cycles =
                    read_versacore_streamer_perf_counter();
                printf("Cluster %d Workload size: M = %d, N = %d, K = %d\n",
                       snrt_cluster_idx(), M, N, K);
                printf("Cluster %d SNAX GEMM Ideal cycles: %d\n",
                       snrt_cluster_idx(), M * K * N);
                printf("Cluster %d SNAX GEMM cycles: %d\n",
                       snrt_cluster_idx(), gemmx_cycles);
                printf("Cluster %d SNAX GEMM Streamer cycles: %d\n",
                       snrt_cluster_idx(), gemmx_streamer_cycles);
                printf("Cluster %d memory-chip to L1 cycles: %u\n",
                       snrt_cluster_idx(), mem_to_l1_cycles);
                printf("Cluster %d memory-chip load + compute cycles: %u\n",
                       snrt_cluster_idx(), end_to_end_cycles);
            }
        }

        // Do not let the next cluster start until the current one is done.
        snrt_global_barrier();
    }

    return err;
}
