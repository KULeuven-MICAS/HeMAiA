// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// smoke test for DIMC cluster & gemmx cluster

// Version 1 function: DIMC loads data to TCDM, GEMMX reads that chunk of data

// Version 2 function: DIMC computes one head, GEMMX reads the result

// Version 3 function: DIMC computes one head, GEMMX reads the result and
// compute MM
#include "snax-dimc-csr.h"
#include "snax-dimc-lib.h"
#include "snrt.h"
// #include "data_mha.h"

#include "data.h"
#include "snax-gemmx-lib.h"
#include "snax-gemmx-params.h"

#include "cfg_gemmx.h"
#include "data_gemmx.h"

uint8_t *tcdm_1_start_addr;  // starting address for GEMMX cluster

int main() {
    // set error value for checking
    int err = 0;

    if (snrt_cluster_idx() == 1) {
        // allocate memory for GEMMX cluster
        tcdm_1_start_addr = (uint8_t *)snrt_cluster_base_addrl();

        // Prepare addresses in TCDM
        int8_t *local_b;
        int32_t *local_d32;

        // Allocate space in TCDM
        local_b = (int8_t *)(snrt_cluster_base_addrl() + delta_local_b);
        local_d32 = (int32_t *)(snrt_cluster_base_addrl() + delta_local_d32);

        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(
                tcdm_1_start_addr, gold,
                gemmx_M * gemmx_K * tileSize * meshRow * sizeof(int8_t));
            snrt_dma_start_1d(
                local_b, B_blocked_col_major,
                gemmx_N * gemmx_K * tileSize * meshCol * sizeof(int8_t));
            snrt_dma_wait_all();
        }

        // Wait for DMA to finish
        snrt_cluster_hw_barrier();

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

            set_gemmx_csr(gemmx_K, gemmx_N, gemmx_M, subtraction_setting, csr0,
                          csr1, shared_bitpacked_shift0,
                          shared_bitpacked_shift1, shared_multiplier0,
                          shared_multiplier1, shared_multiplier2,
                          shared_multiplier3, shared_multiplier4,
                          shared_multiplier5, shared_multiplier6,
                          shared_multiplier7, gemmx_M * gemmx_N, bypassSIMD);
            printf("GEMMX cluster configuration finished\n");

            // Set CSR to start Streamer for conv2d
            set_gemmx_streamer_start();

            // Set CSR to start GEMM
            set_gemmx_start();

            // Poll until Streamer and GEMM accelerator finish
            wait_gemmx_and_streamer();
            printf("GEMMX cluster finished the computation\n");

            // check the final result
            printf("GEMMX cluster starts to check the computed data\n");

            for (int i = 0; i < (gemmx_M * gemmx_N * meshRow * meshCol); i++) {
                if (local_d32[i] != D_golden_blocked_row_major[i]) {
                    err++;
                    printf(
                        "GEMMX cluster failed to verify the computed data at "
                        "index %d, expected %d but got %d\n",
                        i, D_golden_blocked_row_major[i], local_d32[i]);
                }
            }

            if (err == 0) {
                printf(
                    "GEMMX cluster verified the computed data successfully\n");
            } else {
                printf(
                    "GEMMX cluster failed to verify the computed data, err = "
                    "%d\n",
                    err);
            }
        }
    }

    snrt_global_barrier();

    return err;
}
