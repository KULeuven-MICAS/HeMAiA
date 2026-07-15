// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>
//
// M-split GEMM across 4 clusters in 1 compute chip + 1 memory chip.
//
// D = A x B is split along M into 4 row-blocks. Each cluster computes one
// D-block, and all clusters share the same B-block.
//
// Every active cluster DM core fetches its A-block and the shared B-block
// directly from the memory chip into local TCDM. Block -> cluster mapping:
//     cluster 0 -> A0/D0, cluster 1 -> A1/D1, chip 0x10 -> memory chip
//     cluster 2 -> A2/D2, cluster 3 -> A3/D3.
//
// Workflow (cross-cluster sync via snrt_global_barrier() between steps):
//   1. clusters load A-block and B (memchip -> cluster TCDM).
//   2. clusters compute D-block = A x B.
//   3. clusters store D-block back to memchip.
//   4. clusters self-check local TCDM output against the golden output.
//

#include "data.h"

#include "chip_id.h"
#include "gemm_msplit_trace.h"
#include "snax_versacore_lib.h"

// Compute/memory chip coordinates (see target/rtl/cfg/hemaia_mirror_tapeout_4cluster.hjson).
#define COMPUTE_CHIP_ID 0x00
#define MEM_CHIP_LOC_X 0x1
#define MEM_CHIP_LOC_Y 0x0

#ifndef GEMM_MSPLIT_CHECK_RESULT
#define GEMM_MSPLIT_CHECK_RESULT 1
#endif

#ifndef GEMM_MSPLIT_ENABLE_DEBUG_PRINT
#define GEMM_MSPLIT_ENABLE_DEBUG_PRINT 0
#endif

int main() {
    int err = 0;

    uint8_t chip_id = get_current_chip_id();
    uint32_t cluster_id = snrt_cluster_idx();
    uint32_t block = cluster_id;
    int is_active_cluster = (cluster_id < num_clusters);

    if (chip_id != COMPUTE_CHIP_ID) {
        return 0;
    }

    // TCDM layout is identical in every cluster.
    uint32_t base = snrt_cluster_base_addrl();
    int8_t* local_a = (int8_t*)(base + delta_local_a);
    int8_t* local_b = (int8_t*)(base + delta_local_b);
    int32_t* local_d = (int32_t*)(base + delta_local_d);

    // Memory-chip addresses (full 48-bit, memory-chip prefix).
    uint64_t mem_a_all = chiplet_addr_transform_loc(
        MEM_CHIP_LOC_X, MEM_CHIP_LOC_Y,
        (uint64_t)mem_chip_local_base + mem_off_A_base);
    uint64_t mem_a_block =
        mem_a_all + (uint64_t)block * (uint32_t)a_data_length;
    uint64_t mem_b = chiplet_addr_transform_loc(
        MEM_CHIP_LOC_X, MEM_CHIP_LOC_Y,
        (uint64_t)mem_chip_local_base + mem_off_B);
    uint64_t mem_d = chiplet_addr_transform_loc(
        MEM_CHIP_LOC_X, MEM_CHIP_LOC_Y,
        (uint64_t)mem_chip_local_base + mem_off_D_base +
            (uint64_t)block * (uint32_t)d_data_length);

    uint32_t cycles = snrt_mcycle();

    // ---- Step 1: each cluster loads A-block and B from memchip into TCDM ----
    if (is_active_cluster && snrt_is_dm_core()) {
        GEMM_MSPLIT_TRACE_MARKER(BINGO_TRACE_LOAD_A_TCDM_START);
        snrt_dma_start_1d_wideptr(
            chiplet_addr_transform((uint64_t)(uintptr_t)local_a), mem_a_block,
            a_data_length);
        snrt_dma_wait_all();
        GEMM_MSPLIT_TRACE_MARKER(BINGO_TRACE_LOAD_A_TCDM_END);
#if GEMM_MSPLIT_ENABLE_DEBUG_PRINT
        printf("Chip(%x,%x) cluster %u block %u: loaded A-block from memchip into local TCDM\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), cluster_id,
               block);
#endif

        GEMM_MSPLIT_TRACE_MARKER(BINGO_TRACE_LOAD_B_TCDM_START);
        snrt_dma_start_1d_wideptr(
            chiplet_addr_transform((uint64_t)(uintptr_t)local_b), mem_b,
            b_data_length);
        snrt_dma_wait_all();
        GEMM_MSPLIT_TRACE_MARKER(BINGO_TRACE_LOAD_B_TCDM_END);
#if GEMM_MSPLIT_ENABLE_DEBUG_PRINT
        printf("Chip(%x,%x) cluster %u block %u: loaded B-block from memchip into local TCDM\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), cluster_id,
               block);
#endif
    }
    GEMM_MSPLIT_TRACE_MARKER(BINGO_TRACE_GLOBAL_SYNC_START);
    snrt_global_barrier();
    GEMM_MSPLIT_TRACE_MARKER(BINGO_TRACE_GLOBAL_SYNC_END);

    // ---- Step 2: compute D-block = A-block x B (output-stationary) ----
    if (is_active_cluster && snrt_cluster_core_idx() == 0) {
        GEMM_MSPLIT_TRACE_MARKER(BINGO_TRACE_LOAD_CFG_START);
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
        GEMM_MSPLIT_TRACE_MARKER(BINGO_TRACE_LOAD_CFG_END);

        GEMM_MSPLIT_TRACE_MARKER(BINGO_TRACE_PROGRAM_CSR_START);
        set_versacore_streamer_csr(
            delta_local_a, Aslstride, Atlbound, Atlstride,
            set_addr_remap_index_A, transposed_A, channel_en_A,

            delta_local_b, Bslstride, Btlbound, Btlstride, set_addr_remap_index_B,
            transposed_B, channel_en_B,

            delta_local_c, Cslstride, Ctlbound, Ctlstride, set_addr_remap_index_C,
            channel_en_C,

            delta_local_d, D32slstride, D32tlbound, D32tlstride,
            set_addr_remap_index_D32, channel_en_D, array_shape,

            quantization_enable, shift_i, multiplier_i, input_zp_i, output_zp_i,
            int32tofp16_enable, int4_a_enable, int4_b_enable);

        uint32_t subtraction_setting =
            gen_subtraction_config(subtraction_a, subtraction_b);

        set_versacore_csr(1, K, N * M_block, subtraction_setting, array_shape,
                          data_type);

        start_streamer();
        start_versacore();
        GEMM_MSPLIT_TRACE_MARKER(BINGO_TRACE_PROGRAM_CSR_END);
        GEMM_MSPLIT_TRACE_MARKER(BINGO_TRACE_COMPUTE_START);
        wait_versacore_and_streamer();
        GEMM_MSPLIT_TRACE_MARKER(BINGO_TRACE_COMPUTE_END);
    }

    // sync between cluster cores to ensure all cores have finished computing before copying data back to memory chip
    GEMM_MSPLIT_TRACE_MARKER(BINGO_TRACE_CLUSTER_SYNC_START);
    snrt_cluster_hw_barrier();
    GEMM_MSPLIT_TRACE_MARKER(BINGO_TRACE_CLUSTER_SYNC_END);

    // ---- Step 3: store the D-block back to the memory chip ----
    if (is_active_cluster && snrt_is_dm_core()) {
        GEMM_MSPLIT_TRACE_MARKER(BINGO_TRACE_STORE_D_START);
        snrt_dma_start_1d_wideptr(
            mem_d, chiplet_addr_transform((uint64_t)(uintptr_t)local_d),
            d_data_length);
        snrt_dma_wait_all();
        GEMM_MSPLIT_TRACE_MARKER(BINGO_TRACE_STORE_D_END);
    }

    // Sync between clusters to ensure the finish of the layer
    GEMM_MSPLIT_TRACE_MARKER(BINGO_TRACE_GLOBAL_SYNC_START);
    snrt_global_barrier();
    GEMM_MSPLIT_TRACE_MARKER(BINGO_TRACE_GLOBAL_SYNC_END);

    cycles = snrt_mcycle() - cycles;

    if (snrt_cluster_idx() == 0 && snrt_cluster_core_idx() == 0) {
        printf("Chip(%x,%x) cluster %u block %u: gemm-msplit-4cluster-1chip cycles = %u\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), cluster_id, block, cycles);
    }

#if GEMM_MSPLIT_CHECK_RESULT
    // Check each cluster's D-block in parallel. Keep this silent so only the
    // summary prints below need serialization.
    //
    // The golden D lives on the memory chip (mem_off_D_golden), not baked into
    // the device binary. The DM core DMAs this cluster's golden D-block into the
    // TCDM region freed after the compute (the A/B staging area at
    // delta_local_a, sized >= one D-block by the datagen), then core 0 compares
    // it against the computed D-block.
    uint64_t mem_d_golden = chiplet_addr_transform_loc(
        MEM_CHIP_LOC_X, MEM_CHIP_LOC_Y,
        (uint64_t)mem_chip_local_base + mem_off_D_golden +
            (uint64_t)block * (uint32_t)d_data_length);
    int8_t* golden = (int8_t*)(base + delta_local_a);
    if (is_active_cluster && snrt_is_dm_core()) {
        snrt_dma_start_1d_wideptr(
            chiplet_addr_transform((uint64_t)(uintptr_t)golden), mem_d_golden,
            d_data_length);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();
    if (is_active_cluster && snrt_cluster_core_idx() == 0) {
        int8_t* output = (int8_t*)local_d;
        for (int32_t i = 0; i < d_data_length; i++) {
            if (output[i] != golden[i]) {
                err++;
            }
        }
    }
#endif

    snrt_global_barrier();

    // Serialize per-cluster status prints so the output stays readable.
    for (uint32_t print_cluster = 0; print_cluster < num_clusters;
         print_cluster++) {
        if (is_active_cluster && cluster_id == print_cluster &&
            snrt_cluster_core_idx() == 0) {
            printf("Chip(%x,%x) cluster %u block %u: gemm-msplit-4cluster-1chip finish. \n",
                   get_current_chip_loc_x(), get_current_chip_loc_y(), cluster_id,
                   block);

#if GEMM_MSPLIT_CHECK_RESULT
            printf("Chip(%x,%x) cluster %u block %u: gemm-msplit-4cluster-1chip %s, err=%d\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y(), cluster_id,
                   block, err ? "FAIL" : "PASS", err);
#else
            printf("Chip(%x,%x) cluster %u block %u: gemm-msplit-4cluster-1chip result check skipped\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y(), cluster_id,
                   block);
#endif
        }
        snrt_global_barrier();
    }

    return err;
}
