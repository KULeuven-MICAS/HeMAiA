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
// Cluster 0 on compute chip 0x00 fetches all A-blocks and B from the memory
// chip into the compute chip's L3. Every cluster then pulls the data it needs
// from L3 into its own TCDM. Block -> cluster mapping:
//     cluster 0 -> A0/D0, cluster 1 -> A1/D1, chip 0x10 -> memory chip
//     cluster 2 -> A2/D2, cluster 3 -> A3/D3.
//
// Workflow (cross-cluster sync via snrt_global_barrier() between steps):
//   1. cluster 0 loads A0..A3 (memchip -> L3).            | snrt_global_barrier()
//   2. clusters pull their A-block (L3 -> cluster TCDM).  | snrt_global_barrier()
//   3. cluster 0 loads B (memchip -> L3).                 | snrt_global_barrier()
//   4. clusters pull B (L3 -> cluster TCDM), compute D-block = A x B,
//      self-check against the golden output, and store D-block back to memchip.
//

#include "data.h"

#include "chip_id.h"
#include "snax_versacore_lib.h"

// Compute/memory chip coordinates (see target/rtl/cfg/hemaia_mirror_tapeout_4cluster.hjson).
#define COMPUTE_CHIP_ID 0x00
#define MEM_CHIP_LOC_X 0x1
#define MEM_CHIP_LOC_Y 0x0
#define HUB_CLUSTER_ID 0

int main() {
    int err = 0;

    uint8_t chip_id = get_current_chip_id();
    uint32_t cluster_id = snrt_cluster_idx();
    uint32_t block = cluster_id;
    int is_active_cluster = (cluster_id < num_clusters);
    int is_hub_cluster = (cluster_id == HUB_CLUSTER_ID);

    if (chip_id != COMPUTE_CHIP_ID) {
        return 0;
    }

    // TCDM layout is identical in every cluster. L3 staging reuses the generated
    // A and B arrays; cluster 0 overwrites them with the memory-chip contents.
    uint32_t base = snrt_cluster_base_addrl();
    int8_t* local_a = (int8_t*)(base + delta_local_a);
    int8_t* local_b = (int8_t*)(base + delta_local_b);
    int32_t* local_d = (int32_t*)(base + delta_local_d);

    // L3 staging addresses on the compute chip.
    uint64_t l3_a_all = chiplet_addr_transform((uint64_t)(uintptr_t)A);
    uint64_t l3_b = chiplet_addr_transform((uint64_t)(uintptr_t)B);
    uint64_t l3_a_block =
        l3_a_all + (uint64_t)block * (uint32_t)a_data_length;

    // Memory-chip addresses (full 48-bit, memory-chip prefix).
    uint64_t mem_a_all = chiplet_addr_transform_loc(
        MEM_CHIP_LOC_X, MEM_CHIP_LOC_Y,
        (uint64_t)mem_chip_local_base + mem_off_A_base);
    uint64_t mem_b = chiplet_addr_transform_loc(
        MEM_CHIP_LOC_X, MEM_CHIP_LOC_Y,
        (uint64_t)mem_chip_local_base + mem_off_B);
    uint64_t mem_d = chiplet_addr_transform_loc(
        MEM_CHIP_LOC_X, MEM_CHIP_LOC_Y,
        (uint64_t)mem_chip_local_base + mem_off_D_base +
            (uint64_t)block * (uint32_t)d_data_length);

    // ---- Step 1: cluster 0 loads all A-blocks from memchip into L3 ----
    if (is_active_cluster && is_hub_cluster && snrt_is_dm_core()) {
        snrt_dma_start_1d_wideptr(
            l3_a_all, mem_a_all,
            (uint32_t)num_clusters * (uint32_t)a_data_length);
        snrt_dma_wait_all();
        printf("Chip(%x,%x) cluster %u block %u: loaded A-blocks from memchip into L3\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), cluster_id,
               block);
    }
    snrt_global_barrier();

    // ---- Step 2: each cluster pulls its A-block from L3 into local TCDM ----
    if (is_active_cluster && snrt_is_dm_core()) {
        snrt_dma_start_1d_wideptr(
            chiplet_addr_transform((uint64_t)(uintptr_t)local_a), l3_a_block,
            a_data_length);
        snrt_dma_wait_all();
        printf("Chip(%x,%x) cluster %u block %u: loaded A-block from L3 into local TCDM\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), cluster_id,
               block);
    }
    snrt_global_barrier();

    // ---- Step 3: cluster 0 loads the shared B-block from memchip into L3 ----
    if (is_active_cluster && is_hub_cluster && snrt_is_dm_core()) {
        snrt_dma_start_1d_wideptr(l3_b, mem_b, b_data_length);
        snrt_dma_wait_all();
        printf("Chip(%x,%x) cluster %u block %u: loaded B-block from memchip into L3\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), cluster_id,
               block);
    }
    snrt_global_barrier();

    // ---- Step 4a: each cluster pulls B from L3 into local TCDM ----
    if (is_active_cluster && snrt_is_dm_core()) {
        snrt_dma_start_1d_wideptr(
            chiplet_addr_transform((uint64_t)(uintptr_t)local_b), l3_b,
            b_data_length);
        snrt_dma_wait_all();
        printf("Chip(%x,%x) cluster %u block %u: loaded B-block from L3 into local TCDM\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), cluster_id,
               block);
    }
    snrt_global_barrier();

    // ---- Step 4b: compute D-block = A-block x B (output-stationary) ----
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

    if (is_active_cluster && snrt_cluster_core_idx() == 0) {
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
        wait_versacore_and_streamer();

        printf("Chip(%x,%x) cluster %u block %u: gemm-msplit-4cluster-1chip finish\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), cluster_id,
               block);

        int32_t* golden =
            (int32_t*)D + block * (M_block * N * meshRow * meshCol);
        err += check_versacore_result_D32((int32_t*)local_d, (int32_t*)golden,
                                          d_data_length, false);

        printf("Chip(%x,%x) cluster %u block %u: gemm-msplit-4cluster-1chip %s, err=%d\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), cluster_id,
               block, err ? "FAIL" : "PASS", err);
    }
    snrt_cluster_hw_barrier();

    // ---- Step 4c: store the D-block back to the memory chip ----
    if (is_active_cluster && snrt_is_dm_core()) {
        snrt_dma_start_1d_wideptr(
            mem_d, chiplet_addr_transform((uint64_t)(uintptr_t)local_d),
            d_data_length);
        snrt_dma_wait_all();
    }
    snrt_global_barrier();

    return err;
}
