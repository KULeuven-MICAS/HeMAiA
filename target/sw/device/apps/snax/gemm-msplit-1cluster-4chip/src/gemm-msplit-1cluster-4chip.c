// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// M-split GEMM across 4 compute chips + 1 memory chip, hub-and-spoke variant.
//
// D = A x B is split along M into 4 row-blocks. chip 0x10 ("hub") fetches B and
// all A-blocks from the memory chip and distributes them; the other chips pull
// from the hub. Block -> chip mapping:
//     chip 0x00 -> A0/D0,  chip 0x10 -> A1/D1 (hub), chip 0x20 (memchip)
//     chip 0x01 -> A2/D2,  chip 0x11 -> A3/D3.
//
// Workflow (cross-chip sync via snrt_chip_global_barrier() between steps):
//   1. hub loads B (memchip -> TCDM).                       | snrt_chip_global_barrier()
//   2. workers pull B from hub; hub loads A0..A3 (memchip).  | snrt_chip_global_barrier()
//   3. workers pull their A-block from hub; every chip computes D-block = A x B,
//      self-checks vs golden, and stores its D-block back to the memory chip.
//
// snrt_chip_global_barrier() (configured once by snrt_chip_barrier_init) hides the
// per-chip checkpoint counter and chip-id bookkeeping; it requires the host to have
// brought up D2D and zero-initialised the communication buffer (see
// offload_legacy/multi_chip host).

#include "data.h"

#include "chip_id.h"
#include "snax_versacore_lib.h"

// Memory chip coordinate (see target/rtl/cfg/hemaia_tapeout.hjson).
#define MEM_CHIP_LOC_X 0x2
#define MEM_CHIP_LOC_Y 0x0
// The chip that fetches B and all A-blocks from the memchip and distributes them.
#define HUB_CHIP_ID 0x10
// Barrier rectangle: all four compute chips (0x00 .. 0x11).
#define BARRIER_TOP_LEFT 0x00
#define BARRIER_BOTTOM_RIGHT 0x11

// Map a compute chip id to its A/D row-block index.
static int32_t chip_block_index(uint8_t chip_id) {
    switch (chip_id) {
        case 0x00:
            return 0;
        case 0x10:
            return 1;
        case 0x01:
            return 2;
        case 0x11:
            return 3;
        default:
            return -1;
    }
}

int main() {
    int err = 0;

    uint8_t chip_id = get_current_chip_id();
    int32_t block = chip_block_index(chip_id);
    if (block < 0) {
        return -1;  // not a compute chip (all cores agree here)
    }
    int is_hub = (chip_id == HUB_CHIP_ID);

    // Configure the cross-chip barrier once (rectangle + hidden checkpoint reset);
    // every later snrt_chip_global_barrier() is argument-free. All cores call it.
    // The comm buffer is fetched internally, so the app keeps no `cb` pointer.
    snrt_chip_barrier_init(BARRIER_TOP_LEFT, BARRIER_BOTTOM_RIGHT);

    // TCDM layout (identical on every chip). The hub additionally stages all four
    // A-blocks contiguously just after the D region so the workers can pull them.
    uint32_t base = snrt_cluster_base_addrl();
    uint32_t a_stage_off =
        ((uint32_t)delta_local_d + (uint32_t)d_data_length + 63u) & ~63u;

    int8_t* local_a = (int8_t*)(base + delta_local_a);  // worker compute A-block
    int8_t* local_b = (int8_t*)(base + delta_local_b);
    int32_t* local_d = (int32_t*)(base + delta_local_d);

    // Memory-chip addresses (full 48-bit, memchip prefix).
    uint64_t mem_b = chiplet_addr_transform_loc(MEM_CHIP_LOC_X, MEM_CHIP_LOC_Y,
                                                (uint64_t)mem_chip_local_base +
                                                    mem_off_B);
    uint64_t mem_a_all = chiplet_addr_transform_loc(
        MEM_CHIP_LOC_X, MEM_CHIP_LOC_Y,
        (uint64_t)mem_chip_local_base + mem_off_A_base);
    uint64_t mem_d = chiplet_addr_transform_loc(
        MEM_CHIP_LOC_X, MEM_CHIP_LOC_Y,
        (uint64_t)mem_chip_local_base + mem_off_D_base +
            (uint64_t)block * (uint32_t)d_data_length);

    // Hub TCDM addresses that the workers read from (hub prefix + same local base).
    uint64_t hub_b =
        chiplet_addr_transform_full(HUB_CHIP_ID, (uint64_t)(base + delta_local_b));
    uint64_t hub_a = chiplet_addr_transform_full(
        HUB_CHIP_ID,
        (uint64_t)(base + a_stage_off + (uint32_t)block * (uint32_t)a_data_length));

    // ---- Step 1: hub loads the shared B from the memory chip ----
    if (snrt_is_dm_core() && is_hub) {
        snrt_dma_start_1d_wideptr(
            chiplet_addr_transform((uint64_t)(uintptr_t)local_b), mem_b,
            b_data_length);
        snrt_dma_wait_all();
    }
    snrt_chip_global_barrier();  // cross-chip + intra-chip sync (checkpoint hidden)


    // ---- Step 2: workers pull B from hub; hub loads all A-blocks from memchip --
    if (snrt_is_dm_core()) {
        if (is_hub) {
            snrt_dma_start_1d_wideptr(
                chiplet_addr_transform((uint64_t)(base + a_stage_off)), mem_a_all,
                (uint32_t)num_chips * (uint32_t)a_data_length);
        } else {
            snrt_dma_start_1d_wideptr(
                chiplet_addr_transform((uint64_t)(uintptr_t)local_b), hub_b,
                b_data_length);
        }
        snrt_dma_wait_all();
    }
    snrt_chip_global_barrier();  // cross-chip + intra-chip sync (checkpoint hidden)

    // ---- Step 3a: workers pull their A-block from the hub staging area ----
    if (snrt_is_dm_core() && !is_hub) {
        snrt_dma_start_1d_wideptr(
            chiplet_addr_transform((uint64_t)(uintptr_t)local_a), hub_a,
            a_data_length);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    // ---- Step 3b: compute D-block = A-block x B (output-stationary) ----
    // The hub computes directly out of its staging area; workers from local_a.
    uint32_t a_compute =
        is_hub ? (a_stage_off + (uint32_t)block * (uint32_t)a_data_length)
               : (uint32_t)delta_local_a;

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

    if (snrt_global_core_idx() == 0) {
        set_versacore_streamer_csr(
            a_compute, Aslstride, Atlbound, Atlstride, set_addr_remap_index_A,
            transposed_A, channel_en_A,

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

        int32_t* golden =
            (int32_t*)D + (uint32_t)block * (M_block * N * meshRow * meshCol);
        err += check_versacore_result_D32((int32_t*)local_d, (int32_t*)golden,
                                          d_data_length, false);

        printf("Chip(%x,%x) block %d (%s): gemm-msplit-1cluster-4chip %s, err=%d\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), block,
               is_hub ? "HUB" : "WORKER", err ? "FAIL" : "PASS", err);
    }
    snrt_cluster_hw_barrier();

    // ---- Step 3c: store the D-block back to the memory chip ----
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d_wideptr(
            mem_d, chiplet_addr_transform((uint64_t)(uintptr_t)local_d),
            d_data_length);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    return err;
}
