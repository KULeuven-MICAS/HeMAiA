// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "data.h"
#include "snax_versacore_lib.h"

// Chip IDs encode (x << 4) | y: an EAST/WEST pair is 00 <-> 10.
#define SOURCE_CHIP 0x00u
#define PEER_CHIP 0x10u
#define EMPTY 0x13579bdfu
#define READY 0x52454144u
#define B_SENT 0x4253454eu
#define B_ACK 0x4241434bu
#define DONE_PASS 0x600d0000u
#define DONE_FAIL 0x600d0001u

// The UART-loaded image, not main(), initializes these mailboxes. A faster peer
// may already have written READY before this FPGA enters main. Keep them in
// PROGBITS .data and never clear them at runtime. Reload BOTH images before rerun.
static volatile struct {
    uint32_t ready;
    uint32_t b_sent;
    uint32_t b_ack;
    uint32_t done;
} mailbox __attribute__((section(".data.fpga_mailbox"), aligned(64))) = {
    EMPTY, EMPTY, EMPTY, EMPTY};

// Inputs, reference and outputs live in each FPGA's L3. Only working tiles and
// one DMA notification word reside in cluster 0's TCDM. Nonzero sentinels ensure
// these receive/output buffers are also physically present in the UART image.
static volatile int8_t b_received_l3[FPGA_GEMM_B_BYTES]
    __attribute__((section(".data.fpga_buffers"), aligned(64))) = {
        [0 ... FPGA_GEMM_B_BYTES - 1] = -128};
static volatile int32_t d_l3[FPGA_GEMM_D_BYTES / sizeof(int32_t)]
    __attribute__((section(".data.fpga_buffers"), aligned(64))) = {
        [0 ... FPGA_GEMM_D_BYTES / sizeof(int32_t) - 1] = 0x55a55a55};

_Static_assert(FPGA_GEMM_L1_BYTES <= SNRT_TCDM_SIZE / 2,
               "GEMM buffers must leave room for runtime state and core stacks");
_Static_assert(sizeof(a_spatial) / 4 == S_STRIDE_NUM_READER_0,
               "regenerate data.h for the selected cluster");
_Static_assert(sizeof(b_spatial) / 4 == S_STRIDE_NUM_READER_1,
               "regenerate data.h for the selected cluster");
_Static_assert(sizeof(c_spatial) / 4 == S_STRIDE_NUM_READER_WRITER_0,
               "regenerate data.h for the selected cluster");
_Static_assert(sizeof(d_spatial) / 4 == S_STRIDE_NUM_READER_WRITER_1,
               "regenerate data.h for the selected cluster");

static inline uint64_t local_addr(const volatile void *ptr) {
    return chiplet_addr_transform((uint64_t)(uintptr_t)ptr);
}

static inline uint64_t peer_addr(uint8_t peer, const volatile void *ptr) {
    return chiplet_addr_transform_full(peer, (uint64_t)(uintptr_t)ptr);
}

// Only cluster 0's DMA core calls this. A full-width DMA destination preserves
// the chip prefix; casting a remote address to an RV32 pointer would lose it.
static void notify_peer(uint8_t peer, volatile uint32_t *slot,
                        uint32_t value, volatile uint32_t *tx) {
    *tx = value;
    asm volatile("fence rw, rw" ::: "memory");
    snrt_dma_start_1d_wideptr(peer_addr(peer, slot), local_addr(tx), sizeof(*tx));
    snrt_dma_wait_all();
}

static void wait_value(volatile uint32_t *slot, uint32_t expected) {
    while (*slot != expected) {
        asm volatile("nop" ::: "memory");
    }
    asm volatile("fence rw, rw" ::: "memory");
}

static void run_gemm(void) {
    uint32_t a_bounds[T_BOUND_NUM_READER_0];
    uint32_t a_strides[T_STRIDE_NUM_READER_0] = {0};
    uint32_t b_bounds[T_BOUND_NUM_READER_1];
    uint32_t b_strides[T_STRIDE_NUM_READER_1] = {0};
    uint32_t c_bounds[T_BOUND_NUM_READER_WRITER_0];
    uint32_t c_strides[T_STRIDE_NUM_READER_WRITER_0] = {0};
    uint32_t d_bounds[T_BOUND_NUM_READER_WRITER_1];
    uint32_t d_strides[T_STRIDE_NUM_READER_WRITER_1] = {0};
    for (uint32_t i = 0; i < T_BOUND_NUM_READER_0; ++i) a_bounds[i] = 1;
    for (uint32_t i = 0; i < T_BOUND_NUM_READER_1; ++i) b_bounds[i] = 1;
    for (uint32_t i = 0; i < T_BOUND_NUM_READER_WRITER_0; ++i) c_bounds[i] = 1;
    for (uint32_t i = 0; i < T_BOUND_NUM_READER_WRITER_1; ++i) d_bounds[i] = 1;
    a_bounds[0] = b_bounds[0] = FPGA_GEMM_K;
    a_bounds[1] = b_bounds[1] = FPGA_GEMM_N;
    a_bounds[2] = b_bounds[2] = FPGA_GEMM_M;
    a_strides[0] = FPGA_GEMM_A_TILE_BYTES;
    a_strides[2] = FPGA_GEMM_A_TILE_BYTES * FPGA_GEMM_K;
    b_strides[0] = FPGA_GEMM_B_TILE_BYTES;
    b_strides[1] = FPGA_GEMM_B_TILE_BYTES * FPGA_GEMM_K;
    c_bounds[0] = d_bounds[0] = FPGA_GEMM_CD_BEATS;
    c_bounds[1] = d_bounds[1] = FPGA_GEMM_N;
    c_bounds[2] = d_bounds[2] = FPGA_GEMM_M;
    c_strides[0] = d_strides[0] = FPGA_GEMM_CD_BEAT_BYTES;
    c_strides[1] = d_strides[1] = FPGA_GEMM_D_TILE_BYTES;
    c_strides[2] = d_strides[2] = FPGA_GEMM_D_TILE_BYTES * FPGA_GEMM_N;

    set_versacore_streamer_csr(
        FPGA_GEMM_A_OFF, a_spatial, a_bounds, a_strides, 0, 0, a_channels,
        FPGA_GEMM_B_OFF, b_spatial, b_bounds, b_strides, 0, 0, b_channels,
        FPGA_GEMM_D_OFF, c_spatial, c_bounds, c_strides, 0, c_channels,
        FPGA_GEMM_D_OFF, d_spatial, d_bounds, d_strides, 0, d_channels,
        FPGA_GEMM_ARRAY_SHAPE, 0, 0, 0, 0, 0, 0, 0, 0);
    set_versacore_csr(1, FPGA_GEMM_K, FPGA_GEMM_N * FPGA_GEMM_M,
                     0, FPGA_GEMM_ARRAY_SHAPE, 0);
    start_streamer();
    start_versacore();
    wait_versacore_and_streamer();
}

int main(void) {
    uint8_t chip = get_current_chip_id();
    if (chip != SOURCE_CHIP && chip != PEER_CHIP) return 1;
    // Idle clusters join the normal runtime exit reduction after returning.
    // There are no app-level global barriers requiring their participation.
    if (snrt_cluster_idx() != 0) return 0;

    uint8_t peer = chip == SOURCE_CHIP ? PEER_CHIP : SOURCE_CHIP;
    uint32_t block = chip == SOURCE_CHIP ? 0u : 1u;
    uint32_t base = snrt_cluster_base_addrl();
    int8_t *local_a = (int8_t *)(base + FPGA_GEMM_A_OFF);
    int8_t *local_b = (int8_t *)(base + FPGA_GEMM_B_OFF);
    int32_t *local_d = (int32_t *)(base + FPGA_GEMM_D_OFF);
    volatile uint32_t *tx = (volatile uint32_t *)(base + FPGA_GEMM_TX_OFF);
    int errors = 0;

    if (snrt_is_dm_core()) {
        printf("[gemm_fpga] chip %02x ready; M-split rows %u..%u, N=%u K=%u\n",
               chip, block * FPGA_GEMM_M * FPGA_GEMM_MESH_ROW,
               (block + 1) * FPGA_GEMM_M * FPGA_GEMM_MESH_ROW - 1,
               FPGA_GEMM_N * FPGA_GEMM_MESH_COL,
               FPGA_GEMM_K * FPGA_GEMM_TILE_SIZE);
        notify_peer(peer, &mailbox.ready, READY, tx);
        wait_value(&mailbox.ready, READY);

        if (chip == SOURCE_CHIP) {
            // Broadcast shared B from 00's L3 to 10's L3. With two FPGAs there
            // is one receiver; no 0xFF broadcast to absent chips is necessary.
            snrt_dma_start_1d_wideptr(peer_addr(peer, b_received_l3),
                                     local_addr(b_source_l3), FPGA_GEMM_B_BYTES);
            snrt_dma_wait_all();
            notify_peer(peer, &mailbox.b_sent, B_SENT, tx);
            // An observed peer acknowledgement, not local DMA completion,
            // establishes that the remote side received the data.
            wait_value(&mailbox.b_ack, B_ACK);
        } else {
            wait_value(&mailbox.b_sent, B_SENT);
            for (uint32_t i = 0; i < FPGA_GEMM_B_BYTES; ++i) {
                if (b_received_l3[i] != b_source_l3[i]) ++errors;
            }
            printf("[gemm_fpga] chip %02x B broadcast: %s (%u bytes)\n",
                   chip, errors ? "FAIL" : "PASS", FPGA_GEMM_B_BYTES);
            notify_peer(peer, &mailbox.b_ack, B_ACK, tx);
        }

        snrt_dma_start_1d_wideptr(local_addr(local_a),
            local_addr(a_l3 + block * FPGA_GEMM_A_BYTES), FPGA_GEMM_A_BYTES);
        snrt_dma_start_1d_wideptr(local_addr(local_b),
            chip == SOURCE_CHIP ? local_addr(b_source_l3) : local_addr(b_received_l3),
            FPGA_GEMM_B_BYTES);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (snrt_cluster_core_idx() == 0) {
        run_gemm();
        printf("[gemm_fpga] chip %02x GEMM cycles=%u streamer cycles=%u\n",
               chip, read_versacore_perf_counter(),
               read_versacore_streamer_perf_counter());
    }
    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d_wideptr(local_addr(d_l3), local_addr(local_d), FPGA_GEMM_D_BYTES);
        snrt_dma_wait_all();
        asm volatile("fence rw, rw" ::: "memory");
        const int32_t *golden = golden_l3 + block * (FPGA_GEMM_D_BYTES / 4);
        for (uint32_t i = 0; i < FPGA_GEMM_D_BYTES / 4; ++i) {
            if (d_l3[i] != golden[i]) {
                if (errors < 4)
                    printf("[gemm_fpga] chip %02x D[%u]=%d expected=%d\n",
                           chip, i, d_l3[i], golden[i]);
                ++errors;
            }
        }
        printf("[gemm_fpga] chip %02x local L3 D @ 0x%08x: %s, errors=%d\n",
               chip, (uint32_t)(uintptr_t)d_l3, errors ? "FAIL" : "PASS", errors);
        notify_peer(peer, &mailbox.done, errors ? DONE_FAIL : DONE_PASS, tx);
        uint32_t peer_result;
        do {
            peer_result = mailbox.done;
        } while (peer_result != DONE_PASS && peer_result != DONE_FAIL);
        // Return a failure on both hosts if either FPGA detected an error.
        if (peer_result == DONE_FAIL) ++errors;
        printf("[gemm_fpga] chip %02x two-FPGA GEMM: %s\n",
               chip, errors ? "FAIL" : "PASS");
    }
    snrt_cluster_hw_barrier();
    return errors ? 1 : 0;
}
