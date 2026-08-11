// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Cross-DIE (4-chiplet) NONLINEAR ChainGather -- the online-softmax (m, l) moment-merge folded in
// the DMA fabric as the chain crosses the die-to-die links. Each of the four compute chiplets owns
// one (m, l) pair (m at lane 0, l at lane 8; nValid=1); the collector (chip 0x00) gathers the other
// three over D2D with the MonoidJunction MOMENT combine, and the merged (m*, l*) lands locally.
// m* exact; l* uses the writer's exp LUT so it is checked to a fp32 ULP tolerance (DM core is rv32ima).
//
// Two cross-die mechanisms that differ from the single-chip 4-cluster version (see the linear sibling
// snax-xdma-chain-gather-4chip.c for the full rationale):
//   (1) sync via TARGETED cross-chip scalar stores (xchip_store_u32) + local polling -- the
//       snrt_chip_global_barrier() 0xFF broadcast does not route under hemaia_xdma_rr_1c.
//   (2) the gather chain follows physically-ADJACENT D2D hops (snake 0x01->0x11->0x10->0x00); a
//       diagonal hop wedges the backward grant chain and stalls the head forever.

#include "chip_id.h"   // get_current_chip_id, get_chip_baseaddress_value
#include "data.h"
#include "snrt.h"

#define PARTIAL_BYTES (PARTIAL_ELEMS * (int)sizeof(float))
#define COLLECTOR  0x00
#define L_TOL_ULPS 0x20000u
// GATHER_P==2: cross-die P=2 (adjacent source, no middle node -- works today).
// GATHER_P==4: full snake chain (blocked on the middle-node grant RTL bug).
#ifndef GATHER_P
#define GATHER_P 4
#endif

#define READY_OFF 0x400u
#define DONE_OFF  0x480u
#define FLAG_SET  0xA5A5A5A5u

static const uint8_t CHIPS[4]     = {0x00, 0x01, 0x10, 0x11};
static const uint8_t CHAIN_SRC[3] = {0x01, 0x11, 0x10};  // far -> near, all-adjacent D2D hops

static int chip_index(uint8_t c) {
    for (int i = 0; i < 4; i++)
        if (CHIPS[i] == c) return i;
    return -1;
}
static int src_slot(uint8_t c) {
    for (int i = 0; i < 3; i++)
        if (CHAIN_SRC[i] == c) return i;
    return -1;
}

// Targeted cross-chip 32-bit store via the Mseg CSR (0xbc0): aim at one specific chip's prefix
// (chip_id<<8), store to the identical-layout local address, restore Mseg.
static inline void xchip_store_u32(uint8_t target_chip, uint32_t local_addr, uint32_t val) {
    uint32_t tgt_h = (uint32_t)(get_chip_baseaddress_value(target_chip) >> 32);
    uint32_t cur_h = (uint32_t)(get_current_chip_baseaddress_value() >> 32);
    register uint32_t r_h asm("t0") = tgt_h;
    register uint32_t r_v asm("t1") = val;
    register uint32_t r_a asm("t2") = local_addr;
    register uint32_t r_c asm("t3") = cur_h;
    asm volatile(
        "csrw 0xbc0, t0;"
        "sw   t1, 0(t2);"
        "csrw 0xbc0, t3;"
        :
        : "r"(r_h), "r"(r_v), "r"(r_a), "r"(r_c)
        : "memory");
}

int main() {
    uint8_t chip_id = get_current_chip_id();
    int ci = chip_index(chip_id);
    uint32_t tcdm_base = snrt_cluster_base_addrl();
    volatile uint32_t *ready = (volatile uint32_t *)(uintptr_t)(tcdm_base + READY_OFF);
    volatile uint32_t *done  = (volatile uint32_t *)(uintptr_t)(tcdm_base + DONE_OFF);

    if (ci >= 0 && snrt_is_dm_core()) {
        *done = 0;
        if (chip_id == COLLECTOR) { ready[0] = 0; ready[1] = 0; ready[2] = 0; }
    }
    snrt_global_barrier();

    if (ci >= 0 && snrt_is_dm_core()) {
        snrt_dma_start_1d((void *)tcdm_base,
                          &chain_gather_data[ci * PARTIAL_ELEMS], PARTIAL_BYTES);
        snrt_dma_wait_all();
        printf("[DBG chip %x] staged partial\r\n", chip_id);
        int s = src_slot(chip_id);
        if (s >= 0) {
            xchip_store_u32(COLLECTOR, tcdm_base + READY_OFF + (uint32_t)s * 4u, FLAG_SET);
            printf("[DBG chip %x] wrote ready slot %d -> collector\r\n", chip_id, s);
        }
    }

    int err = 0;
    if (chip_id == COLLECTOR && snrt_is_dm_core()) {
        for (int s = 0; s < 3; s++)
            while (ready[s] != FLAG_SET) asm volatile("fence" ::: "memory");
        printf("[DBG chip %x] all sources ready\r\n", chip_id);

        for (uint8_t e = 0; e < XDMA_DST_EXT_NUM; e++) xdma_disable_dst_ext(e);
        for (uint8_t r = 0; r < XDMA_SRC_EXT_NUM; r++) xdma_disable_src_ext(r);
        for (uint8_t j = 0; j < XDMA_DST_JCT_NUM; j++) xdma_disable_dst_junction(j);

        uint64_t local_src = chiplet_addr_transform_full(COLLECTOR, (uint64_t)tcdm_base);
        uint64_t dst_local = chiplet_addr_transform_full(COLLECTOR, (uint64_t)(tcdm_base + PARTIAL_BYTES));
#if GATHER_P == 2
        // P=2 cross-die: collector 0x00 merges ONE adjacent source (0x10). No middle node =>
        // only the working root->head grant. Nonlinear (moment-merge) in-fabric fold, cross-die.
        uint64_t chain[2];
        chain[0] = chiplet_addr_transform_full(0x10, (uint64_t)tcdm_base);  // source/head, adjacent to 0x00
        chain[1] = dst_local;
        uint32_t chain_n = 2;
        uint32_t *gold = (uint32_t *)chain_gather_golden_p2;
#else
        uint64_t chain[4];
        chain[0] = chiplet_addr_transform_full(CHAIN_SRC[0], (uint64_t)tcdm_base);  // 0x01 (head)
        chain[1] = chiplet_addr_transform_full(CHAIN_SRC[1], (uint64_t)tcdm_base);  // 0x11
        chain[2] = chiplet_addr_transform_full(CHAIN_SRC[2], (uint64_t)tcdm_base);  // 0x10 (near dst)
        chain[3] = dst_local;
        uint32_t chain_n = 4;
        uint32_t *gold = (uint32_t *)chain_gather_golden;
#endif

        volatile uint32_t *dbg = (volatile uint32_t *)(uintptr_t)(tcdm_base + PARTIAL_BYTES);
        for (int j = 0; j < PARTIAL_ELEMS; j++) dbg[j] = 0xDEADBEEFu;

        uint32_t jct_csr = (1u << 13) | 1u;  // MonoidJunction MOMENT, nValid=1
        int32_t ret = xdma_chain_gather_1d_full_address(
            local_src, chain, chain_n, PARTIAL_BYTES, WRITER_JCT_MONOIDJUNCTION, jct_csr);
        printf("[DBG chip %x] cfg ret=%d -> xdma_start\r\n", chip_id, ret);
        if (ret != 0) {
            printf("[MonoidGather4c] cfg failed: %d\r\n", ret);
            err = 1;
        } else {
            xdma_task_t task = xdma_start();
            printf("[DBG chip %x] committed remote=%d id=%d -> wait\r\n",
                   chip_id, task.remote, task.task_id);
            volatile uint32_t *d0 = (volatile uint32_t *)(uintptr_t)(tcdm_base + PARTIAL_BYTES);
            uint32_t lf = 0, rf = 0, spins = 0;
            do {
                lf = snax_read_xdma_cfg_reg(XDMA_FINISH_LOCAL_TASK_PTR);
                rf = snax_read_xdma_cfg_reg(XDMA_FINISH_REMOTE_TASK_PTR);
                spins++;
            } while (lf < task.task_id && rf < task.task_id && *d0 == 0xDEADBEEFu &&
                     spins < 500000u);
            printf("[DBG chip %x] poll done: local=%u remote=%u d0=%x spins=%u\r\n",
                   chip_id, lf, rf, *d0, spins);
            printf("[MonoidGather4c] xdma finished in %d cycles (P=%d)\r\n",
                   xdma_last_task_cycle(), GATHER_P);
            volatile uint32_t *rr = (volatile uint32_t *)(uintptr_t)(tcdm_base + PARTIAL_BYTES);
            uint32_t m_got = rr[MOMENT_M_LANE], m_exp = gold[MOMENT_M_LANE];
            uint32_t l_got = rr[MOMENT_L_LANE], l_exp = gold[MOMENT_L_LANE];
            uint32_t l_ulp = (l_got > l_exp) ? (l_got - l_exp) : (l_exp - l_got);
            printf("[MonoidGather4c] m*: got %x exp %x | l*: got %x exp %x (ulp=%x tol=%x)\r\n",
                   m_got, m_exp, l_got, l_exp, l_ulp, L_TOL_ULPS);
            if (m_got != m_exp) { printf("[MonoidGather4c] m* MISMATCH\r\n"); err++; }
            if (l_ulp > L_TOL_ULPS) { printf("[MonoidGather4c] l* OUT OF TOLERANCE\r\n"); err++; }
            printf(err ? "[MonoidGather4c] Check: FAIL (%d)\r\n"
                       : "[MonoidGather4c] Check: PASS\r\n", err);
        }
        for (int s = 0; s < 3; s++)
            xchip_store_u32(CHAIN_SRC[s], tcdm_base + DONE_OFF, FLAG_SET);
    } else if (src_slot(chip_id) >= 0 && snrt_is_dm_core()) {
        while (*done != FLAG_SET) asm volatile("fence" ::: "memory");
        printf("[DBG chip %x] released by collector\r\n", chip_id);
    }
    snrt_global_barrier();
    return err;
}
