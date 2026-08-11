// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Cross-DIE (4-chiplet) linear ChainGather -- the same in-fabric element-wise ADD fold as the
// single-chip 4-cluster version, but the chain now crosses the die-to-die links: each of the four
// compute chiplets (0x00,0x01,0x10,0x11) owns one 64 B partial; the collector (chip 0x00) gathers
// the other three over D2D, folding element-wise along the chain, and the sum lands in its local
// buffer. Root-initiated: only chip 0x00's core issues the gather; the others just stage + sync.
//
// Cross-chip sync note: snrt_chip_global_barrier() uses a 0xFF-broadcast scalar store (announce),
// which hangs under hemaia_xdma_rr_1c (broadcast routing untested here). We instead sync with
// TARGETED cross-chip scalar stores (proven direction) + local polling: each source writes a ready
// flag into the collector's TCDM; the collector polls locally, gathers, then writes a done flag
// back into each source's TCDM. Only cross-chip WRITES to specific chips -- no 0xFF broadcast.

#include "chip_id.h"   // get_current_chip_id, get_chip_baseaddress_value
#include "data.h"
#include "snrt.h"

#define PARTIAL_BYTES (PARTIAL_ELEMS * (int)sizeof(float))
#define COLLECTOR  0x00
// Gather width. GATHER_P==2: cross-die P=2 (adjacent source, no middle node -- works today).
// GATHER_P==4: full snake chain (blocked on the middle-node grant RTL bug -- kept for the fix).
#ifndef GATHER_P
#define GATHER_P 4
#endif

// Sync-flag TCDM offsets, clear of the data region (partial @ 0, dst @ PARTIAL_BYTES=64).
#define READY_OFF 0x400u   // collector-side: 3 ready slots (one per source), one u32 each
#define DONE_OFF  0x480u   // source-side: one done flag the collector sets after the gather
#define FLAG_SET  0xA5A5A5A5u
// DIAG: spoke->spoke cross-chip scalar write probe. 0x10 writes this magic into 0x11's TCDM; 0x11
// reports whether it arrived. Tests whether a NON-HUB chip can write to another NON-HUB chip (the
// cross-chip write pattern the P>=3 middle-node grant needs; Gate-A only ever used hub-origin writes).
#define XCHIP_TEST_OFF 0x500u
#define XCHIP_MAGIC    0x51505150u

// Chain in DATA order (far chiplet first), then the collector's local dst.
// The chain MUST follow physically-ADJACENT D2D hops: the backward grant handshake (root grants its
// prev hop, which grants its prev hop, ... until the head is released) does NOT route between diagonal
// chiplet pairs. Grid: 0x00=[0,0] 0x01=[0,1] 0x10=[1,0] 0x11=[1,1]. The old order [0x11,0x10,0x01] made
// the chain 0x11->0x10->0x01->0x00 whose middle hop 0x10<->0x01 is DIAGONAL, so the 0x01->0x10 grant
// never arrived -> 0x10 never granted 0x11 -> the head stalled with data ready forever (WLF-confirmed:
// head 0x11 write_req_data_valid_i=1, write_req_grant_i=0; root 0x00 already in WAIT_FINISH). Snake
// 0x01->0x11->0x10->0x00 makes every hop adjacent. Sum / moment-merge are commutative, golden unchanged.
static const uint8_t CHIPS[4]   = {0x00, 0x01, 0x10, 0x11};
static const uint8_t CHAIN_SRC[3] = {0x01, 0x11, 0x10};  // far -> near, all-adjacent D2D hops

static int chip_index(uint8_t c) {
    for (int i = 0; i < 4; i++)
        if (CHIPS[i] == c) return i;
    return -1;
}
static int src_slot(uint8_t c) {  // index of a source chip within CHAIN_SRC[], else -1
    for (int i = 0; i < 3; i++)
        if (CHAIN_SRC[i] == c) return i;
    return -1;
}

// Targeted cross-chip 32-bit store via the Mseg CSR (0xbc0). Mirrors announce_chip_checkpoint's
// mechanism but aims at ONE specific chip's prefix instead of the 0xFF broadcast: set Mseg to the
// target chip's high bits (chip_id<<8), store to the (identical-layout) local address, restore Mseg.
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

    // Clear local sync flags before anyone can write them (done: local; ready: collector-only).
    volatile uint32_t *xtest = (volatile uint32_t *)(uintptr_t)(tcdm_base + XCHIP_TEST_OFF);
    if (ci >= 0 && snrt_is_dm_core()) {
        *done = 0;
        *xtest = 0;  // DIAG: clear spoke->spoke probe slot
        if (chip_id == COLLECTOR) { ready[0] = 0; ready[1] = 0; ready[2] = 0; }
    }
    snrt_global_barrier();  // within-chip (proven); orders the clears before staging/writes

    // Each chiplet stages its OWN partial into its cluster TCDM offset 0 (same local offset on
    // every chip, so the collector's cross-chip address hits it).
    if (ci >= 0 && snrt_is_dm_core()) {
        snrt_dma_start_1d((void *)tcdm_base,
                          &chain_gather_data[ci * PARTIAL_ELEMS], PARTIAL_BYTES);
        snrt_dma_wait_all();
        printf("[DBG chip %x] staged partial\r\n", chip_id);  // DIAG
        // sync1: sources announce readiness to the collector via a targeted cross-chip store.
        int s = src_slot(chip_id);
        if (s >= 0) {
            xchip_store_u32(COLLECTOR, tcdm_base + READY_OFF + (uint32_t)s * 4u, FLAG_SET);
            printf("[DBG chip %x] wrote ready slot %d -> collector\r\n", chip_id, s);  // DIAG
        }
        // DIAG spoke->spoke probe: 0x10 (non-hub) writes 0x11 (non-hub) directly.
        if (chip_id == 0x10) {
            xchip_store_u32(0x11, tcdm_base + XCHIP_TEST_OFF, XCHIP_MAGIC);
            printf("[DBG chip 10] spoke->spoke wrote magic to chip 11\r\n");  // DIAG
        }
    }

    int err = 0;
    if (chip_id == COLLECTOR && snrt_is_dm_core()) {
        // sync1 wait: poll the three local ready slots.
        for (int s = 0; s < 3; s++)
            while (ready[s] != FLAG_SET) asm volatile("fence" ::: "memory");
        printf("[DBG chip %x] all sources ready\r\n", chip_id);  // DIAG

        for (uint8_t e = 0; e < XDMA_DST_EXT_NUM; e++) xdma_disable_dst_ext(e);
        for (uint8_t r = 0; r < XDMA_SRC_EXT_NUM; r++) xdma_disable_src_ext(r);
        for (uint8_t j = 0; j < XDMA_DST_JCT_NUM; j++) xdma_disable_dst_junction(j);

        uint64_t local_src = chiplet_addr_transform_full(COLLECTOR, (uint64_t)tcdm_base);
        uint64_t dst_local = chiplet_addr_transform_full(COLLECTOR, (uint64_t)(tcdm_base + PARTIAL_BYTES));
#if GATHER_P == 2
        // P=2 cross-die: collector 0x00 gathers ONE physically-adjacent source (0x10). A length-2
        // chain has NO middle node, so it uses only the root->head grant (WLF-confirmed working);
        // the P>=3 middle-node grant hop is the RTL bug. Isolates the cross-die in-fabric fold.
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

        uint32_t jct_csr = (3u << 4) | 0u;  // ElementwiseJunction: op=ADD, fmt=FP32
        int32_t ret = xdma_chain_gather_1d_full_address(
            local_src, chain, chain_n, PARTIAL_BYTES, WRITER_JCT_ELEMENTWISEJUNCTION, jct_csr);
        printf("[DBG chip %x] cfg ret=%d -> xdma_start\r\n", chip_id, ret);  // DIAG
        if (ret != 0) {
            printf("[ChainGather4c] cfg failed: %d\r\n", ret);
            err = 1;
        } else {
            xdma_task_t task = xdma_start();
            printf("[DBG chip %x] committed remote=%d id=%d -> wait\r\n",
                   chip_id, task.remote, task.task_id);  // DIAG
            // DIAG: bounded dual-counter poll -- which finish counter bumps (local vs remote),
            // and does the fold actually land? (sentinel 0xDEADBEEF at dst[0] pre-fill).
            volatile uint32_t *dst0 = (volatile uint32_t *)(uintptr_t)(tcdm_base + PARTIAL_BYTES);
            uint32_t lf = 0, rf = 0, spins = 0;
            do {
                lf = snax_read_xdma_cfg_reg(XDMA_FINISH_LOCAL_TASK_PTR);
                rf = snax_read_xdma_cfg_reg(XDMA_FINISH_REMOTE_TASK_PTR);
                spins++;
            } while (lf < task.task_id && rf < task.task_id && *dst0 == 0xDEADBEEFu &&
                     spins < 500000u);
            printf("[DBG chip %x] poll done: local=%u remote=%u dst0=%x spins=%u\r\n",
                   chip_id, lf, rf, *dst0, spins);
            printf("[ChainGather4c] xdma finished in %d cycles (P=%d)\r\n",
                   xdma_last_task_cycle(), GATHER_P);
            volatile uint32_t *rr = (volatile uint32_t *)(uintptr_t)(tcdm_base + PARTIAL_BYTES);
            printf("[ChainGather4c] dst[0..3] = %x %x %x %x\r\n", rr[0], rr[1], rr[2], rr[3]);
            for (int j = 0; j < PARTIAL_ELEMS; j++) {
                if (rr[j] != gold[j]) {
                    if (err < 4) printf("[ChainGather4c] MISMATCH %d: got %x exp %x\r\n", j, rr[j], gold[j]);
                    err++;
                }
            }
            printf(err ? "[ChainGather4c] Check: FAIL (%d)\r\n"
                       : "[ChainGather4c] Check: PASS\r\n", err);
        }
        // sync2: release the sources now that the gather has read them.
        for (int s = 0; s < 3; s++)
            xchip_store_u32(CHAIN_SRC[s], tcdm_base + DONE_OFF, FLAG_SET);
    } else if (src_slot(chip_id) >= 0 && snrt_is_dm_core()) {
        // sync2 wait: hold the source until the collector signals it has read our partial.
        while (*done != FLAG_SET) asm volatile("fence" ::: "memory");
        printf("[DBG chip %x] released by collector\r\n", chip_id);  // DIAG
        // DIAG spoke->spoke result: did 0x10's (non-hub) write reach 0x11 (non-hub)?
        if (chip_id == 0x11)
            printf("[DBG chip 11] spoke->spoke recv = %x (expect %x)\r\n", *xtest, XCHIP_MAGIC);
    }
    snrt_global_barrier();  // within-chip clean exit
    return err;
}
