// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Nonlinear ChainGather on a single chip, 4 clusters (hemaia_xdma_rr_4c) -- the online-softmax
// (m, l) moment-merge via the xDMA MonoidJunction (MOMENT combine), the paper's in-fabric
// nonlinear collective. Each cluster owns one (m, l) pair (m at lane 0, l at lane 8; nValid=1);
// the collector gathers all clusters' pairs, folding them along the chain with
//   (m1,l1) (+) (m2,l2) = (max(m1,m2), l1*exp(m1-m*) + l2*exp(m2-m*))
// and the merged (m*, l*) lands in its local buffer. m* is exact (a max); l* uses the writer's
// exp LUT, so it is checked to a fp32 ULP tolerance (the DM core is rv32ima, no FPU -> integer
// bit-pattern math only).

#include "data.h"
#include "snrt.h"

#define PARTIAL_BYTES (PARTIAL_ELEMS * (int)sizeof(float))
// fp32 ULP tolerance for l* (exp-LUT approximation). ~0x20000 ULPs ~= 1.5% at l*~1.19, matching
// the Gate-A moment-merge's 0.02 acceptance bound. The actual ULP delta is printed to tighten later.
#define L_TOL_ULPS 0x20000u

int main() {
    uint32_t tcdm_base = snrt_cluster_base_addrl();
    uint32_t cid = snrt_cluster_idx();

    // Every cluster stages its own (m,l) beat into TCDM offset 0.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d((void *)tcdm_base,
                          &chain_gather_data[cid * PARTIAL_ELEMS], PARTIAL_BYTES);
        snrt_dma_wait_all();
    }
    snrt_global_barrier();

    int err = 0;
    if (cid == 0 && snrt_is_dm_core()) {
        for (uint8_t e = 0; e < XDMA_DST_EXT_NUM; e++) xdma_disable_dst_ext(e);
        for (uint8_t r = 0; r < XDMA_SRC_EXT_NUM; r++) xdma_disable_src_ext(r);
        for (uint8_t j = 0; j < XDMA_DST_JCT_NUM; j++) xdma_disable_dst_junction(j);

        uint64_t h = ((uint64_t)snrt_cluster_base_addrh()) << 32;
        uint64_t local_src = h | (uint64_t)tcdm_base;
        uint64_t dst_local = h | (uint64_t)(tcdm_base + PARTIAL_BYTES);

        uint64_t chain[NUM_CLUSTERS];
        int n = 0;
        for (int c = NUM_CLUSTERS - 1; c >= 1; c--) {
            chain[n++] = h | (uint64_t)(tcdm_base + c * SNRT_CLUSTER_OFFSET);
        }
        chain[n++] = dst_local;

        // Sentinel so we can tell "never written" from "wrong value".
        volatile uint32_t *dbg = (volatile uint32_t *)(uintptr_t)(tcdm_base + PARTIAL_BYTES);
        for (int j = 0; j < PARTIAL_ELEMS; j++) dbg[j] = 0xDEADBEEFu;

        // MonoidJunction CSR(0): [15:13] combineMode = MOMENT(1), [7:0] nValid = 1.
        uint32_t jct_csr = (1u << 13) | 1u;  // 0x2001
        int32_t ret = xdma_chain_gather_1d_full_address(
            local_src, chain, n, PARTIAL_BYTES,
            WRITER_JCT_MONOIDJUNCTION, jct_csr);
        if (ret != 0) {
            printf("[MonoidGather] cfg failed: %d\r\n", ret);
            return 1;
        }

        xdma_task_t task = xdma_start();
        xdma_wait_task(task);
        printf("[MonoidGather] xdma finished in %d cycles\r\n", xdma_last_task_cycle());

        volatile uint32_t *rr = (volatile uint32_t *)(uintptr_t)(tcdm_base + PARTIAL_BYTES);
        uint32_t *gold = (uint32_t *)chain_gather_golden;
        uint32_t m_got = rr[MOMENT_M_LANE], m_exp = gold[MOMENT_M_LANE];
        uint32_t l_got = rr[MOMENT_L_LANE], l_exp = gold[MOMENT_L_LANE];
        uint32_t l_ulp = (l_got > l_exp) ? (l_got - l_exp) : (l_exp - l_got);
        printf("[MonoidGather] m*: got %x exp %x | l*: got %x exp %x (ulp=%x tol=%x)\r\n",
               m_got, m_exp, l_got, l_exp, l_ulp, L_TOL_ULPS);
        if (m_got != m_exp) { printf("[MonoidGather] m* MISMATCH\r\n"); err++; }
        if (l_ulp > L_TOL_ULPS) { printf("[MonoidGather] l* OUT OF TOLERANCE\r\n"); err++; }
        if (err) {
            printf("[MonoidGather] Check: FAIL (%d)\r\n", err);
        } else {
            printf("[MonoidGather] Check: PASS\r\n");
        }
    }

    return err;
}
