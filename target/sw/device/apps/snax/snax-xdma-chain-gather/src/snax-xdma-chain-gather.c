// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// ChainGather bring-up on a single chip, 4 clusters (hemaia_xdma_rr_4c).
//
// Each cluster stages its own 64 B partial (PARTIAL_ELEMS fp32) into TCDM offset 0.
// The collector (cluster 0) issues ONE chain-gather: the xDMA unrolls a cfg to each
// hop, every hop reads its own partial and an in-fabric ElementwiseJunction (per-element
// FP32 ADD) folds the arriving stream with it along the chain. The collector folds its
// own partial too and the summed beat lands in its local dst buffer. Golden = the
// element-wise sum over all clusters. Root-initiated: the participants' cores do nothing.

#include "data.h"
#include "snrt.h"

#define PARTIAL_BYTES (PARTIAL_ELEMS * (int)sizeof(float))

int main() {
    uint32_t tcdm_base = snrt_cluster_base_addrl();
    uint32_t cid = snrt_cluster_idx();

    // Every cluster stages its own partial into TCDM offset 0.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d((void *)tcdm_base,
                          &chain_gather_data[cid * PARTIAL_ELEMS], PARTIAL_BYTES);
        snrt_dma_wait_all();
    }
    snrt_global_barrier();

    int err = 0;
    if (cid == 0 && snrt_is_dm_core()) {
        // Clean writer config: no reader/writer extensions, no stale junctions.
        for (uint8_t e = 0; e < XDMA_DST_EXT_NUM; e++) xdma_disable_dst_ext(e);
        for (uint8_t r = 0; r < XDMA_SRC_EXT_NUM; r++) xdma_disable_src_ext(r);
        for (uint8_t j = 0; j < XDMA_DST_JCT_NUM; j++) xdma_disable_dst_junction(j);

        uint64_t h = ((uint64_t)snrt_cluster_base_addrh()) << 32;
        uint64_t local_src = h | (uint64_t)tcdm_base;                       // collector partial
        uint64_t dst_local = h | (uint64_t)(tcdm_base + PARTIAL_BYTES);     // collector result

        // Chain in DATA order: far source (cluster N-1) first ... near source (cluster 1),
        // ending at the collector's own dst buffer. Every partial is at cluster-local offset 0.
        uint64_t chain[NUM_CLUSTERS];  // (NUM_CLUSTERS-1) remote sources + 1 local dst
        int n = 0;
        for (int c = NUM_CLUSTERS - 1; c >= 1; c--) {
            chain[n++] = h | (uint64_t)(tcdm_base + c * SNRT_CLUSTER_OFFSET);
        }
        chain[n++] = dst_local;

        // DIAG: sentinel-fill the dst so we can tell "writer never wrote" (stays 0xDEADBEEF)
        // from "wrote wrong data". Integer stores only (no FPU on this core).
        volatile uint32_t *dbg = (volatile uint32_t *)(uintptr_t)(tcdm_base + PARTIAL_BYTES);
        for (int jj = 0; jj < PARTIAL_ELEMS; jj++) dbg[jj] = 0xDEADBEEFu;
        // DIAG: show the collector's OWN staged partial (should be chain_gather_data[0..]).
        volatile uint32_t *own = (volatile uint32_t *)(uintptr_t)tcdm_base;
        printf("[ChainGather] own partial [0..3] = %x %x %x %x\r\n", own[0], own[1], own[2], own[3]);

        // ElementwiseJunction CSR(0): [3:0] op = ADD(0), [6:4] fmt = FP32(3).
        uint32_t jct_csr = (3u << 4) | 0u;
        int32_t ret = xdma_chain_gather_1d_full_address(
            local_src, chain, n, PARTIAL_BYTES,
            WRITER_JCT_ELEMENTWISEJUNCTION, jct_csr);
        if (ret != 0) {
            printf("[ChainGather] cfg failed: %d\r\n", ret);
            return 1;
        }

        xdma_task_t task = xdma_start();
        xdma_wait_task(task);
        printf("[ChainGather] xdma finished in %d cycles\r\n", xdma_last_task_cycle());

        // DIAG: dump the dst beat as raw hex (0xDEADBEEF => writer never wrote).
        volatile uint32_t *rr = (volatile uint32_t *)(uintptr_t)(tcdm_base + PARTIAL_BYTES);
        printf("[ChainGather] dst[0..7] = %x %x %x %x %x %x %x %x\r\n",
               rr[0], rr[1], rr[2], rr[3], rr[4], rr[5], rr[6], rr[7]);

        // Check the folded (summed) beat against the golden. The DM core is rv32ima with
        // NO FPU, so compare the raw fp32 BIT PATTERNS as integers (the values are exact
        // integer-valued floats, so the bits must match exactly) -- a `float` compare would
        // emit `flw` and trap as an illegal instruction on this core.
        volatile uint32_t *res = (volatile uint32_t *)(uintptr_t)(tcdm_base + PARTIAL_BYTES);
        uint32_t *gold = (uint32_t *)chain_gather_golden;
        for (int jj = 0; jj < PARTIAL_ELEMS; jj++) {
            if (res[jj] != gold[jj]) {
                printf("[ChainGather] MISMATCH at %d: got 0x%x exp 0x%x\r\n", jj,
                       res[jj], gold[jj]);
                err++;
            }
        }
        if (err) {
            printf("[ChainGather] Check: FAIL (%d mismatches)\r\n", err);
        } else {
            printf("[ChainGather] Check: PASS\r\n");
        }
    }

    return err;
}
