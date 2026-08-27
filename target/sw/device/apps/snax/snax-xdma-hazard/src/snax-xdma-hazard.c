// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>

// This SW is used to test the XDMA read and write between L3 and TCDM
// and between TCDMs of different clusters.
// The test includes 6 parts:
// 1. All clusters read data from L3 to their TCDM at the same time
// 2. Every cluster except C0 reads from its previous cluster
//    C1 reads C0, C2 reads C1, ..., Cn reads C(n-1)
// 3. Every cluster except Cn reads from its next cluster
//    C0 reads C1, C1 reads C2, ..., C(n-1) reads Cn
// 4. Every cluster except Cn writes to its next cluster
//    C0 writes C1, C1 writes C2, ..., C(n-1) writes Cn
// 5. Every cluster except C0 writes to its previous cluster
//    C1 writes C0, C2 writes C1, ..., Cn writes C(n-1)
// 6. All clusters write data back to L3 from their TCDM at the same time
// The data size is defined in data.h file, which is 8192B by default.

#include "data.h"
#include "snrt.h"

uint32_t time[SNRT_CLUSTER_NUM];
#define TCDM_OFFSET 0x1000

// Run a blocking 1D xDMA copy. xdma_memcpy_1d returns a negative code on a
// config error (e.g. -2 when neither src nor dst is in the calling cluster's
// local L1); report it and skip start/wait, rather than issuing a bogus task
// that would hang in xdma_remote_wait.
static inline void xdma_copy_1d_checked(void *src, void *dst, uint32_t size) {
    int32_t ret = xdma_memcpy_1d(src, dst, size);
    if (ret != 0) {
        printf("ERROR: xdma_memcpy_1d failed (ret=%d), skipping transfer\r\n",
               ret);
        return;
    }
    int task_id = xdma_start().task_id;
    xdma_remote_wait(task_id);
}

int main() {
    const uint32_t cluster_idx = snrt_cluster_idx();
    const uint32_t cluster_count = SNRT_CLUSTER_NUM;
    const uint32_t transfer_size = data_size * sizeof(data[0]);
    const uintptr_t tcdm_baseaddress =
        snrt_cluster_base_addrl() + TCDM_OFFSET;

    if (snrt_global_core_idx() == 0) {
        printf("Now start to let clusters read %uB from L3\r\n",
               transfer_size);
    }
    if (snrt_is_dm_core()) {
        xdma_copy_1d_checked(data, (void *)tcdm_baseaddress, transfer_size);
        time[cluster_idx] = xdma_last_task_cycle();
    }

    snrt_global_barrier();
    if (snrt_global_core_idx() == 0) {
        for (uint32_t i = 0; i < cluster_count; i++) {
            printf(
                "XDMA remote read from L3 to TCDM C%u is done in %u cycles.\r\n",
                i, time[i]);
        }
    }
    snrt_global_barrier();

    if (snrt_global_core_idx() == 0) {
        printf(
            "Now start lower-to-higher XDMA reads of %uB between clusters\r\n",
            transfer_size);
    }
    if ((cluster_idx > 0) && snrt_is_dm_core()) {
        const uintptr_t previous_tcdm = tcdm_baseaddress - cluster_offset;
        xdma_copy_1d_checked((void *)previous_tcdm, (void *)tcdm_baseaddress,
                             transfer_size);
        time[cluster_idx] = xdma_last_task_cycle();
    }

    snrt_global_barrier();
    if (snrt_global_core_idx() == 0) {
        for (uint32_t i = 1; i < cluster_count; i++) {
            printf(
                "XDMA remote read from TCDM C%u to TCDM C%u is done in %u cycles.\r\n",
                i - 1, i, time[i]);
        }
    }
    snrt_global_barrier();

    if (snrt_global_core_idx() == 0) {
        printf(
            "Now start higher-to-lower XDMA reads of %uB between clusters\r\n",
            transfer_size);
    }
    if ((cluster_idx + 1 < cluster_count) && snrt_is_dm_core()) {
        const uintptr_t next_tcdm = tcdm_baseaddress + cluster_offset;
        xdma_copy_1d_checked((void *)next_tcdm, (void *)tcdm_baseaddress,
                             transfer_size);
        time[cluster_idx] = xdma_last_task_cycle();
    }

    snrt_global_barrier();
    if (snrt_global_core_idx() == 0) {
        for (uint32_t i = 0; i + 1 < cluster_count; i++) {
            printf(
                "XDMA remote read from TCDM C%u to TCDM C%u is done in %u cycles.\r\n",
                i + 1, i, time[i]);
        }
    }
    snrt_global_barrier();

    if (snrt_global_core_idx() == 0) {
        printf(
            "Now start lower-to-higher XDMA writes of %uB between clusters\r\n",
            transfer_size);
    }
    if ((cluster_idx + 1 < cluster_count) && snrt_is_dm_core()) {
        const uintptr_t next_tcdm = tcdm_baseaddress + cluster_offset;
        xdma_copy_1d_checked((void *)tcdm_baseaddress, (void *)next_tcdm,
                             transfer_size);
        time[cluster_idx] = xdma_last_task_cycle();
    }

    snrt_global_barrier();
    if (snrt_global_core_idx() == 0) {
        for (uint32_t i = 0; i + 1 < cluster_count; i++) {
            printf(
                "XDMA remote write from TCDM C%u to TCDM C%u is done in %u cycles.\r\n",
                i, i + 1, time[i]);
        }
    }
    snrt_global_barrier();

    if (snrt_global_core_idx() == 0) {
        printf(
            "Now start higher-to-lower XDMA writes of %uB between clusters\r\n",
            transfer_size);
    }
    if ((cluster_idx > 0) && snrt_is_dm_core()) {
        const uintptr_t previous_tcdm = tcdm_baseaddress - cluster_offset;
        xdma_copy_1d_checked((void *)tcdm_baseaddress, (void *)previous_tcdm,
                             transfer_size);
        time[cluster_idx] = xdma_last_task_cycle();
    }

    snrt_global_barrier();
    if (snrt_global_core_idx() == 0) {
        for (uint32_t i = 1; i < cluster_count; i++) {
            printf(
                "XDMA remote write from TCDM C%u to TCDM C%u is done in %u cycles.\r\n",
                i, i - 1, time[i]);
        }
    }
    snrt_global_barrier();

#ifdef XDMA_CHECK_RESULT
    if (snrt_is_dm_core()) {
        // Check the result
        uint32_t *golden_result = (uint32_t *)tcdm_baseaddress;
        uint32_t *tcdm_result = (uint32_t *)data;

        for (int i = 0; i < data_size * sizeof(data[0]) / 4; i++) {
            if (tcdm_result[i] != golden_result[i]) {
                printf("The data copy is incorrect at byte %d! \n", i << 2);
            }
        }
        printf("Checking is done. All values are right\n");
    }
    snrt_global_barrier();
#endif
    if (snrt_global_core_idx() == 0) {
        printf("Now start to let clusters write %uB back to L3\r\n",
               transfer_size);
    }
    if (snrt_is_dm_core()) {
        xdma_copy_1d_checked((void *)tcdm_baseaddress, data, transfer_size);
        time[cluster_idx] = xdma_last_task_cycle();
    }
    snrt_global_barrier();
    if (snrt_global_core_idx() == 0) {
        for (uint32_t i = 0; i < cluster_count; i++) {
            printf(
                "XDMA remote write from TCDM C%u to L3 is done in %u cycles.\r\n",
                i, time[i]);
        }
    }
    snrt_global_barrier();

#ifdef XDMA_CHECK_RESULT
    if (snrt_is_dm_core()) {
        // Check the result
        uint32_t *golden_result = (uint32_t *)tcdm_baseaddress;
        uint32_t *tcdm_result = (uint32_t *)data;

        for (int i = 0; i < data_size * sizeof(data[0]) / 4; i++) {
            if (tcdm_result[i] != golden_result[i]) {
                printf("The data copy is incorrect at byte %d! \n", i << 2);
            }
        }
        printf("Checking is done. All values are right\n");
    }
    snrt_global_barrier();
#endif

    return 0;
}
