// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include "data.h"
#include "snrt.h"

#define SOURCE_CHIP_ID 0x10
#define SOURCE_CLUSTER_ID 1
#define CLUSTERS_PER_CHIP 2
#define CHAIN_DEST_COUNT 8
// The four-compute-chip environment has one memory chip at (2, 0).
#define MEM_CHIP_ID 0x20
#define MEM_CHIP_LOCAL_ADDR 0x80000000u

_Static_assert(CHAIN_DEST_COUNT <= XDMA_MAX_DST_COUNT,
               "XDMA needs eight destination slots for this chainwrite");

static int check_data(const volatile uint8_t *actual) {
    int err = 0;
    for (uint32_t i = 0; i < data_size; i++) {
        err += actual[i] != data[i];
    }
    return err;
}

int main(void) {
    int err = 0;
    const uint8_t chip_id = get_current_chip_id();
    const uint32_t cluster_id = snrt_cluster_idx();
    const uint32_t data_bytes = data_size * sizeof(data[0]);
    const uint32_t tcdm_baseaddress = snrt_cluster_base_addrl();
    // CPU pointers stay local to this chip. Only DMA uses full global addresses.
    volatile uint8_t *local_data = (volatile uint8_t *)tcdm_baseaddress;
    const uint64_t mem_dest =
        chiplet_addr_transform_full(MEM_CHIP_ID, MEM_CHIP_LOCAL_ADDR);

    if (snrt_cluster_num() != CLUSTERS_PER_CHIP || data_bytes == 0 ||
        data_bytes % XDMA_WIDTH != 0 || data_bytes > SNRT_TCDM_SIZE / 2) {
        if (snrt_is_dm_core()) {
            printf("Chip %02x cluster %u: expected two clusters and a nonzero "
                   "XDMA-aligned payload fitting twice in TCDM\n",
                   chip_id, cluster_id);
        }
        return 1;
    }

    // All cores on all four compute chips participate in the same barriers.
    snrt_chip_barrier_init(0x00, 0x11);

    if (chip_id == SOURCE_CHIP_ID && cluster_id == SOURCE_CLUSTER_ID &&
        snrt_is_dm_core()) {
        snrt_dma_start_1d((void *)tcdm_baseaddress, data, data_bytes);
        snrt_dma_wait_all();
    } else if (snrt_is_dm_core()) {
        // A missing transfer must fail even if TCDM holds a previous result.
        for (uint32_t i = 0; i < data_size; i++) {
            local_data[i] = (uint8_t)~data[i];
        }
    }

    // Finish receiver initialization before any chip can launch the chainwrite.
    snrt_chip_global_barrier();

    if (chip_id == SOURCE_CHIP_ID && cluster_id == SOURCE_CLUSTER_ID &&
        snrt_is_dm_core()) {
        for (uint32_t i = 0; i < XDMA_DST_EXT_NUM; i++) {
            err += xdma_disable_dst_ext(i) != 0;
        }
        for (uint32_t i = 0; i < XDMA_SRC_EXT_NUM; i++) {
            err += xdma_disable_src_ext(i) != 0;
        }

        // Nine locations including the initialized source, eight destinations:
        // 10:C1 -> 10:C0 -> 00:C1 -> 00:C0 -> 01:C1 -> 01:C0
        //       -> 11:C1 -> 11:C0 -> memory chip.
        const uint8_t chip_order[] = {0x10, 0x00, 0x01, 0x11};
        const uint32_t cluster0_base =
            tcdm_baseaddress - cluster_id * cluster_offset;
        uint64_t dest[CHAIN_DEST_COUNT];
        uint32_t dest_idx = 0;
        for (uint32_t chip = 0; chip < sizeof(chip_order) / sizeof(chip_order[0]);
             chip++) {
            for (int cluster = CLUSTERS_PER_CHIP - 1; cluster >= 0; cluster--) {
                if (chip_order[chip] == SOURCE_CHIP_ID &&
                    cluster == SOURCE_CLUSTER_ID) {
                    continue;
                }
                dest[dest_idx++] = chiplet_addr_transform_full(
                    chip_order[chip], cluster0_base + cluster * cluster_offset);
            }
        }
        dest[dest_idx++] = mem_dest;

        if (err == 0) {
            const uint64_t src =
                chiplet_addr_transform_full(chip_id, tcdm_baseaddress);
            int status = xdma_multicast_1d_full_address(src, dest, dest_idx,
                                                      data_bytes);
            if (status != 0) {
                printf("XDMA chainwrite configuration failed: %d\n", status);
                err++;
            } else {
                xdma_task_t task = xdma_start();
                // Completion propagates back from the final node in the chain.
                xdma_wait_task(task);
                printf("XDMA chainwrite finished in %u cycles\n",
                       xdma_last_task_cycle());
            }
        }
    }

    snrt_cluster_hw_barrier();
    snrt_chip_global_barrier();

    // Each cluster checks its own TCDM against the golden data on its own chip.
    if (snrt_is_dm_core()) {
        err += check_data(local_data);
        printf("Chip %02x cluster %u: %s (%d errors)\n", chip_id, cluster_id,
               err ? "FAIL" : "PASS", err);

        if (chip_id == SOURCE_CHIP_ID && cluster_id == SOURCE_CLUSTER_ID &&
            err == 0) {
            // Verify the ninth location through a DMA readback into a separate
            // local buffer; an RV32 pointer cannot address the memory chip.
            volatile uint8_t *mem_result = local_data + data_bytes;
            snrt_dma_start_1d_wideptr(
                chiplet_addr_transform_full(chip_id, (uint32_t)mem_result),
                mem_dest, data_bytes);
            snrt_dma_wait_all();
            int mem_err = check_data(mem_result);
            err += mem_err;
            printf("Memory chip %02x: %s (%d errors)\n", MEM_CHIP_ID,
                   mem_err ? "FAIL" : "PASS", mem_err);
        }
    }

    // Keep every chip running until all local checks and the readback finish.
    snrt_chip_global_barrier();
    return err;
}
