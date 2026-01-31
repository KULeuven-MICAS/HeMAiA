// Copyright 2024 KU Leuven
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Yunhao Deng <yunhao.deng@kuleuven.be>

// Minimal Bootrom for HeMAiA

// Please avoid using initialized global variable (.data
// region) in C. This will lead to the loss of initial value, as they are
// allocated in spm instead of bootrom. Thus, the initial data will lose.

// For global constant, use "const var"
// For values need to share between functions, use uninitialized global variable
// + initialization function

#include "chip_id.h"
#include "sys_dma_bootrom.h"
#include "occamy_memory_map.h"
#define bool uint8_t
#define true 1
#define false 0
// We only initialize the first 8KB and last 8KB of the TCDM
// This is mainly for the saving of simulation time
// In real silicon, all the mem should be initialized
// In the testing setup, anyway the mem will be handled by the dma engine to load data either from L3 or mempool later
// The first space is for the allocator and the last space is for stack and other data
#define INIT_TCDM_SIZE 0x2000
void bootrom() {
    // Initialize L2
    register uint64_t stack_pointer asm("sp");
    sys_dma_blk_memcpy(
        chiplet_addr_transform(SPM_NARROW_BASE_ADDR),
        chiplet_addr_transform(WIDE_ZERO_MEM_BASE_ADDR),
        stack_pointer - chiplet_addr_transform(SPM_NARROW_BASE_ADDR) - 1024);
    // Initialize Cluster TCDM
    for (uint32_t i = 0; i < N_CLUSTERS_PER_CHIPLET; i++) {
        // CLUSTER_TCDM_BASE_ADDR = QUAD_WIDE_CLUSTER_0_TCDM_BASE_ADDR + i * cluster_offset
        // CLUSTER_TCDM_END_ADDR = CLUSTER_TCDM_BASE_ADDR + CLUSTER_TCDM_SIZE
        // First 8KB
        sys_dma_blk_memcpy(
            chiplet_addr_transform(QUAD_WIDE_CLUSTER_0_TCDM_BASE_ADDR + i * cluster_offset),
            chiplet_addr_transform(WIDE_ZERO_MEM_BASE_ADDR),
            INIT_TCDM_SIZE);
        // Last 8KB
        sys_dma_blk_memcpy(
            chiplet_addr_transform(QUAD_WIDE_CLUSTER_0_TCDM_BASE_ADDR + i * cluster_offset + CLUSTER_TCDM_SIZE - INIT_TCDM_SIZE),
            chiplet_addr_transform(WIDE_ZERO_MEM_BASE_ADDR),
            INIT_TCDM_SIZE);
        }        

}
