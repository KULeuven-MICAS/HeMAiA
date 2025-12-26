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

void bootrom() {
    // Initialize L2
    register uint64_t stack_pointer asm("sp");
    sys_dma_blk_memcpy(
        chiplet_addr_transform(0x70000000),
        chiplet_addr_transform(0x1000000000ULL),
        stack_pointer - chiplet_addr_transform(0x70000000) - 1024);
    // Initialize Cluster TCDM
    for (uint32_t i = 0; i < N_CLUSTERS; i++) {
        sys_dma_blk_memcpy(
            chiplet_addr_transform(QUAD_WIDE_CLUSTER_0_TCDM_BASE_ADDR + i * CLUSTER_TCDM_SIZE),
            chiplet_addr_transform(0x1000000000ULL),
            CLUSTER_TCDM_SIZE);
    }

}
