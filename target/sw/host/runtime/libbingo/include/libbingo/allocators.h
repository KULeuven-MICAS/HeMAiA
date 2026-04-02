// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once

#include <inttypes.h>

#include "bingo_alloc.h"
#include "occamy_memory_map.h"

// The current hierachy is
// L1: Cluster TCDM Memory
// L2: Narrow SoC Shared Memory
// L3: Wide SoC Shared Memory
// L3 will be on the data
// In the future we will have the FPGA D2D Link + external SRAM as a larger L3
#define SPM_WIDE_ALIGNMENT 1024
extern uint64_t __l3_heap_start;
extern uint64_t __wide_spm_end;

extern struct BingoHeapInstance *l2_heap_manager;
extern uint64_t l2_heap_start, l2_heap_size;

extern struct BingoHeapInstance *l3_heap_manager;
extern uint64_t l3_heap_start, l3_heap_size;

// Mempool chiplet allocator
// The mempool chiplet provides large external SRAM accessible via D2D links.
// From the compute chiplet, mempool addresses are translated using
// chiplet_addr_transform_loc(mempool_x, mempool_y, local_addr).
// The heap is initialized on the mempool chip's SPM Wide region.
//
// Default heap layout:
//   SPM Wide base (0x8000_0000) + MEMPOOL_HEAP_OFFSET = heap start
//   The first 64MB is reserved for static data (weights loaded at init).
//   The heap manages the remaining capacity for dynamic allocations.
#define MEMPOOL_HEAP_OFFSET (64 * 1024 * 1024)  // 64MB offset from SPM base
extern uint8_t mempool_chip_loc_x, mempool_chip_loc_y;
