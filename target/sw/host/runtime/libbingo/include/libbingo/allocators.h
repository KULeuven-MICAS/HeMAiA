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
//
// Memchip layout: split the cfg-derived total in half — low half is the
// data region (static weights/inputs), high half is the heap region.
//   [SPM_WIDE_BASE +  0,                  +MEMPOOL_DATA_BYTES)  data
//   [SPM_WIDE_BASE + MEMPOOL_DATA_BYTES,  +MEMPOOL_TOTAL_BYTES) heap
//
// MEMPOOL_TOTAL_SIZE is injected from the cfg via occamy_memory_map.h
// (`hemaia_mem_chip[0].mem_size` in the multichip cfg). Use
// bingo_get_mempool_data_base() for the data region start and
// bingo_get_mempool_heap_manager() / bingo_mempool_alloc() for the heap.
#define MEMPOOL_TOTAL_BYTES        MEMPOOL_TOTAL_SIZE
#define MEMPOOL_DATA_BYTES         (MEMPOOL_TOTAL_BYTES / 2)
#define MEMPOOL_DATA_BASE_OFFSET   0
#define MEMPOOL_HEAP_BASE_OFFSET   MEMPOOL_DATA_BYTES
#define MEMPOOL_HEAP_BYTES         (MEMPOOL_TOTAL_BYTES - MEMPOOL_HEAP_BASE_OFFSET)
extern struct BingoHeapInstance *mempool_heap_manager;
extern uint64_t mempool_heap_base, mempool_heap_capacity;
