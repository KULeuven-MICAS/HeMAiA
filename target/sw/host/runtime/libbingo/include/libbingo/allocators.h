// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once

#include <inttypes.h>

#include "o1heap.h"
#include "o1heap32.h"
#include "occamy_memory_map.h"

// The current hierachy is
// L1: Cluster TCDM Memory
// L2: Narrow SoC Shared Memory
// L3: Wide SoC Shared Memory
// L3 will be on the data
// In the future we will have the FPGA D2D Link + external SRAM as a larger L3
extern uint64_t __l3_heap_start;

extern struct O1HeapInstance *l2_heap_manager;
extern uint64_t l2_heap_start, l2_heap_size;

extern struct O1HeapInstance *l3_heap_manager;
extern uint64_t l3_heap_start, l3_heap_size;