// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Cyril Koenig <cykoenig@iis.ee.ethz.ch>
// Fanchen Kong <fanchen.kong@kuleuven.be>

#pragma once

#include <inttypes.h>

#include "o1heap.h"

// The current hierachy is
// L1: Cluster TCDM Memory
// L2: Narrow SoC Shared Memory
// L3: Wide SoC Shared Memory
// The L2 will be mainly on the mailbox and control
// L3 will be on the data
// In the future we will have the FPGA D2D Link + external SRAM as a larger L3
extern struct O1HeapInstance *l2_heap_manager;
extern uint64_t l2_heap_start_phy, l2_heap_start_virt, l2_heap_size;
extern struct O1HeapInstance *l3_heap_manager;
extern uint64_t l3_heap_start_phy, l3_heap_start_virt, l3_heap_size;