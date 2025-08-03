// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SNRT_H
#define SNRT_H

#include <stddef.h>
#include <stdint.h>

// Occamy specific definitions
#include "occamy_defs.h"
#include "occamy_memory_map.h"
#include "sys_dma.h"

// Vendor
// l1 heap allocator
#include "o1heap.h"
// Forward declarations
#include "snrt_l1_alloc_decls.h" // snrt l1 allocator using o1heap
#include "bingo_decls.h"         // bingo offload dependency
#include "hemaia_cls_decls.h"    // CLS(cluster local storage) decl, which stores the bingo offload unit and the l1 allocator
#include "sw_mailbox_decls.h"    // sw mailbox between the host and the cluster
#include "cluster_interrupt_decls.h"
#include "global_interrupt_decls.h"
#include "memory_decls.h"
#include "sync_decls.h"
#include "team_decls.h"

// Implementation
//#include "alloc.h"

#include "cluster_interrupts.h"
#include "dma.h"
#include "dump.h"
#include "global_interrupts.h"
#include "occamy_device.h"
#include "occamy_memory.h"
#include "perf_cnt.h"
// #include "printf.h"
#include "riscv.h"
// #include "ssr.h"
#include "sync.h"
#include "team.h"
#include "uart.h"
#include "csr.h"

#include "snrt_l1_alloc.h"
#include "bingo.h"
#include "hemaia_cls.h"
#include "sw_mailbox.h"

#endif  // SNRT_H
