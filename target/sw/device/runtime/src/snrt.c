// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#include "snrt.h"

#include "o1heap.c"
#include "cluster_interrupts.c"
#include "dma.c"
#include "global_interrupts.c"
#include "occamy_device.c"
#include "occamy_memory.c"
//#include "occamy_start.c"
#include "hemaia_start.c"
#include "sync.c"
#include "sys_dma.c"
#include "team.c"
#include "uart.c"
#include "snrt_l1_alloc.c"
#include "bingo.c"
#include "hemaia_cls.c"
#include "sw_mailbox.c"
// SNAX runtime
#include "snax_xdma_lib.c"
