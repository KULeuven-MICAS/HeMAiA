// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Nicole Narr <narrn@student.ethz.ch>
// Christopher Reinwardt <creinwar@student.ethz.ch>
// Paul Scheffler <paulsc@iis.ee.ethz.ch>
// Cyril Koenig <cykoenig@iis.ee.ethz.ch>

#pragma once

#include <stdint.h>

static inline void fence() {
    asm volatile("fence" ::: "memory");
}

static inline void fencei() {
    asm volatile("fence.i" ::: "memory");
}

static inline void wfi() {
    asm volatile("wfi" ::: "memory");
}
// Get cycle count since reset
static inline uint64_t get_mcycle() {
    uint64_t mcycle;
    asm volatile("csrr %0, mcycle" : "=r"(mcycle)::"memory");
    return mcycle;
}

#define MIN(a, b) (((a) <= (b)) ? (a) : (b))
#define ALIGN_UP(x, p) (((x) + (p)-1) & ~((p)-1))