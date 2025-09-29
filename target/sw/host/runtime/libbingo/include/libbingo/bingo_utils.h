// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once


#define ALIGN_UP(x, p) (((x) + (p)-1) & ~((p)-1))
// High32 and Low32 extraction macros
#define HIGH32(x) ((uint32_t)(((uint64_t)(x) >> 32) & 0xFFFFFFFF))
#define LOW32(x)  ((uint32_t)(((uint64_t)(x) >> 0) & 0xFFFFFFFF))
// Bit field extraction macro
// Extract bits [high:low] from variable x
#define BINGO_EXTRACT_BITS(x, high, low) (((x) >> (low)) & ((1ULL << ((high) - (low) + 1)) - 1))

// Extract a single bit at position pos from variable x
#define BINGO_EXTRACT_BIT(x, pos) (((x) >> (pos)) & 1)

// 
static inline uint64_t bingo_mcycle() {
    register uint64_t r;
    asm volatile("csrr %0, mcycle" : "=r"(r));
    return r;
}

static inline void bingo_csleep(uint32_t cycles) {
    uint32_t start = bingo_mcycle();
    while ((bingo_mcycle() - start) < cycles) {}
}
