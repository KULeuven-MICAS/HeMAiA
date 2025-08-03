// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once

static inline uint64_t bingo_mcycle() {
    register uint64_t r;
    asm volatile("csrr %0, mcycle" : "=r"(r));
    return r;
}

static inline void bingo_csleep(uint32_t cycles) {
    uint32_t start = bingo_mcycle();
    while ((bingo_mcycle() - start) < cycles) {}
}