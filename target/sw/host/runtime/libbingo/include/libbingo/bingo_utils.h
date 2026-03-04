// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once

#define ALIGN_UP(x, p) (((x) + (p) - 1) & ~((p) - 1))
#define ALIGN_DOWN(x, p) ((x) & ~((p) - 1))

// High32 and Low32 extraction macros
#define HIGH32(x) ((uint32_t)(((uint64_t)(x) >> 32) & 0xFFFFFFFF))
#define LOW32(x)  ((uint32_t)(((uint64_t)(x) >> 0) & 0xFFFFFFFF))
// Bit field extraction macro
// Extract bits [high:low] from variable x
#define BINGO_EXTRACT_BITS(x, high, low) (((x) >> (low)) & ((1ULL << ((high) - (low) + 1)) - 1))

// Extract a single bit at position pos from variable x
#define BINGO_EXTRACT_BIT(x, pos) (((x) >> (pos)) & 1)
// Set a single bit at position pos in 32bit variable x to value v (0 or 1)
#define BINGO_SET_BIT(x, pos, v) ((x) = ((x) & ~(1U << (pos))) | ((!!(v) << (pos))))
#define BINGO_CHIPLET_LOCAL_PTR_AUTO(x) \
    ((__typeof__(&(x))) (uintptr_t)chiplet_addr_transform((uint64_t)(uintptr_t)&(x)))
// Optional: dereferenced (lvalue-style) accessor.
#define BINGO_CHIPLET_LOCAL_REF(x) (*BINGO_CHIPLET_LOCAL_AUTO(x))
#define BINGO_CHIPLET_READW(x) readw((uintptr_t)chiplet_addr_transform((uint64_t)(uintptr_t)&x))
#define BINGO_CHIPLET_READD(x) readd((uintptr_t)chiplet_addr_transform((uint64_t)(uintptr_t)&x))

// ============================================================================
// Bit field encoding macros
// ============================================================================

/// Generic macro to encode a field into a 64-bit word
/// Handles the special case where width=0 by treating it as 1 bit
/// @param value   The value to encode
/// @param width   The bit width of the field (if 0, defaults to 1)
/// @param shift   The bit position where the field starts
#define ENCODE_BITFIELD(value, width, shift) \
    (((uint64_t)(value) & ((1ULL << ((width) ? (width) : 1)) - 1)) << (shift))

/// Macro to calculate the next shift position after a field
/// Handles the special case where field_width=0 by treating it as 1 bit
/// @param current_shift Current shift position
/// @param field_width   Width of the current field (if 0, defaults to 1)
#define NEXT_SHIFT(current_shift, field_width) ((current_shift) + ((field_width) ? (field_width) : 1))

// BINGO HW Task descriptor bit positions and widths
#define TASK_TYPE_WIDTH            1
#define TASK_TYPE_SHIFT            0

#define TASK_ID_WIDTH              12
#define TASK_ID_SHIFT              NEXT_SHIFT(TASK_TYPE_SHIFT, TASK_TYPE_WIDTH)

#define ASSIGNED_CHIPLET_ID_WIDTH     8
#define ASSIGNED_CHIPLET_ID_SHIFT     NEXT_SHIFT(TASK_ID_SHIFT, TASK_ID_WIDTH)

#define ASSIGNED_CLUSTER_ID_WIDTH  N_CLUSTERS_PER_CHIPLET_WIDTH
#define ASSIGNED_CLUSTER_ID_SHIFT  NEXT_SHIFT(ASSIGNED_CHIPLET_ID_SHIFT, ASSIGNED_CLUSTER_ID_WIDTH)
#define ASSIGNED_CORE_ID_WIDTH     N_CORES_PER_CLUSTER_WIDTH
#define ASSIGNED_CORE_ID_SHIFT     NEXT_SHIFT(ASSIGNED_CLUSTER_ID_SHIFT, ASSIGNED_CLUSTER_ID_WIDTH)

#define DEP_CHECK_ENABLED_WIDTH    1
#define DEP_CHECK_ENABLED_SHIFT    NEXT_SHIFT(ASSIGNED_CORE_ID_SHIFT, ASSIGNED_CORE_ID_WIDTH)

#define DEP_CHECK_CODE_WIDTH       N_CORES_PER_CLUSTER
#define DEP_CHECK_CODE_SHIFT       NEXT_SHIFT(DEP_CHECK_ENABLED_SHIFT, DEP_CHECK_ENABLED_WIDTH)

#define DEP_SET_ENABLED_WIDTH      1
#define DEP_SET_ENABLED_SHIFT      NEXT_SHIFT(DEP_CHECK_CODE_SHIFT, DEP_CHECK_CODE_WIDTH)

#define DEP_SET_ALL_CHIPLET_WIDTH  1
#define DEP_SET_ALL_CHIPLET_SHIFT  NEXT_SHIFT(DEP_SET_ENABLED_SHIFT, DEP_SET_ENABLED_WIDTH)

#define DEP_SET_CODE_WIDTH         N_CORES_PER_CLUSTER
#define DEP_SET_CODE_SHIFT         NEXT_SHIFT(DEP_SET_ALL_CHIPLET_SHIFT, DEP_SET_ALL_CHIPLET_WIDTH)

#define DEP_SET_CHIPLET_ID_WIDTH   N_CHIPLETS_WIDTH
#define DEP_SET_CHIPLET_ID_SHIFT   NEXT_SHIFT(DEP_SET_CODE_SHIFT, DEP_SET_CODE_WIDTH)

#define DEP_SET_CLUSTER_ID_WIDTH   N_CLUSTERS_PER_CHIPLET_WIDTH
#define DEP_SET_CLUSTER_ID_SHIFT   NEXT_SHIFT(DEP_SET_CHIPLET_ID_SHIFT, DEP_SET_CHIPLET_ID_WIDTH)

// Mcycle
static inline uint64_t bingo_mcycle() {
    register uint64_t r;
    asm volatile("csrr %0, mcycle" : "=r"(r));
    return r;
}
// Sleep for a number of cycles
static inline void bingo_csleep(uint32_t cycles) {
    uint32_t start = bingo_mcycle();
    while ((bingo_mcycle() - start) < cycles) {}
}
