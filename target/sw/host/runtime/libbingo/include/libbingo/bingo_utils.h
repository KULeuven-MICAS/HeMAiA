// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once

#include <stdint.h>
// The task descriptor layout below is derived from the platform parameters
// (N_CORES_PER_CLUSTER, N_CLUSTERS_PER_CHIPLET, BINGO_DEP_TAG_WIDTH, ...) and is checked
// at compile time, so occamy.h has to be visible here regardless of which header pulled
// this one in first. occamy.h is #pragma once, so the duplicate include is free.
#include "occamy.h"

#define ALIGN_UP(x, p) (((x) + (p) - 1) & ~((p) - 1))
#define ALIGN_DOWN(x, p) ((x) & ~((p) - 1))

// High32 and Low32 extraction macros
#define HIGH32(x) ((uint32_t)(((uint64_t)(x) >> 32) & 0xFFFFFFFF))
#define LOW32(x)  ((uint32_t)(((uint64_t)(x) >> 0) & 0xFFFFFFFF))

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
// Compile-time helpers
// ============================================================================

// _Static_assert is C11, but GCC accepts it in gnu89/gnu99 too (the host runtime builds
// with -std=gnu99), so only a non-GCC pre-C11 compiler needs the array-size fallback.
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)) || defined(__GNUC__)
#define BINGO_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
#define BINGO_STATIC_ASSERT(cond, msg) \
    extern char bingo_static_assert_failed[(cond) ? 1 : -1]
#endif

/// ceil(log2(n)) as an integer constant expression: the preprocessor cannot compute a
/// logarithm, and these widths must be usable in _Static_assert and in shift counts.
#define BINGO_CLOG2(n)                                                         \
    ((n) <=     1 ?  0 : (n) <=     2 ?  1 : (n) <=     4 ?  2 :               \
     (n) <=     8 ?  3 : (n) <=    16 ?  4 : (n) <=    32 ?  5 :               \
     (n) <=    64 ?  6 : (n) <=   128 ?  7 : (n) <=   256 ?  8 :               \
     (n) <=   512 ?  9 : (n) <=  1024 ? 10 : (n) <=  2048 ? 11 :               \
     (n) <=  4096 ? 12 : (n) <=  8192 ? 13 : (n) <= 16384 ? 14 :               \
     (n) <= 32768 ? 15 : 16)

/// The RTL's cf_math_pkg::idx_width(): bits needed to index n things, with a floor of 1.
/// BUG FIX (4): the layout used to use plain clog2 (occamy.h's *_WIDTH defines), which is
/// 0 for a single cluster. The RTL packed struct still has a real 1-bit field there, so a
/// 0-width entry here misplaced every field above it on a 1-cluster chiplet.
#define BINGO_IDX_WIDTH(n) (((n) > 1) ? BINGO_CLOG2(n) : 1)

/// Bits available in the C carrier of struct member `field` of struct `type`.
/// `sizeof` of a member through a null pointer is never evaluated (the offsetof idiom), so
/// this stays an integer constant expression and is usable in _Static_assert.
#define BINGO_CARRIER_BITS(type, field) (8u * (unsigned)sizeof(((type *)0)->field))

/// Fail the BUILD if the carrier of `type::field` cannot hold `width` bits.
/// Descriptor field widths scale with the platform (BINGO_NCORES_HW, chip id width, dep tag
/// width), but the C struct that feeds the encoder has fixed-size members. When a width
/// outgrows its member the value is truncated on the way INTO the encoder, where no bit
/// position is wrong and nothing faults - the dependency edge simply disappears. This turns
/// that into a compile error at the growth step that causes it.
#define BINGO_ASSERT_CARRIER_FITS(type, field, width)                               \
    BINGO_STATIC_ASSERT(BINGO_CARRIER_BITS(type, field) >= (unsigned)(width),       \
                        "task descriptor carrier '" #field "' is narrower than "    \
                        #width " - widen the struct member")

// ============================================================================
// BINGO HW manager task descriptor container
// ============================================================================

/// Width of the descriptor container in bits == bingo_hw_manager_top TaskDescBusWidth.
/// The generated occamy.h carries the value the RTL was elaborated with; the #ifndef keeps
/// this header usable on its own (unit tests, host tools) at the current design point.
#ifndef BINGO_TASK_DESC_WIDTH
#define BINGO_TASK_DESC_WIDTH 128
#endif

/// The descriptor is carried as 64-bit words, LEAST-SIGNIFICANT WORD FIRST: the task list
/// in memory stays a uint64_t array, and the RTL fetch master
/// (bingo_hw_manager_task_queue_master) reads the lowest address into the descriptor's low
/// bits, so ascending address == ascending significance. Swapping the two halves here
/// swaps them in every descriptor the HW manager fetches.
/// occamy.h also emits this one; #ifndef so the generated value wins and the two can
/// never collide, with the consistency check below catching a disagreement.
#ifndef BINGO_TASK_DESC_WORDS
#define BINGO_TASK_DESC_WORDS (BINGO_TASK_DESC_WIDTH / 64)
#endif
/// Stride between two descriptors in the task list, in bytes.
#ifndef BINGO_TASK_DESC_BYTES
#define BINGO_TASK_DESC_BYTES (BINGO_TASK_DESC_WIDTH / 8)
#endif

BINGO_STATIC_ASSERT(BINGO_TASK_DESC_WORDS == BINGO_TASK_DESC_WIDTH / 64,
                    "BINGO_TASK_DESC_WORDS disagrees with BINGO_TASK_DESC_WIDTH");
BINGO_STATIC_ASSERT(BINGO_TASK_DESC_BYTES == BINGO_TASK_DESC_WIDTH / 8,
                    "BINGO_TASK_DESC_BYTES disagrees with BINGO_TASK_DESC_WIDTH");

/// Bytes a task list of `n` descriptors occupies == what bingo_l3_alloc must be asked for.
/// The stride is the descriptor CONTAINER width, not 8: at BINGO_TASK_DESC_WIDTH 128 a list
/// sized with sizeof(uint64_t) per task is half the memory the HW manager will read, and
/// the fetch master walks off the end of the allocation on the second half of the last
/// descriptor. Use this (or bingo_task_desc_store's indexing) rather than open-coding it;
/// see the coordination note on bingo_task_desc_store in bingo_api.h for the emitter side.
#define BINGO_TASK_DESC_LIST_BYTES(n) ((uint64_t)(n) * (uint64_t)BINGO_TASK_DESC_BYTES)

typedef struct {
    uint64_t w[BINGO_TASK_DESC_WORDS];  // w[0] holds descriptor bits [63:0]
} bingo_task_desc_t;

/// Deposit `value` into bits [shift +: width] of a descriptor.
/// The layout is 65+ bits, so a field DOES straddle the 64-bit word boundary (dep_set_tag
/// at the current design point) and the write has to be split across two words.
static inline void bingo_task_desc_set_field(bingo_task_desc_t *desc, uint64_t value,
                                             unsigned width, unsigned shift) {
    if (width == 0U) return;
    const uint64_t mask = (width >= 64U) ? ~(uint64_t)0 : ((1ULL << width) - 1ULL);
    const uint64_t val  = value & mask;
    const unsigned word = shift / 64U;
    const unsigned off  = shift % 64U;
    desc->w[word] |= val << off;
    // width <= 64, so a straddle implies off > 0 and (64 - off) is never a 64-bit shift
    // (which would be undefined behaviour).
    if ((off + width) > 64U) {
        desc->w[word + 1U] |= val >> (64U - off);
    }
}

/// Read bits [shift +: width] back out of a descriptor, straddle included.
static inline uint64_t bingo_task_desc_get_field(const bingo_task_desc_t *desc,
                                                 unsigned width, unsigned shift) {
    if (width == 0U) return 0ULL;
    const uint64_t mask = (width >= 64U) ? ~(uint64_t)0 : ((1ULL << width) - 1ULL);
    const unsigned word = shift / 64U;
    const unsigned off  = shift % 64U;
    uint64_t val = desc->w[word] >> off;
    if ((off + width) > 64U) {
        val |= desc->w[word + 1U] << (64U - off);
    }
    return val & mask;
}

/// Place a field into the descriptor `desc` (an lvalue). This used to return a uint64_t
/// that the caller OR-ed in; a 128-bit descriptor does not fit in a C scalar (and
/// __uint128_t is not available to the 32-bit device builds that see this header), so it
/// now writes through instead.
#define ENCODE_BITFIELD(desc, value, width, shift)                             \
    bingo_task_desc_set_field(&(desc), (uint64_t)(value), (unsigned)(width),   \
                              (unsigned)(shift))

/// Extract bits [high:low] from the descriptor `desc`.
#define BINGO_EXTRACT_BITS(desc, high, low)                                    \
    bingo_task_desc_get_field(&(desc), (unsigned)((high) - (low) + 1),         \
                              (unsigned)(low))

/// Position of the next field: every field advances the cursor by its OWN width.
/// BUG FIX (1): one call site used to pass the *following* field's width, which shifted
/// everything from assigned_cluster_id upwards. The old "0 width counts as 1" coercion is
/// gone as well - it papered over bug 4 and desynced the cursor from $bits() of the RTL
/// packed struct; widths are >= 1 by construction now and asserted below.
#define NEXT_SHIFT(current_shift, field_width) ((current_shift) + (field_width))

/// Cores the HW manager sees per cluster: the cluster's own cores plus the host CVA6,
/// which acts as an extra core in cluster 0 (occamy_quad_ctrl.sv.tpl:
/// BINGO_HW_MANAGER_NR_CORE_PER_CLUSTER = NrCoresPerCluster[0] + 1).
/// BUG FIX (3): the dep codes are one bit per such core and the core id indexes them, so
/// both must be sized from this, not from N_CORES_PER_CLUSTER, which excludes the host.
/// occamy.h emits this too; #ifndef so the generated value wins, checked just below.
#ifndef BINGO_NCORES_HW
#define BINGO_NCORES_HW (N_CORES_PER_CLUSTER + 1)
#endif
BINGO_STATIC_ASSERT(BINGO_NCORES_HW == N_CORES_PER_CLUSTER + 1,
                    "BINGO_NCORES_HW must be N_CORES_PER_CLUSTER + 1 (the host core)");

/// RTL bingo_hw_manager_top ChipIdWidth (cfg hemaia_multichip.chip_id_width). Both
/// chiplet-id fields are this wide because they carry the D2D ROUTING id ((x << 4) | y),
/// not an index into N_CHIPLETS.
/// GENERATED VALUE WINS: occamy.py emits BINGO_CHIP_ID_WIDTH into the generated occamy.h
/// from the same cfg key that elaborates the RTL, exactly as it does for
/// BINGO_TASK_DESC_WIDTH / BINGO_NCORES_HW / BINGO_DEP_TAG_WIDTH. The literal below is a
/// DOCUMENTED FALLBACK, not an independent truth: it is the RTL parameter default and it
/// only applies when this header is compiled standalone (unit tests, host tools) or against
/// a platform header generated before the define existed.
#ifndef BINGO_CHIP_ID_WIDTH
#define BINGO_CHIP_ID_WIDTH 8
#endif
BINGO_STATIC_ASSERT(BINGO_CHIP_ID_WIDTH >= 1 && BINGO_CHIP_ID_WIDTH <= 32,
                    "BINGO_CHIP_ID_WIDTH must be a sane chip_id_t width (1..32)");
#if defined(N_CHIPLETS_X) && defined(N_CHIPLETS_Y)
/// Largest D2D routing id this configuration can name: chip_id = (x << 4) | y, so the
/// far corner of the array is the widest value the two chiplet-id fields ever carry. If
/// this does not fit, a dep set aimed at that chiplet is delivered to the wrong one.
#define BINGO_MAX_CHIP_ROUTING_ID ((((N_CHIPLETS_X) - 1) << 4) | ((N_CHIPLETS_Y) - 1))
BINGO_STATIC_ASSERT(BINGO_CHIP_ID_WIDTH >= BINGO_CLOG2(BINGO_MAX_CHIP_ROUTING_ID + 1),
                    "BINGO_CHIP_ID_WIDTH cannot hold this array's largest chip routing id "
                    "((N_CHIPLETS_X-1) << 4 | (N_CHIPLETS_Y-1))");
#endif

// BINGO HW Task descriptor bit positions and widths.
// MUST mirror the RTL packed struct bingo_hw_manager_task_desc_t (and the Python
// bingo_pack_node) bit-for-bit. A packed struct's LAST-declared member is the LSB, so
// LSB->MSB the order is cond_exec_invert, cond_exec_group_id, cond_exec_en, task_type,
// task_id, assigned_{chiplet,cluster,core}_id, then dep_check_info {en, code, tag} and
// dep_set_info {en, all_chiplet, chiplet_id, cluster_id, code, tag}.
// DARTS Tier 1: Conditional Execution fields
#define COND_EXEC_INVERT_WIDTH     1
#define COND_EXEC_INVERT_SHIFT     0

// RTL bingo_hw_manager_task_desc_t declares cond_exec_group_id as a hard `logic [4:0]`: it
// indexes one of the 32 CERF group bits in the manager's 32-bit cerf_state register. Same
// rule as BINGO_CHIP_ID_WIDTH - if the RTL ever parameterises the group count, occamy.py
// emits BINGO_COND_EXEC_GROUP_ID_WIDTH into occamy.h and it wins here with no edit to this
// file; the literal is the documented fallback for a standalone / pre-define build.
#ifndef BINGO_COND_EXEC_GROUP_ID_WIDTH
#define BINGO_COND_EXEC_GROUP_ID_WIDTH 5
#endif
#define COND_EXEC_GROUP_ID_WIDTH   BINGO_COND_EXEC_GROUP_ID_WIDTH
#define COND_EXEC_GROUP_ID_SHIFT   NEXT_SHIFT(COND_EXEC_INVERT_SHIFT, COND_EXEC_INVERT_WIDTH)

#define COND_EXEC_EN_WIDTH         1
#define COND_EXEC_EN_SHIFT         NEXT_SHIFT(COND_EXEC_GROUP_ID_SHIFT, COND_EXEC_GROUP_ID_WIDTH)

// Task type: 2 bits (00=normal, 01=dummy, 10=gating)
#define TASK_TYPE_WIDTH            2
#define TASK_TYPE_SHIFT            NEXT_SHIFT(COND_EXEC_EN_SHIFT, COND_EXEC_EN_WIDTH)

// RTL bingo_hw_manager_top TaskIdWidth. Same rule as BINGO_CHIP_ID_WIDTH: the generated
// occamy.h value wins when occamy.py emits BINGO_TASK_ID_WIDTH, and the literal below is
// the documented fallback (the RTL parameter default), never an independent number.
#ifndef BINGO_TASK_ID_WIDTH
#define BINGO_TASK_ID_WIDTH 12
#endif
#define TASK_ID_WIDTH              BINGO_TASK_ID_WIDTH
#define TASK_ID_SHIFT              NEXT_SHIFT(TASK_TYPE_SHIFT, TASK_TYPE_WIDTH)

#define ASSIGNED_CHIPLET_ID_WIDTH  BINGO_CHIP_ID_WIDTH
#define ASSIGNED_CHIPLET_ID_SHIFT  NEXT_SHIFT(TASK_ID_SHIFT, TASK_ID_WIDTH)

#define ASSIGNED_CLUSTER_ID_WIDTH  BINGO_IDX_WIDTH(N_CLUSTERS_PER_CHIPLET)
// BUG FIX (1) lives here: this advanced the cursor by ASSIGNED_CLUSTER_ID_WIDTH over the
// ChipIdWidth-wide chiplet id, so every field from here up sat
// (ChipIdWidth - cluster width) bits too low - 6 at 4 clusters, 8 at one.
#define ASSIGNED_CLUSTER_ID_SHIFT  NEXT_SHIFT(ASSIGNED_CHIPLET_ID_SHIFT, ASSIGNED_CHIPLET_ID_WIDTH)

#define ASSIGNED_CORE_ID_WIDTH     BINGO_IDX_WIDTH(BINGO_NCORES_HW)
#define ASSIGNED_CORE_ID_SHIFT     NEXT_SHIFT(ASSIGNED_CLUSTER_ID_SHIFT, ASSIGNED_CLUSTER_ID_WIDTH)

#define DEP_CHECK_ENABLED_WIDTH    1
#define DEP_CHECK_ENABLED_SHIFT    NEXT_SHIFT(ASSIGNED_CORE_ID_SHIFT, ASSIGNED_CORE_ID_WIDTH)

// One bit per core the HW manager tracks, host core included (bug 3).
#define DEP_CHECK_CODE_WIDTH       BINGO_NCORES_HW
#define DEP_CHECK_CODE_SHIFT       NEXT_SHIFT(DEP_CHECK_ENABLED_SHIFT, DEP_CHECK_ENABLED_WIDTH)

// Per-edge identity tag (EnableTaggedDeps). MUST match the hw_manager DepTagWidth,
// so it comes from the generated occamy.h (cfg s1_quadrant.dep_tag_width) rather than
// being hardcoded here. One tag sits at the MSB of each dep_*_info struct (right after
// the dep code).
#define DEP_TAG_WIDTH              BINGO_DEP_TAG_WIDTH

#define DEP_CHECK_TAG_WIDTH        DEP_TAG_WIDTH
#define DEP_CHECK_TAG_SHIFT        NEXT_SHIFT(DEP_CHECK_CODE_SHIFT, DEP_CHECK_CODE_WIDTH)

#define DEP_SET_ENABLED_WIDTH      1
#define DEP_SET_ENABLED_SHIFT      NEXT_SHIFT(DEP_CHECK_TAG_SHIFT, DEP_CHECK_TAG_WIDTH)

#define DEP_SET_ALL_CHIPLET_WIDTH  1
#define DEP_SET_ALL_CHIPLET_SHIFT  NEXT_SHIFT(DEP_SET_ENABLED_SHIFT, DEP_SET_ENABLED_WIDTH)

// ChipIdWidth, not clog2 of the chiplet COUNT: the field carries the same routing id as
// assigned_chiplet_id and the RTL types both as chip_id_t. (clog2 of the count is 0 on a
// single-chiplet cfg, which silently narrows the field to nothing.)
#define DEP_SET_CHIPLET_ID_WIDTH   BINGO_CHIP_ID_WIDTH
#define DEP_SET_CHIPLET_ID_SHIFT   NEXT_SHIFT(DEP_SET_ALL_CHIPLET_SHIFT, DEP_SET_ALL_CHIPLET_WIDTH)

#define DEP_SET_CLUSTER_ID_WIDTH   BINGO_IDX_WIDTH(N_CLUSTERS_PER_CHIPLET)
#define DEP_SET_CLUSTER_ID_SHIFT   NEXT_SHIFT(DEP_SET_CHIPLET_ID_SHIFT, DEP_SET_CHIPLET_ID_WIDTH)

#define DEP_SET_CODE_WIDTH         BINGO_NCORES_HW
#define DEP_SET_CODE_SHIFT         NEXT_SHIFT(DEP_SET_CLUSTER_ID_SHIFT, DEP_SET_CLUSTER_ID_WIDTH)

#define DEP_SET_TAG_WIDTH          DEP_TAG_WIDTH
#define DEP_SET_TAG_SHIFT          NEXT_SHIFT(DEP_SET_CODE_SHIFT, DEP_SET_CODE_WIDTH)

/// Total bits the layout occupies == $bits(bingo_hw_manager_task_desc_t) in the RTL.
/// Everything above it up to BINGO_TASK_DESC_WIDTH is the reserved padding the RTL calls
/// ReservedBitsForTaskDesc, and SW leaves it zero.
#define BINGO_TASK_DESC_LAYOUT_BITS NEXT_SHIFT(DEP_SET_TAG_SHIFT, DEP_SET_TAG_WIDTH)

// Without these the C layout and the RTL struct can disagree on width silently.
BINGO_STATIC_ASSERT(BINGO_TASK_DESC_WIDTH % 64 == 0,
                    "BINGO_TASK_DESC_WIDTH must be a whole number of 64-bit words");
BINGO_STATIC_ASSERT(BINGO_TASK_DESC_LAYOUT_BITS <= BINGO_TASK_DESC_WIDTH,
                    "Task descriptor layout overflows BINGO_TASK_DESC_WIDTH: raise it here "
                    "and TaskDescBusWidth in bingo_hw_manager_top, or shrink a field");
// The two widths this header mirrors from RTL declarations rather than from a cfg key.
// They cannot be checked against a generated number until occamy.py emits one, so check
// them against what the RTL they mirror can physically accept.
BINGO_STATIC_ASSERT(TASK_ID_WIDTH >= 1 && TASK_ID_WIDTH <= 32,
                    "TASK_ID_WIDTH must match bingo_hw_manager_top TaskIdWidth and stay "
                    "inside the 32-bit task-id path SW uses to talk to it");
BINGO_STATIC_ASSERT(COND_EXEC_GROUP_ID_WIDTH >= 1 && COND_EXEC_GROUP_ID_WIDTH <= 5,
                    "cond_exec_group_id indexes the HW manager's 32-bit cerf_state, so it "
                    "is at most 5 bits wide");
// A zero-width field would both vanish from the encoding and stop advancing the cursor,
// silently shifting the whole upper half of the descriptor.
BINGO_STATIC_ASSERT(ASSIGNED_CLUSTER_ID_WIDTH >= 1 && ASSIGNED_CORE_ID_WIDTH >= 1 &&
                    DEP_SET_CLUSTER_ID_WIDTH >= 1 && DEP_CHECK_CODE_WIDTH >= 1 &&
                    DEP_SET_CODE_WIDTH >= 1 && DEP_TAG_WIDTH >= 1,
                    "Every task descriptor field must be at least 1 bit wide");

// Per-Kernel Scratchpad: defined in heterogeneous_runtime.h (shared between host/device)
// bingo_kernel_scratchpad_t, BINGO_KERNEL_SCRATCHPAD_SIZE, BINGO_SP_PROFILE

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
