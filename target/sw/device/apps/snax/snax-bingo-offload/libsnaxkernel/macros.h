// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
//
// Common SNAX kernel-library scaffolding: symbol-table types and macros,
// the BINGO SW-guard helper, and lightweight bit/debug utilities. Every
// kernel partial under libsnaxkernel/ pulls this in; snax_kernel_lib.h
// re-includes it and then aggregates the kernel partials.

#pragma once

#include "snrt.h"
#include "heterogeneous_runtime.h"        // bingo_kernel_scratchpad_t, BINGO_GET_SCRATCHPAD
#include <libbingo/device_kernel_args.h>  // BINGO_KERNEL_ARGS_TRAILER + every args struct typedef

// -------------------------------------------------------------------------
// Debug prints
// -------------------------------------------------------------------------
#ifdef IDMA_DEBUG
#define IDMA_DEBUG_PRINT(...) printf_safe(__VA_ARGS__)
#else
#define IDMA_DEBUG_PRINT(...)
#endif

#ifdef XDMA_DEBUG
#define XDMA_DEBUG_PRINT(...) printf_safe(__VA_ARGS__)
#else
#define XDMA_DEBUG_PRINT(...)
#endif

#ifdef VERSACORE_DEBUG
#define VERSACORE_DEBUG_PRINT(...) printf_safe(__VA_ARGS__)
#else
#define VERSACORE_DEBUG_PRINT(...)
#endif

// -------------------------------------------------------------------------
// SNAX symbol table
// -------------------------------------------------------------------------
#define SNAX_LIB_NAME_MAX_LEN 64
typedef struct __attribute__((packed)) {
    char name[SNAX_LIB_NAME_MAX_LEN]; // function name
    uint32_t addr;                    // function addr
} snax_symbol_t;

#define SNAX_SYMTAB_END_FN_NAME "SYMTAB_END"
#define SNAX_SYMTAB_END_FN_ADDR (uint32_t)(0xBAADF00D)

#define SNAX_LIB_DEFINE      __attribute__((used))
#define SNAX_SYMTAB_SECTION  __attribute__((used, section(".snax_symtab")))
#define SNAX_EXPORT_FUNC(name) {#name, (uint32_t)name}
#define SNAX_SYMTAB_END {SNAX_SYMTAB_END_FN_NAME, SNAX_SYMTAB_END_FN_ADDR}

// -------------------------------------------------------------------------
// Misc helpers
// -------------------------------------------------------------------------
inline uint64_t make_u64(uint32_t hi, uint32_t lo) {
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

#define EXTRACT_BIT(x, pos) (((x) >> (pos)) & 1)

// snrt_l1_malloc + null-check + debug print, with the conventional bail-out
// (return from the surrounding kernel function on failure).
#define BINGO_L1_ALLOC_OR_RETURN(_var, _bytes, _label_str) \
    uint32_t *_var = (uint32_t *)snrt_l1_malloc(_bytes); \
    if (!_var) { \
        printf_safe("ERROR: " _label_str " alloc failed!\r\n"); \
        return; \
    }

// bingo_kernel_scratchpad_t, BINGO_KERNEL_SCRATCHPAD_SIZE, BINGO_SP_PROFILE,
// BINGO_GET_SCRATCHPAD all live in heterogeneous_runtime.h (included above).

// -------------------------------------------------------------------------
// SW Guard — runtime conditional execution for CDFGs
// -------------------------------------------------------------------------
// Bingo schedules conditional data-flow graphs (CDFGs): a node may be
// statically present in the graph but only conditionally run at runtime,
// based on a value computed by an upstream "gating" node. MoE expert
// routing (one shared accelerator dispatching to N experts) is the
// motivating use case, but the same primitive covers any CDFG pattern —
// if/else branches, early-exit on a condition, dynamic-shape pruning,
// masked subgraphs — anything where "should this node run?" is only
// known at runtime.
//
// Two-tier gating
// ---------------
// Conditionality is enforced in two cooperating tiers:
//
//  Tier 1 — HW gating (CERF), implemented in bingo_hw_manager:
//    src/bingo_hw_manager_cond_exec_controller.sv defines a per-chiplet
//    Conditional Execution Register File (CERF) — a small bitmask
//    (16 groups by default) consulted combinationally by the scheduler
//    on every conditional task. The user marks DFG edges as `cond=True`
//    and the compiler pass `bingo_compile_conditional_regions()`
//    (sw/bingo_frontend.py) auto-promotes the source node to a gating
//    task (task_type=2) and assigns CERF group IDs to the conditional
//    targets. On gating-task completion the HW writes a fresh CERF
//    bitmask; conditional tasks whose group bit is 0 are dropped at the
//    scheduler — they never leave the dependency matrix and never
//    reach a core. Skipped tasks still propagate dummy dependency
//    signals so the graph stays consistent.
//
//  Tier 2 — SW guard (this macro):
//    CERF only has 16 groups, so when the conditional fan-out exceeds
//    that (e.g. >16 experts sharing a CERF group, MoE with 64 experts,
//    fine-grained masked nodes) the HW can only fire the whole group.
//    The SW guard then runs *inside* each dispatched task and decides
//    per-node whether to do real work or return immediately. The HW
//    gating saves the dispatch + arg-DMA cost for whole groups; the
//    SW guard saves the kernel-body cost for individual nodes within
//    a fired group. Both layers preserve the dependency invariants —
//    BINGO_RET_SUCC means "I'm done", whether or not real work ran.
//
// Wire layout — every BINGO core-level args struct ends with the
// BINGO_KERNEL_ARGS_TRAILER triple (defined in libbingo/device_kernel_args.h):
//
//   gating_sp_addr   (SW guard; 0 disables this node's guard)
//   cond_node_index  (this node's slot in the activation array)
//   scratchpad_ptr   (this kernel's bingo_kernel_scratchpad_t*)
//
// Both helpers below take the args struct *type* directly and look up each
// field by name via `offsetof(args_t, field)`. Reordering trailer fields
// or appending more args struct fields can never silently shift the
// scratchpad slot — the named anchor stays correct.
//
// SW-guard protocol (per kernel invocation that survived Tier 1):
//   1. Upstream gating kernel writes uint8_t activation[num_nodes]
//      (1 = run downstream node N, 0 = skip).
//   2. Gating kernel stores the array pointer in its own
//      scratchpad.return_value before returning.
//   3. Each guarded downstream node receives `gating_sp_addr` (pointer to
//      the gating kernel's scratchpad) and its `cond_node_index` (its slot
//      in the activation array) via its args trailer.
//   4. BINGO_SW_GUARD_CHECK reads gating_sp->return_value to find the
//      activation array, indexes activation[cond_node_index], and
//      early-returns BINGO_RET_SUCC if it's 0.
//
// When gating_sp_addr == 0 the guard short-circuits to a no-op (single
// load + branch-not-taken), so unguarded nodes pay no measurable cost.
//
// Usage in a guarded kernel:
//   SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_dummy(void *arg) {
//       BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_dummy_args_t);
//       // ↑ if this node is gated off, we already returned BINGO_RET_SUCC.
//       bingo_kernel_scratchpad_t *sp =
//           BINGO_GET_SP(arg, __snax_bingo_kernel_dummy_args_t);
//       // … real work …
//       sp->return_value = result;
//       return BINGO_RET_SUCC;
//   }
//
// Worked example — gating kernel selects 2 of 4 downstream nodes that
// share a single CERF group (so HW already let all four through):
//
//                    [Gating]                  activation = {1, 0, 1, 0}
//                   /  |   |  \                gating_sp->return_value = &activation
//                  v   v   v   v
//               [N0] [N1] [N2] [N3]            cond_node_index = 0,1,2,3
//
//   Each Ni's args trailer holds (gating_sp_addr, i, ...). At entry Ni
//   does BINGO_SW_GUARD_CHECK(arg, args_t):
//     - N0: activation[0]==1 → guard returns nothing, kernel runs.
//     - N1: activation[1]==0 → guard returns BINGO_RET_SUCC immediately.
//     - N2: activation[2]==1 → runs.
//     - N3: activation[3]==0 → skipped.
//   Result: dynamic 2-of-4 routing without recompiling or rescheduling
//   the graph. If N0..N3 were in *different* CERF groups, Tier 1 would
//   already have skipped N1 and N3 at the scheduler and the SW guard
//   would never fire.
#define BINGO_SW_GUARD_CHECK(arg, args_t) \
    do { \
        uint32_t _gsp = *(uint32_t*)((char*)(arg) + offsetof(args_t, gating_sp_addr)); \
        if (_gsp) { \
            bingo_kernel_scratchpad_t* _gating_sp = (bingo_kernel_scratchpad_t*)(uintptr_t)_gsp; \
            uint8_t* _activation = (uint8_t*)(uintptr_t)_gating_sp->return_value; \
            uint32_t _idx = *(uint32_t*)((char*)(arg) + offsetof(args_t, cond_node_index)); \
            if (_activation && !_activation[_idx]) { \
                bingo_kernel_scratchpad_t* _sp = BINGO_GET_SCRATCHPAD(arg, args_t); \
                _sp->return_value = 0; \
                _sp->num_return_values = 0; \
                return BINGO_RET_SUCC; \
            } \
        } \
    } while(0)

// Thin alias over the shared accessor in heterogeneous_runtime.h so device
// kernels can keep their familiar BINGO_GET_SP(arg, args_t) name while the
// actual offset comes from offsetof(args_t, scratchpad_ptr).
#define BINGO_GET_SP(arg, args_t) BINGO_GET_SCRATCHPAD((arg), args_t)
