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

// -------------------------------------------------------------------------
// Debug prints
// -------------------------------------------------------------------------
// #define IDMA_DEBUG
// #define XDMA_DEBUG
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

// VERSACORE_DEBUG_PRINT was previously declared in versacore_hw_param.h.
// After the refactor that file is gone, so declare it here where every
// partial (and the generated core/gemm.h's default-case arm) can see it.
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

// Per-Kernel Scratchpad: defined in heterogeneous_runtime.h (shared host/device)
//   bingo_kernel_scratchpad_t, BINGO_KERNEL_SCRATCHPAD_SIZE, BINGO_SP_PROFILE

// -------------------------------------------------------------------------
// SW Guard for CERF Group Sharing (>32 experts)
// -------------------------------------------------------------------------
// When multiple experts share a CERF group, the HW CERF check only tells
// "this group is active." The SW guard provides per-expert granularity:
//   1. Gating kernel writes uint8_t activation[num_experts] (1=run, 0=skip)
//   2. Gating kernel stores the array pointer in its scratchpad.return_value
//   3. Each expert reads gating_sp->return_value to find the activation array
//   4. Expert checks activation[cond_node_index] — if 0, early-return (skip)
//
// The compiler wires gating_sp_addr and cond_node_index into each expert's args.
// When gating_sp_addr == 0, the SW guard is disabled (no group sharing).
//
// Usage in expert kernels:
//   BINGO_SW_GUARD_CHECK(arg, nfields)
//   nfields = total uint32_t fields in the args struct
//   The last 3 fields are always: gating_sp_addr, cond_node_index, scratchpad_ptr
//   Returns BINGO_RET_SUCC immediately if this expert should be skipped.
//   When gating_sp_addr == 0, the guard is a no-op (single load + branch-not-taken).
#define BINGO_SW_GUARD_CHECK(arg, nfields) \
    do { \
        uint32_t _gsp = ((uint32_t*)(arg))[(nfields) - 3]; /* gating_sp_addr */ \
        if (_gsp) { \
            bingo_kernel_scratchpad_t* _gating_sp = (bingo_kernel_scratchpad_t*)(uintptr_t)_gsp; \
            uint8_t* _activation = (uint8_t*)(uintptr_t)_gating_sp->return_value; \
            uint32_t _idx = ((uint32_t*)(arg))[(nfields) - 2]; /* cond_node_index */ \
            if (_activation && !_activation[_idx]) { \
                bingo_kernel_scratchpad_t* _sp = (bingo_kernel_scratchpad_t*)(uintptr_t) \
                    ((uint32_t*)(arg))[(nfields) - 1]; /* scratchpad_ptr */ \
                _sp->return_value = 0; \
                _sp->num_return_values = 0; \
                return BINGO_RET_SUCC; \
            } \
        } \
    } while(0)

// BINGO_GET_SP: get scratchpad from arg array by field count (last field)
#define BINGO_GET_SP(arg, nfields) \
    ((bingo_kernel_scratchpad_t*)(uintptr_t)((uint32_t*)(arg))[(nfields) - 1])
