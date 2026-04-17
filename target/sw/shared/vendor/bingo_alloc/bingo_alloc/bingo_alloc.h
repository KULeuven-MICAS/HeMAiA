// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Bingo Heap Allocator: A modified O(1) heap allocator for HeMAiA's
// heterogeneous SoC. Derived from o1heap (MIT licensed, Pavel Kirienko).
//
// Key changes from o1heap:
//   - Alignment-based rounding instead of power-of-2 rounding, enabling
//     allocations larger than capacity/2 (e.g., 300 KB from 512 KB TCDM).
//   - Search-within-bin fallback when the first fragment in a bin is too small.
//   - All pointer values are uint64_t so that both 32-bit (Snitch) and 64-bit
//     (CVA6) cores can share the same heap data structure in shared memory.

#ifndef BINGO_ALLOC_H_INCLUDED
#define BINGO_ALLOC_H_INCLUDED

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Alignment for all allocations. Must be a power of 2.
/// 128 bytes ensures compatibility with SNAX accelerator DMA requirements.
#define BINGO_HEAP_ALIGNMENT 128

/// Maximum number of segregated free-list bins (covers sizes up to 2^63).
#define BINGO_HEAP_NUM_BINS 64

/// Runtime diagnostic counters.
typedef struct __attribute__((aligned(8))) {
    uint64_t capacity;          ///< Total usable bytes (heap size minus metadata).
    uint64_t allocated;         ///< Currently allocated bytes.
    uint64_t peak_allocated;    ///< High-water mark of allocated bytes.
    uint64_t peak_request_size; ///< Largest single allocation request seen.
    uint64_t oom_count;         ///< Number of failed allocation attempts.
} BingoHeapDiagnostics;

/// Fragment header — sits at the start of every fragment (allocated or free).
typedef struct __attribute__((aligned(8))) BingoHeapFragmentHeader {
    uint64_t next;  ///< Next fragment in the address-ordered linked list.
    uint64_t prev;  ///< Previous fragment in the address-ordered list.
    uint64_t size;  ///< Total fragment size in bytes (header + payload).
    uint64_t used;  ///< Non-zero if this fragment is currently allocated.
} BingoHeapFragmentHeader;

/// Free fragment — extends the header with free-list pointers.
typedef struct __attribute__((aligned(8))) BingoHeapFragment {
    BingoHeapFragmentHeader header;
    uint64_t next_free;  ///< Next fragment in the same size-class bin.
    uint64_t prev_free;  ///< Previous fragment in the same size-class bin.
} BingoHeapFragment;

/// Heap instance — the root structure stored at the beginning of the arena.
typedef struct __attribute__((aligned(8))) BingoHeapInstance {
    uint64_t __attribute__((aligned(8))) bins[BINGO_HEAP_NUM_BINS];
    uint64_t nonempty_bin_mask;
    BingoHeapDiagnostics diagnostics;
} BingoHeapInstance;

/// Initialize a heap in the provided memory arena.
/// The arena base must be aligned to BINGO_HEAP_ALIGNMENT.
/// Returns the heap handle (same as base), or 0 on failure.
uint64_t bingoHeapInit(uint64_t heap_base_addr, uint64_t heap_size);

/// Allocate a block of at least `amount` bytes.
/// Returns a pointer to the usable region, or 0 if out of memory.
uint64_t bingoHeapMalloc(uint64_t handle, uint64_t amount);

/// Free a previously allocated block. Coalesces with adjacent free fragments.
void bingoHeapFree(uint64_t handle, uint64_t pointer);

/// Return the address of the diagnostics struct inside the heap instance.
uint64_t bingoHeapGetDiagnostics(uint64_t handle);

#ifdef __cplusplus
}
#endif

#endif  // BINGO_ALLOC_H_INCLUDED
