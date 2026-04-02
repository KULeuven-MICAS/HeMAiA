// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Bingo Heap Allocator — derived from o1heap by Pavel Kirienko (MIT).
//
// Modifications vs. original o1heap:
//   1. Alignment-based rounding (not power-of-2) for fragment_size in
//      bingoHeapMalloc(). This removes the ~capacity/2 ceiling on single
//      allocations. A 300 KB tensor can now be allocated from 512 KB TCDM.
//   2. Search-within-bin: when the first fragment in the optimal bin is too
//      small, we walk the bin's free list before falling back to a higher bin.
//      Only the optimal bin needs this search; higher bins are guaranteed to
//      contain fragments >= 2x the optimal bin's minimum, so first-fit works.

#include "bingo_alloc.h"
#include "uart.h"

// ---- Internal constants ----

#define BINGO_HEAP_MAX           18446744073709551615ULL
#define FRAGMENT_SIZE_MIN        (BINGO_HEAP_ALIGNMENT * 2U)
#define FRAGMENT_SIZE_MAX        ((BINGO_HEAP_MAX >> 1U) + 1U)
#define INSTANCE_SIZE_PADDED     ((sizeof(BingoHeapInstance) + BINGO_HEAP_ALIGNMENT - 1U) & ~(BINGO_HEAP_ALIGNMENT - 1U))

// ---- Pointer casting macros ----
// These allow both 32-bit Snitch cores and 64-bit CVA6 to share the same
// heap data structure in shared memory (TCDM / SPM).

#define BINGO_LOW32(x)  ((uint32_t)(((uint64_t)(x) >> 0) & 0xFFFFFFFF))

#define FRAGMENT_PTR(x) \
    ((BingoHeapFragment*) (void*) ((__riscv_xlen == 64) ? (x) : BINGO_LOW32(x)))
#define HANDLE_PTR(x) \
    ((BingoHeapInstance*) (void*) ((__riscv_xlen == 64) ? (x) : BINGO_LOW32(x)))

// ---- Bit-manipulation helpers (software CLZ for portability) ----

static inline uint8_t bingo_clz64(const uint64_t x) {
    uint64_t t = ((uint64_t)1U) << 63U;
    uint8_t r = 0;
    while ((x & t) == 0) {
        t >>= 1U;
        r++;
    }
    return r;
}

static inline uint8_t log2Floor(const uint64_t x) {
    return (uint8_t)(63U - bingo_clz64(x));
}

static inline uint8_t log2Ceil(const uint64_t x) {
    if (x <= 1U) return 0U;
    return (uint8_t)(64U - bingo_clz64(x - 1U));
}

static inline uint64_t pow2(const uint8_t power) {
    return ((uint64_t)1U) << power;
}

// ---- Fragment linked-list helpers ----

/// Link two adjacent fragments in the address-ordered list.
static inline void interlink(uint64_t const left, uint64_t const right) {
    if (left != 0U)  FRAGMENT_PTR(left)->header.next = right;
    if (right != 0U) FRAGMENT_PTR(right)->header.prev = left;
}

/// Insert a free fragment into the appropriate size-class bin.
static inline void rebin(uint64_t const handle, uint64_t const frag) {
    const uint8_t idx = log2Floor(FRAGMENT_PTR(frag)->header.size / FRAGMENT_SIZE_MIN);
    FRAGMENT_PTR(frag)->next_free = HANDLE_PTR(handle)->bins[idx];
    FRAGMENT_PTR(frag)->prev_free = 0U;
    if (HANDLE_PTR(handle)->bins[idx] != 0U) {
        uint64_t cur = HANDLE_PTR(handle)->bins[idx];
        FRAGMENT_PTR(cur)->prev_free = frag;
    }
    HANDLE_PTR(handle)->bins[idx] = frag;
    HANDLE_PTR(handle)->nonempty_bin_mask |= pow2(idx);
}

/// Remove a fragment from its size-class bin.
static inline void unbin(uint64_t const handle, uint64_t const frag) {
    const uint8_t idx = log2Floor(FRAGMENT_PTR(frag)->header.size / FRAGMENT_SIZE_MIN);
    if (FRAGMENT_PTR(frag)->next_free != 0U) {
        uint64_t nf = FRAGMENT_PTR(frag)->next_free;
        FRAGMENT_PTR(nf)->prev_free = FRAGMENT_PTR(frag)->prev_free;
    }
    if (FRAGMENT_PTR(frag)->prev_free != 0U) {
        uint64_t pf = FRAGMENT_PTR(frag)->prev_free;
        FRAGMENT_PTR(pf)->next_free = FRAGMENT_PTR(frag)->next_free;
    }
    if (HANDLE_PTR(handle)->bins[idx] == frag) {
        HANDLE_PTR(handle)->bins[idx] = FRAGMENT_PTR(frag)->next_free;
        if (HANDLE_PTR(handle)->bins[idx] == 0U) {
            HANDLE_PTR(handle)->nonempty_bin_mask &= ~pow2(idx);
        }
    }
}

// ---- Public API ----

uint64_t bingoHeapInit(uint64_t const heap_base_addr, uint64_t const heap_size) {
    uint64_t out = 0U;
    if ((heap_base_addr != 0U) &&
        ((heap_base_addr % BINGO_HEAP_ALIGNMENT) == 0U) &&
        (heap_size >= (INSTANCE_SIZE_PADDED + FRAGMENT_SIZE_MIN)))
    {
        out = heap_base_addr;
        HANDLE_PTR(out)->nonempty_bin_mask = 0UL;
        for (uint64_t i = 0U; i < BINGO_HEAP_NUM_BINS; i++) {
            HANDLE_PTR(out)->bins[i] = 0UL;
        }

        uint64_t capacity = heap_size - INSTANCE_SIZE_PADDED;
        if (capacity > FRAGMENT_SIZE_MAX) {
            capacity = FRAGMENT_SIZE_MAX;
        }
        while ((capacity % FRAGMENT_SIZE_MIN) != 0UL) {
            capacity--;
        }

        uint64_t const frag_start = heap_base_addr + INSTANCE_SIZE_PADDED;
        FRAGMENT_PTR(frag_start)->header.next = 0UL;
        FRAGMENT_PTR(frag_start)->header.prev = 0UL;
        FRAGMENT_PTR(frag_start)->header.size = capacity;
        FRAGMENT_PTR(frag_start)->header.used = 0UL;
        FRAGMENT_PTR(frag_start)->next_free = 0UL;
        FRAGMENT_PTR(frag_start)->prev_free = 0UL;
        rebin(out, frag_start);

        HANDLE_PTR(out)->diagnostics.capacity        = capacity;
        HANDLE_PTR(out)->diagnostics.allocated        = 0UL;
        HANDLE_PTR(out)->diagnostics.peak_allocated   = 0UL;
        HANDLE_PTR(out)->diagnostics.peak_request_size = 0UL;
        HANDLE_PTR(out)->diagnostics.oom_count        = 0UL;
    }
    return out;
}

uint64_t bingoHeapMalloc(uint64_t const handle, const uint64_t amount) {
    uint64_t out = 0UL;
    if ((amount > 0UL) && (amount <= HANDLE_PTR(handle)->diagnostics.capacity)) {
        // ---- KEY CHANGE vs o1heap ----
        // Round to next multiple of FRAGMENT_SIZE_MIN instead of next power of 2.
        // This allows allocations up to (capacity - header), not just capacity/2.
        //
        // Example with 512 KB TCDM (capacity ~508 KB after metadata):
        //   o1heap:      300 KB request -> roundUpToPow2(300KB + 128) = 512 KB -> FAIL
        //   bingo_alloc: 300 KB request -> alignUp(300KB + 128, 256)  = 300.25 KB -> OK
        const uint64_t fragment_size =
            (amount + BINGO_HEAP_ALIGNMENT + FRAGMENT_SIZE_MIN - 1U) & ~(FRAGMENT_SIZE_MIN - 1U);

        // Start from the bin whose minimum size is <= fragment_size.
        // Unlike o1heap (which uses log2Ceil and is guaranteed first-fit),
        // we use log2Floor and may need to search within the bin.
        const uint8_t optimal_bin_index = log2Floor(fragment_size / FRAGMENT_SIZE_MIN);
        const uint64_t candidate_bin_mask = ~(pow2(optimal_bin_index) - 1U);
        const uint64_t suitable_bins = HANDLE_PTR(handle)->nonempty_bin_mask & candidate_bin_mask;

        if (suitable_bins != 0) {
            const uint64_t smallest_bin_mask = suitable_bins & ~(suitable_bins - 1U);
            const uint8_t  bin_index = log2Floor(smallest_bin_mask);

            uint64_t frag_ptr = HANDLE_PTR(handle)->bins[bin_index];

            if (bin_index == optimal_bin_index) {
                // ---- SEARCH WITHIN BIN ----
                // The optimal bin covers sizes [2^idx * MIN, 2^(idx+1) * MIN).
                // Some fragments here may be smaller than fragment_size.
                // Walk the free list until we find one that fits.
                while (frag_ptr != 0U && FRAGMENT_PTR(frag_ptr)->header.size < fragment_size) {
                    frag_ptr = FRAGMENT_PTR(frag_ptr)->next_free;
                }
                if (frag_ptr == 0U) {
                    // No suitable fragment in this bin — try next higher bin.
                    // All fragments in higher bins are guaranteed >= fragment_size.
                    uint64_t higher_bins = suitable_bins & ~smallest_bin_mask;
                    if (higher_bins != 0U) {
                        uint64_t next_mask = higher_bins & ~(higher_bins - 1U);
                        uint8_t  next_idx  = log2Floor(next_mask);
                        frag_ptr = HANDLE_PTR(handle)->bins[next_idx];
                    }
                }
            }
            // else: bin_index > optimal_bin_index -> fragment is guaranteed large enough

            if (frag_ptr != 0U) {
                unbin(handle, frag_ptr);

                const uint64_t leftover = FRAGMENT_PTR(frag_ptr)->header.size - fragment_size;
                FRAGMENT_PTR(frag_ptr)->header.size = fragment_size;

                if (leftover >= FRAGMENT_SIZE_MIN) {
                    uint64_t const new_frag = frag_ptr + fragment_size;
                    FRAGMENT_PTR(new_frag)->header.size = leftover;
                    FRAGMENT_PTR(new_frag)->header.used = 0UL;
                    interlink(new_frag, FRAGMENT_PTR(frag_ptr)->header.next);
                    interlink(frag_ptr, new_frag);
                    rebin(handle, new_frag);
                }

                HANDLE_PTR(handle)->diagnostics.allocated += fragment_size;
                if (HANDLE_PTR(handle)->diagnostics.peak_allocated <
                    HANDLE_PTR(handle)->diagnostics.allocated) {
                    HANDLE_PTR(handle)->diagnostics.peak_allocated =
                        HANDLE_PTR(handle)->diagnostics.allocated;
                }

                FRAGMENT_PTR(frag_ptr)->header.used = 1UL;
                out = frag_ptr + BINGO_HEAP_ALIGNMENT;
            }
        }
    }

    if (amount > HANDLE_PTR(handle)->diagnostics.peak_request_size) {
        HANDLE_PTR(handle)->diagnostics.peak_request_size = amount;
    }
    if ((out == 0UL) && (amount > 0UL)) {
        HANDLE_PTR(handle)->diagnostics.oom_count += 1UL;
    }
    return out;
}

void bingoHeapFree(uint64_t const handle, uint64_t const pointer) {
    if (pointer != 0UL) {
        uint64_t const frag = pointer - BINGO_HEAP_ALIGNMENT;
        FRAGMENT_PTR(frag)->header.used = 0UL;
        HANDLE_PTR(handle)->diagnostics.allocated -= FRAGMENT_PTR(frag)->header.size;

        uint64_t const prev_frag = FRAGMENT_PTR(frag)->header.prev;
        uint64_t const next_frag = FRAGMENT_PTR(frag)->header.next;
        const bool join_left  = (prev_frag != 0UL) && (FRAGMENT_PTR(prev_frag)->header.used == 0UL);
        const bool join_right = (next_frag != 0UL) && (FRAGMENT_PTR(next_frag)->header.used == 0UL);

        if (join_left && join_right) {
            unbin(handle, prev_frag);
            unbin(handle, next_frag);
            FRAGMENT_PTR(prev_frag)->header.size +=
                FRAGMENT_PTR(frag)->header.size + FRAGMENT_PTR(next_frag)->header.size;
            FRAGMENT_PTR(frag)->header.size      = 0UL;
            FRAGMENT_PTR(next_frag)->header.size  = 0UL;
            interlink(prev_frag, FRAGMENT_PTR(next_frag)->header.next);
            rebin(handle, prev_frag);
        } else if (join_left) {
            unbin(handle, prev_frag);
            FRAGMENT_PTR(prev_frag)->header.size += FRAGMENT_PTR(frag)->header.size;
            FRAGMENT_PTR(frag)->header.size = 0UL;
            interlink(prev_frag, next_frag);
            rebin(handle, prev_frag);
        } else if (join_right) {
            unbin(handle, next_frag);
            FRAGMENT_PTR(frag)->header.size += FRAGMENT_PTR(next_frag)->header.size;
            FRAGMENT_PTR(next_frag)->header.size = 0UL;
            interlink(frag, FRAGMENT_PTR(next_frag)->header.next);
            rebin(handle, frag);
        } else {
            rebin(handle, frag);
        }
    }
}

uint64_t bingoHeapGetDiagnostics(uint64_t const handle) {
    return handle + offsetof(BingoHeapInstance, diagnostics);
}
