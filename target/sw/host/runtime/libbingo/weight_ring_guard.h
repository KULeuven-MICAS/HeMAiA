// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// The weight prefetcher's byte guard (host_kernel_lib.h weight_prefetch): which bytes of a ring
// still belong to a chunk the cluster has not read.
//
// A run's chunks sit back to back from the run's first slot (mini_compiler libs/blocks/linear.py
// WeightRings.take_ex, the same rule), so a chunk shorter than a slot leaves the next chunk of
// its run starting inside a LOWER slot than its own: chunk k's bytes can reach down into slots
// below k % n_slots. The slot rule alone (a run waits for the chunks n_slots before its own, the
// owners of its slots' flag and release words) then lets a run of the next wrap overwrite the
// head of a chunk that is not read yet. The guard keeps, per slot, the newest chunk whose bytes
// were pushed into it; a run waits until every chunk whose bytes lie in the slots it writes has
// been read. Without such a chunk the guard is the slot rule exactly.
//
// No dependencies, so the guard builds on the workstation too (mini_compiler
// tests/test_weight_ring_guard.py runs it against WeightRings' placements).

#pragma once

#include <stdint.h>

// slots per ring the guard tracks; WeightRings refuses more (linear.py)
#define WEIGHT_RING_GUARD_MAX_SLOTS 32

typedef struct {
    int64_t top[WEIGHT_RING_GUARD_MAX_SLOTS];   // per slot: the newest chunk with bytes in it, -1: none
} weight_ring_guard_t;

// A prefetcher (part) that starts with `pushed` chunks of the ring already pushed, by an earlier
// part whose runs it does not know: a chunk j lies in slots j % ns - (span - 1) .. j % ns (a run
// of at most `span` chunks, each at most one slot), and only the last ns pushed chunks can still
// be unread, so every slot is held for the newest of those that can reach it.
static inline void weight_ring_guard_init(weight_ring_guard_t *g, uint64_t pushed, uint64_t ns,
                                          uint64_t span) {
    for (uint64_t s = 0; s < WEIGHT_RING_GUARD_MAX_SLOTS; s++) g->top[s] = -1;
    const uint64_t j0 = pushed > ns ? pushed - ns : 0;
    for (uint64_t j = j0; j < pushed; j++) {
        const uint64_t q = j % ns;
        for (uint64_t s = q + 1 > span ? q + 1 - span : 0; s <= q; s++) g->top[s] = (int64_t)j;
    }
}

// Whether a run of `bytes` from the start of slot s0 may be pushed with chunks [0, cons) read:
// every chunk whose bytes lie in the slots it writes has been read.
static inline int weight_ring_guard_free(const weight_ring_guard_t *g, uint64_t s0, uint64_t bytes,
                                         uint64_t sb, uint64_t cons) {
    if (!bytes) return 1;
    const uint64_t s1 = s0 + (bytes - 1) / sb;
    for (uint64_t s = s0; s <= s1 && s < WEIGHT_RING_GUARD_MAX_SLOTS; s++)
        if (g->top[s] >= (int64_t)cons) return 0;
    return 1;
}

// Chunk k, `size` bytes at byte `off` of its run (which starts at slot s0), has been pushed.
static inline void weight_ring_guard_mark(weight_ring_guard_t *g, uint64_t s0, uint64_t off,
                                          uint64_t size, uint64_t sb, uint64_t k) {
    if (!size) return;
    const uint64_t a = s0 + off / sb, b = s0 + (off + size - 1) / sb;
    for (uint64_t s = a; s <= b && s < WEIGHT_RING_GUARD_MAX_SLOTS; s++) g->top[s] = (int64_t)k;
}
