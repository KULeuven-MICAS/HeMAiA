// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// PAGED TASK LISTS (mini compiler list_phases). The gid list format (bingo_hw_scheduler_init_gid_lists) keeps
// three words of every device task of a cluster in its L1 for the whole run: its global id, its args pointer and
// its kernel pointer. Here a cluster's tasks are cut into the args phases of the graph (mini_compiler
// passes/bingo_args_phases.py), and only the resident tasks (the phase loads, the exits) and two phases at a time are in L1:
// the even phases share one window, the odd phases another, and a DM task of the graph copies phase p's entries
// into window p % 2 right before it copies that phase's args (PhaseListLoadArgs). A window is reused once
// every task of the phase before it of the same parity has been dispatched, as the args regions are.
//
// One entry: two words, the task's global id, then its args address with the index of its kernel in the
// cluster's function table in the low six bits (args structs are 64-byte aligned; a task without args has
// address 0).
//
// The header (one a cluster, in its L1; quad_ctrl_global_id_to_dev_id points at it), in words:
//   [0] BINGO_GID_LIST_PAGED_MAGIC   [1] K, the core count   [2] P, the phase count   [3] F, the function count
//   [4] window 0's L1 address        [5] window 1's L1 address
//   [6, 6 + K]                        the resident entries of core r: [rb[r], rb[r + 1])
//   then P rows of K + 1 words        phase p's entries of core r: [pb[p][r], pb[p][r + 1]) of window p % 2
//   then F words                      the kernel addresses
//   then the resident entries         (two words each)
// Each core's entries, resident or of a phase, are in the order its tasks are dispatched.
//
// This file is plain C (stdint only): the device runtime (device/runtime/src/bingo.h, bingo_hw_offload_manager_paged)
// and the host build (libbingo/bingo_list_phases.h) both include it.

#ifndef BINGO_PAGED_LISTS_H
#define BINGO_PAGED_LISTS_H

#include <stdint.h>

#define BINGO_GID_LIST_PAGED_MAGIC 0xB16D0034u
#define BINGO_PAGED_KIDX_BITS      6u
#define BINGO_PAGED_KIDX_MASK      ((1u << BINGO_PAGED_KIDX_BITS) - 1u)
#define BINGO_PAGED_MAX_KERNELS    (1u << BINGO_PAGED_KIDX_BITS)
#define BINGO_PAGED_ENTRY_WORDS    2u
#define BINGO_PAGED_HDR_FIXED      6u

// The header's word offsets, from K, P and F
#define BINGO_PAGED_RB_OFF                   BINGO_PAGED_HDR_FIXED
#define BINGO_PAGED_PB_OFF(k)                (BINGO_PAGED_HDR_FIXED + (k) + 1u)
#define BINGO_PAGED_FN_OFF(k, p)             (BINGO_PAGED_PB_OFF(k) + (p) * ((k) + 1u))
#define BINGO_PAGED_RES_OFF(k, p, f)         (BINGO_PAGED_FN_OFF(k, p) + (f))

// One core's place in its cluster's paged lists
typedef struct {
    const uint32_t *pb;      // the phase rows
    const uint32_t *res;     // the resident entries
    const uint32_t *win[2];  // the two windows
    const uint32_t *w;       // the window of the current phase
    uint32_t stride;         // K + 1
    uint32_t nphases;
    uint32_t core;
    uint32_t q;              // the current phase
    uint32_t cur, end;       // the core's next entry in phase q, and the end of its entries there
    uint32_t r_cur, r_base, r_end;
} bingo_paged_cursor_t;

static inline void bingo_paged_cursor_init(bingo_paged_cursor_t *c, const uint32_t *hdr, const uint32_t *win0,
                                           const uint32_t *win1, uint32_t core) {
    const uint32_t k = hdr[1], p = hdr[2], f = hdr[3];
    c->pb = hdr + BINGO_PAGED_PB_OFF(k);
    c->res = hdr + BINGO_PAGED_RES_OFF(k, p, f);
    c->win[0] = win0;
    c->win[1] = win1;
    c->w = win0;
    c->stride = k + 1u;
    c->nphases = p;
    c->core = core;
    c->q = 0u;
    if (core < k && p) {
        c->cur = c->pb[core];
        c->end = c->pb[core + 1u];
        c->r_base = hdr[BINGO_PAGED_RB_OFF + core];
        c->r_end = hdr[BINGO_PAGED_RB_OFF + core + 1u];
    } else {
        c->cur = c->end = 0u;
        c->r_base = c->r_end = 0u;
        c->nphases = 0u;
    }
    c->r_cur = c->r_base;
}

// The entry of task `gid`, or 0 when it is not this core's. In dispatch order the task is the resident entry at
// the resident cursor or the entry at the phase cursor, so a task costs one or two compares; an entry the manager
// never gets (a task skipped by conditional execution) is stepped over. A task in neither list makes the core
// look in the resident entries from the start, then move on through the later phases, whose windows the graph
// has filled by the time any of their tasks is dispatched. A cursor never goes back to an earlier phase.
static inline const uint32_t *bingo_paged_find(bingo_paged_cursor_t *c, uint32_t gid) {
    uint32_t i;
    if (c->r_cur < c->r_end && c->res[c->r_cur * BINGO_PAGED_ENTRY_WORDS] == gid) {
        return c->res + (c->r_cur++) * BINGO_PAGED_ENTRY_WORDS;
    }
    for (i = c->cur; i < c->end && c->w[i * BINGO_PAGED_ENTRY_WORDS] != gid; i++) {
    }
    if (i < c->end) {
        c->cur = i + 1u;
        return c->w + i * BINGO_PAGED_ENTRY_WORDS;
    }
    for (i = c->r_base; i < c->r_end && c->res[i * BINGO_PAGED_ENTRY_WORDS] != gid; i++) {
    }
    if (i < c->r_end) {
        c->r_cur = i + 1u;
        return c->res + i * BINGO_PAGED_ENTRY_WORDS;
    }
    while (c->q + 1u < c->nphases) {
        const uint32_t *row = c->pb + (++c->q) * c->stride;
        c->w = c->win[c->q & 1u];
        c->end = row[c->core + 1u];
        for (i = row[c->core]; i < c->end && c->w[i * BINGO_PAGED_ENTRY_WORDS] != gid; i++) {
        }
        if (i < c->end) {
            c->cur = i + 1u;
            return c->w + i * BINGO_PAGED_ENTRY_WORDS;
        }
        c->cur = c->end;
    }
    return 0;
}

#endif  // BINGO_PAGED_LISTS_H
