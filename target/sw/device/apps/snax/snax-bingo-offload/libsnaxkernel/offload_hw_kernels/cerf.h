// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// CERF gates that run on a device core.

#pragma once

#include "../macros.h"

// Tile gate (the 'tiles' fork, docs/conditional_dfg_api.md): every slot is built for its
// worst case in `tiles` tiles of `rows` rows; slot s got counts[s] rows, so its tiles
// 0 .. ceil(counts[s] / rows) - 1 run and the rest are skipped. Slot s, tile t is CERF group
// group_base + s * group_block + t; a slot's tiles never cross a 32-bit CERF word.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_cerf_tiles(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    __snax_bingo_kernel_cerf_tiles_args_t *a = (__snax_bingo_kernel_cerf_tiles_args_t *)arg;
    const volatile uint32_t *counts = (const volatile uint32_t *)(uintptr_t)a->counts_addr;
    const uint32_t num_slots = a->num_slots, rows = a->rows, tiles = a->tiles;
    const uint32_t base = a->group_base, block = a->group_block;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_cerf_tiles_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    uint32_t ctrl[BINGO_CERF_WORDS] = {0}, state[BINGO_CERF_WORDS] = {0};
    const uint32_t all = tiles >= 32 ? ~0u : (1u << tiles) - 1u;
    for (uint32_t s = 0; s < num_slots; s++) {
        uint32_t n = (counts[s] + rows - 1u) / rows;
        if (n > tiles) n = tiles;
        const uint32_t on = n >= 32 ? ~0u : (1u << n) - 1u;
        const uint32_t g = base + s * block;
        ctrl[g / 32u] |= all << (g % 32u);
        state[g / 32u] |= on << (g % 32u);
    }
    bingo_dev_cerf_update(ctrl, state);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    BINGO_RETURN_SUCC(sp, 0);
}

// Static gate on a device core (a 'static' fork whose source runs on a device core): give the groups
// in cerf_ctrl the values in cerf_state, word by word, and leave every other group alone.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_cerf_static(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    __snax_bingo_kernel_cerf_static_args_t *a = (__snax_bingo_kernel_cerf_static_args_t *)arg;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_cerf_static_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    bingo_dev_cerf_update(a->cerf_ctrl, a->cerf_state);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    BINGO_RETURN_SUCC(sp, 0);
}

// What a CERF update costs from a device core (the cerf_node workload's micro-benchmark), three ways,
// BINGO_CERF_COST_REPS tries each. Every try gives the groups in cerf_ctrl the bits of cerf_state (even
// tries) or of its complement (odd tries) and waits until the status shows them:
//   0  one store, as a gate does: group 32 * BINGO_CERF_WORDS - 1 through its CERF_SET or CERF_CLR word
//   1  every word through CERF_SET and CERF_CLR (bingo_dev_cerf_update)
//   2  read-modify-write through CERF_STATE: read every status word, merge, write every CERF_STATE word,
//      then CERF_WRITE_EN
// A store to the read-only last CERF_STATUS word before a try and one to CERF_STATUS_0 after it change
// nothing, but the [CERF ISSUE] probe sees them and so brackets the try. Run it only while no task waits on
// a CERF group: it gives the CERF back the state it found, but changes every group meanwhile.
#define BINGO_CERF_COST_REPS 8
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_cerf_cost(void *arg){
    __snax_bingo_kernel_cerf_static_args_t *a = (__snax_bingo_kernel_cerf_static_args_t *)arg;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_cerf_static_args_t);
    volatile uint32_t *const start = (volatile uint32_t *)quad_ctrl_cerf_status_addr(BINGO_CERF_WORDS - 1);
    volatile uint32_t *const end = (volatile uint32_t *)quad_ctrl_cerf_status_addr(0);
    uint32_t found[BINGO_CERF_WORDS], all[BINGO_CERF_WORDS], ctrl[BINGO_CERF_WORDS], want[BINGO_CERF_WORDS],
        cur[BINGO_CERF_WORDS];
    for (uint32_t w = 0; w < BINGO_CERF_WORDS; w++) {
        found[w] = *(volatile uint32_t *)quad_ctrl_cerf_status_addr(w);
        all[w] = ~0u;
        ctrl[w] = a->cerf_ctrl[w];
    }
    for (uint32_t way = 0; way < 3; way++) {
        for (uint32_t r = 0; r < BINGO_CERF_COST_REPS; r++) {
            for (uint32_t w = 0; w < BINGO_CERF_WORDS; w++)
                want[w] = ((r & 1u) ? ~a->cerf_state[w] : a->cerf_state[w]) & ctrl[w];
            *start = 0;
            if (way == 0) {
                bingo_dev_cerf_update_word(BINGO_CERF_WORDS - 1, 1u << 31, (r & 1u) ? 0 : 1u << 31);
            } else if (way == 1) {
                bingo_dev_cerf_update(ctrl, want);
            } else {
                for (uint32_t w = 0; w < BINGO_CERF_WORDS; w++)
                    cur[w] = *(volatile uint32_t *)quad_ctrl_cerf_status_addr(w);
                for (uint32_t w = 0; w < BINGO_CERF_WORDS; w++)
                    *(volatile uint32_t *)quad_ctrl_cerf_state_addr(w) = (cur[w] & ~ctrl[w]) | want[w];
                *(volatile uint32_t *)quad_ctrl_cerf_write_en_addr() = 1;
                for (uint32_t w = 0; w < BINGO_CERF_WORDS; w++)
                    while ((*(volatile uint32_t *)quad_ctrl_cerf_status_addr(w) & ctrl[w]) != want[w]) {
                    }
            }
            *end = 0;
        }
    }
    bingo_dev_cerf_update(all, found);
    BINGO_RETURN_SUCC(sp, 0);
}
