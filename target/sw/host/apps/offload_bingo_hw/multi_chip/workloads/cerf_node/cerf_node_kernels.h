// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Host kernels of the cerf_node workload (main_bingo.py): changing one node of a decode graph
// at run time, by the host rebuilding and relaunching it.

#pragma once

#include "libbingo/bingo_api.h"

#define CERF_NODE_MAX_ROUNDS 64

// A host task that does nothing: the source of a static CERF fork, whose gate the compiler
// splices in right after it on the same core.
__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_cerf_node_nop_args {
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_cerf_node_nop_args_t;

static inline uint64_t __host_bingo_kernel_cerf_node_nop(void *arg){
    bingo_kernel_scratchpad_t *sp =
        (bingo_kernel_scratchpad_t *)(uintptr_t)((__host_bingo_kernel_cerf_node_nop_args_t *)arg)->scratchpad_ptr;
    sp->return_value = 0;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// What a host CERF write costs, piece by piece: host cycles around each step, CERF_NODE_COST_REPS
// times, then one "CNC <step> <median> <min>" line each, on group 127 (word 3, bit 31). No task of the
// graph may wait on a CERF group while this runs.
#define CERF_NODE_COST_REPS 16
#define CERF_NODE_COST_STEPS 9
// Then an update of CERF_NODE_COST_CTRL in every CERF word, CERF_NODE_COST_WAYS ways, as the device's
// __snax_bingo_kernel_cerf_cost does from a core (offload_hw_kernels/cerf.h), plus two with a fence:
//   0  one store, as a gate does: group 127 through CERF_SET_3 or CERF_CLR_3, then its status word
//   1  every word through CERF_SET and CERF_CLR, then every status word
//   2  read-modify-write: read every status word, merge, write every CERF_STATE word and CERF_WRITE_EN,
//      then every status word
//   3  as 2, with a fence in place of the status wait
//   4  the 32-group code: read one status word, merge, write its CERF_STATE word and CERF_WRITE_EN, fence
// Each try changes every controlled group (alternate tries swap the pattern) and is bracketed for the
// [CERF ISSUE] probe by a store to the read-only CERF_STATUS_3 before it and to CERF_STATUS_0 after it,
// which change nothing. "CNW <way> <issue> <total> <total min>": host cycles to the last store issued, and
// to the update seen in the status (or fenced). The CERF gets back the state it had before.
#define CERF_NODE_COST_WAYS 5
#define CERF_NODE_COST_CTRL 0x00FFFF00u
__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_cerf_node_cost_args {
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_cerf_node_cost_args_t;

static uint64_t cerf_node_cost_probe;

static inline uint64_t __host_bingo_kernel_cerf_node_cost(void *arg){
    bingo_kernel_scratchpad_t *sp =
        (bingo_kernel_scratchpad_t *)(uintptr_t)((__host_bingo_kernel_cerf_node_cost_args_t *)arg)->scratchpad_ptr;
    static const char *const name[CERF_NODE_COST_STEPS] = {
        "mcycle", "fence", "load_hit", "load_after_fence", "status_read", "store", "store_fence",
        "store_poll", "update_word"};
    static uint64_t c[CERF_NODE_COST_STEPS][CERF_NODE_COST_REPS];
    const uintptr_t set = (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_cerf_set_addr(3));
    const uintptr_t clr = (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_cerf_clr_addr(3));
    const uintptr_t st = (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_cerf_status_addr(3));
    const uint32_t bit = 1u << 31;
    volatile uint64_t *probe = &cerf_node_cost_probe;
    for (uint32_t r = 0; r < CERF_NODE_COST_REPS; r++) {
        uint64_t t;
        t = bingo_mcycle(); c[0][r] = bingo_mcycle() - t;
        t = bingo_mcycle(); asm volatile("fence" ::: "memory"); c[1][r] = bingo_mcycle() - t;
        (void)*probe;
        t = bingo_mcycle(); (void)*probe; c[2][r] = bingo_mcycle() - t;
        asm volatile("fence" ::: "memory");
        t = bingo_mcycle(); (void)*probe; c[3][r] = bingo_mcycle() - t;
        t = bingo_mcycle(); (void)readw(st); c[4][r] = bingo_mcycle() - t;
        t = bingo_mcycle(); writew(bit, set); c[5][r] = bingo_mcycle() - t;
        while (!(readw(st) & bit)) {
        }
        t = bingo_mcycle(); writew(bit, clr); asm volatile("fence" ::: "memory"); c[6][r] = bingo_mcycle() - t;
        t = bingo_mcycle(); writew(bit, set); while (!(readw(st) & bit)) {
        }
        c[7][r] = bingo_mcycle() - t;
        t = bingo_mcycle(); bingo_cerf_update_word(3, bit, 0); c[8][r] = bingo_mcycle() - t;
    }

    static uint64_t wi[CERF_NODE_COST_WAYS][CERF_NODE_COST_REPS], wt[CERF_NODE_COST_WAYS][CERF_NODE_COST_REPS];
    uintptr_t sa[BINGO_CERF_WORDS], ta[BINGO_CERF_WORDS], ss[BINGO_CERF_WORDS], cs[BINGO_CERF_WORDS];
    uint32_t found[BINGO_CERF_WORDS], want[BINGO_CERF_WORDS], cur[BINGO_CERF_WORDS];
    const uint32_t ctl = CERF_NODE_COST_CTRL, last = BINGO_CERF_WORDS - 1;
    const uintptr_t en = (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_cerf_write_en_addr());
    for (uint32_t w = 0; w < BINGO_CERF_WORDS; w++) {
        sa[w] = (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_cerf_status_addr(w));
        ta[w] = (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_cerf_state_addr(w));
        ss[w] = (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_cerf_set_addr(w));
        cs[w] = (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_cerf_clr_addr(w));
        found[w] = readw(sa[w]);
    }
    for (uint32_t way = 0; way < CERF_NODE_COST_WAYS; way++) {
        for (uint32_t r = 0; r < CERF_NODE_COST_REPS; r++) {
            for (uint32_t w = 0; w < BINGO_CERF_WORDS; w++)
                want[w] = (((w & 1u) ? 0xA5A5A5A5u : 0x5A5A5A5Au) ^ ((r & 1u) ? ~0u : 0u)) & ctl;
            const uint32_t on = (r & 1u) ? 0 : bit;
            writew(0, sa[last]);
            uint64_t t = bingo_mcycle();
            if (way == 0) {
                writew(bit, on ? ss[last] : cs[last]);
                wi[way][r] = bingo_mcycle() - t;
                while ((readw(sa[last]) & bit) != on) {
                }
            } else if (way == 1) {
                for (uint32_t w = 0; w < BINGO_CERF_WORDS; w++) {
                    writew(want[w], ss[w]);
                    writew(~want[w] & ctl, cs[w]);
                }
                wi[way][r] = bingo_mcycle() - t;
                for (uint32_t w = 0; w < BINGO_CERF_WORDS; w++)
                    while ((readw(sa[w]) & ctl) != want[w]) {
                    }
            } else if (way < 4) {
                for (uint32_t w = 0; w < BINGO_CERF_WORDS; w++) cur[w] = readw(sa[w]);
                for (uint32_t w = 0; w < BINGO_CERF_WORDS; w++) writew((cur[w] & ~ctl) | want[w], ta[w]);
                writew(1, en);
                wi[way][r] = bingo_mcycle() - t;
                if (way == 2) {
                    for (uint32_t w = 0; w < BINGO_CERF_WORDS; w++)
                        while ((readw(sa[w]) & ctl) != want[w]) {
                        }
                } else {
                    asm volatile("fence" ::: "memory");
                }
            } else {
                cur[last] = readw(sa[last]);
                writew((cur[last] & ~ctl) | want[last], ta[last]);
                writew(1, en);
                wi[way][r] = bingo_mcycle() - t;
                asm volatile("fence" ::: "memory");
            }
            wt[way][r] = bingo_mcycle() - t;
            writew(0, sa[0]);
        }
    }
    for (uint32_t w = 0; w < BINGO_CERF_WORDS; w++) {
        writew(found[w], ss[w]);
        writew(~found[w], cs[w]);
    }
    for (uint32_t w = 0; w < BINGO_CERF_WORDS; w++)
        while (readw(sa[w]) != found[w]) {
        }

    for (uint32_t k = 0; k < CERF_NODE_COST_STEPS; k++) {
        uint64_t v[CERF_NODE_COST_REPS], lo = ~0ull;
        for (uint32_t r = 0; r < CERF_NODE_COST_REPS; r++) {
            uint64_t x = c[k][r];
            uint32_t i = r;
            while (i > 0 && v[i - 1] > x) { v[i] = v[i - 1]; i--; }
            v[i] = x;
            if (x < lo) lo = x;
        }
        printf_safe("CNC %s %d %d\r\n", name[k], (int)v[CERF_NODE_COST_REPS / 2], (int)lo);
    }
    for (uint32_t k = 0; k < CERF_NODE_COST_WAYS; k++) {
        uint64_t vi[CERF_NODE_COST_REPS], vt[CERF_NODE_COST_REPS];
        for (uint32_t r = 0; r < CERF_NODE_COST_REPS; r++) {
            uint32_t i = r, j = r;
            while (i > 0 && vi[i - 1] > wi[k][r]) { vi[i] = vi[i - 1]; i--; }
            vi[i] = wi[k][r];
            while (j > 0 && vt[j - 1] > wt[k][r]) { vt[j] = vt[j - 1]; j--; }
            vt[j] = wt[k][r];
        }
        printf_safe("CNW %d %d %d %d\r\n", (int)k, (int)vi[CERF_NODE_COST_REPS / 2],
                    (int)vt[CERF_NODE_COST_REPS / 2], (int)vt[0]);
    }
    sp->return_value = 0;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// The host arm, on the host of every compute chiplet. Per round:
//   1. build   (node chiplet only): copy the node's precompiled descriptor from the compiled list
//              into a one-entry list and patch it to run unconditionally, with no dependencies;
//   2. sync    (every chiplet): one 0xFF broadcast chip barrier over the rectangle;
//   3. restart (node chiplet only): launch the one-entry list, then wait for the node's stamp in
//              its cluster's L1.
// Then every chiplet launches its held tail (the exit tasks) and prints its host cycles.
__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_cerf_node_host_arm_args {
    uint64_t node_chip;      // the chiplet whose node is rebuilt and relaunched
    uint64_t node_task_id;   // the node's task id in that chiplet's compiled list
    uint64_t rounds;         // at most CERF_NODE_MAX_ROUNDS
    uint64_t flag_addr;      // the node's sync-probe stamp word (its cluster's L1)
    uint64_t top_left;       // the barrier rectangle
    uint64_t bottom_right;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_cerf_node_host_arm_args_t;

static inline uint64_t __host_bingo_kernel_cerf_node_host_arm(void *arg){
    const __host_bingo_kernel_cerf_node_host_arm_args_t *a =
        (const __host_bingo_kernel_cerf_node_host_arm_args_t *)arg;
    bingo_kernel_scratchpad_t *sp = (bingo_kernel_scratchpad_t *)(uintptr_t)a->scratchpad_ptr;
    const uint8_t chip = get_current_chip_id();
    const int mine = chip == (uint8_t)a->node_chip;
    const uint32_t rounds = a->rounds < CERF_NODE_MAX_ROUNDS ? (uint32_t)a->rounds : CERF_NODE_MAX_ROUNDS;
    volatile comm_buffer_t *cb = (volatile comm_buffer_t *)(uintptr_t)bingo_get_l2_comm_buffer(chip);
    static uint64_t build[CERF_NODE_MAX_ROUNDS], sync[CERF_NODE_MAX_ROUNDS];
    static uint64_t launch[CERF_NODE_MAX_ROUNDS], seen[CERF_NODE_MAX_ROUNDS];

    // The node's slot in the compiled list, and a one-entry list for it (not timed)
    uint64_t *list = bingo_hw_scheduler_list();
    uint64_t *one = 0;
    uint32_t slot = 0;
    volatile uint32_t *flag = (volatile uint32_t *)(uintptr_t)a->flag_addr;
    if (mine) {
        const uint32_t n = bingo_hw_scheduler_list_len();
        for (slot = 0; slot < n; slot++) {
            bingo_task_desc_t d = bingo_task_desc_load(list, slot);
            if (bingo_task_desc_get_field(&d, TASK_ID_WIDTH, TASK_ID_SHIFT) == a->node_task_id) break;
        }
        if (slot == n) {
            printf_safe("Chip(%x, %x): [Host] CERF_NODE FAIL: task %d is not in the list\r\n",
                        get_current_chip_loc_x(), get_current_chip_loc_y(), (int)a->node_task_id);
            return BINGO_RET_FAIL;
        }
        one = (uint64_t *)(uintptr_t)bingo_l2_alloc(chip, BINGO_TASK_DESC_BYTES);
    }

    for (uint32_t r = 0; r < rounds; r++) {
        if (mine) {
            uint64_t t0 = bingo_mcycle();
            bingo_task_list_copy(one, list + (uint64_t)slot * BINGO_TASK_DESC_WORDS, 1);
            bingo_task_desc_patch(one, 0, COND_EXEC_EN_WIDTH, COND_EXEC_EN_SHIFT, 0);
            bingo_task_desc_patch(one, 0, DEP_CHECK_ENABLED_WIDTH, DEP_CHECK_ENABLED_SHIFT, 0);
            bingo_task_desc_patch(one, 0, DEP_SET_ENABLED_WIDTH, DEP_SET_ENABLED_SHIFT, 0);
            asm volatile("fence" ::: "memory");
            build[r] = bingo_mcycle() - t0;
            *flag = 0;
            asm volatile("fence" ::: "memory");
        }
        uint64_t t1 = bingo_mcycle();
        chip_barrier(cb, (uint8_t)a->top_left, (uint8_t)a->bottom_right, (uint8_t)(r + 1));
        sync[r] = bingo_mcycle() - t1;
        if (mine) {
            uint64_t t2 = bingo_mcycle();
            bingo_hw_scheduler_launch((uint64_t)(uintptr_t)one, 1);
            uint64_t t3 = bingo_mcycle();
            while (*flag == 0) {
            }
            launch[r] = t3 - t2;
            seen[r] = bingo_mcycle() - t3;
        }
    }

    bingo_hw_scheduler_launch_held();

    // Short lines: the UART costs simulated time per character, and nothing else runs meanwhile.
    //   CN <chip> <round> <build> <sync> <launch call> <launch -> stamp seen>   (node chiplet)
    //   CN <chip> <round> <sync>                                               (the others)
    for (uint32_t r = 0; r < rounds; r++) {
        if (mine)
            printf_safe("CN %02x %d %d %d %d %d\r\n", chip, (int)r, (int)build[r], (int)sync[r],
                        (int)launch[r], (int)seen[r]);
        else
            printf_safe("CN %02x %d %d\r\n", chip, (int)r, (int)sync[r]);
    }
    sp->return_value = 0;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}
