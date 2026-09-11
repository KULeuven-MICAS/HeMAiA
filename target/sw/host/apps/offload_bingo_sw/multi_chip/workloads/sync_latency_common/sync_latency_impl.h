// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// ARM C — cross-chip synchronization latency of the BINGO *software* manager.
// Shared implementation. One binary sweeps every point up to SYNC_MAX_P:
//     P = 2, 4, 8, 16  x  { one-to-one, one-to-all }
//
// Two workloads include this:
//   sync_latency_smoke_1cluster   SYNC_MAX_P = 4   -- correctness check, ~35 min
//   sync_latency_sweep_1cluster   SYNC_MAX_P = 16  -- the full sweep, hours
// The smoke build touches only chips 0x00, 0x01, 0x10, 0x11; the other chiplets get no
// tasks and idle. It runs on the SAME 16-chiplet RTL, so validating small costs no
// rebuild -- only the graph shrinks.
//
// Companion to the two barriers in 06-chip-sync-latency.md. Same x-axis (participating
// chiplets P) and the same rectangles, so the results drop onto the same plot:
//
//   ONE_TO_ONE   chip 0x00 -> the FURTHEST chiplet in the P-rectangle -> back.
//                One dependency edge each way: the cost of a single remote edge at the
//                worst distance this rectangle contains.
//   ONE_TO_ALL   chip 0x00 -> every other chiplet in the rectangle -> back.
//                Fan-out then fan-in, 2(P-1) edges. This is the shape a barrier has
//                (announce to everyone, wait for everyone), which is what makes it the
//                honest comparison against arms A and B.
//
// ============================================================================
// HOW THE LATENCY IS MEASURED, and why it is a round trip
// ============================================================================
// mcycle is per-core and is NOT synchronized across chiplets, so "producer finished at
// t1 on chip A, consumer started at t2 on chip B" is not a measurable quantity -- that
// subtraction compares two unrelated clocks. So the whole sweep is one CHAIN of round
// trips, and every timestamp is taken on chip 0x00:
//
//   A0 --mids(phase 0)--> A1 --mids(phase 1)--> A2 -- ... --> A9
//   ^^                    ^^
//   anchor tasks, all on chip 0x00, each stamps mcycle once
//
// Phase i's latency is stamp(A_{i+1}) - stamp(A_i): out to the phase's participants and
// back. Both stamps come from the same core on the same chip, so the difference is real.
// It covers TWO edges (out and back), not one; see "Reading the numbers".
//
// ✅ The anchors are FUSED: A_{i+1} is simultaneously the sink of phase i and the source
// of phase i+1. That is what makes the whole sweep fit in one binary -- 41 tasks instead
// of 49 with separate sinks (see "Task budget").
//
// The tasks are __snax_kernel_sync_probe: no work, no printf. All that sits between two
// stamps is the manager resolving edges and the fabric carrying notifications.
//
// ⚠️ PHASE 0 IS AN UNTIMED WARM-UP, and it is not optional. A chiplet's FIRST touch
// costs about 1025 cc more than every later one -- measured: at P = 2 the one-to-one and
// one-to-all phases are the *same graph*, yet they read 4780 and 3755 cc when the first
// of them paid a cold start. The warm-up is a one-to-all over the largest rectangle this
// build sweeps, so every participating chiplet is touched once before anything is timed.
// Without it, every phase that first touches a chiplet is inflated and the P = 2 identity
// check fails.
//
// ⚠️ PHASE 1 IS A LOCAL CONTROL: its mid is on chip 0x00 too, so it has zero cross-chip
// distance and measures the software manager's own per-edge bookkeeping. Subtracting it
// isolates the cross-chip part; without it the remote numbers cannot be attributed to
// the fabric rather than to the scheduler.
//
// ============================================================================
// READING THE NUMBERS
// ============================================================================
//   rt(local)              phase 0, the control
//   one-way, ONE_TO_ONE    ( rt - rt(local) ) / 2      <- both legs are the same shape
//   fan-out+in, ONE_TO_ALL   rt - rt(local)            <- do NOT halve: the legs are
//                                                         different shapes
//                                                         (1->P-1 vs P-1->1)
//
// ⚠️ At P = 2 the two modes are the SAME graph (P-1 = 1), so they MUST agree. They are
// measured separately anyway: that agreement is the harness's own correctness check, and
// it is what caught the missing warm-up. If they disagree, do not read any other number
// on the page -- fix the harness first.
//
// ⚠️ These are UNOVERLAPPED costs -- nothing runs concurrently to hide them. That matches
// how the barriers were measured, which is the point, but it is an upper bound for a
// scheduler, which in a real workload can hide part of the latency behind computation.
// A barrier can hide none of it, so the comparison is conservative in the barrier's
// favour and should be reported that way.
//
// ============================================================================
// TASK BUDGET — why this shape, and what breaks if it grows
// ============================================================================
// ⚠️ bingo_task_create() allocates from the L2 heap in the narrow SPM. That heap is now
// 3/4 of the region (~24 KiB, see bingo_hemaia_system_mmap_init) rather than 1/2, which
// was only about 35 tasks. The full sweep needs 41:
//     11 anchors + 46 mids (15 warm-up + 1 local + 1+1 + 1+3 + 1+7 + 1+15) = 57
// and the heap now takes the whole narrow SPM above the comm buffer less a 1 KiB guard,
// which is 61 tasks at the measured 512 B per task. The SYNC_MAX_P = 4 smoke build needs
// 17. The run prints capacity and peak so the margin is measured, not assumed.
// The run prints the heap's own diagnostics at the end, so the margin is measured rather
// than assumed. If it ever overflows, task creation returns NULL and the graph is
// silently missing edges -- which is why the build error check below refuses to schedule.
//
// ⚠️ A task may have at most BINGO_MAX_REMOTE_SUCC remote successors, and exceeding it
// does not fail loudly: bingo_task_add_depend() prints and DROPS the edge, leaving a
// predecessor count that can never reach zero, i.e. a hang. The P = 16 fan-out needs 15,
// which is why that limit was raised from 8 to 16. The static assert below keeps it
// honest.

#pragma once
#include "host.h"
#include "libbingo/bingo_api.h"

#ifndef SYNC_MAX_P
#error "define SYNC_MAX_P (2, 4, 8 or 16) before including sync_latency_impl.h"
#endif

#define SYNC_SRC_CHIP ((uint8_t)0x00)  // source, and where every timestamp lives

// Rectangle bottom-right per P. Also the furthest chiplet from 0x00 in that rectangle,
// since chip_id = (x << 4) | y and both coordinates are maximal there.
//   P = 2  -> 0x01 (1x2), 1 hop      P = 8  -> 0x13 (2x4), 4 hops
//   P = 4  -> 0x11 (2x2), 2 hops     P = 16 -> 0x33 (4x4), 6 hops
#define SYNC_NUM_RECTS 4u
static const uint8_t SYNC_RECT_BR[SYNC_NUM_RECTS] = {0x01, 0x11, 0x13, 0x33};

#define SYNC_MODE_ONE_TO_ONE 0u
#define SYNC_MODE_ONE_TO_ALL 1u

#define SYNC_NUM_PHASES (2u + 2u * SYNC_NUM_RECTS)  // warm-up + control + 2 modes/rect
#define SYNC_NUM_ANCHORS (SYNC_NUM_PHASES + 1u)
#define SYNC_MAX_TASKS 96u

// The largest one-to-all fan-out in this build is SYNC_MAX_P - 1 remote successors.
// Exceeding the runtime's limit does not fail loudly -- the edge is dropped and the
// schedule stalls -- so catch it at compile time.
_Static_assert(SYNC_MAX_P - 1u <= BINGO_MAX_REMOTE_SUCC,
               "the one-to-all fan-out exceeds BINGO_MAX_REMOTE_SUCC; raise it in "
               "bingo_api.h");

typedef struct {
    uint8_t  br;     // rectangle bottom-right; 0xFF = local control or warm-up
    uint8_t  warmup; // 1 = untimed, not reported
    uint8_t  mode;
    uint32_t p;
    uint32_t hops;   // Manhattan distance to the furthest participant
    uint32_t edges;  // cross-chip edges the round trip traverses
} sync_phase_t;

static inline uint8_t sync_chip_at(uint32_t x, uint32_t y) {
    return (uint8_t)((x << 4) | (y & 0x0Fu));
}
static inline uint32_t sync_rect_count(uint8_t br) {
    return ((uint32_t)(br >> 4) + 1u) * ((uint32_t)(br & 0x0Fu) + 1u);
}
static inline uint32_t sync_hops(uint8_t chip) {  // Manhattan distance from 0x00
    return (uint32_t)(chip >> 4) + (uint32_t)(chip & 0x0Fu);
}
static inline int sync_rect_fits(uint8_t br) {
    return (uint32_t)(br >> 4) < (uint32_t)N_CHIPLETS_X &&
           (uint32_t)(br & 0x0Fu) < (uint32_t)N_CHIPLETS_Y;
}

// Allocate a probe's args on the chiplet that will RUN it, pointing at a stamp word on
// that same chiplet. On every other chiplet this returns 0: that chiplet never executes
// the task, so it never dereferences the pointer. Every chiplet still creates the task,
// in the same order, because bingo_runtime_schedule() indexes the shared list by
// position and remote notifications address tasks by id.
// ⚠️ TWO ADDRESS WIDTHS, and conflating them wedges the machine.
//
// bingo_l3_alloc() returns a FULL address including the 8-bit chiplet prefix in bits
// [47:40]. The device is RV32 and needs the prefix-free low 32 bits; the host is RV64 and
// must keep the prefix to reach its OWN memory.
//
// Truncating first and dereferencing the result on the host sends the access to chiplet
// 0x00 instead of the local chiplet -- a stray remote write on every chip except 0x00.
// That is not a benign mistake: it wedged chip 0x02's host permanently (it stopped
// retiring instructions mid-bingoHeapMalloc and never resumed), and it silently clobbered
// chip 0x00's heap from every other chiplet.
static inline uint32_t sync_make_args(uint8_t owner, uint64_t *stamp_out) {
    if (get_current_chip_id() != owner) return 0;
    __snax_kernel_sync_probe_args_t *args =
        (__snax_kernel_sync_probe_args_t *)bingo_l3_alloc(
            owner, sizeof(__snax_kernel_sync_probe_args_t));
    uint64_t stamp64 = bingo_l3_alloc(owner, sizeof(uint32_t));  // full, prefixed
    *(volatile uint32_t *)(uintptr_t)stamp64 = 0;                // host: keep the prefix
    args->stamp_addr = (uint32_t)stamp64;                        // device: low 32 bits
    if (stamp_out) *stamp_out = stamp64;                         // host reads it back later
    return (uint32_t)(uintptr_t)args;
}

static inline bingo_task_t *sync_probe_task(uint8_t owner, uint64_t *stamp_out) {
    uint32_t args = sync_make_args(owner, stamp_out);
    bingo_task_t *t = bingo_task_create(
        get_device_function("__snax_kernel_sync_probe"), args, owner, 0);
    if (t == NULL) {
        printf("Error: sync probe task creation failed (chip %x) -- L2 task heap full\r\n",
               owner);
    }
    return t;
}

// bingo_task_add_depend() dereferences both arguments, so never hand it a NULL left by
// a failed allocation.
static inline int sync_depend(bingo_task_t *task, bingo_task_t *dep) {
    if (task == NULL || dep == NULL) return -1;
    bingo_task_add_depend(task, dep);
    return 0;
}

static sync_phase_t g_phase[SYNC_NUM_PHASES];
static uint64_t     g_stamp[SYNC_NUM_ANCHORS];  // FULL addresses, chip 0x00 only
static uint32_t     g_num_phases;
static int          g_build_err;

uint32_t __workload_sync_latency(bingo_task_t **task_list) {
    check_kernel_tab_ready();
    if (get_device_function("__snax_kernel_sync_probe") == SNAX_SYMTAB_END_FN_ADDR) {
        printf("Error: __snax_kernel_sync_probe symbol lookup failed!\r\n");
        return 0;
    }

    uint32_t n = 0;
    g_num_phases = 0;
    g_build_err = 0;

    // The first anchor. Every later anchor closes one phase and opens the next.
    bingo_task_t *anchor = sync_probe_task(SYNC_SRC_CHIP, &g_stamp[0]);
    task_list[n++] = anchor;

    // ---- phase 0: UNTIMED WARM-UP -- touch every participating chiplet once ---------
    // One-to-all over the largest rectangle this build sweeps. Its result is discarded;
    // its only job is that no later phase pays a chiplet's first-touch cost.
    {
        uint8_t warm_br = 0;
        for (uint32_t r = 0; r < SYNC_NUM_RECTS; r++) {
            uint8_t br = SYNC_RECT_BR[r];
            if (sync_rect_fits(br) && sync_rect_count(br) <= SYNC_MAX_P) warm_br = br;
        }
        const uint32_t ph = g_num_phases++;
        g_phase[ph].br = 0xFF;
        g_phase[ph].warmup = 1;
        g_phase[ph].mode = SYNC_MODE_ONE_TO_ALL;
        g_phase[ph].p = warm_br ? sync_rect_count(warm_br) : 1;
        g_phase[ph].hops = 0;
        g_phase[ph].edges = 0;

        bingo_task_t *next = sync_probe_task(SYNC_SRC_CHIP, &g_stamp[ph + 1]);
        if (warm_br) {
            for (uint32_t x = 0; x <= (uint32_t)(warm_br >> 4); x++) {
                for (uint32_t y = 0; y <= (uint32_t)(warm_br & 0x0Fu); y++) {
                    uint8_t chip = sync_chip_at(x, y);
                    if (chip == SYNC_SRC_CHIP) continue;
                    bingo_task_t *mid = sync_probe_task(chip, NULL);
                    g_build_err |= sync_depend(mid, anchor);
                    g_build_err |= sync_depend(next, mid);
                    task_list[n++] = mid;
                }
            }
        } else {
            bingo_task_t *mid = sync_probe_task(SYNC_SRC_CHIP, NULL);
            g_build_err |= sync_depend(mid, anchor);
            g_build_err |= sync_depend(next, mid);
            task_list[n++] = mid;
        }
        task_list[n++] = next;
        anchor = next;
    }

    // ---- one phase per (rectangle x mode) -------------------------------------------
    for (uint32_t r = 0; r < SYNC_NUM_RECTS; r++) {
        const uint8_t br = SYNC_RECT_BR[r];
        if (!sync_rect_fits(br)) continue;             // larger than this cfg's grid
        if (sync_rect_count(br) > SYNC_MAX_P) continue;  // above this build's cap

        for (uint32_t mode = SYNC_MODE_ONE_TO_ONE; mode <= SYNC_MODE_ONE_TO_ALL; mode++) {
            const uint32_t p = sync_rect_count(br);
            const uint32_t ph = g_num_phases++;
            g_phase[ph].br = br;
            g_phase[ph].warmup = 0;
            g_phase[ph].mode = (uint8_t)mode;
            g_phase[ph].p = p;
            g_phase[ph].hops = sync_hops(br);
            g_phase[ph].edges = (mode == SYNC_MODE_ONE_TO_ONE) ? 2u : 2u * (p - 1u);

            bingo_task_t *next = sync_probe_task(SYNC_SRC_CHIP, &g_stamp[ph + 1]);

            if (mode == SYNC_MODE_ONE_TO_ONE) {
                bingo_task_t *mid = sync_probe_task(br, NULL);
                g_build_err |= sync_depend(mid, anchor);
                g_build_err |= sync_depend(next, mid);
                task_list[n++] = mid;
            } else {
                for (uint32_t x = 0; x <= (uint32_t)(br >> 4); x++) {
                    for (uint32_t y = 0; y <= (uint32_t)(br & 0x0Fu); y++) {
                        uint8_t chip = sync_chip_at(x, y);
                        if (chip == SYNC_SRC_CHIP) continue;
                        bingo_task_t *mid = sync_probe_task(chip, NULL);
                        g_build_err |= sync_depend(mid, anchor);
                        g_build_err |= sync_depend(next, mid);
                        task_list[n++] = mid;
                    }
                }
            }
            task_list[n++] = next;
            anchor = next;
        }
    }

    // ---- final phase: local control (zero cross-chip distance) ---------------------
    // ⚠️ Deliberately LAST, not between the warm-up and the first remote phase. While a
    // chip00-only phase runs, the remote hosts go idle, and the next remote phase then
    // pays a re-engagement cost -- measured at 287 cc, which broke the P = 2 identity
    // check (the same graph read 3883 vs 3596). Keeping every remote phase contiguous
    // removes it.
    {
        const uint32_t ph = g_num_phases++;
        g_phase[ph].br = 0xFF;
        g_phase[ph].warmup = 0;
        g_phase[ph].mode = SYNC_MODE_ONE_TO_ONE;
        g_phase[ph].p = 1;
        g_phase[ph].hops = 0;
        g_phase[ph].edges = 0;

        bingo_task_t *mid = sync_probe_task(SYNC_SRC_CHIP, NULL);
        bingo_task_t *next = sync_probe_task(SYNC_SRC_CHIP, &g_stamp[ph + 1]);
        g_build_err |= sync_depend(mid, anchor);
        g_build_err |= sync_depend(next, mid);
        task_list[n++] = mid;
        task_list[n++] = next;
        anchor = next;
    }

    asm volatile("fence" ::: "memory");
    return n;
}

int kernel_execution() {
    bingo_task_t *task_list[SYNC_MAX_TASKS] = {0};
    uint32_t num_tasks = __workload_sync_latency(task_list);
    if (num_tasks == 0) return -1;

    // ⚠️ Never schedule a graph that failed to build. A dropped edge means a predecessor
    // count that can never reach zero -- a hang, not a wrong number.
    if (g_build_err) {
        printf("Error: SYNC graph incomplete -- refusing to schedule\r\n");
        return -1;
    }

    bingo_runtime_schedule(task_list, num_tasks);

    // Only chip 0x00 holds the stamps, so only chip 0x00 reports. Printing happens after
    // the whole schedule has drained -- a printf between phases would land inside the
    // next phase's measurement.
    if (get_current_chip_id() == SYNC_SRC_CHIP) {
        // Two passes: the local control is the LAST phase (see above), so its value has
        // to be in hand before any "rt-local" can be printed.
        uint32_t rt[SYNC_NUM_PHASES];
        uint32_t local_rt = 0;
        for (uint32_t ph = 0; ph < g_num_phases; ph++) {
            rt[ph] = *(volatile uint32_t *)(uintptr_t)g_stamp[ph + 1] -
                     *(volatile uint32_t *)(uintptr_t)g_stamp[ph];
            if (!g_phase[ph].warmup && g_phase[ph].br == 0xFF) local_rt = rt[ph];
        }
        for (uint32_t ph = 0; ph < g_num_phases; ph++) {
            if (g_phase[ph].warmup) {
                printf("SYNC[sw] warmup P=%2u (untimed, discarded)      | rt=%u cc\r\n",
                       (unsigned)g_phase[ph].p, (unsigned)rt[ph]);
            } else if (g_phase[ph].br == 0xFF) {
                printf("SYNC[sw] LOCAL  P= 1 hops=0 edges= 0 | rt=%u cc\r\n",
                       (unsigned)rt[ph]);
            } else {
                printf("SYNC[sw] %s P=%2u hops=%u edges=%2u | rt=%u cc | "
                       "rt-local=%d cc\r\n",
                       g_phase[ph].mode == SYNC_MODE_ONE_TO_ONE ? "1to1  " : "1toall",
                       (unsigned)g_phase[ph].p, (unsigned)g_phase[ph].hops,
                       (unsigned)g_phase[ph].edges, (unsigned)rt[ph],
                       (int)(rt[ph] - local_rt));
            }
        }
        // Report the task-heap margin rather than assuming it: this is the limit that
        // caps how big a sweep can be, and it fails by silently dropping edges.
        BingoHeapDiagnostics *d = (BingoHeapDiagnostics *)(uintptr_t)
            bingoHeapGetDiagnostics(bingo_get_l2_heap_manager(get_current_chip_id()));
        printf("SYNC[sw] tasks=%u | L2 heap capacity=%lu peak=%lu oom=%lu\r\n",
               (unsigned)num_tasks, d->capacity, d->peak_allocated, d->oom_count);
    }

    printf("Chip(%x, %x): [Host] All tasks done.\n", get_current_chip_loc_x(),
           get_current_chip_loc_y());
    bingo_close_all_clusters(task_list, num_tasks);
    return 0;
}