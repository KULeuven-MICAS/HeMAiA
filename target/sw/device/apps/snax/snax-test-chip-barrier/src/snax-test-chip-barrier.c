// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Characterisation of BOTH cross-chip barriers, in one run:
//
//   phase 1  snrt_sw_chip_global_barrier()  -- pure software, point-to-point stores only
//   phase 2  snrt_hw_chip_global_barrier()  -- 0xFF broadcast, multicast by the D2D router
//
// Same binary, same boot, same placement, same sweep; only the mechanism differs. That is
// what makes the two sets of numbers comparable -- two separate apps would have been free
// to drift apart in ways nobody would notice.
//
// PER MECHANISM, PER PARTICIPANT COUNT P:
//
//  1. LATENCY. After a couple of aligning barriers, run N_LAT_ITERS barriers back to back
//     with nothing in between. With no work to create skew, what is left is the barrier's
//     own cost -- lambda. Reported as min/avg/max.
//     Durations only: mcycle is per-core and is NOT synchronized across chiplets, so
//     absolute timestamps must never be compared between chips. The announce-vs-wait
//     split is not printed here; both barriers carry BINGO trace markers
//     (BINGO_TRACE_{SW,HW}_CHIP_BARRIER_*, perf_tracing.h) and the decomposition comes
//     from the trace, for every core rather than just this one.
//
//  2. CORRECTNESS. Each round, every participant stamps a strictly increasing tag into its
//     own slot in EVERY participant's TCDM, barriers, then checks that all P slots hold
//     the current tag. A slot holding an older tag means a chip left the barrier before
//     another arrived; the trailing barrier is what also makes "a chip ran a whole round
//     ahead" detectable rather than benign.
//
// SWEEPING P INSIDE ONE BINARY -- the point of this harness.
// The participant set is a rectangle of the compute grid, so a sub-rectangle gives a
// smaller P without touching the RTL: one build, one elaboration, one boot covers every P,
// instead of a full regeneration per point.
//
// ⚠️ THE INVARIANT: barrier counters are per-chip and are compared against what the other
// participants announce, so every chip must hold the same generation at each shared
// barrier. A chip sitting a round out therefore does not skip silently -- it calls
// snrt_chip_barrier_skip() for exactly the number of barriers the participants performed.
// Counters are only advanced, never reset, so a stale arrival slot from a different
// rectangle always reads as an older generation, i.e. "not arrived", never as a spurious
// arrival.
//
// ⚠️ WHY THE RENDEZVOUS BETWEEN RECTANGLES IS ALWAYS THE BROADCAST BARRIER, even in
// phase 1. The software barrier's arrival slots are indexed by a chip's POSITION IN THE
// CURRENT RECTANGLE, so different rectangles reuse the same slots for different chips. A
// chip that has finished skipping reaches the rendezvous early and announces there; with a
// software rendezvous that announce lands in a slot the still-running participants are
// polling, and it reads as an arrival that never happened. The broadcast barrier indexes
// chip_level_checkpoint[] by CHIP ID, which is unique and stable across rectangles, so it
// has no such aliasing. The rendezvous sits outside every measured region, so using it
// does not contaminate phase 1's numbers -- but it does mean this app needs the broadcast
// barrier to work in order to test the software one.
//
// Build/run (4-chiplet default cfg sweeps P = 2, 4; the 16-chiplet cfg sweeps 2, 4, 8, 16):
//   make single-sw HOST_APP_TYPE=offload_legacy CHIP_TYPE=multi_chip \
//                  WORKLOAD=None DEV_APP=snax-test-chip-barrier
//   cd target/sim/automation/test && python3 1_start_multi_chiplet_sim.py \
//        --host-app-type offload_legacy --chip-type multi_chip \
//        --workload None --dev-app snax-test-chip-barrier --engine vcs --waveform 0 \
//        [--cfg target/rtl/cfg/hemaia_16chiplet.hjson]

#include "snrt.h"

#include "chip_id.h"

#define N_LAT_ITERS 32u  // back-to-back barriers timed per participant count
#define N_ROUNDS 4u      // correctness rounds per participant count
#define N_WARMUP 2u      // untimed barriers before each timed loop

// Barriers a participant performs for one rectangle -- and therefore exactly the number a
// non-participant must skip.
#define BARRIERS_PER_RECT (N_WARMUP + N_LAT_ITERS + N_ROUNDS * 2u)

// Rectangles tried, smallest first. Each is anchored at chip 0x00, so chip 0x00 is the
// master throughout and never changes role.
//   0x01 -> 1 x 2 = 2      0x11 -> 2 x 2 = 4
//   0x13 -> 2 x 4 = 8      0x33 -> 4 x 4 = 16
// Entries that do not fit the grid this binary was built for are skipped at run time, so
// the same source runs on a 2x2 cfg (P = 2, 4) and a 4x4 one (P = 2, 4, 8, 16).
#define MAX_RECTS 4u
static const uint8_t RECT_BR[MAX_RECTS] = {0x01, 0x11, 0x13, 0x33};
#define RECT_TL ((uint8_t)0x00)

// The rendezvous rectangle: the whole compute grid, so every chip takes part.
#define FULL_BR ((uint8_t)((((N_CHIPLETS_X)-1) << 4) | (((N_CHIPLETS_Y)-1) & 0x0F)))

// Cross-chip flag slots, one word per participant, at a fixed TCDM offset. Every chip has
// an identical TCDM layout, so participant k's slot is at the same address everywhere and
// a remote write needs only the target's chiplet prefix.
#define FLAG_OFF 0x000u
#define FLAG_BASE_TAG 0xA5A50000u

#define PHASE_SW 0u
#define PHASE_HW 1u
#define N_PHASES 2u

// Broadcast barriers this run performs, worst case: every barrier of the HW phase, plus
// one rendezvous per rectangle in BOTH phases, plus the opening fence of each phase.
#define HW_BARRIER_BUDGET                                     \
    (MAX_RECTS * (BARRIERS_PER_RECT + 1u) + 1u /* HW phase */ \
     + MAX_RECTS + 1u /* SW phase rendezvous + fence */)

_Static_assert(HW_BARRIER_BUDGET <= 255,
               "the broadcast barrier's chip_barrier_checkpoint is 8-bit and this sweep "
               "would wrap it; lower N_LAT_ITERS or N_ROUNDS");
_Static_assert((N_CHIPLETS_X) * (N_CHIPLETS_Y) <= SW_BARRIER_MAX_CHIPS,
               "the compute grid has more chips than the barrier has arrival slots; "
               "raise SW_BARRIER_MAX_CHIPS in heterogeneous_runtime.h");

typedef struct {
    uint32_t p;        // participants; 0 = rectangle does not fit this grid
    uint32_t joined;   // did THIS chip take part
    uint32_t lat_min;
    uint32_t lat_avg;
    uint32_t lat_max;
} sweep_res_t;

// Results are collected and printed only at the very end: a printf between phases would
// skew whichever phase came after it.
// ⚠️ Written by the representative core ONLY. Every core runs main(), and a plain global
// is shared across the cluster, so an unguarded store here lets a non-representative core
// commit its own (never-accumulated) locals over the real measurement.
static sweep_res_t g_res[N_PHASES][MAX_RECTS];

// The correctness tag, derived from position in the sweep rather than from a running
// counter: every chip and every core computes the same value without having to agree on
// how many times a counter was incremented.
static inline uint32_t sweep_tag(uint32_t phase, uint32_t r, uint32_t round) {
    return FLAG_BASE_TAG + (phase * MAX_RECTS + r) * N_ROUNDS + round;
}

// Is `chip_id` inside the rectangle (tl, br)?
static inline int in_rect(uint8_t chip_id, uint8_t tl, uint8_t br) {
    return (chip_id >> 4) >= (tl >> 4) && (chip_id >> 4) <= (br >> 4) &&
           (chip_id & 0x0F) >= (tl & 0x0F) && (chip_id & 0x0F) <= (br & 0x0F);
}

// Widen to the full grid and re-join every chip. Always the broadcast barrier -- see the
// aliasing note in the file header.
static inline void rendezvous(void) {
    snrt_chip_barrier_set_rect(RECT_TL, FULL_BR);
    snrt_chip_global_barrier(SNRT_CHIP_BARRIER_HW);
}

// Sweep every rectangle for one mechanism. Returns an error count.
static int sweep_mechanism(bool use_sw, uint32_t phase) {
    const uint8_t chip_id = get_current_chip_id();
    const uint32_t tcdm_base = snrt_cluster_base_addrl();
    volatile uint32_t* flags = (volatile uint32_t*)(tcdm_base + FLAG_OFF);
    int err = 0;

    rendezvous();  // everyone starts the phase holding the same generation

    for (uint32_t r = 0; r < MAX_RECTS; r++) {
        const uint8_t br = RECT_BR[r];
        sweep_res_t* res = &g_res[phase][r];
        if (snrt_global_core_idx() == 0) {
            res->p = 0;
            res->joined = 0;
            res->lat_min = 0;
            res->lat_avg = 0;
            res->lat_max = 0;
        }

        // Skip rectangles that do not fit this grid.
        if ((br >> 4) >= (uint8_t)(N_CHIPLETS_X) ||
            (br & 0x0F) >= (uint8_t)(N_CHIPLETS_Y)) {
            continue;
        }
        const uint32_t p_count = sw_barrier_participant_count(RECT_TL, br);

        // Every chip is here, in lockstep. Now narrow the rectangle.
        snrt_chip_barrier_set_rect(RECT_TL, br);

        if (!in_rect(chip_id, RECT_TL, br)) {
            // Sitting this one out: advance the counter by exactly what the participants
            // will consume, then wait for them at the rendezvous below.
            snrt_chip_barrier_skip(use_sw, BARRIERS_PER_RECT);
            if (snrt_global_core_idx() == 0) res->p = p_count;
        } else {
            const uint32_t my_idx = sw_barrier_participant_index(chip_id, RECT_TL, br);
            // My slot sits at the SAME address in every chip's TCDM, so one address
            // serves all targets. Exactly one chip ever writes it.
            const uint32_t my_slot_addr = tcdm_base + FLAG_OFF + my_idx * 4u;

            for (uint32_t w = 0; w < N_WARMUP; w++) snrt_chip_global_barrier(use_sw);

            // ---- latency ----
            uint32_t lmin = 0xFFFFFFFFu, lmax = 0, lsum = 0;
            for (uint32_t i = 0; i < N_LAT_ITERS; i++) {
                uint32_t t0 = snrt_mcycle();
                snrt_chip_global_barrier(use_sw);
                uint32_t t1 = snrt_mcycle();
                if (snrt_global_core_idx() == 0) {
                    uint32_t lat = t1 - t0;
                    if (lat < lmin) lmin = lat;
                    if (lat > lmax) lmax = lat;
                    lsum += lat;
                }
            }
            // Commit on the representative core only -- the other cores never entered
            // the accumulation above and still hold the initialisers.
            if (snrt_global_core_idx() == 0) {
                res->p = p_count;
                res->joined = 1;
                res->lat_min = lmin;
                res->lat_max = lmax;
                res->lat_avg = lsum / N_LAT_ITERS;
            }

            // ---- correctness ----
            for (uint32_t round = 0; round < N_ROUNDS; round++) {
                const uint32_t tag = sweep_tag(phase, r, round);

                if (snrt_global_core_idx() == 0) {
                    for (uint32_t p = 0; p < p_count; p++) {
                        uint8_t target = sw_barrier_chip_of_index(p, RECT_TL, br);
                        if (target == chip_id) {
                            flags[my_idx] = tag;
                        } else {
                            snrt_xchip_writew(target, my_slot_addr, tag);
                        }
                    }
                }

                snrt_chip_global_barrier(use_sw);

                if (snrt_global_core_idx() == 0) {
                    // Tags increase strictly across the whole run, so a slot left over
                    // from an earlier rectangle holds a different value and is caught
                    // here rather than passing by coincidence.
                    for (uint32_t p = 0; p < p_count; p++) {
                        if (flags[p] != tag) err++;
                    }
                }

                // Without this second barrier a fast chip could stamp the next round
                // before a slow chip has read this one, which is the violation we want to
                // catch.
                snrt_chip_global_barrier(use_sw);
            }
        }

        rendezvous();
    }

    return err;
}

int main() {
    const uint32_t tcdm_base = snrt_cluster_base_addrl();
    volatile uint32_t* flags = (volatile uint32_t*)(tcdm_base + FLAG_OFF);
    int err = 0;

    // Arms BOTH mechanisms and records the full grid, so every chip is a participant for
    // the first rendezvous.
    snrt_chip_barrier_init(RECT_TL, FULL_BR);

    if (snrt_global_core_idx() == 0) {
        for (uint32_t p = 0; p < SW_BARRIER_MAX_CHIPS; p++) flags[p] = 0;
    }
    snrt_global_barrier();

    // Cross-chip fence: nobody may stamp a remote slot until every chip has zeroed its
    // own. (The init above is intra-chip only.)
    snrt_chip_global_barrier(SNRT_CHIP_BARRIER_HW);

    err += sweep_mechanism(SNRT_CHIP_BARRIER_SW, PHASE_SW);
    err += sweep_mechanism(SNRT_CHIP_BARRIER_HW, PHASE_HW);

    if (snrt_global_core_idx() == 0) {
        for (uint32_t phase = 0; phase < N_PHASES; phase++) {
            for (uint32_t r = 0; r < MAX_RECTS; r++) {
                const sweep_res_t* res = &g_res[phase][r];
                if (res->p == 0 || !res->joined) continue;
                printf("Chip(%x,%x) chip-barrier[%s] %s, err=%d | P=%2u | "
                       "barrier cc min/avg/max %u/%u/%u\n",
                       get_current_chip_loc_x(), get_current_chip_loc_y(),
                       phase == PHASE_SW ? "sw" : "hw", err ? "FAIL" : "PASS", err,
                       (unsigned)res->p, (unsigned)res->lat_min, (unsigned)res->lat_avg,
                       (unsigned)res->lat_max);
            }
        }
    }
    snrt_cluster_hw_barrier();

    return err;
}
