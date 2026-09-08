// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Test and characterisation of the PURE-SOFTWARE cross-chip barrier
// (snrt_sw_chip_global_barrier(), target/sw/device/runtime/src/chip_sync.h).
//
// This is "arm A" of the min-SFR synthetic benchmark: a barrier built from nothing but
// ordinary point-to-point loads and stores. The chip at the rectangle's top-left owns
// the arrival slots in its comm buffer; every chip writes only its own slot, so no
// atomic is required (a remote AMO is de-atomized on the issuing chip by
// i_soc_narrow_wide_amo_adapter and would lose updates -- see chip_sync.h).
//
// It does two things:
//
//  1. LATENCY. After a few aligning barriers, run N_LAT_ITERS barriers back to back with
//     nothing in between. With no work to create skew, what is left is the barrier's own
//     cost -- lambda. Reported as min / mean / max per chip.
//     Durations only: mcycle is per-core and is NOT synchronized across chiplets, so
//     absolute timestamps must never be compared between chips.
//     The announce-vs-wait split is NOT printed here: the barrier carries BINGO trace
//     markers (BINGO_TRACE_SW_CHIP_BARRIER_*, perf_tracing.h) and the decomposition comes
//     from the trace, which also gets it for every core rather than just this one.
//
//  2. CORRECTNESS. Each round, every chip stamps the round tag into its own slot in
//     EVERY participant's TCDM, then barriers, then checks that all P slots hold the
//     current round. A slot holding r-1 means a chip left the barrier before another
//     arrived; a slot holding r+1 means one ran a whole round ahead. The trailing
//     barrier is what makes the second case detectable rather than benign.
//
// All printing happens after both loops: a printf inside either one would dominate the
// measurement.
//
// Build/run (4-chiplet default cfg):
//   make single-sw HOST_APP_TYPE=offload_legacy CHIP_TYPE=multi_chip \
//                  WORKLOAD=None DEV_APP=snax-test-sw-chip-barrier
//   cd target/sim/automation/test && python3 1_start_multi_chiplet_sim.py \
//        --host-app-type offload_legacy --chip-type multi_chip \
//        --workload None --dev-app snax-test-sw-chip-barrier --engine vcs --waveform 0
// Add --cfg target/rtl/cfg/hemaia_16chiplet.hjson for the 4x4 grid.

#include "snrt.h"

#include "chip_id.h"

// Participating rectangle = the whole compute grid, derived from the generated platform
// header so the same source runs on a 2x2 and a 4x4 cfg unchanged.
#define BARRIER_TOP_LEFT ((uint8_t)0x00)
#define BARRIER_BOTTOM_RIGHT \
    ((uint8_t)((((N_CHIPLETS_X)-1) << 4) | (((N_CHIPLETS_Y)-1) & 0x0F)))

#define N_LAT_ITERS 64u  // back-to-back barriers timed for lambda
#define N_ROUNDS 8u      // correctness rounds
#define N_WARMUP 3u      // untimed barriers before the timed loop

// Cross-chip flag slots, one word per participant, at a fixed TCDM offset. Every chip
// has an identical TCDM layout, so chip k's slot is at the same address everywhere and
// a remote write needs only the target's chiplet prefix.
#define FLAG_OFF 0x000u
#define FLAG_TAG(r) (0xA5A50000u | (uint32_t)(r))

_Static_assert((N_CHIPLETS_X) * (N_CHIPLETS_Y) <= SW_BARRIER_MAX_CHIPS,
               "the compute grid has more chips than the barrier has arrival slots; "
               "raise SW_BARRIER_MAX_CHIPS in heterogeneous_runtime.h");

int main() {
    const uint8_t chip_id = get_current_chip_id();
    const uint8_t tl = BARRIER_TOP_LEFT;
    const uint8_t br = BARRIER_BOTTOM_RIGHT;
    const uint32_t P = sw_barrier_participant_count(tl, br);
    int err = 0;

    // Chips outside the rectangle take no part (all cores agree here). With tl = 0x00
    // and br spanning the whole grid this never fires, but the guard keeps the app
    // correct if the rectangle is ever narrowed.
    if ((chip_id >> 4) < (tl >> 4) || (chip_id >> 4) > (br >> 4) ||
        (chip_id & 0x0F) < (tl & 0x0F) || (chip_id & 0x0F) > (br & 0x0F)) {
        return -1;
    }

    const uint32_t tcdm_base = snrt_cluster_base_addrl();
    volatile uint32_t* flags = (volatile uint32_t*)(tcdm_base + FLAG_OFF);
    const uint32_t my_idx = sw_barrier_participant_index(chip_id, tl, br);
    // My slot sits at the SAME address in every chip's TCDM, so one address serves all
    // targets. Exactly one chip ever writes it.
    const uint32_t my_slot_addr = tcdm_base + FLAG_OFF + my_idx * 4u;

    if (snrt_global_core_idx() == 0) {
        for (uint32_t p = 0; p < P; p++) flags[p] = 0;
    }

    snrt_sw_chip_barrier_init(tl, br);

    // Cross-chip fence: nobody may stamp a remote slot until every chip has zeroed its
    // own. (The init above is intra-chip only.)
    snrt_sw_chip_global_barrier();

    // ---- 1: latency ------------------------------------------------------------
    uint32_t lat_min = 0xFFFFFFFFu, lat_max = 0, lat_sum = 0;

    // Align: everyone starts the timed loop together.
    for (uint32_t w = 0; w < N_WARMUP; w++) snrt_sw_chip_global_barrier();

    for (uint32_t i = 0; i < N_LAT_ITERS; i++) {
        uint32_t t0 = snrt_mcycle();
        snrt_sw_chip_global_barrier();
        uint32_t t1 = snrt_mcycle();

        if (snrt_global_core_idx() == 0) {
            uint32_t lat = t1 - t0;
            if (lat < lat_min) lat_min = lat;
            if (lat > lat_max) lat_max = lat;
            lat_sum += lat;
        }
    }

    // ---- 2: correctness --------------------------------------------------------
    for (uint32_t r = 0; r < N_ROUNDS; r++) {
        const uint32_t tag = FLAG_TAG(r);

        if (snrt_global_core_idx() == 0) {
            for (uint32_t p = 0; p < P; p++) {
                uint8_t target = sw_barrier_chip_of_index(p, tl, br);
                if (target == chip_id) {
                    flags[my_idx] = tag;
                } else {
                    snrt_xchip_writew(target, my_slot_addr, tag);
                }
            }
        }

        snrt_sw_chip_global_barrier();

        if (snrt_global_core_idx() == 0) {
            for (uint32_t p = 0; p < P; p++) {
                if (flags[p] != tag) err++;
            }
        }

        // Without this second barrier a fast chip could stamp round r+1 before a slow
        // chip has read round r, which is exactly the violation we want to catch.
        snrt_sw_chip_global_barrier();
    }

    // ---- report ----------------------------------------------------------------
    // Printing only after every loop, because a printf on the measured path would
    // dominate what it is measuring.
    if (snrt_global_core_idx() == 0) {
        printf(
            "Chip(%x,%x) snax-test-sw-chip-barrier %s, err=%d | P=%u | "
            "barrier cc min/avg/max %u/%u/%u\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(), err ? "FAIL" : "PASS",
            err, (unsigned)P, (unsigned)lat_min, (unsigned)(lat_sum / N_LAT_ITERS),
            (unsigned)lat_max);
    }
    snrt_cluster_hw_barrier();

    return err;
}
