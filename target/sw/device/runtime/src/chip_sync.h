// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Cross-chip barriers: the chiplet-spanning analogue of snrt_global_barrier().
//
// Two implementations live here. They have identical semantics and the same call
// pattern, and differ only in how a chip tells the others that it has arrived:
//
//   snrt_hw_chip_global_barrier()  announce = one 0xFF-BROADCAST store, which the D2D
//                                  router multicasts; waiting is a local poll.
//   snrt_sw_chip_global_barrier()  announce = one unicast store to the master chip,
//                                  which pushes the release back out; every wait is a
//                                  poll of local memory. No broadcast, no atomics.
//
// snrt_chip_global_barrier(use_sw) selects between them, so code that wants to switch
// mechanism -- or measure both -- changes one argument rather than its call sites.
//
// All of them hide the manual `checkpoint` integer and the chip-id-prefix bookkeeping
// that the raw chip_barrier() API exposes, and all take their participating-chip
// rectangle from chip_barrier_tl / chip_barrier_br in this chip's own comm_buffer_t.
//
// Usage (call from ALL cores):
//   snrt_chip_barrier_init(top_left_chip_id, bottom_right_chip_id);  // once, arms both
//   ...
//   snrt_chip_global_barrier(SNRT_CHIP_BARRIER_SW);   // at each cross-chip barrier point
//
// CONTRACT, for all of them: every participating chip must execute the same sequence of
// barrier calls, through the same mechanism. The host zero-initialises the comm buffer
// before the device runs.
//
// Included by snrt.h after sync.h / team.h / occamy_device.h / xchip_mem.h, so everything
// these build on is already visible.
#pragma once

//===============================================================
// Broadcast barrier
//===============================================================
// chip_barrier_checkpoint answers "which barrier am I on?". It is per-chip and never
// broadcast: every chip increments its own copy, and because all chips run the same
// barrier sequence, all reach their Nth barrier with counter == N. The only cross-chip
// traffic per barrier is the announce, already encapsulated in announce_chip_checkpoint();
// waiting is a purely local poll.
//
// The checkpoint is 8-bit, so at most 255 barriers per offload.

// Record the participating-chip rectangle and reset this chip's checkpoint counter. Call
// once per chip, from all cores. Only the representative core touches the comm buffer;
// the trailing barrier makes init a clean chip-wide sync point.
// Arms the broadcast barrier only -- see snrt_chip_barrier_init() to arm both.
static inline void snrt_hw_chip_barrier_init(uint8_t top_left_chip_id,
                                             uint8_t bottom_right_chip_id) {
    if (snrt_global_core_idx() == 0) {
        volatile comm_buffer_t* cb = get_communication_buffer();
        cb->chip_barrier_checkpoint = 0;          // re-run safe (host also zeroes)
        cb->chip_barrier_tl = top_left_chip_id;   // record rectangle (local write)
        cb->chip_barrier_br = bottom_right_chip_id;
    }
    snrt_global_barrier();
}

// Barrier over the recorded rectangle. Call from all cores; the per-chip checkpoint
// counter advances by one per call.
static inline void snrt_hw_chip_global_barrier(void) {
    BINGO_TRACE_MARKER(BINGO_TRACE_HW_CHIP_BARRIER_START);

    // 1) All cores / clusters on THIS chip arrive before the counter advances.
    snrt_global_barrier();

    // 2) One representative core per chip (compute core 0) handshakes cross-chip.
    //    announce/wait are pure CSR-Mseg/load sequences with no iDMA dependency,
    //    so a non-DM core is fine here.
    if (snrt_global_core_idx() == 0) {
        volatile comm_buffer_t* cb = get_communication_buffer();
        uint8_t cp = (uint8_t)(++cb->chip_barrier_checkpoint);  // hidden, per-chip

        BINGO_TRACE_MARKER(BINGO_TRACE_HW_CHIP_BARRIER_ANNOUNCE_START);
        announce_chip_checkpoint(cb, cp);                       // broadcast progress
        BINGO_TRACE_MARKER(BINGO_TRACE_HW_CHIP_BARRIER_ANNOUNCE_END);

        BINGO_TRACE_MARKER(BINGO_TRACE_HW_CHIP_BARRIER_WAIT_START);
        wait_chips_checkpoint(cb, cb->chip_barrier_tl,
                              cb->chip_barrier_br, cp);          // local poll others
        BINGO_TRACE_MARKER(BINGO_TRACE_HW_CHIP_BARRIER_WAIT_END);
    }

    // 3) No core leaves until the representative finished the cross-chip wait.
    snrt_global_barrier();

    BINGO_TRACE_MARKER(BINGO_TRACE_HW_CHIP_BARRIER_END);
}

//===============================================================
// Pure-software barrier
//===============================================================
// Built from nothing but ordinary point-to-point stores, on top of the cross-chiplet
// accessors in xchip_mem.h.
//
// WHY NOT A SHARED ATOMIC COUNTER (the direct port of snRuntime's _snrt_barrier)?
// Because a remote atomic cannot work on this platform -- see xchip_mem.h. Arrival is
// recorded in SINGLE-WRITER SLOTS instead: each chip writes only its own
// sw_barrier_arrive[] entry in the master's comm buffer. Distinct addresses, one writer
// each, no atomicity required, and the semantics are unchanged.
//
// Progress is only ever inferred from OBSERVING a value, never from a store retiring,
// because remote stores are fire-and-forget here (again, xchip_mem.h).
//
// The master is the rectangle's top-left chip -- 0x00 for a full grid. Generations are
// 32-bit and compared as signed differences, so this barrier has no 255-call limit.

// How long a waiting chip spins before re-asserting itself. Recovery only; see the use
// site in sw_chip_barrier_wait().
#ifndef SW_BARRIER_REANNOUNCE_SPINS
#define SW_BARRIER_REANNOUNCE_SPINS 4096u
#endif

// Rectangle geometry. chip_id is (x << 4) | y, so the rectangle spanned by (tl, br) is
// x in [tl>>4, br>>4] and y in [tl&0xF, br&0xF] -- the same decode wait_chips_checkpoint()
// uses. Participants are numbered in x-major order so the index is dense in [0, count).
static inline uint32_t sw_barrier_grid_height(uint8_t tl, uint8_t br) {
    return (uint32_t)(br & 0x0F) - (uint32_t)(tl & 0x0F) + 1u;
}

static inline uint32_t sw_barrier_participant_count(uint8_t tl, uint8_t br) {
    return ((uint32_t)(br >> 4) - (uint32_t)(tl >> 4) + 1u) *
           sw_barrier_grid_height(tl, br);
}

static inline uint32_t sw_barrier_participant_index(uint8_t chip_id, uint8_t tl,
                                                    uint8_t br) {
    return ((uint32_t)(chip_id >> 4) - (uint32_t)(tl >> 4)) *
               sw_barrier_grid_height(tl, br) +
           ((uint32_t)(chip_id & 0x0F) - (uint32_t)(tl & 0x0F));
}

static inline uint8_t sw_barrier_chip_of_index(uint32_t idx, uint8_t tl, uint8_t br) {
    uint32_t h = sw_barrier_grid_height(tl, br);
    uint32_t x = (uint32_t)(tl >> 4) + idx / h;
    uint32_t y = (uint32_t)(tl & 0x0F) + idx % h;
    return (uint8_t)((x << 4) | (y & 0x0Fu));
}

// Announce this chip's arrival at the next barrier and return the generation reached,
// which must be handed to sw_chip_barrier_wait(). Kept separate from the wait so the two
// halves can be traced independently: the network term and the skew term.
static inline uint32_t sw_chip_barrier_announce(
    volatile comm_buffer_t* chip_barrier_data_ptr, uint8_t top_left_chip_id,
    uint8_t bottom_right_chip_id) {
    // Explicit read-modify-write rather than ++ on a volatile, which leaves no doubt
    // about how many accesses the volatile qualifier implies. Safe because the generation
    // is local and single-writer.
    uint32_t gen = chip_barrier_data_ptr->sw_barrier_gen + 1u;
    chip_barrier_data_ptr->sw_barrier_gen = gen;
    uint8_t current_chip_id = get_current_chip_id();
    uint32_t idx = sw_barrier_participant_index(current_chip_id, top_left_chip_id,
                                                bottom_right_chip_id);
    volatile uint32_t* slot = &chip_barrier_data_ptr->sw_barrier_arrive[idx];
    if (current_chip_id == top_left_chip_id) {
        *slot = gen;  // the master owns the array; no cross-chip hop needed
    } else {
        snrt_xchip_writew(top_left_chip_id, SNRT_XCHIP_LOCAL_OFF(slot), gen);
    }
    return gen;
}

// Block until every participant has announced `gen`.
//
// Every cross-chip access in the steady state is a one-way STORE: arrivals are pushed to
// the master, the release is pushed back out to each participant, and both waits read
// only local memory. So the D2D links carry a bounded burst of writes per barrier rather
// than a read stream held open for as long as a chip is waiting.
static inline void sw_chip_barrier_wait(
    volatile comm_buffer_t* chip_barrier_data_ptr, uint8_t top_left_chip_id,
    uint8_t bottom_right_chip_id, uint32_t gen) {
    uint8_t current_chip_id = get_current_chip_id();
    uint32_t count =
        sw_barrier_participant_count(top_left_chip_id, bottom_right_chip_id);

    if (current_chip_id == top_left_chip_id) {
        // Master: the arrival slots are in ITS OWN comm buffer, so this poll is purely
        // local. Signed difference tolerates generation wraparound.
        for (uint32_t p = 0; p < count; p++) {
            volatile uint32_t* slot = &chip_barrier_data_ptr->sw_barrier_arrive[p];
            while ((int32_t)(*slot - gen) < 0) {
                asm volatile("fence" ::: "memory");
            }
        }
        // Release by pushing the generation into every other participant's own comm
        // buffer, so their wait never leaves their own die.
        for (uint32_t p = 0; p < count; p++) {
            uint8_t target =
                sw_barrier_chip_of_index(p, top_left_chip_id, bottom_right_chip_id);
            if (target == top_left_chip_id) continue;
            snrt_xchip_writew(
                target,
                SNRT_XCHIP_LOCAL_OFF(&chip_barrier_data_ptr->sw_barrier_release), gen);
        }
        // Publish locally last, so the master's own copy is always authoritative.
        chip_barrier_data_ptr->sw_barrier_release = gen;
    } else {
        uint32_t idx = sw_barrier_participant_index(current_chip_id, top_left_chip_id,
                                                    bottom_right_chip_id);
        uint32_t slot_off =
            SNRT_XCHIP_LOCAL_OFF(&chip_barrier_data_ptr->sw_barrier_arrive[idx]);
        uint32_t release_off =
            SNRT_XCHIP_LOCAL_OFF(&chip_barrier_data_ptr->sw_barrier_release);
        uint32_t spins = 0;
        for (;;) {
            asm volatile("fence" ::: "memory");
            if ((int32_t)(chip_barrier_data_ptr->sw_barrier_release - gen) >= 0) break;
            if (++spins >= SW_BARRIER_REANNOUNCE_SPINS) {
                spins = 0;
                // RECOVERY ONLY, and rare by construction. Stores are fire-and-forget, so
                // a dropped arrival or a dropped release push would never be resent by
                // anyone. Re-assert this chip's arrival, then read the master's
                // authoritative copy once in case it was the release that went missing.
                // One remote access per 4096 spins is negligible traffic, and it turns a
                // permanent hang into a slow barrier.
                snrt_xchip_writew(top_left_chip_id, slot_off, gen);
                if ((int32_t)(snrt_xchip_readw(top_left_chip_id, release_off) - gen) >= 0)
                    break;
            }
        }
    }
}

static inline void sw_chip_barrier(volatile comm_buffer_t* chip_barrier_data_ptr,
                                   uint8_t top_left_chip_id,
                                   uint8_t bottom_right_chip_id) {
    uint32_t gen = sw_chip_barrier_announce(chip_barrier_data_ptr, top_left_chip_id,
                                            bottom_right_chip_id);
    sw_chip_barrier_wait(chip_barrier_data_ptr, top_left_chip_id, bottom_right_chip_id,
                         gen);
}

// Record the participating-chip rectangle and reset this chip's barrier state. Call once
// per chip, from all cores, before the first snrt_sw_chip_global_barrier().
//
// Arms the software barrier only -- see snrt_chip_barrier_init() to arm both.
//
// ⚠️ ORDERING: the master clears its arrival slots when ITS device core reaches this
// point, so in principle a faster chip could announce beforehand and have that announce
// wiped. This is the same window snrt_hw_chip_barrier_init() already lives with (it does not
// clear chip_level_checkpoint[] either), and under offload_legacy each host zeroes its own
// comm buffer and brings up D2D before waking its own snitches. The re-announce guard in
// sw_chip_barrier_wait() makes a lost announce self-healing rather than a hang.
static inline void snrt_sw_chip_barrier_init(uint8_t top_left_chip_id,
                                             uint8_t bottom_right_chip_id) {
    if (snrt_global_core_idx() == 0) {
        volatile comm_buffer_t* cb = get_communication_buffer();
        cb->chip_barrier_tl = top_left_chip_id;
        cb->chip_barrier_br = bottom_right_chip_id;
        cb->sw_barrier_gen = 0;
        cb->sw_barrier_release = 0;
        for (uint32_t p = 0; p < SW_BARRIER_MAX_CHIPS; p++)
            cb->sw_barrier_arrive[p] = 0;
    }
    snrt_global_barrier();
}

// Barrier over the rectangle recorded by snrt_sw_chip_barrier_init(). Call from all cores.
//
// Instrumented with BINGO trace markers rather than cycle counters: they are magic NOPs,
// so they cost nothing architecturally, disappear entirely when BINGO_PERF_TRACING is off,
// and keep the barrier's signature free of measurement plumbing. The ANNOUNCE/WAIT pair
// gives the network-vs-skew split; see perf_tracing.h.
static inline void snrt_sw_chip_global_barrier(void) {
    BINGO_TRACE_MARKER(BINGO_TRACE_SW_CHIP_BARRIER_START);

    // 1) All cores / clusters on THIS chip arrive before the generation advances.
    snrt_global_barrier();

    // 2) One representative core per chip handshakes cross-chip.
    if (snrt_global_core_idx() == 0) {
        volatile comm_buffer_t* cb = get_communication_buffer();
        uint8_t tl = cb->chip_barrier_tl;
        uint8_t br = cb->chip_barrier_br;

        BINGO_TRACE_MARKER(BINGO_TRACE_SW_CHIP_BARRIER_ANNOUNCE_START);
        uint32_t gen = sw_chip_barrier_announce(cb, tl, br);
        BINGO_TRACE_MARKER(BINGO_TRACE_SW_CHIP_BARRIER_ANNOUNCE_END);

        BINGO_TRACE_MARKER(BINGO_TRACE_SW_CHIP_BARRIER_WAIT_START);
        sw_chip_barrier_wait(cb, tl, br, gen);
        BINGO_TRACE_MARKER(BINGO_TRACE_SW_CHIP_BARRIER_WAIT_END);
    }

    // 3) No core leaves until the representative finished the cross-chip wait.
    snrt_global_barrier();

    BINGO_TRACE_MARKER(BINGO_TRACE_SW_CHIP_BARRIER_END);
}

//===============================================================
// Unified entry points
//===============================================================
// The two barriers are interchangeable: same semantics, same contract, same rectangle.
// These let a caller pick the mechanism by argument instead of by function name, which
// is what makes it a one-line change to switch an application over -- or to run the same
// code under both and compare.

// `use_sw` values. Spelled out rather than passed as a bare true/false, because
// `snrt_chip_global_barrier(true)` at a call site says nothing about which barrier ran.
#define SNRT_CHIP_BARRIER_HW false  // 0xFF broadcast, multicast by the D2D router
#define SNRT_CHIP_BARRIER_SW true   // point-to-point stores only

// Record the participating-chip rectangle and arm BOTH barriers. Call once per chip,
// from all cores, before the first barrier of either kind. Costs a handful of extra
// stores over the single-mechanism inits, and in exchange the choice of mechanism stops
// depending on which init was called.
static inline void snrt_chip_barrier_init(uint8_t top_left_chip_id,
                                          uint8_t bottom_right_chip_id) {
    if (snrt_global_core_idx() == 0) {
        volatile comm_buffer_t* cb = get_communication_buffer();
        cb->chip_barrier_tl = top_left_chip_id;  // shared by both mechanisms
        cb->chip_barrier_br = bottom_right_chip_id;
        cb->chip_barrier_checkpoint = 0;         // broadcast barrier state
        cb->sw_barrier_gen = 0;                  // software barrier state
        cb->sw_barrier_release = 0;
        for (uint32_t p = 0; p < SW_BARRIER_MAX_CHIPS; p++)
            cb->sw_barrier_arrive[p] = 0;
    }
    snrt_global_barrier();
}

// Cross-chip barrier through the mechanism named by `use_sw` (SNRT_CHIP_BARRIER_SW /
// _HW). Call from all cores.
//
// ⚠️ Every participating chip must pass the same value at the same barrier. The two
// mechanisms keep separate state and do not observe one another, so a chip taking the
// software path while another takes the broadcast path will not rendezvous -- they will
// both wait for a signal that is being written somewhere else entirely.
//
// `use_sw` is a compile-time constant at almost every call site, so the branch folds away
// and this costs nothing over calling the chosen barrier directly.
static inline void snrt_chip_global_barrier(bool use_sw) {
    if (use_sw) {
        snrt_sw_chip_global_barrier();
    } else {
        snrt_hw_chip_global_barrier();
    }
}
