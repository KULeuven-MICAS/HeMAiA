// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Unified cross-chip barrier, analogous to snrt_global_barrier() but spanning
// chiplets. It hides the manual `checkpoint` integer (and all chip-id-prefix /
// broadcast bookkeeping) that the raw chip_barrier() API exposes.
//
// Two per-chip sync variables live in this chip's own comm_buffer_t and are
// reached only through the local `cb` pointer (no chiplet prefix is ever applied
// at the application level):
//   - chip_barrier_checkpoint : "which barrier am I on?" Local-access only,
//     never broadcast. Every chip increments its own copy; because all chips run
//     the same barrier sequence, all reach their Nth barrier with counter == N.
//   - chip_barrier_tl / chip_barrier_br : the participating-chip rectangle,
//     recorded once by snrt_chip_barrier_init().
// The only cross-chip traffic per barrier is the single announce broadcast
// (chip_level_checkpoint[chip_id] via the 0xFF prefix) already encapsulated in
// announce_chip_checkpoint(); waiting is a purely local poll.
//
// Usage (call from ALL cores):
//   snrt_chip_barrier_init(top_left_chip_id, bottom_right_chip_id);  // once
//   ...
//   snrt_chip_global_barrier();   // at each cross-chip barrier point
//
// Requirement (same contract as the raw chip_barrier()): every participating
// chip must execute the same sequence of snrt_chip_global_barrier() calls. The
// checkpoint path is 8-bit, so at most 255 barriers per offload; the host
// zero-initialises the comm buffer (incl. the counter) before the device runs.
//
// Included by snrt.h AFTER sync.h / team.h / occamy_device.h, so the helpers it
// builds on (snrt_global_barrier, snrt_global_core_idx, get_communication_buffer,
// announce_chip_checkpoint, wait_chips_checkpoint) are already visible.
#pragma once

// Record the participating-chip rectangle for this chip and reset its hidden
// checkpoint counter. Call once per chip, from all cores, before the first
// snrt_chip_global_barrier(). Only the representative core touches the comm
// buffer; the trailing barrier makes init a clean chip-wide sync point.
static inline void snrt_chip_barrier_init(uint8_t top_left_chip_id,
                                          uint8_t bottom_right_chip_id) {
    if (snrt_global_core_idx() == 0) {
        volatile comm_buffer_t* cb = get_communication_buffer();
        cb->chip_barrier_checkpoint = 0;          // re-run safe (host also zeroes)
        cb->chip_barrier_tl = top_left_chip_id;   // record rectangle (local write)
        cb->chip_barrier_br = bottom_right_chip_id;
    }
    snrt_global_barrier();
}

// Cross-chip barrier over the rectangle recorded by snrt_chip_barrier_init().
// Call from all cores; the hidden per-chip checkpoint counter advances by one
// per call.
static inline void snrt_chip_global_barrier(void) {
    // 1) All cores / clusters on THIS chip arrive before the counter advances.
    snrt_global_barrier();

    // 2) One representative core per chip (compute core 0) handshakes cross-chip.
    //    announce/wait are pure CSR-Mseg/load sequences with no iDMA dependency,
    //    so a non-DM core is fine here.
    if (snrt_global_core_idx() == 0) {
        volatile comm_buffer_t* cb = get_communication_buffer();
        uint8_t cp = (uint8_t)(++cb->chip_barrier_checkpoint);  // hidden, per-chip
        announce_chip_checkpoint(cb, cp);                       // broadcast progress
        wait_chips_checkpoint(cb, cb->chip_barrier_tl,
                              cb->chip_barrier_br, cp);          // local poll others
    }

    // 3) No core leaves until the representative finished the cross-chip wait.
    snrt_global_barrier();
}
