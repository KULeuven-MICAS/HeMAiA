// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Core-level bingo kernels: dummy, entry_point, exit. These are the minimal
// control kernels the bingo-hw scheduler uses to mark task boundaries.

#pragma once

#include "../macros.h"

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_dummy(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t dummy_input = ((uint32_t *)arg)[0];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_dummy_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    printf_safe("[Cluster %d Core %d]: Bingo Dummy Kernel: %d\r\n", snrt_cluster_idx(), snrt_cluster_core_idx(), dummy_input);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    BINGO_RETURN_SUCC(sp, 0);
}

// Sync probe: the smallest possible task. No work, no printf -- its whole purpose is to
// be the endpoint of a dependency edge, so the time between two probes measures the
// scheduler and the fabric rather than a kernel.
//
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_sync_probe(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t stamp_buf = ((uint32_t *)arg)[0];
    uint32_t slot      = ((uint32_t *)arg)[1];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_sync_probe_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SYNC_PROBE_START);
    if (stamp_buf) {
        ((volatile uint32_t *)(uintptr_t)stamp_buf)[slot] = snrt_mcycle();
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_SYNC_PROBE_END);

    BINGO_RETURN_SUCC(sp, 0);
}

// Sync mark: the sync probe's stamp, then one 32-bit store of BINGO_SYNC_MARK_MAGIC | slot to
// mark_addr, a word in this chiplet's L1. The testbench's SRAM probe (soc_probes.sv) watches
// L1 for that store and prints every bank array's read and write counters when it sees it, so
// two marks bracket the SRAM accesses between them. mark_addr 0: a plain sync probe.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_sync_mark(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t stamp_buf = ((uint32_t *)arg)[0];
    uint32_t slot      = ((uint32_t *)arg)[1];
    uint32_t mark_addr = ((uint32_t *)arg)[2];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_sync_mark_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SYNC_PROBE_START);
    if (stamp_buf) {
        ((volatile uint32_t *)(uintptr_t)stamp_buf)[slot] = snrt_mcycle();
    }
    if (mark_addr) {
        *(volatile uint32_t *)(uintptr_t)mark_addr = BINGO_SYNC_MARK_MAGIC | (slot & 0xFFFFu);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_SYNC_PROBE_END);

    BINGO_RETURN_SUCC(sp, 0);
}

// Chip barrier: one cross-chip barrier as a BINGO task, run by ONE core of every chip in
// the rectangle. snrt_{sw,hw}_chip_global_barrier() cannot be a task: they open with
// snrt_global_barrier(), which waits for every core of the cluster, and the other cores are
// running other tasks. This calls the single-core halves directly.
//   mode 0: point-to-point software barrier -- the arrival goes to the top-left chip, which
//           collects every arrival and pushes the release back (chip_sync.h)
//   mode 1: 0xFF broadcast barrier -- one broadcast store of this chip's checkpoint, then a
//           poll of the local checkpoint array (heterogeneous_runtime.h)
// Memory chips inside the rectangle run no code and are skipped by both. Both keep their
// per-chip counter in the comm buffer, which the host zeroes (the host chip_barrier kernel,
// op 0) before any chip's first barrier. seq is the barrier's 1-based index in the program:
// every chip runs the same barrier sequence, so the counter must equal it (the 8-bit
// broadcast checkpoint modulo 256), and a mismatch prints a failed check.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_chip_barrier(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t mode = ((uint32_t *)arg)[0];
    uint8_t  tl   = (uint8_t)((uint32_t *)arg)[1];
    uint8_t  br   = (uint8_t)((uint32_t *)arg)[2];
    uint32_t seq  = ((uint32_t *)arg)[3];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_chip_barrier_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    volatile comm_buffer_t* cb = get_communication_buffer();
    uint32_t reached, expect;
    if (mode == 0) {
        BINGO_TRACE_MARKER(BINGO_TRACE_SW_CHIP_BARRIER_START);
        reached = sw_chip_barrier_announce(cb, tl, br);
        sw_chip_barrier_wait(cb, tl, br, reached);
        BINGO_TRACE_MARKER(BINGO_TRACE_SW_CHIP_BARRIER_END);
        expect = seq;
    } else {
        BINGO_TRACE_MARKER(BINGO_TRACE_HW_CHIP_BARRIER_START);
        uint8_t cp = (uint8_t)(cb->chip_barrier_checkpoint + 1u);
        cb->chip_barrier_checkpoint = cp;
        announce_chip_checkpoint(cb, cp);
        wait_chips_checkpoint(cb, tl, br, cp);
        BINGO_TRACE_MARKER(BINGO_TRACE_HW_CHIP_BARRIER_END);
        reached = cp;
        expect = seq & 0xFFu;
    }
    if (seq && reached != expect) {
        printf_safe("Chip %x: [Dev] Check [chip_barrier_seq]: FAIL mode %d barrier %d, counter %d\r\n",
                    get_current_chip_id(), (int)mode, (int)seq, (int)reached);
    }

    BINGO_RETURN_SUCC(sp, reached);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_exit(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t exit_code = ((uint32_t *)arg)[0];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_exit_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    printf_debug("[Cluster %d Core %d]: Exiting with code %d\r\n", snrt_cluster_idx(), snrt_cluster_core_idx(), exit_code);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    sp->return_value = BINGO_RET_EXIT;
    sp->num_return_values = 0;
    return BINGO_RET_EXIT;
}
