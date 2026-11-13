// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Core-level bingo kernels: dummy, stamp, chip_barrier, exit: the control and
// measurement tasks of the bingo-hw flow, none of which computes.

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

// Stamp: the smallest possible task. It does no work and prints nothing. With stamp_buf set it
// stores this core's mcycle into stamp_buf[slot] (a uint32_t array on this chiplet); with
// mark_addr set it then stores BINGO_SRAM_MARK_MAGIC | slot to mark_addr, a word in this chiplet's
// L1, which the testbench's SRAM probe (soc_probes.sv) watches on chiplet 0x00 and answers by
// printing every bank array's read and write counters (slots must come in order: 0, 1, ...); with
// both 0 it only completes. It serves wherever a task is needed for its place in the graph alone:
// a join or ordering point (a generator's probe(), fabric.py), the compiler's dep-tag relays
// (bingo_dfg_transforms.py), an empty kernel whose time is the scheduler's (cerf_moe), a timestamp
// (the time between two stamps measures the scheduler and the fabric rather than a kernel,
// sync_latency_sweep_1cluster), or a measured window's edge, stamp and mark (fabric.py stamp()).
//
// Examples, from a generator (mini_compiler, SnaxBingoKernelStampArgs), each with the graph it makes:
// S.node(chip, cluster, core, name, kernel, args, after) is one task that starts once every task in
// `after` is done; in the figures --> is such a dependency and [chip, cluster, core] where a task runs.
//
// 1. A join point: one task after a and b, which later tasks follow instead of each following both
//    (2 + n edges instead of 2 n, each edge a dependency slot on its consumer's core).
//
//      join = S.node(k, 0, dm, "join", "__snax_bingo_kernel_stamp", SnaxBingoKernelStampArgs(), [a, b])
//      c1 = S.node(k, 1, gemm, "c1", ..., [join])
//      c2 = S.node(k, 1, simd, "c2", ..., [join])
//
//          a --+                                            +--> c1  [k, 1, GEMM]
//              |                                            |
//              +--> join  [k, 0, DM]  no stamp, no mark  ---+
//              |                                            |
//          b --+                                            +--> c2  [k, 1, SIMD]
//
// 2. Two timestamps around `work`: buf[1] - buf[0] is the cycles from t0 to t1, both on one core's
//    clock; chip k's host reads buf after the run.
//
//      buf = BingoMemAlloc("stamps", size=2 * 4, mem_level="L2", chip_id=k, cluster_id=0)
//      t0 = S.node(k, 0, dm, "t0", "__snax_bingo_kernel_stamp", SnaxBingoKernelStampArgs(buf, 0), [ready])
//      work = S.node(k, 1, gemm, "work", ..., [t0])
//      t1 = S.node(k, 0, dm, "t1", "__snax_bingo_kernel_stamp", SnaxBingoKernelStampArgs(buf, 1), [work])
//
//          ready --> t0  [k, 0, DM]  buf[0] = mcycle
//                     |
//                     v
//                    work  [k, 1, GEMM]  the measured task
//                     |
//                     v
//                    t1  [k, 0, DM]  buf[1] = mcycle
//
// 3. A measured window on the root (fabric.py: setup() makes S0_start with stamp(), end_window()
//    S1_end and the report; each stamp stamps slot n into the build's stamp buffer and marks the SRAM
//    counters at the build's mark word).
//
//      start = S.stamp("start", setup_ends)
//      ...                                          # the measured tasks, after start
//      S.end_window(last_tasks)
//
//          every chip's setup (factor loads, zeroing, preloads)
//                     |
//                     v
//          S0_start  [root, 0, DM]  stamp[0] = mcycle, mark 0 --+--> each ring's first post [DM]
//                     |
//                     v
//          the measured tasks, on every chip
//                     |  (the last of them)
//                     v
//          S1_end    [root, 0, DM]  stamp[1] = mcycle, mark 1
//                     |
//                     v
//          report    [root, host]  prints stamp[1] - stamp[0]; the SRAM probe printed every bank
//                     |            array's counters at marks 0 and 1
//                     v
//          checks    [root, host]  the results against the golden
//
//    and the args the host writes for S1_end:
//      __snax_bingo_kernel_stamp_args_t a = {.stamp_buf = stamps, .slot = 1, .mark_addr = mark};
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_stamp(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t stamp_buf = ((uint32_t *)arg)[0];
    uint32_t slot      = ((uint32_t *)arg)[1];
    uint32_t mark_addr = ((uint32_t *)arg)[2];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_stamp_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_STAMP_START);
    if (stamp_buf) {
        ((volatile uint32_t *)(uintptr_t)stamp_buf)[slot] = snrt_mcycle();
    }
    if (mark_addr) {
        *(volatile uint32_t *)(uintptr_t)mark_addr = BINGO_SRAM_MARK_MAGIC | (slot & 0xFFFFu);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_STAMP_END);

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
