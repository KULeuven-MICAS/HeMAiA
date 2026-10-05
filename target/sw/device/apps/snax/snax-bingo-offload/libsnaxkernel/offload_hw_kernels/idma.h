// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Core-level bingo IDMA kernels: 1D copy and broadcast. These program the
// local snitch iDMA to move data between L1 and L3 / other clusters without
// touching the versacore streamer.

#pragma once

#include "../macros.h"

// How many concurrent iDMA descriptors one logical 1-D copy is issued as.
//
// Splitting speeds the transfer itself up, and the array's operand stall falls with it, but
// arming N descriptors costs N times as much configuration -- and that configuration lands
// on the load chain that gates the next dispatch. End to end a split pipeline is slower
// than an unsplit one, so this is left at 1.
//
// The transfer rate that motivates splitting is real, and it is well under what the
// reference sustains for the same bytes. Outstanding-descriptor depth is not what limits
// it, so look elsewhere before raising this.
#ifndef BINGO_IDMA_SPLIT
#define BINGO_IDMA_SPLIT 1
#endif

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_idma_1d_copy(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_idma_1d_copy_args_t);
    if (snrt_is_dm_core()){
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint64_t src_addr = make_u64(((uint32_t *)arg)[0], ((uint32_t *)arg)[1]);
        uint64_t dst_addr = make_u64(((uint32_t *)arg)[2], ((uint32_t *)arg)[3]);
        uint32_t data_size = ((uint32_t *)arg)[4];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_idma_1d_copy_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
        // ISSUE THE COPY AS BINGO_IDMA_SPLIT CONCURRENT TRANSFERS, NOT ONE.
        //
        // A single large L3->L1 load runs at a fixed rate well under what the port can
        // carry, and that rate barely moves whether the array is idle or fully busy. It is
        // therefore not losing to TCDM contention; it is one transfer limited by round-trip
        // latency rather than by bandwidth.
        //
        // Several transfers in flight cover that latency for each other. Bytes moved, burst
        // shape and destination layout are all unchanged -- only the number of outstanding
        // descriptors differs -- so this is free if the limit is elsewhere and roughly
        // linear if it is depth.
        {
            const uint32_t n = (uint32_t)BINGO_IDMA_SPLIT;
            // Split on a 64 B beat so no chunk straddles one; fall back to a single
            // transfer when the size does not divide cleanly.
            const uint32_t chunk = (data_size / n) & ~63u;
            if (n > 1u && chunk != 0u) {
                for (uint32_t i = 0; i < n - 1u; i++)
                    snrt_dma_start_1d_wideptr(dst_addr + i * chunk,
                                              src_addr + i * chunk, chunk);
                const uint32_t done = (n - 1u) * chunk;
                snrt_dma_start_1d_wideptr(dst_addr + done, src_addr + done,
                                          data_size - done);
            } else {
                snrt_dma_start_1d_wideptr(dst_addr, src_addr, data_size);
            }
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
        snrt_dma_wait_all();
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
        IDMA_DEBUG_PRINT("IDMA copy completed\r\n");
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else{
        printf_safe("[Cluster %d Core %d]: Error! IDMA 1D copy should be called from a DM core!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

// A broadcast collective's producer (libs/blocks/collective.py, impl "bcast"): its part into
// this chip's copy of the hand-off array and to the same address on every other compute chip
// (a store to chip 0xFF; the router never delivers a broadcast back to its source, so this
// chip's copy is written locally), then the landed flag the same two ways. All four go out on
// this one iDMA, in order, so a flag that has landed says the part before it has. The flag's
// source is this task's own `value` argument.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_idma_bcast_put(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_idma_bcast_put_args_t);
    if (!snrt_is_dm_core()) {
        printf_safe("[Cluster %d Core %d]: Error! IDMA bcast put should be called from a DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    __snax_bingo_kernel_idma_bcast_put_args_t *a = (__snax_bingo_kernel_idma_bcast_put_args_t *)arg;
    const uint64_t src = make_u64(a->src_addr_hi, a->src_addr_lo);
    const uint64_t dst = make_u64(a->dst_addr_hi, a->dst_addr_lo);
    const uint64_t flag = make_u64(a->flag_addr_hi, a->flag_addr_lo);
    // this chip's prefix: a DMA address whose high word is 0 names chip 0x00
    const uint64_t val = chiplet_addr_transform((uint64_t)(uint32_t)(uintptr_t)&a->value);
    const uint32_t size = a->size;
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_idma_bcast_put_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
    snrt_dma_start_1d_wideptr(dst, src, size);
    snrt_dma_start_1d_wideptr(chiplet_addr_transform_loc(0xF, 0xF, dst), src, size);
    snrt_dma_start_1d_wideptr(flag, val, 4);
    snrt_dma_start_1d_wideptr(chiplet_addr_transform_loc(0xF, 0xF, flag), val, 4);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
    snrt_dma_wait_all();
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
    sp->return_value = (uint32_t)dst;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// A broadcast collective's destination: spin until each producer's landed flag (64 B apart,
// in this chip's memory) holds `value` -- the producers are ordering-only edges, not
// dependencies, so the flags are the proof their parts are here -- then copy the gathered
// array from this chip's memory into L1. No read crosses a link.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_idma_fetch_flagged(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_idma_fetch_flagged_args_t);
    if (!snrt_is_dm_core()) {
        printf_safe("[Cluster %d Core %d]: Error! IDMA flagged fetch should be called from a DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    __snax_bingo_kernel_idma_fetch_flagged_args_t *a = (__snax_bingo_kernel_idma_fetch_flagged_args_t *)arg;
    const uint64_t src = make_u64(a->src_addr_hi, a->src_addr_lo);
    const uint64_t dst = make_u64(a->dst_addr_hi, a->dst_addr_lo);
    const uint64_t flags = make_u64(a->flags_addr_hi, a->flags_addr_lo);
    const uint32_t n = a->n_flags, value = a->value, size = a->size;
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_idma_fetch_flagged_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
    // The poll: ONE strided iDMA copy brings every flag word (each at the head of its 64 B
    // slot) into dst -- free until the data lands there -- and the core checks them in L1.
    // Reading the flags one by one from L3 cost a round trip each: the last
    // producer's flag landed, then the rest were read in turn. A dst too small for the n
    // words falls back to those reads.
    if (size >= 4u * n) {
        volatile uint32_t *seen = (volatile uint32_t *)(uintptr_t)(uint32_t)dst;
        uint32_t polls = 0, i = 0;
        for (;;) {
            snrt_dma_start_2d_wideptr(dst, flags, 4, 4, 64, n);
            snrt_dma_wait_all();
            while (i < n && seen[i] == value) i++;   // flags already seen stay landed
            if (i == n) break;
            // a flag that never lands means a producer never ran: say which, keep waiting
            if (++polls == (1u << 18))
                printf_safe("[Cluster %d]: flagged fetch still waiting for flag %d of %d\r\n",
                            snrt_cluster_idx(), (int)i, (int)n);
        }
    } else {
        for (uint32_t i = 0; i < n; i++) {
            volatile uint32_t *f = (volatile uint32_t *)(uintptr_t)(flags + (uint64_t)i * 64u);
            uint32_t spins = 0;
            while (*f != value) {
                if (++spins == (1u << 22))
                    printf_safe("[Cluster %d]: flagged fetch still waiting for flag %d of %d\r\n",
                                snrt_cluster_idx(), (int)i, (int)n);
            }
        }
    }
    snrt_dma_start_1d_wideptr(dst, src, size);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
    snrt_dma_wait_all();
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
    sp->return_value = (uint32_t)dst;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_idma_broadcast(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_idma_broadcast_args_t);

    // Copy 1d data from src to dst using idma
    // Arg0: uint32_t src_addr_hi
    // Arg1: uint32_t src_addr_lo
    // Arg2: uint32_t dst_addr_hi
    // Arg3: uint32_t dst_addr_lo
    // Arg4: uint32_t size in Byte
    if (snrt_is_dm_core()){
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint64_t src_addr = make_u64(((uint32_t *)arg)[0], ((uint32_t *)arg)[1]);
        uint64_t dst_addr = make_u64(((uint32_t *)arg)[2], ((uint32_t *)arg)[3]);
        uint64_t dst_addr_broadcast = chiplet_addr_transform_loc(0xF, 0xF, dst_addr);
        uint32_t data_size = ((uint32_t *)arg)[4];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_idma_broadcast_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
        snrt_dma_start_1d_wideptr(dst_addr_broadcast, src_addr, data_size);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
        snrt_dma_wait_all();
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
        IDMA_DEBUG_PRINT("IDMA copy completed\r\n");
        IDMA_DEBUG_PRINT("SRC ADDR = %lx\r\n", src_addr);
        IDMA_DEBUG_PRINT("DST ADDR = %lx\r\n", dst_addr_broadcast);
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else{
        printf_safe("[Cluster %d Core %d]: Error! IDMA 1D copy should be called from a DM core!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_idma_pairwise_swap(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_idma_pairwise_swap_args_t);
    // Adjacent element-pair swap over a flat buffer: dst[i] = src[i ^ 1]
    // (x[1],x[0],x[3],x[2],...) — the data half of RoPE rotate_half. In-layer
    // rope_q/rope_k swap a Q/K buffer that only exists at run time, so the swap is
    // produced on the device rather than by the datagen. It is offloaded to the iDMA
    // as two strided element copies (odd->even, even->odd slots); the DM core only
    // issues the descriptors. Works for any operand memory (L1/L3) since the iDMA
    // addresses the full hierarchy.
    // src and dst must NOT alias (the two strided copies would overlap).
    //
    // Arg layout (uint32_t[]):
    //   [0]  src_addr_hi
    //   [1]  src_addr_lo
    //   [2]  dst_addr_hi
    //   [3]  dst_addr_lo
    //   [4]  num_elems    (total element count, must be even)
    //   [5]  elem_bytes   (1=int8, 2=int16/fp16, 4=int32)
    if (snrt_is_dm_core()){
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t *a = (uint32_t *)arg;
        uint64_t src_addr = make_u64(a[0], a[1]);
        uint64_t dst_addr = make_u64(a[2], a[3]);
        uint32_t num_elems = a[4];
        uint32_t elem_bytes = a[5];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_idma_pairwise_swap_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

        uint32_t pairs = num_elems / 2u;         // N/2 pairs
        uint32_t pair_bytes = 2u * elem_bytes;   // stride from one pair to the next
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
        // dst[2k]   = src[2k+1]   (odd elements -> even slots)
        snrt_dma_start_2d_wideptr(dst_addr, src_addr + elem_bytes,
                                  elem_bytes, pair_bytes, pair_bytes, pairs);
        // dst[2k+1] = src[2k]     (even elements -> odd slots)
        snrt_dma_start_2d_wideptr(dst_addr + elem_bytes, src_addr,
                                  elem_bytes, pair_bytes, pair_bytes, pairs);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
        snrt_dma_wait_all();
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
        IDMA_DEBUG_PRINT("IDMA pairwise swap done (%u pairs, %u-byte elems)\r\n", pairs, elem_bytes);
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else{
        printf_safe("[Cluster %d Core %d]: Error! IDMA pairwise swap should be called from a DM core!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}


// Up to three dimensions of runs in one node: `outer` repetitions of `reps` runs of `size`
// bytes. Each repetition is one 2-D iDMA transfer; all of them are issued before one wait,
// so they overlap. The shapes it exists for are not one contiguous run:
//   a gather     the q_pe of several heads (a head apart) into one RoPE operand
//   an append    a cache row into an A layout: 4-byte runs 64 B apart (the key copy), or
//                1-byte runs 4 B apart, 32 times over (the value copy)
//   a tile       Bc tokens of a [512, cap] A-layout operand: 32 runs of 16 Bc bytes
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_idma_2d_copy(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_idma_2d_copy_args_t);
    if (!snrt_is_dm_core()) {
        printf_safe("[Cluster %d Core %d]: Error! IDMA 2D copy should be called from a DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_idma_2d_copy_args_t *a =
        (const __snax_bingo_kernel_idma_2d_copy_args_t *)arg;
    const uint64_t src = make_u64(a->src_addr_hi, a->src_addr_lo);
    const uint64_t dst = make_u64(a->dst_addr_hi, a->dst_addr_lo);
    const uint32_t outer = a->outer ? a->outer : 1u;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_idma_2d_copy_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    if (a->size == 0u || a->reps == 0u) {
        printf_safe("[Cluster %d Core %d]: Error! IDMA 2D copy: size=%d reps=%d must be "
                    "non-zero\r\n", snrt_cluster_idx(), snrt_cluster_core_idx(),
                    (int)a->size, (int)a->reps);
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
    for (uint32_t o = 0; o < outer; o++)
        snrt_dma_start_2d_wideptr(dst + (uint64_t)o * a->dst_outer,
                                  src + (uint64_t)o * a->src_outer, a->size,
                                  a->dst_stride, a->src_stride, a->reps);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
    snrt_dma_wait_all();
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
    sp->return_value = (uint32_t)dst;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// weight_ring.trailer's 64-bit magic (libs/crest.py TRAILER_MAGIC)
#ifndef BINGO_RING_TRAILER_MAGIC
#define BINGO_RING_TRAILER_MAGIC 0x5EED7A11C0DEF1A6ULL
#endif

// The expert-slot record (moe_route.h) one copy reads its source from: slot `slot`, word
// pair 16 + 2 field. The address was decided by the router at run time, so it is looked up
// here, on the DM core, the one engine that reaches every memory.
// One chunk of a weight RING in this chiplet's L3 into an L1 slab. The memory chiplet's
// iDMA PUSHES the weights into the ring (the host's prefetcher schedules it: over the
// half-duplex D2D link a push flows one way, a pull turns the link around per request),
// and after each chunk pushes a 64-B flag holding the chunk's sequence number. Same
// iDMA queue, one AXI ID, one link: the flag lands after the chunk. So: wait for the flag,
// copy the slot (a local L3 read), then hand the slot back by writing the same number
// into its release word, which the prefetcher polls before it reuses the slot.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_idma_ring_load(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_idma_ring_load_args_t);
    if (!snrt_is_dm_core()) {
        printf_safe("[Cluster %d Core %d]: Error! IDMA ring_load should be called from a DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_idma_ring_load_args_t *a =
        (const __snax_bingo_kernel_idma_ring_load_args_t *)arg;
    volatile uint32_t *flag = (volatile uint32_t *)a->flag_addr;
    volatile uint32_t *release = (volatile uint32_t *)a->release_addr;
    const uint64_t src = make_u64(a->src_addr_hi, a->src_addr_lo);
    const uint64_t dst = make_u64(a->dst_addr_hi, a->dst_addr_lo);
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_idma_ring_load_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
    if (a->trailer) {
        // weight_ring.trailer: the record's own last beat says it has landed (the push writes
        // it last, one transfer, writes in order); the slots are zero at boot (.bss)
        while (flag[0] != (uint32_t)BINGO_RING_TRAILER_MAGIC ||
               flag[1] != (uint32_t)(BINGO_RING_TRAILER_MAGIC >> 32)) {
        }
    } else {
        while (*flag != a->seq) {
        }
    }
    snrt_dma_start_1d_wideptr(dst, src, a->size);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
    snrt_dma_wait_all();
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
    if (a->trailer) {
        // cleared before the slot is freed: a later record ending here must not look landed
        flag[0] = 0u;
        flag[1] = 0u;
    }
    *release = a->seq;
    sp->return_value = (uint32_t)dst;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

#define BINGO_MOE_REC_SLOT_WORDS 32u   // 128 B per slot
#define BINGO_MOE_REC_ENTRY_WORD 16u   // the copied 64-B table entry starts here

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_idma_copy_slot(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_idma_copy_slot_args_t);
    if (!snrt_is_dm_core()) {
        printf_safe("[Cluster %d Core %d]: Error! IDMA copy_slot should be called from a DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_idma_copy_slot_args_t *a =
        (const __snax_bingo_kernel_idma_copy_slot_args_t *)arg;
    const volatile uint32_t *w = (const volatile uint32_t *)a->record_addr +
                                 a->slot * BINGO_MOE_REC_SLOT_WORDS +
                                 BINGO_MOE_REC_ENTRY_WORD + 2u * a->field;
    const uint64_t base = make_u64(w[1], w[0]);
    const uint64_t dst = make_u64(a->dst_addr_hi, a->dst_addr_lo);
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_idma_copy_slot_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    if (a->field > 3u || base == 0u) {
        printf_safe("[Cluster %d Core %d]: Error! IDMA copy_slot: slot %d field %d has no "
                    "source (expert %d not staged?)\r\n", snrt_cluster_idx(),
                    snrt_cluster_core_idx(), (int)a->slot, (int)a->field,
                    (int)((const volatile uint32_t *)a->record_addr)
                        [a->slot * BINGO_MOE_REC_SLOT_WORDS]);
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
    snrt_dma_start_1d_wideptr(dst, base + a->offset, a->size);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
    snrt_dma_wait_all();
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
    sp->return_value = (uint32_t)dst;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}
