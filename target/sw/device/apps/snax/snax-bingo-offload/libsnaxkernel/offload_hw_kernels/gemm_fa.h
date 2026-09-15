// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// The two FlashAttention matmuls, as core-level kernels on the GEMM hart.
//
// FlashAttention computes attention TRANSPOSED -- S^T = K.Q^T and O^T = V^T.P^T, which
// is the same arithmetic with the two operands swapped -- so that the score tile lands
// in memory as [Bc, Br]: one 64-B beat is one KEY carrying all Br query scores, one per
// lane. The softmax then reduces per query row by accumulating ALONG BEATS, which
// StreamReduce does for free in the accumulator it already carries, instead of folding
// across the lanes of a beat once per row. That layout is the whole reason the SIMD side
// is cheap, and it is produced HERE, by how the D port walks memory.
//
// WHY THESE ARE NOT __snax_bingo_kernel_gemm_full WITH DIFFERENT ARGUMENTS.
//
// The generic kernel derives a BLOCK-MAJOR D layout: tile (m, n) goes at
// (m*N + n) * tileBytes, each output block laid down whole before the next. That is the
// right default -- it is what xdma_d_to_row_major converts from -- and it is the wrong
// thing here. FlashAttention needs the two N blocks of one M block INTERLEAVED, 32 B
// apart, so that a beat does not straddle two keys. The difference is one descriptor
// field (the N-step stride: meshCol*2 bytes here, a whole tile there), and it is the
// difference between a LANEWISE rowmax that means something and one that maximises over
// a mixture of two keys and two queries.
//
// The second reason is the D write host. The generic path programs it through
// set_versacore_streamer_csr(), which writes a fixed SEVEN user CSRs -- the layout of a
// host carrying a rescale unit AND the FP16 converter. This cluster's host carries the
// converter alone, so its window is TWO registers wide
// (READER_WRITER_EXTENSION_1_CSR_NUM), and the five extra writes land on the streamer's
// own registers: at base+2 that is STREAMER_START_CSR, which launches the streamer in
// the middle of configuring it. The enable bitmask has one bit per extension in
// declaration order, so with the converter alone it is bit 0, not bit 1. These kernels
// therefore program the streamer directly, which is also what the reference
// snax-flashattn.c does, and for the same reason.
//
// COST. Every CSR address below is a compile-time constant, so each write folds through
// the always_inline csrw_ss switch into a single `csrw <imm>`: the whole configuration is
// ~60 cycles against a ~2048-cycle dispatch. Out of line, each access would instead pay a
// jump-table load from L2 plus an indirect jump -- measured, on this core, to be the
// dominant cost of accelerator configuration -- and that is why this file unrolls the
// descriptor writes rather than looping over arrays as the generic kernel does.

#pragma once

#include "../macros.h"
#include "snax_core_roles.h"  // snax_is_gemm_core(), BINGO_REQUIRE_CORE
#include <snax_versacore_lib.h>
#include <gemm_shapes.h>

#if BINGO_NUM_ARRAY_SHAPES != 1
#error "gemm_fa assumes the single-shape split cluster; index the shape table if that changes"
#endif

// The D write host's user-CSR window. Two registers means the Int32ToFp16Converter is
// alone on the port, which is the cluster these kernels are written for; anything else
// changes both the window and the bit position of the converter's enable.
#if defined(READER_WRITER_EXTENSION_1_CSR_BASE) && \
    READER_WRITER_EXTENSION_1_CSR_NUM != 2
#error "the D write host is not converter-only; re-derive the enable bitmask bit position"
#endif

// FP16 is the transport grid of the score tile, and the N-block interleave is laid out on
// it: the offset of the other N block inside a key row is meshCol FP16 elements whether
// the port is emitting FP16 or INT32, so this does NOT follow the output width.
#define BINGO_FA_XPORT_BITS 16u

// Bound on the busy poll. One dispatch here is ~2048 array cycles at Bc = 512; a limit
// three orders of magnitude above that only fires when the engine is genuinely wedged.
#define BINGO_GEMM_FA_SPIN_LIMIT 200000u

// DEBUG KNOB. Set to 0 to make the score matmul emit INT32 instead of FP16, i.e. to run
// the whole dispatch with the Int32ToFp16Converter DISARMED. The result is then wrong --
// the SIMD block reads the tile as FP16 -- but it isolates one question: whether the
// array stalls because the D writer's descriptor was halved for a converter that is not
// actually enabled, in which case the writer expects half the beats the array produces
// and backs up. Leave at 1.
#define BINGO_FA_QK_EMIT_FP16 1

// DEBUG KNOB. Set to 0 to drive the C/D ports with the BLOCK-MAJOR descriptors that
// __snax_bingo_kernel_gemm_full computes -- the layout that is known to complete a
// dispatch on this cluster -- instead of FlashAttention's interleaved one, and with the
// converter forced off so the whole output stage matches a working configuration.
//
// The result is meaningless (the softmax reads a layout that is not there), but it
// answers the one question the descriptor audit cannot: whether this kernel's MECHANICS
// are sound and the interleaved layout is what the streamer will not do, or whether the
// kernel differs from the working path in some other way. Leave at 1.
#define BINGO_FA_CD_INTERLEAVED 1

// ---------------------------------------------------------------------------
// One dispatch.
//
//   emit_fp16 = 1   S^T = K.Q^T. The result is consumed as FP16 by the SIMD block, so
//                   the converter is armed and the tile reaches TCDM in HALF the beats an
//                   INT32 tile would need. C is a zero bias.
//   emit_fp16 = 0   O^T += V^T.P^T. The result accumulates IN PLACE: C and D are the same
//                   buffer, so the matmul's own C input carries the running O across KV
//                   tiles and the accumulation costs nothing extra. INT32, converter off.
// ---------------------------------------------------------------------------
static uint32_t __bingo_gemm_fa_run(uint32_t A_addr, uint32_t B_addr, uint32_t C_addr,
                                    uint32_t D_addr, uint32_t M, uint32_t K, uint32_t N,
                                    uint32_t emit_fp16,
                                    bingo_kernel_scratchpad_t *sp) {
    const uint32_t meshRow  = bingo_gemm_shape_params[0].meshRow;
    const uint32_t tileSize = bingo_gemm_shape_params[0].tileSize;
    const uint32_t meshCol  = bingo_gemm_shape_params[0].meshCol;
    const uint32_t bw       = BINGO_BANK_WIDTH;
    const uint32_t serial   = BINGO_SERIAL_C_D_WIDTH;

    if (M == 0u || K == 0u || N == 0u) {
        printf_safe("[Cluster %d Core %d]: Error! gemm_fa bad shape M=%d K=%d N=%d\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), M, K, N);
        return BINGO_RET_FAIL;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);

    // ---- A (reader 0): the LEFT operand, K tiles of meshRow x tileSize INT8 -----------
    csrw_ss(BASE_PTR_READER_0_LOW, A_addr);
    csrw_ss(S_STRIDE_READER_0_0, bw / 8u);
    const uint32_t a_tile = BINGO_A_ELEM_LEN * tileSize * meshRow / 8u;
    csrw_ss(T_BOUND_READER_0_0, K);
    csrw_ss(T_STRIDE_READER_0_0, a_tile);
    csrw_ss(T_BOUND_READER_0_1, N);
    csrw_ss(T_STRIDE_READER_0_1, 0);
    csrw_ss(T_BOUND_READER_0_2, M);
    csrw_ss(T_STRIDE_READER_0_2, K * a_tile);
    // The reader declares six temporal dimensions; the descriptor uses three and the rest
    // must be neutralised. A bound left at whatever the previous task wrote walks the AGU
    // over memory that is not the operand.
    csrw_ss(T_BOUND_READER_0_3, 1);
    csrw_ss(T_STRIDE_READER_0_3, 0);
    csrw_ss(T_BOUND_READER_0_4, 1);
    csrw_ss(T_STRIDE_READER_0_4, 0);
    csrw_ss(T_BOUND_READER_0_5, 1);
    csrw_ss(T_STRIDE_READER_0_5, 0);
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, 0);
#endif
#ifdef ENABLED_CHANNEL_READER_0
    csrw_ss(ENABLED_CHANNEL_READER_0, bingo_gemm_shape_params[0].channel_en_A[0]);
#endif

    // ---- B (reader 1): the RIGHT operand, K tiles of tileSize x meshCol INT8 ----------
    csrw_ss(BASE_PTR_READER_1_LOW, B_addr);
    csrw_ss(S_STRIDE_READER_1_0, bw / 8u);
    const uint32_t b_tile = BINGO_B_ELEM_LEN * tileSize * meshCol / 8u;
    csrw_ss(T_BOUND_READER_1_0, K);
    csrw_ss(T_STRIDE_READER_1_0, b_tile);
    csrw_ss(T_BOUND_READER_1_1, N);
    csrw_ss(T_STRIDE_READER_1_1, K * b_tile);
    csrw_ss(T_BOUND_READER_1_2, M);
    csrw_ss(T_STRIDE_READER_1_2, 0);
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, 0);
#endif
#ifdef ENABLED_CHANNEL_READER_1
    csrw_ss(ENABLED_CHANNEL_READER_1, bingo_gemm_shape_params[0].channel_en_B[0]);
#endif

    // ---- C: the READ half of the one bidirectional port -------------------------------
    //
    // The channels are a spatial NEST of BINGO_CD_SPATIAL_NUM dimensions, innermost bound
    // BINGO_CD_SPATIAL_BOUND0: channel i sits at sl0*(i%B0) + sl1*((i/B0)%B1). Every number
    // below is derived from the generated gemm_shapes.h, so the narrowing of this port from
    // 32 channels x 2048 b to 16 x 1024 b needed no edit here -- only a regenerated header.
    // Four channels carry 32 B -- one key's meshCol scores -- and the eight groups step by
    // a WHOLE key row of Br = N*meshCol FP16, which is what interleaves the two N blocks
    // in memory at no cost. A beat is then exactly one key, all Br queries.
#if BINGO_FA_CD_INTERLEAVED
    const uint32_t key_row  = N * meshCol * BINGO_FA_XPORT_BITS / 8u;  // Br FP16 = 64 B
    const uint32_t n_step   = meshCol * BINGO_FA_XPORT_BITS / 8u;      // other N block
    const uint32_t c_chunk  = serial * N / 8u;                         // serialised chunk
#else
    // The block-major layout gemm_full derives: the channels lie end to end, the
    // chunks of one block are contiguous, and the N step is a WHOLE output block.
    const uint32_t key_row  = (bw / 8u) * BINGO_CD_SPATIAL_BOUND0;
    const uint32_t n_step   = BINGO_C_ELEM_LEN * meshRow * meshCol / 8u;
    const uint32_t c_chunk  = serial / 8u;
#endif
    const uint32_t blk_i32  = BINGO_C_ELEM_LEN * meshRow * meshCol / 8u;
    csrw_ss(BASE_PTR_READER_WRITER_0_LOW, C_addr);
    csrw_ss(S_STRIDE_READER_WRITER_0_0, bw / 8u);
    csrw_ss(S_STRIDE_READER_WRITER_0_1, key_row);
    csrw_ss(T_BOUND_READER_WRITER_0_0, BINGO_C_ELEM_LEN * meshRow * meshCol / serial);
    csrw_ss(T_STRIDE_READER_WRITER_0_0, c_chunk);
    csrw_ss(T_BOUND_READER_WRITER_0_1, N);
    csrw_ss(T_STRIDE_READER_WRITER_0_1, n_step);
    csrw_ss(T_BOUND_READER_WRITER_0_2, M);
    csrw_ss(T_STRIDE_READER_WRITER_0_2, N * blk_i32);
#ifdef ADDR_REMAP_INDEX_READER_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_0, 0);
#endif
    // C is read in FULL, never broadcast: the port is serialised (meshRow*meshCol*
    // BINGO_C_ELEM_LEN / BINGO_SERIAL_C_D_WIDTH beats per block) and a broadcast
    // operand cannot be spread across a serialised input.
    //
    // ...EXCEPT for the score matmul, whose C is a bias of zeros. That one masks every C
    // channel off instead of streaming the zeros. A disabled channel is not skipped: the
    // requestor still pops the address but suppresses tcdmReq.valid, and the responser
    // substitutes a zero beat (snax readerWriter/DataRequestor.scala, DataResponser.scala),
    // so the array sees exactly the same C = 0 it saw before.
    //
    // What this buys is not the 64 KiB of zeros, though it frees those too. The C READ and
    // the D WRITE are the two halves of ONE ReaderWriter unit sharing ONE set of 8 TCDM
    // ports, with the writer taking absolute priority (ReaderWriter.scala's `sel`). Every
    // C beat therefore costs a port cycle that a D beat could have used, and for the score
    // matmul all of them carried zeros.
#ifdef ENABLED_CHANNEL_READER_WRITER_0
    {
        const uint32_t *c_mask = emit_fp16 ? bingo_channel_en_C_null
                                           : bingo_gemm_shape_params[0].channel_en_C;
        for (uint32_t i = 0; i < ENABLED_CHANNEL_READER_WRITER_0_CSR_NUM; i++)
            csrw_ss(ENABLED_CHANNEL_READER_WRITER_0 + i, c_mask[i]);
    }
#endif

    // ---- D32: the WRITE half of that same port ----------------------------------------
    //
    // Arming the converter HALVES the beats the writer emits -- two INT32 beats merge into
    // one FP16 beat -- so the descriptor must halve with it, or the writer waits for beats
    // that never arrive. Which parts halve is not uniform:
    //
    //   bound0   halves   a transaction still spans one serialised chunk, but covers
    //                     twice the key rows
    //   stride2  halves   an M block is meshRow key rows, and a key row is half the bytes
    //   stride1  does NOT halve -- it is the interleave offset, laid out on the FP16 row,
    //                     so it is the same in both modes. Halving it would drop the
    //                     second N block on top of the first.
    csrw_ss(BASE_PTR_READER_WRITER_1_LOW, D_addr);
    csrw_ss(S_STRIDE_READER_WRITER_1_0, bw / 8u);
    csrw_ss(S_STRIDE_READER_WRITER_1_1, key_row);
    const uint32_t d_bound0 = BINGO_D32_ELEM_LEN * meshRow * meshCol / serial;
    const uint32_t d_stride2 = N * BINGO_D32_ELEM_LEN * meshRow * meshCol / 8u;
#if !BINGO_FA_CD_INTERLEAVED
    emit_fp16 = 0u;   // block-major debug mode: match gemm_full's INT32 output stage
#endif
    csrw_ss(T_BOUND_READER_WRITER_1_0, emit_fp16 ? d_bound0 / 2u : d_bound0);
    csrw_ss(T_STRIDE_READER_WRITER_1_0, c_chunk);
    csrw_ss(T_BOUND_READER_WRITER_1_1, N);
    csrw_ss(T_STRIDE_READER_WRITER_1_1, n_step);
    csrw_ss(T_BOUND_READER_WRITER_1_2, M);
    csrw_ss(T_STRIDE_READER_WRITER_1_2, emit_fp16 ? d_stride2 / 2u : d_stride2);
#ifdef ADDR_REMAP_INDEX_READER_WRITER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_1, 0);
#endif
    // The D write host has no channel mask: the reader_writer declares
    // configurable_channel [1, 0], so only the READ slot has one.

    // The converter, and ONLY the two registers this host owns. extra_loops index 0 is
    // the 2:1 policy, which is what an INT32 -> FP16 narrow is.
#ifdef READER_WRITER_EXTENSION_1_CSR_BASE
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE, emit_fp16 ? 1u : 0u);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 1, 0u);
#endif

    // ---- the accelerator ---------------------------------------------------------------
    // take_in_new_c = 1: every output block starts from C, which is what makes the second
    // matmul's O += P.V accumulation free. ACCUM_BOUND is the contraction depth K;
    // OUTPUT_BOUND is the M*N output blocks one dispatch retires. One array shape and one
    // data type on this cluster, so both selector CSRs are 0.
    set_versacore_csr(1, K, M * N, 0, 0, 0);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    start_versacore_and_streamer();

    // CLEAR THE STREAMER'S START IMMEDIATELY, before anything else -- in particular before
    // the rise-wait below. The reference does this in its first two instructions after
    // launching and it is not incidental: leaving START asserted for the ~130 cycles that
    // 64 busy reads take is long enough to matter, and an engine that re-triggers its
    // descriptor walk under a held START desynchronises from the array, which then waits
    // for operands that never arrive and sits BUSY for ever.
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);

    // Wait for the engine to START before waiting for it to finish.
    //
    // The library's wait polls busy immediately after the two START writes, which is safe
    // only while those writes are SLOW: out of line each goes through the csrw_ss jump
    // table and costs tens of cycles, just enough for busy to rise before the first poll.
    // That is a property of how the caller was compiled, not of the hardware, and it fails
    // silently in the direction that matters -- the wait returns at once, the caller reads
    // a partial performance counter and then reconfigures the engine out from under a
    // running matmul. It never shows up on short tasks; on a large tile it appears as a
    // dispatch reporting FEWER cycles than its own arithmetic floor.
    //
    // Bounded, so a task that completes before we look cannot hang us: if busy never
    // rises, either it already finished (the fall-waits below exit at once, which is
    // correct) or it was never started, which the drain timeout catches.
    for (uint32_t g = 0; g < 64u; g++)
        if (csrr_ss(VERSACORE_BUSY) || csrr_ss(STREAMER_BUSY_CSR)) break;

    // The fall-wait is bounded too. The library's is an unbounded `while (busy)`, which in
    // RTL simulation turns a wedged engine into a run that burns the whole wall-clock
    // budget with nothing to read afterwards; failing the node instead leaves a diagnosis.
    //
    // The accelerator's own START is cleared AFTER the wait, not before it -- again as the
    // reference does. Clearing it while the array is mid-dispatch is not obviously safe,
    // and there is no reason to find out.
    uint32_t spins = 0;
    while (csrr_ss(VERSACORE_BUSY) || csrr_ss(STREAMER_BUSY_CSR)) {
        if (++spins > BINGO_GEMM_FA_SPIN_LIMIT) {
            BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
            // The two busy bits say WHO is stuck; the two performance counters say
            // whether either engine ever ran at all. A streamer counter near zero means
            // the descriptors were never walked; a large one with the array still busy
            // means the operands were delivered and the array is waiting on something
            // else -- almost always the D writer refusing beats because its descriptor
            // is short of what the array will push.
            printf_safe("[Cluster %d Core %d]: Error! gemm_fa timed out (M=%d K=%d N=%d "
                        "fp16=%d): versacore_busy=%d streamer_busy=%d "
                        "versacore_cc=%d streamer_cc=%d\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx(), M, K, N, emit_fp16,
                        (int)csrr_ss(VERSACORE_BUSY), (int)csrr_ss(STREAMER_BUSY_CSR),
                        (int)csrr_ss(VERSACORE_PERFORMANCE_COUNTER),
                        (int)csrr_ss(STREAMER_PERFORMANCE_COUNTER_CSR));
            return BINGO_RET_FAIL;
        }
    }
    csrw_ss(VERSACORE_START_CSR, 0);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);

    sp->return_value = D_addr;
    sp->num_return_values = M * N;
    return BINGO_RET_SUCC;
}

// S^T = K.Q^T -- the score tile, emitted as FP16 for the SIMD block to consume.
// M*meshRow = Bc, N*meshCol = Br, K*tileSize = d.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_gemm_fa_qk(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_gemm_fa_args_t);
    BINGO_REQUIRE_CORE(snax_is_gemm_core(), "gemm_fa_qk", "GEMM");
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_gemm_fa_args_t *a =
        (const __snax_bingo_kernel_gemm_fa_args_t *)arg;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_gemm_fa_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    return __bingo_gemm_fa_run(a->input_A_addr, a->input_B_addr, a->input_C_addr,
                               a->output_D_addr, a->M, a->K, a->N,
                               BINGO_FA_QK_EMIT_FP16, sp);
}

// O^T += V^T.P^T -- accumulated in place in INT32. The caller passes the SAME buffer as C
// and D; that is the online accumulation across KV tiles, and it is the GEMM's own C input
// rather than anything extra.
// M*meshRow = d, N*meshCol = Br, K*tileSize = Bc.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_gemm_fa_pv(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_gemm_fa_args_t);
    BINGO_REQUIRE_CORE(snax_is_gemm_core(), "gemm_fa_pv", "GEMM");
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_gemm_fa_args_t *a =
        (const __snax_bingo_kernel_gemm_fa_args_t *)arg;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_gemm_fa_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    return __bingo_gemm_fa_run(a->input_A_addr, a->input_B_addr, a->input_C_addr,
                               a->output_D_addr, a->M, a->K, a->N, 0u, sp);
}
