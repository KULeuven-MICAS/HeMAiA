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
// The STREAMER is programmed through the ordinary set_versacore_streamer_csr(): what differs
// from gemm_full is the descriptors handed to it, not the way they are written. Its
// `csrw_ss(BASE + i, ...)` loops all have compile-time bounds, so the compiler unrolls them
// and folds every address into an immediate -- the whole call is 47 `csrw` and no indirect
// jump 
//
// What this file does NOT reuse is the library's wait: it polls busy immediately after START
// and then waits for the fall unbounded. Both are wrong here, for the reasons given at the
// launch site below.

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

// Which of the two matmuls this dispatch is, in the trace.
//
// A marker's immediate has to be a compile-time constant -- it rides in an `xori x0, x0,
// imm` -- so the identity cannot be a run-time field. Both arms below are literals and the
// branch is on a parameter both call sites pass as a constant, so it folds away; what
// reaches the trace is one id that says WHICH matmul ran.
#define BINGO_FA_MARK(is_qk, id_qk, id_pv)      \
    do {                                        \
        if (is_qk) BINGO_TRACE_MARKER(id_qk);   \
        else       BINGO_TRACE_MARKER(id_pv);   \
    } while (0)

// Bound on the busy poll. One dispatch here is ~2048 array cycles at Bc = 512; a limit
// three orders of magnitude above that only fires when the engine is genuinely wedged.
#define BINGO_GEMM_FA_SPIN_LIMIT 200000u

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

    BINGO_FA_MARK(emit_fp16, BINGO_TRACE_GEMM_FA_QK_CFG_START,
                              BINGO_TRACE_GEMM_FA_PV_CFG_START);

    const uint32_t a_tile = BINGO_A_ELEM_LEN * tileSize * meshRow / 8u;
    const uint32_t b_tile = BINGO_B_ELEM_LEN * tileSize * meshCol / 8u;

    // The C/D channels are a spatial NEST of BINGO_CD_SPATIAL_NUM dimensions, innermost
    // bound BINGO_CD_SPATIAL_BOUND0: channel i sits at sl0*(i%B0) + sl1*((i/B0)%B1). Every
    // number here comes from the generated gemm_shapes.h, so a reshape of the port needs no
    // edit -- only a regenerated header. The innermost group carries one key's meshCol
    // scores and the groups step by a WHOLE key row of Br = N*meshCol FP16, which is what
    // interleaves the two N blocks in memory at no cost. A beat is then exactly one key,
    // all Br queries.
    const uint32_t key_row = N * meshCol * BINGO_FA_XPORT_BITS / 8u;   // Br FP16 = 64 B
    const uint32_t n_step  = meshCol * BINGO_FA_XPORT_BITS / 8u;       // other N block
    const uint32_t c_chunk = serial * N / 8u;                          // serialised chunk
    const uint32_t blk_i32 = BINGO_C_ELEM_LEN * meshRow * meshCol / 8u;

    // Arming the converter HALVES the beats the writer emits -- two INT32 beats merge into
    // one FP16 beat -- so the D descriptor must halve with it, or the writer waits for beats
    // that never arrive. Which parts halve is not uniform:
    //
    //   bound0   halves   a transaction still spans one serialised chunk, but covers twice
    //                     the key rows
    //   stride2  halves   an M block is meshRow key rows, and a key row is half the bytes
    //   stride1  does NOT halve -- it is the interleave offset, laid out on the FP16 row, so
    //                     it is the same in both modes. Halving it would drop the second N
    //                     block on top of the first.
    const uint32_t d_bound0 = BINGO_D32_ELEM_LEN * meshRow * meshCol / serial;
    const uint32_t d_stride2 = N * BINGO_D32_ELEM_LEN * meshRow * meshCol / 8u;

    // A's reader declares six temporal dimensions; the descriptor uses three and the rest
    // must be neutralised. A bound left at whatever the previous task wrote walks the AGU
    // over memory that is not the operand.
    uint32_t Asl[S_STRIDE_NUM_READER_0] = { bw / 8u };
    uint32_t Atb[T_BOUND_NUM_READER_0]  = { K, N, M, 1u, 1u, 1u };
    uint32_t Ats[T_STRIDE_NUM_READER_0] = { a_tile, 0u, K * a_tile, 0u, 0u, 0u };
    uint32_t Bsl[S_STRIDE_NUM_READER_1] = { bw / 8u };
    uint32_t Btb[T_BOUND_NUM_READER_1]  = { K, N, M };
    uint32_t Bts[T_STRIDE_NUM_READER_1] = { b_tile, K * b_tile, 0u };
    uint32_t Csl[S_STRIDE_NUM_READER_WRITER_0] = { bw / 8u, key_row };
    uint32_t Ctb[T_BOUND_NUM_READER_WRITER_0]  = {
        BINGO_C_ELEM_LEN * meshRow * meshCol / serial, N, M };
    uint32_t Cts[T_STRIDE_NUM_READER_WRITER_0] = { c_chunk, n_step, N * blk_i32 };
    uint32_t Dsl[S_STRIDE_NUM_READER_WRITER_1] = { bw / 8u, key_row };
    uint32_t Dtb[T_BOUND_NUM_READER_WRITER_1]  = {
        emit_fp16 ? d_bound0 / 2u : d_bound0, N, M };
    uint32_t Dts[T_STRIDE_NUM_READER_WRITER_1] = {
        c_chunk, n_step, emit_fp16 ? d_stride2 / 2u : d_stride2 };
    // The D write host has no channel mask of its own -- the reader_writer declares
    // configurable_channel [1, 0], so only the READ slot has one and the library's write of
    // this array is compiled out.
    uint32_t chD[1] = { 0xffffffffu };

    // C is read in FULL, never broadcast: the port is serialised (meshRow*meshCol*
    // BINGO_C_ELEM_LEN / BINGO_SERIAL_C_D_WIDTH beats per block) and a broadcast operand
    // cannot be spread across a serialised input.
    //
    // ...EXCEPT for the score matmul, whose C is a bias of zeros. That one masks every C
    // channel off instead of streaming the zeros. A disabled channel is not skipped: the
    // requestor still pops the address but suppresses tcdmReq.valid, and the responser
    // substitutes a zero beat (snax readerWriter/DataRequestor.scala, DataResponser.scala),
    // so the array sees the same C = 0 a streamed buffer of zeros would have given it.
    //
    // What this buys is memory, not array time. A masked channel still pops its address and
    // still emits its beat, so the serialised C port costs the same BEATS either way and the
    // score matmul's own cycles do not move. What goes away is the 64 KiB buffer of zeros,
    // the iDMA load that would otherwise stage it at the head of the task graph, and the
    // TCDM bandwidth those reads took from everything else sharing the port.
    //
    // Quantisation has nowhere to go on a port carrying the converter alone, so it is off.
    set_versacore_streamer_csr(
        A_addr, Asl, Atb, Ats, 0, 0, (uint32_t *)bingo_gemm_shape_params[0].channel_en_A,
        B_addr, Bsl, Btb, Bts, 0, 0, (uint32_t *)bingo_gemm_shape_params[0].channel_en_B,
        C_addr, Csl, Ctb, Cts, 0,
        emit_fp16 ? (uint32_t *)bingo_channel_en_C_null
                  : (uint32_t *)bingo_gemm_shape_params[0].channel_en_C,
        D_addr, Dsl, Dtb, Dts, 0, chD,
        /*array_shape=*/0, /*quantization_enable=*/0,
        /*shift_i=*/0, /*multiplier_i=*/0, /*input_zp_i=*/0, /*output_zp_i=*/0,
        /*int32tofp16_enable=*/(int32_t)emit_fp16, /*int4_a=*/0, /*int4_b=*/0);

    // ---- the accelerator ---------------------------------------------------------------
    // take_in_new_c = 1: every output block starts from C, which is what makes the second
    // matmul's O += P.V accumulation free. ACCUM_BOUND is the contraction depth K;
    // OUTPUT_BOUND is the M*N output blocks one dispatch retires. One array shape and one
    // data type on this cluster, so both selector CSRs are 0.
    set_versacore_csr(1, K, M * N, 0, 0, 0);
    BINGO_FA_MARK(emit_fp16, BINGO_TRACE_GEMM_FA_QK_CFG_END,
                              BINGO_TRACE_GEMM_FA_PV_CFG_END);

    BINGO_FA_MARK(emit_fp16, BINGO_TRACE_GEMM_FA_QK_RUN_START,
                              BINGO_TRACE_GEMM_FA_PV_RUN_START);
    start_versacore_and_streamer();

    // CLEAR THE STREAMER'S START IMMEDIATELY, before anything else -- in particular before
    // the rise-wait below. An engine that re-triggers its descriptor walk under a held START
    // desynchronises from the array, which then waits for operands that never arrive and
    // sits BUSY for ever, and the ~130 cycles that 64 busy reads take is long enough for
    // that to happen.
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);

    // Wait for the engine to START before waiting for it to finish.
    //
    // Polling busy straight after the two START writes -- as the library's wait does -- is
    // safe only while those writes are SLOW enough for busy to rise first, which is a
    // property of how the caller was compiled, not of the hardware. With the writes folded
    // to single instructions it fails silently in the direction that matters: the wait
    // returns at once, the caller reads a partial performance counter and then reconfigures
    // the engine out from under a running matmul, which shows up as a dispatch reporting
    // FEWER cycles than its own arithmetic floor.
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
    // The accelerator's own START is cleared AFTER the wait, not before it: clearing it
    // while the array is mid-dispatch is not obviously safe.
    uint32_t spins = 0;
    while (csrr_ss(VERSACORE_BUSY) || csrr_ss(STREAMER_BUSY_CSR)) {
        if (++spins > BINGO_GEMM_FA_SPIN_LIMIT) {
            BINGO_FA_MARK(emit_fp16, BINGO_TRACE_GEMM_FA_QK_RUN_END,
                              BINGO_TRACE_GEMM_FA_PV_RUN_END);
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
    BINGO_FA_MARK(emit_fp16, BINGO_TRACE_GEMM_FA_QK_RUN_END,
                              BINGO_TRACE_GEMM_FA_PV_RUN_END);

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
                               a->output_D_addr, a->M, a->K, a->N, 1u, sp);
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
