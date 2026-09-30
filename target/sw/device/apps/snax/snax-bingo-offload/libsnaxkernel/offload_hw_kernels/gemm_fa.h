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
// and folds every address into an immediate, leaving no indirect jump.
//
// The library's wait is not reused. It polls busy immediately after START and then waits
// for the fall unbounded; this file waits by retired-task counter and bounds both spins.
// See the launch site below.

#pragma once

#include "../macros.h"
#include "snax_core_roles.h"  // snax_is_gemm_core(), BINGO_REQUIRE_CORE
#include <snax_versacore_lib.h>
#include <gemm_shapes.h>

// Both matmuls run SHAPE 0, the (16, 4, 16) GEMM shape: every descriptor below reads
// bingo_gemm_shape_params[0] and writes ARRAY_SHAPE_CFG = 0. A cluster may declare more
// shapes -- snax_split_cluster adds (1, 4, 32) for the one-token GEMV (gemv.h) -- and they
// do not concern these kernels. What shape 0 IS is checked at run time, where the table is.
#if BINGO_NUM_ARRAY_SHAPES < 1
#error "gemm_fa needs the (16, 4, 16) GEMM shape as shape 0 of gemm_shapes.h"
#endif

// The D write host's user-CSR window. The Int32ToFp16Converter is alone on the port on the
// cluster these kernels are written for, and its window is one enable word plus its own
// user CSRs:
//
//   2   enable, extra-loop policy                    the plain converter
//   3   enable, extra-loop policy, k (bits [3:0])    built with `shift: 1`: the port writes
//                                                    RNE(x * 2^-k), k clamped to 0..14
//
// Anything else stacks another extension on the port, which moves both the window and the
// bit position of the converter's enable.
//
// WHY THE SHIFT EXISTS. An INT8 x INT8 score over d = 128 reaches 127^2 * 128 = 2,064,512,
// 31x past FP16's 65,504. The converter does not clamp: past that edge it writes +-Inf, the
// row's max becomes Inf, S - m becomes Inf - Inf = NaN, and the whole query row of P, l and
// O is NaN. A power of two only moves the exponent, so RNE(S * 2^-k) keeps the same 11
// significant bits -- no precision is lost -- and the softmax takes the factor back in its
// temperature, a' = a * 2^k. Without it the only way to keep S finite is to shrink the INT8
// operands, which does give up precision.
#if defined(READER_WRITER_EXTENSION_1_CSR_BASE) && READER_WRITER_EXTENSION_1_CSR_NUM == 3
#define BINGO_GEMM_FA_HAS_DSHIFT 1
#elif defined(READER_WRITER_EXTENSION_1_CSR_BASE) && READER_WRITER_EXTENSION_1_CSR_NUM == 2
#define BINGO_GEMM_FA_HAS_DSHIFT 0
#elif defined(READER_WRITER_EXTENSION_1_CSR_BASE)
#error "the D write host is not converter-only; re-derive the enable bitmask bit position"
#else
#define BINGO_GEMM_FA_HAS_DSHIFT 0
#endif
// k is clamped to 14 by the RTL: 2^-14 is FP16's smallest normal, so no integer lands in
// the subnormals. A larger request is refused rather than silently clamped.
#define BINGO_GEMM_FA_DSHIFT_MAX 14u

// PV's two opt-in fixes (the flags word of __snax_bingo_kernel_gemm_fa_args_t).
//
//   B_KMAJOR    B = P^T's 64-byte blocks sit k-major, block (k, n) at (k*N + n)*pitch.
//               That is the order the softmax's INTERLEAVE quantiser writes them in -- four
//               key beats become the two query blocks of one k -- so P needs no copy.
//               The pitch is b_tile (64 B, dense) unless b_pitch says otherwise; see
//               "THE B PITCH" at the launch site.
//   C_COLSCALE  O^T's column q is query q, and the online softmax must scale it by
//               corr[q] before this tile's P.V is added. Int32ColumnScale on the C READ path
//               (READER_WRITER_EXTENSION_0) does exactly that as C streams into the array,
//               so the matmul computes O = corr (.) O + V^T.P^T with no extra pass.
//   D_FP16      the LAST tile of an online softmax. O leaves through the D port as FP16,
//               RNE(O * 2^-d_shift), into a D buffer of its own, while C is still read as
//               the INT32 accumulator -- and scaled, with C_COLSCALE. Without it the last
//               tile writes INT32 and a separate pass would have to narrow 64 KiB of O
//               (d = 512, Br = 32) before anything could normalise it. The score matmul's
//               FP16 path is the same D walk and the same converter; only C differs: QK
//               masks it, this reads it.
#define GEMM_FA_B_KMAJOR   1u
#define GEMM_FA_C_COLSCALE 2u
#define GEMM_FA_D_FP16     4u

// The scaler's window: one enable word, then N (its csr 0), then 32 FP16 factors packed two
// per CSR -- which is byte for byte a 64-byte FP16 beat, so the softmax's corr beat is copied
// word for word. Any other count is a different extension and a different map.
#if defined(READER_WRITER_EXTENSION_0_CSR_BASE) && READER_WRITER_EXTENSION_0_CSR_NUM == 18
#define BINGO_GEMM_FA_HAS_COLSCALE 1
#else
#define BINGO_GEMM_FA_HAS_COLSCALE 0
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
//                   INT32 tile would need. C is a zero bias. d_shift is the converter's
//                   power-of-two scale: the tile is RNE(S * 2^-d_shift).
//   emit_fp16 = 0   O^T += V^T.P^T. The result accumulates IN PLACE: C and D are the same
//                   buffer, so the matmul's own C input carries the running O across KV
//                   tiles and the accumulation costs nothing extra. INT32, converter off.
//
// b_pitch is the distance between two consecutive 64-byte B blocks in the order they are
// stored; 0 means dense (b_tile).
// ---------------------------------------------------------------------------
static uint32_t __bingo_gemm_fa_run(uint32_t A_addr, uint32_t B_addr, uint32_t C_addr,
                                    uint32_t D_addr, uint32_t M, uint32_t K, uint32_t N,
                                    uint32_t emit_fp16, uint32_t perf_addr,
                                    uint32_t flags, uint32_t corr_addr,
                                    uint32_t d_shift, uint32_t b_pitch,
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
    if (meshRow != 16u || tileSize != 4u || meshCol != 16u) {
        printf_safe("[Cluster %d Core %d]: Error! gemm_fa is written for the (16, 4, 16) "
                    "array; shape 0 of this cluster is (%d, %d, %d)\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), (int)meshRow,
                    (int)tileSize, (int)meshCol);
        return BINGO_RET_FAIL;
    }
    // The column scaler rides the C READ path, so it needs a real C (a NULL C masks every
    // channel and the array sees zeros), an INT32 accumulation (the QK score matmul never
    // takes it), no more column blocks than it holds factors for, and the extension itself.
    const uint32_t colscale = (flags & GEMM_FA_C_COLSCALE) != 0u;
    // The D side's width: FP16 for the score matmul and for a PV that ends the recurrence
    // (GEMM_FA_D_FP16), INT32 for a PV that accumulates in place. C's side is separate: the
    // score matmul masks it (its C is a zero bias), every PV reads it unless C_addr is 0.
    const uint32_t d_fp16 = emit_fp16 || (flags & GEMM_FA_D_FP16) != 0u;
    if (colscale && (emit_fp16 || C_addr == 0u || corr_addr == 0u || N > 2u ||
                     !BINGO_GEMM_FA_HAS_COLSCALE)) {
        printf_safe("[Cluster %d Core %d]: Error! gemm_fa C_COLSCALE unusable: fp16=%d "
                    "C=%x corr=%x N=%d built=%d\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), emit_fp16, C_addr,
                    corr_addr, N, BINGO_GEMM_FA_HAS_COLSCALE);
        return BINGO_RET_FAIL;
    }
    // The shift acts on the converter's output, so it means something on the score matmul
    // only. A request the build cannot honour is refused: the plain converter would write
    // +-Inf for every score past 65,504, and the softmax would turn those rows into NaN.
    if (d_shift && (!d_fp16 || !BINGO_GEMM_FA_HAS_DSHIFT ||
                    d_shift > BINGO_GEMM_FA_DSHIFT_MAX)) {
        printf_safe("[Cluster %d Core %d]: Error! gemm_fa d_shift=%d unusable: fp16=%d "
                    "built=%d max=%d\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), d_shift, d_fp16,
                    BINGO_GEMM_FA_HAS_DSHIFT, BINGO_GEMM_FA_DSHIFT_MAX);
        return BINGO_RET_FAIL;
    }
    // An FP16 PV writes a different buffer from the INT32 one it reads: the two walks
    // differ in bytes per block, so C == D would read half-overwritten accumulators.
    if ((flags & GEMM_FA_D_FP16) && (emit_fp16 || C_addr == D_addr)) {
        printf_safe("[Cluster %d Core %d]: Error! gemm_fa D_FP16 is PV's last tile: it "
                    "needs its own D buffer (C=%x D=%x)\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), C_addr, D_addr);
        return BINGO_RET_FAIL;
    }

    BINGO_FA_MARK(emit_fp16, BINGO_TRACE_GEMM_FA_QK_CFG_START,
                              BINGO_TRACE_GEMM_FA_PV_CFG_START);

    const uint32_t a_tile = BINGO_A_ELEM_LEN * tileSize * meshRow / 8u;
    const uint32_t b_tile = BINGO_B_ELEM_LEN * tileSize * meshCol / 8u;
    // THE B PITCH. PV's B stream walks P^T k-major, so with dense 64 B blocks it steps
    // N*64 = 128 B per pass while its A stream (V^T) steps 64 B. TCDM repeats its banks
    // every 256 B, four 64 B groups: B then moves round the groups twice as fast as A, the
    // distance between the two streams keeps turning, and every few passes they want the
    // same banks. Measured on the snax reference: PV at 1.44 cycles per pass. Placing the
    // blocks 160 B apart makes B step 320 B = one rotation + 64 B, A's rate, and once two
    // streams move at the same rate one collision separates them for good: 1.15 cycles per
    // pass. The softmax writes P8 at the same pitch (its p8_pitch); the two must agree.
    const uint32_t b_blk = b_pitch ? b_pitch : b_tile;
    if (b_blk < b_tile || (b_blk & 7u)) {
        printf_safe("[Cluster %d Core %d]: Error! gemm_fa b_pitch=%d: below one %d-byte "
                    "block or not bank-aligned\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), b_pitch, b_tile);
        return BINGO_RET_FAIL;
    }

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
    // B's block order. n-major (block (k, n) at (n*K + k)*b_tile) is how a B operand is
    // normally stored; k-major is how the INTERLEAVE quantiser writes P. The bounds walk
    // k innermost either way -- only where each block is found changes.
    const uint32_t kmaj = (flags & GEMM_FA_B_KMAJOR) != 0u;
    uint32_t Bts[T_STRIDE_NUM_READER_1] = { kmaj ? N * b_blk : b_blk,
                                            kmaj ? b_blk : K * b_blk, 0u };
    uint32_t Csl[S_STRIDE_NUM_READER_WRITER_0] = { bw / 8u, key_row };
    uint32_t Ctb[T_BOUND_NUM_READER_WRITER_0]  = {
        BINGO_C_ELEM_LEN * meshRow * meshCol / serial, N, M };
    uint32_t Cts[T_STRIDE_NUM_READER_WRITER_0] = { c_chunk, n_step, N * blk_i32 };
    uint32_t Dsl[S_STRIDE_NUM_READER_WRITER_1] = { bw / 8u, key_row };
    uint32_t Dtb[T_BOUND_NUM_READER_WRITER_1]  = {
        d_fp16 ? d_bound0 / 2u : d_bound0, N, M };
    uint32_t Dts[T_STRIDE_NUM_READER_WRITER_1] = {
        c_chunk, n_step, d_fp16 ? d_stride2 / 2u : d_stride2 };
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
    // score matmul's own cycles do not move. What goes away is the buffer of zeros itself,
    // the iDMA load that would stage it at the head of the task graph, and the TCDM
    // bandwidth those reads would take from everything else sharing the port.
    //
    // Quantisation has nowhere to go on a port carrying the converter alone, so it is off.
    set_versacore_streamer_csr(
        A_addr, Asl, Atb, Ats, 0, 0, (uint32_t *)bingo_gemm_shape_params[0].channel_en_A,
        B_addr, Bsl, Btb, Bts, 0, 0, (uint32_t *)bingo_gemm_shape_params[0].channel_en_B,
        // A NULL C_addr is the caller saying "this dispatch has no bias" -- the same
        // masking the score matmul gets, available to any dispatch whose C is known to be
        // zero. With every channel masked no request is ever issued, so the pointer is
        // never dereferenced; it is still pointed at something real rather than at 0.
        C_addr ? C_addr : D_addr, Csl, Ctb, Cts, 0,
        (emit_fp16 || C_addr == 0u)
            ? (uint32_t *)bingo_channel_en_C_null
            : (uint32_t *)bingo_gemm_shape_params[0].channel_en_C,
        D_addr, Dsl, Dtb, Dts, 0, chD,
        /*array_shape=*/0, /*quantization_enable=*/0,
        /*shift_i=*/0, /*multiplier_i=*/0, /*input_zp_i=*/0, /*output_zp_i=*/0,
        /*int32tofp16_enable=*/(int32_t)d_fp16, /*int4_a=*/0, /*int4_b=*/0);

#if BINGO_GEMM_FA_HAS_DSHIFT
    // The converter's k, written on EVERY dispatch, 0 on an INT32 PV. The register is
    // latched at START like the rest of the window and otherwise persists, so a k left
    // behind by a score matmul would scale the FP16 output of whatever arms the converter
    // next. (set_versacore_streamer_csr leaves it alone since snax 33c3bfd0, so this write
    // is the only one: k for QK and for an FP16 PV, 0 otherwise.)
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 2, d_fp16 ? d_shift : 0u);
#endif

#if BINGO_GEMM_FA_HAS_COLSCALE
    // ---- the C read path's column scaler ---------------------------------------------
    // Staged like every other streamer CSR and latched at START, so it applies to this
    // dispatch only once the enable is cleared again below. N is the output's column-block
    // count: the scaler walks C in the same (m, n, chunk) order as the descriptor above.
    if (colscale) {
        const volatile uint32_t *f = (const volatile uint32_t *)(uintptr_t)corr_addr;
        csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 1, N);
        csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 2, f[0]);
        csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 3, f[1]);
        csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 4, f[2]);
        csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 5, f[3]);
        csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 6, f[4]);
        csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 7, f[5]);
        csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 8, f[6]);
        csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 9, f[7]);
        csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 10, f[8]);
        csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 11, f[9]);
        csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 12, f[10]);
        csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 13, f[11]);
        csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 14, f[12]);
        csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 15, f[13]);
        csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 16, f[14]);
        csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 17, f[15]);
        csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 0, 1u);
    }
#endif

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
    // Name THIS dispatch BEFORE launching it. A BINGO node owns exactly one dispatch and
    // cannot carry a sequence number between invocations -- a `static` would live in L3 and
    // cost a fabric round trip to read. Sampling the free-running array counter first and
    // waiting for it to advance by one gives the same contract without the state.
    const uint32_t retired_before = csrr_ss(VERSACORE_FINISHED_TASK);
    start_versacore_and_streamer();

    // CLEAR THE STREAMER'S START IMMEDIATELY, before anything else. An engine that
    // re-triggers its descriptor walk under a held START desynchronises from the array,
    // which then waits for operands that never arrive and sits BUSY for ever. Any delay
    // here -- even a short poll loop -- is long enough for that to happen.
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
#if BINGO_GEMM_FA_HAS_COLSCALE
    // DISARM THE SCALER. The enable this dispatch needs was latched at START; left set, it
    // would scale the C of whatever configures this streamer next -- every other GEMM
    // kernel, none of which knows the extension exists.
    if (colscale) csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE + 0, 0u);
#endif

    // Wait for the dispatch by COUNTER, not by polling busy.
    //
    // Polling busy straight after the START writes is only safe while those writes are slow
    // enough for busy to rise first, which is a property of how the caller was compiled
    // rather than of the hardware. When it is not, the wait returns immediately and the
    // caller reads a partial performance counter and reconfigures the engine out from under
    // a running matmul.
    //
    // The ARRAY's retired-task counter is free-running and monotone, so "this dispatch is
    // done" is exactly "it has advanced past the value sampled above". Subtracting before
    // the compare keeps that correct across a counter wrap.
    //
    // WHICH COUNTER MATTERS. It has to be the array's, not the STREAMER's: the streamer
    // counts its data movers done and its writer drains BEFORE the array retires the
    // matmul, so the streamer counter returns early. VERSACORE_BUSY cannot substitute
    // either -- with anything queued behind it the busy flag does not fall between
    // back-to-back dispatches -- so busy is drained once, below, where the result is about
    // to be published, rather than once per dispatch.
    {
        uint32_t rs = 0;
        while ((int32_t)(csrr_ss(VERSACORE_FINISHED_TASK) - retired_before) < 1) {
            if (++rs > BINGO_GEMM_FA_SPIN_LIMIT) break;   // the fall-wait below diagnoses
        }
    }

    // The fall-wait is bounded. An unbounded `while (busy)` turns a wedged engine into a
    // run that burns the whole wall-clock budget with nothing to read afterwards; failing
    // the node instead leaves a diagnosis.
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
    // ---- the array's own account of this dispatch --------------------------------------
    // Read BEFORE the START clear and before any reconfiguration: all four counters reset
    // on config_fire, so a read after the next dispatch's CSR writes returns that
    // dispatch's partial state, not this one's total.
    //
    // This is the number the snax reference reports as `GEMM core busy %`, and reading it
    // is the only way to be comparable to it. A trace SPAN from RUN_START to RUN_END is
    // not the same quantity: it also contains the START writes and the retire poll, so it
    // reports the array as busier than it was, by more on a machine whose dispatch path is
    // slower -- which would flatter exactly the configurations this is meant to measure.
    //
    // perf_addr is L1 and this core is its only writer, so the read-modify-write needs no
    // lock. Cost is five RO CSR reads against a dispatch three orders of magnitude longer.
    if (perf_addr) {
        volatile uint32_t *pf = (volatile uint32_t *)(uintptr_t)perf_addr;
        pf[0] += csrr_ss(VERSACORE_PERFORMANCE_COUNTER);
        pf[1] += csrr_ss(VERSACORE_STALL_A);
        pf[2] += csrr_ss(VERSACORE_STALL_B);
        pf[3] += csrr_ss(VERSACORE_STALL_D);
        pf[4] += 1u;
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
// ---------------------------------------------------------------------------
// Report this cluster's accumulated array counters.
//
// WHY A SEPARATE NODE. The counters have to be read per dispatch (they reset on config
// write) but printed once. Printing from inside the matmul would put a UART write on the
// critical path of the thing being measured; a trailing node runs after the last matmul
// has retired, so the cost lands outside the window every utilisation figure is divided by.
//
// The array's own invariant is
//     stall_a + stall_b + stall_d + passes == performance_counter
// so `passes` is derived rather than measured, and a negative value would mean the counters
// were read across a config write. It is printed as a signed int for exactly that reason.
// ---------------------------------------------------------------------------
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_gemm_perf_report(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_gemm_perf_report_args_t);
    BINGO_REQUIRE_CORE(snax_is_gemm_core(), "gemm_perf_report", "GEMM");
    const __snax_bingo_kernel_gemm_perf_report_args_t *a =
        (const __snax_bingo_kernel_gemm_perf_report_args_t *)arg;
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_gemm_perf_report_args_t);
    volatile uint32_t *pf = (volatile uint32_t *)(uintptr_t)a->perf_addr;
    const uint32_t busy = pf[0], sa = pf[1], sb = pf[2], sd = pf[3], nd = pf[4];
    const int32_t passes = (int32_t)busy - (int32_t)sa - (int32_t)sb - (int32_t)sd;
    printf_safe("[Cluster %d] GEMM-ARRAY busy=%d passes=%d stall_a=%d stall_b=%d "
                "stall_d=%d dispatches=%d ideal=%d\r\n",
                snrt_cluster_idx(), (int)busy, (int)passes, (int)sa, (int)sb, (int)sd,
                (int)nd, (int)a->ideal_cc);
    pf[0] = 0; pf[1] = 0; pf[2] = 0; pf[3] = 0; pf[4] = 0;
    sp->return_value = 0;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_gemm_fa_qk(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_gemm_fa_args_t);
    BINGO_REQUIRE_CORE(snax_is_gemm_core(), "gemm_fa_qk", "GEMM");
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_gemm_fa_args_t *a =
        (const __snax_bingo_kernel_gemm_fa_args_t *)arg;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_gemm_fa_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    // QK's B is Q^T, dense and n-major: neither PV flag nor a B pitch describes it.
    if (a->flags || a->b_pitch) {
        printf_safe("[Cluster %d Core %d]: Error! gemm_fa_qk takes no flags or b_pitch "
                    "(got %d, %d)\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), a->flags, a->b_pitch);
        return BINGO_RET_FAIL;
    }
    return __bingo_gemm_fa_run(a->input_A_addr, a->input_B_addr, a->input_C_addr,
                               a->output_D_addr, a->M, a->K, a->N, 1u,
                               a->perf_addr, 0u, 0u, a->d_shift, 0u, sp);
}

// O^T += V^T.P^T -- accumulated in place in INT32. The caller passes the SAME buffer as C
// and D; that is the online accumulation across KV tiles, and it is the GEMM's own C input
// rather than anything extra. The last tile may instead leave as FP16 (GEMM_FA_D_FP16, with
// d_shift and a D of its own).
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
                               a->output_D_addr, a->M, a->K, a->N, 0u,
                               a->perf_addr, a->flags, a->corr_addr, a->d_shift,
                               a->b_pitch, sp);
}
