// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// The one-token GEMV on VersaCore's (1, 4, 32) array shape, as a core-level kernel on the
// GEMM hart. A port of snax_cluster's dsv2_gemv1_arm / dsv2_gemv_fire / dsv2_gemm_wait
// (sw/apps/dsv2/include/snax-dsv2.h), which passes bit-exact at DeepSeek-V2-Lite's shapes
// on one snax_split_cluster.
//
// WHAT ONE TASK COMPUTES. `groups` GEMVs of depth K = 4 kt and width N = 16 nb:
//
//     y_g (1 x N, FP16) = RNE(x_g . W_g * 2^-k)            g = 0 .. groups-1
//
//   A  x_g in ROW 0 of a 16-row A-layout operand (the (16, 4, 16) A layout: 16 x 4 blocks
//      of 64 B, block k at 64 k), or in a_row (block k's one read word at 8 k: a_blk 8),
//      group g at A + g * a_step. a_step = 0 when the groups
//      share one x (the chunks of one weight matrix), 16 K when each has its own (the
//      absorbed per-head GEMVs). Only channel 0 of the A reader is enabled: it reads rows
//      0 and 1 of each block and the array takes row 0. The other rows are never read.
//   B  W_g in the (16, 4, 16) B layout (4 x 16 blocks of 64 B, n-major: block (n, k) at
//      (n * kt + k) * 64), group g at B + g * K * N. The groups' blocks are consecutive.
//      With w4, INT4 weights instead (see INT4 WEIGHTS below): K * N / 2 bytes a group.
//   D  y_g at D + g * 2 N bytes: FP16, CONTIGUOUS -- a plain row, not the D layout.
//
//   args b_step and d_step are the groups' W and y strides (by default K N, K N / 2 at INT4,
//   and 2 N). With b_step 0 the groups are TOKENS sharing one W: a pass of several tokens
//   runs one group per token over the same chunk, each token's A its own a_row (a_step 2 K)
//   and its outputs a row of y (d_step, y's pitch). Every token still costs its own passes
//   -- the shape has one row -- and the chunk is read once per token.
//
// WHY NOT gemm_full WITH array_shape_idx = 1. gemm_full derives each shape's DENSE tile
// packing from the shape table: for (1, 4, 32) that is a 4 x 32 B block and an A pitch of
// one bank, i.e. operands laid out for that shape alone. The GEMV here keeps the GEMM
// shape's layouts -- the weights are packed ONCE, in the (16, 4, 16) B layout every other
// kernel reads -- and walks them differently:
//
//   * a pass reads TWO 16-column B blocks at the same k: channels 0..7 read block (2p, k),
//     channels 8..15 the next block, kt * 64 bytes on (B's second spatial stride). That is
//     128 weight bytes a pass, twice what the (16, 4, 16) shape takes, and the shape's
//     whole point: a GEMV does one multiply-add per weight byte, so bytes per pass IS its
//     rate.
//   * an odd nb runs HALF width, one block a pass: B's channels 8..15 masked (they read
//     zero, so the upper 16 columns compute 0) and D's upper four channels masked.
//   * the D port's converter runs in its single-beat mode (extra-loop index 3): one
//     output block is one beat, converted in place.
//   * C is masked on every channel: a disabled channel presents zero and issues no TCDM
//     request -- the fresh accumulator every GEMV wants, with no buffer of zeros.
//
// INT4 WEIGHTS (args w4; snax-dsv2.h, WEIGHT WIDTH). B's converter (cfg:
// HasIntlowToInthighConverter) sign-extends the nibbles of a beat's LOW half, nibble i into
// byte i, so the array runs the same passes on exact INT8 values. The two column blocks of a
// pass must then sit together in the low half: W_g is in the PAIRED layout, nibble-packed
// (snax util/layout.py to_b_pairs + pack_int4) -- per pair of blocks, per k, block 2p's 32
// bytes then block 2p + 1's -- one 64-byte run a pass on channels 0..7, the pairs kt * 64
// bytes apart. nb must be even. Half the bytes to load and to read, the same passes. The
// converter's enable is written on EVERY dispatch, so an INT8 task never inherits an INT4
// one's (gemm_full and the FlashAttention matmuls clear it through
// set_versacore_streamer_csr's int4_b_enable).
//
// THE D-PORT SHIFT. The converter writes RNE(acc * 2^-k). k is the smallest value with
// 127^2 * K <= 65,504 * 2^k, or the FP16 output overflows to +-Inf without any error:
// 9 at K = 2,048 and 1,408, 10 at 2,816, 5 at 128, 7 at 512, 8 at 576. k is sticky
// (latched at START), so this kernel writes it on every dispatch.
//
// ONE TASK PER CHUNK. The descriptor is re-armed on every dispatch -- about fifty CSR
// writes against a chunk of kt array passes -- because another kernel (FlashAttention's
// matmuls, gemm_full) may have run on this hart in between and a BINGO node carries no
// state from the last one.

#pragma once

#include "../macros.h"
#include "snax_core_roles.h"  // snax_is_gemm_core(), BINGO_REQUIRE_CORE
#include <snax_versacore_lib.h>
#include <gemm_shapes.h>

// Every CSR the descriptor below writes, and the converter window that has the single-beat
// mode and the shift. The library is shared by every cluster cfg, so a cluster without them
// still BUILDS this file: the body compiles out and the node fails loudly at run time.
#if defined(S_STRIDE_READER_1_1) && defined(T_BOUND_READER_0_5) &&                        \
    defined(T_STRIDE_READER_0_5) && defined(T_BOUND_READER_1_2) &&                        \
    defined(S_STRIDE_READER_WRITER_0_1) && defined(T_BOUND_READER_WRITER_0_2) &&          \
    defined(S_STRIDE_READER_WRITER_1_1) && defined(T_BOUND_READER_WRITER_1_2) &&          \
    defined(ENABLED_CHANNEL_READER_0) && defined(ENABLED_CHANNEL_READER_1) &&             \
    defined(ENABLED_CHANNEL_READER_WRITER_0) && defined(ENABLED_CHANNEL_READER_WRITER_1) && \
    defined(ADDR_REMAP_INDEX_READER_0) && defined(ADDR_REMAP_INDEX_READER_1) &&           \
    defined(ADDR_REMAP_INDEX_READER_WRITER_0) &&                                          \
    defined(ADDR_REMAP_INDEX_READER_WRITER_1) && VERSACORE_HAS_D_SHIFT
#define BINGO_HAS_GEMV 1
#else
#define BINGO_HAS_GEMV 0
#endif

// B's INT4 converter: its host's one enable register (bit 0), when the cfg has it.
#ifdef READER_EXTENSION_1_CSR_BASE
#define BINGO_GEMV_HAS_W4 1
#else
#define BINGO_GEMV_HAS_W4 0
#endif

#define BINGO_GEMV_SHAPE 1u      // ARRAY_SHAPE_CFG of (1, 4, 32) on snax_split_cluster
#define BINGO_GEMV_BLK 64u       // one 16 x 4 A block or 4 x 16 B block, bytes
// Bytes of n weights stored at width w4 ? 4 : 8 (snax-dsv2.h DSV2_WB).
#define BINGO_GEMV_WB(n, w4) ((w4) ? (n) / 2u : (n))
#define BINGO_GEMV_DSHIFT_MAX 14u
#define BINGO_GEMV_SPIN_LIMIT 400000u

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_gemv(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_gemv_args_t);
    BINGO_REQUIRE_CORE(snax_is_gemm_core(), "gemv", "GEMM");
#if !BINGO_HAS_GEMV
    printf_safe("[Cluster %d Core %d]: Error! gemv needs VersaCore's (1, 4, 32) shape with a "
                "two-group B reader, channel masks and the D-port shift, which this RTL cfg "
                "does not generate. Kernel not executed.\r\n",
                snrt_cluster_idx(), snrt_cluster_core_idx());
    return BINGO_RET_FAIL;
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_gemv_args_t *a = (const __snax_bingo_kernel_gemv_args_t *)arg;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_gemv_args_t);
    const uint32_t A_addr = a->input_A_addr, B_addr = a->input_B_addr;
    const uint32_t D_addr = a->output_D_addr;
    const uint32_t kt = a->kt, nb = a->nb, groups = a->groups, a_step = a->a_step;
    const uint32_t k = a->d_shift, w4 = a->w4 ? 1u : 0u;
    const uint32_t b_step = a->b_step, d_step = a->d_step;
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMV_CFG_START);
    // The table says which shape index is (1, 4, 32); a cluster whose second shape is
    // something else would run this descriptor on the wrong array geometry.
#if BINGO_NUM_ARRAY_SHAPES <= BINGO_GEMV_SHAPE
    const bool shape_ok = false;
#else
    const bool shape_ok = bingo_gemm_shape_params[BINGO_GEMV_SHAPE].meshRow == 1u &&
                          bingo_gemm_shape_params[BINGO_GEMV_SHAPE].tileSize == 4u &&
                          bingo_gemm_shape_params[BINGO_GEMV_SHAPE].meshCol == 32u;
#endif
    if (!shape_ok) {
        printf_safe("[Cluster %d Core %d]: Error! gemv: array shape %d of this cluster is "
                    "not (1, 4, 32) (%d shapes in gemm_shapes.h).\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), (int)BINGO_GEMV_SHAPE,
                    (int)BINGO_NUM_ARRAY_SHAPES);
        return BINGO_RET_FAIL;
    }
    if (kt == 0u || nb == 0u || groups == 0u || k > BINGO_GEMV_DSHIFT_MAX) {
        printf_safe("[Cluster %d Core %d]: Error! gemv bad args kt=%d nb=%d groups=%d "
                    "k=%d (k 0..14)\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), (int)kt, (int)nb,
                    (int)groups, (int)k);
        return BINGO_RET_FAIL;
    }
    if (w4 && (!BINGO_GEMV_HAS_W4 || (nb & 1u))) {
        printf_safe("[Cluster %d Core %d]: Error! gemv w4: INT4 weights need B's INT4 "
                    "converter (has %d) and an even nb (%d).\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), (int)BINGO_GEMV_HAS_W4,
                    (int)nb);
        return BINGO_RET_FAIL;
    }

    const uint32_t blk = BINGO_GEMV_BLK;
    const uint32_t w = (nb & 1u) ? 1u : 2u;  // column blocks a pass: 2, or 1 at half width
    const uint32_t nbp = nb / w;              // output blocks per group

    // A -- row 0 of each block, channel 0: k inner, n broadcast, then the group.
    csrw_ss(ENABLED_CHANNEL_READER_0, 0x1u);
    csrw_ss(S_STRIDE_READER_0_0, 8);
    csrw_ss(T_BOUND_READER_0_0, kt);
    csrw_ss(T_STRIDE_READER_0_0, a->a_blk ? a->a_blk : blk);   // 64: A layout; 8: a_row
    csrw_ss(T_BOUND_READER_0_1, nbp);
    csrw_ss(T_STRIDE_READER_0_1, 0);
    csrw_ss(T_BOUND_READER_0_2, groups);
    csrw_ss(T_STRIDE_READER_0_2, a_step);
    csrw_ss(T_BOUND_READER_0_3, 1);
    csrw_ss(T_STRIDE_READER_0_3, 0);
    csrw_ss(T_BOUND_READER_0_4, 1);
    csrw_ss(T_STRIDE_READER_0_4, 0);
    csrw_ss(T_BOUND_READER_0_5, 1);
    csrw_ss(T_STRIDE_READER_0_5, 0);
    csrw_ss(ADDR_REMAP_INDEX_READER_0, 0);
    // B -- w column blocks at the same k a pass: channels 8..15 read the block kt * 64
    // bytes on, or are masked at half width. INT4: the pair is one 64-byte run on channels
    // 0..7, the next pass's 64 bytes on, and the next pair's kt * 64.
    csrw_ss(ENABLED_CHANNEL_READER_1, w4 || w == 1u ? 0xFFu : 0xFFFFu);
    csrw_ss(S_STRIDE_READER_1_0, 8);
    csrw_ss(S_STRIDE_READER_1_1, w4 ? 0u : kt * blk);
    csrw_ss(T_BOUND_READER_1_0, kt);
    csrw_ss(T_STRIDE_READER_1_0, blk);
    csrw_ss(T_BOUND_READER_1_1, nbp);
    csrw_ss(T_STRIDE_READER_1_1, BINGO_GEMV_WB(w * kt * blk, w4));
    csrw_ss(T_BOUND_READER_1_2, groups);
    csrw_ss(T_STRIDE_READER_1_2, b_step);
    csrw_ss(ADDR_REMAP_INDEX_READER_1, 0);
#if BINGO_GEMV_HAS_W4
    csrw_ss(READER_EXTENSION_1_CSR_BASE, w4);  // on or off, every dispatch
#endif
    // C -- masked, one beat per output block.
    csrw_ss(ENABLED_CHANNEL_READER_WRITER_0, 0);
    csrw_ss(S_STRIDE_READER_WRITER_0_0, 8);
    csrw_ss(S_STRIDE_READER_WRITER_0_1, 32);
    csrw_ss(T_BOUND_READER_WRITER_0_0, 1);
    csrw_ss(T_STRIDE_READER_WRITER_0_0, 0);
    csrw_ss(T_BOUND_READER_WRITER_0_1, nbp);
    csrw_ss(T_STRIDE_READER_WRITER_0_1, 32u * w);
    csrw_ss(T_BOUND_READER_WRITER_0_2, groups);
    csrw_ss(T_STRIDE_READER_WRITER_0_2, d_step);
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_0, 0);
    // D -- one beat per block, its 16 w FP16 values on channels 0 .. 4w - 1: 32 w bytes a
    // block, contiguous.
    csrw_ss(ENABLED_CHANNEL_READER_WRITER_1, w == 2u ? 0xFFu : 0xFu);
    csrw_ss(S_STRIDE_READER_WRITER_1_0, 8);
    csrw_ss(S_STRIDE_READER_WRITER_1_1, 32);
    csrw_ss(T_BOUND_READER_WRITER_1_0, nbp);
    csrw_ss(T_STRIDE_READER_WRITER_1_0, 32u * w);
    csrw_ss(T_BOUND_READER_WRITER_1_1, groups);
    csrw_ss(T_STRIDE_READER_WRITER_1_1, d_step);
    csrw_ss(T_BOUND_READER_WRITER_1_2, 1);
    csrw_ss(T_STRIDE_READER_WRITER_1_2, 0);
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_1, 0);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 0, 1u);  // Int32ToFp16 on
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 1, 3u);  // single-beat: a block is one beat
    (void)set_versacore_d_shift(k);                        // RNE(acc * 2^-k)
    // The array: output-stationary, kt passes per block, nbp blocks per group.
    csrw_ss(OVERWRITE_ACCUM, 1);
    csrw_ss(ACCUM_BOUND, kt);
    csrw_ss(OUTPUT_BOUND, nbp * groups);
    csrw_ss(SUBTRACTIONS, 0);
    csrw_ss(ARRAY_SHAPE_CFG, BINGO_GEMV_SHAPE);
    csrw_ss(DATA_TYPE_CFG, 0);
    // Operands and output; the D base doubles as C's (masked, never dereferenced).
    csrw_ss(BASE_PTR_READER_0_LOW, A_addr);
    csrw_ss(BASE_PTR_READER_1_LOW, B_addr);
    csrw_ss(BASE_PTR_READER_WRITER_0_LOW, D_addr);
    csrw_ss(BASE_PTR_READER_WRITER_1_LOW, D_addr);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMV_CFG_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMV_RUN_START);
    // Name this dispatch before launching it, then wait for the array's retired-task
    // counter to pass it (see gemm_fa.h: the streamer's own counter returns early, and busy
    // does not fall between queued dispatches). Subtracting first survives a wrap.
    const uint32_t retired_before = csrr_ss(VERSACORE_FINISHED_TASK);
    csrw_ss(STREAMER_START_CSR, 1);
    csrw_ss(VERSACORE_START_CSR, 1);
    // Clear the streamer's START at once: one held across the walk re-triggers it.
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    uint32_t spins = 0;
    while ((int32_t)(csrr_ss(VERSACORE_FINISHED_TASK) - retired_before) < 1) {
        if (++spins > BINGO_GEMV_SPIN_LIMIT) break;  // the fall-wait below diagnoses
    }
    // The D writer drains after the array retires; the consumer reads y from TCDM, so the
    // node is not done until the streamer is idle too.
    spins = 0;
    while (csrr_ss(VERSACORE_BUSY) || csrr_ss(STREAMER_BUSY_CSR)) {
        if (++spins > BINGO_GEMV_SPIN_LIMIT) {
            BINGO_TRACE_MARKER(BINGO_TRACE_GEMV_RUN_END);
            printf_safe("[Cluster %d Core %d]: Error! gemv timed out (kt=%d nb=%d groups=%d): "
                        "versacore_busy=%d streamer_busy=%d versacore_cc=%d "
                        "streamer_cc=%d\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx(), (int)kt, (int)nb,
                        (int)groups, (int)csrr_ss(VERSACORE_BUSY),
                        (int)csrr_ss(STREAMER_BUSY_CSR),
                        (int)csrr_ss(VERSACORE_PERFORMANCE_COUNTER),
                        (int)csrr_ss(STREAMER_PERFORMANCE_COUNTER_CSR));
            return BINGO_RET_FAIL;
        }
    }
    csrw_ss(VERSACORE_START_CSR, 0);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMV_RUN_END);
    sp->return_value = D_addr;
    sp->num_return_values = nbp * groups;
    return BINGO_RET_SUCC;
#endif
}
