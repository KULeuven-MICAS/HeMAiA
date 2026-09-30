// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// One-token SIMD kernels: the operators of a decode pass that work on ONE row, feeding the
// one-token GEMV (gemv.h). Ports of snax_cluster's dsv2_rmsnorm_row and dsv2_quant_a_row
// (sw/apps/dsv2/include/snax-dsv2.h), which are bit-exact against the DeepSeek-V2-Lite
// golden model (sw/apps/dsv2/util/simd.py) on one snax_split_cluster.
//
// WHY NOT simd_rmsnorm AND simd_fp16_to_int8. Those work a [rows, cols] tile: the row-major
// rmsnorm replicates each row's scalar into a plane for its multiply, and the quantiser
// writes a flat int8 tensor. One token needs neither. Its RMSNorm scalar is a whole-task
// scalar, so STICKY_B latches it and no plane is written; and the GEMV reads its activation
// from ROW 0 of a 16-row A operand, so the quantiser writes that row and nothing else -- a
// quarter of the input beats of a fully written operand, which is the SIMD's limit on this
// path.

#pragma once

#include "simd.h"

// ==========================================================================
// y = x / sqrt(mean(x^2)) over ONE row of n FP16 (n a power of two, a multiple of 32), no
// gain: the gain is folded into the weights that follow the norm. Three tasks, queued
// back to back and drained once:
//
//   reduce   SUMSQ over the row: FP32 accumulate, one splatted FP16 beat        -> ssq
//   map      RSQRT(ssq * 2^-log2(n)), the 1/rms beat                           -> seed
//   ew1      MUL | STICKY_B over [seed, x_0 .. x_last]: the seed is latched and
//            multiplies every later beat, so no plane of 1/rms is ever written -> y
//
// QUANTISED OUTPUT (inv_scale_f32bits != 0). Fp16ToInt8 sits right after EW1 in the
// operator chain, so the multiply's FP16 result goes on to sat127(rne(y * inv)) in the same
// pass and y is n INT8 -- a plain row, two input beats packed per output beat: the norm and
// the activation quantiser as one task, half the bytes out. Bit for bit hwmodel's
// quant_i8(rmsnorm(x), inv), since each operator narrows to FP16 as the golden does.
//
// The seed beat is x - 64: the multiply reads [seed | x] as one flat stream, so the caller
// allocates a beat of headroom directly below x. FP16 caps the sum of squares at 65,504,
// so rms <= sqrt(65504 / n) (5.66 at n = 2,048).
//
// PROGRAMMING, NOT THE SIMD, SET THIS KERNEL'S TIME (rmsnorm_4cluster, 2,048 values, 935 cc
// for 130 beats the SIMD streams in ~260): every pass rewrote all ~22 streamer CSRs at
// ~4.5 cc each, and log2(n) was a 12-step shift loop. All three passes are flat 1-D
// sweeps, so after the reduce is programmed in full (snax_simd_program_flat, inline, no
// shape structs) the map and the multiply rewrite only their addresses and beat counts,
// 4 CSRs each: SimdTop snapshots the whole configuration into its task queue on every
// start, so whatever is not rewritten stays as the pass before set it.
// ==========================================================================
// log2 of a power of two, as popcount(n - 1): no branches (a branch's cold target is an
// instruction-cache miss) and no __builtin_ctz (without Zbb a libgcc call, and this
// runtime is freestanding).
static inline __attribute__((always_inline)) uint32_t bingo_log2_pow2(uint32_t n) {
    uint32_t v = n - 1u;
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    v = (v + (v >> 4)) & 0x0F0F0F0Fu;
    v += v >> 8;
    v += v >> 16;
    return v & 0x3Fu;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_rmsnorm_row(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_rmsnorm_row_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_rmsnorm_row", "SIMD");
#if !(BINGO_HAS_STREAMREDUCE && BINGO_HAS_STREAMMAP && BINGO_HAS_STREAMELEMENTWISE && \
      BINGO_SIMD_HAS_RSQRT)
    BINGO_SIMD_EXT_UNSUPPORTED("simd_rmsnorm_row",
                               "StreamReduce, StreamMap RSQRT and post-map StreamElementwise");
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_simd_rmsnorm_row_args_t *a =
        (const __snax_bingo_kernel_simd_rmsnorm_row_args_t *)arg;
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_rmsnorm_row_args_t);
    const uint32_t x = a->input_addr, seed = a->seed_addr, ssq = a->ssq_addr;
    const uint32_t y = a->output_addr, n = a->cols;
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_START);
    BINGO_SIMD_REQUIRE_LOCAL(x, "simd_rmsnorm_row", "input");
    BINGO_SIMD_REQUIRE_LOCAL(ssq, "simd_rmsnorm_row", "ssq");
    BINGO_SIMD_REQUIRE_LOCAL(y, "simd_rmsnorm_row", "output");
    const uint32_t log2n = bingo_log2_pow2(n);
    if (n < 32u || (n & (n - 1u)) || seed + SIMD_BEAT_BYTES != x || (x & 63u) ||
        (ssq & 63u) || (y & 63u)) {
        printf_safe("[Cluster %d Core %d]: Error! simd_rmsnorm_row: cols=%d must be a power "
                    "of two >= 32, the seed beat must sit directly below x (seed=0x%08x "
                    "x=0x%08x) and every buffer 64-B aligned.\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), (int)n, seed, x);
        return BINGO_RET_FAIL;
    }
    const uint32_t beats = n / 32u;
    snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, beats, SIMD_RED_SUMSQ);
    snax_simd_program_flat(x, beats, ssq, 1u);
    snax_simd_fire();
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, 0x3F800000u - (log2n << 23),
                   0u, SIMD_FUNC_RSQRT);
    snax_simd_program_flat_next(ssq, 1u, seed, 1u);
    snax_simd_fire();
    const uint32_t inv = a->inv_scale_f32bits;
#if BINGO_HAS_FP16TOINT8
    if (inv) {
        snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMELEMENTWISE_1) |
                                                         (1u << SIMD_EXT_FP16TOINT8));
        snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 0, 1u);
        snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1, SIMD_EW_MUL | SIMD_EW_STICKY_B);
        snax_simd_set_op_csr(SIMD_EXT_FP16TOINT8_CSR, 0, inv);
        snax_simd_set_op_csr(SIMD_EXT_FP16TOINT8_CSR, 1, SIMD_QUANT_TAIL(0));
    } else
#else
    if (inv) {
        printf_safe("[Cluster %d Core %d]: Error! simd_rmsnorm_row: an INT8 output needs "
                    "Fp16ToInt8, which this cluster's SIMD does not have.\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
#endif
    {
        snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1u,
                       SIMD_EW_MUL | SIMD_EW_STICKY_B);
    }
    snax_simd_program_flat_next(seed, beats + 1u, y, inv ? beats / 2u : beats);
    snax_simd_fire();
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    if (snax_simd_wait_all_bounded() || snax_simd_bad_config()) {
        printf_safe("[Cluster %d Core %d]: Error! simd_rmsnorm_row did not drain cleanly "
                    "(status %08x)\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(),
                    snax_read_simd_cfg_reg(SIMD_STATUS));
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    sp->return_value = y;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

// ==========================================================================
// n FP16 -> INT8, into ROW `row` of a GEMV's 16-row A operand (16 n bytes; row 0 for the
// token, row 4 for a second one). Fp16ToInt8 alone: sat_127(rne(x * inv_scale)), the
// product in FP32, inv_scale the FP32 bit pattern.
//
// A-block k is 16 rows of four values at 64 k, so value i lands at a + (i/4)*64 + 4*row +
// i%4:
//   read   a beat of 16 values: channels 0, 2, 4, 6, four values each, 8 bytes apart (lane
//          stride 4, the odd channels disabled: they read nothing and present zero)
//   pack   two beats into one output beat, whose 8-byte words hold 4 values each in their
//          low half
//   write  word c to A-block c of the beat's eight (lane stride 64), low half only (byte
//          mask), then eight blocks on
// The other 15 rows are not written. The GEMV reads rows 0 and 1 of each block (one 8-B
// channel) and the array takes row 0, so row 1 has to hold something DEFINED: TCDM that
// was never written reads X on RTL. The caller zeroes the operand once.
//
// SEGMENTS (segs > 1): the input is segs runs of n values, seg_pitch bytes apart, read by
// the reader's third loop; the output is the row of all segs * n values, so segment g lands
// at a + 16 n g -- the A operand of its own GEMV. One task writes every head's operand of an
// absorbed per-head GEMV straight from the heads' slices of q (snax mla_quant_heads).
// ==========================================================================
#define BINGO_SIMD_A_ROW_READ 0x55u   // reader channels 0, 2, 4, 6
#define BINGO_SIMD_A_ROW_WORD 0x0Fu   // a written word's low four bytes: one A row

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_quant_a_row(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_quant_a_row_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_quant_a_row", "SIMD");
#if !BINGO_HAS_FP16TOINT8
    BINGO_SIMD_EXT_UNSUPPORTED("simd_quant_a_row", "Fp16ToInt8");
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_simd_quant_a_row_args_t *a =
        (const __snax_bingo_kernel_simd_quant_a_row_args_t *)arg;
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_quant_a_row_args_t);
    const uint32_t x = a->input_addr, out_a = a->output_addr, n = a->cols, row = a->row;
    const uint32_t inv_bits = a->inv_scale_f32bits;
    const uint32_t segs = a->segs ? a->segs : 1u, seg_pitch = a->seg_pitch;
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_START);
    BINGO_SIMD_REQUIRE_LOCAL(x, "simd_quant_a_row", "input");
    BINGO_SIMD_REQUIRE_LOCAL(out_a, "simd_quant_a_row", "A operand");
    if (n == 0u || (n % 32u) || row >= 16u || (x & 63u) || (out_a & 63u) ||
        (segs > 1u && (SIMD_MAX_DIM < 3 || (seg_pitch & 63u) || seg_pitch < 2u * n))) {
        printf_safe("[Cluster %d Core %d]: Error! simd_quant_a_row: cols=%d must be a "
                    "multiple of 32, row=%d < 16, both buffers 64-B aligned, and %d "
                    "segments %d B apart need a 3-D reader and a 64-B pitch past a "
                    "segment.\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), (int)n, (int)row,
                    (int)segs, (int)seg_pitch);
        return BINGO_RET_FAIL;
    }
    snax_simd_shape_t in, sh_out;
    snax_simd_shape_flat(&in, (void *)x, 1u);
    in.lane_stride = 4u;
    in.lane_mask = BINGO_SIMD_A_ROW_READ;
    in.bound[0] = 2u;       // the two beats of one output beat
    in.stride[0] = 32u;
    in.bound[1] = n / 32u;  // then the next 32 values
    in.stride[1] = 64u;
    in.dim = 2u;
#if SIMD_MAX_DIM >= 3
    if (segs > 1u) {        // then the next segment
        in.bound[2] = segs;
        in.stride[2] = seg_pitch;
        in.dim = 3u;
    }
#endif
    // a_blk 8: a_row, the blocks 8 B apart (row 0 only)
    const uint32_t blk = a->a_blk ? a->a_blk : 64u;
    snax_simd_shape_flat(&sh_out, (void *)(out_a + 4u * row), segs * n / 32u);
    sh_out.lane_stride = blk;
    sh_out.stride[0] = 8u * blk;
    sh_out.byte_mask = BINGO_SIMD_A_ROW_WORD;
    snax_simd_use0(SIMD_EXT_FP16TOINT8);
    snax_simd_set_op_csr(SIMD_EXT_FP16TOINT8_CSR, 0, inv_bits);
    snax_simd_set_op_csr(SIMD_EXT_FP16TOINT8_CSR, 1, 0u);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint32_t rc = simd_run_shapes(&in, &sh_out);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    if (rc != BINGO_RET_SUCC) return BINGO_RET_FAIL;
    sp->return_value = out_a;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

// ==========================================================================
// 16 FP16 rows `pitch` apart -> kt INT8 A blocks: block k (at a + 64 k) holds values
// [4k, 4k + 4) of every row, row r at 4 r. A port of snax mla_quant_q_segment.
//
//   read   a beat is 8 rows x 4 values: lane r at x + r * pitch (8 B, 4 FP16); two beats,
//          rows 0-7 then 8-15 (8 pitch on), make one block; then the next 4 values (8 B)
//   pack   Fp16ToInt8 folds the two beats into one 64-B block
//
// MLA's query operand Q8 [32, 576] is two such segments of its first m-block -- q~ (512
// values, the absorbed query) and the rotated q_pe (64) -- each with its own scale.
// ==========================================================================
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_quant_a_rows(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_quant_a_rows_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_quant_a_rows", "SIMD");
#if !BINGO_HAS_FP16TOINT8
    BINGO_SIMD_EXT_UNSUPPORTED("simd_quant_a_rows", "Fp16ToInt8");
#else
    const __snax_bingo_kernel_simd_quant_a_rows_args_t *a =
        (const __snax_bingo_kernel_simd_quant_a_rows_args_t *)arg;
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_quant_a_rows_args_t);
    const uint32_t x = a->input_addr, pitch = a->pitch, q = a->output_addr, kt = a->kt;
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_START);
    BINGO_SIMD_REQUIRE_LOCAL(x, "simd_quant_a_rows", "input");
    BINGO_SIMD_REQUIRE_LOCAL(q, "simd_quant_a_rows", "A operand");
    if (kt == 0u || (pitch & 7u) || pitch < 8u * kt || (x & 7u) || (q & 63u)) {
        printf_safe("[Cluster %d Core %d]: Error! simd_quant_a_rows: kt=%d, rows %d B apart "
                    "(8-B aligned, at least kt 8-B runs), the output 64-B aligned.\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), (int)kt, (int)pitch);
        return BINGO_RET_FAIL;
    }
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, (void *)x, 1u);
    in.lane_stride = pitch;
    in.bound[0] = 2u;          // rows 0-7, then 8-15
    in.stride[0] = 8u * pitch;
    in.bound[1] = kt;          // then the next 4 values
    in.stride[1] = 8u;
    in.dim = 2u;
    snax_simd_shape_flat(&out, (void *)q, kt);
    snax_simd_use0(SIMD_EXT_FP16TOINT8);
    snax_simd_set_op_csr(SIMD_EXT_FP16TOINT8_CSR, 0, a->inv_scale_f32bits);
    snax_simd_set_op_csr(SIMD_EXT_FP16TOINT8_CSR, 1, 0u);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint32_t rc = simd_run_shapes(&in, &out);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    if (rc != BINGO_RET_SUCC) return BINGO_RET_FAIL;
    sp->return_value = q;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

// ==========================================================================
// The end of MLA's online softmax, per query lane: o~ = O16 (.) c / l. Two tasks (snax
// dsv2_mla_simd, "10 O / l"):
//
//   1  l read twice (stride 0) -> Map RSQRT(a_n * l) on each -> EW1 MUL of the pair
//      -> c / l, into the latch beat directly below O16. sqrt(c / l) squared, not 1 / l:
//      l^2 overflows FP16 long before l does.
//   2  EW1 MUL | STICKY_B over [latch][O16]: every beat (a latent, 32 query lanes) times
//      its lane's factor -> o~
//
// Bit-exact against hwmodel.run_tokens: t16 = map(l, a_n, RSQRT); rsc = t16 * t16;
// ot = O16 * rsc.
// ==========================================================================
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_mla_normalise(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_mla_normalise_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_mla_normalise", "SIMD");
#if !(BINGO_HAS_STREAMMAP && BINGO_HAS_STREAMELEMENTWISE && BINGO_SIMD_HAS_RSQRT)
    BINGO_SIMD_EXT_UNSUPPORTED("simd_mla_normalise",
                               "StreamMap RSQRT and post-map StreamElementwise");
#else
    const __snax_bingo_kernel_simd_mla_normalise_args_t *a =
        (const __snax_bingo_kernel_simd_mla_normalise_args_t *)arg;
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_mla_normalise_args_t);
    const uint32_t l = a->l_addr, latch = a->latch_addr, o = a->input_addr;
    const uint32_t y = a->output_addr, beats = a->beats;
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_START);
    BINGO_SIMD_REQUIRE_LOCAL(l, "simd_mla_normalise", "l");
    BINGO_SIMD_REQUIRE_LOCAL(o, "simd_mla_normalise", "O16");
    BINGO_SIMD_REQUIRE_LOCAL(y, "simd_mla_normalise", "output");
    if (beats == 0u || latch + SIMD_BEAT_BYTES != o || (l & 63u) || (o & 63u) ||
        (y & 63u)) {
        printf_safe("[Cluster %d Core %d]: Error! simd_mla_normalise: the latch beat must sit "
                    "directly below O16 (latch=0x%08x O16=0x%08x), every buffer 64-B "
                    "aligned, beats=%d > 0.\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), latch, o, (int)beats);
        return BINGO_RET_FAIL;
    }
    snax_simd_shape_t in, out;
    snax_simd_shape_broadcast(&in, (void *)l, 2u);
    snax_simd_shape_flat(&out, (void *)latch, 1u);
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMMAP) |
                                                     (1u << SIMD_EXT_STREAMELEMENTWISE_1));
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 0, a->a_n_f32bits);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 1, 0u);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 2, SIMD_FUNC_RSQRT);
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 0, 2u);
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1, SIMD_EW_MUL);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    snax_simd_shape_flat(&in, (void *)latch, 1u + beats);
    snax_simd_shape_flat(&out, (void *)y, beats);
    snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1u,
                   SIMD_EW_MUL | SIMD_EW_STICKY_B);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    const uint32_t bad = snax_simd_wait_all_bounded() || snax_simd_bad_config();
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, 0u);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    if (bad) {
        printf_safe("[Cluster %d Core %d]: Error! simd_mla_normalise did not drain cleanly "
                    "(status %08x)\r\n", snrt_cluster_idx(), snrt_cluster_core_idx(),
                    snax_read_simd_cfg_reg(SIMD_STATUS));
        return BINGO_RET_FAIL;
    }
    sp->return_value = y;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

// ==========================================================================
// Softmax over ONE row (snax dsv2_softmax_row, the router's): the row max and sum fold
// across lanes, so each comes out splatted over a beat. Five tasks, queued back to back:
//
//   1  Reduce MAX over x                                   -> m           (tmp)
//   2  Map LINEAR a = -1                                   -> -m          (x's latch)
//   3  EW0 ADD|STICKY_B [-m][x] -> Map EXP -> Reduce ADD|TAP -> [e][s]    (e)
//   4  [s][s] at stride 0: EW0 MUL|STICKY_B -> Map RSQRT   -> 1/s         (e's latch)
//   5  EW1 MUL|STICKY_B [1/s][e]                           -> p
//
// 1/s = rsqrt(s * s) is exact for a positive s whose square fits FP16 (s <= 64: the router's
// 64 probabilities, each at most 1 after the max is subtracted).
// ==========================================================================
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_softmax_row(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_softmax_row_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_softmax_row", "SIMD");
#if !(BINGO_HAS_PREMAP_ELEMENTWISE && BINGO_HAS_STREAMMAP && BINGO_HAS_STREAMREDUCE && \
      BINGO_HAS_STREAMELEMENTWISE && BINGO_SIMD_HAS_RSQRT && BINGO_SIMD_EW0_HAS_MUL)
    BINGO_SIMD_EXT_UNSUPPORTED("simd_softmax_row",
                               "StreamElementwise_0 (MUL, ADD), StreamMap (EXP, RSQRT), "
                               "StreamReduce and StreamElementwise_1");
#else
    const __snax_bingo_kernel_simd_softmax_row_args_t *a =
        (const __snax_bingo_kernel_simd_softmax_row_args_t *)arg;
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_softmax_row_args_t);
    const uint32_t x = a->x_addr, tmp = a->tmp_addr, e = a->e_addr, p = a->p_addr;
    const uint32_t beats = a->beats;
    const uint32_t xl = x - SIMD_BEAT_BYTES, el = e - SIMD_BEAT_BYTES;
    const uint32_t s = e + beats * SIMD_BEAT_BYTES;
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_START);
    BINGO_SIMD_REQUIRE_LOCAL(xl, "simd_softmax_row", "x's latch");
    BINGO_SIMD_REQUIRE_LOCAL(tmp, "simd_softmax_row", "tmp");
    BINGO_SIMD_REQUIRE_LOCAL(el, "simd_softmax_row", "e's latch");
    BINGO_SIMD_REQUIRE_LOCAL(p, "simd_softmax_row", "p");
    if (beats == 0u || ((x | tmp | e | p) & 63u)) {
        printf_safe("[Cluster %d Core %d]: Error! simd_softmax_row: beats=%d, every buffer "
                    "64-B aligned.\r\n", snrt_cluster_idx(), snrt_cluster_core_idx(),
                    (int)beats);
        return BINGO_RET_FAIL;
    }
    snax_simd_shape_t in, out;
    // 1 the row's max
    snax_simd_shape_flat(&in, (void *)x, beats);
    snax_simd_shape_flat(&out, (void *)tmp, 1u);
    snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, beats, SIMD_RED_MAX);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    // 2 -m into x's latch
    snax_simd_shape_flat(&in, (void *)tmp, 1u);
    snax_simd_shape_flat(&out, (void *)xl, 1u);
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, SIMD_F32_NEG_ONE, 0u,
                   SIMD_FUNC_LINEAR);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    // 3 e = exp(x - m) and its sum, one sweep
    snax_simd_shape_flat(&in, (void *)xl, 1u + beats);
    snax_simd_shape_flat(&out, (void *)e, beats + 1u);
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                                     (1u << SIMD_EXT_STREAMMAP) |
                                                     (1u << SIMD_EXT_STREAMREDUCE));
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_0_CSR, 0, 1u);
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_0_CSR, 1, SIMD_EW_ADD | SIMD_EW_STICKY_B);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 0, SIMD_F32_ONE);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 1, 0u);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 2, SIMD_FUNC_EXP);
    snax_simd_set_op_csr(SIMD_EXT_STREAMREDUCE_CSR, 0, beats);
    snax_simd_set_op_csr(SIMD_EXT_STREAMREDUCE_CSR, 1, SIMD_RED_ADD | SIMD_RED_TAP);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    // 4 1/s = rsqrt(s * s) into e's latch
    snax_simd_shape_broadcast(&in, (void *)s, 2u);
    snax_simd_shape_flat(&out, (void *)el, 1u);
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                                     (1u << SIMD_EXT_STREAMMAP));
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_0_CSR, 0, 1u);
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_0_CSR, 1, SIMD_EW_MUL | SIMD_EW_STICKY_B);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 0, SIMD_F32_ONE);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 1, 0u);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 2, SIMD_FUNC_RSQRT);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    // 5 p = e / s
    snax_simd_shape_flat(&in, (void *)el, 1u + beats);
    snax_simd_shape_flat(&out, (void *)p, beats);
    snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1u,
                   SIMD_EW_MUL | SIMD_EW_STICKY_B);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    const uint32_t bad = snax_simd_wait_all_bounded() || snax_simd_bad_config();
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, 0u);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    if (bad) {
        printf_safe("[Cluster %d Core %d]: Error! simd_softmax_row did not drain cleanly "
                    "(status %08x)\r\n", snrt_cluster_idx(), snrt_cluster_core_idx(),
                    snax_read_simd_cfg_reg(SIMD_STATUS));
        return BINGO_RET_FAIL;
    }
    sp->return_value = p;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

// ==========================================================================
// SwiGLU of one token into ROW 0 of the down GEMV's A operand (snax dsv2_swiglu, a8 path):
//
//   1  Map SILU on gate                                        -> sg
//   2  EW1 MUL over [sg, up] (the lower address first: MUL commutes) -> Fp16ToInt8, with
//      the row write of simd_quant_a_row, each value from its two operands
//
// Each stage narrows to FP16, so a8 = sat127(rne(RNE(silu(gate) * up) * inv)), bit for bit
// hwmodel.moe_experts. inv comes from the FP32 word at inv_addr when that is non-zero: an
// expert SLOT's scale, which the router decides at run time.
// ==========================================================================
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_swiglu_a_row(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_swiglu_a_row_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_swiglu_a_row", "SIMD");
#if !(BINGO_HAS_STREAMMAP && BINGO_HAS_STREAMELEMENTWISE && BINGO_HAS_FP16TOINT8)
    BINGO_SIMD_EXT_UNSUPPORTED("simd_swiglu_a_row",
                               "StreamMap SILU, post-map StreamElementwise and Fp16ToInt8");
#else
    const __snax_bingo_kernel_simd_swiglu_a_row_args_t *a =
        (const __snax_bingo_kernel_simd_swiglu_a_row_args_t *)arg;
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_swiglu_a_row_args_t);
    const uint32_t g = a->g_addr, sg = a->sg_addr, out_a = a->output_addr, n = a->inter;
    const uint32_t up = g + 2u * n;
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_START);
    BINGO_SIMD_REQUIRE_LOCAL(g, "simd_swiglu_a_row", "gate|up");
    BINGO_SIMD_REQUIRE_LOCAL(sg, "simd_swiglu_a_row", "silu scratch");
    BINGO_SIMD_REQUIRE_LOCAL(out_a, "simd_swiglu_a_row", "A operand");
    if (n == 0u || (n % 32u) || ((g | sg | out_a) & 63u) || SIMD_MAX_DIM < 3) {
        printf_safe("[Cluster %d Core %d]: Error! simd_swiglu_a_row: inter=%d must be a "
                    "multiple of 32 and every buffer 64-B aligned (3-D reader needed).\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), (int)n);
        return BINGO_RET_FAIL;
    }
    uint32_t inv = a->inv_scale_f32bits;
    if (a->inv_addr) inv = *(volatile uint32_t *)a->inv_addr;
    // ROWS: token r's [gate | up] at g + r g_pitch, its A row at out_a + r a_pitch; the two
    // tasks of a row share sg, so a row waits for the one before it. A_BLK 8 writes a_row
    // (4 values and 4 untouched bytes a block, the caller zeroes them once), 64 row 0 of the
    // 16-row A layout.
    const uint32_t rows = a->rows ? a->rows : 1u, blk = a->a_blk ? a->a_blk : 64u;
    uint32_t bad = 0u;
    for (uint32_t r = 0; r < rows && !bad; r++) {
        const uint32_t gr = g + r * a->g_pitch, upr = gr + 2u * n;
        const uint32_t outr = out_a + r * a->a_pitch;
        if (r > 0u) {
            bad = snax_simd_wait_all_bounded() || snax_simd_bad_config();
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, 0u);
            if (bad) break;
        }
        snax_simd_shape_t in, out;
        // 1 sg = silu(gate)
        snax_simd_shape_flat(&in, (void *)gr, n / 32u);
        snax_simd_shape_flat(&out, (void *)sg, n / 32u);
        snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, SIMD_F32_ONE, 0u,
                       SIMD_FUNC_SILU);
        snax_simd_program_fast(&in, &out);
        snax_simd_fire();
        // 2 (sg * up) quantised into the row: operands d apart, the lower first
        const uint32_t lo = sg < upr ? sg : upr, d = sg < upr ? upr - sg : sg - upr;
        snax_simd_shape_flat(&in, (void *)lo, 1u);
        in.lane_stride = 4u;
        in.lane_mask = BINGO_SIMD_A_ROW_READ;
        in.bound[0] = 2u;           // the two operands of one value
        in.stride[0] = d;
        in.bound[1] = 2u;           // the two beats of one output beat
        in.stride[1] = 32u;
#if SIMD_MAX_DIM >= 3
        in.bound[2] = n / 32u;      // then the next 32 values
        in.stride[2] = 64u;
#endif
        in.dim = 3u;
        snax_simd_shape_flat(&out, (void *)outr, n / 32u);
        out.lane_stride = blk;
        out.stride[0] = 8u * blk;
        out.byte_mask = BINGO_SIMD_A_ROW_WORD;
        snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMELEMENTWISE_1) |
                                                         (1u << SIMD_EXT_FP16TOINT8));
        snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 0, 2u);
        snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1, SIMD_EW_MUL);
        snax_simd_set_op_csr(SIMD_EXT_FP16TOINT8_CSR, 0, inv);
        snax_simd_set_op_csr(SIMD_EXT_FP16TOINT8_CSR, 1, 0u);
        snax_simd_program_fast(&in, &out);
        snax_simd_fire();
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    bad = bad || snax_simd_wait_all_bounded() || snax_simd_bad_config();
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, 0u);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    if (bad) {
        printf_safe("[Cluster %d Core %d]: Error! simd_swiglu_a_row did not drain cleanly "
                    "(status %08x)\r\n", snrt_cluster_idx(), snrt_cluster_core_idx(),
                    snax_read_simd_cfg_reg(SIMD_STATUS));
        return BINGO_RET_FAIL;
    }
    sp->return_value = out_a;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}
