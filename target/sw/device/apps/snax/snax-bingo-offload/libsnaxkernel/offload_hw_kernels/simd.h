// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
//
// Core-level bingo SIMD kernels: the FP16 stream-operator primitives, and the fused
// whole-ops (softmax / rmsnorm / silu / swiglu / rope) that make up an LLM layer's
// epilogue. Every kernel here runs on the SIMD core, hart 1.
//
// THE OPERATOR CHAIN IS A FIXED LINEAR ORDER:
//
//     EW0 -> Map -> Reduce -> EW1 -> Fp16ToInt8
//
// One reader stream in, one writer stream out, operators applied in that order. So what
// can be fused into ONE pass over memory is decided by the HARDWARE, not by this file: a
// single task computes at most
//
//     write = quant( ew1( reduce( map( ew0( read ) ) ) ) )
//
// over one affine iteration space. Anything needing two passes over the same data, two
// output dtypes, or an operator out of that order is two tasks.
//
// THREE PROPERTIES OF THE CHAIN DO THE WORK IN THE KERNELS BELOW.
//
//  1. EW0 SITS BEFORE THE MAP. A combine that must happen BEFORE the pointwise
//     transform -- f(x - s) -- has somewhere to go, so softmax's `exp(x - max)` and its
//     row sum are ONE pass:
//         read [x, bcast] -> EW0(ADD) -> Map(EXP) -> Reduce(ADD|TAP) -> write
//     The generated header names the two elementwise instances
//     SIMD_EXT_STREAMELEMENTWISE_0 and _1; there is NO bare name, and choosing the
//     wrong one is silent -- EW1 would apply the subtract AFTER the exponential.
//
//  2. STICKY-B. SIMD_EW_STICKY_B latches the FIRST beat of a task as operand B and
//     combines every later beat against it, so a broadcast operand need not be
//     materialised at all: place the scalar beat immediately BEFORE its data and the
//     broadcast pass disappears. One task latches one beat, so this serves a whole-task
//     scalar (the rows == 1 paths); a per-ROW scalar still needs a real broadcast.
//
//  3. TAP. SIMD_RED_TAP passes the row through AND appends the reduction as a trailing
//     beat: N beats in, N+1 out. Softmax needs both halves -- the maximum and the tile
//     it came from -- so without TAP the next task would re-read the whole tile. The
//     pass-through also decides where the copy LANDS, which is what lets the next task
//     find its broadcast operand already adjacent to its data.
//
// A NOTE ON SIMD_RED_LANEWISE, because it is the one mode these kernels deliberately do
// NOT use. It emits the per-lane accumulators instead of folding them -- the reduction
// ACROSS beats rather than across lanes. A [rows, D] row-major tensor reduces ALONG a
// row, which spans both beats and the 32 lanes inside each beat, so the horizontal fold
// is unavoidable here. LANEWISE pays off only when the data is TRANSPOSED so that one
// lane is one row; an attention score tile stored [Bc, Br] is exactly that case, and it
// is why a transposed softmax reduces for free where this one does not.
//
// ADDRESSES ARE LOCAL. The SIMD block has no AXI port: it reads and writes this
// cluster's TCDM and nothing else. The BINGO arg structs carry 64-bit hi/lo pairs, so
// the kernels below take the low word and CHECK it is local -- an address that is not
// would otherwise be silently truncated into whatever TCDM sits at that offset.

#pragma once

#include "../macros.h"
#include "snax_core_roles.h"  // snax_is_simd_core(), BINGO_REQUIRE_CORE
#include "snax_fp16_math.h"   // integer recip_f16 / sqrt_f16 / f16_to_f32bits

// ==========================================================================
// Operator availability
//
// The stream operators are OPTIONAL hardware: the generated snax-simd-addr.h defines a
// SIMD_EXT_<NAME> index only for the ones the cluster cfg actually built. A cfg without
// one of them must still BUILD this library -- it is shared by every cfg -- so the
// restriction lives at CALL time: the kernel body is dead-code-eliminated (the
// BINGO_HAS_* flag is a literal 0) and the node fails loudly in the runtime instead of
// breaking everyone's build. The 0xFF sentinel keeps the unreachable body compiling and
// is out of range for every cfg, so even a mis-gated call cannot program a real
// operator.
// ==========================================================================
#include <snax_simd_lib.h>

#ifdef SIMD_EXT_STREAMMAP
#define BINGO_HAS_STREAMMAP 1
#else
#define BINGO_HAS_STREAMMAP 0
#define SIMD_EXT_STREAMMAP 0xFFu
#define SIMD_EXT_STREAMMAP_CSR 0u
#endif

#ifdef SIMD_EXT_STREAMREDUCE
#define BINGO_HAS_STREAMREDUCE 1
#else
#define BINGO_HAS_STREAMREDUCE 0
#define SIMD_EXT_STREAMREDUCE 0xFFu
#define SIMD_EXT_STREAMREDUCE_CSR 0u
#endif

// The two elementwise instances are separate operators with separate ids, and which one
// a kernel wants is a correctness question, not a preference: EW0 is the PRE-map
// combine, EW1 the POST-map one. BINGO_HAS_PREMAP_ELEMENTWISE gates the kernels whose
// fusion depends on EW0 existing.
#ifdef SIMD_EXT_STREAMELEMENTWISE_0
#define BINGO_HAS_PREMAP_ELEMENTWISE 1
#else
#define BINGO_HAS_PREMAP_ELEMENTWISE 0
#define SIMD_EXT_STREAMELEMENTWISE_0 0xFFu
#define SIMD_EXT_STREAMELEMENTWISE_0_CSR 0u
#endif

#ifdef SIMD_EXT_STREAMELEMENTWISE_1
#define BINGO_HAS_STREAMELEMENTWISE 1
#else
#define BINGO_HAS_STREAMELEMENTWISE 0
#define SIMD_EXT_STREAMELEMENTWISE_1 0xFFu
#define SIMD_EXT_STREAMELEMENTWISE_1_CSR 0u
#endif

#ifdef SIMD_EXT_FP16TOINT8
#define BINGO_HAS_FP16TOINT8 1
#else
#define BINGO_HAS_FP16TOINT8 0
#define SIMD_EXT_FP16TOINT8 0xFFu
#define SIMD_EXT_FP16TOINT8_CSR 0u
#endif

// Refuse a kernel whose operator this cfg did not generate.
#define BINGO_SIMD_EXT_UNSUPPORTED(kname, exts)                              \
    do {                                                                     \
        printf_safe("[Cluster %d Core %d]: Error! " kname " needs the SIMD " \
                    exts " operator, which this RTL cfg does not generate. " \
                    "Kernel not executed.\r\n",                              \
                    snrt_cluster_idx(), snrt_cluster_core_idx());            \
        return BINGO_RET_FAIL;                                               \
    } while (0)

// Output precision of a fused kernel's SINGLE writer pass. Chosen by the kernel-name
// WRAPPER (__..._f16_f16 / __..._f16_i8), never by a user arg -- so the user's kernel
// args stay { input, output, rows, cols } with no HW or quant detail leaking through.
#define SIMD_OUT_F16 0u
#define SIMD_OUT_I8 1u

// Baked Fp16ToInt8 scales. The producer's range is known per op, so the user supplies no
// scale: softmax output is in [0,1] (127.0), rmsnorm is ~[-2,2] (64.0), silu/swiglu are
// wider (16.0).
#define BINGO_SIMD_I8_SCALE_UNIT 0x42FE0000u  // 127.0f
#define BINGO_SIMD_I8_SCALE_NORM 0x42800000u  // 64.0f
#define BINGO_SIMD_I8_SCALE_ACT 0x41800000u   // 16.0f

// ==========================================================================
// Address handling
// ==========================================================================
static inline bool simd_addr_is_local(uint64_t addr) {
    uint32_t lo = (uint32_t)addr;
    return (lo >= snrt_l1_start_addr()) && (lo < snrt_l1_end_addr());
}

#define BINGO_SIMD_REQUIRE_LOCAL(_addr, _kname, _what)                          \
    do {                                                                        \
        if (!simd_addr_is_local(_addr)) {                                       \
            printf_safe("[Cluster %d Core %d]: Error! " _kname " " _what        \
                        " 0x%08x is not in this cluster's L1. The SIMD block "  \
                        "has no AXI port.\r\n",                                 \
                        snrt_cluster_idx(), snrt_cluster_core_idx(),            \
                        (uint32_t)(_addr));                                     \
            return BINGO_RET_FAIL;                                              \
        }                                                                       \
    } while (0)

// ==========================================================================
// Pass helpers
//
// Each runs ONE SIMD task and waits for it. They take local pointers and shapes, arm the
// operators with constant-address writes, and drain with a BOUNDED poll so a wedged
// engine reports itself instead of hanging the simulator.
//
// The AGU is 3-D with a single destination, so a shape is at most {operand, beat, row}
// and there is no destination-slot or multicast bookkeeping to do.
// ==========================================================================

// Run the currently-armed chain over `in` -> `out`, full geometry program. Returns
// BINGO_RET_SUCC, or BINGO_RET_FAIL on a drain timeout / degenerate geometry.
static inline uint32_t simd_run_shapes(const snax_simd_shape_t *in,
                                       const snax_simd_shape_t *out) {
    snax_simd_program_fast(in, out);
    snax_simd_fire();
    if (snax_simd_wait_all_bounded()) {
        printf_safe("[Cluster %d Core %d]: SIMD task drain timeout (status %08x)\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(),
                    snax_read_simd_cfg_reg(SIMD_STATUS));
        return BINGO_RET_FAIL;
    }
    // Sticky bit: a task whose AGU never became busy was DROPPED, not run. Nothing else
    // reports it -- the counters advance and the output buffer simply keeps its old
    // contents, which is indistinguishable from a correct run of the wrong data.
    if (snax_simd_bad_config()) {
        printf_safe("[Cluster %d Core %d]: SIMD refused a degenerate task "
                    "(bad-config sticky set); check the shape bounds.\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    return BINGO_RET_SUCC;
}

// Per-row reduction, row -> one splatted scalar beat. 2-D reader {beat, row}; 1-D writer.
// `mode` is SIMD_RED_MAX / _ADD / _SUMSQ, optionally OR-ed with TAP / FP32OUT / LANEWISE.
static inline uint32_t simd_pass_reduce(void *src, void *dst, uint32_t rows,
                                        uint32_t beats, uint32_t mode,
                                        uint32_t dst_beats) {
    snax_simd_shape_t in, out;
    snax_simd_shape_rows(&in, src, rows, beats, beats * SIMD_BEAT_BYTES);
    snax_simd_shape_flat(&out, dst, dst_beats);
    snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, beats, mode);
    return simd_run_shapes(&in, &out);
}

// out = func(a*x + b) over `flat` beats, optional fused Fp16ToInt8 (which HALVES the
// beats the writer emits, so the caller's dst_beats must halve with it).
static inline uint32_t simd_pass_map(void *src, void *dst, uint32_t flat,
                                     uint32_t a_bits, uint32_t b_bits,
                                     uint32_t func, uint32_t out_dt,
                                     uint32_t inv_scale) {
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, src, flat);
    snax_simd_shape_flat(&out, dst, (out_dt == SIMD_OUT_I8) ? (flat / 2u) : flat);
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, a_bits, b_bits, func);
    if (out_dt == SIMD_OUT_I8)
        snax_simd_arm1(SIMD_EXT_FP16TOINT8, SIMD_EXT_FP16TOINT8_CSR, inv_scale);
    return simd_run_shapes(&in, &out);
}

// Broadcast one beat per row to `beats` beats per row, with a fused StreamMap LINEAR
// (out = a*x) applied on the fly -- a = -1.0 does the negate INSIDE the broadcast, which
// is why softmax needs no DM-core negate loop. `dst_row_stride` lets the destination be
// TAP-padded so a later 2-operand pass keeps a constant interleave delta across rows.
static inline uint32_t simd_pass_bcast_map(void *src_beats, void *dst,
                                           uint32_t rows, uint32_t beats,
                                           uint32_t dst_row_stride,
                                           uint32_t a_bits) {
    snax_simd_shape_t in, out;
    // Reader: {beat (stride 0 -> repeat), row}. One beat per row, presented `beats` times.
    snax_simd_shape_2d(&in, src_beats, beats, 0u, rows, SIMD_BEAT_BYTES);
    snax_simd_shape_rows(&out, dst, rows, beats, dst_row_stride);
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, a_bits, 0u,
                   SIMD_FUNC_LINEAR);
    return simd_run_shapes(&in, &out);
}

// Two-operand elementwise over rows that are `src_row_stride` bytes apart, of which only
// the first `beats` beats are data (set src_row_stride = beats*SIMD_BEAT_BYTES for the
// packed case). Reader is 3-D {operand, beat, row}; writer is packed 1-D.
//
// The reader AGU strides FORWARD only, so the base must be the LOWER operand -- a
// "negative" stride wraps, reads outside TCDM and STALLS the task forever. The swap is
// valid because MUL and ADD commute; see snax_simd_ew2_base().
//
// `ext` picks WHICH elementwise instance: EW1 for a combine that follows the map (the
// classic case), EW0 for one that must precede it.
static inline uint32_t simd_pass_ew2(void *src_a, void *src_b, void *dst,
                                     uint32_t rows, uint32_t beats,
                                     uint32_t src_row_stride, uint32_t ext,
                                     uint32_t ext_csr, uint32_t op,
                                     uint32_t out_dt, uint32_t inv_scale) {
    uint32_t operand_stride;
    void *base = snax_simd_ew2_base(src_a, src_b, &operand_stride);
    snax_simd_shape_t in, out;
    snax_simd_shape_clear(&in);
    in.base = base;
    in.lane_stride = SIMD_LANE_BYTES;
    in.lane_mask = 0xFFFFFFFFu;
    in.byte_mask = 0xFFFFFFFFu;
    in.dim = 3;
    in.bound[0] = 2u;
    in.stride[0] = operand_stride;
    in.bound[1] = beats;
    in.stride[1] = SIMD_BEAT_BYTES;
    in.bound[2] = rows;
    in.stride[2] = src_row_stride;
    uint32_t flat = rows * beats;
    snax_simd_shape_flat(&out, dst, (out_dt == SIMD_OUT_I8) ? (flat / 2u) : flat);
    snax_simd_use2(ext, ext_csr, 2u, op);
    if (out_dt == SIMD_OUT_I8)
        snax_simd_arm1(SIMD_EXT_FP16TOINT8, SIMD_EXT_FP16TOINT8_CSR, inv_scale);
    return simd_run_shapes(&in, &out);
}

// THE FUSED EPILOGUE PASS.
//
//     read [x, bcast]  ->  EW0(op_pre)  ->  Map(func)  ->  Reduce(mode|TAP)  ->  write
//
// The combine, the pointwise transform and the fold in ONE sweep over the tile, with no
// intermediate written between them. This is what EW0 sitting upstream of the map buys:
// `exp(x - s)` needs no materialised `x - s`.
//
// The reader interleaves the two operands (x and the per-row broadcast) exactly as
// simd_pass_ew2 does, so the same row-stride rules apply -- pass row_b for packed
// operands, pad_row for TAP-padded ones. Note the asymmetry this pass creates: its
// READER is packed (both operands are plain [rows, D] tensors) while its WRITER is
// TAP-padded, because the reduce appends a scalar per row. rows*(beats+1) out.
static inline uint32_t simd_pass_ew_map_reduce(void *src_x, void *src_b, void *dst,
                                               uint32_t rows, uint32_t beats,
                                               uint32_t src_row_stride,
                                               uint32_t ew_op, uint32_t func,
                                               uint32_t red_mode) {
    uint32_t operand_stride;
    void *base = snax_simd_ew2_base(src_x, src_b, &operand_stride);
    snax_simd_shape_t in, out;
    snax_simd_shape_clear(&in);
    in.base = base;
    in.lane_stride = SIMD_LANE_BYTES;
    in.lane_mask = 0xFFFFFFFFu;
    in.byte_mask = 0xFFFFFFFFu;
    in.dim = 3;
    in.bound[0] = 2u;
    in.stride[0] = operand_stride;
    in.bound[1] = beats;
    in.stride[1] = SIMD_BEAT_BYTES;
    in.bound[2] = rows;
    in.stride[2] = src_row_stride;
    snax_simd_shape_flat(&out, dst, rows * (beats + 1u));
    // All three operators in ONE enable mask: a single csrw, then the operator CSRs at
    // constant addresses. Order in the mask is irrelevant -- the CHAIN order is fixed in
    // hardware by the cfg's reader_extensions list.
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR,
                            (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                (1u << SIMD_EXT_STREAMMAP) |
                                (1u << SIMD_EXT_STREAMREDUCE));
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 0, 2u);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 1, ew_op);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 0, SIMD_F32_ONE);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 1, 0u);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 2, func);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 0, beats);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 1, red_mode);
    return simd_run_shapes(&in, &out);
}

// Map |> Reduce in one pass, no pre-combine: out = reduce(map(a*x + b)) per row, TAP
// optional.
static inline uint32_t simd_pass_map_reduce(void *src, void *dst, uint32_t rows,
                                            uint32_t beats, uint32_t a_bits,
                                            uint32_t b_bits, uint32_t func,
                                            uint32_t red_mode, uint32_t dst_beats) {
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, src, rows * beats);
    snax_simd_shape_flat(&out, dst, dst_beats);
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMMAP) |
                                                     (1u << SIMD_EXT_STREAMREDUCE));
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 0, a_bits);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 1, b_bits);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 2, func);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 0, beats);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 1, red_mode);
    return simd_run_shapes(&in, &out);
}


// ==========================================================================
// Scratch pools
//
// The bingo L1 heap malloc/free is a real free-list allocator (~5k cc per call) and the
// scratch these kernels need is small and identical from call to call. Allocate ONCE,
// lazily, and reuse: one heap call for the life of the program. Only a tile larger than
// the pool falls back to a per-call malloc/free. `s_pool` is per-kernel and touched only
// by the SIMD core, so there is no race.
// ==========================================================================
#define BINGO_SIMD_SCRATCH_POOL 8192u

// ==========================================================================
// PRIMITIVES -- one operator (or one fused chain) per kernel, shapes from the args.
// These are the building blocks a host DFG composes; the fused whole-ops below are what
// it should use when the whole op fits in one node.
// ==========================================================================

// StreamReduce: per-row reduction (row -> scalar). `op` = MAX/ADD/SUMSQ, optionally
// OR-ed with SIMD_RED_TAP (pass the row through and append the scalar),
// SIMD_RED_FP32OUT (keep the scalar in FP32 -- the FP16 narrow WRAPS to garbage rather
// than saturating to inf, so use it whenever the sum can exceed FP16 range) or
// SIMD_RED_LANEWISE (emit the per-lane partials; only meaningful for a TRANSPOSED
// layout, see the header).
//
// Arg layout: src[2], dst[2], beats, op, rows, csr_mode (ignored), dst_bound0.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_stream_reduce(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_stream_reduce_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_stream_reduce", "SIMD");
#if !BINGO_HAS_STREAMREDUCE
    BINGO_SIMD_EXT_UNSUPPORTED("simd_stream_reduce", "StreamReduce");
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t *a = (uint32_t *)arg;
    uint64_t src_addr = make_u64(a[0], a[1]);
    uint64_t dst_addr = make_u64(a[2], a[3]);
    uint32_t beats = a[4];
    uint32_t op = a[5];
    uint32_t rows = a[6];
    uint32_t dst_bound0 = a[8];
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_stream_reduce_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_SIMD_REQUIRE_LOCAL(src_addr, "simd_stream_reduce", "src");
    BINGO_SIMD_REQUIRE_LOCAL(dst_addr, "simd_stream_reduce", "dst");

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_START);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint32_t rc = simd_pass_reduce((void *)(uint32_t)src_addr, (void *)(uint32_t)dst_addr,
                                   rows, beats, op, dst_bound0);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    if (rc != BINGO_RET_SUCC) return BINGO_RET_FAIL;
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

// StreamMap: out = func(a*x + b) per element, over `rows*beats` flat beats. Optional
// fused FP16->INT8 (out_dtype = 1; then dst_bound0 = rows*beats/2).
//
// A nonzero a_addr is the runtime-scale escape hatch: a producer kernel wrote the FP32
// bits into that L1 word (the dequant scale), so read it instead of the baked
// immediate.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_stream_map(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_stream_map_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_stream_map", "SIMD");
#if !BINGO_HAS_STREAMMAP
    BINGO_SIMD_EXT_UNSUPPORTED("simd_stream_map", "StreamMap");
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t *a = (uint32_t *)arg;
    uint64_t src_addr = make_u64(a[0], a[1]);
    uint64_t dst_addr = make_u64(a[2], a[3]);
    uint32_t beats = a[4];
    uint32_t a_f32bits = a[5];
    uint32_t b_f32bits = a[6];
    uint32_t func = a[7];
    uint32_t rows = a[8];
    uint32_t out_dtype = a[11];
    uint32_t inv_scale = a[12];
    uint32_t a_addr_hi = a[13];
    uint32_t a_addr_lo = a[14];
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_stream_map_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    if (out_dtype == SIMD_OUT_I8 && !BINGO_HAS_FP16TOINT8)
        BINGO_SIMD_EXT_UNSUPPORTED("simd_stream_map with out_dtype=int8", "Fp16ToInt8");
    BINGO_SIMD_REQUIRE_LOCAL(src_addr, "simd_stream_map", "src");
    BINGO_SIMD_REQUIRE_LOCAL(dst_addr, "simd_stream_map", "dst");

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_START);
    if (a_addr_lo || a_addr_hi) a_f32bits = *(volatile uint32_t *)a_addr_lo;
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint32_t rc = simd_pass_map((void *)(uint32_t)src_addr, (void *)(uint32_t)dst_addr,
                                rows * beats, a_f32bits, b_f32bits, func, out_dtype,
                                inv_scale);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    if (rc != BINGO_RET_SUCC) return BINGO_RET_FAIL;
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

// MERGED StreamMap |> StreamReduce in ONE task: per row,
// out = reduce(reduce_op, map(func, a*x + b)).
//
// TWO OUTPUT MODES, by the SIMD_RED_TAP bit in `reduce_op`:
//   TAP SET   -- the mapped row passes through 1:1 AND the row's scalar is appended as a
//                trailing beat: a PADDED [rows, beats+1] tensor. dst_bound0 =
//                rows*(beats+1). MIND THE PADDED ROW STRIDE -- a downstream consumer must
//                either be single-row or address rows at (beats+1)*64 explicitly; a flat
//                reader would walk the scalar beats as if they were data.
//   TAP CLEAR -- only the per-row scalars, exactly like stream_reduce but over the MAPPED
//                values. dst_bound0 = rows.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_stream_map_reduce(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_stream_map_reduce_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_stream_map_reduce", "SIMD");
#if !BINGO_HAS_STREAMMAP || !BINGO_HAS_STREAMREDUCE
    BINGO_SIMD_EXT_UNSUPPORTED("simd_stream_map_reduce", "StreamMap+StreamReduce");
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t *a = (uint32_t *)arg;
    uint64_t src_addr = make_u64(a[0], a[1]);
    uint64_t dst_addr = make_u64(a[2], a[3]);
    uint32_t beats = a[4];
    uint32_t a_f32bits = a[5];
    uint32_t b_f32bits = a[6];
    uint32_t func = a[7];
    uint32_t reduce_op = a[8];
    uint32_t rows = a[9];
    uint32_t dst_bound0 = a[11];
    uint32_t a_addr_hi = a[12];
    uint32_t a_addr_lo = a[13];
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_stream_map_reduce_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_SIMD_REQUIRE_LOCAL(src_addr, "simd_stream_map_reduce", "src");
    BINGO_SIMD_REQUIRE_LOCAL(dst_addr, "simd_stream_map_reduce", "dst");

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_START);
    if (a_addr_lo || a_addr_hi) a_f32bits = *(volatile uint32_t *)a_addr_lo;
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint32_t rc = simd_pass_map_reduce((void *)(uint32_t)src_addr,
                                       (void *)(uint32_t)dst_addr, rows, beats,
                                       a_f32bits, b_f32bits, func, reduce_op,
                                       dst_bound0);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    if (rc != BINGO_RET_SUCC) return BINGO_RET_FAIL;
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

// Fp16ToInt8: out[i] = clamp(round(in[i] * inv_scale), -128, 127) over rows*beats flat
// beats. The narrow is chained after an IDENTITY StreamMap (LINEAR a=1, b=0) so the
// reader emits the fp16 stream the quant lane consumes. inv_scale = FP32 bits of
// 127/max|x|; dst_bound0 = rows*beats/2 (int8 packs two elements per fp16 lane).
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_fp16_to_int8(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_fp16_to_int8_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_fp16_to_int8", "SIMD");
#if !BINGO_HAS_STREAMMAP || !BINGO_HAS_FP16TOINT8
    BINGO_SIMD_EXT_UNSUPPORTED("simd_fp16_to_int8", "StreamMap+Fp16ToInt8");
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t *a = (uint32_t *)arg;
    uint64_t src_addr = make_u64(a[0], a[1]);
    uint64_t dst_addr = make_u64(a[2], a[3]);
    uint32_t beats = a[4];
    uint32_t rows = a[5];
    uint32_t inv_scale = a[6];
    uint32_t inv_scale_hi = a[9];
    uint32_t inv_scale_lo = a[10];
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_fp16_to_int8_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_SIMD_REQUIRE_LOCAL(src_addr, "simd_fp16_to_int8", "src");
    BINGO_SIMD_REQUIRE_LOCAL(dst_addr, "simd_fp16_to_int8", "dst");

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_START);
    // The requant scale is per-tensor and computed at run time (max|x| via MAX(x) +
    // MAX(-x), then a host reciprocal) -> read it from L1 when an address is given.
    if (inv_scale_lo || inv_scale_hi) inv_scale = *(volatile uint32_t *)inv_scale_lo;
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint32_t rc = simd_pass_map((void *)(uint32_t)src_addr, (void *)(uint32_t)dst_addr,
                                rows * beats, SIMD_F32_ONE, 0u, SIMD_FUNC_LINEAR,
                                SIMD_OUT_I8, inv_scale);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    if (rc != BINGO_RET_SUCC) return BINGO_RET_FAIL;
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

// StreamElementwise: out = op(operand_0, operand_1) over two interleaved streams, across
// rows*beats beats, with optional fused FP16->INT8.
//
// This uses EW1, the POST-map instance. A combine that must PRECEDE a map belongs in
// EW0 and is reachable through the fused kernels below, not through this generic one.
//
// REQUIRED LAYOUT: the two operand buffers must be the same size and identically laid
// out, so operand_0[beat] and operand_1[beat] are a CONSTANT stride apart, and that
// stride must be small (the two buffers near each other in L1). The kernel bases the
// reader at the lower operand and strides up, which is valid because MUL and ADD commute
// -- so callers need not pre-order the operands.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_stream_elementwise(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_stream_elementwise_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_stream_elementwise", "SIMD");
#if !BINGO_HAS_STREAMELEMENTWISE
    BINGO_SIMD_EXT_UNSUPPORTED("simd_stream_elementwise", "StreamElementwise");
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t *a = (uint32_t *)arg;
    uint64_t src_addr = make_u64(a[0], a[1]);
    uint64_t dst_addr = make_u64(a[2], a[3]);
    uint32_t beats = a[4];
    uint32_t operand_stride = a[5];
    uint32_t operand_count = a[6];
    uint32_t op = a[7];
    uint32_t rows = a[8];
    uint32_t out_dtype = a[11];
    uint32_t inv_scale = a[12];
    uint32_t src_b_addr_hi = a[13];
    uint32_t src_b_addr_lo = a[14];
    uint32_t src_row_stride = a[15];
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_stream_elementwise_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    if (out_dtype == SIMD_OUT_I8 && !BINGO_HAS_FP16TOINT8)
        BINGO_SIMD_EXT_UNSUPPORTED("simd_stream_elementwise with out_dtype=int8",
                                   "Fp16ToInt8");
    if (operand_count != 2u) {
        printf_safe("[Cluster %d Core %d]: Error! simd_stream_elementwise supports "
                    "operand_count=2 only (got %d).\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), operand_count);
        return BINGO_RET_FAIL;
    }
    BINGO_SIMD_REQUIRE_LOCAL(src_addr, "simd_stream_elementwise", "src_a");
    BINGO_SIMD_REQUIRE_LOCAL(dst_addr, "simd_stream_elementwise", "dst");

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_START);
    // Either the caller gave a second operand address (preferred -- the stride is derived
    // here and the swap handled) or it baked a stride into the args.
    void *src_a = (void *)(uint32_t)src_addr;
    void *src_b;
    if (src_b_addr_lo || src_b_addr_hi) {
        BINGO_SIMD_REQUIRE_LOCAL(make_u64(src_b_addr_hi, src_b_addr_lo),
                                 "simd_stream_elementwise", "src_b");
        src_b = (void *)src_b_addr_lo;
    } else {
        src_b = (void *)((uint32_t)src_addr + operand_stride);
    }
    // src_row_stride == 0 means the flat/packed case: rows are `beats` beats apart.
    if (src_row_stride == 0u) src_row_stride = beats * SIMD_BEAT_BYTES;
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint32_t rc = simd_pass_ew2(src_a, src_b, (void *)(uint32_t)dst_addr, rows, beats,
                                src_row_stride, SIMD_EXT_STREAMELEMENTWISE_1,
                                SIMD_EXT_STREAMELEMENTWISE_1_CSR, op, out_dtype,
                                inv_scale);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    if (rc != BINGO_RET_SUCC) return BINGO_RET_FAIL;
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

// ==========================================================================
// Fused FP16 softmax -- the WHOLE pipeline in ONE kernel.
//
// out[r, :] = softmax(x[r, :]), D = cols fp16 per row. Replaces the ~11-node host DFG
// (reduce-MAX, host negate, broadcast, sub-max, merged EXP+Sexp, gather, host
// reciprocal, broadcast, normalize-MUL, quant): everything runs on the SIMD core, so the
// host only does Load / Store / Check.
//
// The two per-row SCALAR transforms are done here as INTEGER bit-ops on the fp16 pattern
// -- the cluster cores are rv32ima with no FPU, so a float op would trap:
//   -max : XOR 0x8000 (sign flip);  1/Sexp : recip_f16() (integer divu).
//
// TWO PATHS, chosen by `rows`:
//
//   rows == 1 -- FAST PATH. One row shares one -max and one 1/Sexp, so both fold into
//   StreamMap's scalar CSRs (b and a) and every broadcast and elementwise pass vanishes:
//       reduce(MAX)                            x    -> bt
//       [core] b = -max
//       map(EXP, b=-max) |> reduce(ADD,TAP)    x    -> expb     (subtract folded in)
//       [core] a = 1/Sexp
//       map(LINEAR, a=1/Sexp) [+ quant]        expb -> out
//   3 SIMD tasks, no scratch beyond bt and expb.
//
//   rows > 1 -- GENERAL PATH. Each row has its OWN -max and 1/Sexp, and StreamMap's a,b
//   are single scalars, so the per-row scalars must be broadcast and applied
//   elementwise. EW0 sits upstream of the map, so the subtract, the exponential and the
//   sum are still ONE pass.
//       reduce(MAX)                                  x      -> bt
//       bcast_map(a=-1)                              bt     -> bc     (negate fused in)
//       EW0(ADD) |> map(EXP) |> reduce(ADD,TAP)      x,bc   -> expb   <-- 3 passes in 1
//       [core] per-row 1/Sexp, splatted
//       bcast                                        bt     -> bc
//       EW1(MUL) [+ quant]                           expb,bc-> out
//   5 SIMD tasks, and no `x - max` intermediate: the fused pass never writes one.
//
// Arg layout (__snax_bingo_kernel_simd_softmax_args_t): input[2], output[2], rows, cols.
// ==========================================================================
static inline uint32_t __snax_bingo_kernel_simd_softmax(void *arg, uint32_t out_prec) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_softmax_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_softmax", "SIMD");
#if !BINGO_HAS_STREAMMAP || !BINGO_HAS_STREAMREDUCE || !BINGO_HAS_STREAMELEMENTWISE
    (void)out_prec;
    BINGO_SIMD_EXT_UNSUPPORTED("simd_softmax",
                               "StreamMap+StreamReduce+StreamElementwise");
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t *a = (uint32_t *)arg;
    uint64_t in_addr = make_u64(a[0], a[1]);
    uint64_t out_addr = make_u64(a[2], a[3]);
    uint32_t rows = a[4];
    uint32_t cols = a[5];
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_softmax_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    uint32_t out_i8 = (out_prec == SIMD_OUT_I8);
    if (out_i8 && !BINGO_HAS_FP16TOINT8)
        BINGO_SIMD_EXT_UNSUPPORTED("simd_softmax_f16_i8", "Fp16ToInt8");
    if (rows > 1u && !BINGO_HAS_PREMAP_ELEMENTWISE)
        BINGO_SIMD_EXT_UNSUPPORTED("simd_softmax with rows>1",
                                   "StreamElementwise_0 (pre-map)");
    BINGO_SIMD_REQUIRE_LOCAL(in_addr, "simd_softmax", "input");
    BINGO_SIMD_REQUIRE_LOCAL(out_addr, "simd_softmax", "output");

    uint32_t out_dt = out_i8 ? SIMD_OUT_I8 : SIMD_OUT_F16;
    uint32_t beats = cols >> 5u;                                    // 32 fp16 per beat
    uint32_t inv_scale = out_i8 ? BINGO_SIMD_I8_SCALE_UNIT : 0u;    // out in [0,1]
    uint32_t row_b = beats * SIMD_BEAT_BYTES;                       // packed row pitch
    uint32_t pad_row = (beats + 1u) * SIMD_BEAT_BYTES;              // TAP row pitch
    uint32_t pad_b = rows * pad_row;

    // Scratch: [ bt | bc | expb ]. `xs` is GONE -- the fused EW0|Map|Reduce pass never
    // materialises x - max. bt holds the per-row scalar beats (max, then 1/Sexp); bc the
    // broadcast operand; expb the TAP-padded exp tile.
    uint32_t bt_off = 0u;
    uint32_t bc_off = rows * SIMD_BEAT_BYTES;
    uint32_t expb_off = bc_off + pad_b;
    uint32_t scratch_bytes = expb_off + pad_b + 64u;
    static uint32_t s_pool = 0u;
    uint32_t scratch_lo, from_pool = 0u;
    if (scratch_bytes <= BINGO_SIMD_SCRATCH_POOL) {
        if (!s_pool) s_pool = snrt_l1_malloc(BINGO_SIMD_SCRATCH_POOL);
        scratch_lo = s_pool;
        from_pool = 1u;
    } else {
        scratch_lo = snrt_l1_malloc(scratch_bytes);
    }
    if (!scratch_lo) {
        printf_safe("[Cluster %d Core %d]: softmax L1 scratch alloc failed!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    uint32_t base_lo = (scratch_lo + 63u) & ~63u;
    void *bt = (void *)(base_lo + bt_off);
    void *bc = (void *)(base_lo + bc_off);
    void *expb = (void *)(base_lo + expb_off);
    volatile uint16_t *bt_l = (volatile uint16_t *)bt;
    volatile uint16_t *expb_l = (volatile uint16_t *)expb;
    uint32_t sum_h = beats * 32u;  // lane offset of the TAP trailing scalar beat

    uint32_t rc;

    // reduce(MAX): x -> bt, one splatted scalar beat per row.
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    rc = simd_pass_reduce((void *)(uint32_t)in_addr, bt, rows, beats, SIMD_RED_MAX, rows);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);

    if (rows == 1u) {
        // [core] b = -max: read the splatted max, sign-flip, widen to fp32 bits.
        BINGO_TRACE_MARKER(BINGO_TRACE_SCALAR_RUN_START);
        uint32_t b_bits = 0u;
        if (rc == BINGO_RET_SUCC)
            b_bits = f16_to_f32bits((uint16_t)(bt_l[0] ^ 0x8000u));
        BINGO_TRACE_MARKER(BINGO_TRACE_SCALAR_RUN_END);
        // map(EXP, a=1, b=-max) |> reduce(ADD, TAP): x -> expb, subtract folded in.
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
        if (rc == BINGO_RET_SUCC)
            rc = simd_pass_map_reduce((void *)(uint32_t)in_addr, expb, rows, beats,
                                      SIMD_F32_ONE, b_bits, SIMD_FUNC_EXP,
                                      SIMD_RED_ADD | SIMD_RED_TAP, beats + 1u);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
        // [core] a = 1/Sexp from the TAP trailing beat.
        BINGO_TRACE_MARKER(BINGO_TRACE_SCALAR_RUN_START);
        uint32_t a_bits = 0u;
        if (rc == BINGO_RET_SUCC) a_bits = f16_to_f32bits(recip_f16(expb_l[sum_h]));
        BINGO_TRACE_MARKER(BINGO_TRACE_SCALAR_RUN_END);
        // map(LINEAR, a=1/Sexp) [+ quant]: expb -> out. The trailing Sexp beat sits just
        // past `beats`, so a flat reader of `beats` beats skips it.
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
        if (rc == BINGO_RET_SUCC)
            rc = simd_pass_map(expb, (void *)(uint32_t)out_addr, beats, a_bits, 0u,
                               SIMD_FUNC_LINEAR, out_dt, inv_scale);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    } else {
        // bcast_map(a = -1): bt -> bc, PACKED (row_b). The negate happens inside the
        // broadcast -- no separate core loop.
        //
        // Packed, not padded, because the fused pass below reads bc alongside x and BOTH
        // operands must share one row stride: the 3-D reader has a single stride per
        // dimension, so a packed x and a padded bc cannot be interleaved. x is the
        // caller's buffer and is packed by definition, so bc matches it. (The TAP padding
        // appears only on the fused pass's OUTPUT, and only the normalize pass below --
        // which reads that output -- uses pad_row.)
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
        if (rc == BINGO_RET_SUCC)
            rc = simd_pass_bcast_map(bt, bc, rows, beats, row_b, SIMD_F32_NEG_ONE);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
        // THE FUSED PASS: EW0(ADD) -> Map(EXP) -> Reduce(ADD|TAP). x + (-max), then exp,
        // then the row sum, in ONE sweep over the tile. This is the three old passes
        // (broadcast-subtract, exp, sum) collapsed into one, and the `xs` intermediate
        // they needed is gone with them -- a full [rows, D] write and read removed.
        //
        // The reduce re-inits per row off its own operandCount = beats counter, so the
        // reader needs no row dimension of its own beyond the packed walk.
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
        if (rc == BINGO_RET_SUCC)
            rc = simd_pass_ew_map_reduce((void *)(uint32_t)in_addr, bc, expb, rows, beats,
                                         row_b, SIMD_EW_ADD, SIMD_FUNC_EXP,
                                         SIMD_RED_ADD | SIMD_RED_TAP);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
        // [core] per-row 1/Sexp, splatted across bt's row beat (two lanes per store).
        BINGO_TRACE_MARKER(BINGO_TRACE_SCALAR_RUN_START);
        if (rc == BINGO_RET_SUCC) {
            uint32_t pad_row_h = (beats + 1u) * 32u;
            for (uint32_t r = 0u; r < rows; r++) {
                uint16_t inv = recip_f16(expb_l[r * pad_row_h + sum_h]);
                uint32_t inv2 = ((uint32_t)inv << 16) | inv;
                volatile uint32_t *row32 = (volatile uint32_t *)(bt_l + r * 32u);
                for (uint32_t l = 0u; l < 16u; l++) row32[l] = inv2;
            }
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_SCALAR_RUN_END);
        // bcast bt(1/Sexp) -> bc, padded (identity map, a = 1).
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
        if (rc == BINGO_RET_SUCC)
            rc = simd_pass_bcast_map(bt, bc, rows, beats, pad_row, SIMD_F32_ONE);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
        // EW1(MUL) [+ quant]: out = exp * (1/Sexp). Both operands padded, packed write.
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
        if (rc == BINGO_RET_SUCC)
            rc = simd_pass_ew2(expb, bc, (void *)(uint32_t)out_addr, rows, beats, pad_row,
                               SIMD_EXT_STREAMELEMENTWISE_1,
                               SIMD_EXT_STREAMELEMENTWISE_1_CSR, SIMD_EW_MUL, out_dt,
                               inv_scale);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    }

    if (!from_pool) snrt_l1_free(scratch_lo);
    if (rc != BINGO_RET_SUCC) {
        printf_safe("[Cluster %d Core %d]: softmax pass failed!\r\n", snrt_cluster_idx(),
                    snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    sp->return_value = (uint32_t)out_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_softmax_f16_f16(void *arg) {
    return __snax_bingo_kernel_simd_softmax(arg, SIMD_OUT_F16);
}
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_softmax_f16_i8(void *arg) {
    return __snax_bingo_kernel_simd_softmax(arg, SIMD_OUT_I8);
}

// ==========================================================================
// Fused FP16 RMSNorm -- the WHOLE pipeline in ONE kernel.
//
// out[r, :] = x[r, :] / sqrt(mean_j x[r,j]^2). Same shape and args as softmax; the
// differences are the reduction (SUMSQ, not MAX), the scalar (inv_rms = 1/sqrt(Sxx/N)
// via the integer sqrt_f16 + recip_f16 -- no FPU on this core), and that the normalize
// reads x DIRECTLY: no exp, no sub-max, no TAP padding anywhere. N = cols is taken to be
// a power of two, so the mean is an exponent subtract.
//
//   rows == 1 : reduce(SUMSQ) -> [core] inv_rms -> map(LINEAR, a=inv_rms) [+ quant]
//   rows >  1 : reduce(SUMSQ) -> [core] per-row inv_rms -> bcast -> EW1(MUL) [+ quant]
//
// Unlike softmax, rmsnorm has nothing for EW0 to do: its only combine is the final
// multiply, and nothing precedes it.
// ==========================================================================
static inline uint32_t __snax_bingo_kernel_simd_rmsnorm(void *arg, uint32_t out_prec) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_rmsnorm_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_rmsnorm", "SIMD");
#if !BINGO_HAS_STREAMMAP || !BINGO_HAS_STREAMREDUCE || !BINGO_HAS_STREAMELEMENTWISE
    (void)out_prec;
    BINGO_SIMD_EXT_UNSUPPORTED("simd_rmsnorm",
                               "StreamMap+StreamReduce+StreamElementwise");
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t *a = (uint32_t *)arg;
    uint64_t in_addr = make_u64(a[0], a[1]);
    uint64_t out_addr = make_u64(a[2], a[3]);
    uint32_t rows = a[4];
    uint32_t cols = a[5];
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_rmsnorm_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    uint32_t out_i8 = (out_prec == SIMD_OUT_I8);
    if (out_i8 && !BINGO_HAS_FP16TOINT8)
        BINGO_SIMD_EXT_UNSUPPORTED("simd_rmsnorm_f16_i8", "Fp16ToInt8");
    BINGO_SIMD_REQUIRE_LOCAL(in_addr, "simd_rmsnorm", "input");
    BINGO_SIMD_REQUIRE_LOCAL(out_addr, "simd_rmsnorm", "output");

    uint32_t out_dt = out_i8 ? SIMD_OUT_I8 : SIMD_OUT_F16;
    uint32_t beats = cols >> 5u;
    uint32_t inv_scale = out_i8 ? BINGO_SIMD_I8_SCALE_NORM : 0u;  // out ~[-2,2]
    uint32_t row_b = beats * SIMD_BEAT_BYTES;
    uint32_t tot_b = rows * row_b;
    uint32_t log2D = 0u;
    for (uint32_t t = cols; t > 1u; t >>= 1u) log2D++;

    // Scratch: [ bt (rows scalar beats) | bc (broadcast, packed) ].
    uint32_t bt_off = 0u, bc_off = rows * SIMD_BEAT_BYTES;
    uint32_t scratch_bytes = bc_off + tot_b + 64u;
    static uint32_t s_pool = 0u;
    uint32_t scratch_lo, from_pool = 0u;
    if (scratch_bytes <= BINGO_SIMD_SCRATCH_POOL) {
        if (!s_pool) s_pool = snrt_l1_malloc(BINGO_SIMD_SCRATCH_POOL);
        scratch_lo = s_pool;
        from_pool = 1u;
    } else {
        scratch_lo = snrt_l1_malloc(scratch_bytes);
    }
    if (!scratch_lo) {
        printf_safe("[Cluster %d Core %d]: rmsnorm L1 scratch alloc failed!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    uint32_t base_lo = (scratch_lo + 63u) & ~63u;
    void *bt = (void *)(base_lo + bt_off);
    void *bc = (void *)(base_lo + bc_off);
    volatile uint16_t *bt_l = (volatile uint16_t *)bt;

    uint32_t rc;

    // reduce(SUMSQ): x -> bt.
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    rc = simd_pass_reduce((void *)(uint32_t)in_addr, bt, rows, beats, SIMD_RED_SUMSQ,
                          rows);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);

    if (rows == 1u) {
        // [core] inv_rms = 1/sqrt(ssq/N). The mean is an exponent subtract because N is a
        // power of two; then the integer sqrt and reciprocal, widened to fp32 bits.
        BINGO_TRACE_MARKER(BINGO_TRACE_SCALAR_RUN_START);
        uint32_t a_bits = 0u;
        if (rc == BINGO_RET_SUCC) {
            uint16_t ssq = bt_l[0];
            uint32_t Es = (ssq >> 10) & 0x1Fu;
            uint16_t mean = (uint16_t)(((Es - log2D) << 10) | (ssq & 0x3FFu));
            a_bits = f16_to_f32bits(recip_f16(sqrt_f16(mean)));
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_SCALAR_RUN_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
        if (rc == BINGO_RET_SUCC)
            rc = simd_pass_map((void *)(uint32_t)in_addr, (void *)(uint32_t)out_addr,
                               rows * beats, a_bits, 0u, SIMD_FUNC_LINEAR, out_dt,
                               inv_scale);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    } else {
        BINGO_TRACE_MARKER(BINGO_TRACE_SCALAR_RUN_START);
        if (rc == BINGO_RET_SUCC) {
            for (uint32_t r = 0u; r < rows; r++) {
                uint16_t ssq = bt_l[r * 32u];
                uint32_t Es = (ssq >> 10) & 0x1Fu;
                uint16_t mean = (uint16_t)(((Es - log2D) << 10) | (ssq & 0x3FFu));
                uint16_t inv = recip_f16(sqrt_f16(mean));
                uint32_t inv2 = ((uint32_t)inv << 16) | inv;
                volatile uint32_t *row32 = (volatile uint32_t *)(bt_l + r * 32u);
                for (uint32_t l = 0u; l < 16u; l++) row32[l] = inv2;
            }
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_SCALAR_RUN_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
        if (rc == BINGO_RET_SUCC)
            rc = simd_pass_bcast_map(bt, bc, rows, beats, row_b, SIMD_F32_ONE);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
        if (rc == BINGO_RET_SUCC)
            rc = simd_pass_ew2((void *)(uint32_t)in_addr, bc, (void *)(uint32_t)out_addr,
                               rows, beats, row_b, SIMD_EXT_STREAMELEMENTWISE_1,
                               SIMD_EXT_STREAMELEMENTWISE_1_CSR, SIMD_EW_MUL, out_dt,
                               inv_scale);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    }

    if (!from_pool) snrt_l1_free(scratch_lo);
    if (rc != BINGO_RET_SUCC) {
        printf_safe("[Cluster %d Core %d]: rmsnorm pass failed!\r\n", snrt_cluster_idx(),
                    snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    sp->return_value = (uint32_t)out_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_rmsnorm_f16_f16(void *arg) {
    return __snax_bingo_kernel_simd_rmsnorm(arg, SIMD_OUT_F16);
}
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_rmsnorm_f16_i8(void *arg) {
    return __snax_bingo_kernel_simd_rmsnorm(arg, SIMD_OUT_I8);
}

// ==========================================================================
// Fused FP16 SiLU -- one StreamMap pass: out = silu(x) = x*sigmoid(x), elementwise over
// [rows, cols]. The user gives { input, output, rows, cols }; the writer dtype is picked
// by the wrapper name.
// ==========================================================================
static inline uint32_t __snax_bingo_kernel_simd_silu(void *arg, uint32_t out_prec) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_silu_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_silu", "SIMD");
#if !BINGO_HAS_STREAMMAP
    (void)out_prec;
    BINGO_SIMD_EXT_UNSUPPORTED("simd_silu", "StreamMap");
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t *a = (uint32_t *)arg;
    uint64_t in_addr = make_u64(a[0], a[1]);
    uint64_t out_addr = make_u64(a[2], a[3]);
    uint32_t rows = a[4];
    uint32_t cols = a[5];
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_silu_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    uint32_t out_i8 = (out_prec == SIMD_OUT_I8);
    if (out_i8 && !BINGO_HAS_FP16TOINT8)
        BINGO_SIMD_EXT_UNSUPPORTED("simd_silu_f16_i8", "Fp16ToInt8");
    BINGO_SIMD_REQUIRE_LOCAL(in_addr, "simd_silu", "input");
    BINGO_SIMD_REQUIRE_LOCAL(out_addr, "simd_silu", "output");

    uint32_t beats = cols >> 5u;
    uint32_t inv_scale = out_i8 ? BINGO_SIMD_I8_SCALE_ACT : 0u;
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint32_t rc = simd_pass_map((void *)(uint32_t)in_addr, (void *)(uint32_t)out_addr,
                                rows * beats, SIMD_F32_ONE, 0u, SIMD_FUNC_SILU,
                                out_i8 ? SIMD_OUT_I8 : SIMD_OUT_F16, inv_scale);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    if (rc != BINGO_RET_SUCC) {
        printf_safe("[Cluster %d Core %d]: silu StreamMap pass failed!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    sp->return_value = (uint32_t)out_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_silu_f16_f16(void *arg) {
    return __snax_bingo_kernel_simd_silu(arg, SIMD_OUT_F16);
}
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_silu_f16_i8(void *arg) {
    return __snax_bingo_kernel_simd_silu(arg, SIMD_OUT_I8);
}

// ==========================================================================
// Fused FP16 SwiGLU: out = silu(gate) * up, elementwise over [rows, cols].
//
//   sg  = silu(gate)      StreamMap SILU,        gate  -> sg scratch
//   out = sg (.) up       StreamElementwise MUL, sg,up -> out
//
// TWO passes, and the second one is what EW1 is for: SwiGLU is silu(a)*b, so the
// multiply FOLLOWS the map. It cannot collapse into one chain, because the two operands
// enter at different points -- `gate` through the map, `up` only at the combine -- while
// the chain reads ONE interleaved stream. Fusing it would need the map to apply to just
// one of the two interleaved operands, which the datapath does not express.
// ==========================================================================
static inline uint32_t __snax_bingo_kernel_simd_swiglu(void *arg, uint32_t out_prec) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_swiglu_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_swiglu", "SIMD");
#if !BINGO_HAS_STREAMMAP || !BINGO_HAS_STREAMELEMENTWISE
    (void)out_prec;
    BINGO_SIMD_EXT_UNSUPPORTED("simd_swiglu", "StreamMap+StreamElementwise");
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t *a = (uint32_t *)arg;
    uint64_t gate_addr = make_u64(a[0], a[1]);
    uint64_t up_addr = make_u64(a[2], a[3]);
    uint64_t out_addr = make_u64(a[4], a[5]);
    uint32_t rows = a[6];
    uint32_t cols = a[7];
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_swiglu_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    uint32_t out_i8 = (out_prec == SIMD_OUT_I8);
    if (out_i8 && !BINGO_HAS_FP16TOINT8)
        BINGO_SIMD_EXT_UNSUPPORTED("simd_swiglu_f16_i8", "Fp16ToInt8");
    BINGO_SIMD_REQUIRE_LOCAL(gate_addr, "simd_swiglu", "gate");
    BINGO_SIMD_REQUIRE_LOCAL(up_addr, "simd_swiglu", "up");
    BINGO_SIMD_REQUIRE_LOCAL(out_addr, "simd_swiglu", "output");

    uint32_t beats = cols >> 5u;
    uint32_t row_b = beats * SIMD_BEAT_BYTES;
    uint32_t tot_b = rows * row_b;
    uint32_t inv_scale = out_i8 ? BINGO_SIMD_I8_SCALE_ACT : 0u;

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_START);
    uint32_t scratch_bytes = tot_b + 64u;
    static uint32_t s_pool = 0u;
    uint32_t scratch_lo, from_pool = 0u;
    if (scratch_bytes <= BINGO_SIMD_SCRATCH_POOL) {
        if (!s_pool) s_pool = snrt_l1_malloc(BINGO_SIMD_SCRATCH_POOL);
        scratch_lo = s_pool;
        from_pool = 1u;
    } else {
        scratch_lo = snrt_l1_malloc(scratch_bytes);
    }
    if (!scratch_lo) {
        printf_safe("[Cluster %d Core %d]: swiglu L1 scratch alloc failed!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    void *sg = (void *)((scratch_lo + 63u) & ~63u);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_END);

    // 1) sg = silu(gate).
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint32_t rc = simd_pass_map((void *)(uint32_t)gate_addr, sg, rows * beats,
                                SIMD_F32_ONE, 0u, SIMD_FUNC_SILU, SIMD_OUT_F16, 0u);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    // 2) out = sg (.) up [+ quant]. Both operands packed, so the row stride is row_b.
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    if (rc == BINGO_RET_SUCC)
        rc = simd_pass_ew2(sg, (void *)(uint32_t)up_addr, (void *)(uint32_t)out_addr,
                           rows, beats, row_b, SIMD_EXT_STREAMELEMENTWISE_1,
                           SIMD_EXT_STREAMELEMENTWISE_1_CSR, SIMD_EW_MUL,
                           out_i8 ? SIMD_OUT_I8 : SIMD_OUT_F16, inv_scale);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);

    if (!from_pool) snrt_l1_free(scratch_lo);
    if (rc != BINGO_RET_SUCC) {
        printf_safe("[Cluster %d Core %d]: swiglu pass failed!\r\n", snrt_cluster_idx(),
                    snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    sp->return_value = (uint32_t)out_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_swiglu_f16_f16(void *arg) {
    return __snax_bingo_kernel_simd_swiglu(arg, SIMD_OUT_F16);
}
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_swiglu_f16_i8(void *arg) {
    return __snax_bingo_kernel_simd_swiglu(arg, SIMD_OUT_I8);
}

// ==========================================================================
// Fused FP16 RoPE (interleaved / complex-rotation convention):
//   xswap = adjacent fp16-pair swap of x
//   out   = x (.) cos_full  +  xswap (.) sin_signed
//
// cos_full (cos duplicated per pair) and sin_signed (sin with alternating sign) are
// PRECOMPUTED tables -- the sign flip is not a datapath op. x is the runtime Q/K, so the
// swap has to be computed on device, which is what makes in-layer rope_q / rope_k
// possible at all.
//
// THE SWAP RUNS ON THE CORE, not on a DMA. Only hart 3 has the DMA ISA -- the cluster
// asserts that at most one core carries it -- so a dm* instruction here traps with an
// illegal instruction. The swap is therefore plain loads and stores: a [rows*cols/2]
// halfword shuffle, which is real work on a scalar core. If RoPE ever lands on the
// critical path, split it into an xDMA-core node that produces xswap and a SIMD-core
// node that does the two multiplies and the add.
// ==========================================================================
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_rope(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_rope_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_rope", "SIMD");
#if !BINGO_HAS_STREAMELEMENTWISE
    BINGO_SIMD_EXT_UNSUPPORTED("simd_rope", "StreamElementwise");
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t *a = (uint32_t *)arg;
    uint64_t x_addr = make_u64(a[0], a[1]);
    uint64_t cos_addr = make_u64(a[2], a[3]);
    uint64_t sin_addr = make_u64(a[4], a[5]);
    uint64_t out_addr = make_u64(a[6], a[7]);
    uint32_t cols = a[8];
    uint32_t rows = a[9];
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_rope_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_SIMD_REQUIRE_LOCAL(x_addr, "simd_rope", "x");
    BINGO_SIMD_REQUIRE_LOCAL(cos_addr, "simd_rope", "cos");
    BINGO_SIMD_REQUIRE_LOCAL(sin_addr, "simd_rope", "sin");
    BINGO_SIMD_REQUIRE_LOCAL(out_addr, "simd_rope", "out");

    uint32_t beats = cols >> 5u;
    uint32_t row_beats = rows * beats;
    uint32_t tot_b = row_beats * SIMD_BEAT_BYTES;
    uint32_t num_elems = row_beats * 32u;
    uint32_t row_b = beats * SIMD_BEAT_BYTES;

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_START);
    // Scratch [ xswap | tmp1 | tmp2 ], 64-byte aligned for the beat reads.
    uint32_t scratch_bytes = 3u * tot_b + 64u;
    static uint32_t s_pool = 0u;
    uint32_t scratch_lo, from_pool = 0u;
    if (scratch_bytes <= BINGO_SIMD_SCRATCH_POOL) {
        if (!s_pool) s_pool = snrt_l1_malloc(BINGO_SIMD_SCRATCH_POOL);
        scratch_lo = s_pool;
        from_pool = 1u;
    } else {
        scratch_lo = snrt_l1_malloc(scratch_bytes);
    }
    if (!scratch_lo) {
        printf_safe("[Cluster %d Core %d]: rope L1 scratch alloc failed!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    uint32_t base_lo = (scratch_lo + 63u) & ~63u;
    void *xswap = (void *)base_lo;
    void *tmp1 = (void *)(base_lo + tot_b);
    void *tmp2 = (void *)(base_lo + 2u * tot_b);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_END);

    // xswap: exchange each adjacent fp16 pair. Word-at-a-time, which is exactly one
    // halfword rotate per 32-bit word -- the pairs are aligned, so no cross-word case.
    BINGO_TRACE_MARKER(BINGO_TRACE_SCALAR_RUN_START);
    {
        const volatile uint32_t *src = (const volatile uint32_t *)(uint32_t)x_addr;
        volatile uint32_t *dst = (volatile uint32_t *)xswap;
        uint32_t words = num_elems >> 1u;
        for (uint32_t i = 0u; i < words; i++) {
            uint32_t w = src[i];
            dst[i] = (w >> 16) | (w << 16);
        }
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_SCALAR_RUN_END);

    // tmp1 = x (.) cos_full
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint32_t rc = simd_pass_ew2((void *)(uint32_t)x_addr, (void *)(uint32_t)cos_addr,
                                tmp1, rows, beats, row_b, SIMD_EXT_STREAMELEMENTWISE_1,
                                SIMD_EXT_STREAMELEMENTWISE_1_CSR, SIMD_EW_MUL,
                                SIMD_OUT_F16, 0u);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    // tmp2 = xswap (.) sin_signed
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    if (rc == BINGO_RET_SUCC)
        rc = simd_pass_ew2(xswap, (void *)(uint32_t)sin_addr, tmp2, rows, beats, row_b,
                           SIMD_EXT_STREAMELEMENTWISE_1,
                           SIMD_EXT_STREAMELEMENTWISE_1_CSR, SIMD_EW_MUL, SIMD_OUT_F16,
                           0u);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    // out = tmp1 (+) tmp2
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    if (rc == BINGO_RET_SUCC)
        rc = simd_pass_ew2(tmp1, tmp2, (void *)(uint32_t)out_addr, rows, beats, row_b,
                           SIMD_EXT_STREAMELEMENTWISE_1,
                           SIMD_EXT_STREAMELEMENTWISE_1_CSR, SIMD_EW_ADD, SIMD_OUT_F16,
                           0u);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);

    if (!from_pool) snrt_l1_free(scratch_lo);
    if (rc != BINGO_RET_SUCC) {
        printf_safe("[Cluster %d Core %d]: rope pass failed!\r\n", snrt_cluster_idx(),
                    snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    sp->return_value = (uint32_t)out_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}
