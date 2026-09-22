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

// Drain the SIMD block before the fused exp pass reads the -m_new prefix.
//
// DEFAULT 0: this was TRIED AND IT DOES NOT WORK. Blocking the SIMD core mid-softmax
// perturbs timing into the known 4-issuer xDMA deadlock -- at NSCORE=3 the read-stall and
// wide-send watchdogs fire on clusters 3 and 1 (xdma_stall_watchdog.sv:75, ~4.16-4.18 ms),
// and at the shipping NSCORE=2 the run produces no checks at all. Kept behind the flag as a
// record of a negative result, not as a fix.
//
// A correct fix has to establish the prefix WITHOUT stalling the SIMD core -- e.g. seed each
// s16 prefix from the idle xDMA with a dependency edge into the FIRST softmax that uses that
// buffer. That is not the same as the setup-time pre-fill, which has no such edge and is
// simply overwritten (and, filled with 0 or NEG_INF16, corrupts the result: 33x and 0.750x).
#ifndef BINGO_FA_DRAIN_BEFORE_FUSED
#define BINGO_FA_DRAIN_BEFORE_FUSED 0
#endif

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
        snax_simd_arm2(SIMD_EXT_FP16TOINT8, SIMD_EXT_FP16TOINT8_CSR, inv_scale,
                       SIMD_QUANT_TAIL(0));
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
        snax_simd_arm2(SIMD_EXT_FP16TOINT8, SIMD_EXT_FP16TOINT8_CSR, inv_scale,
                       SIMD_QUANT_TAIL(0));
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
// SIZED AGAINST THE LARGEST TILE THE SWEEP RUNS, not picked round.
//
// The fused softmax is the hungriest: it needs the per-row scalar beats plus TWO
// TAP-padded copies of the tile,
//
//     rows*64  +  2 * rows*(beats+1)*64  +  64
//
// which at the sweep's largest point (rows = 8, cols = 256, so beats = 8) is 9792 B. A
// tile that does not fit falls out of the pool onto a per-call snrt_l1_malloc/free, and
// that allocator costs several times the operator chain it wraps.
//
// This is a POOL, not a bound: a tile larger than it still works, it just pays the
// allocator. Raising it further is cheap L1 but not free, so it tracks the sweep grid
// rather than the largest tile imaginable.
#define BINGO_SIMD_SCRATCH_POOL 16384u

// How many experts one MoE combine can index. Bounds a stack array, so it is a cap on
// the kernel, not on the graph: a fork with more branches than this still compiles and
// still skips correctly in hardware -- the combine just refuses to fold them.
#define BINGO_MOE_MAX_EXPERTS 32u

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
// MoE combine: out = SUM over selected e of weight[e] * y[e], elementwise.
//
//   acc = w[a0] * y[a0]                 StreamMap LINEAR
//   acc = acc + w[aj] * y[aj]           StreamMap LINEAR, then StreamElementwise ADD
//
// The weights are FP32 BIT PATTERNS, loaded and handed to the map's scale CSR
// without being interpreted. That is the whole reason this runs on a hart with no
// FPU: the renormalising divide already happened on the host, in the gating kernel,
// and what arrives here is the answer.
//
// Only the experts the gate selected are read. A loser's landing slot is never
// written by anyone -- its whole branch was skipped -- so touching it would fold in
// whatever the last dispatch left there.
//
// The accumulator PING-PONGS between two scratch buffers instead of accumulating in
// place: simd_pass_ew2 reads a 3-D interleave of its two operands and writes a packed
// 1-D stream, so dst == src_b is not a safe aliasing. Alternating also lets the LAST
// add write straight to out_addr, so the caller never has to ask where the answer is.
// k selected experts cost 2k - 1 passes.
// ==========================================================================
static inline uint32_t __snax_bingo_kernel_simd_moe_combine(void *arg,
                                                            uint32_t out_prec) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_moe_combine_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_moe_combine", "SIMD");
#if !BINGO_HAS_STREAMMAP || !BINGO_HAS_STREAMELEMENTWISE
    (void)out_prec;
    BINGO_SIMD_EXT_UNSUPPORTED("simd_moe_combine", "StreamMap+StreamElementwise");
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t *a = (uint32_t *)arg;
    uint64_t out_addr = make_u64(a[0], a[1]);
    uint64_t src_base = make_u64(a[2], a[3]);
    uint32_t src_stride = a[4];
    uint32_t num_inputs = a[5];
    uint32_t rows = a[6];
    uint32_t cols = a[7];
    uint64_t act_addr = make_u64(a[8], a[9]);
    uint64_t weight_addr = make_u64(a[10], a[11]);
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_moe_combine_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    (void)out_prec;
    BINGO_SIMD_REQUIRE_LOCAL(out_addr, "simd_moe_combine", "output");
    BINGO_SIMD_REQUIRE_LOCAL(src_base, "simd_moe_combine", "operand base");
    if (num_inputs)
        BINGO_SIMD_REQUIRE_LOCAL(src_base + (uint64_t)(num_inputs - 1u) * src_stride,
                                 "simd_moe_combine", "last operand slot");

    uint32_t beats = cols >> 5u;
    uint32_t row_b = beats * SIMD_BEAT_BYTES;
    uint32_t tot_b = rows * row_b;

    // Which experts ran. Scalar byte loads, so the array may sit in L3 next to the
    // weights; this is the only place the routing decision is read.
    const uint8_t *act = (const uint8_t *)(uint32_t)act_addr;
    const uint32_t *wbits = (const uint32_t *)(uint32_t)weight_addr;
    uint32_t sel[BINGO_MOE_MAX_EXPERTS];
    uint32_t n_act = 0u;
    for (uint32_t e = 0; e < num_inputs && n_act < BINGO_MOE_MAX_EXPERTS; e++)
        if (act[e]) sel[n_act++] = e;

    if (n_act == 0u) {
        // top_k >= 1 makes this impossible, so it means the gating kernel never
        // ran or wrote somewhere else. Silence here would look like a zero output.
        printf_safe("[Cluster %d Core %d]: moe_combine got NO active expert of "
                    "%d -- gating never published its decision\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), num_inputs);
        return BINGO_RET_FAIL;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_START);
    // Two accumulators plus one staging buffer for the operand being scaled.
    uint32_t scratch_bytes = 3u * (tot_b + 64u);
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
        printf_safe("[Cluster %d Core %d]: moe_combine L1 scratch alloc failed!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    uint32_t base = (scratch_lo + 63u) & ~63u;
    uint32_t pad = (tot_b + 63u) & ~63u;
    void *acc_buf[2] = {(void *)base, (void *)(base + pad)};
    void *stage = (void *)(base + 2u * pad);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_END);

    void *out = (void *)(uint32_t)out_addr;
    #define MOE_SRC(e) ((void *)((uint32_t)src_base + (e) * src_stride))

    // First winner: scale it into the accumulator, or straight out when it is alone.
    void *acc = (n_act == 1u) ? out : acc_buf[0];
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint32_t rc = simd_pass_map(MOE_SRC(sel[0]), acc, rows * beats,
                                wbits[sel[0]], 0u, SIMD_FUNC_LINEAR,
                                SIMD_OUT_F16, 0u);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);

    for (uint32_t j = 1u; j < n_act && rc == BINGO_RET_SUCC; j++) {
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
        rc = simd_pass_map(MOE_SRC(sel[j]), stage, rows * beats,
                           wbits[sel[j]], 0u, SIMD_FUNC_LINEAR, SIMD_OUT_F16, 0u);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
        if (rc != BINGO_RET_SUCC) break;
        // The last add lands in out_addr; the others alternate accumulators so the
        // destination is never also an operand.
        void *dst = (j == n_act - 1u)
                        ? out
                        : ((acc == acc_buf[0]) ? acc_buf[1] : acc_buf[0]);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
        rc = simd_pass_ew2(acc, stage, dst, rows, beats, row_b,
                           SIMD_EXT_STREAMELEMENTWISE_1,
                           SIMD_EXT_STREAMELEMENTWISE_1_CSR, SIMD_EW_ADD,
                           SIMD_OUT_F16, 0u);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
        acc = dst;
    }
    #undef MOE_SRC

    if (!from_pool) snrt_l1_free(scratch_lo);
    if (rc != BINGO_RET_SUCC) {
        printf_safe("[Cluster %d Core %d]: moe_combine pass failed!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    sp->return_value = (uint32_t)out_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_moe_combine_f16(void *arg) {
    return __snax_bingo_kernel_simd_moe_combine(arg, SIMD_OUT_F16);
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

// ==========================================================================
// FlashAttention online-softmax epilogue -- the WHOLE per-tile SIMD half in ONE kernel.
//
// One KV tile of FlashAttention is three engine dispatches:
//
//     GEMM   S16^T = K8 . Q8^T                     [Bc, Br] fp16
//     SIMD   the online softmax, and P8^T = int8(P)          <- THIS KERNEL
//     GEMM   O32^T += V8^T . P8^T
//
// The SIMD half is ELEVEN accelerator tasks. They are fired back to back into the
// block's 2-deep task queue with no wait between them, so task i's engine time overlaps
// task i+1's CSR programming; only the last one is waited on. Making them eleven BINGO
// nodes instead would destroy exactly that -- each would have to retire before the next
// was released -- and would pay eleven dispatches for what is one indivisible state
// update. Hence one kernel.
//
// THE RECURRENCE. Per query row, carried across KV tiles in `arena`:
//     m_new = max(m_old, rowmax(S))
//     corr  = exp(m_old - m_new)          how much the PAST must shrink
//     P     = exp(S - m_new)
//     l_new = corr * l_old + rowsum(P)
//     O    *= corr                        the GEMM adds P.V on top
// Nothing is approximated: re-basing l and O onto the new maximum makes the tiled answer
// the full-matrix answer.
//
// EVERYTHING IS TRANSPOSED, and that is what makes this cheap. S is stored [Bc, Br], so
// one 64-byte beat holds one score per QUERY ROW -- lane k is query row k. The softmax
// reduces per query row, which now runs ALONG BEATS, so SIMD_RED_LANEWISE returns it
// straight out of the accumulator the engine already carries: no horizontal fold, no
// treeBuf serialisation, no scalar drain, and m is ONE beat rather than Br scalars.
// Br is therefore fixed at 32 = SIMD_BEAT_BYTES/2 lanes and is not a parameter.
//
// ADJACENCY IS THE LAYOUT RULE. A task pairs operands by reading one flat stream, so
// every pair below must be physically adjacent, and the arena is ONE contiguous block
// for that reason alone. Moving a field silently feeds a task the wrong beat rather
// than faulting:
//
//   [ negmS | s16 .. Bc .. | rmax | mrun | mnew | delta | corrL | lrun | lnew |
//     p16 .. Bc .. | rsum | lsc | corrO | oacc .. d .. ]
//     ^^^^^^^^^^^^^^^^^^^^^^^^^                          the tap writes Bc+1 beats from
//     |             |        |                           s16, so rowmax LANDS in rmax
//     |             |        [rmax][mrun] -> m_new, and later [-m_new][mrun] -> delta
//     |             sticky-B latch for exp(S - m_new): one flat read of 1+Bc beats
//     [corrL][lrun] -> corr*l_old   [rsum][lsc] -> l_new   [corrO][oacc] -> O *= corr
//
// WHAT THE CALLER OWES US. `s16_src` is the GEMM's D32 output for this tile, Bc fp16
// beats -- the Int32ToFp16Converter on that port means it arrives as fp16 and at HALF
// the beats an INT32 tile would need. `p8_dst` receives Bc/2 beats. Both are ordinary
// BINGO L1 allocations; the BINGO edge from the producing GEMM node is what guarantees
// s16_src is complete, which is why this kernel needs no sync counters and no barrier.
//
// Arg layout (__snax_bingo_kernel_simd_fa_softmax_args_t):
//   [0..1] s16_src   (GEMM D32 output this tile, fp16 [Bc, 32])
//   [2..3] p8_dst    (int8 P^T for the next GEMM, Bc/2 beats)
//   [4..5] arena     (the contiguous block above; see SIMD_FA_ARENA_BYTES)
//   [6]    bc        (keys in this tile = beats of S)
//   [7]    dhead     (head dimension = beats of O)
//   [8]    tile_idx  (0 seeds m, l and O; every later tile carries them forward)
// ==========================================================================

// Shapes are cached in the arena rather than rebuilt per call. Building twelve of them
// is several hundred instructions, which against a ~2400-cycle tile is not noise; and
// they live in the ARENA rather than in a static, so they are L1 by construction and
// there is no question about where a .bss object lands.
//
// The MEMO THAT GUARDS THEM has to live there too -- see SIMD_FA_MEMO_OFF below. Putting
// the data in L1 while leaving its key in .bss costs more than rebuilding the shapes.
#define SIMD_FA_NUM_SHAPES 12
// A FIXED reservation, deliberately NOT sizeof(snax_simd_shape_t) * N.
//
// The struct carries SIMD_MAX_DIM bounds and strides, and SIMD_MAX_DIM is the cluster's
// reader_agu_temporal_dimension -- so its size is a property of the hjson. Deriving the
// reservation from it would make every DATA offset in the arena depend on the cfg too,
// and the host has to know those offsets: SnaxBingoKernelSimdFaSoftmaxArgs.layout() is
// how a workload reads the recurrence back (`arena.view(layout(bc, d)["mrun"])`). With a
// fixed block the data layout is a function of (bc, dhead) alone, which both sides
// already agree on exactly.
//
// Keep this in step with SnaxBingoKernelSimdFaSoftmaxArgs.SHAPE_BYTES.
#define SIMD_FA_SHAPES_BYTES 1024u

// The geometry memo: "the shapes in THIS arena are already built, for this (bc, dhead)".
//
// It lives in the arena's shape reservation -- i.e. in L1 -- and deliberately NOT in a
// function-scope static. The device image links entirely into L3 and this core has no data
// cache, so a `static` read here is a blocking fabric round trip, an order of magnitude
// worse again while the other clusters stream K/V. That is a large share of
// BINGO_TRACE_SIMD_CFG for twelve bytes that never need to leave the cluster.
//
// Keying it on the arena is also required for correctness: with NQ query tiles the arena
// ALTERNATES every invocation, so one shared static could never hold the right answer and
// the memo would miss every time from NQ=2 onwards. Per
// arena, it hits every tile but the first.
#define SIMD_FA_MEMO_OFF   (SIMD_FA_SHAPES_BYTES - 16u)
#define SIMD_FA_MEMO_MAGIC 0x5AFA5117u

// a[10], the geometry mode. SELF is the default and the only value a workload that has no
// prologue node may use: it rebuilds at tile 0 unconditionally, so uninitialised L1 can
// never be mistaken for a built geometry. PROLOGUE builds and returns. PRIMED says "a
// PROLOGUE node for THIS arena is an ancestor of this node in the graph", which is what
// makes it safe to trust the memo on the very first tile.
#define SIMD_FA_GEOM_SELF     0u
#define SIMD_FA_GEOM_PROLOGUE 1u
#define SIMD_FA_GEOM_PRIMED   2u
// Like PRIMED, and ALSO: the block's shared CSRs still hold a same-geometry softmax
// program, so snax_simd_program_fast() can be skipped and only the six per-task CSRs
// written. Only the host may assert this -- it is the one that knows what else ran on
// this core. See the geom_mode comment in main_bingo.py.
#define SIMD_FA_GEOM_CSR_PRIMED 3u

typedef struct {
    uint32_t magic, bc, dhead;
} simd_fa_memo_t;

// Total arena bytes for a (bc, dhead) tile. The host allocates exactly this.
//   negmS 1 | s16 bc | rmax 1 | mrun 1 | mnew 1 | delta 1 | corrL 1 | lrun 1 | lnew 1 |
//   p16 bc | rsum 1 | lsc 1 | corrO 1 | oacc dhead      =  2*bc + dhead + 11 beats
#define SIMD_FA_ARENA_BEATS(bc, dhead) (2u * (bc) + (dhead) + 11u)
#define SIMD_FA_ARENA_BYTES(bc, dhead) \
    (SIMD_FA_SHAPES_BYTES + SIMD_FA_ARENA_BEATS(bc, dhead) * SIMD_BEAT_BYTES)

// Shape slots, named so the launch sequence below reads as the recurrence does.
enum {
    FA_SH_TAP_IN = 0, FA_SH_TAP_OUT,      // 1     rowmax, tile passed through
    FA_SH_MNEW_IN,    FA_SH_MNEW_OUT,     // 2     m_new = max(m_old, rowmax)
    FA_SH_NEGM_IN,    FA_SH_NEGM_OUT,     // 3+4   -m_new, fanned out to both latches
    FA_SH_DELTA_IN,   FA_SH_DELTA_OUT,    // 5     delta = m_old - m_new
    FA_SH_CORR_IN,    FA_SH_CORR_OUT,     // 6+7   corr = exp(delta), to both latches
    FA_SH_P_IN,       FA_SH_P_OUT,        // 8+9   P = exp(S - m_new) AND rowsum, fused
    FA_SH_LSC_IN,     FA_SH_LSC_OUT,      // 11    corr * l_old
    FA_SH_LNEW_IN,    FA_SH_LNEW_OUT,     // 13    l_new = corr*l_old + rowsum
    FA_SH_ORS_IN,     FA_SH_ORS_OUT,      // 14    O *= corr
    FA_SH_CMT_IN,     FA_SH_CMT_OUT,      // 15+16 commit m, l for the next tile
    FA_SH_COUNT
};

#if BINGO_HAS_PREMAP_ELEMENTWISE && BINGO_HAS_STREAMMAP && \
    BINGO_HAS_STREAMREDUCE && BINGO_HAS_STREAMELEMENTWISE && BINGO_HAS_FP16TOINT8

// The arena layout, derived in ONE place. Both the shape builder and the seeding below
// take their pointers from here, because two copies of this arithmetic is exactly how a
// layout silently drifts -- and a drifted layout does not fault, it feeds a task the
// beat next door.
// FA_SH_COUNT, not SIMD_FA_NUM_SHAPES: the enum carries an IN and an OUT per task, so the
// real occupancy is 20 shapes (880 B at SIMD_MAX_DIM=3), not 12. The bound is the MEMO
// offset rather than the whole reservation, because the memo sits at its tail and an
// overrun into it would be silent.
_Static_assert(FA_SH_COUNT * sizeof(snax_simd_shape_t) <= SIMD_FA_MEMO_OFF,
               "the FA shape cache outgrew its fixed reservation; raise "
               "SIMD_FA_SHAPES_BYTES here AND SHAPE_BYTES in bingo_kernel_args.py");

typedef struct {
    uint8_t *rmax, *mrun, *mnew, *delta;
    uint8_t *corrL, *lrun, *lnew, *rsum, *lsc, *corrO, *oacc;
} simd_fa_layout_t;

static void simd_fa_layout(simd_fa_layout_t *L, uint32_t arena, uint32_t bc,
                           uint32_t dhead) {
    const uint32_t B = SIMD_BEAT_BYTES;
    uint32_t t = arena + SIMD_FA_SHAPES_BYTES;
    L->rmax  = (uint8_t *)t;  t += B;   // the tap lands the rowmax exactly here
    L->mrun  = (uint8_t *)t;  t += B;
    L->mnew  = (uint8_t *)t;  t += B;
    L->delta = (uint8_t *)t;  t += B;
    L->corrL = (uint8_t *)t;  t += B;
    L->lrun  = (uint8_t *)t;  t += B;
    L->lnew  = (uint8_t *)t;  t += B;
    L->rsum  = (uint8_t *)t;  t += B;   // and the rowsum exactly here
    L->lsc   = (uint8_t *)t;  t += B;
    L->corrO = (uint8_t *)t;  t += B;
    L->oacc  = (uint8_t *)t;  t += dhead * B;
    (void)t;
}

// Build every task geometry. Called once per (arena, bc, dhead); the only things that
// change per tile are the two bases re-pointed in the kernel.
static void simd_fa_build_shapes(uint32_t arena, uint32_t bc, uint32_t dhead) {
    snax_simd_shape_t *sh = (snax_simd_shape_t *)arena;
    simd_fa_layout_t L;
    simd_fa_layout(&L, arena, bc, dhead);

    // 1  rowmax in one pass: TAP passes the tile through and appends the per-lane maxima.
    //    The input base is re-pointed per tile; `s16` is the kernel's own copy, which the
    //    fused pass below then reads with negmS latched in front of it.
    snax_simd_shape_flat(&sh[FA_SH_TAP_IN],  (void *)0, bc);
    // ONE beat out, not bc+1. Without SIMD_RED_TAP the reduce emits only its result, so
    // task 1 does not copy the tile into the arena: the fused exp pass reads the GEMM's own
    // output buffer, and the arena never carries a copy of the tile.
    snax_simd_shape_flat(&sh[FA_SH_TAP_OUT], L.rmax, 1);
    // 2  m_new = max(m_old, rowmax): LANEWISE over the adjacent pair [rmax][mrun].
    snax_simd_shape_flat(&sh[FA_SH_MNEW_IN],  L.rmax, 2);
    snax_simd_shape_flat(&sh[FA_SH_MNEW_OUT], L.mnew, 1);
    // 3+4  -m_new to BOTH latches in one task: read m_new twice at stride 0 and fan the
    //      negated value out, once before s16 and once into the rmax slot (task 2 has
    //      consumed it, so it now pairs with mrun). One fill+drain instead of two.
    snax_simd_shape_broadcast(&sh[FA_SH_NEGM_IN], L.mnew, 2);
    // -m_new goes to rmax (where task 5 reads it) and to the one-beat prefix in front of
    // the live score buffer (where the fused pass latches it). The second address changes
    // per tile, so stride[0] is patched at dispatch.
    snax_simd_shape_2d(&sh[FA_SH_NEGM_OUT], L.rmax, 2, 0, 1, 0);
    // 5  delta = m_old - m_new: LANEWISE ADD over [-m_new][m_old].
    snax_simd_shape_flat(&sh[FA_SH_DELTA_IN],  L.rmax, 2);
    snax_simd_shape_flat(&sh[FA_SH_DELTA_OUT], L.delta, 1);
    // 6+7  corr = exp(delta), fanned out the same way: corrL before l_old, corrO before O.
    snax_simd_shape_broadcast(&sh[FA_SH_CORR_IN], L.delta, 2);
    snax_simd_shape_2d(&sh[FA_SH_CORR_OUT], L.corrL, 2,
                       (uint32_t)(L.corrO - L.corrL), 1, 0);
    // 8+9  THE FUSED PASS. EW0 sticky-ADD latches -m_new and subtracts it from every beat,
    //      Map exponentiates, Reduce sums LANEWISE and TAPs the result:
    //        read [negmS][S^T x bc]  ->  write [P x bc][rowsum]
    //      S^T - m_new stays inside the chain and is never written out.
    // Base patched at dispatch to the live score buffer's prefix beat.
    snax_simd_shape_flat(&sh[FA_SH_P_IN],  (void *)0, 1u + bc);
    // base patched per dispatch to the live p8 buffer: bc/2 INT8 beats + 1 fp16 tail
    snax_simd_shape_flat(&sh[FA_SH_P_OUT], (void *)0, bc / 2u + 1u);
    // 11 corr * l_old: sticky MUL over [corrL][lrun]; the seed emits nothing.
    snax_simd_shape_flat(&sh[FA_SH_LSC_IN],  L.corrL, 2);
    snax_simd_shape_flat(&sh[FA_SH_LSC_OUT], L.lsc, 1);
    // 13 l_new = corr*l_old + rowsum: LANEWISE ADD over the adjacent pair [rsum][lsc].
    // [lsc][rsum]: the row sum lives in the p8 buffer's trailing beat, so this is a 2-beat
    // STRIDED read, not an adjacent pair. ADD is commutative, so taking lsc
    // first keeps the stride positive (the arena is allocated before the p8 buffers).
    snax_simd_shape_2d(&sh[FA_SH_LNEW_IN], L.lsc, 2, 0, 1, 0);
    snax_simd_shape_flat(&sh[FA_SH_LNEW_OUT], L.lnew, 1);
    // 14 O *= corr: sticky MUL, latch corrO then the O tile, written back in place past
    //    the consumed latch.
    snax_simd_shape_flat(&sh[FA_SH_ORS_IN],  L.corrO, 1u + dhead);
    snax_simd_shape_flat(&sh[FA_SH_ORS_OUT], L.oacc, dhead);
    // 15+16 commit. m_old and l_old must stay live all tile -- delta needs m_old after
    //    m_new exists, and corr*l_old needs l_old while l_new is forming -- so the new
    //    pair lives in separate beats and is copied over the old one last. Two sources
    //    and two destinations, each a constant stride apart, so one strided 2-beat task
    //    commits both. An identity map is how a copy is expressed here.
    snax_simd_shape_2d(&sh[FA_SH_CMT_IN],  L.mnew, 2, (uint32_t)(L.lnew - L.mnew), 1, 0);
    snax_simd_shape_2d(&sh[FA_SH_CMT_OUT], L.mrun, 2, (uint32_t)(L.lrun - L.mrun), 1, 0);
}

// Seed the recurrence. m starts at the most negative FINITE fp16 rather than -inf, so
// max(m, rowmax) is just rowmax and exp(m - m_new) underflows to 0 as it should, with no
// inf arithmetic anywhere near the exponential.
//
// O is zeroed with the core's own stores rather than an identity map, because a map
// computes a*x + b over the EXISTING contents: 0 * inf is NaN, and freshly allocated L1
// is not guaranteed to be anything. It costs dhead beats of stores once per query tile,
// not once per KV tile. If that ever matters, zero it from the host with the xDMA's
// memset instead -- the cfg has HasVerilogMemset.
// Word fill, unrolled eight wide.
//
// NOT `volatile`. A volatile destination forbids the compiler from unrolling or reordering
// anything: it emits store/increment/compare/branch per word, roughly four instructions for
// every four bytes, which over a whole O accumulator dominates everything else the kernel
// does. volatile is not what makes these stores visible to the accelerator either; the
// compiler barrier in the caller is.
//
// The tail loop is not dead code by accident: every count used here is a multiple of
// SIMD_BEAT_BYTES / 4, which is 16 on this cluster, so the unrolled body covers it
// exactly -- but that is a property of the current SIMD width, not a guarantee.
static inline void simd_fa_fill_u32(uint32_t *p, uint32_t v, uint32_t n) {
    uint32_t i = 0;
    for (; i + 8u <= n; i += 8u) {
        p[i + 0] = v; p[i + 1] = v; p[i + 2] = v; p[i + 3] = v;
        p[i + 4] = v; p[i + 5] = v; p[i + 6] = v; p[i + 7] = v;
    }
    for (; i < n; i++) p[i] = v;
}

static void simd_fa_init_state(uint32_t arena, uint32_t bc, uint32_t dhead) {
    simd_fa_layout_t L;
    simd_fa_layout(&L, arena, bc, dhead);
    simd_fa_fill_u32((uint32_t *)L.mrun, 0xFBFFFBFFu, SIMD_BEAT_BYTES / 4u);  // -65504
    simd_fa_fill_u32((uint32_t *)L.lrun, 0u, SIMD_BEAT_BYTES / 4u);           // l = 0
    // O is NOT zeroed here any more. It is dhead beats, and zeroing it with this core's
    // stores sat on the critical path with both engines idle behind it. The workload is
    // expected to hand that to an engine that is idle at the time -- the ArenaOzero node
    // in the FA graphs does it on the xDMA. A workload that calls this kernel WITHOUT
    // arranging that zero will read a stale O on its first KV tile.
    (void)dhead;
    // The stores must land before the first snax_simd_fire() reads the arena. This is
    // what volatile was standing in for, and it costs nothing.
    __asm__ volatile("" ::: "memory");
}

#endif  // the full chain

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_fa_softmax(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_fa_softmax_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_fa_softmax", "SIMD");
#if !(BINGO_HAS_PREMAP_ELEMENTWISE && BINGO_HAS_STREAMMAP && BINGO_HAS_STREAMREDUCE && \
      BINGO_HAS_STREAMELEMENTWISE && BINGO_HAS_FP16TOINT8)
    BINGO_SIMD_EXT_UNSUPPORTED(
        "simd_fa_softmax",
        "StreamElementwise_0+StreamMap+StreamReduce+StreamElementwise_1+Fp16ToInt8");
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t *a = (uint32_t *)arg;
    uint64_t s16_src = make_u64(a[0], a[1]);
    uint64_t p8_dst = make_u64(a[2], a[3]);
    uint64_t arena_a = make_u64(a[4], a[5]);
    uint32_t bc = a[6];
    uint32_t dhead = a[7];
    uint32_t tile_idx = a[8];
    uint32_t seed_state = a[9];
    uint32_t geom_mode = a[10];
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_fa_softmax_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_SIMD_REQUIRE_LOCAL(s16_src, "simd_fa_softmax", "s16_src");
    BINGO_SIMD_REQUIRE_LOCAL(p8_dst, "simd_fa_softmax", "p8_dst");
    BINGO_SIMD_REQUIRE_LOCAL(arena_a, "simd_fa_softmax", "arena");
    // bc must be even (the quantiser packs 2:1) and both must be non-zero: a zero bound
    // is a DEGENERATE task, which the block DROPS rather than running -- the counters
    // advance and the output keeps its old contents, which is indistinguishable from a
    // correct run on stale data.
    if (bc == 0u || dhead == 0u || (bc & 1u)) {
        printf_safe("[Cluster %d Core %d]: Error! simd_fa_softmax bad geometry "
                    "bc=%d dhead=%d (bc must be even and both non-zero)\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), bc, dhead);
        return BINGO_RET_FAIL;
    }

    uint32_t arena = (uint32_t)arena_a;
    snax_simd_shape_t *sh = (snax_simd_shape_t *)arena;

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_START);
    // Geometry is fixed for a whole FA run of one query tile, so build it once per arena.
    // Re-pointing the two bases that do change is two stores.
    //
    // tile_idx 0 is the first invocation for a given arena -- the workload walks j (the KV
    // tile) outermost -- so it rebuilds UNCONDITIONALLY unless a SHAPES_ONLY prologue has
    // already primed this arena. The memo is therefore only ever TRUSTED after a write this
    // run made itself, never required to be valid on a cold arena, and uninitialised L1
    // cannot be mistaken for a built geometry.
    //
    // WHY A PROLOGUE EXISTS AT ALL. The build is a few hundred scalar stores into TCDM,
    // and their cost is set by what else is using the TCDM ports: with the iDMA streaming a
    // K/V tile through the same ports each store costs several times what it does on an
    // idle cluster. It is neither icache nor the CSR writes, it is port contention, so the
    // cure is to not do it on the critical path. A prologue node does the build before the
    // load chain has spun up, and every real tile after it takes the memo path instead.
    //
    // THE COLD TEST MUST COME FIRST, AND || MUST SHORT-CIRCUIT IT.
    //
    // TCDM is `tc_sram` with SimInit = "none" (snitch_data_mem.sv passes no SimInit and
    // the default is "none"), and +vcs+initreg is applied only to netlist builds, not to
    // sim_rtl. A word this run has not written therefore reads back X. Comparing the memo
    // of a COLD arena puts that X into a branch condition, the PC goes X with it, and the
    // hart is gone: no fault, no exception, no timeout, just a core that never retires
    // again.
    //
    // So the cold test has to be the LEFT operand, where C's || never evaluates the memo
    // on an arena nobody has written. The boolean value of the condition is the same
    // either way; only the evaluation order is. The memo is then read only when a write
    // is guaranteed to have happened already: by an earlier tile (tile_idx > 0) or by a
    // PROLOGUE ancestor (PRIMED).
    //
    // The reference has the same hazard and solves it the same way round: it clears its
    // shape table up front (snax-flashattn-decode.c, snax_simd_shapes_clear) precisely
    // because .l1 is NOLOAD and "a shape that is declared but never filled programs the
    // AGU from whatever was in TCDM".
    simd_fa_memo_t *memo = (simd_fa_memo_t *)(arena + SIMD_FA_MEMO_OFF);
    const uint32_t shapes_ready = (geom_mode == SIMD_FA_GEOM_PRIMED ||
                                   geom_mode == SIMD_FA_GEOM_CSR_PRIMED);
    const uint32_t cold = (tile_idx == 0u && !shapes_ready);
    if (cold || memo->magic != SIMD_FA_MEMO_MAGIC ||
        memo->bc != bc || memo->dhead != dhead) {
        simd_fa_build_shapes(arena, bc, dhead);
        memo->magic = SIMD_FA_MEMO_MAGIC;
        memo->bc    = bc;
        memo->dhead = dhead;
    }
    // The PROLOGUE does NOT return here, because the rest of the config is where the cost
    // is: a handful of instruction-cache line refills in this function's base-re-pointing
    // block and snax_simd_program_fast(). A refill is cheap on an idle fabric and expensive
    // while the iDMA streams a tile through the same path, which is exactly when the first
    // real tile runs. Doing them in the prologue moves the refills into a quiet window and
    // every later tile hits in the icache.
    //
    // Nothing else changes: no beat is moved, no accelerator task is queued, and the running
    // state is left alone -- seed_state is 0 for a prologue node, so init_state below does
    // not run and the xDMA memsets remain the only writer of m and l.
    if (tile_idx == 0u && seed_state) simd_fa_init_state(arena, bc, dhead);
    sh[FA_SH_TAP_IN].base = (void *)(uint32_t)s16_src;

    // The caller hands us the GEMM's D buffer, which the workload allocated with ONE BEAT
    // of headroom in front of it. That beat is where -m_new goes, so the fused exp pass can
    // read [-m_new][the whole tile] as one contiguous stream straight out of the GEMM's
    // output -- no copy. rmax's address comes from a shape that already points at it rather
    // than from re-walking the layout.
    {
        const uint32_t negm = (uint32_t)s16_src - SIMD_BEAT_BYTES;
        const uint32_t rmax = (uint32_t)sh[FA_SH_MNEW_IN].base;
        sh[FA_SH_P_IN].base = (void *)negm;
        const uint32_t rsum_p8 = (uint32_t)p8_dst + (bc / 2u) * SIMD_BEAT_BYTES;
        sh[FA_SH_P_OUT].base = (void *)(uint32_t)p8_dst;
        sh[FA_SH_LNEW_IN].stride[0] = rsum_p8 - (uint32_t)sh[FA_SH_LNEW_IN].base;
        sh[FA_SH_NEGM_OUT].stride[0] = negm - rmax;   // arena is allocated first: positive
    }

    // One full program establishes every CSR no task below varies -- address high word,
    // spatial stride, channel and byte masks, temporal dims 1 and 2. From here each task
    // writes only its six 1-D CSRs. It is done per invocation, not once ever, because
    // another node may have used the block since.
    if (geom_mode != SIMD_FA_GEOM_CSR_PRIMED)
        snax_simd_program_fast(&sh[FA_SH_TAP_IN], &sh[FA_SH_TAP_OUT]);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_CFG_END);

    // A PROLOGUE stops HERE -- after the whole config, before the first task is fired.
    // The CSRs it has just written are shared state that the next invocation rewrites
    // unconditionally (snax_simd_program_fast is "done per invocation, not once ever"),
    // so leaving them programmed is not a handover, only a warm cache.
    if (geom_mode == SIMD_FA_GEOM_PROLOGUE) return BINGO_RET_SUCC;

    // ---- the online softmax, in full ------------------------------------------------
    // Every task below is one or two beats except 1, 8+9, 10 and 14. That ratio is the
    // point: the STATE UPDATE is dominated by per-task start and drain, not by compute,
    // which is why they are fired back to back with no wait between them.
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_START);
    // 1  rowmax over the tile the GEMM has already written as fp16.
    snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, bc,
                   SIMD_RED_MAX | SIMD_RED_LANEWISE);
    snax_simd_program_1d(&sh[FA_SH_TAP_IN], &sh[FA_SH_TAP_OUT]);
    snax_simd_fire();
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_START);
    // 2  m_new = max(m_old, rowmax)
    snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, 2,
                   SIMD_RED_MAX | SIMD_RED_LANEWISE);
    snax_simd_program_1d(&sh[FA_SH_MNEW_IN], &sh[FA_SH_MNEW_OUT]);
    snax_simd_fire();
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_START);
    // 3+4  -m_new into both latches, one task
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, SIMD_F32_NEG_ONE, 0,
                   SIMD_FUNC_LINEAR);
    snax_simd_program_1d(&sh[FA_SH_NEGM_IN], &sh[FA_SH_NEGM_OUT]);
    snax_simd_fire();
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_START);
    // 5  delta = m_old - m_new
    snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, 2,
                   SIMD_RED_ADD | SIMD_RED_LANEWISE);
    snax_simd_program_1d(&sh[FA_SH_DELTA_IN], &sh[FA_SH_DELTA_OUT]);
    snax_simd_fire();
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_START);
    // 6+7  corr = exp(delta), to both places a latch is needed
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, SIMD_F32_ONE, 0,
                   SIMD_FUNC_EXP);
    snax_simd_program_1d(&sh[FA_SH_CORR_IN], &sh[FA_SH_CORR_OUT]);
    snax_simd_fire();
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_START);
    // 8+9  P = exp(S - m_new) AND its rowsum, in ONE pass over the tile. EW0 is upstream
    //      of Map, so the per-lane subtract happens before the exponential and the tile
    //      is never written out in between. Sticky-B suppresses the seed beat, so EW0
    //      emits exactly the bc data beats.
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR,
                            (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                (1u << SIMD_EXT_STREAMMAP) |
                                (1u << SIMD_EXT_STREAMREDUCE) |
                                (1u << SIMD_EXT_FP16TOINT8));
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 0, 1);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 1,
                            SIMD_EW_ADD | SIMD_EW_STICKY_B);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 0, SIMD_F32_ONE);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 1, 0);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 2, SIMD_FUNC_EXP);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 0, bc);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 1,
                            SIMD_RED_ADD | SIMD_RED_LANEWISE | SIMD_RED_TAP);
    // The quantiser narrows the bc data beats 2:1 and lets the TAPPED row sum through
    // unnarrowed, so this one pass emits both the operand the next GEMM reads and the
    // fp16 statistic the recurrence needs. tailPeriod counts DATA beats before each tail,
    // so bc (even) is right -- bc+1 would be the full period and is odd.
    // The scale is 127.0, not 1.0. P = exp(S - m) lies in [0,1], so a scale of 1.0 rounds
    // every element to 0 or 1 -- P degenerates to a BINARY MASK and the PV matmul that
    // consumes it computes nothing meaningful. Nothing upstream of O catches this, since m
    // and the row sum are both correct either way; only a check that reads O back does.
    // BINGO_SIMD_I8_SCALE_UNIT is what the general simd_softmax kernel uses for the same
    // [0,1] range.
    //
    // O comes out 127x larger as a result. Harmless here -- O is never normalised by l in
    // this benchmark -- but a real FA epilogue dividing by l MUST account for it, because
    // l is accumulated from the UNSCALED fp16 P.
    snax_write_simd_cfg_reg(SIMD_EXT_FP16TOINT8_CSR + 0, BINGO_SIMD_I8_SCALE_UNIT);
    snax_write_simd_cfg_reg(SIMD_EXT_FP16TOINT8_CSR + 1, SIMD_QUANT_TAIL(bc));
#if BINGO_FA_DRAIN_BEFORE_FUSED
    // DRAIN BEFORE THE FUSED PASS READS THE -m_new PREFIX.
    //
    // This is the one producer/consumer pair in the softmax that is not a plain arena slot:
    // tasks 3+4 write -m_new into the beat immediately BELOW the score tile
    // (`negm = s16_src - SIMD_BEAT_BYTES`), and the fused pass below streams
    // [negmS][S^T x bc] starting AT that beat, sticky-latching it as the value subtracted
    // from every score. Every other task reads state written a task or more earlier; this
    // one reads the immediately preceding task's output.
    //
    // Tasks are deliberately fired back to back with no wait, so the ordering rests on the
    // block retiring task N before task N+1's first beat enters the chain. MEASURED: at
    // NSCORE=3 the fused pass reads X out of `fa_s16_2`'s prefix (0x44300 -- the buffer used
    // exactly once under the 0,1,2,0 rotation, so nothing rewrote it first), 516 ns before
    // the SIMD writes X into `fa_p8_2`. One bounded drain per softmax closes it.
    if (snax_simd_wait_all_bounded()) {
        printf_safe("[Cluster %d Core %d]: SIMD drain timeout before fused pass\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
#endif
    snax_simd_program_1d(&sh[FA_SH_P_IN], &sh[FA_SH_P_OUT]);
    snax_simd_fire();
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_END);

    // 10 quantise is GONE -- the fused pass above emits INT8 directly.

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_START);
    // 14 O *= corr. HOISTED to here, directly after the quantise, because P8 and the
    //    rescaled O are the only two things the next GEMM consumes. It depends on corr
    //    and on nothing below, so nothing stops it running now -- and everything after
    //    this point only prepares the NEXT tile.
    //
    // NOTE (2026-09-21): this task writes `L.oacc`, the arena's own O region, which nothing
    // reads -- Store_o reads fa_oacc32_{q}, a separate allocation PV accumulates into. So it
    // is DEAD: removing it (and the 8,192 B region, and the 1,024 B fa_cz alongside) passed
    // 12/12.
    //
    // It is restored, and the reason is a measurement one. Deleting it edits the device
    // library, which shifts every kernel address and the .rodata dispatch table -- an effect
    // measured at 3.7 pp ON ITS OWN, because .rodata reads from a cluster kernel are
    // blocking fabric round trips. The removal arm came out 62.1% -> 59.5%, which is inside
    // that noise band and therefore says nothing either way. (An earlier -140 cc reading
    // quoted here was worse than useless: it was taken from a bingo_trace.json that was
    // still being written, when the UART showed 8 of 12 checks.)
    //
    // To actually price it, build a matched baseline with the SAME device library and
    // compare against that. The REAL bug here is separate and still open: PV applies no corr
    // at all, so fa_oacc32's recurrence is incomplete whether or not this task exists.
    snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1,
                   SIMD_EW_MUL | SIMD_EW_STICKY_B);
    snax_simd_program_1d(&sh[FA_SH_ORS_IN], &sh[FA_SH_ORS_OUT]);
    snax_simd_fire();
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_START);
    // 11 corr * l_old
    snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1,
                   SIMD_EW_MUL | SIMD_EW_STICKY_B);
    snax_simd_program_1d(&sh[FA_SH_LSC_IN], &sh[FA_SH_LSC_OUT]);
    snax_simd_fire();
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_START);
    // 13 l_new = corr*l_old + rowsum
    snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, 2,
                   SIMD_RED_ADD | SIMD_RED_LANEWISE);
    snax_simd_program_1d(&sh[FA_SH_LNEW_IN], &sh[FA_SH_LNEW_OUT]);
    snax_simd_fire();
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_START);
    // 15+16 commit the running state for the next KV tile.
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, SIMD_F32_ONE, 0,
                   SIMD_FUNC_LINEAR);
    snax_simd_program_1d(&sh[FA_SH_CMT_IN], &sh[FA_SH_CMT_OUT]);
    snax_simd_fire();
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_TASK_END);

    // ONE drain for all eleven. Bounded, so a wedged engine reports itself instead of
    // hanging the simulation, and the node fails instead of letting the next GEMM read a
    // half-written P8.
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_DRAIN_START);
    uint32_t rc = snax_simd_wait_all_bounded();
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_DRAIN_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    if (rc) {
        printf_safe("[Cluster %d Core %d]: simd_fa_softmax drain timeout tile %d "
                    "(submitted %d finished %d status %08x)\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), tile_idx,
                    snax_simd_submitted(), snax_simd_finished(),
                    snax_read_simd_cfg_reg(SIMD_STATUS));
        return BINGO_RET_FAIL;
    }
    if (snax_simd_bad_config()) {
        printf_safe("[Cluster %d Core %d]: simd_fa_softmax: the block REFUSED a task as "
                    "degenerate (bad-config sticky). Check bc/dhead and the arena.\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    sp->return_value = (uint32_t)p8_dst;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}
