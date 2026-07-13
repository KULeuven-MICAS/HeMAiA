// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
//
// Core-level bingo xDMA kernels. Everything that programs the snax xDMA
// engine — generic 1D/6D copies, layout-transform kernels between row-major
// and versacore-tiled A/B/D, 2D helpers (transpose, submatrix, expand,
// concat, pad, gather). No versacore streamer CSR writes here.

#pragma once

#include "../macros.h"

// ==========================================================================
// Optional xDMA reader-extension availability
//
// The streaming-SIMD extensions (StreamMap, StreamReduce, StreamElementwise,
// Fp16ToInt8) are an EXTENSION to the base xDMA and are only present if the RTL
// cfg asked for them; the generated snax_xdma_addr.h defines a READER_EXT_* index
// only for the ones that exist. A cfg without them must still be able to BUILD
// this library — it just may not CALL those kernels. So each extension gets:
//
//   BINGO_HAS_<EXT>  1/0 compile-time flag. The kernels branch on it, and because
//                    it is a literal the compiler folds the branch away: on a cfg
//                    that has the extension the check costs nothing, and on one
//                    that lacks it the whole body is dead-code-eliminated down to
//                    the BINGO_XDMA_EXT_UNSUPPORTED report.
//   READER_EXT_<EXT> sentinel 0xFF when absent, so the (unreachable) body still
//                    compiles. 0xFF is out of range for every cfg, and
//                    xdma_enable_src_ext() rejects any index >= XDMA_SRC_EXT_NUM,
//                    so even a mis-gated call cannot program the wrong extension.
//
// The WRITER_EXT_* extensions below (ElementwiseAdd, Transposer) are NOT listed
// here on purpose: those kernels already carry a real CPU fallback and so keep
// their plain #ifdef, which selects an implementation rather than failing.
// ==========================================================================
#if defined(READER_EXT_STREAMMAP)
#define BINGO_HAS_STREAMMAP 1
#else
#define BINGO_HAS_STREAMMAP 0
#define READER_EXT_STREAMMAP 0xFFu
#endif

#if defined(READER_EXT_STREAMREDUCE)
#define BINGO_HAS_STREAMREDUCE 1
#else
#define BINGO_HAS_STREAMREDUCE 0
#define READER_EXT_STREAMREDUCE 0xFFu
#endif

#if defined(READER_EXT_STREAMELEMENTWISE)
#define BINGO_HAS_STREAMELEMENTWISE 1
#else
#define BINGO_HAS_STREAMELEMENTWISE 0
#define READER_EXT_STREAMELEMENTWISE 0xFFu
#endif

#if defined(READER_EXT_FP16TOINT8)
#define BINGO_HAS_FP16TOINT8 1
#else
#define BINGO_HAS_FP16TOINT8 0
#define READER_EXT_FP16TOINT8 0xFFu
#endif

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_1d_copy(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_1d_copy_args_t);
    // Copy 1d data from src to dst using xdma
    // Arg0: uint32_t src_addr_hi
    // Arg1: uint32_t src_addr_lo
    // Arg2: uint32_t dst_addr_hi
    // Arg3: uint32_t dst_addr_lo
    // Arg4: uint32_t size in Byte

    if (snrt_is_dm_core())
    {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint64_t src_addr = make_u64(((uint32_t *)arg)[0], ((uint32_t *)arg)[1]);
        uint64_t dst_addr = make_u64(((uint32_t *)arg)[2], ((uint32_t *)arg)[3]);
        uint32_t data_size = ((uint32_t *)arg)[4];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_1d_copy_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
        xdma_disable_all_extensions();
        BINGO_XDMA_TRY(xdma_memcpy_1d_full_addr(src_addr, dst_addr, data_size), "xdma_1d_copy");
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        xdma_task_t task_id = xdma_start();
        xdma_wait_task(task_id);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
        XDMA_DEBUG_PRINT("XDMA copy completed\n");
        XDMA_DEBUG_PRINT("SRC ADDR = %lx\n", src_addr);
        XDMA_DEBUG_PRINT("DST ADDR = %lx\n", dst_addr);
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else{
        printf_safe("[Cluster %d Core %d]: Error! XDMA copy should be called from a DM core!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

// ==========================================================================
// xDMA data layout transformation kernels
// These use the xDMA N-dimensional memcpy with AGU stride configuration
// to perform reshape and transpose operations in hardware.
// ==========================================================================

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_6d(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_6d_args_t);
    // Generic 6D xDMA transfer with explicit AGU strides and bounds.
    // This is the low-level kernel that exposes the full streamer AGU
    // configuration to the user. The maximum streamer dimension is 6
    // (1 spatial + up to 5 temporal, but we use a fixed 6-entry layout
    // so the struct is fixed-size and no variable-length parsing is needed).
    //
    // Unused dimensions should have stride=0 and bound=1.
    //
    // Arg layout (__snax_bingo_kernel_xdma_6d_args_t):
    //   [0]  src_addr_hi
    //   [1]  src_addr_lo
    //   [2]  dst_addr_hi
    //   [3]  dst_addr_lo
    //   [4]  spatial_stride_src
    //   [5]  spatial_stride_dst
    //   [6]  num_temporal_dims     (1..5, number of active temporal dimensions)
    //   [7..11]  temporal_strides_src[5]  (unused dims set to 0)
    //   [12..16] temporal_bounds_src[5]   (unused dims set to 1)
    //   [17..21] temporal_strides_dst[5]  (unused dims set to 0)
    //   [22..26] temporal_bounds_dst[5]   (unused dims set to 1)

    if (snrt_is_dm_core())
    {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t *a = (uint32_t *)arg;
        uint64_t src_addr = make_u64(a[0], a[1]);
        uint64_t dst_addr = make_u64(a[2], a[3]);
        uint32_t spatial_stride_src = a[4];
        uint32_t spatial_stride_dst = a[5];
        uint32_t num_dims = a[6];
        uint32_t *t_strides_src = &a[7];
        uint32_t *t_bounds_src  = &a[12];
        uint32_t *t_strides_dst = &a[17];
        uint32_t *t_bounds_dst  = &a[22];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_6d_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

        // Disable all extensions (pure AGU data movement)

        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
        xdma_disable_all_extensions();
        BINGO_XDMA_TRY(xdma_memcpy_nd_full_addr(
            src_addr, dst_addr,
            spatial_stride_src, spatial_stride_dst,
            num_dims, t_strides_src, t_bounds_src,
            num_dims, t_strides_dst, t_bounds_dst,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        ), "xdma_6d");
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);

        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        xdma_task_t task_id = xdma_start();
        xdma_wait_task(task_id);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else {
        printf_safe("[Cluster %d Core %d]: Error! xDMA 6d should be called from DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

// ==========================================================================
// xDMA writer ElementwiseAdd kernels
//
// What the HW does
// ----------------
// The HasElementwiseAdd writer extension sits in the xDMA *writer* datapath and
// accumulates `num_operands` consecutive vectors that flow through it into a
// single output vector — an N:1 reduction. The xDMA bus is 512-bit, so one
// "vector"/"tile" is 512b = 16 int32 elements (elementWidth=32 in the cfg). The
// extension keeps a per-lane accumulator: it loads on the 1st of every group of
// `num_operands` inputs, adds the next (num_operands-1), and emits the sum, then
// resets for the next group. So the reader feeds num_operands x more vectors
// than the writer stores.
//
// How we drive the AGU
// --------------------
// To make the N operands of one output tile arrive back-to-back, the reader AGU
// uses the operand index as its INNERMOST temporal dim, then iterates tiles:
//   reader: dim0 = operand  [bound=num_operands, stride=operand_stride]
//           dim1 = tile      [bound=tiles,        stride=64 bytes]
//   writer: dim0 = tile      [bound=tiles,        stride=64 bytes]
// where tiles = num_int32_elem_per_operand / 16. Reader address for (operand o,
// tile t) = src_base + o*operand_stride + t*64.
//
// Worked example: sum 3 operands, 32 int32 each (tiles = 32/16 = 2)
// ----------------------------------------------------------------
//   L1 layout (operand_stride = 32*4 = 128 bytes):
//     src_base+0x00 : A[0..15] | A[16..31]          (operand 0, tiles A0,A1)
//     src_base+0x80 : B[0..15] | B[16..31]          (operand 1, tiles B0,B1)
//     src_base+0x100: C[0..15] | C[16..31]          (operand 2, tiles C0,C1)
//   Reader emission order (operand inner, tile outer):
//     A0,B0,C0, A1,B1,C1
//   Writer extension groups every num_operands=3:
//     out tile0 = A0+B0+C0   (dst+0x00, 16 int32)
//     out tile1 = A1+B1+C1   (dst+0x40, 16 int32)
//   => dst[i] = A[i] + B[i] + C[i] for i in 0..31, in one streaming pass.
//
// Why this kernel exists
// ----------------------
// It fuses the GEMM K-split partial-sum reduction (D = D0 + D1 + ... ) into a
// single xDMA pass, replacing the sequential host int32-add chain
// (__host_bingo_kernel_add_i32) that walks L3<->host once per pair.
//
// Two entry points
// ----------------
//   __snax_bingo_kernel_xdma_elementwise_add        : general N-operand form,
//       caller supplies src_base, num_operands, operand_stride.
//   __snax_bingo_kernel_xdma_elementwise_add_ab : convenience dst = a + b;
//       derives the stride from src_a/src_b. The two buffers may be in EITHER order:
//       the wrapper bases the reader at the LOWER address and strides up to the higher
//       (valid because add is commutative), so the caller need not pre-order them.
//
// Constraints / fallback
// ----------------------
//   - num_int32_elem_per_operand must be a multiple of 16 (one 512b bus word).
//   - The reader AGU strides FORWARD only (an unsigned-wrapping "negative" stride
//     reads out of range and stalls). The 2-operand convenience wrappers (add_ab,
//     StreamElementwise w/ src_b_addr) hide this by swapping to the lower base for
//     commutative ops; the GENERAL N-operand form (explicit operand_stride) still
//     requires operands laid out at a constant ascending stride.
//   - Falls back to a plain CPU int32 accumulate when the writer extension is
//     not present in the generated HW (WRITER_EXT_ELEMENTWISEADDBIT32 undefined).
// ==========================================================================
static inline uint32_t xdma_elementwise_add_run(
    uint64_t src_base, uint64_t dst_addr,
    uint32_t num_int32_elem_per_operand, uint32_t num_operands,
    uint32_t operand_stride)
{
    // Each output vector is one 512b bus word = XDMA_WIDTH/4 = 16 int32, so the
    // per-operand element count must be a whole, non-zero number of bus words;
    // otherwise the tile count truncates and the transfer would be wrong.
    if (num_int32_elem_per_operand == 0 ||
        (num_int32_elem_per_operand % (XDMA_WIDTH / 4)) != 0) {
        printf_safe("[Cluster %d Core %d]: Error! xDMA elementwise_add "
                    "num_int32_elem_per_operand=%u must be a positive multiple of %u\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(),
                    num_int32_elem_per_operand, (unsigned)(XDMA_WIDTH / 4));
        return BINGO_RET_FAIL;
    }
    uint32_t tiles = num_int32_elem_per_operand / (XDMA_WIDTH / 4);  // 16 int32/vector
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
#ifdef WRITER_EXT_ELEMENTWISEADDBIT32
    xdma_disable_all_extensions();
    uint32_t csr[1] = { num_operands };
    xdma_enable_dst_ext(WRITER_EXT_ELEMENTWISEADDBIT32, csr);

    // Reader: dim0 = operand index (inner), dim1 = output tiles.
    uint32_t ts_src[2] = { operand_stride, XDMA_WIDTH };
    uint32_t tb_src[2] = { num_operands,   tiles      };
    // Writer: one accumulated vector per output tile.
    uint32_t ts_dst[1] = { XDMA_WIDTH };
    uint32_t tb_dst[1] = { tiles      };
    BINGO_XDMA_TRY(xdma_memcpy_nd_full_addr(
        src_base, dst_addr,
        XDMA_WIDTH / XDMA_SPATIAL_CHAN, XDMA_WIDTH / XDMA_SPATIAL_CHAN,
        2, ts_src, tb_src, 1, ts_dst, tb_dst,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF), "xdma_elementwise_add");
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
    xdma_task_t task_id = xdma_start();
    xdma_wait_task(task_id);
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
    xdma_disable_dst_ext(WRITER_EXT_ELEMENTWISEADDBIT32);
#else
    // CPU fallback: dst[e] = sum_o src[o][e] over int32 elements.
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
    volatile int32_t *dst = (volatile int32_t *)(uint32_t)dst_addr;
    for (uint32_t e = 0; e < num_int32_elem_per_operand; e++) {
        int32_t acc = 0;
        for (uint32_t o = 0; o < num_operands; o++) {
            volatile int32_t *src =
                (volatile int32_t *)((uint32_t)src_base + o * operand_stride);
            acc += src[e];
        }
        dst[e] = acc;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
    (void)tiles;
#endif
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_elementwise_add(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_elementwise_add_args_t);
    // General N-operand form: dst[i] = sum over o of operand_o[i].
    //   [0] src_addr_hi  [1] src_addr_lo  [2] dst_addr_hi  [3] dst_addr_lo
    //   [4] num_int32_elem_per_operand (multiple of 16)
    //   [5] num_operands  [6] operand_stride (bytes between operands)
    //
    // The N operands must be EVENLY SPACED and ASCENDING: operand o lives at
    // src_base + o*operand_stride (the reader only strides forward). Use this
    // when partials are a regular array, e.g. a contiguous [N, M] int32 block
    // -> num_operands = N, operand_stride = M*4.
    //
    // The 2-operand _ab variant below is just this kernel specialized to
    // num_operands = 2 with operand_stride derived from the two addresses:
    //   add_ab(a, b, dst, n)  ==  add(a, dst, n, 2, b - a).
    if (snrt_is_dm_core()) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t *a = (uint32_t *)arg;
        uint64_t src_base = make_u64(a[0], a[1]);
        uint64_t dst_addr = make_u64(a[2], a[3]);
        uint32_t num_int32_elem_per_operand = a[4];
        uint32_t num_operands = a[5];
        uint32_t operand_stride = a[6];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_elementwise_add_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        if (xdma_elementwise_add_run(src_base, dst_addr, num_int32_elem_per_operand,
                                     num_operands, operand_stride) != BINGO_RET_SUCC)
            return BINGO_RET_FAIL;
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else {
        printf_safe("[Cluster %d Core %d]: Error! xDMA elementwise_add should be called from a DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_elementwise_add_ab(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_elementwise_add_ab_args_t);
    // Two-operand convenience form: dst = a + b (int32, num_int32_elements, mult of 16).
    //   [0] src_a_hi [1] src_a_lo [2] src_b_hi [3] src_b_lo
    //   [4] dst_hi   [5] dst_lo   [6] num_int32_elements
    //
    // Identical HW path to the N-operand kernel above, fixed to num_operands = 2
    // with the stride derived from the two addresses instead of passed in:
    //   add_ab(a, b, dst, n)  ==  add(a, dst, n, 2, b - a).
    // Use this when you have two independent buffers and don't want to compute a
    // stride. The two buffers may be in EITHER order: the reader AGU only strides
    // FORWARD, so the body bases at the LOWER address and strides up to the higher
    // (a swap; valid because add is commutative). See the StreamElementwise header
    // for the full layout contract + the HW sign-extend TODO that would drop the swap.
    if (snrt_is_dm_core()) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t *a = (uint32_t *)arg;
        uint64_t src_a = make_u64(a[0], a[1]);
        uint64_t src_b = make_u64(a[2], a[3]);
        uint64_t dst_addr = make_u64(a[4], a[5]);
        uint32_t num_int32_elements = a[6];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_elementwise_add_ab_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        // The reader AGU strides FORWARD only, so the base must be the LOWER of the two
        // operands. If src_b sits below src_a, swap (base = the lower, stride up to the
        // higher); valid because add is commutative (a + b == b + a).
        uint64_t lo_base;
        uint32_t operand_stride;
        if ((uint32_t)src_b >= (uint32_t)src_a) {
            lo_base = src_a;
            operand_stride = (uint32_t)src_b - (uint32_t)src_a;
        } else {
            lo_base = src_b;
            operand_stride = (uint32_t)src_a - (uint32_t)src_b;
        }
        if (xdma_elementwise_add_run(lo_base, dst_addr, num_int32_elements, 2, operand_stride) != BINGO_RET_SUCC)
            return BINGO_RET_FAIL;
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else {
        printf_safe("[Cluster %d Core %d]: Error! xDMA elementwise_add_ab should be called from a DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

// ==========================================================================
// xDMA FP16 streaming-SIMD primitives (reader extensions)
//
// The 3 generic ops the LLM layers (softmax / rmsnorm / silu / swiglu / rope)
// decompose into. Each is a thin wrapper that programs one reader extension,
// launches a single xDMA task over `rows` x `beats` 64-byte beats, and waits.
// A row is `beats` x 64-byte beats (64 B = 32 FP16); `rows` independent rows are
// processed in one dispatch (rows=1 is the back-compatible single-row case). The
// op/func selector and FP32-bit operands come in as args, so the named sub-ops
// (reduce_max, map_exp, ew_mul, ...) are just these 3 kernels invoked with preset
// args by the Python minicompiler.
//
// Multi-row: `reduce` is per-row (the reduce counter wraps at operandCount=beats,
// re-initing each row and emitting one splatted scalar beat per row, so its reader
// gains a `rows` outer dim and its writer emits `rows` beats). `map`/`elementwise`
// are element-wise, so multi-row is just a larger flat outer bound `rows*beats`.
//
// CSR encodings (passed verbatim into the extension CSRs by the caller):
//   StreamMap        func: 0=LINEAR(a*x+b)  1=EXP  2=SILU   CSR={a_f32,b_f32,func}
//   StreamReduce     op:   0=MAX  1=ADD  2=SUMSQ            CSR={beats,op}
//                    op bit8  (0x100) = TAP        : pass the row through 1:1, then
//                                                    emit the scalar as a trailing beat.
//                    op bit9  (0x200) = OUT_FP32   : emit the per-row scalar in FP32
//                                                    (no narrow to the FP16 transport).
//                                                    Use whenever the reduction can
//                                                    exceed ~6e4 (e.g. SUMSQ of unscaled
//                                                    activations), since the FP16 narrow
//                                                    wraps to garbage (NOT inf) on overflow.
//                                                    The host then reads the scalar as
//                                                    `float` at stride 16 (FP32/beat)
//                                                    instead of `uint16_t` at stride 32.
//   StreamElementwise op:  0=MUL  1=ADD                     CSR={operand_count,op}
//   Fp16ToInt8       CSR={inv_scale_f32}   (fused quant: FP16 stream -> packed INT8)
//
// Sticky CSR: the row AGU shape is identical across a same-shape group, so
// csr_mode selects FULL (full xdma_memcpy_nd_full_addr) for the first op of a
// group, or STICKY (xdma_retask_1d: 5 writes, reuse the persisted shape) for the
// rest. dst_bound0 is the WRITER beat count the caller computes.
// ==========================================================================

// Shared launch for the stream primitives: program the AGU (full config or sticky
// retask), run one xDMA task, and wait. The reader extension(s) must already be
// enabled by the caller. The src AGU has `src_dims` temporal dims (2 for the flat
// {beat, 1} stream shape; 3 for the PADDED-ROW {operand, beat, row} shape the
// elementwise op uses to read a TAP-padded tensor); dst is 1D {64-byte beat, dst_bound0}.
// Returns BINGO_RET_SUCC, or BINGO_RET_FAIL (via BINGO_XDMA_TRY) on a cfg error.
static inline uint32_t xdma_stream_launch(
    uint64_t src_addr, uint64_t dst_addr,
    uint32_t *src_str, uint32_t *src_bnd, uint32_t src_dims,
    uint32_t dst_bound0, uint32_t csr_mode)
{
    if (csr_mode == 0u) {  // FULL: completely configure the AGU (default)
        uint32_t dst_str[1] = { XDMA_WIDTH };
        uint32_t dst_bnd[1] = { dst_bound0 };
        BINGO_XDMA_TRY(xdma_memcpy_nd_full_addr(
            src_addr, dst_addr,
            XDMA_WIDTH / XDMA_SPATIAL_CHAN, XDMA_WIDTH / XDMA_SPATIAL_CHAN,
            src_dims, src_str, src_bnd, 1, dst_str, dst_bnd,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF), "xdma_stream");
    } else {               // STICKY: reuse the persisted same-shape config (opt-in)
        BINGO_XDMA_TRY(xdma_retask_1d(
            (void *)(uint32_t)src_addr, (void *)(uint32_t)dst_addr, dst_bound0),
            "xdma_stream_retask");
    }
    xdma_task_t task_id = xdma_start();
    xdma_wait_task(task_id);
    return BINGO_RET_SUCC;
}

// StreamReduce op-CSR flag bits, OR'd into `op` by the caller (the minicompiler).
// The op CSR is passed verbatim into the extension, so these need no datapath change
// here — only the host's read-back stride changes for OUT_FP32 (see comment above).
#ifndef REDUCE_OP_TAP
#define REDUCE_OP_TAP   (1u << 8)   // pass the row through, then emit the scalar beat
#endif
#ifndef REDUCE_OUT_FP32
#define REDUCE_OUT_FP32 (1u << 9)   // emit the per-row scalar in FP32 (no FP16 narrow)
#endif

// StreamReduce: per-row reduction (row -> scalar). op = MAX/ADD/SUMSQ. Runs `rows`
// independent reductions in one dispatch: the reader is 2D {beats inner, rows
// outer}, the writer emits one splatted scalar beat per row (dst_bound0 = rows).
// OR REDUCE_OUT_FP32 into `op` to keep the scalar in FP32 (read back at stride 16).
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_stream_reduce(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_stream_reduce_args_t);
    if (snrt_is_dm_core()) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t *a = (uint32_t *)arg;
        uint64_t src_addr = make_u64(a[0], a[1]);
        uint64_t dst_addr = make_u64(a[2], a[3]);
        uint32_t beats      = a[4];
        uint32_t op         = a[5];
        uint32_t rows       = a[6];
        uint32_t csr_mode   = a[7];
        uint32_t dst_bound0 = a[8];
        bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_stream_reduce_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        if (!BINGO_HAS_STREAMREDUCE)
            BINGO_XDMA_EXT_UNSUPPORTED("xdma_stream_reduce", "StreamReduce");
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
        uint32_t src_str[2] = { XDMA_WIDTH, beats * XDMA_WIDTH };
        uint32_t src_bnd[2] = { beats, rows };
        uint32_t csr_red[2] = { beats, op };
        xdma_enable_src_ext(READER_EXT_STREAMREDUCE, csr_red);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        uint32_t rc = xdma_stream_launch(src_addr, dst_addr, src_str, src_bnd, 2u, dst_bound0, csr_mode);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
        xdma_disable_src_ext(READER_EXT_STREAMREDUCE);
        if (rc != BINGO_RET_SUCC) return BINGO_RET_FAIL;
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else {
        printf_safe("[Cluster %d Core %d]: Error! xDMA stream_reduce should be called from a DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

// StreamMap: out = func(a*x + b) per element, over `rows*beats` flat beats.
// a_f32bits/b_f32bits are FP32-bit immediates (a defaults to 1.0). Optional fused
// FP16->INT8 quant (out_dtype=1, pass dst_bound0 = rows*beats/2).
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_stream_map(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_stream_map_args_t);
    if (snrt_is_dm_core()) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t *a = (uint32_t *)arg;
        uint64_t src_addr = make_u64(a[0], a[1]);
        uint64_t dst_addr = make_u64(a[2], a[3]);
        uint32_t beats         = a[4];
        uint32_t a_f32bits     = a[5];
        uint32_t b_f32bits     = a[6];
        uint32_t func          = a[7];
        uint32_t rows          = a[8];
        uint32_t csr_mode      = a[9];
        uint32_t dst_bound0    = a[10];
        uint32_t out_dtype     = a[11];
        uint32_t inv_scale     = a[12];
        uint32_t a_addr_hi     = a[13];
        uint32_t a_addr_lo     = a[14];
        bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_stream_map_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        // The fused quant lane is only touched at out_dtype=1, so a cfg without
        // Fp16ToInt8 can still run every fp16 map — only the int8 output is refused.
        if (!BINGO_HAS_STREAMMAP)
            BINGO_XDMA_EXT_UNSUPPORTED("xdma_stream_map", "StreamMap");
        if (out_dtype == 1u && !BINGO_HAS_FP16TOINT8)
            BINGO_XDMA_EXT_UNSUPPORTED("xdma_stream_map with out_dtype=int8", "Fp16ToInt8");
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
        uint32_t flat = rows * beats;
        uint32_t src_str[2] = { XDMA_WIDTH, flat * XDMA_WIDTH };
        uint32_t src_bnd[2] = { flat, 1 };
        // Optional runtime 'a': a nonzero a_addr means a kernel wrote the FP32 bits
        // into that L1 word (the dequant qsc); read it instead of the baked immediate.
        if (a_addr_lo || a_addr_hi) a_f32bits = *(volatile uint32_t *)a_addr_lo;
        uint32_t csr_map[3] = { a_f32bits, b_f32bits, func };
        xdma_enable_src_ext(READER_EXT_STREAMMAP, csr_map);
        if (out_dtype == 1u) {
            uint32_t csr_q[1] = { inv_scale };
            xdma_enable_src_ext(READER_EXT_FP16TOINT8, csr_q);
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        uint32_t rc = xdma_stream_launch(src_addr, dst_addr, src_str, src_bnd, 2u, dst_bound0, csr_mode);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
        if (out_dtype == 1u) xdma_disable_src_ext(READER_EXT_FP16TOINT8);
        xdma_disable_src_ext(READER_EXT_STREAMMAP);
        if (rc != BINGO_RET_SUCC) return BINGO_RET_FAIL;
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else {
        printf_safe("[Cluster %d Core %d]: Error! xDMA stream_map should be called from a DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

// ==========================================================================
// MERGED StreamMap -||> StreamReduce: the map AND the reduce in ONE xDMA task.
//
// Per row: out = reduce(reduce_op, map(func, a*x + b)). Both reader extensions are
// enabled for a single task. The reader extensions are chained in the hjson's list
// order (StreamMap, StreamReduce, StreamElementwise, Fp16ToInt8), so the map's output
// feeds the reduce INSIDE the datapath: the mapped row is never written out and
// re-read, and two tasks collapse into one.
//
// This is the fusion the snax-xdma-softmax-fold / -rmsnorm-fold apps measure. On
// softmax it replaces the deployed 2-task pair
//     map(EXP)    -> expb        [task 1, writes rows*beats beats]
//     reduce(ADD) -> sum[rows]   [task 2, RE-READS all of expb]
// with one task that emits both, removing a whole pass over the tensor and a whole
// task setup (~600 cc of CSR orchestration; at rows=4,D=64 the fused task is 242 cc
// vs 86+197=283 cc for the pair).
//
// TWO OUTPUT MODES, selected by the TAP bit (REDUCE_OP_TAP, 0x100) in `reduce_op`:
//
//   TAP SET -- "map and reduce, keep both". The mapped row is passed through 1:1 AND
//     the row's reduction scalar is appended as a TRAILING BEAT. The output is a
//     PADDED [rows, beats+1] beat tensor:
//         row r = [ mapped beat 0 ... mapped beat beats-1 | splatted scalar beat ]
//     dst_bound0 = rows*(beats+1); dst buffer = rows*(beats+1)*64 bytes.
//     MIND THE PADDED ROW STRIDE. A downstream consumer of the mapped data must
//     either be single-row (rows=1: a flat reader of `beats` beats simply stops
//     before the trailing scalar) or address rows at the (beats+1)*64-byte stride
//     explicitly -- a flat [rows,D] reader would walk the scalar beats as if they
//     were data. This is the one cost of the fusion.
//
//   TAP CLEAR -- "reduce of a mapped stream", no passthrough. Only the per-row scalar
//     beats are written, exactly like stream_reduce but over the MAPPED values (e.g.
//     SUMSQ of a dequantized GEMM output without materializing the dequantized tile).
//     dst_bound0 = rows; dst buffer = rows*64 bytes.
//
// OR REDUCE_OUT_FP32 (0x200) into reduce_op to keep the per-row scalar in FP32 instead
// of narrowing it to the FP16 transport -- see the stream_reduce header; use it
// whenever the reduction can exceed fp16 range (the narrow wraps to garbage, not inf).
//
// The reduce runs over the MAPPED values, so ordering is map-then-reduce and nothing
// else: an elementwise op can never precede the map in one task (it sits later in the
// chain). Softmax's x-max subtract therefore stays its own StreamElementwise pass; the
// scalar `b` only folds the subtract in when every row shares one b (i.e. rows == 1).
//
// Arg layout (__snax_bingo_kernel_xdma_stream_map_reduce_args_t):
//   [0..1] src_addr   [2..3] dst_addr
//   [4]  beats        (beats per row; also the reduce's operandCount)
//   [5]  a_f32bits    (StreamMap a, FP32 bits; 1.0f = 0x3F800000)
//   [6]  b_f32bits    (StreamMap b, FP32 bits)
//   [7]  func         (StreamMap: 0=LINEAR 1=EXP 2=SILU)
//   [8]  reduce_op    (StreamReduce: 0=MAX 1=ADD 2=SUMSQ; |0x100=TAP; |0x200=OUT_FP32)
//   [9]  rows
//   [10] csr_mode     (0=FULL, 1=STICKY)
//   [11] dst_bound0   (writer beats: rows*(beats+1) with TAP, rows without)
//   [12..13] a_addr   (optional runtime 'a', as in stream_map: if nonzero, read the
//                      FP32 bits from that L1 word instead of the a_f32bits immediate)
// ==========================================================================
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_stream_map_reduce(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_stream_map_reduce_args_t);
    if (snrt_is_dm_core()) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t *a = (uint32_t *)arg;
        uint64_t src_addr = make_u64(a[0], a[1]);
        uint64_t dst_addr = make_u64(a[2], a[3]);
        uint32_t beats      = a[4];
        uint32_t a_f32bits  = a[5];
        uint32_t b_f32bits  = a[6];
        uint32_t func       = a[7];
        uint32_t reduce_op  = a[8];
        uint32_t rows       = a[9];
        uint32_t csr_mode   = a[10];
        uint32_t dst_bound0 = a[11];
        uint32_t a_addr_hi  = a[12];
        uint32_t a_addr_lo  = a[13];
        bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_stream_map_reduce_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        // The merge chains both reader extensions in one task, so both must exist.
        if (!BINGO_HAS_STREAMMAP || !BINGO_HAS_STREAMREDUCE)
            BINGO_XDMA_EXT_UNSUPPORTED("xdma_stream_map_reduce", "StreamMap+StreamReduce");
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
        // Reader is the flat [rows*beats] beat stream, as in stream_map: the reduce
        // re-inits per row off its own operandCount=beats counter, so the reader does
        // not need a row dim of its own. The writer beat count (dst_bound0) is what
        // differs between the TAP and non-TAP modes, and the caller supplies it.
        uint32_t flat = rows * beats;
        uint32_t src_str[2] = { XDMA_WIDTH, flat * XDMA_WIDTH };
        uint32_t src_bnd[2] = { flat, 1 };
        // Optional runtime 'a' (same escape hatch as stream_map): a nonzero a_addr means
        // a kernel wrote the FP32 bits into that L1 word (the dequant qsc); read it
        // instead of the baked immediate.
        if (a_addr_lo || a_addr_hi) a_f32bits = *(volatile uint32_t *)a_addr_lo;
        uint32_t csr_map[3] = { a_f32bits, b_f32bits, func };
        uint32_t csr_red[2] = { beats, reduce_op };
        // BOTH extensions on for the one task -- this is the merge.
        xdma_enable_src_ext(READER_EXT_STREAMMAP, csr_map);
        xdma_enable_src_ext(READER_EXT_STREAMREDUCE, csr_red);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        uint32_t rc = xdma_stream_launch(src_addr, dst_addr, src_str, src_bnd, 2u, dst_bound0, csr_mode);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
        xdma_disable_src_ext(READER_EXT_STREAMREDUCE);
        xdma_disable_src_ext(READER_EXT_STREAMMAP);
        if (rc != BINGO_RET_SUCC) return BINGO_RET_FAIL;
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else {
        printf_safe("[Cluster %d Core %d]: Error! xDMA stream_map_reduce should be called from a DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

// Fp16ToInt8: out[i] = clamp(round(in[i] * inv_scale), -128, 127) over `rows*beats` flat beats.
// A dedicated fp16 activation -> int8 GEMM-operand requant on the xDMA (HasFp16ToInt8 datapath),
// replacing the host CVA6 quantize_f16i8 (which reads the fp16 from L3 element-by-element -- the host
// makespan wall). The narrow is chained after an IDENTITY StreamMap (LINEAR a=1,b=0) so the reader
// emits the fp16 stream the FP16TOINT8 lane consumes -- the same proven path stream_map takes at
// out_dtype=1, just with a fixed passthrough map and no `func`/`a`/`b` for the caller to set.
// inv_scale = FP32 bits of 127/max|x| (the symmetric per-tensor scale; the producer computes max|x|
// via MAX(x)+MAX(-x) reduces). dst_bound0 = rows*beats/2 (int8 packs two elements per fp16 lane).
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_fp16_to_int8(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_fp16_to_int8_args_t);
    if (snrt_is_dm_core()) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t *a = (uint32_t *)arg;
        uint64_t src_addr = make_u64(a[0], a[1]);
        uint64_t dst_addr = make_u64(a[2], a[3]);
        uint32_t beats      = a[4];
        uint32_t rows       = a[5];
        uint32_t inv_scale     = a[6];   // FP32 bits of 127/max|x| (compile-time immediate)
        uint32_t csr_mode      = a[7];
        uint32_t dst_bound0    = a[8];
        uint32_t inv_scale_hi  = a[9];   // optional: if (hi|lo)!=0, read the runtime inv_scale
        uint32_t inv_scale_lo  = a[10];  // (127/max|x|, FP32 bits) that requant_scale wrote to L1
        bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_fp16_to_int8_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        // The narrow is chained after an identity StreamMap, so this one needs both.
        if (!BINGO_HAS_STREAMMAP || !BINGO_HAS_FP16TOINT8)
            BINGO_XDMA_EXT_UNSUPPORTED("xdma_fp16_to_int8", "StreamMap+Fp16ToInt8");
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
        uint32_t flat = rows * beats;
        uint32_t src_str[2] = { XDMA_WIDTH, flat * XDMA_WIDTH };
        uint32_t src_bnd[2] = { flat, 1 };
        // The requant scale is per-tensor and computed at run time (max|x| via MAX(x)+MAX(-x) on the
        // xDMA, then requant_scale on the host) -> read it from L1 when an address is given.
        if (inv_scale_lo || inv_scale_hi) inv_scale = *(volatile uint32_t *)inv_scale_lo;
        uint32_t csr_map[3] = { 0x3F800000u, 0u, 0u };        // identity: a=1.0, b=0, func=LINEAR
        xdma_enable_src_ext(READER_EXT_STREAMMAP, csr_map);
        uint32_t csr_q[1] = { inv_scale };
        xdma_enable_src_ext(READER_EXT_FP16TOINT8, csr_q);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        uint32_t rc = xdma_stream_launch(src_addr, dst_addr, src_str, src_bnd, 2u, dst_bound0, csr_mode);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
        xdma_disable_src_ext(READER_EXT_FP16TOINT8);
        xdma_disable_src_ext(READER_EXT_STREAMMAP);
        if (rc != BINGO_RET_SUCC) return BINGO_RET_FAIL;
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else {
        printf_safe("[Cluster %d Core %d]: Error! xDMA fp16_to_int8 should be called from a DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

// StreamElementwise: out = op(operand_0, operand_1, ...) over `operand_count`
// interleaved streams spaced operand_stride bytes apart (the reader reads
// {op0[beat], op1[beat], ...} as the inner AGU dim), across `rows*beats` flat
// beats. Optional fused FP16->INT8.
//
// REQUIRED DATA LAYOUT (read this before laying out the operands)
// ---------------------------------------------------------------
// The reader AGU computes operand addresses as base + o*operand_stride and can only
// stride FORWARD: operand_stride must be a small, ASCENDING byte offset. The 2-operand
// mode passes the operands as two separate addresses (src_a = a[0:1], src_b = a[13:14])
// and derives operand_stride = src_b - src_a at run time. So the two operand buffers:
//   - must be the SAME size and identically laid out ([rows, beats], one 64-B beat per
//     (row,beat)), so operand_0[beat] and operand_1[beat] are a constant stride apart, and
//   - the stride must be small relative to the 48-bit AxiAddressWidth (a few KB, i.e. the
//     two buffers near each other in L1) -- a wrapped "negative" stride reads out of range
//     and the reader STALLS (the xdma never completes -> hang, no error).
//
// To tolerate EITHER operand order, this kernel SWAPS when src_b < src_a: it bases the
// reader at the lower address and strides up to the higher. That is only valid because the
// StreamElementwise ops in use (MUL=0, ADD=1) are COMMUTATIVE -- op(a,b)==op(b,a) per beat
// and the output beat order is unchanged. Callers (e.g. bingo, whose allocator may place
// the broadcast operand below the input) therefore need NOT pre-order the operands.
//
// TODO (Solution B -- make this fully op-agnostic, incl. NON-commutative ops):
//   The swap only works for commutative ops. The clean order-preserving fix is a 1-line,
//   backward-compatible HW change in the reader AGU: SIGN-EXTEND the 32-bit temporal-stride
//   CSR word to the 48-bit stride instead of zero-extending it --
//     snax_cluster .../readerWriter/AddressGenUnit.scala, connectWithList():
//       temporalStrides(i) := remainingCSR.head
//         ->  temporalStrides(i) := remainingCSR.head.asSInt.pad(param.addressWidth).asUInt
//   Then a negative operand_stride (src_b below src_a) carries the right two's-complement
//   bits and `ptr + o*stride` wraps (UInt add, already truncating) onto the lower operand
//   WITHOUT reordering -- so any op and any layout work, at zero SW cost. A positive stride
//   sign-extends to itself, so existing callers are unaffected. Once that lands, drop the
//   swap below and just pass operand_stride = src_b - src_a unconditionally.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_stream_elementwise(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_stream_elementwise_args_t);
    if (snrt_is_dm_core()) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t *a = (uint32_t *)arg;
        uint64_t src_addr = make_u64(a[0], a[1]);
        uint64_t dst_addr = make_u64(a[2], a[3]);
        uint32_t beats          = a[4];
        uint32_t operand_stride = a[5];
        uint32_t operand_count  = a[6];
        uint32_t op             = a[7];
        uint32_t rows           = a[8];
        uint32_t csr_mode       = a[9];
        uint32_t dst_bound0     = a[10];
        uint32_t out_dtype      = a[11];
        uint32_t inv_scale      = a[12];
        uint32_t src_b_addr_hi  = a[13];
        uint32_t src_b_addr_lo  = a[14];
        uint32_t src_row_stride = a[15];
        bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_stream_elementwise_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        // As in stream_map: the quant lane is only used at out_dtype=1, so a cfg
        // without Fp16ToInt8 still runs every fp16 elementwise op.
        if (!BINGO_HAS_STREAMELEMENTWISE)
            BINGO_XDMA_EXT_UNSUPPORTED("xdma_stream_elementwise", "StreamElementwise");
        if (out_dtype == 1u && !BINGO_HAS_FP16TOINT8)
            BINGO_XDMA_EXT_UNSUPPORTED("xdma_stream_elementwise with out_dtype=int8", "Fp16ToInt8");
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
        // If a 2nd operand address is given, derive the operand stride from the two
        // separate buffers at run time. The reader AGU can only stride FORWARD, so the
        // base must be the LOWER of the two operands. If operand 1 (src_b) sits below
        // operand 0 (src_a) -- e.g. the bingo allocator places the broadcast operand
        // before the input -- swap them: read from the lower address and stride up to
        // the higher. Safe because the StreamElementwise op is commutative (MUL/ADD):
        // op(op0,op1) == op(op1,op0) per beat and the output beat order is unchanged.
        // (A non-commutative op would need real per-operand AGU bases / signed strides.)
        if (src_b_addr_lo || src_b_addr_hi) {
            uint32_t a_lo = (uint32_t)src_addr;
            if (src_b_addr_lo >= a_lo) {
                operand_stride = src_b_addr_lo - a_lo;
            } else {
                src_addr = make_u64(src_b_addr_hi, src_b_addr_lo);  // base = the lower operand
                operand_stride = a_lo - src_b_addr_lo;
            }
        }
        // Interleaved src. Two reader shapes:
        //   src_row_stride == 0 (default): 2D {operand, beat} over a FLAT [rows,D] tensor --
        //     the operands are packed, so one flat beat counter walks the whole tensor.
        //   src_row_stride != 0: 3D {operand, beat, row} over PADDED rows -- each row occupies
        //     src_row_stride bytes of which only the first `beats` beats are data. This is how
        //     an operand produced by the merged map+reduce in TAP mode is consumed: its rows are
        //     (beats+1)*64 bytes apart and the trailing scalar beat must be SKIPPED, which a flat
        //     reader cannot do. BOTH operands must share src_row_stride (so the interleave delta
        //     stays constant across rows) -- the broadcast operand is therefore written at the
        //     same padded row stride by the xDMA broadcast pass that builds it.
        // The writer stays 1D: the output is packed [rows,D] (or int8), never padded.
        uint32_t src_str[3] = { operand_stride, XDMA_WIDTH, src_row_stride };
        uint32_t src_bnd[3] = { operand_count, beats, rows };
        uint32_t src_dims   = 3u;
        if (src_row_stride == 0u) {          // flat: collapse the beat/row dims into one
            src_bnd[1] = rows * beats;
            src_dims   = 2u;
        }
        uint32_t csr_ew[2]  = { operand_count, op };
        xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISE, csr_ew);
        if (out_dtype == 1u) {
            uint32_t csr_q[1] = { inv_scale };
            xdma_enable_src_ext(READER_EXT_FP16TOINT8, csr_q);
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        uint32_t rc = xdma_stream_launch(src_addr, dst_addr, src_str, src_bnd, src_dims, dst_bound0, csr_mode);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
        if (out_dtype == 1u) xdma_disable_src_ext(READER_EXT_FP16TOINT8);
        xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISE);
        if (rc != BINGO_RET_SUCC) return BINGO_RET_FAIL;
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else {
        printf_safe("[Cluster %d Core %d]: Error! xDMA stream_elementwise should be called from a DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

// One 2-operand StreamElementwise pass for the fused rope kernel:
// dst = op(src_a, src_b) over rows*beats 64-byte beats (FULL config, no quant). The
// reader strides FORWARD only, so base at the LOWER operand and stride up to the
// higher -- valid because the ops used are commutative (MUL=0, ADD=1). Enables and
// disables READER_EXT_STREAMELEMENTWISE around the launch. See the standalone
// __snax_bingo_kernel_xdma_stream_elementwise above for the full rationale.
// Callers must check BINGO_HAS_STREAMELEMENTWISE first — this helper assumes the
// extension exists (on a cfg without it, its only caller returns before reaching here).
static inline uint32_t xdma_stream_ew2(uint64_t src_a, uint64_t src_b, uint64_t dst,
                                       uint32_t rows, uint32_t beats, uint32_t op)
{
    uint32_t a_lo = (uint32_t)src_a, b_lo = (uint32_t)src_b;
    uint64_t base = src_a;
    uint32_t operand_stride;
    if (b_lo >= a_lo) { operand_stride = b_lo - a_lo; }
    else { base = src_b; operand_stride = a_lo - b_lo; }
    uint32_t src_str[2] = { operand_stride, XDMA_WIDTH };
    uint32_t src_bnd[2] = { 2u, rows * beats };
    uint32_t csr_ew[2]  = { 2u, op };
    xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISE, csr_ew);
    uint32_t rc = xdma_stream_launch(base, dst, src_str, src_bnd, 2u, rows * beats, 0u /*FULL*/);
    xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISE);
    return rc;
}

// Fused FP16 RoPE (interleaved / complex-rotation convention) -- ALL DMA-engine ops:
//   xswap = adjacent fp16-pair swap of x       (iDMA, two strided 2-byte copies)
//   tmp1  = x     (.) cos_full                 (xDMA StreamElementwise MUL)
//   tmp2  = xswap (.) sin_signed               (xDMA StreamElementwise MUL)
//   out   = tmp1  (+) tmp2                      (xDMA StreamElementwise ADD)
// cos_full (cos duplicated per pair) and sin_signed (sin with alternating sign) are
// PRECOMPUTED tables -- the sign flip is not a DMA op. x is the runtime Q/K, so the
// swap is computed ON-DEVICE here (this is what enables in-layer rope_q/rope_k, which
// cannot precompute xswap). The scratch block [xswap | tmp1 | tmp2] is allocated from
// L1 here and freed on exit. Folds the standalone snax-xdma-rope flow into one kernel.
//
// Arg layout (uint32_t[]):
//   [0..1]  x_addr     (input,  fp16 [rows, D], D = beats*32 elems/row)
//   [2..3]  cos_addr   (cos_full table, same shape)
//   [4..5]  sin_addr   (sin_signed table, same shape)
//   [6..7]  out_addr   (output, same shape)
//   [8]     beats      (D = beats*32 fp16 per row)
//   [9]     rows       (rows / token positions)
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_rope(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_rope_args_t);
    if (snrt_is_dm_core()) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t *a = (uint32_t *)arg;
        uint64_t x_addr   = make_u64(a[0], a[1]);
        uint64_t cos_addr = make_u64(a[2], a[3]);
        uint64_t sin_addr = make_u64(a[4], a[5]);
        uint64_t out_addr = make_u64(a[6], a[7]);
        uint32_t beats = a[8];
        uint32_t rows  = a[9];
        bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_rope_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        // All three of rope's passes are StreamElementwise MUL/ADD; without the
        // extension there is no DMA-only path left, so refuse the call.
        if (!BINGO_HAS_STREAMELEMENTWISE)
            BINGO_XDMA_EXT_UNSUPPORTED("xdma_rope", "StreamElementwise");
        uint32_t row_beats = rows * beats;            // total 64-byte beats
        uint32_t tot_b     = row_beats * XDMA_WIDTH;  // bytes per [rows, D] fp16 buffer
        uint32_t num_elems = row_beats * 32u;         // fp16 elements (32 per 64-B beat)

        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
        // One L1 scratch block [xswap | tmp1 | tmp2], 64-byte aligned for the beat reads.
        uint32_t scratch_lo = snrt_l1_malloc(3u * tot_b + 64u);
        if (!scratch_lo) {
            printf_safe("[Cluster %d Core %d]: rope L1 scratch alloc failed!\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }
        uint32_t base_lo = (scratch_lo + 63u) & ~63u;
        uint64_t xswap = chiplet_addr_transform((uint64_t)base_lo);
        uint64_t tmp1  = xswap + tot_b;
        uint64_t tmp2  = xswap + 2u * tot_b;
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);

        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        // 1) xswap = adjacent fp16-pair swap of x (dst[i] = x[i^1]) on the iDMA.
        uint32_t pairs = num_elems / 2u;
        snrt_dma_start_2d_wideptr(xswap,      x_addr + 2u, 2u, 4u, 4u, pairs);  // xswap[2k]   = x[2k+1]
        snrt_dma_start_2d_wideptr(xswap + 2u, x_addr,      2u, 4u, 4u, pairs);  // xswap[2k+1] = x[2k]
        snrt_dma_wait_all();
        // 2) tmp1 = x (.) cos ; 3) tmp2 = xswap (.) sin ; 4) out = tmp1 (+) tmp2.
        uint32_t rc = xdma_stream_ew2(x_addr, cos_addr, tmp1, rows, beats, 0u /*MUL*/);
        if (rc == BINGO_RET_SUCC) rc = xdma_stream_ew2(xswap, sin_addr, tmp2, rows, beats, 0u /*MUL*/);
        if (rc == BINGO_RET_SUCC) rc = xdma_stream_ew2(tmp1, tmp2, out_addr, rows, beats, 1u /*ADD*/);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);

        snrt_l1_free(scratch_lo);
        if (rc != BINGO_RET_SUCC) {
            printf_safe("[Cluster %d Core %d]: rope StreamElementwise pass failed!\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }
        sp->return_value = (uint32_t)out_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else {
        printf_safe("[Cluster %d Core %d]: Error! xDMA rope should be called from a DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

// ==========================================================================
// High-level xDMA kernels: user provides shapes, kernel computes AGU config.
// These wrap the low-level reshape kernel with automatic stride computation.
// ==========================================================================

// SW 2D transpose [M,N] -> [N,M] at element granularity. Correct for any
// element width — used for widths the HW transposer has no native mode for
// (int32) and when the Transposer extension is absent. The DM-core CPU loop is
// not coherent with L3, so non-local (L3) operands are staged through L1 (src
// via xdma_layout_stage_in, dst via xdma_layout_stage_out + flush), matching the
// HW path's L1-or-L3 contract. Returns BINGO_RET_SUCC, or BINGO_RET_FAIL on L1
// staging-alloc failure.
static inline uint32_t xdma_cpu_transpose_2d(uint64_t src_addr, uint64_t dst_addr,
                                             uint32_t M, uint32_t N, uint32_t elem_bytes)
{
    uint32_t bytes = M * N * elem_bytes;
    xdma_layout_stage_t si;
    xdma_layout_stage_out_t so;
    if (xdma_layout_stage_in(&si, src_addr, bytes) != 0 ||
        xdma_layout_stage_out(&so, dst_addr, bytes) != 0) {
        xdma_layout_stage_free(&si);   // frees src scratch if staged; no-op otherwise
        return BINGO_RET_FAIL;
    }
    volatile uint8_t *src = (volatile uint8_t *)(uint32_t)si.xdma_src;
    volatile uint8_t *dst = (volatile uint8_t *)so.l1;
    for (uint32_t r = 0; r < M; r++)
        for (uint32_t c = 0; c < N; c++)
            for (uint32_t b = 0; b < elem_bytes; b++)
                dst[(c * M + r) * elem_bytes + b] = src[(r * N + c) * elem_bytes + b];
    xdma_layout_stage_out_flush(&si, &so, dst_addr, bytes);   // transfer + flush + free both
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_transpose_2d(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_transpose_2d_args_t);
    // Transpose a 2D matrix [M, N] -> [N, M].
    // Element-width dispatch:
    //   - int8  (elem_bytes=1): HW transposer, 8-bit mode  (CSR0=0)
    //   - int16 (elem_bytes=2): HW transposer, 16-bit mode (CSR0=1)
    //   - int32 (elem_bytes=4): SW transpose (no native 32-bit HW mode)
    // When the Transposer extension is absent (WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16
    // undefined), all widths use the SW transpose.
    //
    // HW path constraints: M % 8 == 0, N * elem_bytes % 8 == 0.
    //
    // Arg layout (uint32_t[]):
    //   [0]  src_addr_hi
    //   [1]  src_addr_lo
    //   [2]  dst_addr_hi
    //   [3]  dst_addr_lo
    //   [4]  M            (source rows)
    //   [5]  N            (source cols)
    //   [6]  elem_bytes   (1=int8, 2=int16, 4=int32)

    if (snrt_is_dm_core())
    {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t *a = (uint32_t *)arg;
        uint64_t src_addr = make_u64(a[0], a[1]);
        uint64_t dst_addr = make_u64(a[2], a[3]);
        uint32_t M = a[4];
        uint32_t N = a[5];
        uint32_t elem_bytes = a[6];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_transpose_2d_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

#ifdef WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16
        // ── HW Transposer path (8-bit / 16-bit native modes) ──
        // The 8x8 Transposer in the writer datapath has two native element-width
        // modes, selected by CSR0 (0 = 8-bit, 1 = 16-bit; from the cfg's
        // elementWidth:[8,16]). In a native mode the transposer does the
        // element-aware transpose internally (assembling tpt beats per 8x8
        // tile), so it works on the writer side. 32-bit has NO native mode — and
        // the byte-mode multi-beat compose that would emulate it does not hold on
        // the writer side — so 32-bit falls through to the SW transpose below.
        //
        //   tile_width = 8; tpt = ceil(8*8*elem_bits/512) beats per 8x8 tile
        //   spatial_stride_src = N*elem_bytes; spatial_stride_dst = M*elem_bytes
        //   src strides: [8, tile_w*elem_bytes, N*tile_w*elem_bytes]
        //   dst strides: [8, M*tile_w*elem_bytes, tile_w*elem_bytes]
        //
        // ── Working principle: the Transposer writes into the writer's LOCAL
        //    memory, so a fused transpose+transfer is a READ-from-another-memory,
        //    never a write-into-another-memory ─────────────────────────────────
        // The writer emits its 8 spatial channels at spatial_stride_dst
        // (= M*elem_bytes) — one transposed row-slice per channel — while the
        // Transposer reorders the bytes WITHIN each 8x8 tile. That per-channel
        // scatter is realized as 8 independently-addressed requests ONLY when the
        // writer targets its OWN LOCAL TCDM ("only local write", XDMACtrl). So the
        // transpose is fused correctly into the data move exactly when the move
        // READS from another memory (a remote cluster's TCDM, or L3) and WRITES
        // the transposed result into LOCAL TCDM. See the worked example
        //   target/sw/device/apps/snax/snax-xdma-transpose
        // — its experiment group sets src = a *remote* cluster's TCDM
        // (tcdm_baseaddress + cluster_offset) and dst = its *local* TCDM, enables
        // this writer Transposer, and the host check is byte-exact.
        //
        // The mirror image — transposing straight INTO another memory (writing the
        // transpose out to L3 / off-cluster) — is not a fused operation the cluster
        // xDMA can express: there is no local writer at that destination, so the
        // writer's per-channel scatter cannot be placed there and the global write
        // is drained as one CONTIGUOUS AXI burst
        // (xdma_axi_adapter/xdma_burst_reshaper). The byte transpose then degrades
        // to 8-byte-word granularity (only the 8-byte-aligned columns, c%8==0, come
        // out transposed). Therefore "transpose then move OUT to L3" must be split
        // into two steps: transpose into LOCAL memory, then a plain move out.
        //
        // SW BYPASS (implemented below): when dst is non-local, transpose into a
        // LOCAL L1 scratch (scatter into local TCDM = correct), then a plain
        // CONTIGUOUS copy L1->dst (xdma_layout_stage_out + _flush). A contiguous
        // move carries no per-channel scatter, so it is exact to any memory. For a
        // local dst this is a zero-copy passthrough (the fused path above,
        // unchanged). Cost: one L1 scratch (M*N*elem_bytes) + an L1->dst 1D DMA.
        //
        // To instead fuse transpose+write-to-L3 in a single op (NOT done here;
        // needs RTL/gen changes) one of:
        //  (1) Give the mem-system xDMA a Transposer. hemaia_mem_system.sv's
        //      hemaia_xdma writes L3 through its OWN local tcdm_req_o, so ITS
        //      writes are local/scatter-capable. It has no Transposer today
        //      (hemaia_xdma_cfg.writer_extensions is empty) but is built by the
        //      SAME snax.xdma.xdmaTop.XDMATopGen as this cluster xDMA, so adding
        //      `HasTransposer:{row:[8,8],col:[8,8],elementWidth:[8,16]}` to
        //      hemaia_xdma_cfg.writer_extensions + regen lets the HOST
        //      (hemaia-xdma-lib.h) drive the L3-side engine to READ L1 + transpose
        //      + write L3 locally — i.e. L3 reading L1 and transposing into itself.
        //  (2) Carry spatial_stride_dst (+ per-channel strobes) on the xDMA
        //      toRemote descriptor and have xdma_axi_adapter emit scattered AW +
        //      real w_strb instead of a contiguous burst (changes the
        //      cross-cluster write protocol).
        if (elem_bytes == 1 || elem_bytes == 2) {
            uint32_t tile_w = 8;
            uint32_t tpt = (tile_w * tile_w * (elem_bytes * 8) + 511) / 512; // transfers per transpose (8x8 tile of elem_bytes*8-bit elems / 512b bus)

            // The xDMA reader is tied to local L1, so stage src into local L1 if
            // it isn't already there (zero-copy fast path when src is already
            // local).
            uint32_t bytes = M * N * elem_bytes;
            xdma_layout_stage_t st;
            if (xdma_layout_stage_in(&st, src_addr, bytes) != 0) {
                printf_safe("[Cluster %d Core %d]: transpose_2d L1 alloc failed!\r\n",
                            snrt_cluster_idx(), snrt_cluster_core_idx());
                return BINGO_RET_FAIL;
            }

            // SW BYPASS: the writer must scatter into LOCAL memory for a correct
            // byte transpose. If dst is non-local, transpose into a local L1
            // scratch and flush it contiguously to dst afterwards; if dst is
            // local, write straight through (so = zero-copy passthrough).
            bool dst_local = xdma_addr_in_local_l1(dst_addr);
            xdma_layout_stage_out_t so;
            uint64_t xpose_dst = dst_addr;
            if (!dst_local) {
                if (xdma_layout_stage_out(&so, dst_addr, bytes) != 0) {
                    xdma_layout_stage_free(&st);
                    printf_safe("[Cluster %d Core %d]: transpose_2d bypass L1 alloc failed!\r\n",
                                snrt_cluster_idx(), snrt_cluster_core_idx());
                    return BINGO_RET_FAIL;
                }
                // Writer target = the local scratch (chiplet-transformed like the
                // staged reader src), so the write scatters into local L1.
                xpose_dst = chiplet_addr_transform((uint64_t)so.l1);
                XDMA_DEBUG_PRINT("[transpose_2d] non-local dst 0x%llx -> local-scratch bypass\r\n",
                                 (unsigned long long)dst_addr);
            }

            // Disable all, then enable the transposer on the writer side.
            // CSR0 selects the element width the transposer transposes at:
            // 0 = 8-bit (int8), 1 = 16-bit (int16).
            BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
            xdma_disable_all_extensions();
            uint32_t tp_csr[1] = { (elem_bytes == 2) ? 1u : 0u };
            xdma_enable_dst_ext(WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16, tp_csr);

            uint32_t spatial_stride_src = N * elem_bytes;
            uint32_t spatial_stride_dst = M * elem_bytes;

            uint32_t t_strides_src[3] = {
                8,                          // dim0: within one transfer
                tile_w * elem_bytes,              // dim1: next tile horizontally
                N * tile_w * elem_bytes           // dim2: next tile-row
            };
            uint32_t t_bounds_src[3] = {
                tpt,                        // transfers per 8x8 tile
                N / tile_w,                 // tiles across columns
                M / tile_w                  // tiles across rows
            };

            // Writer: transposed tile placement
            // After transpose, each 8x8 block's rows/cols are swapped.
            // dim1 stride = M*tile_w*elem_bytes (stride to next column-block in transposed output)
            // dim2 stride = tile_w*elem_bytes   (stride to next row-block in transposed output)
            uint32_t t_strides_dst[3] = {
                8,                          // dim0: within one transfer
                M * tile_w * elem_bytes,          // dim1: next column-block in output
                tile_w * elem_bytes               // dim2: next row-block in output
            };
            uint32_t t_bounds_dst[3] = {
                tpt,
                N / tile_w,
                M / tile_w
            };

            // Transpose into xpose_dst (the local scratch when bypassing, else dst).
            xdma_memcpy_nd_full_addr(
                st.xdma_src, xpose_dst,
                spatial_stride_src, spatial_stride_dst,
                3, t_strides_src, t_bounds_src,
                3, t_strides_dst, t_bounds_dst,
                0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
            );
            BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);

            BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
            xdma_task_t task_id = xdma_start();
            xdma_wait_task(task_id);
            BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);

            // Disable transposer after use
            xdma_disable_dst_ext(WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16);
            if (dst_local) {
                xdma_layout_stage_free(&st);              // src staging only
            } else {
                // Contiguous copy local scratch -> real (non-local) dst, then free
                // both the dst scratch and the src staging buffer.
                xdma_layout_stage_out_flush(&st, &so, dst_addr, bytes);
            }
            sp->return_value = (uint32_t)dst_addr;
            sp->num_return_values = 0;
            return BINGO_RET_SUCC;
        }
#endif
        // ── SW transpose ──
        // Correct for any element width; reached for 32-bit (no native HW
        // transposer mode) or when the Transposer extension is absent.
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        uint32_t sw_rc = xdma_cpu_transpose_2d(src_addr, dst_addr, M, N, elem_bytes);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
        if (sw_rc != BINGO_RET_SUCC) {
            printf_safe("[Cluster %d Core %d]: transpose_2d CPU-fallback L1 alloc failed!\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else {
        printf_safe("[Cluster %d Core %d]: Error! transpose_2d should be called from DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_submatrix_2d(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_submatrix_2d_args_t);
    // Extract a sub-region A[row_start:row_end, col_start:col_end] from [src_rows, src_cols].
    // Tile-level: 8 spatial channels span 8 consecutive rows, temporal dims
    // iterate within rows (by 8 bytes) and across groups of 8 rows.
    // Constraints: out_rows % 8 == 0, out_cols*elem_bytes % 8 == 0, col_start*elem_bytes % 8 == 0.
    //
    // Arg layout (uint32_t[]):
    //   [0]  src_addr_hi
    //   [1]  src_addr_lo
    //   [2]  dst_addr_hi
    //   [3]  dst_addr_lo
    //   [4]  src_rows
    //   [5]  src_cols
    //   [6]  row_start     (inclusive)
    //   [7]  row_end       (exclusive)
    //   [8]  col_start     (inclusive)
    //   [9]  col_end       (exclusive)
    //   [10] elem_bytes

    if (snrt_is_dm_core())
    {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t *a = (uint32_t *)arg;
        uint64_t src_addr = make_u64(a[0], a[1]);
        uint64_t dst_addr = make_u64(a[2], a[3]);
        uint32_t src_cols  = a[5];
        uint32_t row_start = a[6];
        uint32_t row_end   = a[7];
        uint32_t col_start = a[8];
        uint32_t col_end   = a[9];
        uint32_t elem_bytes      = a[10];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_submatrix_2d_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

        uint32_t out_rows = row_end - row_start;
        uint32_t out_cols = col_end - col_start;
        uint32_t src_row_bytes = src_cols * elem_bytes;
        uint32_t out_row_bytes = out_cols * elem_bytes;

        // Offset source to start of sub-region
        src_addr += (uint64_t)(row_start * src_cols + col_start) * elem_bytes;

        // Spatial: 8 channels across 8 consecutive rows
        uint32_t spatial_stride_src = src_row_bytes;
        uint32_t spatial_stride_dst = out_row_bytes;

        // Temporal: dim0 within row (8-byte chunks), dim1 across groups of 8 rows
        uint32_t t_strides_src[2] = { 8, src_row_bytes * 8 };
        uint32_t t_bounds_src[2]  = { out_row_bytes / 8, out_rows / 8 };
        uint32_t t_strides_dst[2] = { 8, out_row_bytes * 8 };
        uint32_t t_bounds_dst[2]  = { out_row_bytes / 8, out_rows / 8 };


        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
        xdma_disable_all_extensions();
        BINGO_XDMA_TRY(xdma_memcpy_nd_full_addr(
            src_addr, dst_addr,
            spatial_stride_src, spatial_stride_dst,
            2, t_strides_src, t_bounds_src,
            2, t_strides_dst, t_bounds_dst,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        ), "xdma_submatrix_2d");
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);

        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        xdma_task_t task_id = xdma_start();
        xdma_wait_task(task_id);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else {
        printf_safe("[Cluster %d Core %d]: Error! xDMA submatrix_2d should be called from DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_expand_2d(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_expand_2d_args_t);
    // Broadcast a single row [1, N] to [M, N] by repeating it M times.
    // Tile-level: spatial_stride_src=0 makes all 8 channels read the same row.
    // 8 dst channels write to 8 consecutive output rows.
    // Constraints: M % 8 == 0, N*elem_bytes % 8 == 0.
    //
    // Arg layout (uint32_t[]):
    //   [0]  src_addr_hi
    //   [1]  src_addr_lo
    //   [2]  dst_addr_hi
    //   [3]  dst_addr_lo
    //   [4]  M            (number of output rows / broadcast factor)
    //   [5]  N            (row width, shared by src and dst)
    //   [6]  elem_bytes

    if (snrt_is_dm_core())
    {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t *a = (uint32_t *)arg;
        uint64_t src_addr = make_u64(a[0], a[1]);
        uint64_t dst_addr = make_u64(a[2], a[3]);
        uint32_t M    = a[4];
        uint32_t N    = a[5];
        uint32_t elem_bytes = a[6];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_expand_2d_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

        uint32_t row_bytes = N * elem_bytes;

        // Spatial: src stride=0 (all channels read same row), dst stride=row_bytes
        uint32_t spatial_stride_src = 0;
        uint32_t spatial_stride_dst = row_bytes;

        // Temporal: dim0 within row, dim1 across groups of 8 output rows
        uint32_t t_strides_src[2] = { 8, 0 };
        uint32_t t_bounds_src[2]  = { row_bytes / 8, M / 8 };
        uint32_t t_strides_dst[2] = { 8, row_bytes * 8 };
        uint32_t t_bounds_dst[2]  = { row_bytes / 8, M / 8 };


        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
        xdma_disable_all_extensions();
        BINGO_XDMA_TRY(xdma_memcpy_nd_full_addr(
            src_addr, dst_addr,
            spatial_stride_src, spatial_stride_dst,
            2, t_strides_src, t_bounds_src,
            2, t_strides_dst, t_bounds_dst,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        ), "xdma_expand_2d");
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);

        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        xdma_task_t task_id = xdma_start();
        xdma_wait_task(task_id);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else {
        printf_safe("[Cluster %d Core %d]: Error! xDMA expand_2d should be called from DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_concat_2d(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_concat_2d_args_t);
    // Copy one input chunk to an offset position in a larger output tensor.
    // Concat of N inputs = N invocations, each placing its chunk at a different offset.
    //
    // Arg layout (uint32_t[]):
    //   [0] src_addr_hi  [1] src_addr_lo
    //   [2] dst_addr_hi  [3] dst_addr_lo
    //   [4] src_rows     [5] src_cols
    //   [6] dst_rows     [7] dst_cols
    //   [8] axis         (0=row, 1=col)
    //   [9] offset       (element offset along axis)
    //   [10] elem_bytes

    if (snrt_is_dm_core())
    {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t *a = (uint32_t *)arg;
        uint64_t src_addr = make_u64(a[0], a[1]);
        uint64_t dst_addr = make_u64(a[2], a[3]);
        uint32_t src_rows = a[4];
        uint32_t src_cols = a[5];
        uint32_t dst_rows = a[6];
        uint32_t dst_cols = a[7];
        uint32_t axis     = a[8];
        uint32_t offset   = a[9];
        uint32_t elem_bytes     = a[10];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_concat_2d_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

        // Apply offset to destination base address
        if (axis == 0) {
            // Row concat: offset entire rows
            dst_addr += (uint64_t)offset * dst_cols * elem_bytes;
        } else {
            // Column concat: offset within each row
            dst_addr += (uint64_t)offset * elem_bytes;
        }

        uint32_t src_row_bytes = src_cols * elem_bytes;
        uint32_t dst_row_bytes = dst_cols * elem_bytes;

        // Spatial: 8 channels across 8 consecutive rows
        uint32_t spatial_stride_src = src_row_bytes;
        uint32_t spatial_stride_dst = dst_row_bytes;

        // Temporal: dim0 within row, dim1 across groups of 8 rows
        uint32_t t_strides_src[2] = { 8, src_row_bytes * 8 };
        uint32_t t_bounds_src[2]  = { src_row_bytes / 8, src_rows / 8 };
        uint32_t t_strides_dst[2] = { 8, dst_row_bytes * 8 };
        uint32_t t_bounds_dst[2]  = { src_row_bytes / 8, src_rows / 8 };


        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
        xdma_disable_all_extensions();
        BINGO_XDMA_TRY(xdma_memcpy_nd_full_addr(
            src_addr, dst_addr,
            spatial_stride_src, spatial_stride_dst,
            2, t_strides_src, t_bounds_src,
            2, t_strides_dst, t_bounds_dst,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        ), "xdma_concat_2d");
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);

        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        xdma_task_t task_id = xdma_start();
        xdma_wait_task(task_id);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else {
        printf_safe("[Cluster %d Core %d]: Error! xDMA concat_2d should be called from DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_pad_2d(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_pad_2d_args_t);
    // Zero-fill output, then strided-copy source into the padded interior.
    // Phase 1: CPU scalar memset (AGU-only, no Memset extension)
    // Phase 2: xDMA strided copy to padded interior
    //
    // Arg layout (uint32_t[]):
    //   [0] src_addr_hi  [1] src_addr_lo
    //   [2] dst_addr_hi  [3] dst_addr_lo
    //   [4] src_rows     [5] src_cols
    //   [6] pad_top      [7] pad_bottom
    //   [8] pad_left     [9] pad_right
    //   [10] elem_bytes

    if (snrt_is_dm_core())
    {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t *a = (uint32_t *)arg;
        uint64_t src_addr = make_u64(a[0], a[1]);
        uint64_t dst_addr = make_u64(a[2], a[3]);
        uint32_t src_rows   = a[4];
        uint32_t src_cols   = a[5];
        uint32_t pad_top    = a[6];
        uint32_t pad_bottom = a[7];
        uint32_t pad_left   = a[8];
        uint32_t pad_right  = a[9];
        uint32_t elem_bytes       = a[10];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_pad_2d_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

        uint32_t dst_rows = src_rows + pad_top + pad_bottom;
        uint32_t dst_cols = src_cols + pad_left + pad_right;
        uint32_t total_bytes = dst_rows * dst_cols * elem_bytes;

        // Phase 1: CPU zero-fill the entire output buffer
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
        volatile uint32_t *dst32 = (volatile uint32_t *)(uint32_t)dst_addr;
        for (uint32_t i = 0; i < total_bytes / 4; i++) {
            dst32[i] = 0;
        }
        // Handle remaining bytes
        volatile uint8_t *dst8 = (volatile uint8_t *)(uint32_t)dst_addr;
        for (uint32_t i = (total_bytes / 4) * 4; i < total_bytes; i++) {
            dst8[i] = 0;
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);

        // Phase 2: xDMA strided copy of source into padded interior
        uint64_t dst_interior = dst_addr + (uint64_t)(pad_top * dst_cols + pad_left) * elem_bytes;

        uint32_t src_row_bytes = src_cols * elem_bytes;
        uint32_t dst_row_bytes = dst_cols * elem_bytes;

        // Spatial: 8 channels across 8 consecutive rows
        uint32_t spatial_stride_src = src_row_bytes;
        uint32_t spatial_stride_dst = dst_row_bytes;

        // Temporal: dim0 within row, dim1 across groups of 8 rows
        uint32_t t_strides_src[2] = { 8, src_row_bytes * 8 };
        uint32_t t_bounds_src[2]  = { src_row_bytes / 8, src_rows / 8 };
        uint32_t t_strides_dst[2] = { 8, dst_row_bytes * 8 };
        uint32_t t_bounds_dst[2]  = { src_row_bytes / 8, src_rows / 8 };


        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
        xdma_disable_all_extensions();
        BINGO_XDMA_TRY(xdma_memcpy_nd_full_addr(
            src_addr, dst_interior,
            spatial_stride_src, spatial_stride_dst,
            2, t_strides_src, t_bounds_src,
            2, t_strides_dst, t_bounds_dst,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        ), "xdma_pad_2d");
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);

        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        xdma_task_t task_id = xdma_start();
        xdma_wait_task(task_id);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else {
        printf_safe("[Cluster %d Core %d]: Error! xDMA pad_2d should be called from DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_gather_2d(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_gather_2d_args_t);
    // Gather rows by arithmetic stride: src[start], src[start+stride], ...
    // Source reads with large temporal stride (skips rows); destination writes contiguously.
    //
    // Arg layout (uint32_t[]):
    //   [0] src_addr_hi  [1] src_addr_lo
    //   [2] dst_addr_hi  [3] dst_addr_lo
    //   [4] src_rows      (total rows in source, for bounds checking)
    //   [5] src_cols      (cols per row)
    //   [6] num_indices   (number of rows to gather)
    //   [7] index_start   (first row index)
    //   [8] index_stride  (stride between indices; 1=contiguous)
    //   [9] elem_bytes

    if (snrt_is_dm_core())
    {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t *a = (uint32_t *)arg;
        uint64_t src_addr = make_u64(a[0], a[1]);
        uint64_t dst_addr = make_u64(a[2], a[3]);
        uint32_t src_rows     = a[4];
        uint32_t src_cols     = a[5];
        uint32_t num_indices  = a[6];
        uint32_t index_start  = a[7];
        uint32_t index_stride = a[8];
        uint32_t elem_bytes         = a[9];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_gather_2d_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

        // Offset source to the first gathered row
        uint64_t src_base = src_addr + (uint64_t)index_start * src_cols * elem_bytes;

        uint32_t row_bytes = src_cols * elem_bytes;

        // Spatial: src channels span gathered rows, dst channels span consecutive rows
        uint32_t spatial_stride_src = index_stride * row_bytes;
        uint32_t spatial_stride_dst = row_bytes;

        // Temporal: dim0 within row, dim1 across groups of 8 gathered rows
        uint32_t t_strides_src[2] = { 8, spatial_stride_src * 8 };
        uint32_t t_bounds_src[2]  = { row_bytes / 8, num_indices / 8 };
        uint32_t t_strides_dst[2] = { 8, row_bytes * 8 };
        uint32_t t_bounds_dst[2]  = { row_bytes / 8, num_indices / 8 };


        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
        xdma_disable_all_extensions();
        BINGO_XDMA_TRY(xdma_memcpy_nd_full_addr(
            src_base, dst_addr,
            spatial_stride_src, spatial_stride_dst,
            2, t_strides_src, t_bounds_src,
            2, t_strides_dst, t_bounds_dst,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        ), "xdma_gather_2d");
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);

        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        xdma_task_t task_id = xdma_start();
        xdma_wait_task(task_id);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
        sp->return_value = (uint32_t)dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else {
        printf_safe("[Cluster %d Core %d]: Error! xDMA gather_2d should be called from DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

// ==========================================================================
// VersaCore blocked-layout conversion kernels (tile-shape-parameterized)
//
// Each kernel converts between row-major (logical 2D) and one of the three
// VersaCore blocked layouts {A, B, D}. Arguments are parameterized by the
// scheduler's tile dimensions so the same kernel works for any DSE-chosen
// tiling. See HeMAiA/util/sim/xdma/layout_convert.py for the Python reference.
//
// Layout definitions (elem_bytes=1 for INT8, 4 for INT32/FP32):
//   A-layout [M_T, K_T, meshRow, tileSize]:
//     A[m,k,r,s] ↔ R[m*meshRow+r, k*tileSize+s]
//   B-layout [N_T, K_T, meshCol, tileSize]:
//     B[n,k,c,s] ↔ R[k*tileSize+s, n*meshCol+c]
//   D-layout [M_T, N_T, meshRow, meshCol]:
//     D[m,n,r,c] ↔ R[m*meshRow+r, n*meshCol+c]
//
// ─── xDMA AGU constraints (recap) ─────────────────────────────────────────
//   XDMA_SPATIAL_CHAN = 8     (8 hardware channels, fixed)
//   bytes / channel beat = 8  (each channel transfers 8 contiguous bytes)
//   XDMA_{SRC,DST}_TEMP_DIM = 5 each (≤ 5 temporal dims per side)
//   spatial_stride may be 0 (broadcast, e.g. xdma_expand_2d) or any value
//     ≥ 8 bytes; consecutive channels' 8-byte beats must not overlap.
//   HW Transposer extension (defined(WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16)) is
//     fixed at an 8x8-byte block; element-aware widths (uint16/uint32) are
//     handled by issuing tpt = (8·8·elem_bits)/512 beats per logical block,
//     mirroring __snax_bingo_kernel_xdma_transpose_2d.
//
// xDMA's reader is tied to this cluster's L1; its writer addresses the global
// space. Only the source needs L1 staging when not local — the helper
// `xdma_layout_stage_in` (snax_xdma_lib.h) handles that, and xDMA writes the
// layout-transformed result directly to the user's dst (no stage-out).
//
// ─── Spatial-axis decision tree per kernel family ─────────────────────────
// The 4-axis loop (m, k|n, r, s|c) is mapped onto (spatial=8) x ≤5 temporal
// dims. We pick the spatial axis at runtime; the first matching path wins.
//
// A↔R kernels — axes (m, k, r, s); inner row = tileSize·elem_bytes bytes:
//   Path 1: meshRow == 8 && (tileSize·elem_bytes) % 8 == 0     spatial = r
//   Path 2: meshRow > 8 && meshRow % 8 == 0 && …%8 == 0  spatial = r_inner
//   Path 3: meshRow ∈ {1,2,4} && (tileSize·elem_bytes) % 64==0 spatial = s_chunk_inner
//   Path 4: (tileSize·elem_bytes) %8==0 && K_T %8==0           spatial = k_inner
//   Path 5: (tileSize·elem_bytes) %8==0 && M_T %8==0           spatial = m_inner
//   else  : CPU fallback
//
// D↔R kernels — axes (m, n, r, c); inner row = meshCol·elem_bytes bytes:
//   Path 1: meshRow == 8 && (meshCol·elem_bytes) % 8 == 0      spatial = r
//   Path 2: meshRow > 8 && meshRow % 8 == 0 && …%8 == 0  spatial = r_inner
//   Path 3: meshRow ∈ {1,2,4} && (meshCol·elem_bytes) % 64==0  spatial = c_chunk_inner
//   Path 4: (meshCol·elem_bytes) %8==0 && N_T %8==0            spatial = n_inner
//   Path 5: (meshCol·elem_bytes) %8==0 && M_T %8==0            spatial = m_inner
//   else  : CPU fallback
//
// B↔R kernels — axes (n, k, c, s); HW Transposer + AGU sub-block iteration:
//   HW path: defined(WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16) && tileSize %8==0 && meshCol %8==0
//     For each (n,k) tile, decompose into (meshCol/8) x (tileSize/8) element
//     sub-blocks; each sub-block goes through one HW Transposer block. The
//     AGU iterates tpt x c_sub x s_sub x k x n  → 5 temporal dims (max).
//   else  : CPU fallback
//
// ─── Resulting coverage matrix ────────────────────────────────────────────
// The mesh and elem_bytes are COMPILE-TIME constants of each kernel (see the wrapper
// blocks below), so the decision trees above fold: every kernel below IS exactly one
// path. This table therefore names kernels, not runtime outcomes. It is DERIVED from
// the conditions above -- if you change them, regenerate it.
//
// array_shape (mR,tS,mC) = snax_versacore_to_cluster.hjson spatial_unrolling[0]:
//   idx 0 = (32,2,32)   idx 1 = (1,16,32)   idx 2 = (16,8,16)
// elem_bytes: e1 = INT8, e2 = FP16, e4 = INT32/FP32.
//
//   e#  shape (mR,tS,mC) | A<->R                            | D<->R                            | B<->R
//   ----------------------------------------------------------------------------------------------------------
//   e1  0 (32, 2, 32)    | _M32K2  CPU (1)                  | _M32N32 HW p2                    | _K2N32  CPU (2)
//   e1  1 (1, 16, 32)    | _M1K16  HW p4/p5 (3)             | _M1N32  HW p4/p5 (3)             | _K16N32 HW
//   e1  2 (16, 8, 16)    | _M16K8  HW p2                    | _M16N16 HW p2                    | _K8N16  HW
//   e2  0 (32, 2, 32)    | _M32K2  CPU (1)                  | _M32N32 HW p2                    | _K2N32  CPU (2)
//   e2  1 (1, 16, 32)    | _M1K16  HW p4/p5 (3)             | _M1N32  HW p3                    | _K16N32 HW
//   e2  2 (16, 8, 16)    | _M16K8  HW p2                    | _M16N16 HW p2                    | _K8N16  HW
//   e4  0 (32, 2, 32)    | _M32K2  HW p2                    | _M32N32 HW p2                    | _K2N32  CPU (2)
//   e4  1 (1, 16, 32)    | _M1K16  HW p3                    | _M1N32  HW p3                    | _K16N32 HW
//   e4  2 (16, 8, 16)    | _M16K8  HW p2                    | _M16N16 HW p2                    | _K8N16  HW
//
//   (1) A-tile inner row = tileSize*elem_bytes = 2 (e1) or 4 (e2) < 8 bytes; cannot form
//       an 8-byte beat contiguous in both row-major src and packed A dst. CPU.
//   (2) Same root cause: tileSize=2 prevents 8x8-byte sub-blocking of the B-tile, for
//       every elem_bytes -- the HW Transposer needs tileSize%8 and meshCol%8.
//   (3) meshRow=1 with a 16-byte inner row misses paths 1-3, so it needs K_T%8 (p4) or
//       M_T%8 (p5), else CPU. These are the ONLY kernels whose path still depends on a
//       runtime tile count; everything else is decided at compile time.
//   CPU paths run on the DM core, which is not coherent with L3, so they stage
//   non-local (L3) operands through L1 via xdma_cpu_stage_* (snax_xdma_lib.h).
//
// NOTE: this table used to list five array shapes -- (32,2,32), (1,8,64), (4,8,64),
// (8,8,64), (8,32,8) -- which are NOT this RTL build's. Corrected against the cfg.
// ==========================================================================

// Dispatch helper for the layout-convert HW paths: configures an AGU-only
// or AGU+Transposer xDMA transfer with the given strides/bounds, kicks it,
// and waits for completion. Keeps the per-path bodies short.
static inline void xdma_layout_run(
    uint64_t src, uint64_t dst,
    uint32_t spatial_stride_src, uint32_t spatial_stride_dst,
    uint32_t ndims,
    uint32_t *t_strides_src, uint32_t *t_bounds_src,
    uint32_t *t_strides_dst, uint32_t *t_bounds_dst,
    bool use_transposer)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
    xdma_disable_all_extensions();
#ifdef WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16
    uint32_t tp_csr[1] = {0};  // 8x8-byte (8-bit) transpose mode
    if (use_transposer)
        xdma_enable_dst_ext(WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16, tp_csr);
#else
    (void)use_transposer;
#endif
    xdma_memcpy_nd_full_addr(
        src, dst,
        spatial_stride_src, spatial_stride_dst,
        ndims, t_strides_src, t_bounds_src,
        ndims, t_strides_dst, t_bounds_dst,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
    xdma_task_t task_id = xdma_start();
    xdma_wait_task(task_id);
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);

#ifdef WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16
    if (use_transposer) xdma_disable_dst_ext(WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16);
#endif
}

// D-layout → row-major. See section banner for the path table.
//   Coverage: paths 1–5 cover all 5 VersaCore array_shapes via paths 1–3.
//   Arg layout: src_hi/lo, dst_hi/lo, M_T, N_T, meshRow, meshCol, elem_bytes.
//   Strides used by all paths (factored out for readability):
//     row_bytes_dst       = N_T * meshCol * elem_bytes   (full row of row-major R)
//     tile_bytes_dst_skip = meshRow * row_bytes_dst (advance past meshRow rows)
//     tile_bytes_src      = meshRow * meshCol * elem_bytes (one D-tile worth of bytes)
//     row_bytes_src       = meshCol * elem_bytes         (one row inside a D-tile)
static inline uint32_t __xdma_d_to_row_major_impl(void *arg, uint32_t meshRow, uint32_t meshCol, uint32_t elem_bytes)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_d_to_row_major_args_t);
    if (!snrt_is_dm_core()) {
        printf_safe("[Cluster %d Core %d]: Error! d_to_row_major must be called from DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t *a = (uint32_t *)arg;
    uint64_t src_addr = make_u64(a[0], a[1]);
    uint64_t dst_addr = make_u64(a[2], a[3]);
    uint32_t M_T = a[4], N_T = a[5];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_d_to_row_major_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    uint32_t bytes     = M_T * N_T * meshRow * meshCol * elem_bytes;
    uint32_t row_b_src = meshCol * elem_bytes;          // 1 row of a D-tile
    uint32_t row_b_dst = N_T * meshCol * elem_bytes;    // 1 row of row-major R
    uint32_t tile_b    = meshRow * meshCol * elem_bytes;
    bool hw_done = false;

    // Decide which HW path applies before staging, so we don't allocate L1
    // for a transfer that ends up on the CPU loop.
    int path = 0;
    if ((row_b_src % 8) == 0) {
        if (meshRow == 8)                                                       path = 1;
        else if (meshRow > 8 && (meshRow % 8) == 0)                             path = 2;
        else if ((meshRow == 1 || meshRow == 2 || meshRow == 4)
                 && (row_b_src % 64) == 0)                                      path = 3;
        else if ((N_T % 8) == 0)                                                path = 4;
        else if ((M_T % 8) == 0)                                                path = 5;
    }

    if (path != 0) {
        xdma_layout_stage_t st;
        if (xdma_layout_stage_in(&st, src_addr, bytes) != 0) {
            printf_safe("[Cluster %d Core %d]: d_to_row_major L1 alloc failed!\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }
        uint32_t inner_beats = row_b_src / 8;       // 8-byte chunks per tile-row

        if (path == 1) {
            // Path 1: spatial = r (8 rows of one D-tile).
            uint32_t ts_src[3] = { 8,           tile_b,             N_T * tile_b        };
            uint32_t tb_src[3] = { inner_beats, N_T,                M_T                 };
            uint32_t ts_dst[3] = { 8,           row_b_src,          meshRow * row_b_dst };
            uint32_t tb_dst[3] = { inner_beats, N_T,                M_T                 };
            xdma_layout_run(st.xdma_src, dst_addr, row_b_src, row_b_dst,
                            3, ts_src, tb_src, ts_dst, tb_dst, false);
        } else if (path == 2) {
            // Path 2: spatial = r_inner; r_outer is a new temporal dim.
            uint32_t r_outer = meshRow / 8;
            uint32_t ts_src[4] = { 8,           8 * row_b_src,      tile_b,             N_T * tile_b };
            uint32_t tb_src[4] = { inner_beats, r_outer,            N_T,                M_T          };
            uint32_t ts_dst[4] = { 8,           8 * row_b_dst,      row_b_src,          meshRow * row_b_dst };
            uint32_t tb_dst[4] = { inner_beats, r_outer,            N_T,                M_T          };
            xdma_layout_run(st.xdma_src, dst_addr, row_b_src, row_b_dst,
                            4, ts_src, tb_src, ts_dst, tb_dst, false);
        } else if (path == 3) {
            // Path 3: spatial = c_chunk_inner (8 chunks x 8 bytes covering a
            // 64-byte slice of one D-tile row in one xDMA spatial sweep).
            uint32_t c_outer = row_b_src / 64;
            uint32_t ts_src[4] = { 64,      row_b_src,  tile_b,    N_T * tile_b };
            uint32_t tb_src[4] = { c_outer, meshRow,    N_T,       M_T          };
            uint32_t ts_dst[4] = { 64,      row_b_dst,  row_b_src, meshRow * row_b_dst };
            uint32_t tb_dst[4] = { c_outer, meshRow,    N_T,       M_T          };
            xdma_layout_run(st.xdma_src, dst_addr, 8, 8,
                            4, ts_src, tb_src, ts_dst, tb_dst, false);
        } else if (path == 4) {
            // Path 4: spatial = n_inner (8 D-tiles in n direction in parallel).
            uint32_t n_outer = N_T / 8;
            uint32_t ts_src[4] = { 8,           row_b_src,  8 * tile_b,    N_T * tile_b };
            uint32_t tb_src[4] = { inner_beats, meshRow,    n_outer,       M_T          };
            uint32_t ts_dst[4] = { 8,           row_b_dst,  8 * row_b_src, meshRow * row_b_dst };
            uint32_t tb_dst[4] = { inner_beats, meshRow,    n_outer,       M_T          };
            xdma_layout_run(st.xdma_src, dst_addr, tile_b, row_b_src,
                            4, ts_src, tb_src, ts_dst, tb_dst, false);
        } else /* path == 5 */ {
            // Path 5: spatial = m_inner (8 different m row-blocks in parallel).
            uint32_t m_outer = M_T / 8;
            uint32_t ts_src[4] = { 8,           row_b_src,  tile_b,     8 * N_T * tile_b };
            uint32_t tb_src[4] = { inner_beats, meshRow,    N_T,        m_outer          };
            uint32_t ts_dst[4] = { 8,           row_b_dst,  row_b_src,  8 * meshRow * row_b_dst };
            uint32_t tb_dst[4] = { inner_beats, meshRow,    N_T,        m_outer          };
            xdma_layout_run(st.xdma_src, dst_addr,
                            N_T * tile_b, meshRow * row_b_dst,
                            4, ts_src, tb_src, ts_dst, tb_dst, false);
        }
        xdma_layout_stage_free(&st);
        hw_done = true;
    }

    if (!hw_done) {
        // CPU fallback (no HW path matched the shape). Stage non-local (L3)
        // operands through L1: the DM core is not coherent with L3.
        uint32_t N_cols = N_T * meshCol;
        xdma_layout_stage_t si;
        xdma_layout_stage_out_t so;
        if (xdma_layout_stage_in(&si, src_addr, bytes) != 0 ||
            xdma_layout_stage_out(&so, dst_addr, bytes) != 0) {
            xdma_layout_stage_free(&si);   // frees src scratch if staged; no-op otherwise
            printf_safe("[Cluster %d Core %d]: d_to_row_major CPU-fallback L1 alloc failed!\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }
        volatile uint8_t *src = (volatile uint8_t *)(uint32_t)si.xdma_src;
        volatile uint8_t *dst = (volatile uint8_t *)so.l1;
        for (uint32_t m = 0; m < M_T; m++)
        for (uint32_t n = 0; n < N_T; n++)
        for (uint32_t r = 0; r < meshRow; r++)
        for (uint32_t c = 0; c < meshCol; c++) {
            uint32_t src_off = (((m * N_T + n) * meshRow + r) * meshCol + c) * elem_bytes;
            uint32_t dst_off = ((m * meshRow + r) * N_cols + n * meshCol + c) * elem_bytes;
            for (uint32_t b = 0; b < elem_bytes; b++) dst[dst_off + b] = src[src_off + b];
        }
        xdma_layout_stage_out_flush(&si, &so, dst_addr, bytes);   // transfer + flush + free both
    }
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// The 9 runnable xdma_d_to_row_major kernels = (array shape) x (elem_bytes). Each binds its mesh and
// element width as compile-time constants, so the AGU-path decision tree above folds away and
// the wrapper IS its path -- there is no runtime `if` left to pick the wrong one.
#define BINGO_DEF_XDMA_D_TO_ROW_MAJOR(suffix, d1, d2, eb)                            \
    SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_d_to_row_major_##suffix(void *arg)     \
    { return __xdma_d_to_row_major_impl(arg, (d1), (d2), (eb)); }

BINGO_DEF_XDMA_D_TO_ROW_MAJOR(e1_M32N32, 32, 32, 1)
BINGO_DEF_XDMA_D_TO_ROW_MAJOR(e2_M32N32, 32, 32, 2)
BINGO_DEF_XDMA_D_TO_ROW_MAJOR(e4_M32N32, 32, 32, 4)
BINGO_DEF_XDMA_D_TO_ROW_MAJOR(e1_M1N32, 1, 32, 1)
BINGO_DEF_XDMA_D_TO_ROW_MAJOR(e2_M1N32, 1, 32, 2)
BINGO_DEF_XDMA_D_TO_ROW_MAJOR(e4_M1N32, 1, 32, 4)
BINGO_DEF_XDMA_D_TO_ROW_MAJOR(e1_M16N16, 16, 16, 1)
BINGO_DEF_XDMA_D_TO_ROW_MAJOR(e2_M16N16, 16, 16, 2)
BINGO_DEF_XDMA_D_TO_ROW_MAJOR(e4_M16N16, 16, 16, 4)

// row-major → A-layout. See section banner for the path table.
//   Coverage: paths 1–5; array_shape 0 INT8 (tileSize·elem_bytes=2) lacks any
//   8-byte beat that's contiguous in both src (row-major rows) and dst
//   (packed A) → CPU. Other shapes hit a HW path.
//   Arg layout: src_hi/lo, dst_hi/lo, M_T, K_T, meshRow, tileSize, elem_bytes.
static inline uint32_t __xdma_row_major_to_a_impl(void *arg, uint32_t meshRow, uint32_t tileSize, uint32_t elem_bytes)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_row_major_to_a_args_t);
    if (!snrt_is_dm_core()) {
        printf_safe("[Cluster %d Core %d]: Error! row_major_to_a must be called from DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t *a = (uint32_t *)arg;
    uint64_t src_addr = make_u64(a[0], a[1]);
    uint64_t dst_addr = make_u64(a[2], a[3]);
    uint32_t M_T = a[4], K_T = a[5];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_row_major_to_a_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    uint32_t bytes     = M_T * K_T * meshRow * tileSize * elem_bytes;
    uint32_t row_b_blk = tileSize * elem_bytes;          // 1 row inside one A-tile (dst-packed)
    uint32_t row_b_rm  = K_T * tileSize * elem_bytes;    // 1 row of row-major R (src-side)
    uint32_t tile_b    = meshRow * tileSize * elem_bytes;
    bool hw_done = false;

    int path = 0;
    if ((row_b_blk % 8) == 0) {
        if (meshRow == 8)                                                       path = 1;
        else if (meshRow > 8 && (meshRow % 8) == 0)                             path = 2;
        else if ((meshRow == 1 || meshRow == 2 || meshRow == 4)
                 && (row_b_blk % 64) == 0)                                      path = 3;
        else if ((K_T % 8) == 0)                                                path = 4;
        else if ((M_T % 8) == 0)                                                path = 5;
    }

    if (path != 0) {
        xdma_layout_stage_t st;
        if (xdma_layout_stage_in(&st, src_addr, bytes) != 0) {
            printf_safe("[Cluster %d Core %d]: row_major_to_a L1 alloc failed!\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }
        uint32_t inner_beats = row_b_blk / 8;

        if (path == 1) {
            // Path 1: spatial = r (8 channels = 8 rows of an A-tile).
            uint32_t ts_src[3] = { 8,           row_b_blk,  meshRow * row_b_rm };
            uint32_t tb_src[3] = { inner_beats, K_T,        M_T                };
            uint32_t ts_dst[3] = { 8,           tile_b,     K_T * tile_b       };
            uint32_t tb_dst[3] = { inner_beats, K_T,        M_T                };
            xdma_layout_run(st.xdma_src, dst_addr, row_b_rm, row_b_blk,
                            3, ts_src, tb_src, ts_dst, tb_dst, false);
        } else if (path == 2) {
            uint32_t r_outer = meshRow / 8;
            uint32_t ts_src[4] = { 8,           8 * row_b_rm,  row_b_blk,  meshRow * row_b_rm };
            uint32_t tb_src[4] = { inner_beats, r_outer,       K_T,        M_T                };
            uint32_t ts_dst[4] = { 8,           8 * row_b_blk, tile_b,     K_T * tile_b       };
            uint32_t tb_dst[4] = { inner_beats, r_outer,       K_T,        M_T                };
            xdma_layout_run(st.xdma_src, dst_addr, row_b_rm, row_b_blk,
                            4, ts_src, tb_src, ts_dst, tb_dst, false);
        } else if (path == 3) {
            uint32_t s_outer = row_b_blk / 64;
            uint32_t ts_src[4] = { 64,      row_b_rm,  row_b_blk,  meshRow * row_b_rm };
            uint32_t tb_src[4] = { s_outer, meshRow,   K_T,        M_T                };
            uint32_t ts_dst[4] = { 64,      row_b_blk, tile_b,     K_T * tile_b       };
            uint32_t tb_dst[4] = { s_outer, meshRow,   K_T,        M_T                };
            xdma_layout_run(st.xdma_src, dst_addr, 8, 8,
                            4, ts_src, tb_src, ts_dst, tb_dst, false);
        } else if (path == 4) {
            // Path 4: spatial = k_inner (8 different k-tiles in parallel,
            // each at the same r within its tile).
            uint32_t k_outer = K_T / 8;
            uint32_t ts_src[4] = { 8,           row_b_rm,  8 * row_b_blk, meshRow * row_b_rm };
            uint32_t tb_src[4] = { inner_beats, meshRow,   k_outer,       M_T                };
            uint32_t ts_dst[4] = { 8,           row_b_blk, 8 * tile_b,    K_T * tile_b       };
            uint32_t tb_dst[4] = { inner_beats, meshRow,   k_outer,       M_T                };
            xdma_layout_run(st.xdma_src, dst_addr, row_b_blk, tile_b,
                            4, ts_src, tb_src, ts_dst, tb_dst, false);
        } else /* path == 5 */ {
            uint32_t m_outer = M_T / 8;
            uint32_t ts_src[4] = { 8,           row_b_rm,  row_b_blk,  8 * meshRow * row_b_rm };
            uint32_t tb_src[4] = { inner_beats, meshRow,   K_T,        m_outer                };
            uint32_t ts_dst[4] = { 8,           row_b_blk, tile_b,     8 * K_T * tile_b       };
            uint32_t tb_dst[4] = { inner_beats, meshRow,   K_T,        m_outer                };
            xdma_layout_run(st.xdma_src, dst_addr,
                            meshRow * row_b_rm, K_T * tile_b,
                            4, ts_src, tb_src, ts_dst, tb_dst, false);
        }
        xdma_layout_stage_free(&st);
        hw_done = true;
    }

    if (!hw_done) {
        uint32_t K_cols = K_T * tileSize;
        // CPU fallback: stage non-local (L3) operands through L1.
        xdma_layout_stage_t si;
        xdma_layout_stage_out_t so;
        if (xdma_layout_stage_in(&si, src_addr, bytes) != 0 ||
            xdma_layout_stage_out(&so, dst_addr, bytes) != 0) {
            xdma_layout_stage_free(&si);   // frees src scratch if staged; no-op otherwise
            printf_safe("[Cluster %d Core %d]: row_major_to_a CPU-fallback L1 alloc failed!\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }
        volatile uint8_t *src = (volatile uint8_t *)(uint32_t)si.xdma_src;
        volatile uint8_t *dst = (volatile uint8_t *)so.l1;
        for (uint32_t m = 0; m < M_T; m++)
        for (uint32_t k = 0; k < K_T; k++)
        for (uint32_t r = 0; r < meshRow; r++)
        for (uint32_t s = 0; s < tileSize; s++) {
            uint32_t src_off = ((m * meshRow + r) * K_cols + k * tileSize + s) * elem_bytes;
            uint32_t dst_off = (((m * K_T + k) * meshRow + r) * tileSize + s) * elem_bytes;
            for (uint32_t b = 0; b < elem_bytes; b++) dst[dst_off + b] = src[src_off + b];
        }
        xdma_layout_stage_out_flush(&si, &so, dst_addr, bytes);   // transfer + flush + free both
    }
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// The 9 runnable xdma_row_major_to_a kernels = (array shape) x (elem_bytes). Each binds its mesh and
// element width as compile-time constants, so the AGU-path decision tree above folds away and
// the wrapper IS its path -- there is no runtime `if` left to pick the wrong one.
#define BINGO_DEF_XDMA_ROW_MAJOR_TO_A(suffix, d1, d2, eb)                            \
    SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_row_major_to_a_##suffix(void *arg)     \
    { return __xdma_row_major_to_a_impl(arg, (d1), (d2), (eb)); }

BINGO_DEF_XDMA_ROW_MAJOR_TO_A(e1_M32K2, 32, 2, 1)
BINGO_DEF_XDMA_ROW_MAJOR_TO_A(e2_M32K2, 32, 2, 2)
BINGO_DEF_XDMA_ROW_MAJOR_TO_A(e4_M32K2, 32, 2, 4)
BINGO_DEF_XDMA_ROW_MAJOR_TO_A(e1_M1K16, 1, 16, 1)
BINGO_DEF_XDMA_ROW_MAJOR_TO_A(e2_M1K16, 1, 16, 2)
BINGO_DEF_XDMA_ROW_MAJOR_TO_A(e4_M1K16, 1, 16, 4)
BINGO_DEF_XDMA_ROW_MAJOR_TO_A(e1_M16K8, 16, 8, 1)
BINGO_DEF_XDMA_ROW_MAJOR_TO_A(e2_M16K8, 16, 8, 2)
BINGO_DEF_XDMA_ROW_MAJOR_TO_A(e4_M16K8, 16, 8, 4)

// row-major → B-layout. B↔R is per-(n,k)-tile transpose-then-tile, so we
// drive the xDMA Transposer writer extension (mirrors xdma_transpose_2d).
// The Transposer is fixed at 8x8-byte blocks, so the (n,k) B-tile is
// decomposed into (meshCol/8) c-sub x (tileSize/8) s-sub element sub-blocks
// and the AGU iterates them in addition to (n, k).
//   Coverage: HW path when defined(WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16) && tileSize%8==0 && meshCol%8==0
//             (all VersaCore array_shapes 1..4); CPU when tileSize=2 (shape 0).
//   The CPU path stages non-local (L3) operands through L1 (xdma_cpu_stage_*).
//   Arg layout: src_hi/lo, dst_hi/lo, K_T, N_T, tileSize, meshCol, elem_bytes.
static inline uint32_t __xdma_row_major_to_b_impl(void *arg, uint32_t tileSize, uint32_t meshCol, uint32_t elem_bytes)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_row_major_to_b_args_t);
    if (!snrt_is_dm_core()) {
        printf_safe("[Cluster %d Core %d]: Error! row_major_to_b must be called from DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t *a = (uint32_t *)arg;
    uint64_t src_addr = make_u64(a[0], a[1]);
    uint64_t dst_addr = make_u64(a[2], a[3]);
    uint32_t K_T = a[4], N_T = a[5];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_row_major_to_b_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    uint32_t bytes  = K_T * tileSize * N_T * meshCol * elem_bytes;
    bool hw_done = false;

#ifdef WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16
    if ((tileSize % 8) == 0 && (meshCol % 8) == 0) {
        xdma_layout_stage_t st;
        if (xdma_layout_stage_in(&st, src_addr, bytes) != 0) {
            printf_safe("[Cluster %d Core %d]: row_major_to_b L1 alloc failed!\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }

        const uint32_t tile_w   = 8;                                    // HW transposer block
        uint32_t tpt            = (tile_w * tile_w * (elem_bytes * 8) + 511) / 512;
        uint32_t row_b_src      = N_T * meshCol * elem_bytes;                 // R row width
        uint32_t b_tile_b       = tileSize * meshCol * elem_bytes;            // one B-tile bytes
        uint32_t c_subs         = meshCol  / tile_w;                    // # 8-elem_bytes c chunks
        uint32_t s_subs         = tileSize / tile_w;                    // # 8-elem_bytes s chunks

        // 8 channels = 8 byte-rows of one transposer block (= 8 c values).
        // Each channel writes to a c_inner position in B; spatial_stride_dst
        // crosses one c step inside the B-tile = tileSize·elem_bytes bytes.
        uint32_t spatial_stride_src = row_b_src;
        uint32_t spatial_stride_dst = tileSize * elem_bytes;

        // 5 temporal dims max:
        //   [0] tpt          : within transposer block (1 for elem_bytes=1, 4 for elem_bytes=4)
        //   [1] c_sub        : next 8 c-elements inside the B-tile
        //   [2] s_sub        : next 8 s-elements inside the B-tile
        //   [3] k            : next R row-tile / next (n,k) tile in B
        //   [4] n            : next R col-tile / next n-group in B
        uint32_t ts_src[5] = {
            8,                          // tpt: advance 8 bytes within channel
            tile_w * elem_bytes,              // c_sub: +8 cols in R
            tile_w * row_b_src,         // s_sub: +8 rows in R
            tileSize * row_b_src,       // k: +tileSize rows in R
            meshCol * elem_bytes              // n: +meshCol cols in R
        };
        uint32_t tb_src[5] = { tpt, c_subs, s_subs, K_T, N_T };
        uint32_t ts_dst[5] = {
            8,                          // tpt: 8 bytes within channel
            tile_w * tileSize * elem_bytes,   // c_sub: +8 c-rows in B-tile (each c-row = tileSize·elem_bytes)
            tile_w * elem_bytes,              // s_sub: +8 s-bytes in each c-row of B-tile
            b_tile_b,                   // k: next (n,k) tile within n-group
            K_T * b_tile_b              // n: next n-group of K_T tiles in B
        };
        uint32_t tb_dst[5] = { tpt, c_subs, s_subs, K_T, N_T };

        xdma_layout_run(st.xdma_src, dst_addr,
                        spatial_stride_src, spatial_stride_dst,
                        5, ts_src, tb_src, ts_dst, tb_dst, true);

        xdma_layout_stage_free(&st);
        hw_done = true;
    }
#endif

    if (!hw_done) {
        uint32_t N_cols = N_T * meshCol;
        // CPU fallback: stage non-local (L3) operands through L1 — the DM core
        // is not coherent with L3, so a direct CPU deref there returns garbage.
        xdma_layout_stage_t si;
        xdma_layout_stage_out_t so;
        if (xdma_layout_stage_in(&si, src_addr, bytes) != 0 ||
            xdma_layout_stage_out(&so, dst_addr, bytes) != 0) {
            xdma_layout_stage_free(&si);   // frees src scratch if staged; no-op otherwise
            printf_safe("[Cluster %d Core %d]: row_major_to_b CPU-fallback L1 alloc failed!\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }
        volatile uint8_t *src = (volatile uint8_t *)(uint32_t)si.xdma_src;
        volatile uint8_t *dst = (volatile uint8_t *)so.l1;
        for (uint32_t n = 0; n < N_T; n++)
        for (uint32_t k = 0; k < K_T; k++)
        for (uint32_t c = 0; c < meshCol; c++)
        for (uint32_t s = 0; s < tileSize; s++) {
            uint32_t src_off = ((k * tileSize + s) * N_cols + n * meshCol + c) * elem_bytes;
            uint32_t dst_off = (((n * K_T + k) * meshCol + c) * tileSize + s) * elem_bytes;
            for (uint32_t b = 0; b < elem_bytes; b++) dst[dst_off + b] = src[src_off + b];
        }
        xdma_layout_stage_out_flush(&si, &so, dst_addr, bytes);   // transfer + flush + free both
    }
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// The 9 runnable xdma_row_major_to_b kernels = (array shape) x (elem_bytes). Each binds its mesh and
// element width as compile-time constants, so the AGU-path decision tree above folds away and
// the wrapper IS its path -- there is no runtime `if` left to pick the wrong one.
#define BINGO_DEF_XDMA_ROW_MAJOR_TO_B(suffix, d1, d2, eb)                            \
    SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_row_major_to_b_##suffix(void *arg)     \
    { return __xdma_row_major_to_b_impl(arg, (d1), (d2), (eb)); }

BINGO_DEF_XDMA_ROW_MAJOR_TO_B(e1_K2N32, 2, 32, 1)
BINGO_DEF_XDMA_ROW_MAJOR_TO_B(e2_K2N32, 2, 32, 2)
BINGO_DEF_XDMA_ROW_MAJOR_TO_B(e4_K2N32, 2, 32, 4)
BINGO_DEF_XDMA_ROW_MAJOR_TO_B(e1_K16N32, 16, 32, 1)
BINGO_DEF_XDMA_ROW_MAJOR_TO_B(e2_K16N32, 16, 32, 2)
BINGO_DEF_XDMA_ROW_MAJOR_TO_B(e4_K16N32, 16, 32, 4)
BINGO_DEF_XDMA_ROW_MAJOR_TO_B(e1_K8N16, 8, 16, 1)
BINGO_DEF_XDMA_ROW_MAJOR_TO_B(e2_K8N16, 8, 16, 2)
BINGO_DEF_XDMA_ROW_MAJOR_TO_B(e4_K8N16, 8, 16, 4)

// A-layout → row-major. Inverse of row_major_to_a: src/dst stride arrays
// are swapped versus the forward kernel; same path-selection logic.
//   Coverage: paths 1–5; same CPU-only cell as forward (array_shape 0 INT8).
//   Arg layout: src_hi/lo, dst_hi/lo, M_T, K_T, meshRow, tileSize, elem_bytes.
static inline uint32_t __xdma_a_to_row_major_impl(void *arg, uint32_t meshRow, uint32_t tileSize, uint32_t elem_bytes)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_a_to_row_major_args_t);
    if (!snrt_is_dm_core()) {
        printf_safe("[Cluster %d Core %d]: Error! a_to_row_major must be called from DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t *a = (uint32_t *)arg;
    uint64_t src_addr = make_u64(a[0], a[1]);
    uint64_t dst_addr = make_u64(a[2], a[3]);
    uint32_t M_T = a[4], K_T = a[5];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_a_to_row_major_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    uint32_t bytes     = M_T * K_T * meshRow * tileSize * elem_bytes;
    uint32_t row_b_blk = tileSize * elem_bytes;          // 1 row inside one A-tile (src-packed)
    uint32_t row_b_rm  = K_T * tileSize * elem_bytes;    // 1 row of row-major R (dst-side)
    uint32_t tile_b    = meshRow * tileSize * elem_bytes;
    bool hw_done = false;

    int path = 0;
    if ((row_b_blk % 8) == 0) {
        if (meshRow == 8)                                                       path = 1;
        else if (meshRow > 8 && (meshRow % 8) == 0)                             path = 2;
        else if ((meshRow == 1 || meshRow == 2 || meshRow == 4)
                 && (row_b_blk % 64) == 0)                                      path = 3;
        else if ((K_T % 8) == 0)                                                path = 4;
        else if ((M_T % 8) == 0)                                                path = 5;
    }

    if (path != 0) {
        xdma_layout_stage_t st;
        if (xdma_layout_stage_in(&st, src_addr, bytes) != 0) {
            printf_safe("[Cluster %d Core %d]: a_to_row_major L1 alloc failed!\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }
        uint32_t inner_beats = row_b_blk / 8;

        if (path == 1) {
            uint32_t ts_src[3] = { 8,           tile_b,    K_T * tile_b       };
            uint32_t tb_src[3] = { inner_beats, K_T,       M_T                };
            uint32_t ts_dst[3] = { 8,           row_b_blk, meshRow * row_b_rm };
            uint32_t tb_dst[3] = { inner_beats, K_T,       M_T                };
            xdma_layout_run(st.xdma_src, dst_addr, row_b_blk, row_b_rm,
                            3, ts_src, tb_src, ts_dst, tb_dst, false);
        } else if (path == 2) {
            uint32_t r_outer = meshRow / 8;
            uint32_t ts_src[4] = { 8,           8 * row_b_blk, tile_b,    K_T * tile_b       };
            uint32_t tb_src[4] = { inner_beats, r_outer,       K_T,       M_T                };
            uint32_t ts_dst[4] = { 8,           8 * row_b_rm,  row_b_blk, meshRow * row_b_rm };
            uint32_t tb_dst[4] = { inner_beats, r_outer,       K_T,       M_T                };
            xdma_layout_run(st.xdma_src, dst_addr, row_b_blk, row_b_rm,
                            4, ts_src, tb_src, ts_dst, tb_dst, false);
        } else if (path == 3) {
            uint32_t s_outer = row_b_blk / 64;
            uint32_t ts_src[4] = { 64,      row_b_blk, tile_b,    K_T * tile_b       };
            uint32_t tb_src[4] = { s_outer, meshRow,   K_T,       M_T                };
            uint32_t ts_dst[4] = { 64,      row_b_rm,  row_b_blk, meshRow * row_b_rm };
            uint32_t tb_dst[4] = { s_outer, meshRow,   K_T,       M_T                };
            xdma_layout_run(st.xdma_src, dst_addr, 8, 8,
                            4, ts_src, tb_src, ts_dst, tb_dst, false);
        } else if (path == 4) {
            uint32_t k_outer = K_T / 8;
            uint32_t ts_src[4] = { 8,           row_b_blk, 8 * tile_b,    K_T * tile_b       };
            uint32_t tb_src[4] = { inner_beats, meshRow,   k_outer,       M_T                };
            uint32_t ts_dst[4] = { 8,           row_b_rm,  8 * row_b_blk, meshRow * row_b_rm };
            uint32_t tb_dst[4] = { inner_beats, meshRow,   k_outer,       M_T                };
            xdma_layout_run(st.xdma_src, dst_addr, tile_b, row_b_blk,
                            4, ts_src, tb_src, ts_dst, tb_dst, false);
        } else /* path == 5 */ {
            uint32_t m_outer = M_T / 8;
            uint32_t ts_src[4] = { 8,           row_b_blk, tile_b,    8 * K_T * tile_b       };
            uint32_t tb_src[4] = { inner_beats, meshRow,   K_T,       m_outer                };
            uint32_t ts_dst[4] = { 8,           row_b_rm,  row_b_blk, 8 * meshRow * row_b_rm };
            uint32_t tb_dst[4] = { inner_beats, meshRow,   K_T,       m_outer                };
            xdma_layout_run(st.xdma_src, dst_addr,
                            K_T * tile_b, meshRow * row_b_rm,
                            4, ts_src, tb_src, ts_dst, tb_dst, false);
        }
        xdma_layout_stage_free(&st);
        hw_done = true;
    }

    if (!hw_done) {
        uint32_t K_cols = K_T * tileSize;
        // CPU fallback: stage non-local (L3) operands through L1.
        xdma_layout_stage_t si;
        xdma_layout_stage_out_t so;
        if (xdma_layout_stage_in(&si, src_addr, bytes) != 0 ||
            xdma_layout_stage_out(&so, dst_addr, bytes) != 0) {
            xdma_layout_stage_free(&si);   // frees src scratch if staged; no-op otherwise
            printf_safe("[Cluster %d Core %d]: a_to_row_major CPU-fallback L1 alloc failed!\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }
        volatile uint8_t *src = (volatile uint8_t *)(uint32_t)si.xdma_src;
        volatile uint8_t *dst = (volatile uint8_t *)so.l1;
        for (uint32_t m = 0; m < M_T; m++)
        for (uint32_t k = 0; k < K_T; k++)
        for (uint32_t r = 0; r < meshRow; r++)
        for (uint32_t s = 0; s < tileSize; s++) {
            uint32_t src_off = (((m * K_T + k) * meshRow + r) * tileSize + s) * elem_bytes;
            uint32_t dst_off = ((m * meshRow + r) * K_cols + k * tileSize + s) * elem_bytes;
            for (uint32_t b = 0; b < elem_bytes; b++) dst[dst_off + b] = src[src_off + b];
        }
        xdma_layout_stage_out_flush(&si, &so, dst_addr, bytes);   // transfer + flush + free both
    }
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// The 9 runnable xdma_a_to_row_major kernels = (array shape) x (elem_bytes). Each binds its mesh and
// element width as compile-time constants, so the AGU-path decision tree above folds away and
// the wrapper IS its path -- there is no runtime `if` left to pick the wrong one.
#define BINGO_DEF_XDMA_A_TO_ROW_MAJOR(suffix, d1, d2, eb)                            \
    SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_a_to_row_major_##suffix(void *arg)     \
    { return __xdma_a_to_row_major_impl(arg, (d1), (d2), (eb)); }

BINGO_DEF_XDMA_A_TO_ROW_MAJOR(e1_M32K2, 32, 2, 1)
BINGO_DEF_XDMA_A_TO_ROW_MAJOR(e2_M32K2, 32, 2, 2)
BINGO_DEF_XDMA_A_TO_ROW_MAJOR(e4_M32K2, 32, 2, 4)
BINGO_DEF_XDMA_A_TO_ROW_MAJOR(e1_M1K16, 1, 16, 1)
BINGO_DEF_XDMA_A_TO_ROW_MAJOR(e2_M1K16, 1, 16, 2)
BINGO_DEF_XDMA_A_TO_ROW_MAJOR(e4_M1K16, 1, 16, 4)
BINGO_DEF_XDMA_A_TO_ROW_MAJOR(e1_M16K8, 16, 8, 1)
BINGO_DEF_XDMA_A_TO_ROW_MAJOR(e2_M16K8, 16, 8, 2)
BINGO_DEF_XDMA_A_TO_ROW_MAJOR(e4_M16K8, 16, 8, 4)

// B-layout → row-major. Inverse of row_major_to_b: same per-(n,k)-tile
// transpose, src/dst stride arrays swapped, AGU iterates the same
// (c_sub, s_sub, k, n) sub-block grid.
//   Coverage: same as forward — HW for shapes 1..4, CPU for shape 0.
//   Arg layout: src_hi/lo, dst_hi/lo, K_T, N_T, tileSize, meshCol, elem_bytes.
static inline uint32_t __xdma_b_to_row_major_impl(void *arg, uint32_t tileSize, uint32_t meshCol, uint32_t elem_bytes)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_b_to_row_major_args_t);
    if (!snrt_is_dm_core()) {
        printf_safe("[Cluster %d Core %d]: Error! b_to_row_major must be called from DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t *a = (uint32_t *)arg;
    uint64_t src_addr = make_u64(a[0], a[1]);
    uint64_t dst_addr = make_u64(a[2], a[3]);
    uint32_t K_T = a[4], N_T = a[5];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_b_to_row_major_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    uint32_t bytes  = K_T * tileSize * N_T * meshCol * elem_bytes;
    bool hw_done = false;

#ifdef WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16
    if ((tileSize % 8) == 0 && (meshCol % 8) == 0) {
        xdma_layout_stage_t st;
        if (xdma_layout_stage_in(&st, src_addr, bytes) != 0) {
            printf_safe("[Cluster %d Core %d]: b_to_row_major L1 alloc failed!\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }

        const uint32_t tile_w   = 8;
        uint32_t tpt            = (tile_w * tile_w * (elem_bytes * 8) + 511) / 512;
        uint32_t row_b_dst      = N_T * meshCol * elem_bytes;
        uint32_t b_tile_b       = tileSize * meshCol * elem_bytes;
        uint32_t c_subs         = meshCol  / tile_w;
        uint32_t s_subs         = tileSize / tile_w;

        // Reader: 8 channels each read one byte-row of the src 8x8 block;
        // within a B-tile, byte-rows (c values) are tileSize·elem_bytes apart.
        uint32_t spatial_stride_src = tileSize * elem_bytes;
        // Writer: after transpose, channel c_inner writes 8 bytes representing
        // one s position; consecutive channels write to consecutive R rows.
        uint32_t spatial_stride_dst = row_b_dst;

        // 5 temporal dims:
        //   [0] tpt   : within transposer block
        //   [1] c_sub : next 8 c-elements of the B-tile
        //   [2] s_sub : next 8 s-elements of the B-tile
        //   [3] k     : next (n,k+1) tile in B / next R row-tile
        //   [4] n     : next n-group in B / next R col-tile
        uint32_t ts_src[5] = {
            8,                          // tpt
            tile_w * tileSize * elem_bytes,   // c_sub: +8 c-rows in B-tile
            tile_w * elem_bytes,              // s_sub: +8 s-bytes in each c-row of B-tile
            b_tile_b,                   // k: next (n,k) tile within n-group
            K_T * b_tile_b              // n: next n-group of K_T tiles
        };
        uint32_t tb_src[5] = { tpt, c_subs, s_subs, K_T, N_T };
        uint32_t ts_dst[5] = {
            8,                          // tpt
            tile_w * elem_bytes,              // c_sub: +8 cols in R (same row band)
            tile_w * row_b_dst,         // s_sub: +8 rows down in R
            tileSize * row_b_dst,       // k: +tileSize rows in R
            meshCol * elem_bytes              // n: +meshCol cols in R
        };
        uint32_t tb_dst[5] = { tpt, c_subs, s_subs, K_T, N_T };

        xdma_layout_run(st.xdma_src, dst_addr,
                        spatial_stride_src, spatial_stride_dst,
                        5, ts_src, tb_src, ts_dst, tb_dst, true);

        xdma_layout_stage_free(&st);
        hw_done = true;
    }
#endif

    if (!hw_done) {
        uint32_t N_cols = N_T * meshCol;
        // CPU fallback: stage non-local (L3) operands through L1.
        xdma_layout_stage_t si;
        xdma_layout_stage_out_t so;
        if (xdma_layout_stage_in(&si, src_addr, bytes) != 0 ||
            xdma_layout_stage_out(&so, dst_addr, bytes) != 0) {
            xdma_layout_stage_free(&si);   // frees src scratch if staged; no-op otherwise
            printf_safe("[Cluster %d Core %d]: b_to_row_major CPU-fallback L1 alloc failed!\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }
        volatile uint8_t *src = (volatile uint8_t *)(uint32_t)si.xdma_src;
        volatile uint8_t *dst = (volatile uint8_t *)so.l1;
        for (uint32_t n = 0; n < N_T; n++)
        for (uint32_t k = 0; k < K_T; k++)
        for (uint32_t c = 0; c < meshCol; c++)
        for (uint32_t s = 0; s < tileSize; s++) {
            uint32_t src_off = (((n * K_T + k) * meshCol + c) * tileSize + s) * elem_bytes;
            uint32_t dst_off = ((k * tileSize + s) * N_cols + n * meshCol + c) * elem_bytes;
            for (uint32_t b = 0; b < elem_bytes; b++) dst[dst_off + b] = src[src_off + b];
        }
        xdma_layout_stage_out_flush(&si, &so, dst_addr, bytes);   // transfer + flush + free both
    }
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// The 9 runnable xdma_b_to_row_major kernels = (array shape) x (elem_bytes). Each binds its mesh and
// element width as compile-time constants, so the AGU-path decision tree above folds away and
// the wrapper IS its path -- there is no runtime `if` left to pick the wrong one.
#define BINGO_DEF_XDMA_B_TO_ROW_MAJOR(suffix, d1, d2, eb)                            \
    SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_b_to_row_major_##suffix(void *arg)     \
    { return __xdma_b_to_row_major_impl(arg, (d1), (d2), (eb)); }

BINGO_DEF_XDMA_B_TO_ROW_MAJOR(e1_K2N32, 2, 32, 1)
BINGO_DEF_XDMA_B_TO_ROW_MAJOR(e2_K2N32, 2, 32, 2)
BINGO_DEF_XDMA_B_TO_ROW_MAJOR(e4_K2N32, 2, 32, 4)
BINGO_DEF_XDMA_B_TO_ROW_MAJOR(e1_K16N32, 16, 32, 1)
BINGO_DEF_XDMA_B_TO_ROW_MAJOR(e2_K16N32, 16, 32, 2)
BINGO_DEF_XDMA_B_TO_ROW_MAJOR(e4_K16N32, 16, 32, 4)
BINGO_DEF_XDMA_B_TO_ROW_MAJOR(e1_K8N16, 8, 16, 1)
BINGO_DEF_XDMA_B_TO_ROW_MAJOR(e2_K8N16, 8, 16, 2)
BINGO_DEF_XDMA_B_TO_ROW_MAJOR(e4_K8N16, 8, 16, 4)

// row-major → D-layout. Inverse of d_to_row_major: src/dst stride arrays
// are swapped versus the forward kernel; same path-selection logic.
//   Coverage: paths 1–5 cover all 5 VersaCore array_shapes via paths 1–3.
//   Arg layout: src_hi/lo, dst_hi/lo, M_T, N_T, meshRow, meshCol, elem_bytes.
static inline uint32_t __xdma_row_major_to_d_impl(void *arg, uint32_t meshRow, uint32_t meshCol, uint32_t elem_bytes)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_row_major_to_d_args_t);
    if (!snrt_is_dm_core()) {
        printf_safe("[Cluster %d Core %d]: Error! row_major_to_d must be called from DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t *a = (uint32_t *)arg;
    uint64_t src_addr = make_u64(a[0], a[1]);
    uint64_t dst_addr = make_u64(a[2], a[3]);
    uint32_t M_T = a[4], N_T = a[5];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_row_major_to_d_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    uint32_t bytes     = M_T * N_T * meshRow * meshCol * elem_bytes;
    uint32_t row_b_blk = meshCol * elem_bytes;          // 1 row of a D-tile (dst-side packing)
    uint32_t row_b_rm  = N_T * meshCol * elem_bytes;    // 1 row of row-major R (src-side)
    uint32_t tile_b    = meshRow * meshCol * elem_bytes;
    bool hw_done = false;

    int path = 0;
    if ((row_b_blk % 8) == 0) {
        if (meshRow == 8)                                                       path = 1;
        else if (meshRow > 8 && (meshRow % 8) == 0)                             path = 2;
        else if ((meshRow == 1 || meshRow == 2 || meshRow == 4)
                 && (row_b_blk % 64) == 0)                                      path = 3;
        else if ((N_T % 8) == 0)                                                path = 4;
        else if ((M_T % 8) == 0)                                                path = 5;
    }

    if (path != 0) {
        xdma_layout_stage_t st;
        if (xdma_layout_stage_in(&st, src_addr, bytes) != 0) {
            printf_safe("[Cluster %d Core %d]: row_major_to_d L1 alloc failed!\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }
        uint32_t inner_beats = row_b_blk / 8;

        if (path == 1) {
            // Path 1: spatial = r (8 channels = 8 rows of a D-tile).
            uint32_t ts_src[3] = { 8,           row_b_blk,  meshRow * row_b_rm };
            uint32_t tb_src[3] = { inner_beats, N_T,        M_T                };
            uint32_t ts_dst[3] = { 8,           tile_b,     N_T * tile_b       };
            uint32_t tb_dst[3] = { inner_beats, N_T,        M_T                };
            xdma_layout_run(st.xdma_src, dst_addr, row_b_rm, row_b_blk,
                            3, ts_src, tb_src, ts_dst, tb_dst, false);
        } else if (path == 2) {
            uint32_t r_outer = meshRow / 8;
            uint32_t ts_src[4] = { 8,           8 * row_b_rm,  row_b_blk,  meshRow * row_b_rm };
            uint32_t tb_src[4] = { inner_beats, r_outer,       N_T,        M_T                };
            uint32_t ts_dst[4] = { 8,           8 * row_b_blk, tile_b,     N_T * tile_b       };
            uint32_t tb_dst[4] = { inner_beats, r_outer,       N_T,        M_T                };
            xdma_layout_run(st.xdma_src, dst_addr, row_b_rm, row_b_blk,
                            4, ts_src, tb_src, ts_dst, tb_dst, false);
        } else if (path == 3) {
            // Path 3: spatial = c_chunk_inner; one xDMA spatial sweep covers
            // a 64-byte slice of a D-tile row.
            uint32_t c_outer = row_b_blk / 64;
            uint32_t ts_src[4] = { 64,      row_b_rm,  row_b_blk,  meshRow * row_b_rm };
            uint32_t tb_src[4] = { c_outer, meshRow,   N_T,        M_T                };
            uint32_t ts_dst[4] = { 64,      row_b_blk, tile_b,     N_T * tile_b       };
            uint32_t tb_dst[4] = { c_outer, meshRow,   N_T,        M_T                };
            xdma_layout_run(st.xdma_src, dst_addr, 8, 8,
                            4, ts_src, tb_src, ts_dst, tb_dst, false);
        } else if (path == 4) {
            uint32_t n_outer = N_T / 8;
            uint32_t ts_src[4] = { 8,           row_b_rm,  8 * row_b_blk, meshRow * row_b_rm };
            uint32_t tb_src[4] = { inner_beats, meshRow,   n_outer,       M_T                };
            uint32_t ts_dst[4] = { 8,           row_b_blk, 8 * tile_b,    N_T * tile_b       };
            uint32_t tb_dst[4] = { inner_beats, meshRow,   n_outer,       M_T                };
            xdma_layout_run(st.xdma_src, dst_addr, row_b_blk, tile_b,
                            4, ts_src, tb_src, ts_dst, tb_dst, false);
        } else /* path == 5 */ {
            uint32_t m_outer = M_T / 8;
            uint32_t ts_src[4] = { 8,           row_b_rm,  row_b_blk,  8 * meshRow * row_b_rm };
            uint32_t tb_src[4] = { inner_beats, meshRow,   N_T,        m_outer                };
            uint32_t ts_dst[4] = { 8,           row_b_blk, tile_b,     8 * N_T * tile_b       };
            uint32_t tb_dst[4] = { inner_beats, meshRow,   N_T,        m_outer                };
            xdma_layout_run(st.xdma_src, dst_addr,
                            meshRow * row_b_rm, N_T * tile_b,
                            4, ts_src, tb_src, ts_dst, tb_dst, false);
        }
        xdma_layout_stage_free(&st);
        hw_done = true;
    }

    if (!hw_done) {
        uint32_t N_cols = N_T * meshCol;
        // CPU fallback: stage non-local (L3) operands through L1.
        xdma_layout_stage_t si;
        xdma_layout_stage_out_t so;
        if (xdma_layout_stage_in(&si, src_addr, bytes) != 0 ||
            xdma_layout_stage_out(&so, dst_addr, bytes) != 0) {
            xdma_layout_stage_free(&si);   // frees src scratch if staged; no-op otherwise
            printf_safe("[Cluster %d Core %d]: row_major_to_d CPU-fallback L1 alloc failed!\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }
        volatile uint8_t *src = (volatile uint8_t *)(uint32_t)si.xdma_src;
        volatile uint8_t *dst = (volatile uint8_t *)so.l1;
        for (uint32_t m = 0; m < M_T; m++)
        for (uint32_t n = 0; n < N_T; n++)
        for (uint32_t r = 0; r < meshRow; r++)
        for (uint32_t c = 0; c < meshCol; c++) {
            uint32_t src_off = ((m * meshRow + r) * N_cols + n * meshCol + c) * elem_bytes;
            uint32_t dst_off = (((m * N_T + n) * meshRow + r) * meshCol + c) * elem_bytes;
            for (uint32_t b = 0; b < elem_bytes; b++) dst[dst_off + b] = src[src_off + b];
        }
        xdma_layout_stage_out_flush(&si, &so, dst_addr, bytes);   // transfer + flush + free both
    }
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// The 9 runnable xdma_row_major_to_d kernels = (array shape) x (elem_bytes). Each binds its mesh and
// element width as compile-time constants, so the AGU-path decision tree above folds away and
// the wrapper IS its path -- there is no runtime `if` left to pick the wrong one.
#define BINGO_DEF_XDMA_ROW_MAJOR_TO_D(suffix, d1, d2, eb)                            \
    SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_row_major_to_d_##suffix(void *arg)     \
    { return __xdma_row_major_to_d_impl(arg, (d1), (d2), (eb)); }

BINGO_DEF_XDMA_ROW_MAJOR_TO_D(e1_M32N32, 32, 32, 1)
BINGO_DEF_XDMA_ROW_MAJOR_TO_D(e2_M32N32, 32, 32, 2)
BINGO_DEF_XDMA_ROW_MAJOR_TO_D(e4_M32N32, 32, 32, 4)
BINGO_DEF_XDMA_ROW_MAJOR_TO_D(e1_M1N32, 1, 32, 1)
BINGO_DEF_XDMA_ROW_MAJOR_TO_D(e2_M1N32, 1, 32, 2)
BINGO_DEF_XDMA_ROW_MAJOR_TO_D(e4_M1N32, 1, 32, 4)
BINGO_DEF_XDMA_ROW_MAJOR_TO_D(e1_M16N16, 16, 16, 1)
BINGO_DEF_XDMA_ROW_MAJOR_TO_D(e2_M16N16, 16, 16, 2)
BINGO_DEF_XDMA_ROW_MAJOR_TO_D(e4_M16N16, 16, 16, 4)
