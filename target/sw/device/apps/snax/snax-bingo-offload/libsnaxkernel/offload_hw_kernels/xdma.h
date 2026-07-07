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
        int task_id = xdma_start();
        xdma_wait_task(src_addr, dst_addr, task_id);
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
        int task_id = xdma_start();
        xdma_wait_task(src_addr, dst_addr, task_id);
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
// where tiles = num_elem_per_operand / 16. Reader address for (operand o,
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
//       derives operand_stride = src_b - src_a (so src_b must be the higher
//       address — the reader can only stride forward, see below).
//
// Constraints / fallback
// ----------------------
//   - num_elem_per_operand must be a multiple of 16 (one 512b bus word).
//   - operands must be laid out at a constant FORWARD stride (ascending
//     addresses); a wrapped "negative" stride makes the reader generate an
//     out-of-range address and stall.
//   - Falls back to a plain CPU int32 accumulate when the writer extension is
//     not present in the generated HW (WRITER_EXT_ELEMENTWISEADDBIT32 undefined).
// ==========================================================================
static inline uint32_t xdma_elementwise_add_run(
    uint64_t src_base, uint64_t dst_addr,
    uint32_t num_elem_per_operand, uint32_t num_operands,
    uint32_t operand_stride)
{
    // Each output vector is one 512b bus word = XDMA_WIDTH/4 = 16 int32, so the
    // per-operand element count must be a whole, non-zero number of bus words;
    // otherwise the tile count truncates and the transfer would be wrong.
    if (num_elem_per_operand == 0 ||
        (num_elem_per_operand % (XDMA_WIDTH / 4)) != 0) {
        printf_safe("[Cluster %d Core %d]: Error! xDMA elementwise_add "
                    "num_elem_per_operand=%u must be a positive multiple of %u\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(),
                    num_elem_per_operand, (unsigned)(XDMA_WIDTH / 4));
        return BINGO_RET_FAIL;
    }
    uint32_t tiles = num_elem_per_operand / (XDMA_WIDTH / 4);  // 16 int32/vector
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
    int task_id = xdma_start();
    xdma_wait_task(src_base, dst_addr, task_id);
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
    xdma_disable_dst_ext(WRITER_EXT_ELEMENTWISEADDBIT32);
#else
    // CPU fallback: dst[e] = sum_o src[o][e] over int32 elements.
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
    volatile int32_t *dst = (volatile int32_t *)(uint32_t)dst_addr;
    for (uint32_t e = 0; e < num_elem_per_operand; e++) {
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
    //   [4] num_elem_per_operand (multiple of 16)
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
        uint32_t num_elem_per_operand = a[4];
        uint32_t num_operands = a[5];
        uint32_t operand_stride = a[6];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_elementwise_add_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        if (xdma_elementwise_add_run(src_base, dst_addr, num_elem_per_operand,
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
    // Two-operand convenience form: dst = a + b (int32, num_int32 elems, mult of 16).
    //   [0] src_a_hi [1] src_a_lo [2] src_b_hi [3] src_b_lo
    //   [4] dst_hi   [5] dst_lo   [6] num_int32
    //
    // Identical HW path to the N-operand kernel above, fixed to num_operands = 2
    // with the stride derived from the two addresses instead of passed in:
    //   add_ab(a, b, dst, n)  ==  add(a, dst, n, 2, b - a).
    // Use this when you have two independent buffers and don't want to compute a
    // stride. The two buffers can be anywhere, BUT src_b must be at a HIGHER
    // address than src_a (the reader only strides forward; see below).
    if (snrt_is_dm_core()) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t *a = (uint32_t *)arg;
        uint64_t src_a = make_u64(a[0], a[1]);
        uint64_t src_b = make_u64(a[2], a[3]);
        uint64_t dst_addr = make_u64(a[4], a[5]);
        uint32_t num_int32 = a[6];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_elementwise_add_ab_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        // The reader AGU strides FORWARD from src_a by (src_b - src_a) to reach
        // operand b, so src_b must lie at a higher address than src_a (a wrapped
        // "negative" stride would generate an out-of-range read and stall).
        uint32_t operand_stride = (uint32_t)src_b - (uint32_t)src_a;
        if (xdma_elementwise_add_run(src_a, dst_addr, num_int32, 2, operand_stride) != BINGO_RET_SUCC)
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
// High-level xDMA kernels: user provides shapes, kernel computes AGU config.
// These wrap the low-level reshape kernel with automatic stride computation.
// ==========================================================================

// SW 2D transpose [M,N] -> [N,M] at element granularity. Correct for any
// element width — used for widths the HW transposer has no native mode for
// (int32) and when the Transposer extension is absent. src/dst must be local L1.
static inline void xdma_cpu_transpose_2d(uint64_t src_addr, uint64_t dst_addr,
                                         uint32_t M, uint32_t N, uint32_t elem_bytes)
{
    volatile uint8_t *src = (volatile uint8_t *)(uint32_t)src_addr;
    volatile uint8_t *dst = (volatile uint8_t *)(uint32_t)dst_addr;
    for (uint32_t r = 0; r < M; r++)
        for (uint32_t c = 0; c < N; c++)
            for (uint32_t b = 0; b < elem_bytes; b++)
                dst[(c * M + r) * elem_bytes + b] = src[(r * N + c) * elem_bytes + b];
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
        if (elem_bytes == 1 || elem_bytes == 2) {
            uint32_t tile_w = 8;
            uint32_t tpt = (tile_w * tile_w * (elem_bytes * 8) + 511) / 512; // transfers per transpose (8x8 tile of elem_bytes*8-bit elems / 512b bus)

            // The xDMA reader is tied to local L1, so stage src into local L1 if
            // it isn't already there (zero-copy fast path when src is already
            // local). dst is written by the global-addressing writer, so it
            // needs no staging.
            uint32_t bytes = M * N * elem_bytes;
            xdma_layout_stage_t st;
            if (xdma_layout_stage_in(&st, src_addr, bytes) != 0) {
                printf_safe("[Cluster %d Core %d]: transpose_2d L1 alloc failed!\r\n",
                            snrt_cluster_idx(), snrt_cluster_core_idx());
                return BINGO_RET_FAIL;
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

            xdma_memcpy_nd_full_addr(
                st.xdma_src, dst_addr,
                spatial_stride_src, spatial_stride_dst,
                3, t_strides_src, t_bounds_src,
                3, t_strides_dst, t_bounds_dst,
                0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
            );
            BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);

            BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
            int task_id = xdma_start();
            xdma_wait_task(st.xdma_src, dst_addr, task_id);
            BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);

            // Disable transposer after use
            xdma_disable_dst_ext(WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16);
            xdma_layout_stage_free(&st);
            sp->return_value = (uint32_t)dst_addr;
            sp->num_return_values = 0;
            return BINGO_RET_SUCC;
        }
#endif
        // ── SW transpose ──
        // Correct for any element width; reached for 32-bit (no native HW
        // transposer mode) or when the Transposer extension is absent.
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        xdma_cpu_transpose_2d(src_addr, dst_addr, M, N, elem_bytes);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
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
        int task_id = xdma_start();
        xdma_wait_task(src_addr, dst_addr, task_id);
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
        int task_id = xdma_start();
        xdma_wait_task(src_addr, dst_addr, task_id);
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
        int task_id = xdma_start();
        xdma_wait_task(src_addr, dst_addr, task_id);
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
        int task_id = xdma_start();
        xdma_wait_task(src_addr, dst_interior, task_id);
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
        int task_id = xdma_start();
        xdma_wait_task(src_base, dst_addr, task_id);
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
// ─── Resulting coverage matrix (VersaCore array_shapes x kernel x elem_bytes) ───
//   array_shape (mR,tS,mC) | A↔R INT8         | A↔R INT32        | D↔R         | B↔R
//   ─────────────────────────────────────────────────────────────────────────────────
//   0 (32,4,32)            | CPU¹             | HW (path 2)      | HW (path 2) | CPU²
//   1 (1,8,64)             | HW (p4 if K_T%8) | HW (p4 if K_T%8) | HW (path 3) | HW
//   2 (4,8,64)             | HW (p4 if K_T%8) | HW (p4 if K_T%8) | HW (path 3) | HW
//   3 (8,8,64)             | HW (path 1)      | HW (path 1)      | HW (path 1) | HW
//   4 (8,32,8)             | HW (path 1)      | HW (path 1)      | HW (path 1) | HW
//
//   ¹ A-tile inner row = tileSize·elem_bytes = 4 < 8 bytes; can't form an 8-byte
//     beat that's contiguous in both row-major src and packed A dst. CPU.
//   ² Same root cause: tileSize=4 prevents 8x8-byte sub-blocking of B-tile.
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
    int task_id = xdma_start();
    xdma_wait_task(src, dst, task_id);
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
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_d_to_row_major(void *arg)
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
    uint32_t meshRow = a[6], meshCol = a[7];
    uint32_t elem_bytes = a[8];
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
        // CPU fallback (no HW path matched the shape).
        uint32_t N_cols = N_T * meshCol;
        volatile uint8_t *src = (volatile uint8_t *)(uint32_t)src_addr;
        volatile uint8_t *dst = (volatile uint8_t *)(uint32_t)dst_addr;
        for (uint32_t m = 0; m < M_T; m++)
        for (uint32_t n = 0; n < N_T; n++)
        for (uint32_t r = 0; r < meshRow; r++)
        for (uint32_t c = 0; c < meshCol; c++) {
            uint32_t src_off = (((m * N_T + n) * meshRow + r) * meshCol + c) * elem_bytes;
            uint32_t dst_off = ((m * meshRow + r) * N_cols + n * meshCol + c) * elem_bytes;
            for (uint32_t b = 0; b < elem_bytes; b++) dst[dst_off + b] = src[src_off + b];
        }
    }
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// row-major → A-layout. See section banner for the path table.
//   Coverage: paths 1–5; array_shape 0 INT8 (tileSize·elem_bytes=4) lacks any
//   8-byte beat that's contiguous in both src (row-major rows) and dst
//   (packed A) → CPU. Other shapes hit a HW path.
//   Arg layout: src_hi/lo, dst_hi/lo, M_T, K_T, meshRow, tileSize, elem_bytes.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_row_major_to_a(void *arg)
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
    uint32_t meshRow = a[6], tileSize = a[7];
    uint32_t elem_bytes = a[8];
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
        volatile uint8_t *src = (volatile uint8_t *)(uint32_t)src_addr;
        volatile uint8_t *dst = (volatile uint8_t *)(uint32_t)dst_addr;
        for (uint32_t m = 0; m < M_T; m++)
        for (uint32_t k = 0; k < K_T; k++)
        for (uint32_t r = 0; r < meshRow; r++)
        for (uint32_t s = 0; s < tileSize; s++) {
            uint32_t src_off = ((m * meshRow + r) * K_cols + k * tileSize + s) * elem_bytes;
            uint32_t dst_off = (((m * K_T + k) * meshRow + r) * tileSize + s) * elem_bytes;
            for (uint32_t b = 0; b < elem_bytes; b++) dst[dst_off + b] = src[src_off + b];
        }
    }
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// row-major → B-layout. B↔R is per-(n,k)-tile transpose-then-tile, so we
// drive the xDMA Transposer writer extension (mirrors xdma_transpose_2d).
// The Transposer is fixed at 8x8-byte blocks, so the (n,k) B-tile is
// decomposed into (meshCol/8) c-sub x (tileSize/8) s-sub element sub-blocks
// and the AGU iterates them in addition to (n, k).
//   Coverage: HW path when defined(WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16) && tileSize%8==0 && meshCol%8==0
//             (all VersaCore array_shapes 1..4); CPU when tileSize=4 (shape 0).
//   Arg layout: src_hi/lo, dst_hi/lo, K_T, N_T, tileSize, meshCol, elem_bytes.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_row_major_to_b(void *arg)
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
    uint32_t tileSize = a[6], meshCol = a[7];
    uint32_t elem_bytes = a[8];
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
        volatile uint8_t *src = (volatile uint8_t *)(uint32_t)src_addr;
        volatile uint8_t *dst = (volatile uint8_t *)(uint32_t)dst_addr;
        for (uint32_t n = 0; n < N_T; n++)
        for (uint32_t k = 0; k < K_T; k++)
        for (uint32_t c = 0; c < meshCol; c++)
        for (uint32_t s = 0; s < tileSize; s++) {
            uint32_t src_off = ((k * tileSize + s) * N_cols + n * meshCol + c) * elem_bytes;
            uint32_t dst_off = (((n * K_T + k) * meshCol + c) * tileSize + s) * elem_bytes;
            for (uint32_t b = 0; b < elem_bytes; b++) dst[dst_off + b] = src[src_off + b];
        }
    }
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// A-layout → row-major. Inverse of row_major_to_a: src/dst stride arrays
// are swapped versus the forward kernel; same path-selection logic.
//   Coverage: paths 1–5; same CPU-only cell as forward (array_shape 0 INT8).
//   Arg layout: src_hi/lo, dst_hi/lo, M_T, K_T, meshRow, tileSize, elem_bytes.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_a_to_row_major(void *arg)
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
    uint32_t meshRow = a[6], tileSize = a[7];
    uint32_t elem_bytes = a[8];
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
        volatile uint8_t *src = (volatile uint8_t *)(uint32_t)src_addr;
        volatile uint8_t *dst = (volatile uint8_t *)(uint32_t)dst_addr;
        for (uint32_t m = 0; m < M_T; m++)
        for (uint32_t k = 0; k < K_T; k++)
        for (uint32_t r = 0; r < meshRow; r++)
        for (uint32_t s = 0; s < tileSize; s++) {
            uint32_t src_off = (((m * K_T + k) * meshRow + r) * tileSize + s) * elem_bytes;
            uint32_t dst_off = ((m * meshRow + r) * K_cols + k * tileSize + s) * elem_bytes;
            for (uint32_t b = 0; b < elem_bytes; b++) dst[dst_off + b] = src[src_off + b];
        }
    }
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// B-layout → row-major. Inverse of row_major_to_b: same per-(n,k)-tile
// transpose, src/dst stride arrays swapped, AGU iterates the same
// (c_sub, s_sub, k, n) sub-block grid.
//   Coverage: same as forward — HW for shapes 1..4, CPU for shape 0.
//   Arg layout: src_hi/lo, dst_hi/lo, K_T, N_T, tileSize, meshCol, elem_bytes.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_b_to_row_major(void *arg)
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
    uint32_t tileSize = a[6], meshCol = a[7];
    uint32_t elem_bytes = a[8];
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
        volatile uint8_t *src = (volatile uint8_t *)(uint32_t)src_addr;
        volatile uint8_t *dst = (volatile uint8_t *)(uint32_t)dst_addr;
        for (uint32_t n = 0; n < N_T; n++)
        for (uint32_t k = 0; k < K_T; k++)
        for (uint32_t c = 0; c < meshCol; c++)
        for (uint32_t s = 0; s < tileSize; s++) {
            uint32_t src_off = (((n * K_T + k) * meshCol + c) * tileSize + s) * elem_bytes;
            uint32_t dst_off = ((k * tileSize + s) * N_cols + n * meshCol + c) * elem_bytes;
            for (uint32_t b = 0; b < elem_bytes; b++) dst[dst_off + b] = src[src_off + b];
        }
    }
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// row-major → D-layout. Inverse of d_to_row_major: src/dst stride arrays
// are swapped versus the forward kernel; same path-selection logic.
//   Coverage: paths 1–5 cover all 5 VersaCore array_shapes via paths 1–3.
//   Arg layout: src_hi/lo, dst_hi/lo, M_T, N_T, meshRow, meshCol, elem_bytes.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_row_major_to_d(void *arg)
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
    uint32_t meshRow = a[6], meshCol = a[7];
    uint32_t elem_bytes = a[8];
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
        volatile uint8_t *src = (volatile uint8_t *)(uint32_t)src_addr;
        volatile uint8_t *dst = (volatile uint8_t *)(uint32_t)dst_addr;
        for (uint32_t m = 0; m < M_T; m++)
        for (uint32_t n = 0; n < N_T; n++)
        for (uint32_t r = 0; r < meshRow; r++)
        for (uint32_t c = 0; c < meshCol; c++) {
            uint32_t src_off = ((m * meshRow + r) * N_cols + n * meshCol + c) * elem_bytes;
            uint32_t dst_off = (((m * N_T + n) * meshRow + r) * meshCol + c) * elem_bytes;
            for (uint32_t b = 0; b < elem_bytes; b++) dst[dst_off + b] = src[src_off + b];
        }
    }
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

