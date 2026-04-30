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
        xdma_memcpy_1d_full_addr(src_addr, dst_addr, data_size);
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
        xdma_memcpy_nd_full_addr(
            src_addr, dst_addr,
            spatial_stride_src, spatial_stride_dst,
            num_dims, t_strides_src, t_bounds_src,
            num_dims, t_strides_dst, t_bounds_dst,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        );
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
// High-level xDMA kernels: user provides shapes, kernel computes AGU config.
// These wrap the low-level reshape kernel with automatic stride computation.
// ==========================================================================

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_transpose_2d(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_transpose_2d_args_t);
    // Transpose a 2D matrix [M, N] -> [N, M].
    // If the xDMA was generated with the 8x8-tile transposer reader
    // extension (READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16 is defined in
    // snax_xdma_addr.h), uses the HW transposer for tile-level 8x8
    // transpose. Otherwise falls back to a CPU byte-by-byte transpose.
    //
    // HW path constraints: M % 8 == 0, N * elem % 8 == 0.
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
        uint32_t elem = a[6];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_transpose_2d_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

#ifdef READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16
        // ── HW Transposer path ──
        // The Transposer extension transposes each 8x8 tile of uint8 elements
        // (or tile_width x tile_width for other element widths) in the reader
        // data path. The AGU iterates tile-by-tile across the matrix.
        //
        // Reference: snax-xdma-transpose datagen.py
        // For MN→MN transpose with enable_transpose=true, uint8:
        //   tile_width = 8
        //   transfer_per_transpose = 1 (8x8x8bit = 512bit fits in one bus word)
        //   spatial_stride_src = N * elem (row width in src)
        //   spatial_stride_dst = M * elem (row width in dst, which is NxM transposed)
        //   temporal: dim0=transfer_per_transpose, dim1=tiles_across_cols, dim2=tiles_across_rows
        //   src strides: [8, tile_width*elem, N*tile_width*elem]
        //   dst strides: [8, M*tile_width*elem, tile_width*elem]
        {
            uint32_t tile_w = 8;
            uint32_t elem_bits = elem * 8;
            uint32_t tpt = (tile_w * tile_w * elem_bits + 511) / 512; // transfers per transpose

            // Disable all, then enable transposer on reader side
            BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
            xdma_disable_all_extensions();
            xdma_enable_src_ext(0, NULL);

            uint32_t spatial_stride_src = N * elem;
            uint32_t spatial_stride_dst = M * elem;

            uint32_t t_strides_src[3] = {
                8,                          // dim0: within one transfer
                tile_w * elem,              // dim1: next tile horizontally
                N * tile_w * elem           // dim2: next tile-row
            };
            uint32_t t_bounds_src[3] = {
                tpt,                        // transfers per 8x8 tile
                N / tile_w,                 // tiles across columns
                M / tile_w                  // tiles across rows
            };

            // Writer: transposed tile placement
            // After transpose, each 8x8 block's rows/cols are swapped.
            // dim1 stride = M*tile_w*elem (stride to next column-block in transposed output)
            // dim2 stride = tile_w*elem   (stride to next row-block in transposed output)
            uint32_t t_strides_dst[3] = {
                8,                          // dim0: within one transfer
                M * tile_w * elem,          // dim1: next column-block in output
                tile_w * elem               // dim2: next row-block in output
            };
            uint32_t t_bounds_dst[3] = {
                tpt,
                N / tile_w,
                M / tile_w
            };

            xdma_memcpy_nd_full_addr(
                src_addr, dst_addr,
                spatial_stride_src, spatial_stride_dst,
                3, t_strides_src, t_bounds_src,
                3, t_strides_dst, t_bounds_dst,
                0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
            );
            BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);

            BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
            int task_id = xdma_start();
            xdma_wait_task(src_addr, dst_addr, task_id);
            BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);

            // Disable transposer after use
            xdma_disable_src_ext(0);
        }
#else
        // ── CPU fallback (no Transposer extension) ──
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        volatile uint8_t *src = (volatile uint8_t *)(uint32_t)src_addr;
        volatile uint8_t *dst = (volatile uint8_t *)(uint32_t)dst_addr;
        for (uint32_t r = 0; r < M; r++) {
            for (uint32_t c = 0; c < N; c++) {
                for (uint32_t b = 0; b < elem; b++) {
                    dst[(c * M + r) * elem + b] = src[(r * N + c) * elem + b];
                }
            }
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
#endif
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
    // Constraints: out_rows % 8 == 0, out_cols*elem % 8 == 0, col_start*elem % 8 == 0.
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
        uint32_t elem      = a[10];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_submatrix_2d_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

        uint32_t out_rows = row_end - row_start;
        uint32_t out_cols = col_end - col_start;
        uint32_t src_row_bytes = src_cols * elem;
        uint32_t out_row_bytes = out_cols * elem;

        // Offset source to start of sub-region
        src_addr += (uint64_t)(row_start * src_cols + col_start) * elem;

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
        xdma_memcpy_nd_full_addr(
            src_addr, dst_addr,
            spatial_stride_src, spatial_stride_dst,
            2, t_strides_src, t_bounds_src,
            2, t_strides_dst, t_bounds_dst,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        );
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
    // Constraints: M % 8 == 0, N*elem % 8 == 0.
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
        uint32_t elem = a[6];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_expand_2d_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

        uint32_t row_bytes = N * elem;

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
        xdma_memcpy_nd_full_addr(
            src_addr, dst_addr,
            spatial_stride_src, spatial_stride_dst,
            2, t_strides_src, t_bounds_src,
            2, t_strides_dst, t_bounds_dst,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        );
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
        uint32_t elem     = a[10];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_concat_2d_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

        // Apply offset to destination base address
        if (axis == 0) {
            // Row concat: offset entire rows
            dst_addr += (uint64_t)offset * dst_cols * elem;
        } else {
            // Column concat: offset within each row
            dst_addr += (uint64_t)offset * elem;
        }

        uint32_t src_row_bytes = src_cols * elem;
        uint32_t dst_row_bytes = dst_cols * elem;

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
        xdma_memcpy_nd_full_addr(
            src_addr, dst_addr,
            spatial_stride_src, spatial_stride_dst,
            2, t_strides_src, t_bounds_src,
            2, t_strides_dst, t_bounds_dst,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        );
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
        uint32_t elem       = a[10];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_pad_2d_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

        uint32_t dst_rows = src_rows + pad_top + pad_bottom;
        uint32_t dst_cols = src_cols + pad_left + pad_right;
        uint32_t total_bytes = dst_rows * dst_cols * elem;

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
        uint64_t dst_interior = dst_addr + (uint64_t)(pad_top * dst_cols + pad_left) * elem;

        uint32_t src_row_bytes = src_cols * elem;
        uint32_t dst_row_bytes = dst_cols * elem;

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
        xdma_memcpy_nd_full_addr(
            src_addr, dst_interior,
            spatial_stride_src, spatial_stride_dst,
            2, t_strides_src, t_bounds_src,
            2, t_strides_dst, t_bounds_dst,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        );
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
        uint32_t elem         = a[9];
        bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_gather_2d_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

        // Offset source to the first gathered row
        uint64_t src_base = src_addr + (uint64_t)index_start * src_cols * elem;

        uint32_t row_bytes = src_cols * elem;

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
        xdma_memcpy_nd_full_addr(
            src_base, dst_addr,
            spatial_stride_src, spatial_stride_dst,
            2, t_strides_src, t_bounds_src,
            2, t_strides_dst, t_bounds_dst,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        );
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
// tiling. See HeMAiA/util/sim/layout_convert.py for the Python reference.
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
//   HW Transposer extension (defined(READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16)) is
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
// A↔R kernels — axes (m, k, r, s); inner row = tileSize·elem bytes:
//   Path 1: meshRow == 8 && (tileSize·elem) % 8 == 0     spatial = r
//   Path 2: meshRow > 8 && meshRow % 8 == 0 && …%8 == 0  spatial = r_inner
//   Path 3: meshRow ∈ {1,2,4} && (tileSize·elem) % 64==0 spatial = s_chunk_inner
//   Path 4: (tileSize·elem) %8==0 && K_T %8==0           spatial = k_inner
//   Path 5: (tileSize·elem) %8==0 && M_T %8==0           spatial = m_inner
//   else  : CPU fallback
//
// D↔R kernels — axes (m, n, r, c); inner row = meshCol·elem bytes:
//   Path 1: meshRow == 8 && (meshCol·elem) % 8 == 0      spatial = r
//   Path 2: meshRow > 8 && meshRow % 8 == 0 && …%8 == 0  spatial = r_inner
//   Path 3: meshRow ∈ {1,2,4} && (meshCol·elem) % 64==0  spatial = c_chunk_inner
//   Path 4: (meshCol·elem) %8==0 && N_T %8==0            spatial = n_inner
//   Path 5: (meshCol·elem) %8==0 && M_T %8==0            spatial = m_inner
//   else  : CPU fallback
//
// B↔R kernels — axes (n, k, c, s); HW Transposer + AGU sub-block iteration:
//   HW path: defined(READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16) && tileSize %8==0 && meshCol %8==0
//     For each (n,k) tile, decompose into (meshCol/8) x (tileSize/8) element
//     sub-blocks; each sub-block goes through one HW Transposer block. The
//     AGU iterates tpt x c_sub x s_sub x k x n  → 5 temporal dims (max).
//   else  : CPU fallback
//
// ─── Resulting coverage matrix (VersaCore array_shapes x kernel x elem) ───
//   array_shape (mR,tS,mC) | A↔R INT8         | A↔R INT32        | D↔R         | B↔R
//   ─────────────────────────────────────────────────────────────────────────────────
//   0 (32,4,32)            | CPU¹             | HW (path 2)      | HW (path 2) | CPU²
//   1 (1,8,64)             | HW (p4 if K_T%8) | HW (p4 if K_T%8) | HW (path 3) | HW
//   2 (4,8,64)             | HW (p4 if K_T%8) | HW (p4 if K_T%8) | HW (path 3) | HW
//   3 (8,8,64)             | HW (path 1)      | HW (path 1)      | HW (path 1) | HW
//   4 (8,32,8)             | HW (path 1)      | HW (path 1)      | HW (path 1) | HW
//
//   ¹ A-tile inner row = tileSize·elem = 4 < 8 bytes; can't form an 8-byte
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
#ifdef READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16
    if (use_transposer) xdma_enable_src_ext(0, NULL);
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

#ifdef READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16
    if (use_transposer) xdma_disable_src_ext(0);
#endif
}

// D-layout → row-major. See section banner for the path table.
//   Coverage: paths 1–5 cover all 5 VersaCore array_shapes via paths 1–3.
//   Arg layout: src_hi/lo, dst_hi/lo, M_T, N_T, meshRow, meshCol, elem_bytes.
//   Strides used by all paths (factored out for readability):
//     row_bytes_dst       = N_T * meshCol * elem   (full row of row-major R)
//     tile_bytes_dst_skip = meshRow * row_bytes_dst (advance past meshRow rows)
//     tile_bytes_src      = meshRow * meshCol * elem (one D-tile worth of bytes)
//     row_bytes_src       = meshCol * elem         (one row inside a D-tile)
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
    uint32_t elem = a[8];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_d_to_row_major_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    uint32_t bytes     = M_T * N_T * meshRow * meshCol * elem;
    uint32_t row_b_src = meshCol * elem;          // 1 row of a D-tile
    uint32_t row_b_dst = N_T * meshCol * elem;    // 1 row of row-major R
    uint32_t tile_b    = meshRow * meshCol * elem;
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
            uint32_t src_off = (((m * N_T + n) * meshRow + r) * meshCol + c) * elem;
            uint32_t dst_off = ((m * meshRow + r) * N_cols + n * meshCol + c) * elem;
            for (uint32_t b = 0; b < elem; b++) dst[dst_off + b] = src[src_off + b];
        }
    }
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// row-major → A-layout. See section banner for the path table.
//   Coverage: paths 1–5; array_shape 0 INT8 (tileSize·elem=4) lacks any
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
    uint32_t elem = a[8];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_row_major_to_a_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    uint32_t bytes     = M_T * K_T * meshRow * tileSize * elem;
    uint32_t row_b_blk = tileSize * elem;          // 1 row inside one A-tile (dst-packed)
    uint32_t row_b_rm  = K_T * tileSize * elem;    // 1 row of row-major R (src-side)
    uint32_t tile_b    = meshRow * tileSize * elem;
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
            uint32_t src_off = ((m * meshRow + r) * K_cols + k * tileSize + s) * elem;
            uint32_t dst_off = (((m * K_T + k) * meshRow + r) * tileSize + s) * elem;
            for (uint32_t b = 0; b < elem; b++) dst[dst_off + b] = src[src_off + b];
        }
    }
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// row-major → B-layout. B↔R is per-(n,k)-tile transpose-then-tile, so we
// drive the xDMA Transposer reader extension (mirrors xdma_transpose_2d).
// The Transposer is fixed at 8x8-byte blocks, so the (n,k) B-tile is
// decomposed into (meshCol/8) c-sub x (tileSize/8) s-sub element sub-blocks
// and the AGU iterates them in addition to (n, k).
//   Coverage: HW path when defined(READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16) && tileSize%8==0 && meshCol%8==0
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
    uint32_t elem = a[8];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_row_major_to_b_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    uint32_t bytes  = K_T * tileSize * N_T * meshCol * elem;
    bool hw_done = false;

#ifdef READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16
    if ((tileSize % 8) == 0 && (meshCol % 8) == 0) {
        xdma_layout_stage_t st;
        if (xdma_layout_stage_in(&st, src_addr, bytes) != 0) {
            printf_safe("[Cluster %d Core %d]: row_major_to_b L1 alloc failed!\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }

        const uint32_t tile_w   = 8;                                    // HW transposer block
        uint32_t elem_bits      = elem * 8;
        uint32_t tpt            = (tile_w * tile_w * elem_bits + 511) / 512;
        uint32_t row_b_src      = N_T * meshCol * elem;                 // R row width
        uint32_t b_tile_b       = tileSize * meshCol * elem;            // one B-tile bytes
        uint32_t c_subs         = meshCol  / tile_w;                    // # 8-elem c chunks
        uint32_t s_subs         = tileSize / tile_w;                    // # 8-elem s chunks

        // 8 channels = 8 byte-rows of one transposer block (= 8 c values).
        // Each channel writes to a c_inner position in B; spatial_stride_dst
        // crosses one c step inside the B-tile = tileSize·elem bytes.
        uint32_t spatial_stride_src = row_b_src;
        uint32_t spatial_stride_dst = tileSize * elem;

        // 5 temporal dims max:
        //   [0] tpt          : within transposer block (1 for elem=1, 4 for elem=4)
        //   [1] c_sub        : next 8 c-elements inside the B-tile
        //   [2] s_sub        : next 8 s-elements inside the B-tile
        //   [3] k            : next R row-tile / next (n,k) tile in B
        //   [4] n            : next R col-tile / next n-group in B
        uint32_t ts_src[5] = {
            8,                          // tpt: advance 8 bytes within channel
            tile_w * elem,              // c_sub: +8 cols in R
            tile_w * row_b_src,         // s_sub: +8 rows in R
            tileSize * row_b_src,       // k: +tileSize rows in R
            meshCol * elem              // n: +meshCol cols in R
        };
        uint32_t tb_src[5] = { tpt, c_subs, s_subs, K_T, N_T };
        uint32_t ts_dst[5] = {
            8,                          // tpt: 8 bytes within channel
            tile_w * tileSize * elem,   // c_sub: +8 c-rows in B-tile (each c-row = tileSize·elem)
            tile_w * elem,              // s_sub: +8 s-bytes in each c-row of B-tile
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
            uint32_t src_off = ((k * tileSize + s) * N_cols + n * meshCol + c) * elem;
            uint32_t dst_off = (((n * K_T + k) * meshCol + c) * tileSize + s) * elem;
            for (uint32_t b = 0; b < elem; b++) dst[dst_off + b] = src[src_off + b];
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
    uint32_t elem = a[8];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_a_to_row_major_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    uint32_t bytes     = M_T * K_T * meshRow * tileSize * elem;
    uint32_t row_b_blk = tileSize * elem;          // 1 row inside one A-tile (src-packed)
    uint32_t row_b_rm  = K_T * tileSize * elem;    // 1 row of row-major R (dst-side)
    uint32_t tile_b    = meshRow * tileSize * elem;
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
            uint32_t src_off = (((m * K_T + k) * meshRow + r) * tileSize + s) * elem;
            uint32_t dst_off = ((m * meshRow + r) * K_cols + k * tileSize + s) * elem;
            for (uint32_t b = 0; b < elem; b++) dst[dst_off + b] = src[src_off + b];
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
    uint32_t elem = a[8];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_b_to_row_major_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    uint32_t bytes  = K_T * tileSize * N_T * meshCol * elem;
    bool hw_done = false;

#ifdef READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16
    if ((tileSize % 8) == 0 && (meshCol % 8) == 0) {
        xdma_layout_stage_t st;
        if (xdma_layout_stage_in(&st, src_addr, bytes) != 0) {
            printf_safe("[Cluster %d Core %d]: b_to_row_major L1 alloc failed!\r\n",
                        snrt_cluster_idx(), snrt_cluster_core_idx());
            return BINGO_RET_FAIL;
        }

        const uint32_t tile_w   = 8;
        uint32_t elem_bits      = elem * 8;
        uint32_t tpt            = (tile_w * tile_w * elem_bits + 511) / 512;
        uint32_t row_b_dst      = N_T * meshCol * elem;
        uint32_t b_tile_b       = tileSize * meshCol * elem;
        uint32_t c_subs         = meshCol  / tile_w;
        uint32_t s_subs         = tileSize / tile_w;

        // Reader: 8 channels each read one byte-row of the src 8x8 block;
        // within a B-tile, byte-rows (c values) are tileSize·elem apart.
        uint32_t spatial_stride_src = tileSize * elem;
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
            tile_w * tileSize * elem,   // c_sub: +8 c-rows in B-tile
            tile_w * elem,              // s_sub: +8 s-bytes in each c-row of B-tile
            b_tile_b,                   // k: next (n,k) tile within n-group
            K_T * b_tile_b              // n: next n-group of K_T tiles
        };
        uint32_t tb_src[5] = { tpt, c_subs, s_subs, K_T, N_T };
        uint32_t ts_dst[5] = {
            8,                          // tpt
            tile_w * elem,              // c_sub: +8 cols in R (same row band)
            tile_w * row_b_dst,         // s_sub: +8 rows down in R
            tileSize * row_b_dst,       // k: +tileSize rows in R
            meshCol * elem              // n: +meshCol cols in R
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
            uint32_t src_off = (((n * K_T + k) * meshCol + c) * tileSize + s) * elem;
            uint32_t dst_off = ((k * tileSize + s) * N_cols + n * meshCol + c) * elem;
            for (uint32_t b = 0; b < elem; b++) dst[dst_off + b] = src[src_off + b];
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
    uint32_t elem = a[8];
    bingo_kernel_scratchpad_t* sp = BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_row_major_to_d_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    uint32_t bytes     = M_T * N_T * meshRow * meshCol * elem;
    uint32_t row_b_blk = meshCol * elem;          // 1 row of a D-tile (dst-side packing)
    uint32_t row_b_rm  = N_T * meshCol * elem;    // 1 row of row-major R (src-side)
    uint32_t tile_b    = meshRow * meshCol * elem;
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
            uint32_t src_off = ((m * meshRow + r) * N_cols + n * meshCol + c) * elem;
            uint32_t dst_off = (((m * N_T + n) * meshRow + r) * meshCol + c) * elem;
            for (uint32_t b = 0; b < elem; b++) dst[dst_off + b] = src[src_off + b];
        }
    }
    sp->return_value = (uint32_t)dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

