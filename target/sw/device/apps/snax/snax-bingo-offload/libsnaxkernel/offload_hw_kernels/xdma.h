// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
//
// Core-level bingo xDMA kernels: DATA MOVEMENT ONLY. Generic 1D/6D copies,
// layout-transform kernels between row-major and versacore-tiled A/B/D, 2D
// helpers (transpose, submatrix, expand, concat, pad, gather), and the writer
// junction ops (elementwise_add). No versacore streamer CSR writes here.
//
// The FP16 stream operators and the fused whole-ops built on them live in
// simd.h, not here: on the split cluster the SIMD block is a separate engine.
// This file is what is genuinely the xDMA's -- the AGU, the Transposer (a wire
// permutation, not compute), Memset, the writer junctions, and the AXI path
// that makes cross-cluster and cross-chiplet transfers possible at all.

#pragma once

#include "../macros.h"
#include "snax_core_roles.h"  // snax_is_xdma_core()

// WHICH HART. These kernels program the SNAX xDMA through `csrw_ss(XDMA_CFG_ADDR + n)`,
// and XDMA_CFG_ADDR is 960 -- a CSR NUMBER, so the window is HART-LOCAL. The accelerator
// therefore has to be driven from the hart it is attached to, which the generated role map
// names SNAX_CORE_XDMA.
//
// The guards below must NOT use `snax_is_xdma_core()`. That is right only where one core
// carries both roles, which is not the case on snax_split_cluster, and the two are easy to
// confuse because the cluster hjson overloads the name: `xdma: true` on a core is the
// DMA-ISA boolean (the classic Snitch iDMA, role "idma"), while `snax_xdma_cfg` is the
// SNAX xDMA accelerator (role "xdma"). `snax_is_xdma_core()` finds the former.
//
// The iDMA hart has no SNAX accelerator, so CSR writes aimed at it land in a window with
// nothing behind it: no fault, no data moved, and a completion flag that reads back "done"
// because it never read anything.

// PUT THE MULTICAST DESTINATION SLOTS BACK TO ZERO.
//
// Slots 1..15 of XDMA_DST_ADDR_PTR live in the CSR bank and SURVIVE the task that set
// them, exactly like the writer junctions the chain gather clears on entry. Every kernel
// that programs its CSRs longhand -- __snax_bingo_kernel_xdma_memset below is the one that
// matters -- writes only slot 0 and relies on the rest being zero.

// Callers invoke it AFTER their wait, so it costs nothing the transfer was not paying.
static inline void bingo_xdma_disarm_multicast_slots(uint32_t armed)
{
    const uint32_t n = (armed == 0u || armed > XDMA_MAX_DST_COUNT)
                           ? XDMA_MAX_DST_COUNT : armed;
    for (uint32_t i = 1; i < n; i++) {
        snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB + i * 2, 0);
        snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_MSB + i * 2, 0);
    }
}

// Fill a local L1 region with a repeating 32-bit pattern, on the xDMA's writer path.
//
// The CSR addresses here are all COMPILE-TIME CONSTANTS, written out longhand rather than
// through snax_xdma_memcpy_nd(). That is not style: csrw_ss is a switch over every CSR
// address, so a constant folds to a one-cycle `csrw imm` while a COMPUTED one degrades to
// a jump-table load out of .rodata -- which this target maps to main memory -- plus an
// indirect jump, on every write. The general path also writes far more than a fill needs:
// 30 of its writes zero multicast slots this never uses.
//
// THAT LAST SENTENCE IS AN INVARIANT THIS KERNEL DEPENDS ON, not just an observation:
// skipping those 30 writes is only safe while slots 1..15 are zero on entry. Anything that
// arms them must call bingo_xdma_disarm_multicast_slots(n) before it yields the core --
// see that function for what happens when it does not.
//
// The reader channels are disabled deliberately. A disabled channel issues NO TCDM request
// at all, so the beat the writer sees comes entirely from the Memset extension -- no read,
// and no fetch of a constant from main memory, which is the whole point.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_memset(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_memset_args_t);
    BINGO_REQUIRE_CORE(snax_is_xdma_core(), "xdma_memset", "XDMA");
#ifndef WRITER_EXT_VERILOGMEMSET
    // The Memset extension is optional RTL. Refuse at call time rather than #error, so a
    // config without it still builds every other kernel in this file.
    printf_safe("[Cluster %d Core %d]: Error! xdma_memset needs WRITER_EXT_VERILOGMEMSET, "
                "which this cfg does not instantiate\r\n",
                snrt_cluster_idx(), snrt_cluster_core_idx());
    (void)arg;
    return BINGO_RET_FAIL;
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    __snax_bingo_kernel_xdma_memset_args_t *a =
        (__snax_bingo_kernel_xdma_memset_args_t *)arg;
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_memset_args_t);
    const uint32_t dst = a->dst_addr_lo;
    const uint32_t bytes = a->size;
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    if (bytes == 0u || (bytes & 63u)) {
        printf_safe("[Cluster %d Core %d]: Error! xdma_memset size=%d must be a non-zero "
                    "multiple of 64\r\n", snrt_cluster_idx(), snrt_cluster_core_idx(),
                    (int)bytes);
        return BINGO_RET_FAIL;
    }
    if (!xdma_addr_in_local_l1(((uint64_t)a->dst_addr_hi << 32) | dst)) {
        printf_safe("[Cluster %d Core %d]: Error! xdma_memset dst is not in local L1\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
    xdma_disable_all_extensions();
    // src == dst: the reader is off, so its address only has to be legal, never read.
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_LSB, dst);
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_MSB, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB, dst);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_MSB, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_SPATIAL_STRIDE_PTR, 8);
    snax_write_xdma_cfg_reg(XDMA_DST_SPATIAL_STRIDE_PTR, 8);
    // One temporal dimension: `beats` beats of 64 B. XDMA_WR_*_DIMS writes EVERY generated
    // dimension at a literal CSR address and gives the unused ones the neutral
    // bound=1/stride=0 -- so the runtime dim picks values, never addresses.
    uint32_t fill_bnd[1] = { bytes / 64u };
    uint32_t fill_str[1] = { 64u };
    XDMA_WR_SRC_DIMS(1u, fill_bnd, fill_str);
    XDMA_WR_DST_DIMS(1u, fill_bnd, fill_str);
    snax_write_xdma_cfg_reg(XDMA_SRC_ENABLED_CHAN_PTR, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_CHAN_PTR, 0xFFFFFFFFu);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_BYTE_PTR, 0xFFFFFFFFu);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLE_PTR, 1u << WRITER_EXT_VERILOGMEMSET);
    snax_write_xdma_cfg_reg(XDMA_DST_EXT_CSR_PTR, a->pattern);
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
    xdma_task_t task_id = xdma_start();
    xdma_wait_task(task_id);
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);

    sp->return_value = dst;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_1d_copy(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_1d_copy_args_t);
    // Copy 1d data from src to dst using xdma
    // Arg0: uint32_t src_addr_hi
    // Arg1: uint32_t src_addr_lo
    // Arg2: uint32_t dst_addr_hi
    // Arg3: uint32_t dst_addr_lo
    // Arg4: uint32_t size in Byte

    if (snax_is_xdma_core())
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
        printf_safe("[Cluster %d Core %d]: Error! XDMA copy must run on the xDMA core!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}

// ONE READ FROM MAIN MEMORY, N WRITES INTO N CLUSTERS' L1.
//
// The fan-out is in the writer's destination-address slots, so this is ONE transfer with
// ONE finish, not N copies. That is what separates it from a software loop over
// __snax_bingo_kernel_xdma_1d_copy: the loop's single issuer serialises and idles on the
// WAR edge, while the hardware commits every destination from the same read stream.
//
// Used by FlashAttention when the four clusters hold four query heads of ONE GQA group:
// they share K and V byte for byte, so pulling those bytes four times over the quadrant's
// single 512-bit path is four times the port time for one byte of information. See
// docs/fa_decomposition_hierarchy.md for the balance-point argument.
//
// No extension is armed. Multicast is base streamer behaviour -- the destination slots are
// always there, and the unicast path spends 15 of its config writes ZEROING them -- so
// this needs no #ifdef on an optional RTL block. It does disable any extension a previous
// task left armed, for the same reason the gather does: extensions and junctions live in
// their own CSR banks and survive the task that set them.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_multicast(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_multicast_args_t);
    BINGO_REQUIRE_CORE(snax_is_xdma_core(), "xdma_multicast", "XDMA");

    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    __snax_bingo_kernel_xdma_multicast_args_t *a =
        (__snax_bingo_kernel_xdma_multicast_args_t *)arg;
    uint64_t src = make_u64(a->src_addr_hi, a->src_addr_lo);
    uint32_t dst_num = a->dst_num;
    uint32_t size = a->size;
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_multicast_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    // The HW bound is a generated define, so it can only be checked here. A dst_num past
    // it would have its extra slots silently dropped by xdma_multicast_nd_full_address,
    // which is a wrong-data bug rather than a failure.
    if (dst_num < 1 || dst_num > BINGO_XDMA_MCAST_MAX || dst_num > XDMA_MAX_DST_COUNT) {
        printf_safe("[Cluster %d Core %d]: Error! xDMA multicast dst_num=%u must be "
                    "1..%u (arg struct) and <= %u (HW)\r\n", snrt_cluster_idx(),
                    snrt_cluster_core_idx(), dst_num,
                    (unsigned)BINGO_XDMA_MCAST_MAX, (unsigned)XDMA_MAX_DST_COUNT);
        return BINGO_RET_FAIL;
    }

    uint64_t dst[BINGO_XDMA_MCAST_MAX];
    for (uint32_t i = 0; i < dst_num; i++) {
        dst[i] = make_u64(a->dst_hi[i], a->dst_lo[i]);
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
    xdma_disable_all_extensions();
    BINGO_XDMA_TRY(xdma_multicast_1d_full_address(src, dst, dst_num, size),
                   "xdma_multicast");
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
    xdma_task_t task = xdma_start();
    xdma_wait_task(task);
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);

    bingo_xdma_disarm_multicast_slots(dst_num);

    sp->return_value = (uint32_t)dst[0];
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
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

    if (snax_is_xdma_core())
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
// single xDMA pass. The alternative is the sequential host int32-add chain
// (__host_bingo_kernel_add_i32), which walks L3<->host once per pair.
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

// ==========================================================================
// ChainGather -- the collective fold, in the fabric.
//
// Every other kernel in this file MOVES data. This one REDUCES it on the way: the
// collector's xDMA walks a chain of remote partials and a writer junction folds each
// arriving stream with that node's local read, so P partials cost one pass and the
// collector's buffer receives the answer, not the operands.
//
// For FlashAttention this is the cross-cluster epilogue. Each cluster holds its own
// (m, l) for the same query rows, and combining them is the online-softmax merge --
// m* = max m_c, l* = sum exp(m_c - m*) l_c -- which is precisely what
// WRITER_JCT_MONOIDJUNCTION computes. The O numerators then fold with the SAME chain
// under WRITER_JCT_ELEMENTWISEJUNCTION (ADD, FP32), after each cluster has rescaled
// its O by exp(m_c - m*). Two gathers and one rescale replace a gather-then-reduce
// that would move every partial to one cluster first.
//
// Three things here are not optional, each learned from a failure that looked like
// something else. They are cheap; keep them.
// ==========================================================================
// ==========================================================================
// Pack a FlashAttention (m, l) partial into the monoid junction's lane geometry.
//
// Plain scalar work, deliberately: it runs on the xDMA core so that the gather that
// consumes it is the very next thing that core does, with no cross-core handoff for 256
// bytes. That core is rv32ima with NO FPU, so the FP16 -> FP32 widening is done on the
// bit pattern. It is the whole reason this is a kernel and not two lines in the caller.
// ==========================================================================
static inline uint32_t bingo_f16_to_f32_bits(uint16_t h)
{
    uint32_t s = (uint32_t)(h >> 15) << 31;
    uint32_t e = (uint32_t)((h >> 10) & 0x1Fu);
    uint32_t m = (uint32_t)(h & 0x3FFu);
    if (e == 0u) {
        if (m == 0u) {
            return s;                       // +-0
        }
        // Subnormal: value = m * 2^-24. Shift until bit 10 is set; after k shifts the
        // value is (1+f) * 2^(-14-k), so the FP32 exponent field is 113 - k. `e`
        // counts -k and wraps, which is fine -- the sum is a small positive number.
        while ((m & 0x400u) == 0u) {
            m <<= 1;
            e--;
        }
        m &= 0x3FFu;
        return s | ((e + 113u) << 23) | (m << 13);
    }
    if (e == 0x1Fu) {
        return s | 0x7F800000u | (m << 13);  // inf / NaN, payload preserved
    }
    return s | ((e + 112u) << 23) | (m << 13);   // 127 - 15 = 112
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_pack_fa_partial(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_pack_fa_partial_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    __snax_bingo_kernel_pack_fa_partial_args_t *a =
        (__snax_bingo_kernel_pack_fa_partial_args_t *)arg;
    const uint16_t *src_m = (const uint16_t *)(uintptr_t)make_u64(a->src_m_addr_hi,
                                                                 a->src_m_addr_lo);
    const uint16_t *src_l = (const uint16_t *)(uintptr_t)make_u64(a->src_l_addr_hi,
                                                                 a->src_l_addr_lo);
    uint32_t *dst = (uint32_t *)(uintptr_t)make_u64(a->dst_addr_hi, a->dst_addr_lo);
    uint32_t n_rows = a->n_rows, S = a->slots;
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_pack_fa_partial_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    // F = 2 fields, so the beat holds 2*S lanes and a 512b FP32 beat has 16 of them.
    // A larger S would run one field's slots into the next field's lanes with no fault
    // and no wrong-looking address -- just a fold of the wrong operands.
    if (S == 0u || S > 8u || (S & (S - 1u)) != 0u || n_rows == 0u || (n_rows % S) != 0u) {
        printf_safe("[Cluster %d Core %d]: Error! pack_fa_partial slots=%u must be a "
                    "power of two in 1..8 and must divide n_rows=%u\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), S, n_rows);
        return BINGO_RET_FAIL;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
    const uint32_t LANES = 16u;              // FP32 lanes in one 512b beat
    for (uint32_t beat = 0; beat < n_rows / S; beat++) {
        uint32_t *out = dst + beat * LANES;
        for (uint32_t j = 0; j < S; j++) {
            uint32_t row = beat * S + j;
            out[j]     = bingo_f16_to_f32_bits(src_m[row]);   // field 0: the key
            out[S + j] = bingo_f16_to_f32_bits(src_l[row]);   // field 1: the exp twist
        }
        // Lanes past 2*S are never read at this geometry, but a gather folds whatever
        // is in the beat if someone later widens S. Zero is the additive identity and
        // -inf would be the max identity, so leave no junk to inherit.
        for (uint32_t k = 2u * S; k < LANES; k++) {
            out[k] = 0u;
        }
    }
    __asm__ volatile("" ::: "memory");
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
    sp->return_value = (uint32_t)(uintptr_t)dst;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// Bound for the post-finish settle spin below. Large enough that a real transfer's
// last store always lands inside it, small enough that a dead one reports in well
// under a microsecond of simulated time instead of hanging the run.
#define BINGO_XDMA_GATHER_SETTLE_SPINS 4096u

#ifdef XDMA_DST_JCT_ENABLE_PTR
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_chain_gather(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_chain_gather_args_t);
    if (!snax_is_xdma_core()) {
        printf_safe("[Cluster %d Core %d]: Error! xDMA chain_gather must run on the "
                    "xDMA core!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    __snax_bingo_kernel_xdma_chain_gather_args_t *a =
        (__snax_bingo_kernel_xdma_chain_gather_args_t *)arg;
    uint64_t local_src = make_u64(a->local_src_hi, a->local_src_lo);
    uint32_t chain_num = a->chain_num;
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_xdma_chain_gather_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    if (chain_num < 2 || chain_num > BINGO_XDMA_CHAIN_MAX ||
        chain_num > XDMA_MAX_DST_COUNT) {
        printf_safe("[Cluster %d Core %d]: Error! xDMA chain_gather chain_num=%u must be "
                    "2..%u (arg struct) and <= %u (HW)\r\n", snrt_cluster_idx(),
                    snrt_cluster_core_idx(), chain_num,
                    (unsigned)BINGO_XDMA_CHAIN_MAX, (unsigned)XDMA_MAX_DST_COUNT);
        return BINGO_RET_FAIL;
    }

    uint64_t chain[BINGO_XDMA_CHAIN_MAX];
    for (uint32_t i = 0; i < chain_num; i++) {
        chain[i] = make_u64(a->chain_hi[i], a->chain_lo[i]);
    }
    // The collector's own buffer is the last hop, and the settle spin below needs a
    // value it can watch change.
    uint32_t dst_addr = (uint32_t)chain[chain_num - 1];

    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
    // (1) A STALE JUNCTION IS NOT INERT. Junctions live in their own CSR bank and
    // survive the previous task; one left armed re-folds this transfer with whatever
    // it was pointed at. Clear extensions AND junctions before arming.
    xdma_disable_all_extensions();
    for (uint8_t j = 0; j < XDMA_DST_JCT_NUM; j++) xdma_disable_dst_junction(j);

    // (2) SENTINEL, so "the writer never wrote" is distinguishable from "it wrote the
    // wrong value", and so the settle spin below has something to observe.
    *(volatile uint32_t *)(uintptr_t)dst_addr = 0xDEADBEEFu;
    __asm__ volatile("fence" ::: "memory");

    BINGO_XDMA_TRY(xdma_chain_gather_1d_full_address(local_src, chain, chain_num,
                                                     a->size, (uint8_t)a->junction,
                                                     a->jct_csr0),
                   "xdma_chain_gather");
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);

    // (3) A SPURIOUS FINISH READS AS SUCCESS. If a finish left standing by the previous
    // gather is already at or past the id this transfer is about to get, xdma_wait_task
    // returns immediately: the task "completes" in a few tens of cycles and nothing moved.
    // The finish counter must lag the commit counter at arming time, so check it here
    // rather than debugging the silence afterwards.
    uint32_t finish_before = snax_read_xdma_cfg_reg(XDMA_FINISH_REMOTE_TASK_PTR);
    uint32_t commit_before = snax_read_xdma_cfg_reg(XDMA_COMMIT_REMOTE_TASK_PTR);
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
    xdma_task_t task = xdma_start();
    if (task.remote && finish_before > commit_before) {
        printf_safe("[Cluster %d Core %d]: Error! xDMA chain_gather saw a standing "
                    "finish (remote finish=%u > commit=%u) before task %u; the gather "
                    "would return without moving data\r\n", snrt_cluster_idx(),
                    snrt_cluster_core_idx(), finish_before, commit_before, task.task_id);
        return BINGO_RET_FAIL;
    }
    xdma_wait_task(task);

    // (4) THE FINISH COUNTER IS NOT A DATA BARRIER. It can bump a few cycles before the
    // writer's last store lands in TCDM, so a consumer scheduled right behind this node
    // reads the sentinel. A cold gather is slow enough to hide it; a warm one loses the
    // race every time. Bounded, so a genuinely dead transfer still returns and reports
    // rather than hanging the simulation.
    {
        volatile uint32_t *settle = (volatile uint32_t *)(uintptr_t)dst_addr;
        uint32_t s = 0;
        for (; s < BINGO_XDMA_GATHER_SETTLE_SPINS; s++) {
            if (settle[0] != 0xDEADBEEFu) break;
            __asm__ volatile("fence" ::: "memory");
        }
        __asm__ volatile("fence" ::: "memory");
        if (s == BINGO_XDMA_GATHER_SETTLE_SPINS) {
            printf_safe("[Cluster %d Core %d]: Error! xDMA chain_gather destination "
                        "0x%x still holds the sentinel after the transfer finished; "
                        "no folded beat arrived\r\n", snrt_cluster_idx(),
                        snrt_cluster_core_idx(), dst_addr);
            xdma_disable_dst_junction((uint8_t)a->junction);
            bingo_xdma_disarm_multicast_slots(0);   // 0 = sweep all; see below
            return BINGO_RET_FAIL;
        }
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);

    xdma_disable_dst_junction((uint8_t)a->junction);
    // The gather arms the SAME destination slots the multicast does (the library calls it
    // "the mirror of multicast" and routes it through xdma_multicast_1d_full_address), so
    // it owes the same disarm. It has never been caught by this only because it is the
    // last xDMA task in its workload -- an accident of scheduling, not a property.
    //
    // SWEEPS ALL 15 ON PURPOSE. The multicast kernel passes its own dst_num because it
    // knows it; how many slots the gather's chain arms is a property of the chain
    // geometry rather than of anything in scope here, and guessing it low would leave a
    // slot armed -- the exact bug this function exists to prevent. Being the last xDMA
    // task in its workload, its teardown is not on anyone's critical path.
    bingo_xdma_disarm_multicast_slots(0);
    sp->return_value = dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}
#else
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_xdma_chain_gather(void *arg)
{
    // Refuse at CALL time, not compile time: the junctions are an optional part of the
    // xDMA in the cluster cfg, and a #error here would stop every build on a cfg that
    // simply does not have them.
    (void)arg;
    printf_safe("[Cluster %d Core %d]: Error! xDMA chain_gather needs a writer junction "
                "(HasElementwiseJunction / HasMonoidJunction) and this cfg has none\r\n",
                snrt_cluster_idx(), snrt_cluster_core_idx());
    return BINGO_RET_FAIL;
}
#endif  // XDMA_DST_JCT_ENABLE_PTR

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
    if (snax_is_xdma_core()) {
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
        printf_safe("[Cluster %d Core %d]: Error! xDMA elementwise_add must run on the xDMA core!\r\n",
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
    if (snax_is_xdma_core()) {
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
        printf_safe("[Cluster %d Core %d]: Error! xDMA elementwise_add_ab must run on the xDMA core!\r\n",
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
// xdma_cpu_transpose_2d moved to snax_xdma_lib.h (runtime/snax/xdma); it now returns the
// lib's native int32_t 0/-1. Callers here test only `!= BINGO_RET_SUCC` (== 0), so the
// -1 error code is transparent.

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

    if (snax_is_xdma_core())
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
        // local dst this is a zero-copy passthrough (the fused path above).
        // Cost: one L1 scratch (M*N*elem_bytes) + an L1->dst 1D DMA.
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

    if (snax_is_xdma_core())
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

    if (snax_is_xdma_core())
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

    if (snax_is_xdma_core())
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

    if (snax_is_xdma_core())
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

    if (snax_is_xdma_core())
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
//   HW path: defined(WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16) && tileSize %8==0
//            && meshCol %8==0 && elem_bytes == 1
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
//   e2  1 (1, 16, 32)    | _M1K16  HW p4/p5 (3)             | _M1N32  HW p3                    | _K16N32 CPU (4)
//   e2  2 (16, 8, 16)    | _M16K8  HW p2                    | _M16N16 HW p2                    | _K8N16  CPU (4)
//   e4  0 (32, 2, 32)    | _M32K2  HW p2                    | _M32N32 HW p2                    | _K2N32  CPU (2)
//   e4  1 (1, 16, 32)    | _M1K16  HW p3                    | _M1N32  HW p3                    | _K16N32 CPU (4)
//   e4  2 (16, 8, 16)    | _M16K8  HW p2                    | _M16N16 HW p2                    | _K8N16  CPU (4)
//
//   (1) A-tile inner row = tileSize*elem_bytes = 2 (e1) or 4 (e2) < 8 bytes; cannot form
//       an 8-byte beat contiguous in both row-major src and packed A dst. CPU.
//   (2) Same root cause: tileSize=2 prevents 8x8-byte sub-blocking of the B-tile, for
//       every elem_bytes -- the HW Transposer needs tileSize%8 and meshCol%8.
//   (3) meshRow=1 with a 16-byte inner row misses paths 1-3, so it needs K_T%8 (p4) or
//       M_T%8 (p5), else CPU. These are the ONLY kernels whose path still depends on a
//       runtime tile count; everything else is decided at compile time.
//   (4) The B↔R HW path drives the 8x8 BYTE-granular Transposer, whose block is an 8x8
//       ELEMENT block only at elem_bytes=1; wider elements take the CPU loop.
//   CPU paths run on the DM core, which is not coherent with L3, so they stage
//   non-local (L3) operands through L1 via xdma_cpu_stage_* (snax_xdma_lib.h).
// ==========================================================================

// xdma_layout_run moved to snax_xdma_lib.h (runtime/snax/xdma).

// D-layout → row-major. See section banner for the path table.
//   Coverage: paths 1–5; which one each kernel takes is in that table.
//   Arg layout: src_hi/lo, dst_hi/lo, M_T, N_T, meshRow, meshCol, elem_bytes.
//   Strides used by all paths (factored out for readability):
//     row_bytes_dst       = N_T * meshCol * elem_bytes   (full row of row-major R)
//     tile_bytes_dst_skip = meshRow * row_bytes_dst (advance past meshRow rows)
//     tile_bytes_src      = meshRow * meshCol * elem_bytes (one D-tile worth of bytes)
//     row_bytes_src       = meshCol * elem_bytes         (one row inside a D-tile)
static inline uint32_t __xdma_d_to_row_major_impl(void *arg, uint32_t meshRow, uint32_t meshCol, uint32_t elem_bytes)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_d_to_row_major_args_t);
    if (!snax_is_xdma_core()) {
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
        // The CPU fallback is a real cost the model must price, so it emits the same
        // XDMA_RUN markers as the HW path (xdma_layout_run). The sweep pairs events to
        // configs POSITIONALLY: a config that emitted no event would shift every later
        // measurement onto the wrong config.
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        for (uint32_t m = 0; m < M_T; m++)
        for (uint32_t n = 0; n < N_T; n++)
        for (uint32_t r = 0; r < meshRow; r++)
        for (uint32_t c = 0; c < meshCol; c++) {
            uint32_t src_off = (((m * N_T + n) * meshRow + r) * meshCol + c) * elem_bytes;
            uint32_t dst_off = ((m * meshRow + r) * N_cols + n * meshCol + c) * elem_bytes;
            for (uint32_t b = 0; b < elem_bytes; b++) dst[dst_off + b] = src[src_off + b];
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
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
    if (!snax_is_xdma_core()) {
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
        // The CPU fallback is a real cost the model must price, so it emits the same
        // XDMA_RUN markers as the HW path (xdma_layout_run). The sweep pairs events to
        // configs POSITIONALLY: a config that emitted no event would shift every later
        // measurement onto the wrong config.
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        for (uint32_t m = 0; m < M_T; m++)
        for (uint32_t k = 0; k < K_T; k++)
        for (uint32_t r = 0; r < meshRow; r++)
        for (uint32_t s = 0; s < tileSize; s++) {
            uint32_t src_off = ((m * meshRow + r) * K_cols + k * tileSize + s) * elem_bytes;
            uint32_t dst_off = (((m * K_T + k) * meshRow + r) * tileSize + s) * elem_bytes;
            for (uint32_t b = 0; b < elem_bytes; b++) dst[dst_off + b] = src[src_off + b];
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
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
//   Coverage: HW path when defined(WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16) && tileSize%8==0
//             && meshCol%8==0 && elem_bytes==1 (see the gate below); CPU otherwise.
//   The CPU path stages non-local (L3) operands through L1 (xdma_cpu_stage_*).
//   Arg layout: src_hi/lo, dst_hi/lo, K_T, N_T, tileSize, meshCol, elem_bytes.
static inline uint32_t __xdma_row_major_to_b_impl(void *arg, uint32_t tileSize, uint32_t meshCol, uint32_t elem_bytes)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_row_major_to_b_args_t);
    if (!snax_is_xdma_core()) {
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
    // The HW transposer path requires elem_bytes == 1.
    //
    // WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16 is an 8x8 BYTE-granular block transposer, so
    // one 8x8 byte block is one 8x8 ELEMENT block only when an element is a single byte. The
    // stride scheme below is built on that identity; at 2- or 4-byte elements it does not
    // compose and the transposed tile is wrong. Wider elements therefore take the CPU loop,
    // which is slower but correct at any width. A width can join the HW path once the
    // transposer has a native mode for it.
    if ((tileSize % 8) == 0 && (meshCol % 8) == 0 && elem_bytes == 1) {
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
        // The CPU fallback is a real cost the model must price, so it emits the same
        // XDMA_RUN markers as the HW path (xdma_layout_run). The sweep pairs events to
        // configs POSITIONALLY: a config that emitted no event would shift every later
        // measurement onto the wrong config.
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        for (uint32_t n = 0; n < N_T; n++)
        for (uint32_t k = 0; k < K_T; k++)
        for (uint32_t c = 0; c < meshCol; c++)
        for (uint32_t s = 0; s < tileSize; s++) {
            uint32_t src_off = ((k * tileSize + s) * N_cols + n * meshCol + c) * elem_bytes;
            uint32_t dst_off = (((n * K_T + k) * meshCol + c) * tileSize + s) * elem_bytes;
            for (uint32_t b = 0; b < elem_bytes; b++) dst[dst_off + b] = src[src_off + b];
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
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
    if (!snax_is_xdma_core()) {
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
        // The CPU fallback is a real cost the model must price, so it emits the same
        // XDMA_RUN markers as the HW path (xdma_layout_run). The sweep pairs events to
        // configs POSITIONALLY: a config that emitted no event would shift every later
        // measurement onto the wrong config.
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        for (uint32_t m = 0; m < M_T; m++)
        for (uint32_t k = 0; k < K_T; k++)
        for (uint32_t r = 0; r < meshRow; r++)
        for (uint32_t s = 0; s < tileSize; s++) {
            uint32_t src_off = (((m * K_T + k) * meshRow + r) * tileSize + s) * elem_bytes;
            uint32_t dst_off = ((m * meshRow + r) * K_cols + k * tileSize + s) * elem_bytes;
            for (uint32_t b = 0; b < elem_bytes; b++) dst[dst_off + b] = src[src_off + b];
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
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
//   Coverage: same as the forward kernel — HW only at elem_bytes==1 with
//             tileSize%8==0 && meshCol%8==0; CPU otherwise.
//   Arg layout: src_hi/lo, dst_hi/lo, K_T, N_T, tileSize, meshCol, elem_bytes.
static inline uint32_t __xdma_b_to_row_major_impl(void *arg, uint32_t tileSize, uint32_t meshCol, uint32_t elem_bytes)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_b_to_row_major_args_t);
    if (!snax_is_xdma_core()) {
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
    // The HW transposer path requires elem_bytes == 1.
    //
    // WRITER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16 is an 8x8 BYTE-granular block transposer, so
    // one 8x8 byte block is one 8x8 ELEMENT block only when an element is a single byte. The
    // stride scheme below is built on that identity; at 2- or 4-byte elements it does not
    // compose and the transposed tile is wrong. Wider elements therefore take the CPU loop,
    // which is slower but correct at any width. A width can join the HW path once the
    // transposer has a native mode for it.
    if ((tileSize % 8) == 0 && (meshCol % 8) == 0 && elem_bytes == 1) {
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
        // The CPU fallback is a real cost the model must price, so it emits the same
        // XDMA_RUN markers as the HW path (xdma_layout_run). The sweep pairs events to
        // configs POSITIONALLY: a config that emitted no event would shift every later
        // measurement onto the wrong config.
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        for (uint32_t n = 0; n < N_T; n++)
        for (uint32_t k = 0; k < K_T; k++)
        for (uint32_t c = 0; c < meshCol; c++)
        for (uint32_t s = 0; s < tileSize; s++) {
            uint32_t src_off = (((n * K_T + k) * meshCol + c) * tileSize + s) * elem_bytes;
            uint32_t dst_off = ((k * tileSize + s) * N_cols + n * meshCol + c) * elem_bytes;
            for (uint32_t b = 0; b < elem_bytes; b++) dst[dst_off + b] = src[src_off + b];
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
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
//   Coverage: paths 1–5; which one each kernel takes is in the section-banner table.
//   Arg layout: src_hi/lo, dst_hi/lo, M_T, N_T, meshRow, meshCol, elem_bytes.
static inline uint32_t __xdma_row_major_to_d_impl(void *arg, uint32_t meshRow, uint32_t meshCol, uint32_t elem_bytes)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_xdma_row_major_to_d_args_t);
    if (!snax_is_xdma_core()) {
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
        // The CPU fallback is a real cost the model must price, so it emits the same
        // XDMA_RUN markers as the HW path (xdma_layout_run). The sweep pairs events to
        // configs POSITIONALLY: a config that emitted no event would shift every later
        // measurement onto the wrong config.
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        for (uint32_t m = 0; m < M_T; m++)
        for (uint32_t n = 0; n < N_T; n++)
        for (uint32_t r = 0; r < meshRow; r++)
        for (uint32_t c = 0; c < meshCol; c++) {
            uint32_t src_off = ((m * meshRow + r) * N_cols + n * meshCol + c) * elem_bytes;
            uint32_t dst_off = (((m * N_T + n) * meshRow + r) * meshCol + c) * elem_bytes;
            for (uint32_t b = 0; b < elem_bytes; b++) dst[dst_off + b] = src[src_off + b];
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);
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
