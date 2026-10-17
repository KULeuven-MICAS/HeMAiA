// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// A mixture of experts over a prompt of T tokens (a prefill), its experts placed on the chips
// ahead of time and its routing computed at run time. The graph is static: each expert's
// rows (its tokens, padded to whole 8-row tiles) are fixed when the workload is built; which
// token sits in which row, and with which weight, is data these kernels read.
//
//   moe_topk_rows        the top k of every token's probabilities -> the ROUTE TABLE, one
//                        32-B entry a token: k expert ids (uint8, 0xFF past k) at bytes 0..7,
//                        their FP16 weights (0 past k) at bytes 8..23, bytes 24..31 zero
//   moe_dispatch         the route table -> per expert slot of this chip its ROW LIST: the
//                        tokens routed to it in token order and their weights, padded with a
//                        zero row (weight 0) to the slot's capacity
//   idma_gather_rows     the listed rows of a matrix (X in L3) into a row-major L1 tile
//   simd_mul_tiles       a D-layout matrix times a one-row-tile scale tile, row tile by row
//                        tile (the per-column dequantisation), in place or row-major out
//   simd_deq_swiglu      a gate|up GEMM output dequantised, then silu(gate) * up, row tile by
//                        row tile, into a D-layout A16
//   simd_quant_rows      A16 rows times each row's route weight, quantised to INT8 rows of a
//                        row-major GEMM A operand
//   simd_scatter_add_rows
//                        an expert's output rows added into the token rows they belong to,
//                        optionally dequantised first (simd_mul_tiles' pass, in place)
//
// D LAYOUT (array shape 3, FP16): blocks of 8 rows x 16 columns, 256 B, row r of a block at
// r * 32 B; blocks in (row tile, column tile) order, so row i of a matrix of NT column tiles
// is NT beats, 256 B apart, from (i / 8) * NT * 256 + (i % 8) * 32. Every SIMD task here is a
// sequence of passes over such rows or row tiles, each one programmed and run in turn.

#pragma once

#include "../macros.h"

#define BINGO_MOE_PF_ENTRY_BYTES 32u
#define BINGO_MOE_PF_W_OFFSET 8u
#define BINGO_MOE_PF_MAX_N 64u
#define BINGO_MOE_PF_MAX_K 8u
#define BINGO_MOE_PF_MAX_SLOTS 4u
#define BINGO_MOE_PF_D_BLOCK 256u    // one D-layout block: 8 rows x 16 FP16

// ============================================================== route

// The k best keys, largest first, after kk (larger than the last one) takes the last one's
// place: kk moves up past every key it beats. Returns the new last (smallest) key.
static inline uint32_t moe_pf_insert(uint32_t *key, uint32_t k, uint32_t kk) {
    uint32_t j = k - 1u;
    while (j > 0u && key[j - 1u] < kk) {
        key[j] = key[j - 1u];
        j--;
    }
    key[j] = kk;
    return key[k - 1u];
}

// The key of value i of a row: value << 8 | (255 - i).
#define MOE_PF_KEY(word, i) ((((word) >> (16u * ((i) & 1u))) & 0xFFFFu) << 8 | (255u - (i)))

// Key kk into the six best keys k0 >= ... >= k5 when it beats k5: it moves up past every
// key it beats, the keys below it each move down one.
#define MOE_PF_INS6(kk)                                                                    \
    do {                                                                                   \
        const uint32_t x_ = (kk);                                                          \
        if (x_ > k5) {                                                                     \
            if (x_ <= k4) { k5 = x_; }                                                     \
            else { k5 = k4;                                                                \
            if (x_ <= k3) { k4 = x_; }                                                     \
            else { k4 = k3;                                                                \
            if (x_ <= k2) { k3 = x_; }                                                     \
            else { k3 = k2;                                                                \
            if (x_ <= k1) { k2 = x_; }                                                     \
            else { k2 = k1;                                                                \
            if (x_ <= k0) { k1 = x_; }                                                     \
            else { k1 = k0; k0 = x_; } } } } }                                             \
        }                                                                                  \
    } while (0)

// One row's top 6 (the DeepSeek-V2 router's k), its keys in registers: they start at 0,
// below every real key (whose index byte is at least 255 - 63), so the first six values go
// in through the same insertion as the rest. The entry is written as eight words.
static inline void moe_pf_top6_row(const uint32_t *pw, uint32_t nw, volatile uint32_t *o) {
    const uint32_t M = 0x80008000u;
    uint32_t k0 = 0u, k1 = 0u, k2 = 0u, k3 = 0u, k4 = 0u, k5 = 0u;
    uint32_t tau2 = 0u, ib = 255u;                      // ib = 255 - the next value's index
    uint32_t w = 0u;
    for (; w + 2u <= nw; w += 2u, ib -= 4u) {
        const uint32_t w0 = pw[w], w1 = pw[w + 1u];
        if (((((w0 | M) - tau2) | ((w1 | M) - tau2)) & M) == 0u) continue;
        MOE_PF_INS6(((w0 << 16) >> 8) | ib);
        MOE_PF_INS6(((w0 >> 16) << 8) | (ib - 1u));
        MOE_PF_INS6(((w1 << 16) >> 8) | (ib - 2u));
        MOE_PF_INS6(((w1 >> 16) << 8) | (ib - 3u));
        tau2 = (k5 >> 8) | ((k5 >> 8) << 16);
    }
    if (w < nw) {                                       // an odd word left
        const uint32_t w0 = pw[w];
        MOE_PF_INS6(((w0 << 16) >> 8) | ib);
        MOE_PF_INS6(((w0 >> 16) << 8) | (ib - 1u));
    }
    // id = 255 - (key & 0xFF) = 0xFF ^ (key & 0xFF); ids 6 and 7 are 0xFF
    o[0] = 0xFFFFFFFFu ^ ((k0 & 0xFFu) | (k1 & 0xFFu) << 8 | (k2 & 0xFFu) << 16 |
                          (k3 & 0xFFu) << 24);
    o[1] = 0xFFFFFFFFu ^ ((k4 & 0xFFu) | (k5 & 0xFFu) << 8);
    o[2] = (k0 >> 8) | (k1 >> 8) << 16;
    o[3] = (k2 >> 8) | (k3 >> 8) << 16;
    o[4] = (k4 >> 8) | (k5 >> 8) << 16;
    o[5] = 0u;
    o[6] = 0u;
    o[7] = 0u;
}
#undef MOE_PF_INS6

// Tokens [t0, t1) of moe_topk_rows: see the kernel.
static void moe_pf_topk_range(uint32_t p_base, uint32_t p_pitch, uint32_t k, uint32_t nw,
                              uint32_t out_base, uint32_t t0, uint32_t t1) {
    const uint32_t M = 0x80008000u;
    for (uint32_t t = t0; t < t1; t++) {
        const uint32_t *pw = (const uint32_t *)(p_base + t * p_pitch);
        if (k == 6u) {
            moe_pf_top6_row(pw, nw, (volatile uint32_t *)(out_base +
                                                          t * BINGO_MOE_PF_ENTRY_BYTES));
            continue;
        }
        uint32_t key[BINGO_MOE_PF_MAX_K];
        for (uint32_t i = 0; i < k; i++) {             // the first k, sorted
            const uint32_t kk = MOE_PF_KEY(pw[i >> 1], i);
            uint32_t j = i;
            while (j > 0u && key[j - 1u] < kk) {
                key[j] = key[j - 1u];
                j--;
            }
            key[j] = kk;
        }
        uint32_t last = key[k - 1u];
        uint32_t w = k >> 1;
        if (k & 1u) {                                   // the rest of the word value k - 1 is in
            const uint32_t kk = MOE_PF_KEY(pw[w], k);
            if (kk > last) last = moe_pf_insert(key, k, kk);
            w++;
        }
        uint32_t tau2 = (last >> 8) | ((last >> 8) << 16);
        for (; w + 2u <= nw; w += 2u) {
            const uint32_t w0 = pw[w], w1 = pw[w + 1u];
            if (((((w0 | M) - tau2) | ((w1 | M) - tau2)) & M) == 0u) continue;
            const uint32_t i = 2u * w;
            uint32_t kk = MOE_PF_KEY(w0, i);
            if (kk > last) last = moe_pf_insert(key, k, kk);
            kk = MOE_PF_KEY(w0, i + 1u);
            if (kk > last) last = moe_pf_insert(key, k, kk);
            kk = MOE_PF_KEY(w1, i + 2u);
            if (kk > last) last = moe_pf_insert(key, k, kk);
            kk = MOE_PF_KEY(w1, i + 3u);
            if (kk > last) last = moe_pf_insert(key, k, kk);
            tau2 = (last >> 8) | ((last >> 8) << 16);
        }
        if (w < nw) {                                   // an odd word left
            const uint32_t w0 = pw[w], i = 2u * w;
            uint32_t kk = MOE_PF_KEY(w0, i);
            if (kk > last) last = moe_pf_insert(key, k, kk);
            kk = MOE_PF_KEY(w0, i + 1u);
            if (kk > last) last = moe_pf_insert(key, k, kk);
        }
        // id j = 255 - (key & 0xFF) = 0xFF ^ (key & 0xFF): XOR into a 0xFF byte
        uint32_t id0 = 0xFFFFFFFFu, id1 = 0xFFFFFFFFu;
        uint32_t wt[BINGO_MOE_PF_MAX_K / 2u] = {0u, 0u, 0u, 0u};
        for (uint32_t j = 0; j < k; j++) {
            const uint32_t x = (key[j] & 0xFFu) << (8u * (j & 3u));
            if (j < 4u) id0 ^= x; else id1 ^= x;
            wt[j >> 1] |= (key[j] >> 8) << (16u * (j & 1u));
        }
        volatile uint32_t *o = (volatile uint32_t *)(out_base + t * BINGO_MOE_PF_ENTRY_BYTES);
        o[0] = id0;
        o[1] = id1;
        o[2] = wt[0];
        o[3] = wt[1];
        o[4] = wt[2];
        o[5] = wt[3];
        o[6] = 0u;
        o[7] = 0u;
    }
}

// The top k of `n` FP16 probabilities of each token t in [t0, t1) at p_addr + t * p_pitch:
// largest first, ties to the lower index (hwmodel.top_k16). The probabilities are a
// softmax's, never negative, so their bit patterns order as unsigned integers below 0x8000.
//
// Each candidate is a KEY, value << 8 | (255 - index): one unsigned compare orders two
// candidates by value and, on equal values, puts the lower index first. The first k keys are
// sorted, largest first; a later key goes in when it beats the last (smallest) one. Two words,
// four values, are first tested against the last key's value tau at once: bit 15 of each half
// of (word | 0x8000_8000) - (tau | tau << 16) says value >= tau (the halves cannot borrow from
// each other, as every value is below 0x8000), which skips most of the row.
//
// The entry is assembled in registers and written as eight words: ids 0xFF past k, weights 0
// past k, bytes 24..31 zero. k = 6 runs moe_pf_top6_row.
//
// ctr_addr != 0: several tasks share [t0, t1) through the word at ctr_addr (zero before the
// first of them runs), each taking `chunk` tokens at a time until the range is done.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_topk_rows(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_moe_topk_rows_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_moe_topk_rows_args_t *a =
        (const __snax_bingo_kernel_moe_topk_rows_args_t *)arg;
    const uint32_t n = a->n, k = a->k, t0 = a->t0, t1 = a->t1;
    const uint32_t p_base = a->p_addr, p_pitch = a->p_pitch, out_base = a->out_addr;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_moe_topk_rows_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    if (n == 0u || n > BINGO_MOE_PF_MAX_N || (n % 2u) != 0u || k == 0u ||
        k > BINGO_MOE_PF_MAX_K || k > n || t1 < t0 || (p_base % 4u) != 0u ||
        (p_pitch % 4u) != 0u || (out_base % 4u) != 0u) {
        printf_safe("[Cluster %d Core %d]: Error! moe_topk_rows n=%d k=%d tokens [%d, %d) "
                    "(n even, rows and table word-aligned)\r\n", snrt_cluster_idx(),
                    snrt_cluster_core_idx(), (int)n, (int)k, (int)t0, (int)t1);
        return BINGO_RET_FAIL;
    }
    const uint32_t nw = n / 2u;
    BINGO_TRACE_MARKER(BINGO_TRACE_SCALAR_RUN_START);
    if (a->ctr_addr != 0u) {
        // Shared work: every task of the range takes `chunk` tokens at a time from the counter
        // (zero before the first task), so the cores that reach the range early do most of it.
        uint32_t *ctr = (uint32_t *)(uintptr_t)a->ctr_addr;
        const uint32_t ch = a->chunk != 0u ? a->chunk : 1u;
        for (;;) {
            const uint32_t s0 = t0 + __atomic_fetch_add(ctr, ch, __ATOMIC_RELAXED);
            if (s0 >= t1) break;
            moe_pf_topk_range(p_base, p_pitch, k, nw, out_base, s0, s0 + ch < t1 ? s0 + ch : t1);
        }
    } else {
        moe_pf_topk_range(p_base, p_pitch, k, nw, out_base, t0, t1);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_SCALAR_RUN_END);
    sp->return_value = out_base;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// The route table -> the row lists of this chip's expert slots. Slot s (expert ids byte s)
// takes 4 * cap[s] bytes from list_addr, slots back to back: cap[s] uint16 token rows, then
// cap[s] uint16 FP16 weights. A row past the slot's tokens is pad_row with weight 0. Every
// slot must get exactly cnt[s] tokens (the graph's shapes were fixed from the same route);
// another count is an error, reported and written to err_addr.
//
// An expert id selects its slot through a 256-entry table (0xFF: not on this chip); a token's
// ids are read as two words and looked up byte by byte, the bytes past k too: the table
// holds 0xFF there, which no slot's expert has.
//
// state_addr != 0: the task does tokens [t0, t1) only and keeps the slots' counts in the four
// words at state_addr (this cluster's L1) for the next task of the same lists: the task with
// t0 = 0 fills the lists with padding and starts the counts at zero, the one with t1 = tokens
// checks them. The tasks of one set of lists run in token order (one core).
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dispatch(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_moe_dispatch_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_moe_dispatch_args_t *a =
        (const __snax_bingo_kernel_moe_dispatch_args_t *)arg;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_moe_dispatch_args_t);
    const uint32_t ns = a->n_slots, k = a->k, tokens = a->tokens, table = a->table_addr;
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    if (ns == 0u || ns > BINGO_MOE_PF_MAX_SLOTS || k == 0u || k > BINGO_MOE_PF_MAX_K ||
        (table % 4u) != 0u || (a->list_addr % 2u) != 0u) {
        printf_safe("[Cluster %d Core %d]: Error! moe_dispatch slots=%d k=%d (table "
                    "word-aligned)\r\n", snrt_cluster_idx(), snrt_cluster_core_idx(), (int)ns,
                    (int)k);
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_SCALAR_RUN_START);
    uint32_t slot_w[256u / 4u];
    for (uint32_t q = 0; q < 256u / 4u; q++) slot_w[q] = 0xFFFFFFFFu;
    uint8_t *slot_of = (uint8_t *)slot_w;
    volatile uint16_t *rows[BINGO_MOE_PF_MAX_SLOTS];
    volatile uint16_t *ws[BINGO_MOE_PF_MAX_SLOTS];
    uint32_t cap[BINGO_MOE_PF_MAX_SLOTS], got[BINGO_MOE_PF_MAX_SLOTS];
    volatile uint32_t *state = (volatile uint32_t *)(uintptr_t)a->state_addr;
    const uint32_t t_first = state ? a->t0 : 0u, t_end = state ? a->t1 : tokens;
    if (t_first > t_end || t_end > tokens) {
        printf_safe("[Cluster %d Core %d]: Error! moe_dispatch tokens [%d, %d) of %d\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), (int)t_first, (int)t_end,
                    (int)tokens);
        return BINGO_RET_FAIL;
    }
    uint32_t off = 0u;
    for (uint32_t s = 0; s < ns; s++) {
        cap[s] = a->cap[s];
        rows[s] = (volatile uint16_t *)(a->list_addr + off);
        ws[s] = (volatile uint16_t *)(a->list_addr + off + 2u * cap[s]);
        off += 4u * cap[s];
        const uint32_t eid = (a->expert_ids >> (8u * s)) & 0xFFu;
        if (eid == 0xFFu) {
            printf_safe("[Cluster %d Core %d]: Error! moe_dispatch slot %d has expert id 0xFF, "
                        "the table's empty id\r\n", snrt_cluster_idx(), snrt_cluster_core_idx(),
                        (int)s);
            return BINGO_RET_FAIL;
        }
        slot_of[eid] = (uint8_t)s;
        if (t_first != 0u) {                            // a later range: the lists are started
            got[s] = state[s];
            continue;
        }
        got[s] = 0u;
        for (uint32_t i = 0; i < cap[s]; i++) {
            rows[s][i] = (uint16_t)a->pad_row;
            ws[s][i] = 0u;
        }
    }
#define MOE_PF_DISP(word, j)                                                              \
    do {                                                                                  \
        const uint32_t s_ = slot_of[((word) >> (8u * ((j) & 3u))) & 0xFFu];               \
        if (s_ != 0xFFu) {                                                                \
            const uint32_t g_ = got[s_]++;                                                \
            if (g_ < cap[s_]) {                                                           \
                rows[s_][g_] = (uint16_t)t;                                               \
                ws[s_][g_] = w[j];                                                        \
            }                                                                             \
        }                                                                                 \
    } while (0)
    for (uint32_t t = t_first; t < t_end; t++) {
        const uint32_t *e = (const uint32_t *)(table + t * BINGO_MOE_PF_ENTRY_BYTES);
        const uint32_t id0 = e[0], id1 = e[1];
        const uint16_t *w = (const uint16_t *)(table + t * BINGO_MOE_PF_ENTRY_BYTES +
                                               BINGO_MOE_PF_W_OFFSET);
        MOE_PF_DISP(id0, 0u);
        MOE_PF_DISP(id0, 1u);
        MOE_PF_DISP(id0, 2u);
        MOE_PF_DISP(id0, 3u);
        if (k > 4u) {
            MOE_PF_DISP(id1, 4u);
            MOE_PF_DISP(id1, 5u);
        }
        if (k > 6u) {
            MOE_PF_DISP(id1, 6u);
            MOE_PF_DISP(id1, 7u);
        }
    }
#undef MOE_PF_DISP
    if (state) {
        for (uint32_t s = 0; s < ns; s++) state[s] = got[s];
        if (t_end != tokens) {                          // not the last range: nothing to check yet
            sp->return_value = a->list_addr;
            sp->num_return_values = 0;
            return BINGO_RET_SUCC;
        }
    }
    uint32_t bad = 0u;
    for (uint32_t s = 0; s < ns; s++) {
        if (got[s] != a->cnt[s]) {
            printf_safe("[Cluster %d Core %d]: Error! moe_dispatch slot %d (expert %d): %d "
                        "tokens routed, the graph has %d\r\n", snrt_cluster_idx(),
                        snrt_cluster_core_idx(), (int)s,
                        (int)((a->expert_ids >> (8u * s)) & 0xFFu), (int)got[s],
                        (int)a->cnt[s]);
            bad = 1u;
        }
    }
    if (a->err_addr) *(volatile uint32_t *)a->err_addr = bad;
    BINGO_TRACE_MARKER(BINGO_TRACE_SCALAR_RUN_END);
    sp->return_value = a->list_addr;
    sp->num_return_values = 0;
    return bad ? BINGO_RET_FAIL : BINGO_RET_SUCC;
}

// ============================================================== gather

// Row ids[i] of the matrix at src (row r at src + r * row_bytes, any memory the iDMA reaches)
// into dst + i * dst_pitch, for i < n: one 1-D transfer a row, then one wait.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_idma_gather_rows(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_idma_gather_rows_args_t);
    if (!snrt_is_dm_core()) {
        printf_safe("[Cluster %d Core %d]: Error! idma_gather_rows should be called from a DM "
                    "core!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_idma_gather_rows_args_t *a =
        (const __snax_bingo_kernel_idma_gather_rows_args_t *)arg;
    const uint64_t src = make_u64(a->src_addr_hi, a->src_addr_lo);
    const uint64_t dst = make_u64(a->dst_addr_hi, a->dst_addr_lo);
    const volatile uint16_t *ids = (const volatile uint16_t *)a->ids_addr;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_idma_gather_rows_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
    for (uint32_t i = 0; i < a->n; i++) {
        snrt_dma_start_1d_wideptr(dst + (uint64_t)i * a->dst_pitch,
                                  src + (uint64_t)ids[i] * a->row_bytes, a->row_bytes);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
    snrt_dma_wait_all();
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
    sp->return_value = (uint32_t)dst;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// ============================================================== SIMD passes

// The end of a sequence of queued tasks: wait for every submitted task, then the sticky
// bad-config bit (a task the AGU refused was dropped, not run).
static inline uint32_t moe_pf_drain(const char *kname) {
    if (snax_simd_wait_all_bounded()) {
        printf_safe("[Cluster %d Core %d]: %s SIMD drain timeout (status %08x)\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), kname,
                    snax_read_simd_cfg_reg(SIMD_STATUS));
        return BINGO_RET_FAIL;
    }
    if (snax_simd_bad_config()) {
        printf_safe("[Cluster %d Core %d]: %s SIMD refused a degenerate task\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), kname);
        return BINGO_RET_FAIL;
    }
    return BINGO_RET_SUCC;
}

#if BINGO_HAS_STREAMELEMENTWISE
// Program (not run) one EW1 pass over the pairs (a[j], b[j]), j < beats, beats `beat_stride`
// apart in both operands, written to dst as an out_bound1 x out_bound0 sweep of the given
// strides. The reader starts at the lower operand; moe_pf_ew2_repoint moves the pass.
static inline void moe_pf_ew2_program(void *src_a, void *src_b, uint32_t beats,
                                      uint32_t beat_stride, void *dst, uint32_t out_bound0,
                                      uint32_t out_stride0, uint32_t out_bound1,
                                      uint32_t out_stride1, uint32_t op) {
    uint32_t operand_stride;
    void *base = snax_simd_ew2_base(src_a, src_b, &operand_stride);
    snax_simd_shape_t in, out;
    snax_simd_shape_clear(&in);
    in.base = base;
    in.lane_stride = SIMD_LANE_BYTES;
    in.lane_mask = 0xFFFFFFFFu;
    in.byte_mask = 0xFFFFFFFFu;
    in.dim = 2;
    in.bound[0] = 2u;
    in.stride[0] = operand_stride;
    in.bound[1] = beats;
    in.stride[1] = beat_stride;
    snax_simd_shape_2d(&out, dst, out_bound0, out_stride0, out_bound1, out_stride1);
    snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 2u, op);
    snax_simd_program_fast(&in, &out);
}

// The programmed pass moved to the operands a, b and the output dst: the reader's base (the
// lower operand), its operand stride and the writer's base.
static inline void moe_pf_ew2_repoint(uint32_t a, uint32_t b, uint32_t dst) {
    snax_write_simd_cfg_reg(SIMD_SRC_ADDR_PTR_LSB, a < b ? a : b);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 0u, a < b ? b - a : a - b);
    snax_write_simd_cfg_reg(SIMD_DST_ADDR_PTR_LSB, dst);
}

// One such pass, run to completion.
static inline uint32_t moe_pf_ew2(void *src_a, void *src_b, uint32_t beats,
                                  uint32_t beat_stride, void *dst, uint32_t out_bound0,
                                  uint32_t out_stride0, uint32_t out_bound1,
                                  uint32_t out_stride1, uint32_t op) {
    moe_pf_ew2_program(src_a, src_b, beats, beat_stride, dst, out_bound0, out_stride0,
                       out_bound1, out_stride1, op);
    snax_simd_fire();
    return moe_pf_drain("moe_pf_ew2");
}
#endif

// x (D layout, `tiles` row tiles of `tile_beats` beats, x_tile_stride bytes apart) times the
// one-row-tile scale tile s, elementwise: row tile m into dst + m * dst_tile_stride, or with
// dst_pitch != 0 into a row-major matrix (row 8 m + r at dst + (8 m + r) * dst_pitch, its
// tile_beats / 8 column tiles 32 B apart). dst may be x.
//
// One pass a row tile, all of one geometry: programmed once, then each tile moves the pass
// (moe_pf_ew2_repoint) and is submitted to the SIMD's task queue without waiting; the tiles'
// passes touch disjoint rows.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_mul_tiles(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_mul_tiles_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_mul_tiles", "SIMD");
#if !BINGO_HAS_STREAMELEMENTWISE
    BINGO_SIMD_EXT_UNSUPPORTED("simd_mul_tiles", "StreamElementwise");
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_simd_mul_tiles_args_t *a =
        (const __snax_bingo_kernel_simd_mul_tiles_args_t *)arg;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_simd_mul_tiles_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_SIMD_REQUIRE_LOCAL(a->x_addr, "simd_mul_tiles", "x");
    BINGO_SIMD_REQUIRE_LOCAL(a->s_addr, "simd_mul_tiles", "scale");
    BINGO_SIMD_REQUIRE_LOCAL(a->dst_addr, "simd_mul_tiles", "dst");
    const uint32_t nt = a->tile_beats / 8u;
    if (a->tile_beats == 0u || (a->tile_beats % 8u) != 0u || a->tiles == 0u) {
        printf_safe("[Cluster %d Core %d]: Error! simd_mul_tiles tiles=%d tile_beats=%d\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), (int)a->tiles,
                    (int)a->tile_beats);
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    for (uint32_t m = 0; m < a->tiles; m++) {
        const uint32_t x = a->x_addr + m * a->x_tile_stride;
        const uint32_t d = a->dst_pitch ? a->dst_addr + 8u * m * a->dst_pitch
                                        : a->dst_addr + m * a->dst_tile_stride;
        if (m == 0u && a->dst_pitch) {
            // the reader takes a block's 8 rows in turn, then the next column tile
            moe_pf_ew2_program((void *)x, (void *)a->s_addr, a->tile_beats, SIMD_BEAT_BYTES,
                               (void *)d, 8u, a->dst_pitch, nt, SIMD_BEAT_BYTES, SIMD_EW_MUL);
        } else if (m == 0u) {
            moe_pf_ew2_program((void *)x, (void *)a->s_addr, a->tile_beats, SIMD_BEAT_BYTES,
                               (void *)d, a->tile_beats, SIMD_BEAT_BYTES, 1u, 0u, SIMD_EW_MUL);
        } else {
            moe_pf_ew2_repoint(x, a->s_addr, d);
        }
        snax_simd_fire();
    }
    const uint32_t rc = moe_pf_drain("simd_mul_tiles");
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    if (rc != BINGO_RET_SUCC) return BINGO_RET_FAIL;
    sp->return_value = a->dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

// A gate|up GEMM output g ([tiles][2 nb][8][16] FP16: each row tile's nb gate blocks, then its
// nb up blocks) times its scale tile s (2 nb blocks, one row tile's), in place; then
// silu(gate) * up of every row tile into out + m * out_tile_stride (nb blocks). sg_addr: the
// scratch the silu goes through, as large as g.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_deq_swiglu(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_deq_swiglu_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_deq_swiglu", "SIMD");
#if !BINGO_HAS_STREAMMAP || !BINGO_HAS_STREAMELEMENTWISE
    BINGO_SIMD_EXT_UNSUPPORTED("simd_deq_swiglu", "StreamMap+StreamElementwise");
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_simd_deq_swiglu_args_t *a =
        (const __snax_bingo_kernel_simd_deq_swiglu_args_t *)arg;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_simd_deq_swiglu_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_SIMD_REQUIRE_LOCAL(a->g_addr, "simd_deq_swiglu", "g");
    BINGO_SIMD_REQUIRE_LOCAL(a->s_addr, "simd_deq_swiglu", "scale");
    BINGO_SIMD_REQUIRE_LOCAL(a->sg_addr, "simd_deq_swiglu", "scratch");
    BINGO_SIMD_REQUIRE_LOCAL(a->out_addr, "simd_deq_swiglu", "out");
    const uint32_t half = a->nb * BINGO_MOE_PF_D_BLOCK;      // one row tile's gate (or up)
    const uint32_t tile = 2u * half;
    const uint32_t hb = a->nb * 8u;                          // beats in it
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint32_t rc = BINGO_RET_SUCC;
    // 1) g (.) s, row tile by row tile, in place: programmed once, each tile's pass moved
    // and queued, all drained before the silu reads g
    for (uint32_t m = 0; m < a->tiles; m++) {
        const uint32_t gm = a->g_addr + m * tile;
        if (m == 0u) {
            moe_pf_ew2_program((void *)gm, (void *)a->s_addr, 2u * hb, SIMD_BEAT_BYTES,
                               (void *)gm, 2u * hb, SIMD_BEAT_BYTES, 1u, 0u, SIMD_EW_MUL);
        } else {
            moe_pf_ew2_repoint(gm, a->s_addr, gm);
        }
        snax_simd_fire();
    }
    rc = moe_pf_drain("simd_deq_swiglu");
    // 2) silu over the whole of g (the up halves' silu is never read)
    if (rc == BINGO_RET_SUCC)
        rc = simd_pass_map((void *)a->g_addr, (void *)a->sg_addr, a->tiles * 2u * hb,
                           SIMD_F32_ONE, 0u, SIMD_FUNC_SILU, SIMD_OUT_F16, 0u);
    // 3) silu(gate) (.) up: in every row tile the gate half of sg against the up half of g,
    // the same distance apart in each; out row tile m at out + m * out_tile_stride
    if (rc == BINGO_RET_SUCC) {
        uint32_t operand_stride;
        void *base = snax_simd_ew2_base((void *)a->sg_addr, (void *)(a->g_addr + half),
                                        &operand_stride);
        snax_simd_shape_t in, out;
        snax_simd_shape_clear(&in);
        in.base = base;
        in.lane_stride = SIMD_LANE_BYTES;
        in.lane_mask = 0xFFFFFFFFu;
        in.byte_mask = 0xFFFFFFFFu;
        in.dim = 3;
        in.bound[0] = 2u;
        in.stride[0] = operand_stride;
        in.bound[1] = hb;
        in.stride[1] = SIMD_BEAT_BYTES;
        in.bound[2] = a->tiles;
        in.stride[2] = tile;
        snax_simd_shape_2d(&out, (void *)a->out_addr, hb, SIMD_BEAT_BYTES, a->tiles,
                           a->out_tile_stride);
        snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 2u,
                       SIMD_EW_MUL);
        rc = simd_run_shapes(&in, &out);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    if (rc != BINGO_RET_SUCC) return BINGO_RET_FAIL;
    sp->return_value = a->out_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

// Rows i < rows of a D-layout A16 (row i's nb beats 256 B apart from src + (i / 8) *
// src_tile_stride + (i % 8) * 32), each times its route weight (FP16 w[i]; w_addr 0: 1),
// quantised with the static inv_scale into row i of a row-major INT8 matrix at dst + i *
// dst_pitch. An odd nb reads one beat more and writes 16 values more at the row's end.
//
// One pass a row, all of one geometry: it is programmed once, and each row then sets only
// its two base addresses and its weight and is submitted to the SIMD's task queue (which
// snapshots the whole task), so the core programs row i+1 while row i runs.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_quant_rows(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_quant_rows_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_quant_rows", "SIMD");
#if !BINGO_HAS_STREAMMAP || !BINGO_HAS_FP16TOINT8
    BINGO_SIMD_EXT_UNSUPPORTED("simd_quant_rows", "StreamMap+Fp16ToInt8");
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_simd_quant_rows_args_t *a =
        (const __snax_bingo_kernel_simd_quant_rows_args_t *)arg;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_simd_quant_rows_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_SIMD_REQUIRE_LOCAL(a->src_addr, "simd_quant_rows", "src");
    BINGO_SIMD_REQUIRE_LOCAL(a->dst_addr, "simd_quant_rows", "dst");
    const uint32_t nb2 = (a->nb + 1u) & ~1u;     // the quantiser packs two beats into one
    if (a->nb == 0u || (a->dst_addr % SIMD_BEAT_BYTES) != 0u ||
        (a->dst_pitch % SIMD_BEAT_BYTES) != 0u) {
        printf_safe("[Cluster %d Core %d]: Error! simd_quant_rows nb=%d dst=0x%08x pitch=%d "
                    "(both beat-aligned)\r\n", snrt_cluster_idx(), snrt_cluster_core_idx(),
                    (int)a->nb, (unsigned)a->dst_addr, (int)a->dst_pitch);
        return BINGO_RET_FAIL;
    }
    const volatile uint16_t *w = (const volatile uint16_t *)a->w_addr;
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    if (a->rows == 0u) {
        sp->return_value = a->dst_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    }
    snax_simd_shape_t in, out;
    snax_simd_shape_2d(&in, (void *)a->src_addr, nb2, BINGO_MOE_PF_D_BLOCK, 1u, 0u);
    snax_simd_shape_flat(&out, (void *)a->dst_addr, nb2 / 2u);
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR,
                   a->w_addr ? bingo_f16_to_f32bits(w[0]) : SIMD_F32_ONE, 0u, SIMD_FUNC_LINEAR);
    snax_simd_arm2(SIMD_EXT_FP16TOINT8, SIMD_EXT_FP16TOINT8_CSR, a->inv_scale,
                   SIMD_QUANT_TAIL(0));
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    for (uint32_t i = 1; i < a->rows; i++) {
        snax_write_simd_cfg_reg(SIMD_SRC_ADDR_PTR_LSB,
                                a->src_addr + (i / 8u) * a->src_tile_stride +
                                    (i % 8u) * SIMD_BEAT_BYTES);
        snax_write_simd_cfg_reg(SIMD_DST_ADDR_PTR_LSB, a->dst_addr + i * a->dst_pitch);
        if (a->w_addr) {
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 0u, bingo_f16_to_f32bits(w[i]));
        }
        snax_simd_fire();
    }
    const uint32_t rc = moe_pf_drain("simd_quant_rows");
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    if (rc != BINGO_RET_SUCC) return BINGO_RET_FAIL;
    sp->return_value = a->dst_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}

// y[ids[i]] += d[i] for i < n, FP16, both D layout over nt column tiles: y's row t at y_addr
// + (t / 8) * y_tile_stride + (t % 8) * 32, d's row i likewise from d_addr; nt beats a row,
// 256 B apart. The ids are distinct (one expert's tokens).
//
// One EW1 pass a row over the pairs (y row, d row): programmed once, then each row sets its
// reader base (the lower of the two rows), the operand stride between them and its writer
// base, and is submitted to the task queue without waiting; the rows' passes write disjoint
// rows of y, so they need no order among themselves.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_simd_scatter_add_rows(void *arg) {
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_simd_scatter_add_rows_args_t);
    BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_scatter_add_rows", "SIMD");
#if !BINGO_HAS_STREAMELEMENTWISE
    BINGO_SIMD_EXT_UNSUPPORTED("simd_scatter_add_rows", "StreamElementwise");
#else
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_simd_scatter_add_rows_args_t *a =
        (const __snax_bingo_kernel_simd_scatter_add_rows_args_t *)arg;
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_simd_scatter_add_rows_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_SIMD_REQUIRE_LOCAL(a->y_addr, "simd_scatter_add_rows", "y");
    BINGO_SIMD_REQUIRE_LOCAL(a->d_addr, "simd_scatter_add_rows", "d");
    const volatile uint16_t *ids = (const volatile uint16_t *)a->ids_addr;
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    if (a->n == 0u) {
        sp->return_value = a->y_addr;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    }
    if (a->s_addr) {
        // the dequantisation of the n rows' row tiles, in place, before they are added
        BINGO_SIMD_REQUIRE_LOCAL(a->s_addr, "simd_scatter_add_rows", "scale");
        const uint32_t tiles = (a->n + 7u) / 8u;
        for (uint32_t m = 0; m < tiles; m++) {
            const uint32_t x = a->d_addr + m * a->d_tile_stride;
            if (m == 0u) {
                moe_pf_ew2_program((void *)x, (void *)a->s_addr, 8u * a->nt, SIMD_BEAT_BYTES,
                                   (void *)x, 8u * a->nt, SIMD_BEAT_BYTES, 1u, 0u, SIMD_EW_MUL);
            } else {
                moe_pf_ew2_repoint(x, a->s_addr, x);
            }
            snax_simd_fire();
        }
        if (moe_pf_drain("simd_scatter_add_rows") != BINGO_RET_SUCC) return BINGO_RET_FAIL;
    }
    for (uint32_t i = 0; i < a->n; i++) {
        const uint32_t t = ids[i];
        const uint32_t yr = a->y_addr + (t / 8u) * a->y_tile_stride + (t % 8u) * SIMD_BEAT_BYTES;
        const uint32_t dr = a->d_addr + (i / 8u) * a->d_tile_stride + (i % 8u) * SIMD_BEAT_BYTES;
        if (i == 0u) {
            moe_pf_ew2_program((void *)yr, (void *)dr, a->nt, BINGO_MOE_PF_D_BLOCK, (void *)yr,
                               a->nt, BINGO_MOE_PF_D_BLOCK, 1u, 0u, SIMD_EW_ADD);
        } else {
            moe_pf_ew2_repoint(yr, dr, yr);
        }
        snax_simd_fire();
    }
    const uint32_t rc = moe_pf_drain("simd_scatter_add_rows");
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    if (rc != BINGO_RET_SUCC) return BINGO_RET_FAIL;
    sp->return_value = a->y_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
#endif
}
