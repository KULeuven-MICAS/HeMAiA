// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// The router's decision, as data the rest of the graph can read: the top k experts, and
// for each one an EXPERT-SLOT RECORD naming its weights. A port of snax dsv2_top_k16 plus
// the slot table snax-dsv2-moe.h builds in core code.
//
// WHY A RECORD AND NOT A BRANCH. A slot's GEMVs are the same nodes whatever expert the
// router picks; only the bytes they stream differ. So the graph is static and the choice is
// an ADDRESS: every load of a slot is idma_copy_slot, which reads its source from the
// record at run time. Nothing is skipped and nothing branches on the id.
//
// THE RECORD, k slots of 128 B in this cluster's L1 (other clusters pull a copy):
//
//     word  0      expert id
//     word  1      its routing weight as FP32 bits (the combine's Map scale)
//     word  2      its routing weight, FP16 bits
//     words 3-15   0
//     words 16-31  its entry of the expert table, 64 B, copied verbatim:
//                    16/17 gate|up weights   18/19 their factors   (lo, hi)
//                    20/21 down weights      22/23 their factors
//                    24    the SwiGLU output's inv scale, FP32 bits
//                    25-31 0
//
// A table entry of an expert whose weights were never staged is all zero; the route
// refuses it here, by name, rather than letting a slot stream from address 0.
//
// A PASS (tokens > 1, a speculative pass's tokens): the slots are the union of the tokens'
// top k, token 0's in its order then each later token's new ones (hwmodel.union_order), and
// there is one record per token, back to back: the same ids and table entries, the token's
// own weights -- 0 for an expert it did not pick, so the combine adds it nothing. The graph
// streams every union slot once for all the tokens.

#pragma once

#include "../macros.h"
#include "snax_core_roles.h"  // snax_is_idma_core(), BINGO_REQUIRE_CORE

#define BINGO_MOE_ROUTE_MAX_N 64u
#define BINGO_MOE_ROUTE_MAX_K 8u
#define BINGO_MOE_ROUTE_MAX_T 4u     // tokens of a pass
#define BINGO_MOE_ROUTE_MAX_U 32u    // the union's slots
#define BINGO_MOE_TABLE_ENTRY_BYTES 64u
#define BINGO_MOE_REC_SLOT_BYTES 128u

// FP16 -> FP32 bits, EXACT for every finite value: the combine multiplies by this word, and
// the golden (numpy) widens a subnormal exactly. snax_fp16_math.h's f16_to_f32bits flushes
// subnormals to zero, which is right for its softmax scalars and wrong for a routed weight.
static inline uint32_t bingo_f16_to_f32bits(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    if (e == 0x1Fu) return sign | 0x7F800000u | (m << 13);
    if (e != 0u) return sign | ((e + 112u) << 23) | (m << 13);
    if (m == 0u) return sign;
    e = 113u;                                  // m * 2^-24: normalise the leading one
    while (!(m & 0x400u)) {
        m <<= 1;
        e--;
    }
    return sign | (e << 23) | ((m & 0x3FFu) << 13);
}

// FP16 bit patterns -> integers in value order (snax dsv2_mono16).
static inline int32_t bingo_mono16(uint16_t h) {
    const int32_t mag = (int32_t)(h & 0x7FFFu);
    return (h & 0x8000u) ? -mag : mag;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_route(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_moe_route_args_t);
    BINGO_REQUIRE_CORE(snax_is_idma_core(), "moe_route", "iDMA");
    const __snax_bingo_kernel_moe_route_args_t *a =
        (const __snax_bingo_kernel_moe_route_args_t *)arg;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_moe_route_args_t);
    const uint32_t n = a->n, k = a->k;
    const uint32_t T = a->tokens ? a->tokens : 1u, U = T > 1u ? a->union_n : k;
    if (n == 0u || n > BINGO_MOE_ROUTE_MAX_N || k == 0u || k > BINGO_MOE_ROUTE_MAX_K ||
        k > n || (a->record_addr & 63u) || T > BINGO_MOE_ROUTE_MAX_T || U < k ||
        U > BINGO_MOE_ROUTE_MAX_U) {
        printf_safe("[Cluster %d Core %d]: Error! moe_route: n=%d (<= %d), k=%d (<= %d), "
                    "tokens=%d (<= %d), union %d (<= %d), the record 64-B aligned\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), (int)n,
                    (int)BINGO_MOE_ROUTE_MAX_N, (int)k, (int)BINGO_MOE_ROUTE_MAX_K, (int)T,
                    (int)BINGO_MOE_ROUTE_MAX_T, (int)U, (int)BINGO_MOE_ROUTE_MAX_U);
        return BINGO_RET_FAIL;
    }
    volatile uint32_t *rec = (volatile uint32_t *)a->record_addr;
    const uint64_t table = make_u64(a->table_addr_hi, a->table_addr_lo);
    // Each token's top k, largest first; equal values go to the lower index
    // (hwmodel.top_k16). ONE pass: each probability is read once and inserted into a sorted
    // list of k -- after the entries it equals, so of equal values the lower index stays
    // ahead. The earlier k passes of k volatile loads, key conversions and 64-bit mask tests
    // cost 24.6 us of the layer's critical path (every routed expert waits for this record);
    // this is ~n * k compares.
    uint32_t ids[BINGO_MOE_ROUTE_MAX_T][BINGO_MOE_ROUTE_MAX_K];
    uint16_t w16s[BINGO_MOE_ROUTE_MAX_T][BINGO_MOE_ROUTE_MAX_K];
    for (uint32_t t = 0; t < T; t++) {
        const volatile uint16_t *p = (const volatile uint16_t *)(a->p_addr + t * a->p_pitch);
        int32_t keys[BINGO_MOE_ROUTE_MAX_K];
        uint32_t have = 0;
        // the k-th key in a register (below every key until the list is full), and the key
        // without a branch: the loop over n probabilities dominates this kernel, which sits on
        // the layer's critical path (every routed expert waits for the record)
        int32_t thr = INT32_MIN;
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t h = p[i];
            const int32_t sgn = -(int32_t)(h >> 15);                 // 0, or -1 if negative
            const int32_t key = ((int32_t)(h & 0x7FFFu) ^ sgn) - sgn;   // bingo_mono16
            if (key <= thr) continue;
            uint32_t j = have < k ? have++ : k - 1;      // the slot it enters from
            while (j > 0 && key > keys[j - 1]) {
                keys[j] = keys[j - 1];
                ids[t][j] = ids[t][j - 1];
                j--;
            }
            keys[j] = key;
            ids[t][j] = i;
            if (have == k) thr = keys[k - 1];
        }
        for (uint32_t r = 0; r < k; r++) w16s[t][r] = p[ids[t][r]];
        if (a->ids_addr) {          // candidates folded across chips: their expert ids
            const volatile uint32_t *map = (const volatile uint32_t *)a->ids_addr;
            for (uint32_t r = 0; r < k; r++) ids[t][r] = map[ids[t][r]];
        }
    }
    // The slots: one token's k, or the pass's UNION -- token 0's experts in its order, then
    // each later token's new ones in its order (hwmodel.union_order). The graph was built
    // for exactly U slots; another count is a routing it cannot run.
    uint32_t uid[BINGO_MOE_ROUTE_MAX_U];
    uint32_t nu = 0;
    for (uint32_t t = 0; t < T && nu <= U; t++) {
        for (uint32_t r = 0; r < k && nu <= U; r++) {
            uint32_t seen = 0;
            for (uint32_t q = 0; q < nu && !seen; q++) seen = uid[q] == ids[t][r];
            if (seen) continue;
            if (nu < U) uid[nu] = ids[t][r];
            nu++;                               // U + 1: one too many, and the loops stop
        }
    }
    if (nu != U) {
        printf_safe("[Cluster %d Core %d]: Error! moe_route: the %d tokens' top %d are %s%d "
                    "experts, the graph runs %d slots\r\n", snrt_cluster_idx(),
                    snrt_cluster_core_idx(), (int)T, (int)k, nu > U ? "more than " : "",
                    (int)(nu > U ? U : nu), (int)U);
        return BINGO_RET_FAIL;
    }
    // Token t's record: U slots, slot s = expert uid[s] with token t's weight (0 if it did
    // not pick it). Token 0's slots get their table entries by iDMA straight from the table
    // (U transfers of 64 B); each further token's record takes all U entries from token 0's
    // in ONE 2-D transfer. Measured at 4 tokens, 22 slots: the core copying the entries word
    // by word, 55 us; 88 transfers of 64 B, 45 us (the iDMA's short queue serialises them).
    for (uint32_t s = 0; s < U; s++)
        snrt_dma_start_1d_wideptr(
            chiplet_addr_transform((uint64_t)(uint32_t)(
                rec + s * (BINGO_MOE_REC_SLOT_BYTES / 4u) + 16)),
            table + (uint64_t)uid[s] * BINGO_MOE_TABLE_ENTRY_BYTES, BINGO_MOE_TABLE_ENTRY_BYTES);
    if (T > 1u) {
        snrt_dma_wait_all();
        for (uint32_t t = 1; t < T; t++)
            snrt_dma_start_2d_wideptr(
                chiplet_addr_transform((uint64_t)(uint32_t)(
                    rec + t * U * (BINGO_MOE_REC_SLOT_BYTES / 4u) + 16)),
                chiplet_addr_transform((uint64_t)(uint32_t)(rec + 16)),
                BINGO_MOE_TABLE_ENTRY_BYTES, BINGO_MOE_REC_SLOT_BYTES, BINGO_MOE_REC_SLOT_BYTES,
                U);
    }
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t s = 0; s < U; s++) {
            volatile uint32_t *slot = rec + (t * U + s) * (BINGO_MOE_REC_SLOT_BYTES / 4u);
            uint16_t w16 = 0;
            for (uint32_t r = 0; r < k; r++)
                if (ids[t][r] == uid[s]) w16 = w16s[t][r];
            slot[0] = uid[s];
            slot[1] = w16 ? bingo_f16_to_f32bits(w16) : 0u;
            slot[2] = w16;
            for (uint32_t i = 3; i < 16u; i++) slot[i] = 0u;
        }
    }
    snrt_dma_wait_all();
    uint32_t refused = 0;
    // The ids go to the UART only on request: this kernel is on the layer's critical path
    // (every routed expert waits for the record), and eight printf_safe calls through the
    // UART cost ~500 us -- as long as the rest of the routing together. The host's check of
    // the record verifies the same thing.
#if defined(BINGO_MOE_ROUTE_VERBOSE) && BINGO_MOE_ROUTE_VERBOSE
    printf_safe("[Cluster %d] MoE route: %d slots", snrt_cluster_idx(), (int)U);
    for (uint32_t s = 0; s < U; s++) printf_safe(" %d", (int)uid[s]);
    printf_safe("\r\n");
#endif
    for (uint32_t s = 0; s < U; s++) {
        const volatile uint32_t *slot = rec + s * (BINGO_MOE_REC_SLOT_BYTES / 4u);
        if (!(slot[16] | slot[17]) || !(slot[20] | slot[21])) refused++;
    }
    if (refused) {
        printf_safe("[Cluster %d Core %d]: Error! moe_route: %d of the chosen experts have no "
                    "weights staged (a zero table entry)\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), (int)refused);
        return BINGO_RET_FAIL;
    }
    BINGO_RETURN_SUCC(sp, a->record_addr);
}

// FP32 bits -> FP16 bits, round to nearest even, for a finite value: a logit widened from
// FP16, which comes back exactly. Overflow saturates to infinity; a value below FP16's
// normals flushes to 0.
static inline uint16_t bingo_f32bits_to_f16(uint32_t f) {
    const uint16_t sign = (uint16_t)((f >> 16) & 0x8000u);
    const int32_t e = (int32_t)((f >> 23) & 0xFFu) - 127 + 15;
    uint32_t m = f & 0x7FFFFFu;
    if (e >= 31) return sign | 0x7C00u;
    if (e <= 0) return sign;
    uint32_t h = ((uint32_t)e << 10) | (m >> 13);
    const uint32_t rest = m & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) h++;   // carries into the exponent
    return sign | (uint16_t)h;
}

#define BINGO_MOE_PAD_F16 0xFBFFu          // -65504: no max or softmax picks it
#define BINGO_MOE_PAD_F32 0xC77FE000u      // -65504 as FP32
#define BINGO_MOE_CAND 8u                  // the TopK junction's candidates

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_router_pack(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_moe_router_pack_args_t);
    const __snax_bingo_kernel_moe_router_pack_args_t *a =
        (const __snax_bingo_kernel_moe_router_pack_args_t *)arg;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_moe_router_pack_args_t);
    const uint32_t n = a->n;
    if (a->mode > 2u || (a->mode < 2u && (n == 0u || n > BINGO_MOE_CAND))) {
        printf_safe("[Cluster %d Core %d]: Error! moe_router_pack: mode %d, n=%d (1..%d)\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), (int)a->mode, (int)n,
                    (int)BINGO_MOE_CAND);
        return BINGO_RET_FAIL;
    }
    if (a->mode == 0u) {
        const volatile uint16_t *lg = (const volatile uint16_t *)a->logits_addr;
        volatile uint16_t *row = (volatile uint16_t *)a->row_addr;
        for (uint32_t i = 0; i < 32u; i++) row[i] = i < n ? lg[i] : (uint16_t)BINGO_MOE_PAD_F16;
    } else if (a->mode == 1u) {
        const volatile uint16_t *row = (const volatile uint16_t *)a->logits_addr;
        // the n logits in descending order, equal values by the lower index (insertion sort)
        uint32_t idx[BINGO_MOE_CAND];
        int32_t key[BINGO_MOE_CAND];
        for (uint32_t i = 0; i < n; i++) {
            const int32_t kk = bingo_mono16(row[i]);
            uint32_t j = i;
            while (j > 0 && kk > key[j - 1]) {
                key[j] = key[j - 1];
                idx[j] = idx[j - 1];
                j--;
            }
            key[j] = kk;
            idx[j] = i;
        }
        volatile uint32_t *tk = (volatile uint32_t *)a->topk_addr;
        for (uint32_t i = 0; i < BINGO_MOE_CAND; i++) {
            tk[i] = i < n ? bingo_f16_to_f32bits(row[idx[i]]) : BINGO_MOE_PAD_F32;
            tk[8u + i] = i < n ? a->id_base + idx[i] : 0xFFFFFFFFu;
        }
        // the softmax-statistics group: the key beat (m at lane 0), then the value beat (l at
        // lane 0), FP16; row 0 is the only one the junction folds, the other lanes -65504
        volatile uint32_t *st = (volatile uint32_t *)a->stats_addr;
        const uint32_t m16 = *(const volatile uint16_t *)a->m_addr;
        const uint32_t s16 = *(const volatile uint16_t *)a->s_addr;
        const uint32_t pad2 = BINGO_MOE_PAD_F16 | (BINGO_MOE_PAD_F16 << 16);
        for (uint32_t i = 0; i < 16u; i++) {
            st[i] = i == 0u ? (BINGO_MOE_PAD_F16 << 16) | m16 : pad2;
            st[16u + i] = i == 0u ? (BINGO_MOE_PAD_F16 << 16) | s16 : pad2;
        }
    } else {
        const volatile uint32_t *tk = (const volatile uint32_t *)a->topk_addr;
        const volatile uint16_t *st = (const volatile uint16_t *)a->stats_addr;
        volatile uint16_t *row = (volatile uint16_t *)a->row_addr;
        volatile uint16_t *neg_m = (volatile uint16_t *)(a->row_addr - 64u);
        volatile uint16_t *s = (volatile uint16_t *)a->s_addr;
        volatile uint32_t *ids = (volatile uint32_t *)a->ids_addr;
        const uint16_t m16 = st[0];         // the key beat's lane 0: m*
        const uint16_t l16 = st[32];        // the value beat's lane 0: l*, FP16 from the junction
        for (uint32_t i = 0; i < 32u; i++) {
            row[i] = i < BINGO_MOE_CAND ? bingo_f32bits_to_f16(tk[i])
                                         : (uint16_t)BINGO_MOE_PAD_F16;
            neg_m[i] = (uint16_t)(m16 ^ 0x8000u);
            s[i] = l16;
        }
        for (uint32_t i = 0; i < BINGO_MOE_CAND; i++) ids[i] = tk[8u + i];
    }
    BINGO_RETURN_SUCC(sp, 0);
}
