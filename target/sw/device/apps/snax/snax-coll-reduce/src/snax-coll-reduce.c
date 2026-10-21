// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// E2 reduce under offload_legacy: every chip runs this binary on its cluster 0, and every
// cross-chip synchronisation is software -- point-to-point stores and local polling, the
// primitive the software chip barrier (chip_sync.h) is built from. No hardware scheduler is
// involved, so a reduce emulated in software pays for every hand-off it needs.
//
// The sum of the participants' partials (FP16) lands in the root's OUT. Implementations:
//
//   in fabric: the xDMA ChainGather, whose stream folds a chip's buffer in that chip's writer
//   junction as it passes
//     xdma   one ChainGather by the root: the partials fold in transit in the writer junction
//            of every chip the stream passes
//     xtree  binomial tree on snake positions: at each level a two-chip ChainGather through
//            the giver's writer junction, after the giver's software flag
//     x2l    two-level tree: groups of COLL_GROUP consecutive snake chips each fold into their
//            group root (the member nearest the root) with one ChainGather, all at once; then
//            the root folds the group roots' sums with one more. Software flags: member ->
//            group root, group root -> root. P <= COLL_GROUP is the plain xdma chain
//   in software: the iDMA moves, the SIMD adds
//     pull   the root pulls every partial and folds them on its SIMD, pipelined over two slots
//     chain  chip i waits for chip i+1's partial sum, adds its own, pushes the sum to chip
//            i-1; pipelined in COLL_CHAIN_CHUNKS chunks
//     tree   binomial tree: log2 P levels of push + add, each receive behind a software
//            clear-to-send and a software flag
//
// THE PARTICIPANTS of a point are the first P chips of the snake: the compute columns walked
// in turn, down one column and up the next, so every prefix of length 2, 4, 8 and 16 is a
// rectangle of compute chips and consecutive chips are D2D neighbours (bar the one step that
// crosses a memory column).
//
// A POINT, all chips in lockstep:
//   grid barrier                    (software, through the comm buffer; untimed)
//   root: SRAM mark, t0, GO to every participant
//   participant: wait GO, READY to the root, run its role
//   root: run its role; a role that reads other chips' memory from the root waits for every
//         READY first. The root then waits for every READY, t1, SRAM mark
//   grid barrier, then the untimed check and the report
// An EMPTY point is GO -> READY alone. The latency of a point is t1 - t0 minus the median
// empty point at the same P; its SRAM accesses are the difference of its two marks, minus
// the empty point's.
//
// READS, ONE SOURCE AT A TIME. Reads issued back to back by one iDMA to chips on different
// routes return out of order under its one AXI ID, and the R beat at the head of the D2D RX
// stream then waits for one queued behind it: the link deadlocks. The root fold keeps one
// source in flight.
//
// HAND-OFFS. Data goes out on the iDMA, and its flag goes out on the same iDMA after it, into
// the same memory as the data, so a flag that has landed proves the data before it has. A
// remote write's B response is generated at the sender, so a flag the core stored after the
// DMA completed would prove nothing. A receive buffer that is reused within a point also
// needs a clear-to-send from the receiver before the sender may push into it.
//
// FLAG VALUES are the point's sequence number k, the same on every chip; a flag slot is only
// compared with k, so a value left by an earlier point never matches and no slot is cleared.

#include "snrt.h"

#include "chip_id.h"
#include "data.h"
#include "occamy_base_addr.h"
#include "snax_core_roles.h"
#include "snax_simd_lib.h"

#define N COLL_NBYTES
#define MAXC 16u
#define ROOT_IDX 0u

// ---- implementations -------------------------------------------------------------------
// The ids are the bits of COLL_IMPL_FILTER (datagen.py, params.hjson "impls").
#define I_EMPTY 0u
#define I_XDMA 2u    // one xDMA chain
#define I_PULL 3u    // root fold
#define I_CHAIN 5u   // software chain
#define I_TREE 6u    // software binomial tree
#define I_XTREE 10u  // binomial tree of two-chip xDMA chains, SW sync
#define I_X2L 12u    // two-level tree of xDMA chains, groups of COLL_GROUP, SW sync

#ifndef COLL_GROUP
#define COLL_GROUP 4u   // chips per first-level group of the two-level tree (params.hjson "group")
#endif

#define PRIM_NAME "reduce"   // the first word of every [COLL] line
static const uint32_t IMPLS[] = {I_XDMA, I_XTREE, I_X2L, I_PULL, I_CHAIN, I_TREE};
#define N_IMPLS (sizeof(IMPLS) / sizeof(IMPLS[0]))

static const char *impl_name(uint32_t i) {
    switch (i) {
        case I_EMPTY: return "empty";
        case I_XDMA: return "xdma";
        case I_PULL: return "pull";
        case I_CHAIN: return "chain";
        case I_TREE: return "tree";
        case I_XTREE: return "xtree";
        case I_X2L: return "x2l";
    }
    return "?";
}

// ---- memory map --------------------------------------------------------------------------
// Cluster 0's TCDM, the same local addresses on every chip: a remote address is the local one
// with the chip's prefix. The bottom 4 KiB is left to the L1 allocator's header; the stacks and
// the cluster-local storage sit at the top.
#define TCDM ((uint32_t)SNRT_TCDM_START_ADDR)
#define CTRL (TCDM + 0x1000u)
#define C_GO (CTRL + 0x0000u)
#define C_VAL (CTRL + 0x0040u)     // the word every DMA-written flag is copied from: k
#define C_MARK (CTRL + 0x0080u)    // the SRAM probe's mark word (chip 0x00 only)
#define C_HALT (CTRL + 0x00C0u)    // cluster 1's release (in cluster 1's TCDM)
#define C_READY (CTRL + 0x0100u)   // [16] at the root
#define C_CTS (CTRL + 0x0D00u)     // [16] clear-to-send, by level
#define C_SYNC (CTRL + 0x1100u)    // [64] data flags in L1, by step
#define C_ZERO (CTRL + 0x3000u)    // 4 KiB never written: the source of every zero fill
#define CTRL_SIZE 0x4000u
#define SLOT(base, i) ((base) + 64u * (uint32_t)(i))

#define IN (TCDM + 0x08000u)    // this chip's partial
#define OUT (TCDM + 0x28000u)   // result
#define TMP (TCDM + 0x48000u)   // receive and scratch
_Static_assert(N <= 0x20000u, "the L1 buffers are 128 KiB apart");
_Static_assert(TMP + 0x20000u + 0x2000u <= TCDM + 0x80000u, "L1 buffers run into the stacks");

// ---- per-core context ----------------------------------------------------------------------
typedef struct {
    uint8_t S[MAXC];   // the snake: S[i] = chip id at position i
    uint32_t nsnake;
    uint8_t self;
    uint32_t me;       // my snake position
    uint32_t P;
    uint32_t pi;       // index of P in coll_ps
    uint32_t k;        // point sequence number, the flag value
    uint32_t slot;     // next SRAM mark slot (root DM only)
    int dm, simd, xd;
    uint32_t errs;
    volatile comm_buffer_t *cb;
    uint8_t br;        // the grid barrier's bottom-right chip
} ctx_t;

static inline void cl_bar(void) { snrt_cluster_hw_barrier(); }

static inline void pause_cc(uint32_t n) {
    uint32_t t0 = snrt_mcycle();
    while (snrt_mcycle() - t0 < n) {
    }
}

// Spin until the word at `addr` (this chip, L1 or L3) holds v. The pause reads only mcycle, so
// a wait costs one memory read per poll however long it lasts.
static inline void wait_eq(uint32_t addr, uint32_t v) {
    volatile uint32_t *p = (volatile uint32_t *)(uintptr_t)addr;
    uint32_t spins = 0;
    while (*p != v) {
        pause_cc(16);
        if (++spins == (1u << 15))   // ~1 ms of simulated time: report, keep waiting
            printf("[COLL] chip %02x still waiting at %08x for %u (holds %u)\r\n",
                   get_current_chip_id(), addr, v, *p);
    }
}

static inline uint64_t loc(uint32_t a) { return chiplet_addr_transform((uint64_t)a); }
static inline uint64_t rem(uint8_t chip, uint32_t a) {
    return chiplet_addr_transform_full(chip, (uint64_t)a);
}

static inline void dma(uint64_t dst, uint64_t src, uint32_t n) {
    snrt_dma_start_1d_wideptr(dst, src, n);
}
static inline void dma_wait(void) { snrt_dma_wait_all(); }
// A flag behind the data: k, copied by the iDMA from C_VAL.
static inline void dma_flag(uint64_t dst) { dma(dst, loc(C_VAL), 4); }

// A control-only signal (GO, READY, clear-to-send, a tree's level flag): a core store.
static inline void sig(ctx_t *c, uint8_t chip, uint32_t addr, uint32_t v) {
    if (chip == c->self)
        *(volatile uint32_t *)(uintptr_t)addr = v;
    else
        snrt_xchip_writew(chip, addr, v);
}

static inline void mark(ctx_t *c) {
    *(volatile uint32_t *)(uintptr_t)C_MARK = 0x5A110000u | (c->slot & 0xFFFFu);
    c->slot++;
}

// The software grid barrier over every compute chip (master 0x00), through the comm buffer.
static inline void grid_barrier(ctx_t *c) { sw_chip_barrier(c->cb, 0x00, c->br); }

// ---- the snake -------------------------------------------------------------------------------
static int is_compute(uint8_t id) {
#ifdef HEMAIA_IS_COMPUTE_CHIP
    return (int)HEMAIA_IS_COMPUTE_CHIP(id);
#else
    (void)id;
    return 1;
#endif
}

static void build_snake(ctx_t *c) {
    uint32_t n = 0, col = 0;
    for (uint32_t x = 0; x < (uint32_t)N_CHIPLETS_X; x++) {
        uint32_t any = 0;
        for (uint32_t y = 0; y < (uint32_t)N_CHIPLETS_Y; y++) any |= is_compute((x << 4) | y);
        if (!any) continue;
        for (uint32_t j = 0; j < (uint32_t)N_CHIPLETS_Y; j++) {
            uint32_t y = (col & 1u) ? (uint32_t)N_CHIPLETS_Y - 1u - j : j;
            uint8_t id = (uint8_t)((x << 4) | y);
            if (is_compute(id) && n < MAXC) c->S[n++] = id;
        }
        col++;
    }
    c->nsnake = n;
    c->me = 0xFFu;
    for (uint32_t i = 0; i < n; i++)
        if (c->S[i] == c->self) c->me = i;
}

// ---- engines ---------------------------------------------------------------------------------
// out = a + b, FP16, on the SIMD core. Every core calls it; one cluster barrier.
static void simd_add(ctx_t *c, uint32_t out, uint32_t a, uint32_t b, uint32_t nbytes) {
    if (c->simd) {
        uint32_t stride;
        void *base = snax_simd_ew2_base((void *)(uintptr_t)a, (void *)(uintptr_t)b, &stride);
        uint32_t beats = nbytes / SIMD_BEAT_BYTES, rows = 1;
        if (nbytes % 4096u == 0u) {
            beats = 4096u / SIMD_BEAT_BYTES;
            rows = nbytes / 4096u;
        }
        snax_simd_shape_t in, o;
        snax_simd_shape_clear(&in);
        in.base = base;
        in.lane_stride = SIMD_LANE_BYTES;
        in.lane_mask = 0xFFFFFFFFu;
        in.byte_mask = 0xFFFFFFFFu;
        in.dim = 3;
        in.bound[0] = 2u;
        in.stride[0] = stride;
        in.bound[1] = beats;
        in.stride[1] = SIMD_BEAT_BYTES;
        in.bound[2] = rows;
        in.stride[2] = beats * SIMD_BEAT_BYTES;
        snax_simd_shape_flat(&o, (void *)(uintptr_t)out, rows * beats);
        snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 2u,
                       SIMD_EW_ADD);
        snax_simd_program_fast(&in, &o);
        snax_simd_fire();
        if (snax_simd_wait_all_bounded() || snax_simd_bad_config()) {
            printf("[COLL] chip %02x: SIMD add failed (status %08x)\r\n", c->self,
                   snax_read_simd_cfg_reg(SIMD_STATUS));
            c->errs++;
        }
    }
    cl_bar();
}

// An xDMA ChainGather issued by this chip: N bytes from `src` travel through the writer junction
// of every chip in chain[0..n-2], which adds the buffer at that address, and land at chain[n-1],
// this chip's `dst`. `src` and `dst` must differ: dst holds a sentinel in its first and last word
// until the data lands there.
// Every core calls it; one cluster barrier.
static void xdma_chain_fold(ctx_t *c, uint64_t src, const uint64_t *chain,
                                                    uint32_t n, uint32_t dst) {
    if (c->xd) {
        xdma_disable_all_extensions();
        for (uint8_t j = 0; j < XDMA_DST_JCT_NUM; j++) xdma_disable_dst_junction(j);
        volatile uint32_t *s = (volatile uint32_t *)(uintptr_t)dst;
        const uint32_t last = N / 4u - 1u;
        s[0] = 0xDEADBEEFu;
        s[last] = 0xDEADBEEFu;
        asm volatile("fence" ::: "memory");
        // ElementwiseJunction CSR(0): [3:0] op ADD = 0, [6:4] fmt FP16 = 0
        if (xdma_chain_gather_1d_full_address(src, (uint64_t *)chain, n, N,
                                              WRITER_JCT_ELEMENTWISEJUNCTION, 0u) != 0) {
            printf("[COLL] chip %02x: chain gather cfg failed\r\n", c->self);
            c->errs++;
        } else {
            uint32_t fin = snax_read_xdma_cfg_reg(XDMA_FINISH_REMOTE_TASK_PTR);
            uint32_t com = snax_read_xdma_cfg_reg(XDMA_COMMIT_REMOTE_TASK_PTR);
            xdma_task_t t = xdma_start();
            if (t.remote && fin > com) {
                printf("[COLL] chip %02x: standing xDMA finish %u > commit %u\r\n", c->self,
                       fin, com);
                c->errs++;
            }
            xdma_wait_task(t);
            // the finish counter can move before the writer's last store lands: the result is
            // whole when both ends hold data (a taker's sum is read by the next level at once)
            uint32_t spins = 0;
            while ((s[0] == 0xDEADBEEFu || s[last] == 0xDEADBEEFu) && spins < 4096u) {
                spins++;
                asm volatile("fence" ::: "memory");
            }
            if (spins == 4096u) {
                printf("[COLL] chip %02x: chain gather left the sentinel\r\n", c->self);
                c->errs++;
            }
        }
        xdma_disable_dst_junction(WRITER_JCT_ELEMENTWISEJUNCTION);
        for (uint32_t i = 1; i < XDMA_MAX_DST_COUNT; i++) {
            snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB + i * 2, 0);
            snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_MSB + i * 2, 0);
        }
    }
    cl_bar();
}

// The root's xDMA ChainGather over S[0..P-1]: the root's own IN travels out to S[P-1] and back
// through every participant's writer junction, which adds that chip's IN, and lands in the
// root's `dst`. Every core calls it; one cluster barrier.
static void xdma_chain_reduce(ctx_t *c, uint32_t dst) {
    uint64_t chain[MAXC];
    uint32_t n = 0;
    for (uint32_t i = c->P - 1u; i >= 1u; i--) chain[n++] = rem(c->S[i], IN);
    chain[n++] = rem(c->self, dst);
    xdma_chain_fold(c, rem(c->self, IN), chain, n, dst);
}

// The binomial tree on snake positions: at level l, position p with bit l set (and no lower
// bit) hands its sum to p - 2^l, and position p with the low l + 1 bits clear takes p + 2^l's.
// How many sums position p takes before it hands its own on (the root: all log2 P).
static uint32_t tree_takes(uint32_t p, uint32_t P) {
    uint32_t n = 0;
    for (uint32_t b = 1; b < P; b <<= 1) {
        if ((p & (2u * b - 1u)) == b) break;
        if ((p & (2u * b - 1u)) == 0u && p + b < P) n++;
    }
    return n;
}

// The reduce on that tree, folded by the xDMA: at each level the taker runs a two-chip
// ChainGather that carries its sum out through the giver's writer junction, which adds the
// giver's sum, and back. Software sync per level: the giver signals the taker once its own sum
// is final. A sum alternates between TMP and OUT, so a fold's source and destination differ
// and the last sum lands in OUT; a chip that took no sum hands on IN. The root ends with the
// sum in OUT. Every core calls it.
static void xdma_tree_reduce(ctx_t *c) {
    const uint32_t P = c->P, me = c->me, k = c->k;
    const uint32_t R = tree_takes(me, P);
    uint32_t acc = IN, m = 0;
    for (uint32_t b = 1, l = 0; b < P; b <<= 1, l++) {
        if ((me & (2u * b - 1u)) == b) {
            if (c->dm) sig(c, c->S[me - b], SLOT(C_SYNC, l), k);
            break;
        }
        if ((me & (2u * b - 1u)) == 0u && me + b < P) {
            m++;
            const uint32_t dst = ((R - m) & 1u) ? TMP : OUT;
            const uint32_t give = tree_takes(me + b, P) ? OUT : IN;
            if (c->dm) wait_eq(SLOT(C_SYNC, l), k);
            cl_bar();
            const uint64_t chain[2] = {rem(c->S[me + b], give), rem(c->self, dst)};
            xdma_chain_fold(c, rem(c->self, acc), chain, 2u, dst);
            acc = dst;
        }
    }
}

static void wait_ready_all(ctx_t *c) {
    for (uint32_t i = 1; i < c->P; i++) wait_eq(SLOT(C_READY, i), c->k);
}

// The group root of snake positions first..last: the member fewest mesh hops from the root
// chip, so with column-sized groups the group roots sit in the root's row.
static uint32_t group_root(ctx_t *c, uint32_t first, uint32_t last) {
    const int rx = c->S[ROOT_IDX] >> 4, ry = c->S[ROOT_IDX] & 0xF;
    uint32_t best = first, best_d = 0xFFu;
    for (uint32_t p = first; p <= last; p++) {
        int dx = (c->S[p] >> 4) - rx, dy = (c->S[p] & 0xF) - ry;
        uint32_t d = (uint32_t)((dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy));
        if (d < best_d) {
            best_d = d;
            best = p;
        }
    }
    return best;
}

// The two-level tree of xDMA chains. Level 1: the participants split into groups of
// COLL_GROUP consecutive snake positions, and every group root runs one ChainGather over its
// group (the farthest member first), all groups at once; a group of one keeps its IN as its
// sum. Level 2: the root runs one ChainGather over the group roots' sums (the farthest group
// first) into OUT. Software sync: every member signals its group root (its IN may be read),
// and every other group root signals the root once its group's sum is whole. The root's own
// group sum goes to TMP, the others' to OUT, so a fold's source and destination differ.
// P <= COLL_GROUP is one group: the plain xdma chain. Every core calls it.
static void xdma_two_level_reduce(ctx_t *c) {
    const uint32_t P = c->P, me = c->me, k = c->k, G = COLL_GROUP;
    const uint32_t K = (P + G - 1u) / G;   // groups
    if (K == 1u) {
        if (c->dm && me == ROOT_IDX) wait_ready_all(c);
        cl_bar();
        if (me == ROOT_IDX) xdma_chain_reduce(c, OUT);
        return;
    }
    const uint32_t g = me / G, first = g * G, last = (first + G < P ? first + G : P) - 1u;
    const uint32_t gr = group_root(c, first, last);
    if (me != gr) {
        if (c->dm) sig(c, c->S[gr], SLOT(C_SYNC, me), k);
        return;
    }
    // level 1: this group into its sum
    const uint32_t sum = (last == first) ? IN : (me == ROOT_IDX) ? TMP : OUT;
    if (last > first) {
        if (c->dm)
            for (uint32_t p = first; p <= last; p++)
                if (p != gr) wait_eq(SLOT(C_SYNC, p), k);
        cl_bar();
        uint64_t chain[MAXC];
        uint32_t n = 0;
        for (uint32_t d = last - first; d >= 1u; d--) {
            if (gr + d <= last) chain[n++] = rem(c->S[gr + d], IN);
            if (gr >= first + d) chain[n++] = rem(c->S[gr - d], IN);
        }
        chain[n++] = rem(c->self, sum);
        xdma_chain_fold(c, rem(c->self, IN), chain, n, sum);
    }
    if (me != ROOT_IDX) {
        if (c->dm) sig(c, c->S[ROOT_IDX], SLOT(C_SYNC, 32u + g), k);
        return;
    }
    // level 2: the group roots' sums into the root's OUT
    if (c->dm)
        for (uint32_t j = 1; j < K; j++) wait_eq(SLOT(C_SYNC, 32u + j), k);
    cl_bar();
    uint64_t chain[MAXC];
    uint32_t n = 0;
    for (uint32_t j = K - 1u; j >= 1u; j--) {
        const uint32_t f = j * G, l = (f + G < P ? f + G : P) - 1u;
        chain[n++] = rem(c->S[group_root(c, f, l)], l > f ? OUT : IN);
    }
    chain[n++] = rem(c->self, OUT);
    xdma_chain_fold(c, rem(c->self, sum), chain, n, OUT);
}

// The root pulls every other participant's IN and folds it into OUT on the SIMD, over two
// 64 KiB slots of TMP: the pull of the next half overlaps the add of this one.
static void fold_all(ctx_t *c, uint32_t P) {
    const uint32_t H = N / 2u;
    const uint32_t T = 2u * (P - 1u);
    if (c->dm) dma(loc(TMP), rem(c->S[(c->me + 1u) % P], IN), H);
    for (uint32_t t = 0; t < T; t++) {
        uint32_t h = t & 1u, src = 1u + t / 2u;
        if (c->dm) {
            dma_wait();
            if (t + 1u < T) {
                uint32_t h1 = (t + 1u) & 1u, src1 = 1u + (t + 1u) / 2u;
                dma(loc(TMP + h1 * H), rem(c->S[(c->me + src1) % P], IN) + h1 * H, H);
            }
        }
        cl_bar();
        uint32_t acc = (src == 1u) ? IN : OUT;
        simd_add(c, OUT + h * H, acc + h * H, TMP + h * H, H);
    }
}

// ---- the roles ---------------------------------------------------------------------------------
static void role(ctx_t *c, uint32_t impl) {
    const uint32_t P = c->P, me = c->me, k = c->k;
    const int root = (me == ROOT_IDX);
    if (impl == I_EMPTY) return;
    if (impl == I_XDMA) {
        if (c->dm && root) wait_ready_all(c);
        cl_bar();
        if (root) xdma_chain_reduce(c, OUT);
    } else if (impl == I_XTREE) {
        xdma_tree_reduce(c);
    } else if (impl == I_X2L) {
        xdma_two_level_reduce(c);
    } else if (impl == I_PULL) {
        if (c->dm && root) wait_ready_all(c);
        cl_bar();
        if (root) fold_all(c, P);
    } else if (impl == I_CHAIN) {
        // chip me receives the running sum from me + 1 into TMP, chunk by chunk, adds its IN
        // and pushes the sum on to me - 1; the far end pushes its IN
        const uint32_t C = COLL_CHAIN_CHUNKS, Q = N / C;
        const int far = (me == P - 1u);
        for (uint32_t q = 0; q < C; q++) {
            if (!far) {
                if (c->dm) wait_eq(SLOT(C_SYNC, q), k);
                cl_bar();
                simd_add(c, OUT + q * Q, IN + q * Q, TMP + q * Q, Q);
            }
            if (c->dm && !root) {
                uint8_t nx = c->S[me - 1u];
                dma(rem(nx, TMP + q * Q), loc((far ? IN : OUT) + q * Q), Q);
                dma_flag(rem(nx, SLOT(C_SYNC, q)));
            }
        }
        if (c->dm) dma_wait();
    } else if (impl == I_TREE) {
        // binomial tree on snake positions: at level l, me with bit l set (and no lower bit)
        // sends its sum to me - 2^l; me with the low l+1 bits clear receives from me + 2^l.
        // TMP is reused at every level, so a receiver clears its sender to send.
        uint32_t L = 0;
        while ((1u << L) < P) L++;
        int have = 0;   // OUT holds a running sum
        for (uint32_t l = 0; l < L; l++) {
            uint32_t b = 1u << l;
            if ((me & (2u * b - 1u)) == b) {
                // send, then done
                if (c->dm) {
                    uint8_t to = c->S[me - b];
                    if (l > 0) wait_eq(SLOT(C_CTS, l), k);
                    dma(rem(to, TMP), loc(have ? OUT : IN), N);
                    dma_flag(rem(to, SLOT(C_SYNC, l)));
                    dma_wait();
                }
                break;
            }
            if ((me & (2u * b - 1u)) == 0u && me + b < P) {
                if (c->dm) {
                    if (l > 0) sig(c, c->S[me + b], SLOT(C_CTS, l), k);
                    wait_eq(SLOT(C_SYNC, l), k);
                }
                cl_bar();
                simd_add(c, OUT, have ? OUT : IN, TMP, N);
                have = 1;
            }
        }
    }
}

// ---- checks ------------------------------------------------------------------------------------
// Compare `span` bytes at `buf` (this chip, L1 or L3) with the golden, one word every
// sample_stride(span) bytes. gold_sampled: the golden is already that sample array;
// otherwise it is a contiguous buffer sampled the same way.
static uint32_t check_samples(uint32_t buf, uint32_t span, uint32_t gold, int gold_sampled) {
    uint32_t stride = 64u * ((span + COLL_SAMPLE_SPAN - 1u) / COLL_SAMPLE_SPAN) + 4u;
    uint32_t ns = (span - 4u) / stride + 1u;
    snrt_dma_start_2d_wideptr(loc(TMP), loc(buf), 4, 4, stride, ns);
    if (gold_sampled)
        dma(loc(TMP + 0x4000u), loc(gold), ns * 4u);
    else
        snrt_dma_start_2d_wideptr(loc(TMP + 0x4000u), loc(gold), 4, 4, stride, ns);
    dma_wait();
    volatile uint32_t *got = (volatile uint32_t *)(uintptr_t)TMP;
    volatile uint32_t *exp = (volatile uint32_t *)(uintptr_t)(TMP + 0x4000u);
    uint32_t bad = 0;
    for (uint32_t i = 0; i < ns; i++) {
        if (got[i] != exp[i]) {
            if (bad < 3)
                printf("[COLL] chip %02x: sample %u at +%u: got %08x want %08x\r\n",
                       get_current_chip_id(), i, i * stride, got[i], exp[i]);
            bad++;
        }
    }
    return bad;
}

static void check(ctx_t *c, uint32_t impl) {
    const uint32_t P = c->P, me = c->me;
    if (c->dm && me == ROOT_IDX) {   // the sum lands at the root only
        uint32_t bad = check_samples(OUT, N, (uint32_t)(uintptr_t)coll_gold_sum[c->pi], 1);
        printf("Check [%s_%s_P%u_c%02x]: %s (%u bad samples)\r\n", PRIM_NAME,
                   impl_name(impl), P, c->self, bad ? "FAIL" : "PASS", bad);
        c->errs += bad;
    }
    cl_bar();
}

// ---- a point -----------------------------------------------------------------------------------
static void point(ctx_t *c, uint32_t impl, uint32_t rep, int do_check) {
    c->k++;
    const uint32_t k = c->k, P = c->P, me = c->me;
    const int part = me < P, root = (me == ROOT_IDX);
    uint32_t t0 = 0, t1 = 0, s0 = 0;
    if (c->dm) {
        *(volatile uint32_t *)(uintptr_t)C_VAL = k;
        grid_barrier(c);
        if (root) {
            s0 = c->slot;
            mark(c);
            t0 = snrt_mcycle();
            for (uint32_t i = 1; i < P; i++) sig(c, c->S[i], C_GO, k);
        } else if (part) {
            wait_eq(C_GO, k);
            sig(c, c->S[0], SLOT(C_READY, me), k);
        }
    }
    cl_bar();
    if (part) role(c, impl);
    if (c->dm && root) {
        wait_ready_all(c);
        t1 = snrt_mcycle();
        mark(c);
    }
    cl_bar();
    if (c->dm) grid_barrier(c);
    cl_bar();
    if (c->dm && root)
        printf("[COLL] %s %s P=%u r=%u k=%u rt=%u s=%u\r\n", PRIM_NAME, impl_name(impl), P,
               rep, k, t1 - t0, s0);
    if (do_check) check(c, impl);
    cl_bar();
}

static void zero(uint32_t dst, uint32_t n) {
    // a 2-D DMA whose source does not advance: the 4 KiB zero block, n / 4 KiB times
    snrt_dma_start_2d_wideptr(loc(dst), loc(C_ZERO), 4096, 4096, 0, n / 4096u);
}

int main() {
    ctx_t c;
    c.self = get_current_chip_id();
    c.dm = snrt_is_dm_core();
    c.simd = snax_is_simd_core();
    c.xd = snax_is_xdma_core();
    c.k = 0;
    c.slot = 0;
    c.errs = 0;
    c.cb = get_communication_buffer();
    c.br = (uint8_t)((((uint32_t)N_CHIPLETS_X - 1u) << 4) | ((uint32_t)N_CHIPLETS_Y - 1u));

    // Cluster 1 and up take no part. They wait in their own TCDM until cluster 0 is done,
    // so the exit's reduction (a spin in L3) does not run beside the measurement. They read
    // one L3 word every ~32 K cycles while they wait: the testbench's LIVE watchdog stops the
    // simulation after 200 K cycles without main-memory traffic, and the root's pull + fold of
    // 15 partials runs L1 to L1 for longer than that.
    if (snrt_cluster_idx() != 0) {
        if (c.dm) {
            uint32_t halt = C_HALT + snrt_cluster_idx() * SNRT_CLUSTER_OFFSET;
            *(volatile uint32_t *)(uintptr_t)halt = 0;
            volatile uint32_t *h = (volatile uint32_t *)(uintptr_t)halt;
            volatile const uint32_t *l3 = (volatile const uint32_t *)coll_ps;
            uint32_t n = 0;
            while (*h != 0x600DF00Du) {
                pause_cc(1024);
                if ((++n & 31u) == 0u) (void)l3[0];
            }
        }
        cl_bar();
        return 0;
    }

    build_snake(&c);
    if (c.me == 0xFFu || c.nsnake < COLL_NCHIPS) {
        if (c.dm)
            printf("[COLL] chip %02x: not on the snake (%u compute chips, data for %u)\r\n",
                   c.self, c.nsnake, (uint32_t)COLL_NCHIPS);
        return 1;
    }

    // Setup: zero the control words and every buffer (an unwritten word reads X), stage
    // this chip's partial, then one grid barrier before anyone may write another chip.
    if (c.dm) {
        volatile uint32_t *z = (volatile uint32_t *)(uintptr_t)CTRL;
        for (uint32_t i = 0; i < CTRL_SIZE / 4u; i++) z[i] = 0;
        asm volatile("fence" ::: "memory");
        zero(OUT, 0x20000u);
        zero(TMP, 0x20000u);
        dma(loc(IN), loc((uint32_t)(uintptr_t)&coll_in[c.me][0]), N);
        dma_wait();
        if (c.self == 0x00)
            printf("[COLL] %s: %u B per chip, %u impls, snake %02x..%02x, %u repeats\r\n",
                   PRIM_NAME, (uint32_t)N, (uint32_t)N_IMPLS, c.S[0], c.S[c.nsnake - 1u],
                   (uint32_t)COLL_REPEATS);
        grid_barrier(&c);
    }
    cl_bar();

    for (uint32_t pi = 0; pi < COLL_NP; pi++) {
        c.P = coll_ps[pi];
        c.pi = pi;
        if (c.dm) grid_barrier(&c);
        cl_bar();
        for (uint32_t r = 0; r < COLL_EMPTY_REPEATS; r++) point(&c, I_EMPTY, r, 0);
        for (uint32_t ii = 0; ii < N_IMPLS; ii++) {
            if (!((COLL_IMPL_FILTER >> IMPLS[ii]) & 1u)) continue;   // params.hjson "impls"
            for (uint32_t r = 0; r < COLL_REPEATS; r++)
                point(&c, IMPLS[ii], r, r + 1u == COLL_REPEATS);
        }
    }

    // release cluster 1
    if (c.dm) {
        grid_barrier(&c);
        for (uint32_t cl = 1; cl < (uint32_t)N_CLUSTERS_PER_CHIPLET; cl++)
            *(volatile uint32_t *)(uintptr_t)(C_HALT + cl * SNRT_CLUSTER_OFFSET) = 0x600DF00Du;
        if (c.self == 0x00)
            printf("[COLL] %s done: %u errors on chip 00\r\n", PRIM_NAME, c.errs);
        else if (c.errs)
            printf("[COLL] %s done: %u errors on chip %02x\r\n", PRIM_NAME, c.errs, c.self);
    }
    cl_bar();
    return c.errs ? 1 : 0;
}
