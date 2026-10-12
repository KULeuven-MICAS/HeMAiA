// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// E2 collectives under offload_legacy: every chip runs this binary on its cluster 0, and every
// cross-chip synchronisation is software -- point-to-point stores and local polling, the
// primitive the software chip barrier (chip_sync.h) is built from. No hardware scheduler is
// involved, so a collective emulated in software pays for every hand-off it needs.
//
// ONE PRIMITIVE PER BUILD (COLL_PRIM, set by the snax-coll-<primitive> app):
//
//   broadcast   d2d    one D2D multicast of the root's buffer into every participant's L1,
//                      then a landed flag; the destinations only wait on it
//               pull   every destination pulls the root's buffer with its iDMA
//   reduce      xdma   one xDMA ChainGather by the root: the partials fold in transit in the
//                      writer junction of every chip the stream passes
//               pull   the root pulls every partial and folds them on its SIMD, pipelined
//               chain  software chain: chip i waits for chip i+1's partial sum, adds its own,
//                      pushes the sum to chip i-1; pipelined in COLL_CHAIN_CHUNKS chunks
//               tree   software binomial tree: log2 P levels of push + add, each receive
//                      behind a software clear-to-send and a software flag
//               xtree  the same tree folded by the xDMA: at each level a two-chip ChainGather
//                      through the giver's writer junction, after the giver's software flag
//   all-gather  d2d    every producer multicasts its part into every participant's L3, in
//                      turn (producer i waits for producer i-1's landed flag)
//               d2dn   every producer multicasts at once, without waiting for the others
//                      (runs only when params.hjson names it: overlapping multicasts
//                      that cross in opposite directions deadlock the D2D router)
//               a2a    every chip pushes its part into every other chip, behind a landed flag
//               ring   software ring: P-1 steps, each chip forwards the part it received
//                      to the next chip on the ring cycle (build_ring)
//               rd     software recursive doubling: log2 P exchanges of doubling blocks
//   all-reduce  xdma   the xDMA ChainGather into the root, then one D2D multicast of the sum
//               xtree  the xDMA tree (as in reduce) into the root, then the same multicast
//               a2a    every chip pushes its partial into every other chip's L3, behind a
//                      landed flag, and folds the P-1 it receives on its SIMD
//               ring   software ring all-reduce: reduce-scatter then all-gather, 2(P-1)
//                      steps of N/P bytes
//               rd     software recursive doubling: log2 P exchanges of the whole buffer
//
// THE PARTICIPANTS of a point are the first P chips of the snake: the compute columns walked
// in turn, down one column and up the next, so every prefix of length 2, 4, 8 and 16 is a
// rectangle of compute chips and consecutive chips are D2D neighbours (bar the one step that
// crosses a memory column). The multicast fence of every chip is set to that rectangle, so a
// 0xFF multicast reaches the P participants and no other chip.
//
// A POINT, all chips in lockstep:
//   grid barrier                    (software, through the comm buffer; untimed)
//   root: SRAM mark, t0, GO to every participant
//   participant: wait GO, READY to the root, run its role
//   root: run its role; a role that reads or writes other chips' memory from the root waits
//         for every READY first. The root then waits for every DONE (the primitives whose
//         output is on every participant) and every READY, t1, SRAM mark
//   grid barrier, then the untimed check and the report
// An EMPTY point is GO -> READY alone. The latency of a point is t1 - t0 minus the median
// empty point at the same P; its SRAM accesses are the difference of its two marks, minus
// the empty point's.
//
// READS, ONE SOURCE AT A TIME. Reads issued back to back by one iDMA to chips on different
// routes return out of order under its one AXI ID, and the R beat at the head of the D2D RX
// stream then waits for one queued behind it: the link deadlocks. Every pull arm keeps one
// source in flight, and the all-to-all arms push instead of pulling.
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
#include "hemaia_d2d_link_peripheral.h"
#include "occamy_base_addr.h"
#include "snax_core_roles.h"
#include "snax_simd_lib.h"

#define PRIM_BCAST 0
#define PRIM_REDUCE 2
#define PRIM_ALLGATHER 3
#define PRIM_ALLREDUCE 4

#ifndef COLL_PRIM
#error "COLL_PRIM is set by the snax-coll-<primitive> app Makefile"
#endif

#define N COLL_NBYTES
#define MAXC 16u
#define ROOT_IDX 0u

// ---- implementations -------------------------------------------------------------------
#define I_EMPTY 0u
#define I_D2D 1u     // broadcast, all-gather
#define I_XDMA 2u    // reduce, all-reduce (all-reduce: xDMA chain + D2D multicast)
#define I_PULL 3u
#define I_CHAIN 5u   // reduce
#define I_TREE 6u    // reduce
#define I_RING 7u    // all-gather, all-reduce
#define I_RD 8u      // all-gather, all-reduce
#define I_A2A 9u     // all-gather, all-reduce: all-to-all push
#define I_XTREE 10u  // reduce, all-reduce: binomial tree of two-chip xDMA chains, SW sync
#define I_D2DN 11u   // all-gather: every producer's D2D multicast at once, no turns

#if COLL_PRIM == PRIM_BCAST
#define PRIM_NAME "broadcast"
#define DST_OUTPUT 1
static const uint32_t IMPLS[] = {I_D2D, I_PULL};
#elif COLL_PRIM == PRIM_REDUCE
#define PRIM_NAME "reduce"
#define DST_OUTPUT 0
static const uint32_t IMPLS[] = {I_XDMA, I_XTREE, I_PULL, I_CHAIN, I_TREE};
#elif COLL_PRIM == PRIM_ALLGATHER
#define PRIM_NAME "all_gather"
#define DST_OUTPUT 1
#define USE_BIG 1
static const uint32_t IMPLS[] = {I_D2D, I_D2DN, I_A2A, I_RING, I_RD};
#elif COLL_PRIM == PRIM_ALLREDUCE
#define PRIM_NAME "all_reduce"
#define DST_OUTPUT 1
#define USE_BIG 1
static const uint32_t IMPLS[] = {I_XDMA, I_XTREE, I_A2A, I_RING, I_RD};
#else
#error "unknown COLL_PRIM"
#endif
#define N_IMPLS (sizeof(IMPLS) / sizeof(IMPLS[0]))

static const char *impl_name(uint32_t i) {
    switch (i) {
        case I_EMPTY: return "empty";
        case I_D2D: return "d2d";
        case I_XDMA: return "xdma";
        case I_PULL: return "pull";
        case I_CHAIN: return "chain";
        case I_TREE: return "tree";
        case I_RING: return "ring";
        case I_XTREE: return "xtree";
        case I_D2DN: return "d2dn";
        case I_RD: return "rd";
        case I_A2A: return "a2a";
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
#define C_DONE (CTRL + 0x0500u)    // [16] at the root
#define C_LFLAG (CTRL + 0x0900u)   // [16] landed flags in L1
#define C_CTS (CTRL + 0x0D00u)     // [16] clear-to-send, by level
#define C_SYNC (CTRL + 0x1100u)    // [64] data flags in L1, by step
#define C_ZERO (CTRL + 0x3000u)    // 4 KiB never written: the source of every zero fill
#define CTRL_SIZE 0x4000u
#define SLOT(base, i) ((base) + 64u * (uint32_t)(i))

#define IN (TCDM + 0x08000u)    // this chip's partial / the root's broadcast buffer
#define OUT (TCDM + 0x28000u)   // result
#define TMP (TCDM + 0x48000u)   // receive and scratch
_Static_assert(N <= 0x20000u, "the L1 buffers are 128 KiB apart");
_Static_assert(TMP + 0x20000u + 0x2000u <= TCDM + 0x80000u, "L1 buffers run into the stacks");

#ifdef USE_BIG
// The all-gather destination (all-reduce: the received partials), MAXC parts, and its landed
// flags, in L3: the same local address on every chip (one image). A flag for data in L3 lives
// in L3 too, behind the data on the same path into the same memory.
static uint8_t coll_big[MAXC * N] __attribute__((aligned(4096)));
static uint32_t coll_l3flag[64 * 16] __attribute__((aligned(64)));
#define BIG ((uint32_t)(uintptr_t)coll_big)
#define L3FLAG ((uint32_t)(uintptr_t)coll_l3flag)
#endif

// ---- per-core context ----------------------------------------------------------------------
typedef struct {
    uint8_t S[MAXC];   // the snake: S[i] = chip id at position i
    uint32_t nsnake;
    uint8_t self;
    uint32_t me;       // my snake position
    uint8_t R[MAXC];   // the ring arms' cycle through the P participants: chip ids
    uint8_t Rpos[MAXC];   // the snake position of R[i]
    uint32_t rp;       // my position on the ring
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
__attribute__((unused)) static inline uint64_t mcast(uint32_t a) { return chiplet_addr_transform_full(0xFF, (uint64_t)a); }

static inline void dma(uint64_t dst, uint64_t src, uint32_t n) {
    snrt_dma_start_1d_wideptr(dst, src, n);
}
static inline void dma_wait(void) { snrt_dma_wait_all(); }
// A flag behind the data: k, copied by the iDMA from C_VAL.
static inline void dma_flag(uint64_t dst) { dma(dst, loc(C_VAL), 4); }

// A control-only signal (GO, READY, DONE, clear-to-send): a core store.
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

// ---- the snake and the multicast fence ------------------------------------------------------
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

// The ring arms' cycle through the P participants. A snake prefix closes with one step back to
// its first chip. On the whole 4 x 4 compute grid around a memory column that step is four
// hops through the memory column, against the snake's own step across it; RING16 visits the
// same 16 chips in single hops and two crossings, one per direction in different rows. Used
// when it holds exactly the participants; COLL_RING_SNAKE keeps the snake.
static const uint8_t RING16[16] = {0x00, 0x01, 0x02, 0x03, 0x13, 0x33, 0x43, 0x42,
                                   0x41, 0x40, 0x30, 0x31, 0x32, 0x12, 0x11, 0x10};

static void build_ring(ctx_t *c) {
    const uint32_t P = c->P;
    for (uint32_t i = 0; i < P; i++) {
        c->R[i] = c->S[i];
        asm volatile("" ::: "memory");
    }
#if !COLL_RING_SNAKE
    if (P == 16u) {
        uint32_t ok = 1;
        for (uint32_t i = 0; i < 16u; i++) {
            uint32_t hits = 0;
            for (uint32_t j = 0; j < 16u; j++) hits += RING16[j] == c->S[i];
            if (hits != 1u) ok = 0;
        }
        if (ok)
            for (uint32_t i = 0; i < 16u; i++) {
                c->R[i] = RING16[i];
                asm volatile("" ::: "memory");   // a byte loop, not a call to memcpy (no libc)
            }
    }
#endif
    c->rp = 0xFFu;
    for (uint32_t i = 0; i < P; i++) {
        if (c->R[i] == c->self) c->rp = i;
        for (uint32_t j = 0; j < P; j++)
            if (c->S[j] == c->R[i]) c->Rpos[i] = (uint8_t)j;
    }
}

// The fence for participants S[0..P-1]: deliver here if this chip is one of them, forward
// toward a neighbour inside their bounding box. The box of a snake prefix holds no other
// compute chip; a memory chip in it (P = 16 on a grid with a memory column) only forwards.
static void set_fence(ctx_t *c, uint32_t P) {
    uint32_t x0 = 15, x1 = 0, y0 = 15, y1 = 0;
    for (uint32_t i = 0; i < P; i++) {
        uint32_t x = c->S[i] >> 4, y = c->S[i] & 0xFu;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
    }
    int x = c->self >> 4, y = c->self & 0xF;
#define INBOX(xx, yy)                                                                    \
    ((xx) >= (int)x0 && (xx) <= (int)x1 && (yy) >= (int)y0 && (yy) <= (int)y1 &&        \
     (xx) < (int)N_CHIPLETS_X && (yy) < (int)N_CHIPLETS_Y)
    uint32_t v = 0;
    if (c->me < P) {
        v |= 1u << 4;
        if (INBOX(x + 1, y)) v |= 1u << 0;   // east
        if (INBOX(x - 1, y)) v |= 1u << 1;   // west
        if (INBOX(x, y - 1)) v |= 1u << 2;   // north
        if (INBOX(x, y + 1)) v |= 1u << 3;   // south
    }
#undef INBOX
    volatile uint32_t *reg = (volatile uint32_t *)(uintptr_t)(
        HEMAIA_D2D_LINK_BASE_ADDR + HEMAIA_D2D_LINK_MULTICAST_FENCE_REGISTER_REG_OFFSET);
    *reg = v;
    asm volatile("fence" ::: "memory");
    if (*reg != v) {
        printf("[COLL] chip %02x: multicast fence reads %x, wrote %x\r\n", c->self, *reg, v);
        c->errs++;
    }
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
__attribute__((unused)) static void xdma_chain_fold(ctx_t *c, uint64_t src, const uint64_t *chain,
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
__attribute__((unused)) static void xdma_chain_reduce(ctx_t *c, uint32_t dst) {
    uint64_t chain[MAXC];
    uint32_t n = 0;
    for (uint32_t i = c->P - 1u; i >= 1u; i--) chain[n++] = rem(c->S[i], IN);
    chain[n++] = rem(c->self, dst);
    xdma_chain_fold(c, rem(c->self, IN), chain, n, dst);
}

// The binomial tree on snake positions: at level l, position p with bit l set (and no lower
// bit) hands its sum to p - 2^l, and position p with the low l + 1 bits clear takes p + 2^l's.
// How many sums position p takes before it hands its own on (the root: all log2 P).
__attribute__((unused)) static uint32_t tree_takes(uint32_t p, uint32_t P) {
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
__attribute__((unused)) static void xdma_tree_reduce(ctx_t *c) {
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

// The root pulls every other participant's IN and folds it into OUT on the SIMD, over two
// 64 KiB slots of TMP: the pull of the next half overlaps the add of this one.
// Fold the other P - 1 partials into OUT on the SIMD, over two 64 KiB slots of TMP: the copy
// of the next half overlaps the add of this one. pushed = 0: pull them from the other chips'
// IN (S[me + j]); pushed = 1: they were pushed into this chip's L3 slot of each sender (j-th
// from S[me - j], the order they are due) behind a landed flag.
__attribute__((unused)) static uint64_t fold_src(ctx_t *c, uint32_t P, uint32_t j, int pushed) {
#ifdef USE_BIG
    if (pushed) return loc(BIG + ((c->me + P - j) % P) * N);
#else
    (void)pushed;
#endif
    return rem(c->S[(c->me + j) % P], IN);
}

__attribute__((unused)) static void fold_all(ctx_t *c, uint32_t P, int pushed) {
    const uint32_t H = N / 2u;
    const uint32_t T = 2u * (P - 1u);
#ifdef USE_BIG
#define FOLD_WAIT(j) \
    if (pushed) wait_eq(SLOT(L3FLAG, (c->me + P - (j)) % P), c->k)
#else
#define FOLD_WAIT(j)
#endif
    if (c->dm) {
        FOLD_WAIT(1u);
        dma(loc(TMP), fold_src(c, P, 1u, pushed), H);
    }
    for (uint32_t t = 0; t < T; t++) {
        uint32_t h = t & 1u, src = 1u + t / 2u;
        if (c->dm) {
            dma_wait();
            if (t + 1u < T) {
                uint32_t h1 = (t + 1u) & 1u, src1 = 1u + (t + 1u) / 2u;
                if (h1 == 0u) FOLD_WAIT(src1);
                dma(loc(TMP + h1 * H), fold_src(c, P, src1, pushed) + h1 * H, H);
            }
        }
        cl_bar();
        uint32_t acc = (src == 1u) ? IN : OUT;
        simd_add(c, OUT + h * H, acc + h * H, TMP + h * H, H);
    }
#undef FOLD_WAIT
}

// ---- the roles ---------------------------------------------------------------------------------
static void role(ctx_t *c, uint32_t impl) {
    const uint32_t P = c->P, me = c->me, k = c->k;
    const int root = (me == ROOT_IDX);
    (void)P;
    (void)k;
    if (impl == I_EMPTY) return;

#if COLL_PRIM == PRIM_BCAST
    if (impl == I_D2D) {
        if (c->dm && root) {
            wait_ready_all(c);
            dma(mcast(OUT), loc(IN), N);
            dma_flag(mcast(SLOT(C_LFLAG, 0)));
            dma_wait();
        } else if (c->dm) {
            wait_eq(SLOT(C_LFLAG, 0), k);
        }
    } else if (impl == I_PULL) {
        if (c->dm && !root) {
            dma(loc(OUT), rem(c->S[0], IN), N);
            dma_wait();
        }
    }
#elif COLL_PRIM == PRIM_REDUCE
    if (impl == I_XDMA) {
        if (c->dm && root) wait_ready_all(c);
        cl_bar();
        if (root) xdma_chain_reduce(c, OUT);
    } else if (impl == I_XTREE) {
        xdma_tree_reduce(c);
    } else if (impl == I_PULL) {
        if (c->dm && root) wait_ready_all(c);
        cl_bar();
        if (root) fold_all(c, P, 0);
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
#elif COLL_PRIM == PRIM_ALLGATHER
    // every participant's BIG holds part i at i * N
    if (impl == I_D2D) {
        if (c->dm) {
            if (root) wait_ready_all(c);
            else wait_eq(SLOT(L3FLAG, me - 1u), k);   // the ring: one multicast in flight
            dma(mcast(BIG + me * N), loc(IN), N);
            dma_flag(mcast(SLOT(L3FLAG, me)));
            // this chip's own slot after the multicast: the next producer waits on the flag
            dma(loc(BIG + me * N), loc(IN), N);
            for (uint32_t i = 0; i < P; i++)
                if (i != me) wait_eq(SLOT(L3FLAG, i), k);
            dma_wait();
        }
    } else if (impl == I_D2DN) {
        // every producer multicasts at once, without waiting for the others: the overlapping
        // multicasts the d2d arm's turns avoid. Two that cross in opposite directions each lock a
        // router's outputs for a whole burst and wait at the other's router: a deadlock
        if (c->dm) {
            if (root) wait_ready_all(c);
            dma(mcast(BIG + me * N), loc(IN), N);
            dma_flag(mcast(SLOT(L3FLAG, me)));
            dma(loc(BIG + me * N), loc(IN), N);
            for (uint32_t i = 0; i < P; i++)
                if (i != me) wait_eq(SLOT(L3FLAG, i), k);
            dma_wait();
        }
    } else if (impl == I_A2A) {
        if (c->dm) {
            for (uint32_t j = 1; j < P; j++) {
                uint8_t to = c->S[(me + j) % P];
                dma(rem(to, BIG + me * N), loc(IN), N);
                dma_flag(rem(to, SLOT(L3FLAG, me)));
            }
            dma(loc(BIG + me * N), loc(IN), N);
            for (uint32_t i = 0; i < P; i++)
                if (i != me) wait_eq(SLOT(L3FLAG, i), k);
            dma_wait();
        }
    } else if (impl == I_RING) {
        if (c->dm) {
            const uint32_t rp = c->rp;
            uint8_t nx = c->R[(rp + 1u) % P];
            for (uint32_t s = 0; s + 1u < P; s++) {
                uint32_t p = c->Rpos[(rp + P - s) % P];   // the part sent at step s (its slot)
                if (s > 0) wait_eq(SLOT(L3FLAG, p), k);
                dma(rem(nx, BIG + p * N), loc(s == 0 ? IN : BIG + p * N), N);
                dma_flag(rem(nx, SLOT(L3FLAG, p)));
                if (s == 0) dma(loc(BIG + me * N), loc(IN), N);
            }
            wait_eq(SLOT(L3FLAG, c->Rpos[(rp + 1u) % P]), k);
            dma_wait();
        }
    } else if (impl == I_RD) {
        if (c->dm) {
            for (uint32_t b = 1, l = 0; b < P; b <<= 1, l++) {
                uint32_t base = me & ~(b - 1u);
                uint8_t to = c->S[me ^ b];
                if (b == 1u) {
                    dma(rem(to, BIG + me * N), loc(IN), N);
                } else {
                    // the block includes this chip's own slot, copied by an earlier transfer;
                    // one transfer's writes are not ordered before the next one's reads
                    dma_wait();
                    dma(rem(to, BIG + base * N), loc(BIG + base * N), b * N);
                }
                dma_flag(rem(to, SLOT(L3FLAG, 16u + l)));
                if (b == 1u) dma(loc(BIG + me * N), loc(IN), N);   // own slot, behind the send
                wait_eq(SLOT(L3FLAG, 16u + l), k);
            }
            dma_wait();
        }
    }
#elif COLL_PRIM == PRIM_ALLREDUCE
    if (impl == I_XDMA || impl == I_XTREE) {
        // the sum into the root's OUT (the chain, or the tree), then one D2D multicast of it
        if (impl == I_XDMA) {
            if (c->dm && root) wait_ready_all(c);
            cl_bar();
            if (root) xdma_chain_reduce(c, OUT);
        } else {
            xdma_tree_reduce(c);
        }
        if (c->dm && root) {
            dma(mcast(OUT), loc(OUT), N);
            dma_flag(mcast(SLOT(C_LFLAG, 0)));
            dma_wait();
        } else if (c->dm) {
            wait_eq(SLOT(C_LFLAG, 0), k);
        }
    } else if (impl == I_A2A) {
        if (c->dm) {
            for (uint32_t j = 1; j < P; j++) {
                uint8_t to = c->S[(me + j) % P];
                dma(rem(to, BIG + me * N), loc(IN), N);
                dma_flag(rem(to, SLOT(L3FLAG, me)));
            }
        }
        fold_all(c, P, 1);
        if (c->dm) dma_wait();
    } else if (impl == I_RING) {
        // reduce-scatter: at step s send chunk (rp - s), receive chunk (rp - s - 1) into
        // TMP slot s and add IN into it (into OUT at the last step: that chunk is final);
        // all-gather: at step s send chunk (rp + 1 - s) from OUT into the next chip's OUT.
        // (rp: this chip's position on the ring)
        const uint32_t Q = N / P, rp = c->rp;
        uint8_t nx = c->R[(rp + 1u) % P];
        uint32_t prev_res = IN + rp * Q;   // where the chunk sent at step s sits
        for (uint32_t s = 0; s + 1u < P; s++) {
            if (c->dm) {
                dma(rem(nx, TMP + s * Q), loc(prev_res), Q);
                dma_flag(rem(nx, SLOT(C_SYNC, s)));
                wait_eq(SLOT(C_SYNC, s), k);
            }
            cl_bar();
            uint32_t cr = (rp + 2u * P - s - 1u) % P;
            uint32_t dst = (s + 2u == P) ? OUT + cr * Q : TMP + s * Q;
            simd_add(c, dst, IN + cr * Q, TMP + s * Q, Q);
            prev_res = dst;
        }
        if (c->dm) {
            dma_wait();   // the last reduce-scatter send has left OUT's neighbour, TMP
            for (uint32_t s = 0; s + 1u < P; s++) {
                uint32_t ca = (rp + 1u + P - s) % P;
                if (s > 0) wait_eq(SLOT(C_SYNC, P - 1u + s - 1u), k);
                dma(rem(nx, OUT + ca * Q), loc(OUT + ca * Q), Q);
                dma_flag(rem(nx, SLOT(C_SYNC, P - 1u + s)));
            }
            wait_eq(SLOT(C_SYNC, P - 1u + P - 2u), k);
            dma_wait();
        }
    } else if (impl == I_RD) {
        // recursive doubling: at level l exchange the running sum with me ^ 2^l into TMP
        // (reused, so behind a clear-to-send from level 1 on) and add
        uint32_t l = 0;
        for (uint32_t b = 1; b < P; b <<= 1, l++) {
            uint8_t to = c->S[me ^ b];
            if (c->dm) {
                if (l > 0) wait_eq(SLOT(C_CTS, l), k);
                dma(rem(to, TMP), loc(l == 0 ? IN : OUT), N);
                dma_flag(rem(to, SLOT(C_SYNC, l)));
                dma_wait();
                wait_eq(SLOT(C_SYNC, l), k);
            }
            cl_bar();
            simd_add(c, OUT, l == 0 ? IN : OUT, TMP, N);
            if (c->dm && (b << 1) < P) sig(c, c->S[me ^ (b << 1)], SLOT(C_CTS, l + 1u), k);
        }
    }
#endif
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
    if (c->dm && me < P) {
        uint32_t bad = 0;
        int mine = 0;
        uint32_t in0 = (uint32_t)(uintptr_t)&coll_in[0][0];
#if COLL_PRIM == PRIM_BCAST
        mine = me != ROOT_IDX;
        if (mine) bad = check_samples(OUT, N, in0, 0);
#elif COLL_PRIM == PRIM_REDUCE
        mine = me == ROOT_IDX;
        if (mine) bad = check_samples(OUT, N, (uint32_t)(uintptr_t)coll_gold_sum[c->pi], 1);
#elif COLL_PRIM == PRIM_ALLGATHER
        mine = 1;
        bad = check_samples(BIG, P * N, in0, 0);
#elif COLL_PRIM == PRIM_ALLREDUCE
        mine = 1;
        bad = check_samples(OUT, N, (uint32_t)(uintptr_t)coll_gold_sum[c->pi], 1);
#endif
        (void)in0;
        if (mine && (bad || me == ROOT_IDX || me == P - 1u))
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
    if (c->dm && part) {
        if (root) {
            if (DST_OUTPUT && impl != I_EMPTY)
                for (uint32_t i = 1; i < P; i++) wait_eq(SLOT(C_DONE, i), k);
            wait_ready_all(c);
            t1 = snrt_mcycle();
            mark(c);
        } else if (DST_OUTPUT && impl != I_EMPTY) {
            sig(c, c->S[0], SLOT(C_DONE, me), k);
        }
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
    // so the exit's reduction (a spin in L3) does not run beside the measurement. In a reduce
    // build they also read one L3 word every ~32 K cycles while they wait: the testbench's LIVE
    // watchdog stops the simulation after 200 K cycles without main-memory traffic, and the
    // root's pull + fold of 15 partials runs L1 to L1 for longer than that. The other builds
    // do not read: their points touch L3 throughout, and the all-gather arms, whose iDMA streams
    // parts out of L3 to the next chip, stopped mid-stream when this read landed beside them.
    if (snrt_cluster_idx() != 0) {
        if (c.dm) {
            uint32_t halt = C_HALT + snrt_cluster_idx() * SNRT_CLUSTER_OFFSET;
            *(volatile uint32_t *)(uintptr_t)halt = 0;
            volatile uint32_t *h = (volatile uint32_t *)(uintptr_t)halt;
#if COLL_PRIM == PRIM_REDUCE
            volatile const uint32_t *l3 = (volatile const uint32_t *)coll_ps;
            uint32_t n = 0;
            while (*h != 0x600DF00Du) {
                pause_cc(1024);
                if ((++n & 31u) == 0u) (void)l3[0];
            }
#else
            while (*h != 0x600DF00Du) pause_cc(1024);
#endif
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
#ifdef USE_BIG
        zero(BIG, MAXC * N);
        zero(L3FLAG, 4096u);
#endif
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
        build_ring(&c);
        if (c.dm) {
            set_fence(&c, c.P);
            grid_barrier(&c);
        }
        cl_bar();
        for (uint32_t r = 0; r < COLL_EMPTY_REPEATS; r++) point(&c, I_EMPTY, r, 0);
        for (uint32_t ii = 0; ii < N_IMPLS; ii++) {
            if (!((COLL_IMPL_FILTER >> IMPLS[ii]) & 1u)) continue;   // params.hjson "impls"
            for (uint32_t r = 0; r < COLL_REPEATS; r++)
                point(&c, IMPLS[ii], r, r + 1u == COLL_REPEATS);
        }
    }

    // every chip's fence back to the whole grid, then release cluster 1
    if (c.dm) {
        set_fence(&c, c.nsnake);
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
