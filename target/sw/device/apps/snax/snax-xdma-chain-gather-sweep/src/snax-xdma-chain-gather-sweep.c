// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Cross-die ChainGather SCALING SWEEP on the 16-chiplet platform (hemaia_16chiplet).
//
// This one app replaces the four earlier ChainGather bring-ups (single-chip linear,
// single-chip monoid, 4-chiplet linear, 4-chiplet monoid). It measures how the in-fabric
// fold scales with the number of participating dies: for each chain width P in SWEEP_P
// (2, 4, 8, 16) it runs BOTH junctions and reports the latency of each.
//
//   LINEAR  ElementwiseJunction, per-element FP32 ADD  -> byte-exact check.
//   MOMENT  MonoidJunction, the nonlinear online-softmax (m, l) merge
//           (m1,l1) (+) (m2,l2) = (max(m1,m2), l1*exp(m1-m*) + l2*exp(m2-m*))
//           -> m* exact, l* checked to a ULP tolerance (writer exp LUT).
//
// THE GRID AND THE SNAKE.  hemaia_16chiplet places 16 compute chiplets on a 4x4 grid plus a
// memory chiplet at [4,0]. chip_id = (x << 4) | y (chip_id.h), so with x the column and y the
// row the grid is
//
//        y=0    y=1    y=2    y=3
//   x=0  0x00   0x01   0x02   0x03
//   x=1  0x10   0x11   0x12   0x13
//   x=2  0x20   0x21   0x22   0x23
//   x=3  0x30   0x31   0x32   0x33
//
// The chain MUST follow physically ADJACENT D2D hops. The backward grant handshake (the root
// grants its previous hop, which grants its previous hop, ... until the head is released) does
// not route between non-neighbouring chiplets, so a chain with a diagonal hop stalls with the
// head holding data forever. SNAKE[] below is a boustrophedon Hamiltonian path over the grid
// that starts at the collector, so EVERY consecutive pair is a real N/S/E/W neighbour -- and,
// crucially, every PREFIX of it is itself a valid all-adjacent chain. That prefix property is
// what lets one table and one data set serve every P in the sweep.
//
// The chiplet at snake index i owns partial i, so a width-P round folds partials [0 .. P-1]
// and the golden for that round is the datagen's fold over the same prefix.
//
// CHAIN LENGTH.  A width-P gather programs P xDMA dst slots (P-1 remote sources + the
// collector's own dst buffer), so P == NUM_CHIPLETS == 16 sits exactly at the hardware cap
// XDMA_MAX_DST_COUNT. The static assert below fails the build rather than letting
// xdma_multicast_1d_full_address() reject the round at run time.
//
// CROSS-CHIP SYNC.  snrt_chip_global_barrier() announces via a 0xFF-BROADCAST scalar store,
// whose routing is not exercised on this platform and which hangs here. We instead use
// TARGETED cross-chip scalar stores (the proven direction) plus local polling: each source
// writes a ready flag into the collector's TCDM, the collector polls locally, gathers, then
// writes a done flag back into each source. Each round owns its own flag slots, so no round
// ever has to clear another's flags and there is no clear/set race between rounds.

#include "chip_id.h"  // get_current_chip_id, get_chip_baseaddress_value, chiplet_addr_transform_full
#include "data.h"
#include "snrt.h"

#define PARTIAL_BYTES (PARTIAL_ELEMS * (int)sizeof(float))

// The collector is the chiplet at snake index 0 -- chip 0x00, the boot chip.
#define COLLECTOR 0x00

// Two junctions measured per width.
#define MODE_LIN 0
#define MODE_MOM 1
#define NUM_MODES 2
#define NUM_ROUNDS (NUM_MODES * NUM_SWEEP_P)

// TCDM map. The four data beats live in the first 256 B; the sync flags sit well clear of them.
#define LIN_SRC_OFF 0x000u                    // this chiplet's linear partial
#define LIN_DST_OFF 0x040u                    // collector's linear result
#define MOM_SRC_OFF 0x080u                    // this chiplet's (m, l) beat
#define MOM_DST_OFF 0x0C0u                    // collector's merged (m*, l*)
#define READY_OFF 0x400u                      // collector side: ready[round][snake_idx], u32
#define DONE_OFF 0x600u                       // source side:    done[round],            u32
#define FLAG_SET 0xA5A5A5A5u

// fp32 ULP tolerance for l* (exp-LUT approximation). ~0x20000 ULPs ~= 1.5% at l* ~ 1.19,
// matching the Gate-A moment-merge acceptance bound. The measured delta is printed so the
// bound can be tightened once the sweep has run.
#define L_TOL_ULPS 0x20000u

// Bounded wait. A sweep is worth more when one hanging width still yields the others, so the
// collector polls with a spin cap instead of spinning forever: a stuck round is reported as
// TIMEOUT and the sweep carries on.
#define WAIT_SPINS 1000000u

// Boustrophedon Hamiltonian path over the 4x4 grid, starting at the collector. Every
// consecutive pair differs in exactly one coordinate by one, and so does every prefix.
//   0x00 -> 0x10 -> 0x20 -> 0x30   (down column y=0)
//   0x31 <- 0x21 <- 0x11 <- 0x01   (back along y=1)
//   0x02 -> 0x12 -> 0x22 -> 0x32   (down y=2)
//   0x33 <- 0x23 <- 0x13 <- 0x03   (back along y=3)
static const uint8_t SNAKE[NUM_CHIPLETS] = {
    0x00, 0x10, 0x20, 0x30,
    0x31, 0x21, 0x11, 0x01,
    0x02, 0x12, 0x22, 0x32,
    0x33, 0x23, 0x13, 0x03,
};

// Index of a chip within SNAKE[], or -1 if it is not on the grid.
static int snake_index(uint8_t chip) {
    for (int i = 0; i < NUM_CHIPLETS; i++) {
        if (SNAKE[i] == chip) return i;
    }
    return -1;
}

// Targeted cross-chip 32-bit store via the Mseg CSR (0xbc0). This mirrors
// announce_chip_checkpoint()'s mechanism but aims at ONE specific chip's prefix instead of the
// 0xFF broadcast: set Mseg to the target chip's high bits, store to the (identical-layout)
// local address, restore Mseg.
static inline void xchip_store_u32(uint8_t target_chip, uint32_t local_addr, uint32_t val) {
    uint32_t tgt_h = (uint32_t)(get_chip_baseaddress_value(target_chip) >> 32);
    uint32_t cur_h = (uint32_t)(get_current_chip_baseaddress_value() >> 32);
    register uint32_t r_h asm("t0") = tgt_h;
    register uint32_t r_v asm("t1") = val;
    register uint32_t r_a asm("t2") = local_addr;
    register uint32_t r_c asm("t3") = cur_h;
    asm volatile(
        "csrw 0xbc0, t0;"
        "sw   t1, 0(t2);"
        "csrw 0xbc0, t3;"
        :
        : "r"(r_h), "r"(r_v), "r"(r_a), "r"(r_c)
        : "memory");
}

#ifdef XDMA_DST_JCT_ENABLE_PTR

_Static_assert(NUM_CHIPLETS <= XDMA_MAX_DST_COUNT,
    "a width-NUM_CHIPLETS chain needs NUM_CHIPLETS xDMA dst slots (NUM_CHIPLETS-1 remote "
    "sources + the local dst); raise the xDMA multicast width or lower num_chiplets.");

// Poll the finish counter the HW actually bumped, bounded. Returns 1 on completion, 0 on
// timeout.
static int xdma_wait_task_bounded(xdma_task_t task) {
    for (uint32_t s = 0; s < WAIT_SPINS; s++) {
        uint32_t f = task.remote
                         ? snax_read_xdma_cfg_reg(XDMA_FINISH_REMOTE_TASK_PTR)
                         : snax_read_xdma_cfg_reg(XDMA_FINISH_LOCAL_TASK_PTR);
        if (f >= task.task_id) return 1;
    }
    return 0;
}

// One measured round: fold the first `width` partials into the collector's dst buffer.
// Returns the number of check errors; *cycles gets the xDMA task latency (0 on timeout).
static int run_round(int mode, int width, uint32_t tcdm_base, const float *golden,
                     uint32_t *task_cycles, uint32_t *wall_cycles) {
    const uint32_t src_off = (mode == MODE_LIN) ? LIN_SRC_OFF : MOM_SRC_OFF;
    const uint32_t dst_off = (mode == MODE_LIN) ? LIN_DST_OFF : MOM_DST_OFF;
    const char *tag = (mode == MODE_LIN) ? "lin" : "mom";

    // Clean writer config: no reader/writer extensions, no stale junction from the last round.
    for (uint8_t e = 0; e < XDMA_DST_EXT_NUM; e++) xdma_disable_dst_ext(e);
    for (uint8_t r = 0; r < XDMA_SRC_EXT_NUM; r++) xdma_disable_src_ext(r);
    for (uint8_t j = 0; j < XDMA_DST_JCT_NUM; j++) xdma_disable_dst_junction(j);

    uint64_t local_src = chiplet_addr_transform_full(COLLECTOR, (uint64_t)(tcdm_base + src_off));
    uint64_t dst_local = chiplet_addr_transform_full(COLLECTOR, (uint64_t)(tcdm_base + dst_off));

    // Chain in DATA order: far source first (snake index width-1), down to the nearest source
    // (snake index 1), ending at the collector's own dst buffer. Every partial sits at the same
    // local offset on every chiplet, so a chain entry is just that chiplet's prefix + offset.
    uint64_t chain[NUM_CHIPLETS];
    int n = 0;
    for (int i = width - 1; i >= 1; i--) {
        chain[n++] = chiplet_addr_transform_full(SNAKE[i], (uint64_t)(tcdm_base + src_off));
    }
    chain[n++] = dst_local;

    // Sentinel-fill the dst so "the writer never wrote" is distinguishable from "it wrote the
    // wrong value". Integer stores only -- the DM core is rv32ima with no FPU.
    volatile uint32_t *dbg = (volatile uint32_t *)(uintptr_t)(tcdm_base + dst_off);
    for (int j = 0; j < PARTIAL_ELEMS; j++) dbg[j] = 0xDEADBEEFu;

    uint8_t junction;
    uint32_t jct_csr;
    if (mode == MODE_LIN) {
        // ElementwiseJunction CSR(0): [3:0] op = ADD(0), [6:4] fmt = FP32(3).
        junction = WRITER_JCT_ELEMENTWISEJUNCTION;
        jct_csr = (3u << 4) | 0u;
    } else {
        // MonoidJunction CSR(0): [15:13] combineMode = MOMENT(1), [7:0] nValid = 1.
        junction = WRITER_JCT_MONOIDJUNCTION;
        jct_csr = (1u << 13) | 1u;
    }

    int32_t ret = xdma_chain_gather_1d_full_address(local_src, chain, (uint32_t)n,
                                                    PARTIAL_BYTES, junction, jct_csr);
    if (ret != 0) {
        printf("[Sweep] P=%d %s: cfg FAILED (%d)\r\n", width, tag, ret);
        *task_cycles = 0;
        *wall_cycles = 0;
        return 1;
    }

    uint32_t t0 = snrt_mcycle();
    xdma_task_t task = xdma_start();
    int done = xdma_wait_task_bounded(task);
    uint32_t t1 = snrt_mcycle();

    if (!done) {
        printf("[Sweep] P=%d %s: TIMEOUT after %u spins (remote=%d id=%u)\r\n",
               width, tag, WAIT_SPINS, task.remote, task.task_id);
        *task_cycles = 0;
        *wall_cycles = t1 - t0;
        return 1;
    }

    *task_cycles = xdma_last_task_cycle();
    *wall_cycles = t1 - t0;

    // Check. Compare raw fp32 BIT PATTERNS as integers: the DM core is rv32ima with NO FPU, so
    // a float compare would emit flw and trap as an illegal instruction.
    volatile uint32_t *res = (volatile uint32_t *)(uintptr_t)(tcdm_base + dst_off);
    const uint32_t *gold = (const uint32_t *)golden;
    int err = 0;

    if (mode == MODE_LIN) {
        // Small integer-valued fp32 operands -> the sum is exact -> byte-exact compare.
        for (int j = 0; j < PARTIAL_ELEMS; j++) {
            if (res[j] != gold[j]) {
                if (err < 4) {
                    printf("[Sweep] P=%d lin MISMATCH at %d: got %x exp %x\r\n",
                           width, j, res[j], gold[j]);
                }
                err++;
            }
        }
    } else {
        // m* is a max, so exact; l* goes through the writer's exp LUT, so ULP-bounded.
        uint32_t m_got = res[MOMENT_M_LANE], m_exp = gold[MOMENT_M_LANE];
        uint32_t l_got = res[MOMENT_L_LANE], l_exp = gold[MOMENT_L_LANE];
        uint32_t l_ulp = (l_got > l_exp) ? (l_got - l_exp) : (l_exp - l_got);
        printf("[Sweep] P=%d mom m*: got %x exp %x | l*: got %x exp %x (ulp=%x tol=%x)\r\n",
               width, m_got, m_exp, l_got, l_exp, l_ulp, L_TOL_ULPS);
        if (m_got != m_exp) {
            printf("[Sweep] P=%d mom m* MISMATCH\r\n", width);
            err++;
        }
        if (l_ulp > L_TOL_ULPS) {
            printf("[Sweep] P=%d mom l* OUT OF TOLERANCE\r\n", width);
            err++;
        }
    }
    return err;
}

#endif  // XDMA_DST_JCT_ENABLE_PTR

int main() {
    uint8_t chip_id = get_current_chip_id();
    int me = snake_index(chip_id);
    uint32_t tcdm_base = snrt_cluster_base_addrl();

    volatile uint32_t *ready = (volatile uint32_t *)(uintptr_t)(tcdm_base + READY_OFF);
    volatile uint32_t *done = (volatile uint32_t *)(uintptr_t)(tcdm_base + DONE_OFF);

#ifndef XDMA_DST_JCT_ENABLE_PTR
    // This build's xDMA has no writer junctions, so there is no in-fabric fold to measure.
    // Refuse at run time rather than breaking the build of every other configuration.
    if (me == 0 && snrt_is_dm_core()) {
        printf("[Sweep] SKIPPED: this build has no xDMA writer junctions "
               "(XDMA_DST_JCT_ENABLE_PTR undefined); ChainGather is unavailable.\r\n");
    }
    (void)ready;
    (void)done;
    return 0;
#else
    if (me < 0) {
        // Not a chiplet of the 4x4 grid (should not happen on hemaia_16chiplet).
        return 0;
    }

    // Clear this chiplet's own flag slots before anyone can write them. `done` is written by
    // the collector into THIS chip; `ready` is only ever written into the collector.
    if (snrt_is_dm_core()) {
        for (int r = 0; r < NUM_ROUNDS; r++) done[r] = 0;
        if (chip_id == COLLECTOR) {
            for (int r = 0; r < NUM_ROUNDS; r++) {
                for (int i = 0; i < NUM_CHIPLETS; i++) ready[r * NUM_CHIPLETS + i] = 0;
            }
        }
    }
    snrt_global_barrier();  // within-chip; orders the clears before any staging or flag write

    // Every chiplet stages BOTH of its partials at the same local offsets on every chip, so a
    // chain entry is just that chiplet's address prefix plus the offset.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d((void *)(uintptr_t)(tcdm_base + LIN_SRC_OFF),
                          &chain_gather_data[me * PARTIAL_ELEMS], PARTIAL_BYTES);
        snrt_dma_start_1d((void *)(uintptr_t)(tcdm_base + MOM_SRC_OFF),
                          &chain_gather_moment_data[me * PARTIAL_ELEMS], PARTIAL_BYTES);
        snrt_dma_wait_all();
    }
    snrt_global_barrier();

    if (!snrt_is_dm_core()) {
        snrt_global_barrier();  // compute cores: just join the exit barrier
        return 0;
    }

    int err = 0;
    uint32_t task_cyc[NUM_ROUNDS];
    uint32_t wall_cyc[NUM_ROUNDS];
    int round_err[NUM_ROUNDS];
    for (int r = 0; r < NUM_ROUNDS; r++) {
        task_cyc[r] = 0;
        wall_cyc[r] = 0;
        round_err[r] = 0;
    }

    for (int mode = 0; mode < NUM_MODES; mode++) {
        for (int pi = 0; pi < NUM_SWEEP_P; pi++) {
            const int width = SWEEP_P[pi];
            const int r = mode * NUM_SWEEP_P + pi;
            const int participating = (me < width);

            if (!participating) continue;

            if (chip_id != COLLECTOR) {
                // Announce readiness to the collector, then hold until it has read our partial.
                // Holding matters: a source must not return from main() (and let its chiplet
                // signal EOC) while the collector is still gathering from it.
                xchip_store_u32(COLLECTOR,
                                tcdm_base + READY_OFF + (uint32_t)(r * NUM_CHIPLETS + me) * 4u,
                                FLAG_SET);
                while (done[r] != FLAG_SET) asm volatile("fence" ::: "memory");
                continue;
            }

            // Collector: wait for every source of THIS round.
            for (int s = 1; s < width; s++) {
                while (ready[r * NUM_CHIPLETS + s] != FLAG_SET) {
                    asm volatile("fence" ::: "memory");
                }
            }

            const float *golden = (mode == MODE_LIN)
                                      ? &chain_gather_golden_lin[pi * PARTIAL_ELEMS]
                                      : &chain_gather_golden_mom[pi * PARTIAL_ELEMS];
            round_err[r] = run_round(mode, width, tcdm_base, golden, &task_cyc[r], &wall_cyc[r]);
            err += round_err[r];

            // Release this round's sources.
            for (int s = 1; s < width; s++) {
                xchip_store_u32(SNAKE[s], tcdm_base + DONE_OFF + (uint32_t)r * 4u, FLAG_SET);
            }
        }
    }

    if (chip_id == COLLECTOR) {
        printf("\r\n[Sweep] ChainGather scaling on %d chiplets -- xDMA task cycles\r\n",
               NUM_CHIPLETS);
        printf("[Sweep]   P  fold  task_cc  wall_cc  result\r\n");
        for (int mode = 0; mode < NUM_MODES; mode++) {
            for (int pi = 0; pi < NUM_SWEEP_P; pi++) {
                const int r = mode * NUM_SWEEP_P + pi;
                printf("[Sweep] %3d  %s  %7u  %7u  %s\r\n", SWEEP_P[pi],
                       (mode == MODE_LIN) ? "lin" : "mom", task_cyc[r], wall_cyc[r],
                       round_err[r] ? "FAIL" : "PASS");
            }
        }
        printf(err ? "[Sweep] Check: FAIL (%d)\r\n" : "[Sweep] Check: PASS (%d)\r\n", err);
    }

    snrt_global_barrier();  // within-chip clean exit
    return err;
#endif  // XDMA_DST_JCT_ENABLE_PTR
}
