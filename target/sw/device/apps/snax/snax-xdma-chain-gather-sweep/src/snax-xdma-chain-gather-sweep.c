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
// RTL-simulation budgets, calibrated against MEASURED sim speed, not against silicon.
// A 16-chiplet VCS run advances a DM core at only a few hundred instructions per wall-clock
// second, so a spin budget is a wall-clock budget: 300k spins cost well over an hour per
// stuck round. The fold itself completes in a few hundred CYCLES and a 3-hop cross-chip
// store in ~1k, so 15k spins (~60k+ cycles) is still orders of magnitude of headroom while
// keeping a failing round to a couple of minutes.
#define WAIT_SPINS 15000u
// One budget for the WHOLE round's cross-chip flag wait, not per flag: on failure we want to
// learn which sources are missing immediately, not to pay the timeout once per source. This
// one stays short on purpose -- every partial is staged before the barrier in main(), so a
// source the collector is waiting on should already be there.
#define SYNC_SPINS 15000u
// The FINAL release, on the other hand, has to cover the collector traversing the ENTIRE
// sweep. A chiplet that takes part only in the widest rounds reaches the end of its own loop
// while the collector is still working through the narrow rounds it sits out -- on a 4x4 grid
// at P=2 that is fourteen chiplets idling through every narrow round. Budget per round.
#define RELEASE_SPINS (50000u * (uint32_t)NUM_ROUNDS)
// How long to wait for the folded beat to become visible after the finish counter bumps.
// This is a data-visibility settle, not a completion wait, so it is deliberately short.
#define SETTLE_SPINS 2000u

// Boustrophedon Hamiltonian path over the compute grid, BUILT AT RUNTIME from the generated
// N_CHIPLETS_X / N_CHIPLETS_Y so the same binary is correct on any rectangular platform.
// Row y is walked west-to-east when y is even and east-to-west when y is odd, so every
// consecutive pair differs in exactly one coordinate by one -- and so does every PREFIX,
// which is what lets one table serve every P in the sweep.
//
//   4x4: 0x00 0x10 0x20 0x30 | 0x31 0x21 0x11 0x01 | 0x02 0x12 0x22 0x32 | 0x33 0x23 0x13 0x03
//   2x2: 0x00 0x10 0x11 0x01                (the chain the 4-chiplet bring-up used)
//
// Index 0 is the collector, so a width-P round folds the first P entries.
static uint8_t SNAKE[NUM_CHIPLETS];

static void build_snake(void) {
    int i = 0;
    for (int y = 0; y < N_CHIPLETS_Y; y++) {
        for (int k = 0; k < N_CHIPLETS_X; k++) {
            const int x = (y & 1) ? (N_CHIPLETS_X - 1 - k) : k;
            if (i < NUM_CHIPLETS) SNAKE[i++] = (uint8_t)((x << 4) | y);
        }
    }
}

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

_Static_assert(NUM_CHIPLETS == N_CHIPLETS,
    "the generated data set has NUM_CHIPLETS partials but the platform reports N_CHIPLETS "
    "chiplets; set num_chiplets in the app's params.hjson to match the RTL cfg.");
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

// Wait for a cross-chip sync flag, bounded. Returns 1 when the flag arrived, 0 on timeout.
// Every wait in this app is bounded: check_finish requires ALL chiplets to report a status,
// so one lost flag would otherwise turn into the runner's 4-hour wall-clock timeout with no
// diagnostic. Timing out and reporting which chip/round/slot stalled is far more useful.
static int wait_flag_bounded(volatile uint32_t *flag, uint32_t budget) {
    for (uint32_t s = 0; s < budget; s++) {
        if (*flag == FLAG_SET) return 1;
        asm volatile("fence" ::: "memory");
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
        // MonoidJunction CSR(0) names a GEOMETRY, not an operator (MonoidJunction.scala):
        //   [7:0] nValid | [11:8] n | [21:18] nExp | [25:22] nAdd | [27:26] sigma
        //   [28] keyPol (0=max) | [29] keyMul (0 = the (R,max) key monoid)
        // The online-softmax partial is (m, l): key m plus ONE value coordinate, so n = 1
        // (F = n+1 = 2 fields) and l takes the exp twist, so nExp = 1, nAdd = 0. Lanes are
        // field-major, lane = field*S + slot with S = 1 << sigma, so sigma = 3 (S = 8) puts
        // m at lane 0 and l at lane 8 -- exactly MOMENT_M_LANE / MOMENT_L_LANE. nValid = 1
        // leaves only slot 0 live; every other slot is fed its field's identity.
        //
        // NOT the old StreamMomentMergeRt encoding ((1<<13)|1, combineMode at [15:13]).
        // Under this layout that word decodes to n = 0, sigma = 0 -- a key-only geometry
        // with S = 1 -- so the l value at lane 8 was never read and the fold wrote nothing.
        junction = WRITER_JCT_MONOIDJUNCTION;
        jct_csr = (1u /*nValid*/) | (1u << 8 /*n*/) | (1u << 18 /*nExp*/) |
                  (0u << 22 /*nAdd*/) | (3u << 26 /*sigma=3 -> S=8*/);
    }

    int32_t ret = xdma_chain_gather_1d_full_address(local_src, chain, (uint32_t)n,
                                                    PARTIAL_BYTES, junction, jct_csr);
    if (ret != 0) {
        printf("[Sweep] P=%d %s: cfg FAILED (%d)\r\n", width, tag, ret);
        *task_cycles = 0;
        *wall_cycles = 0;
        return 1;
    }

    // SPURIOUS-FINISH PROBE. If a leftover finish/grant from the PREVIOUS gather is still
    // standing, the finish counter is already at or past the id this transfer is about to be
    // given, so xdma_wait_task returns immediately and the round "completes" in ~26 cycles
    // having moved nothing -- exactly the observed second-gather symptom. Sample the counters
    // BEFORE the start so a stale one is visible.
    uint32_t fl_before = snax_read_xdma_cfg_reg(XDMA_FINISH_LOCAL_TASK_PTR);
    uint32_t fr_before = snax_read_xdma_cfg_reg(XDMA_FINISH_REMOTE_TASK_PTR);
    uint32_t cl_before = snax_read_xdma_cfg_reg(XDMA_COMMIT_LOCAL_TASK_PTR);
    uint32_t cr_before = snax_read_xdma_cfg_reg(XDMA_COMMIT_REMOTE_TASK_PTR);

    uint32_t t0 = snrt_mcycle();
    xdma_task_t task = xdma_start();
    printf("[Sweep] P=%d %s: commit(l=%u r=%u) finish_before(l=%u r=%u) -> task id=%u remote=%d\r\n",
           width, tag, cl_before, cr_before, fl_before, fr_before, task.task_id, task.remote);
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

    // SETTLE BARRIER -- do not delete.
    // The finish counter is NOT a sufficient barrier for the folded beat being visible to
    // this core: it can bump a few cycles before the writer's last store has landed in TCDM.
    // Reading the result immediately then returns the pre-filled sentinel and the round is
    // scored a false MISMATCH. The first gather of a program hides this (it is slow enough
    // that the check loses the race anyway); a warm second gather completes in ~31 cycles and
    // loses it every time -- which is exactly the "only the first gather works" symptom.
    //
    // The old 4-chiplet bring-up app never hit this only because it printf'd twice between
    // the wait and the check, which incidentally gave the write time to land. Depending on a
    // printf for correctness is not a barrier, so wait explicitly: poll until the destination
    // stops reading as the sentinel, bounded so a genuinely dead transfer still reports.
    {
        volatile uint32_t *settle = (volatile uint32_t *)(uintptr_t)(tcdm_base + dst_off);
        for (uint32_t s = 0; s < SETTLE_SPINS; s++) {
            if (settle[0] != 0xDEADBEEFu) break;
            asm volatile("fence" ::: "memory");
        }
        asm volatile("fence" ::: "memory");
    }

    // The junction's own verdict on THIS transfer (XDMA_JCT_STATUS is cleared on writerStart):
    // [0] sticky cfg-error, [1] sticky starved (an operand never arrived), [2] live.
#ifdef XDMA_JCT_STATUS
    {
        uint32_t st = snax_read_xdma_cfg_reg(XDMA_JCT_STATUS);
        printf("[Sweep] P=%d %s: jct_status=%x (cfgerr=%d starved=%d live=%d)\r\n",
               width, tag, st, (int)(st & 1u), (int)((st >> 1) & 1u), (int)((st >> 2) & 1u));
    }
#endif

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
    build_snake();
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

    // EXPERIMENT: iterate modes in reverse (MOMENT first) to separate "the monoid junction is
    // broken" from "any SECOND gather in a program fails" -- mom had only ever run 2nd.
#define MODE_ORDER_REVERSED 0
    for (int mi = 0; mi < NUM_MODES; mi++) {
        const int mode = MODE_ORDER_REVERSED ? (NUM_MODES - 1 - mi) : mi;
        for (int pi = 0; pi < NUM_SWEEP_P; pi++) {
            const int width = SWEEP_P[pi];
            const int r = mode * NUM_SWEEP_P + pi;
            const int participating = (me < width);

            if (!participating) continue;

            if (chip_id != COLLECTOR) {
                // Announce readiness to the collector and move on. Every partial was staged
                // before the barrier in main() and is never modified afterwards, so this is a
                // liveness signal, not a data dependency -- what keeps this chiplet alive
                // while the collector gathers is the SINGLE final release below.
                //
                // It used to hold here, per round, on `done[r]`. That deadlocks against the
                // sweep's own shape: a source announces round r as soon as it reaches it, but
                // the collector may still be several rounds behind, working through widths
                // this source sits out. The wait then expires on the first wide round after a
                // run of narrow ones and the chiplet reports a spurious error, even though
                // every gather passed. Measured on the 2x2 bring-up: snake indices 2 and 3
                // timed out three times each; index 1, which is in every round, never did.
                xchip_store_u32(COLLECTOR,
                                tcdm_base + READY_OFF + (uint32_t)(r * NUM_CHIPLETS + me) * 4u,
                                FLAG_SET);
                // One line per participating round. This is what separates "the source never
                // got here" from "the source ran but its cross-chip store never landed" --
                // the two have identical symptoms at the collector.
                printf("[Sweep] chip %x: ready sent, round %d (P=%d %s)\r\n",
                       chip_id, r, width, (mode == MODE_LIN) ? "lin" : "mom");
                continue;
            }

            // Collector: wait for every source of THIS round under ONE shared budget. A
            // missing source means its partial was never staged, so the fold would read
            // stale TCDM -- report exactly which chips are missing and skip the round
            // rather than measure garbage.
            int sources_ready = 0;
            for (uint32_t spin = 0; spin < SYNC_SPINS; spin++) {
                int missing = 0;
                for (int s = 1; s < width; s++) {
                    if (ready[r * NUM_CHIPLETS + s] != FLAG_SET) missing++;
                }
                if (missing == 0) { sources_ready = 1; break; }
                asm volatile("fence" ::: "memory");
            }
            if (!sources_ready) {
                for (int s = 1; s < width; s++) {
                    if (ready[r * NUM_CHIPLETS + s] != FLAG_SET) {
                        printf("[Sweep] P=%d %s: MISSING ready from chip %x (snake idx %d)\r\n",
                               width, (mode == MODE_LIN) ? "lin" : "mom", SNAKE[s], s);
                    }
                }
            }
            if (!sources_ready) {
                round_err[r] = 1;
                err++;
                continue;
            }

            const float *golden = (mode == MODE_LIN)
                                      ? &chain_gather_golden_lin[pi * PARTIAL_ELEMS]
                                      : &chain_gather_golden_mom[pi * PARTIAL_ELEMS];
            round_err[r] = run_round(mode, width, tcdm_base, golden, &task_cyc[r], &wall_cyc[r]);
            err += round_err[r];
        }
    }

    // FINAL RELEASE. A source must not return from main() -- and let its chiplet signal EOC --
    // while the collector is still gathering from it. One release at the end of the whole
    // sweep is both sufficient (no source ever modifies its partial) and race-free, which a
    // per-round release is not: see the source branch above.
    if (chip_id == COLLECTOR) {
        for (int s = 1; s < NUM_CHIPLETS; s++) {
            xchip_store_u32(SNAKE[s], tcdm_base + DONE_OFF, FLAG_SET);
        }
    } else if (!wait_flag_bounded(&done[0], RELEASE_SPINS)) {
        printf("[Sweep] chip %x: TIMEOUT waiting for the final release\r\n", chip_id);
        err++;
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
