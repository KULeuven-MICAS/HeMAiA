// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Cross-die in-fabric REDUCTION sweep on the HeMAiA chiplet mesh.
//
// Two reduction SHAPES over the same partials, the same junctions and the same goldens, so
// the only variable is the shape:
//
//   CHAIN  one flat width-P ChainGather: P-1 remote sources folded hop by hop into the
//          collector. Cost grows linearly in P.
//   TREE   a two-stage fold: every row folds itself into its own x=0 chiplet IN PARALLEL,
//          then the x=0 column folds those row results into chip 0x00. Cost is
//          chain(RX) + chain(RY) instead of chain(RX*RY).
//
// and two junctions per shape:
//
//   LINEAR  ElementwiseJunction, per-element FP32 ADD  -> byte-exact check.
//   MOMENT  MonoidJunction, the nonlinear online-softmax (m, l) merge
//           (m1,l1) (+) (m2,l2) = (max(m1,m2), l1*exp(m1-m*) + l2*exp(m2-m*))
//           -> m* exact, l* checked to a ULP tolerance (writer exp LUT).
//
// Both junctions are associative AND commutative, which is exactly what licenses the tree:
// re-associating the fold cannot change the answer, so the tree's result is checked against
// the SAME golden as the chain's. That is a real correctness statement, not a convenience.
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
// EVERY HOP IS ONE D2D LINK -- a deliberate convention, NOT a hardware rule. An earlier
// version of this comment claimed the backward grant handshake cannot route between
// non-neighbours. That is wrong. hemaia_d2d_link_router_rc.sv is X-first / Y-second unicast
// with a Y fallback, and its only reachability caveat is that the active chip array be convex
// (a full rectangle is); the grant is an ordinary unicast narrow write to
// get_cluster_end_addr(src) - MMIOGrantOffset and routes like any other packet. Nothing in the
// D2D link or the xDMA software asserts adjacency.
//
// What a multi-hop chain hop really costs is that it drags chiplets which are NOT in the chain
// into the chain's dependency graph, while hemaia_d2d_link_aw_gate_recorder serialises W
// bursts per destination ("not multiple W transactions from different src ... to the same dst
// at the same time") and the write-aware router keeps a single serialised soc->D2D stream.
// That is the machinery behind the P>=3 deadlock root-caused during bring-up -- a middle node
// issuing its data-forward AW before its W data existed, locking an output with no W to
// release it (fixed by gating that AW on w_valid). A longer hop holds more order-sensitive
// shared resources across more chips, so confining every hop to one link keeps each hop's
// AW/W/grant local and the cyclic-wait surface as small as it can be.
//
// SNAKE[] below is a boustrophedon Hamiltonian path over the grid that starts at the
// collector, so every consecutive pair is a real N/S/E/W neighbour -- and, crucially, every
// PREFIX of it is itself a single-link chain. That prefix property is what lets one table and
// one data set serve every P in the sweep.
//
// The chiplet at snake index i owns partial i, so a width-P round folds partials [0 .. P-1]
// and the golden for that round is the datagen's fold over the same prefix.
//
// WHY THE TREE IS ROWS-THEN-COLUMN, AND NOT A BINARY TREE.  The single-link convention above
// binds BOTH stages, and rows and columns are the two families of straight lines a mesh has. A
// balanced binary tree over the snake would pair segments whose collectors are far apart -- at
// P=16, G=4 they are 0x00, 0x31, 0x02, 0x33 -- so every stage-2 hop would span several links.
// The router would carry that; what it gives up is the containment argument above. It is also
// not obviously faster: a binary tree's hops span 1, 2, 4, 8 links, so the per-hop cost does
// not stay at one link's, whereas rows-then-column minimises total hop distance -- the usual
// reason it is the standard all-reduce shape on a mesh. Worth adding a `btree` shape and
// letting the measurement settle it rather than the argument.
//
//   stage 1   row y is the contiguous snake run [y*RX .. y*RX+RX-1]; folding it toward x=0 is
//             the walk (RX-1, y) -> ... -> (1, y) -> (0, y), every step one hop east-to-west.
//   stage 2   the row results sit on the x=0 column, and (0, RY-1) -> ... -> (0, 1) -> (0, 0)
//             is every step one hop south-to-north.
//
// The snake alternates each row's direction, but the row fold does not care: it always walks
// x downwards, which is all-adjacent either way. The set of chiplets in RY full rows is
// exactly the snake prefix of length RX*RY, so chain and tree fold the SAME partials.
//
// A tree therefore exists only when the width covers at least two FULL rows. On the 4x4 grid
// that is P=8 and P=16; P=2 and P=4 are a single (partial) row -- four collinear chiplets have
// no second dimension to fold along, and the tree degenerates to the chain. On the 2x2 grid
// P=4 is two full rows and does have a tree. Widths with no tree are reported "n/a", not
// failed.
//
// SYNCHRONISATION FOR THE TREE -- a SUBMISSION barrier, not a completion barrier.  Stage 2
// does NOT wait for stage 1 to finish. The one ordering guarantee it does need is:
//
//     every stage-2 participant must already have its own stage-1 task SUBMITTED to its own
//     xDMA queue by the time the stage-2 cfg reaches it.
//
// Then the queue does the completion ordering for free: a stage-2 cfg arriving at (0,y) queues
// behind that node's stage-1 task, and a stage-1 task retires only once its row result has
// landed in that node's own TCDM. Measured at snax level: with the straggler still programmed
// before stage 2 is issued a 1600-cycle skew (>9x a whole row fold) passes every round; with
// it programmed after, it fails at ZERO skew, folding the sentinel. So each row collector sets
// a "submitted" flag right after xdma_start(), and chip 0x00 waits for those flags -- one flag
// round-trip -- before issuing stage 2. Waiting for COMPLETION instead would throw away the
// overlap and most of the speed-up. D2D lengthens cfg-arrival times relative to task times,
// which WIDENS the inversion window, so the flag matters more here than on the snax bench.
//
// CHAIN LENGTH.  A width-P gather programs P xDMA dst slots (P-1 remote sources + the
// collector's own dst buffer), so P == NUM_CHIPLETS == 16 sits exactly at the hardware cap
// XDMA_MAX_DST_COUNT. The static assert below fails the build rather than letting
// xdma_multicast_1d_full_address() reject the round at run time.
//
// CROSS-CHIP SYNC.  snrt_chip_global_barrier() announces via a 0xFF-BROADCAST scalar store,
// whose routing is not exercised on this platform and which hangs here. We instead use
// TARGETED cross-chip scalar stores (the proven direction) plus local polling. Each round owns
// its own flag slots, so no round ever has to clear another's flags and there is no clear/set
// race between rounds.

#include "chip_id.h"  // get_current_chip_id, get_chip_baseaddress_value, chiplet_addr_transform_full
#include "data.h"
#include "snrt.h"

#define PARTIAL_BYTES (PARTIAL_ELEMS * (int)sizeof(float))

// The collector is the chiplet at snake index 0 -- chip 0x00, the boot chip. It is also the
// tree's row-0 collector and its final root.
#define COLLECTOR 0x00

// Two junctions measured per width.
#define MODE_LIN 0
#define MODE_MOM 1
#define NUM_MODES 2

// Two reduction shapes measured per (width, junction).
#define ALGO_CHAIN 0
#define ALGO_TREE 1
#define NUM_ALGOS 2

#define NUM_ROUNDS (NUM_ALGOS * NUM_MODES * NUM_SWEEP_P)

// TCDM map. Eight 64 B data beats first, then the per-round row slots, then the flag tables.
// Everything after the beats is sized from NUM_ROUNDS / NUM_CHIPLETS so the regions cannot
// silently overlap when the sweep grows -- an earlier layout had READY running straight
// through DONE and only escaped by luck, because the aliased slot happened to be one nobody
// ever wrote.
#define BEAT_BYTES 0x40u
#define LIN_SRC_OFF 0x000u   // this chiplet's linear partial
#define LIN_DST_OFF 0x040u   // collector's linear CHAIN result
#define MOM_SRC_OFF 0x080u   // this chiplet's (m, l) beat
#define MOM_DST_OFF 0x0C0u   // collector's merged (m*, l*) CHAIN result
#define LIN_TREE_OFF 0x100u  // collector's linear TREE result
#define MOM_TREE_OFF 0x140u  // collector's merged TREE result

// Tree stage-1 output, one slot PER ROUND on every row collector. Per-round slots mean the
// DATA cannot race even though the flags are only loosely coupled: chip 0x00 reads row slot r
// while the row collector may already be writing round r+1, and those are different slots.
// (The s2done flag below is a separate concern -- it protects the xDMA CONFIG plane, not this
// buffer.)
#define ROW_OFF 0x200u
#define ROW_WORDS (NUM_ROUNDS * (int)(BEAT_BYTES / 4u))

// ready[round][snake_idx]: written cross-chip by a source INTO the node that gathers from it
// -- chip 0x00 for a chain round, the row's x=0 chiplet for a tree round.
#define READY_OFF (ROW_OFF + (uint32_t)ROW_WORDS * 4u)
#define READY_WORDS (NUM_ROUNDS * NUM_CHIPLETS)
// submitted[round][row]: written cross-chip by a row collector INTO chip 0x00 once its stage-1
// task is in its own xDMA queue. This is the submission barrier described in the header.
#define SUBMIT_OFF (READY_OFF + (uint32_t)READY_WORDS * 4u)
#define SUBMIT_WORDS (NUM_ROUNDS * NUM_CHIPLETS)
// s2done[round]: written cross-chip by chip 0x00 into every row collector once stage 2 has
// retired. A row collector must not reprogram its xDMA for the next round while stage 2 of
// this one is still FORWARDING through it -- it is a middle hop for stage 2 immediately after
// having been the collector of stage 1, which is exactly the role change that a stranded AW
// descriptor in the adapter used to break. Unlike the per-round release the chain sources used
// to hold on, this one cannot deadlock against the sweep's shape: 0x00 waits for every row's
// submission each round, so a row collector is never more than about one round ahead of it.
#define S2DONE_OFF (SUBMIT_OFF + (uint32_t)SUBMIT_WORDS * 4u)
#define S2DONE_WORDS NUM_ROUNDS
// One final release, broadcast by the collector when the whole sweep is done.
#define DONE_OFF (S2DONE_OFF + (uint32_t)S2DONE_WORDS * 4u)
#define SYNC_END (DONE_OFF + 4u)
#define FLAG_SET 0xA5A5A5A5u

// fp32 ULP tolerance for l* (exp-LUT approximation). ~0x20000 ULPs ~= 1.5% at l* ~ 1.19,
// matching the Gate-A moment-merge acceptance bound. The measured delta is printed so the
// bound can be tightened once the sweep has run; measured values so far are 7-27 ULP, four
// orders of magnitude inside this, so it is currently far too loose to catch a regression.
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
// The submission barrier, on the other hand, may have to wait for a row collector that is
// still working through EARLIER rounds, so it gets the per-round budget.
#define SUBMIT_SPINS (20000u * (uint32_t)NUM_ROUNDS)
// The FINAL release has to cover the collector traversing the ENTIRE sweep. A chiplet that
// takes part only in the widest rounds reaches the end of its own loop while the collector is
// still working through the narrow rounds it sits out -- on a 4x4 grid at P=2 that is fourteen
// chiplets idling through every narrow round. Budget per round.
#define RELEASE_SPINS (50000u * (uint32_t)NUM_ROUNDS)
// How long to wait for a folded beat to become visible after the finish counter bumps.
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

// chip_id is (x << 4) | y throughout HeMAiA (chip_id.h).
static inline int chip_x(uint8_t c) { return (int)(c >> 4); }
static inline int chip_y(uint8_t c) { return (int)(c & 0x0Fu); }
static inline uint8_t chip_at(int x, int y) { return (uint8_t)(((x & 0xF) << 4) | (y & 0xF)); }

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
_Static_assert(SYNC_END <= 0x4000u,
    "the sync tables have grown past 16 KiB of TCDM; shrink sweep_p or move them.");
_Static_assert(LIN_TREE_OFF + BEAT_BYTES <= MOM_TREE_OFF &&
                   MOM_TREE_OFF + BEAT_BYTES <= ROW_OFF,
    "the data beats overlap the per-round row slots.");

// How many FULL rows a width-P tree spans, or 0 if this width has no tree.
// A tree needs at least two full rows: one row is a straight line with no second dimension to
// fold along, and a partial row would make stage 1 fold a set the golden does not describe.
static int tree_rows(int width) {
    if (N_CHIPLETS_X <= 1) return 0;
    if (width % N_CHIPLETS_X != 0) return 0;
    const int ry = width / N_CHIPLETS_X;
    if (ry < 2 || ry > N_CHIPLETS_Y) return 0;
    return ry;
}

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

// No reader/writer extensions and no stale junction from the last round.
static void clear_xdma_plugins(void) {
    for (uint8_t e = 0; e < XDMA_DST_EXT_NUM; e++) xdma_disable_dst_ext(e);
    for (uint8_t r = 0; r < XDMA_SRC_EXT_NUM; r++) xdma_disable_src_ext(r);
    for (uint8_t j = 0; j < XDMA_DST_JCT_NUM; j++) xdma_disable_dst_junction(j);
}

// The junction and its CSR(0) for a mode. Both tree stages use the SAME junction the chain
// does -- re-associating an associative fold does not change the operator.
static void junction_for_mode(int mode, uint8_t *junction, uint32_t *jct_csr) {
    if (mode == MODE_LIN) {
        // ElementwiseJunction CSR(0): [3:0] op = ADD(0), [6:4] fmt = FP32(3).
        *junction = WRITER_JCT_ELEMENTWISEJUNCTION;
        *jct_csr = (3u << 4) | 0u;
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
        *junction = WRITER_JCT_MONOIDJUNCTION;
        *jct_csr = (1u /*nValid*/) | (1u << 8 /*n*/) | (1u << 18 /*nExp*/) |
                   (0u << 22 /*nAdd*/) | (3u << 26 /*sigma=3 -> S=8*/);
    }
}

// Fill a beat with the sentinel so "the writer never wrote" is distinguishable from "it wrote
// the wrong value". Integer stores only -- the DM core is rv32ima with no FPU.
static void sentinel_fill(uint32_t addr) {
    volatile uint32_t *p = (volatile uint32_t *)(uintptr_t)addr;
    for (int j = 0; j < PARTIAL_ELEMS; j++) p[j] = 0xDEADBEEFu;
}

// SETTLE BARRIER -- do not delete.
// The finish counter is NOT a sufficient barrier for the folded beat being visible to this
// core: it can bump a few cycles before the writer's last store has landed in TCDM. Reading
// the result immediately then returns the pre-filled sentinel and the round is scored a false
// MISMATCH. The first gather of a program hides this (it is slow enough that the check loses
// the race anyway); a warm second gather completes in ~31 cycles and loses it every time --
// which is exactly the "only the first gather works" symptom.
//
// The old 4-chiplet bring-up app never hit this only because it printf'd twice between the
// wait and the check, which incidentally gave the write time to land. Depending on a printf
// for correctness is not a barrier, so wait explicitly: poll until the destination stops
// reading as the sentinel, bounded so a genuinely dead transfer still reports.
static void settle_dst(uint32_t addr) {
    volatile uint32_t *settle = (volatile uint32_t *)(uintptr_t)addr;
    for (uint32_t s = 0; s < SETTLE_SPINS; s++) {
        if (settle[0] != 0xDEADBEEFu) break;
        asm volatile("fence" ::: "memory");
    }
    asm volatile("fence" ::: "memory");
}

// The junction's own verdict on the LAST transfer (XDMA_JCT_STATUS is cleared on writerStart):
// [0] sticky cfg-error, [1] sticky starved (an operand never arrived), [2] live.
// NOTE a zero here is NOT evidence of health on a failing round: both sticky bits are cleared
// by the writer-start pulse that even a dead round issues, and neither can fire on a round
// that never forms a chain. Gate every verdict on the DATA.
static void print_jct_status(const char *algo, int width, const char *tag) {
#ifdef XDMA_JCT_STATUS
    uint32_t st = snax_read_xdma_cfg_reg(XDMA_JCT_STATUS);
    printf("[Sweep] %s P=%d %s: jct_status=%x (cfgerr=%d starved=%d live=%d)\r\n", algo, width,
           tag, st, (int)(st & 1u), (int)((st >> 1) & 1u), (int)((st >> 2) & 1u));
#else
    (void)algo;
    (void)width;
    (void)tag;
#endif
}

// SPURIOUS-FINISH PROBE. If a leftover finish/grant from the PREVIOUS gather is still standing,
// the finish counter is already at or past the id this transfer is about to be given, so
// xdma_wait_task returns immediately and the round "completes" in ~26 cycles having moved
// nothing -- exactly the second-gather symptom that three RTL defects produced in turn.
// finish_before MUST equal commit on every round; it is visible a whole round before the data
// check fails, because xdma_wait_task_bounded() compares with >=.
static xdma_task_t start_and_probe(const char *algo, int width, const char *tag) {
    uint32_t fl = snax_read_xdma_cfg_reg(XDMA_FINISH_LOCAL_TASK_PTR);
    uint32_t fr = snax_read_xdma_cfg_reg(XDMA_FINISH_REMOTE_TASK_PTR);
    uint32_t cl = snax_read_xdma_cfg_reg(XDMA_COMMIT_LOCAL_TASK_PTR);
    uint32_t cr = snax_read_xdma_cfg_reg(XDMA_COMMIT_REMOTE_TASK_PTR);
    xdma_task_t task = xdma_start();
    printf("[Sweep] %s P=%d %s: commit(l=%u r=%u) finish_before(l=%u r=%u) -> task id=%u "
           "remote=%d\r\n",
           algo, width, tag, cl, cr, fl, fr, task.task_id, task.remote);
    return task;
}

// Compare the folded beat at `addr` with `golden`. Returns the number of errors.
static int check_result(int mode, int width, const char *algo, uint32_t addr,
                        const float *golden) {
    // Compare raw fp32 BIT PATTERNS as integers: the DM core is rv32ima with NO FPU, so a
    // float compare would emit flw and trap as an illegal instruction.
    volatile uint32_t *res = (volatile uint32_t *)(uintptr_t)addr;
    const uint32_t *gold = (const uint32_t *)golden;
    int err = 0;

    if (mode == MODE_LIN) {
        // Small integer-valued fp32 operands -> the sum is exact -> byte-exact compare.
        for (int j = 0; j < PARTIAL_ELEMS; j++) {
            if (res[j] != gold[j]) {
                if (err < 4) {
                    printf("[Sweep] %s P=%d lin MISMATCH at %d: got %x exp %x\r\n", algo, width,
                           j, res[j], gold[j]);
                }
                err++;
            }
        }
    } else {
        // m* is a max, so exact; l* goes through the writer's exp LUT, so ULP-bounded.
        uint32_t m_got = res[MOMENT_M_LANE], m_exp = gold[MOMENT_M_LANE];
        uint32_t l_got = res[MOMENT_L_LANE], l_exp = gold[MOMENT_L_LANE];
        uint32_t l_ulp = (l_got > l_exp) ? (l_got - l_exp) : (l_exp - l_got);
        printf("[Sweep] %s P=%d mom m*: got %x exp %x | l*: got %x exp %x (ulp=%x tol=%x)\r\n",
               algo, width, m_got, m_exp, l_got, l_exp, l_ulp, L_TOL_ULPS);
        if (m_got != m_exp) {
            printf("[Sweep] %s P=%d mom m* MISMATCH\r\n", algo, width);
            err++;
        }
        if (l_ulp > L_TOL_ULPS) {
            printf("[Sweep] %s P=%d mom l* OUT OF TOLERANCE\r\n", algo, width);
            err++;
        }
    }
    return err;
}

//======================================================================================
// CHAIN: one flat width-P gather into the collector.
//======================================================================================
static int run_chain_round(int mode, int width, uint32_t tcdm_base, const float *golden,
                           uint32_t *task_cycles, uint32_t *wall_cycles) {
    const uint32_t src_off = (mode == MODE_LIN) ? LIN_SRC_OFF : MOM_SRC_OFF;
    const uint32_t dst_off = (mode == MODE_LIN) ? LIN_DST_OFF : MOM_DST_OFF;
    const char *tag = (mode == MODE_LIN) ? "lin" : "mom";

    clear_xdma_plugins();

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

    sentinel_fill(tcdm_base + dst_off);

    uint8_t junction;
    uint32_t jct_csr;
    junction_for_mode(mode, &junction, &jct_csr);

    if (xdma_chain_gather_1d_full_address(local_src, chain, (uint32_t)n, PARTIAL_BYTES, junction,
                                          jct_csr) != 0) {
        printf("[Sweep] chain P=%d %s: cfg FAILED\r\n", width, tag);
        *task_cycles = 0;
        *wall_cycles = 0;
        return 1;
    }

    uint32_t t0 = snrt_mcycle();
    xdma_task_t task = start_and_probe("chain", width, tag);
    int done = xdma_wait_task_bounded(task);
    uint32_t t1 = snrt_mcycle();

    if (!done) {
        printf("[Sweep] chain P=%d %s: TIMEOUT after %u spins (remote=%d id=%u)\r\n", width, tag,
               WAIT_SPINS, task.remote, task.task_id);
        *task_cycles = 0;
        *wall_cycles = t1 - t0;
        return 1;
    }

    *task_cycles = xdma_last_task_cycle();
    *wall_cycles = t1 - t0;

    settle_dst(tcdm_base + dst_off);
    print_jct_status("chain", width, tag);
    return check_result(mode, width, "chain", tcdm_base + dst_off, golden);
}

//======================================================================================
// TREE stage 1: fold row `y` into its own x=0 chiplet. Run by every row collector, in
// parallel across rows. Returns 0 and the started task on success.
//======================================================================================
static int tree_start_row_fold(int mode, int y, int r, uint32_t tcdm_base, xdma_task_t *task) {
    const uint32_t src_off = (mode == MODE_LIN) ? LIN_SRC_OFF : MOM_SRC_OFF;
    const uint32_t row_off = ROW_OFF + (uint32_t)r * BEAT_BYTES;
    const char *tag = (mode == MODE_LIN) ? "lin" : "mom";
    const uint8_t self = chip_at(0, y);

    clear_xdma_plugins();
    sentinel_fill(tcdm_base + row_off);

    // Walk the row east-to-west: (RX-1,y) -> ... -> (1,y) -> dst at (0,y). Every step is one
    // hop, whichever way the snake happens to run through this row.
    uint64_t local_src = chiplet_addr_transform_full(self, (uint64_t)(tcdm_base + src_off));
    uint64_t chain[N_CHIPLETS_X];
    int n = 0;
    for (int x = N_CHIPLETS_X - 1; x >= 1; x--) {
        chain[n++] = chiplet_addr_transform_full(chip_at(x, y), (uint64_t)(tcdm_base + src_off));
    }
    chain[n++] = chiplet_addr_transform_full(self, (uint64_t)(tcdm_base + row_off));

    uint8_t junction;
    uint32_t jct_csr;
    junction_for_mode(mode, &junction, &jct_csr);

    if (xdma_chain_gather_1d_full_address(local_src, chain, (uint32_t)n, PARTIAL_BYTES, junction,
                                          jct_csr) != 0) {
        printf("[Sweep] tree row %d %s: stage-1 cfg FAILED\r\n", y, tag);
        return -1;
    }
    // The width printed here is the ROW width -- this probe is one row fold, not the
    // whole tree round.
    *task = start_and_probe("tree-s1(row)", N_CHIPLETS_X, tag);
    return 0;
}

//======================================================================================
// TREE stage 2: fold the x=0 column's row results into chip 0x00. Run by 0x00 only, and only
// once every row collector has flagged that its stage-1 task is submitted.
//======================================================================================
static int tree_start_column_fold(int mode, int ry, int r, uint32_t tcdm_base,
                                  xdma_task_t *task) {
    const uint32_t row_off = ROW_OFF + (uint32_t)r * BEAT_BYTES;
    const uint32_t tree_off = (mode == MODE_LIN) ? LIN_TREE_OFF : MOM_TREE_OFF;
    const char *tag = (mode == MODE_LIN) ? "lin" : "mom";

    clear_xdma_plugins();
    sentinel_fill(tcdm_base + tree_off);

    // (0,RY-1) -> ... -> (0,1) -> dst at (0,0). Every step is one hop north.
    uint64_t local_src = chiplet_addr_transform_full(COLLECTOR, (uint64_t)(tcdm_base + row_off));
    uint64_t chain[N_CHIPLETS_Y];
    int n = 0;
    for (int y = ry - 1; y >= 1; y--) {
        chain[n++] = chiplet_addr_transform_full(chip_at(0, y), (uint64_t)(tcdm_base + row_off));
    }
    chain[n++] = chiplet_addr_transform_full(COLLECTOR, (uint64_t)(tcdm_base + tree_off));

    uint8_t junction;
    uint32_t jct_csr;
    junction_for_mode(mode, &junction, &jct_csr);

    if (xdma_chain_gather_1d_full_address(local_src, chain, (uint32_t)n, PARTIAL_BYTES, junction,
                                          jct_csr) != 0) {
        printf("[Sweep] tree %s: stage-2 cfg FAILED\r\n", tag);
        return -1;
    }
    // ...and here it is the COLUMN height, for the same reason.
    *task = start_and_probe("tree-s2(col)", ry, tag);
    return 0;
}

#endif  // XDMA_DST_JCT_ENABLE_PTR

int main() {
    build_snake();
    uint8_t chip_id = get_current_chip_id();
    int me = snake_index(chip_id);
    uint32_t tcdm_base = snrt_cluster_base_addrl();

    volatile uint32_t *ready = (volatile uint32_t *)(uintptr_t)(tcdm_base + READY_OFF);
    volatile uint32_t *submitted = (volatile uint32_t *)(uintptr_t)(tcdm_base + SUBMIT_OFF);
    volatile uint32_t *s2done = (volatile uint32_t *)(uintptr_t)(tcdm_base + S2DONE_OFF);
    volatile uint32_t *done = (volatile uint32_t *)(uintptr_t)(tcdm_base + DONE_OFF);

#ifndef XDMA_DST_JCT_ENABLE_PTR
    // This build's xDMA has no writer junctions, so there is no in-fabric fold to measure.
    // Refuse at run time rather than breaking the build of every other configuration.
    if (me == 0 && snrt_is_dm_core()) {
        printf("[Sweep] SKIPPED: this build has no xDMA writer junctions "
               "(XDMA_DST_JCT_ENABLE_PTR undefined); ChainGather is unavailable.\r\n");
    }
    (void)ready;
    (void)submitted;
    (void)s2done;
    (void)done;
    return 0;
#else
    if (me < 0) {
        // Not a chiplet of the compute grid.
        return 0;
    }
    const int my_x = chip_x(chip_id);
    const int my_y = chip_y(chip_id);

    // Clear this chiplet's own flag slots before anyone can write them. EVERY chip clears the
    // whole READY table now, not just chip 0: in a tree round a source announces to its ROW
    // collector, so any x=0 chiplet can be a receiver.
    if (snrt_is_dm_core()) {
        *done = 0;
        for (int i = 0; i < READY_WORDS; i++) ready[i] = 0;
        for (int i = 0; i < SUBMIT_WORDS; i++) submitted[i] = 0;
        for (int i = 0; i < S2DONE_WORDS; i++) s2done[i] = 0;
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
    int round_na[NUM_ROUNDS];
    for (int r = 0; r < NUM_ROUNDS; r++) {
        task_cyc[r] = 0;
        wall_cyc[r] = 0;
        round_err[r] = 0;
        round_na[r] = 0;
    }

    for (int ai = 0; ai < NUM_ALGOS; ai++) {
        for (int mode = 0; mode < NUM_MODES; mode++) {
            for (int pi = 0; pi < NUM_SWEEP_P; pi++) {
                const int width = SWEEP_P[pi];
                const int r = (ai * NUM_MODES + mode) * NUM_SWEEP_P + pi;
                const char *tag = (mode == MODE_LIN) ? "lin" : "mom";
                const float *golden = (mode == MODE_LIN)
                                          ? &chain_gather_golden_lin[pi * PARTIAL_ELEMS]
                                          : &chain_gather_golden_mom[pi * PARTIAL_ELEMS];

                //==========================================================================
                // CHAIN
                //==========================================================================
                if (ai == ALGO_CHAIN) {
                    if (me >= width) continue;

                    if (chip_id != COLLECTOR) {
                        // Announce readiness to the collector and move on. Every partial was
                        // staged before the barrier above and is never modified afterwards, so
                        // this is a liveness signal, not a data dependency -- what keeps this
                        // chiplet alive while the collector gathers is the SINGLE final
                        // release at the end.
                        //
                        // It used to hold here, per round, on a per-round release flag. That
                        // deadlocks against the sweep's own shape: a source announces round r
                        // as soon as it reaches it, but the collector may still be several
                        // rounds behind, working through widths this source sits out. The wait
                        // then expires on the first wide round after a run of narrow ones and
                        // the chiplet reports a spurious error even though every gather passed.
                        // Measured on the 2x2 bring-up: snake indices 2 and 3 timed out three
                        // times each; index 1, which is in every round, never did.
                        xchip_store_u32(
                            COLLECTOR,
                            tcdm_base + READY_OFF + (uint32_t)(r * NUM_CHIPLETS + me) * 4u,
                            FLAG_SET);
                        continue;
                    }

                    // Collector: wait for every source of THIS round under ONE shared budget.
                    // A missing source means its partial was never staged, so the fold would
                    // read stale TCDM -- report exactly which chips are missing and skip the
                    // round rather than measure garbage.
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
                                printf("[Sweep] chain P=%d %s: MISSING ready from chip %x "
                                       "(snake idx %d)\r\n",
                                       width, tag, SNAKE[s], s);
                            }
                        }
                        round_err[r] = 1;
                        err++;
                        continue;
                    }

                    round_err[r] = run_chain_round(mode, width, tcdm_base, golden, &task_cyc[r],
                                                   &wall_cyc[r]);
                    err += round_err[r];
                    continue;
                }

                //==========================================================================
                // TREE
                //==========================================================================
                const int ry = tree_rows(width);
                if (ry == 0) {
                    // Fewer than two full rows: no second dimension to fold along, so the tree
                    // IS the chain. Report n/a rather than a duplicate measurement.
                    round_na[r] = 1;
                    continue;
                }
                if (my_y >= ry) continue;  // this chiplet's row is not in the tree at this width

                if (my_x != 0) {
                    // Row member: announce to MY ROW COLLECTOR, not to chip 0. The row
                    // collector is the node that gathers this partial in stage 1.
                    xchip_store_u32(chip_at(0, my_y),
                                    tcdm_base + READY_OFF +
                                        (uint32_t)(r * NUM_CHIPLETS + me) * 4u,
                                    FLAG_SET);
                    continue;
                }

                // ---- Row collector (x == 0) ----
                int row_ready = 0;
                for (uint32_t spin = 0; spin < SYNC_SPINS; spin++) {
                    int missing = 0;
                    for (int x = 1; x < N_CHIPLETS_X; x++) {
                        const int idx = snake_index(chip_at(x, my_y));
                        if (idx < 0 || ready[r * NUM_CHIPLETS + idx] != FLAG_SET) missing++;
                    }
                    if (missing == 0) { row_ready = 1; break; }
                    asm volatile("fence" ::: "memory");
                }
                if (!row_ready) {
                    printf("[Sweep] tree P=%d %s: row %d MISSING a member's ready\r\n", width,
                           tag, my_y);
                    if (chip_id == COLLECTOR) {
                        // The root cannot form the tree; release the rows so they do not
                        // strand behind a stage-2 flag that will never be sent.
                        for (int y = 1; y < ry; y++) {
                            xchip_store_u32(chip_at(0, y),
                                            tcdm_base + S2DONE_OFF + (uint32_t)r * 4u, FLAG_SET);
                        }
                        round_err[r] = 1;
                    }
                    err++;
                    continue;
                }

                uint32_t t0 = snrt_mcycle();
                xdma_task_t s1;
                if (tree_start_row_fold(mode, my_y, r, tcdm_base, &s1) != 0) {
                    if (chip_id == COLLECTOR) round_err[r] = 1;
                    err++;
                    continue;
                }

                // SUBMISSION barrier, not a completion barrier: flag as soon as the stage-1
                // task is in this node's own xDMA queue. Stage 2 may then arrive at any time
                // and will queue behind it.
                if (chip_id != COLLECTOR) {
                    xchip_store_u32(COLLECTOR,
                                    tcdm_base + SUBMIT_OFF +
                                        (uint32_t)(r * NUM_CHIPLETS + my_y) * 4u,
                                    FLAG_SET);
                    // A row collector still has to retire its own stage-1 before it may reuse
                    // its xDMA, and must not leave main() while 0x00 is still gathering. Its
                    // row slot for THIS round is never rewritten, so no release handshake is
                    // needed -- the next round writes a different slot.
                    if (!xdma_wait_task_bounded(s1)) {
                        printf("[Sweep] tree P=%d %s: row %d stage-1 TIMEOUT (id=%u)\r\n", width,
                               tag, my_y, s1.task_id);
                        err++;
                    }
                    // Hold until stage 2 has finished forwarding through this node, so the
                    // next round's programming cannot land on top of an in-flight forward.
                    if (!wait_flag_bounded(&s2done[r], SUBMIT_SPINS)) {
                        printf("[Sweep] tree P=%d %s: row %d never saw stage-2 complete\r\n",
                               width, tag, my_y);
                        err++;
                    }
                    continue;
                }

                // ---- chip 0x00: also the tree's root ----
                // Wait for every OTHER row collector to have submitted, while our own stage 1
                // is still in flight. This is the whole synchronisation cost of the tree.
                int all_submitted = 1;
                for (int y = 1; y < ry; y++) {
                    if (!wait_flag_bounded(&submitted[r * NUM_CHIPLETS + y], SUBMIT_SPINS)) {
                        printf("[Sweep] tree P=%d %s: row %d never reported its stage-1 "
                               "submission\r\n",
                               width, tag, y);
                        all_submitted = 0;
                    }
                }
                if (!all_submitted) {
                    for (int y = 1; y < ry; y++) {
                        xchip_store_u32(chip_at(0, y),
                                        tcdm_base + S2DONE_OFF + (uint32_t)r * 4u, FLAG_SET);
                    }
                    round_err[r] = 1;
                    err++;
                    continue;
                }

                // Our own stage 1 must retire before stage 2 can start anyway -- 0x00 issues
                // both, so its own queue orders them -- and reading the counter here is the
                // only way to attribute the two stages separately.
                if (!xdma_wait_task_bounded(s1)) {
                    printf("[Sweep] tree P=%d %s: stage-1 TIMEOUT at the root (id=%u)\r\n", width,
                           tag, s1.task_id);
                    for (int y = 1; y < ry; y++) {
                        xchip_store_u32(chip_at(0, y),
                                        tcdm_base + S2DONE_OFF + (uint32_t)r * 4u, FLAG_SET);
                    }
                    round_err[r] = 1;
                    err++;
                    continue;
                }
                const uint32_t s1_cc = xdma_last_task_cycle();
                settle_dst(tcdm_base + ROW_OFF + (uint32_t)r * BEAT_BYTES);

                xdma_task_t s2;
                if (tree_start_column_fold(mode, ry, r, tcdm_base, &s2) != 0) {
                    for (int y = 1; y < ry; y++) {
                        xchip_store_u32(chip_at(0, y),
                                        tcdm_base + S2DONE_OFF + (uint32_t)r * 4u, FLAG_SET);
                    }
                    round_err[r] = 1;
                    err++;
                    continue;
                }
                int s2_done = xdma_wait_task_bounded(s2);
                uint32_t t1 = snrt_mcycle();
                // Release the row collectors on EVERY path out of here, including the failing
                // one: a round that wedges must not also strand every other chiplet.
                for (int y = 1; y < ry; y++) {
                    xchip_store_u32(chip_at(0, y),
                                    tcdm_base + S2DONE_OFF + (uint32_t)r * 4u, FLAG_SET);
                }
                if (!s2_done) {
                    printf("[Sweep] tree P=%d %s: stage-2 TIMEOUT (id=%u)\r\n", width, tag,
                           s2.task_id);
                    task_cyc[r] = 0;
                    wall_cyc[r] = t1 - t0;
                    round_err[r] = 1;
                    err++;
                    continue;
                }
                const uint32_t s2_cc = xdma_last_task_cycle();

                // The tree's fabric cost is the sum of the two stages' xDMA task latencies:
                // rows fold in parallel, so one row fold plus the column fold is the critical
                // path. wall_cc additionally carries the SW programming and the submission
                // barrier.
                task_cyc[r] = s1_cc + s2_cc;
                wall_cyc[r] = t1 - t0;
                printf("[Sweep] tree P=%d %s: %dx%d, stage1=%u stage2=%u sum=%u\r\n", width, tag,
                       N_CHIPLETS_X, ry, s1_cc, s2_cc, task_cyc[r]);

                const uint32_t tree_off = (mode == MODE_LIN) ? LIN_TREE_OFF : MOM_TREE_OFF;
                settle_dst(tcdm_base + tree_off);
                print_jct_status("tree", width, tag);
                round_err[r] = check_result(mode, width, "tree", tcdm_base + tree_off, golden);
                err += round_err[r];
            }
        }
    }

    // FINAL RELEASE. A source must not return from main() -- and let its chiplet signal EOC --
    // while the collector is still gathering from it. One release at the end of the whole
    // sweep is both sufficient (no source ever modifies a buffer another round still reads)
    // and race-free, which a per-round release is not: see the chain source branch above.
    if (chip_id == COLLECTOR) {
        for (int s = 1; s < NUM_CHIPLETS; s++) {
            xchip_store_u32(SNAKE[s], tcdm_base + DONE_OFF, FLAG_SET);
        }
    } else if (!wait_flag_bounded(done, RELEASE_SPINS)) {
        printf("[Sweep] chip %x: TIMEOUT waiting for the final release\r\n", chip_id);
        err++;
    }

    if (chip_id == COLLECTOR) {
        printf("\r\n[Sweep] In-fabric reduction on %d chiplets (%dx%d) -- xDMA task cycles\r\n",
               NUM_CHIPLETS, N_CHIPLETS_X, N_CHIPLETS_Y);
        printf("[Sweep]  algo    P  fold  task_cc  wall_cc  result\r\n");
        for (int ai = 0; ai < NUM_ALGOS; ai++) {
            for (int mode = 0; mode < NUM_MODES; mode++) {
                for (int pi = 0; pi < NUM_SWEEP_P; pi++) {
                    const int r = (ai * NUM_MODES + mode) * NUM_SWEEP_P + pi;
                    const char *algo = (ai == ALGO_CHAIN) ? "chain" : " tree";
                    if (round_na[r]) {
                        printf("[Sweep] %s %4d  %s        -        -  n/a (needs >=2 full "
                               "rows)\r\n",
                               algo, SWEEP_P[pi], (mode == MODE_LIN) ? "lin" : "mom");
                        continue;
                    }
                    printf("[Sweep] %s %4d  %s  %7u  %7u  %s\r\n", algo, SWEEP_P[pi],
                           (mode == MODE_LIN) ? "lin" : "mom", task_cyc[r], wall_cyc[r],
                           round_err[r] ? "FAIL" : "PASS");
                }
            }
        }
        printf(err ? "[Sweep] Check: FAIL (%d)\r\n" : "[Sweep] Check: PASS (%d)\r\n", err);
    }

    snrt_global_barrier();  // within-chip clean exit
    return err;
#endif  // XDMA_DST_JCT_ENABLE_PTR
}
