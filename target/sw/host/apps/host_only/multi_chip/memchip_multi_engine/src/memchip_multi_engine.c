// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// The memory chip's push engines, and memory chips that hold different data.
//
// Built for hemaia_cmc_16MBL3_1cluster:
//     C(0,0)  M(1,0)  C(2,0)        M(1,0) = memory chip A, between the compute chips
//             M(1,1)                M(1,1) = memory chip B, reached only through A
// Each memory chip was loaded with its own SRAM image and its own HBM image (datagen.py:
// build/mempool.bin for A, build/mempool_chip_1_1.bin for B, an HBM manifest tagged
// chip=<id>), word i of each being (tag << 48) | i with a tag per image.
//
// Chip 00 drives everything; the other compute chip only receives. After the links go to
// DDR:
//   1. Each memory chip holds its own images (CPU loads over D2D).
//   2. Every engine of every memory chip copies HBM -> its own SRAM, checked.
//   3. PARALLEL: A's engine 2 pushes HBM -> chip 20, engine 1 HBM -> chip 00, at once.
//   4. SERIAL:   A's engine 0 pushes both, one after the other (the old single engine).
//   5. B's engines push its HBM to chip 20 and chip 00: through A, B's east port being
//      unavailable (the router falls back north), and checked for B's data. Both streams
//      share the one link B -> A.
//   6. A's engine 3 pushes HBM -> B's SRAM over the memory-chip link, checked.
// The east stream is always launched first. Programming an engine is register writes from
// chip 00, over the west link; once a west stream runs, the link is its sender's until it
// drains (half-duplex), and a second engine programmed behind it would start only then.
// A receiver times its own buffer: the first word landing, the last word landing (local
// polls, which put nothing on a link). Chip 20 reports its times to chip 00; the chips'
// mcycle counters start together and run at the same clock, so chip 00 can line them up.

#include "../data/data.h"
#include "host.h"

#if N_MEM_CHIPS < 2 || N_CHIPLETS < 2
#error "memchip_multi_engine needs two memory chips and two compute chips (hemaia_cmc_16MBL3_1cluster)"
#endif

#define MEM_A    MEM_CHIP_ID_0
#define MEM_B    MEM_CHIP_ID_1
#define DRIVER   CHIPLET_ID_0
#define RECEIVER CHIPLET_ID_1
#define HOST_MHZ ((uint64_t)HEMAIA_SIM_CLK_MHZ / HEMAIA_CORE_CLK_DIV)

#define XFER       (128ULL * 1024)  // bytes per stream
#define N_PHASES   4                // 3, 4 (two receiver windows each), 5, and spare
// Chip-local L3: receive buffers from 4 MiB (the program sits at the bottom, the stack at
// the top), the receiver's reports to chip 00 at 3 MiB.
#define RX_BASE    (SPM_WIDE_BASE_ADDR + 4ULL * 1024 * 1024)
#define REPORT     (SPM_WIDE_BASE_ADDR + 3ULL * 1024 * 1024)
#define REPORT_OK  0x52455054ULL    // "REPT"
_Static_assert(WIDE_SPM_SIZE >= 8 * 1024 * 1024, "needs an L3 of 8 MiB or more");

// Offsets into the HBM images, every stream its own words.
#define HBM_OFF(phase, stream) ((uint64_t)((phase) * 2 + (stream)) * XFER)
// Memory chip SRAM: the loaded image first, then where step 2 and step 6 copy to.
#define L4_ENGINE_DST (2ULL * 1024 * 1024)
#define L4_FROM_A     (4ULL * 1024 * 1024)

typedef struct {
    volatile uint64_t magic;  // REPORT_OK + phase once filled
    volatile uint64_t t_first, t_last, bad;
} report_t;

static uint32_t errors = 0;
static uint8_t me;

static uint64_t word(uint64_t tag, uint64_t byte_off) { return (tag << 48) | (byte_off / 8); }

static uint64_t rx_buf(uint8_t chip, int phase) {
    return chiplet_addr_transform_full(chip, RX_BASE + (uint64_t)phase * XFER);
}
static volatile report_t* report(uint8_t chip, int phase) {
    return (volatile report_t*)(uintptr_t)chiplet_addr_transform_full(
        chip, REPORT + (uint64_t)phase * sizeof(report_t));
}
static uint64_t hbm(uint8_t mem, uint64_t off) {
    return chiplet_addr_transform_full(mem, HBM_BASE_ADDR + off);
}
static uint64_t l4(uint8_t mem, uint64_t off) {
    return chiplet_addr_transform_full(mem, SPM_WIDE_BASE_ADDR + off);
}

static void say(const char* s) {
    printf("Chip(%x, %x): [ME] %s\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), s);
}

// Words [0, n) of `buf`, every `stride`-th and the last, against the image `tag` from
// `src_off`. A remote buffer gets a wide stride: every word read is a D2D round trip.
#define LOCAL  61
#define REMOTE 509
static uint64_t count_bad(volatile uint64_t* buf, uint64_t tag, uint64_t src_off, uint64_t n,
                          uint64_t stride) {
    uint64_t bad = 0;
    asm volatile("fence" ::: "memory");
    for (uint64_t i = 0; i < n / 8; i += stride)
        if (buf[i] != word(tag, src_off + 8 * i)) bad++;
    if (buf[n / 8 - 1] != word(tag, src_off + n - 8)) bad++;
    return bad;
}

static void expect(const char* what, uint64_t got, uint64_t want) {
    if (got != want) {
        printf("Chip(%x, %x): [ME] FAIL %s: %lx, expected %lx\r\n", get_current_chip_loc_x(),
               get_current_chip_loc_y(), what, got, want);
        errors++;
    }
}

// Wait for the first and then the last word of this chip's buffer for `phase`; return
// their mcycle times in *t_first / *t_last.
static void receive(int phase, uint64_t tag, uint64_t src_off, uint64_t* t_first,
                    uint64_t* t_last) {
    volatile uint64_t* b = (volatile uint64_t*)(uintptr_t)rx_buf(me, phase);
    const uint64_t first = word(tag, src_off), last = word(tag, src_off + XFER - 8);
    uint64_t v;
    do {
        asm volatile("fence" ::: "memory");
        v = b[0];
    } while (v != first);
    *t_first = mcycle();
    do {
        asm volatile("fence" ::: "memory");
        v = b[XFER / 8 - 1];
    } while (v != last);
    *t_last = mcycle();
}

static void line(const char* what, uint64_t n, uint64_t cyc) {
    printf("Chip(%x, %x): [ME] %s: %d B in %d cycles = %d MB/s\r\n", get_current_chip_loc_x(),
           get_current_chip_loc_y(), what, (int)n, (int)cyc,
           (int)(n * HOST_MHZ / (cyc ? cyc : 1)));
}

// The receiver: phases 0..2 each deliver XFER bytes to it (from A's engine 2, A's engine
// 0, B's engine 1). Buffers are cleared before the links go up, so nothing races the
// clearing; the report goes to chip 00.
static int run_receiver(void) {
    static const uint64_t tags[3] = {HBM_TAG_A, HBM_TAG_A, HBM_TAG_B};
    for (int p = 0; p < 3; p++) {
        uint64_t t0, t1;
        receive(p, tags[p], HBM_OFF(p, 1), &t0, &t1);
        uint64_t bad = count_bad((volatile uint64_t*)(uintptr_t)rx_buf(me, p), tags[p],
                                 HBM_OFF(p, 1), XFER, LOCAL);
        volatile report_t* r = report(DRIVER, p);
        r->t_first = t0;
        r->t_last = t1;
        r->bad = bad;
        asm volatile("fence" ::: "memory");
        r->magic = REPORT_OK + p;
        asm volatile("fence" ::: "memory");
        errors += (uint32_t)bad;
        line(p == 0 ? "rx from A engine 2 (parallel)"
                    : p == 1 ? "rx from A engine 0 (serial)" : "rx from B engine 1 (via A)",
             XFER, t1 - t0);
    }
    return errors ? -1 : 0;
}

// Return `rc` once the UART has sent everything: the simulation ends when the last chip
// returns, and would cut its last line.
static int finish(int rc) {
    while (!is_transmit_done(get_current_chip_baseaddress())) {
    }
    return rc;
}

static void wait_report(int p, uint64_t* t0, uint64_t* t1) {
    volatile report_t* r = report(DRIVER, p);
    uint64_t m;
    do {
        asm volatile("fence" ::: "memory");
        m = r->magic;
    } while (m != REPORT_OK + p);
    *t0 = r->t_first;
    *t1 = r->t_last;
    expect("receiver's check", r->bad, 0);
}

// Program engine `k` of memory chip `mem` to push XFER bytes of its HBM from `src_off` to
// chip `to`'s buffer for `phase`, and launch it.
static void push(uint8_t mem, int k, uint8_t to, int phase, uint64_t src_off) {
    sys_dma_engine_memcpy(mem, k, rx_buf(to, phase), hbm(mem, src_off), XFER);
}

static void both(const char* what, int p, uint64_t launch) {
    uint64_t f0, l0, f1, l1;
    receive(p, p == 2 ? HBM_TAG_B : HBM_TAG_A, HBM_OFF(p, 0), &f0, &l0);
    expect("chip 00's check",
           count_bad((volatile uint64_t*)(uintptr_t)rx_buf(me, p),
                     p == 2 ? HBM_TAG_B : HBM_TAG_A, HBM_OFF(p, 0), XFER, LOCAL),
           0);
    wait_report(p, &f1, &l1);
    printf("Chip(%x, %x): [ME] %s: launch %d, chip 00 [%d, %d], chip 20 [%d, %d]\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), what, (int)launch, (int)f0,
           (int)l0, (int)f1, (int)l1);
    line("  chip 00 stream", XFER, l0 - f0);
    line("  chip 20 stream", XFER, l1 - f1);
    line("  both, launch to the later last word", 2 * XFER, (l0 > l1 ? l0 : l1) - launch);
}

int main() {
    me = get_current_chip_id();
    // Before any traffic: the receive buffers' first and last words, and chip 00's report
    // slots. Nothing reaches them before the links are up.
    for (int p = 0; p < N_PHASES; p++) {
        volatile uint64_t* b = (volatile uint64_t*)(uintptr_t)rx_buf(me, p);
        b[0] = 0;
        b[XFER / 8 - 1] = 0;
        volatile report_t* r = report(me, p);
        r->magic = 0;
    }
    asm volatile("fence" ::: "memory");
    hemaia_d2d_link_initialize_grid(me);
    int ddr_err = hemaia_d2d_link_ddr_on_grid(me);
    init_uart(get_current_chip_baseaddress(), 32, 1);
    if (ddr_err) {
        say("a memory chip's DDR register did not read back");
        return -1;
    }
    if (me == RECEIVER) return finish(run_receiver());
    if (me != DRIVER) return 0;

    printf("Chip(%x, %x): [ME] host %d MHz, memory chips %x (%d engines) and %x (%d), "
           "%d B per stream\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), (int)HOST_MHZ, MEM_A,
           MEM_CHIP_NUM_SYS_IDMA_0, MEM_B, MEM_CHIP_NUM_SYS_IDMA_1, (int)XFER);

    // 1. Each memory chip holds its own images.
    const uint64_t last = HBM_IMAGE_BYTES - 8, l4_last = L4_IMAGE_BYTES - 8;
    expect("A's SRAM word 0", *(volatile uint64_t*)(uintptr_t)l4(MEM_A, 0), word(L4_TAG_A, 0));
    expect("A's SRAM last", *(volatile uint64_t*)(uintptr_t)l4(MEM_A, l4_last),
           word(L4_TAG_A, l4_last));
    expect("B's SRAM word 0", *(volatile uint64_t*)(uintptr_t)l4(MEM_B, 0), word(L4_TAG_B, 0));
    expect("B's SRAM last", *(volatile uint64_t*)(uintptr_t)l4(MEM_B, l4_last),
           word(L4_TAG_B, l4_last));
    expect("A's HBM word 0", *(volatile uint64_t*)(uintptr_t)hbm(MEM_A, 0), word(HBM_TAG_A, 0));
    expect("A's HBM last", *(volatile uint64_t*)(uintptr_t)hbm(MEM_A, last), word(HBM_TAG_A, last));
    expect("B's HBM word 0", *(volatile uint64_t*)(uintptr_t)hbm(MEM_B, 0), word(HBM_TAG_B, 0));
    expect("B's HBM last", *(volatile uint64_t*)(uintptr_t)hbm(MEM_B, last), word(HBM_TAG_B, last));
    say(errors ? "1. per-chip images: FAILED" : "1. per-chip images: each memory chip holds its own");

    // 2. Every engine: 4 KiB HBM -> its own SRAM, on the memory chip.
    static const uint8_t mems[2] = {MEM_A, MEM_B};
    static const int n_eng[2] = {MEM_CHIP_NUM_SYS_IDMA_0, MEM_CHIP_NUM_SYS_IDMA_1};
    for (int m = 0; m < 2; m++) {
        const uint64_t tag = m ? HBM_TAG_B : HBM_TAG_A;
        for (int k = 0; k < n_eng[m]; k++) {
            const uint64_t src = 8 * XFER + (uint64_t)k * 4096, dst = L4_ENGINE_DST + k * 4096;
            uint32_t id = sys_dma_engine_memcpy(mems[m], k, l4(mems[m], dst), hbm(mems[m], src),
                                                4096);
            sys_dma_engine_wait(mems[m], k, id);
            uint64_t bad = count_bad((volatile uint64_t*)(uintptr_t)l4(mems[m], dst), tag, src,
                                     4096, 64);
            if (bad) {
                printf("Chip(%x, %x): [ME] FAIL memory chip %x engine %d: %d words wrong\r\n",
                       get_current_chip_loc_x(), get_current_chip_loc_y(), mems[m], k, (int)bad);
                errors += bad;
            }
        }
    }
    say(errors ? "2. engines: FAILED" : "2. engines: every engine of both memory chips copies");

    // 3. Parallel: A's engine 2 pushes to the east chip, engine 1 to the west one, at once.
    uint64_t t = mcycle();
    push(MEM_A, 2, RECEIVER, 0, HBM_OFF(0, 1));
    push(MEM_A, 1, DRIVER, 0, HBM_OFF(0, 0));
    both("3. PARALLEL, A engines 2 (east) + 1 (west)", 0, t);

    // 4. Serial: A's engine 0 pushes both, east then west.
    t = mcycle();
    push(MEM_A, 0, RECEIVER, 1, HBM_OFF(1, 1));
    push(MEM_A, 0, DRIVER, 1, HBM_OFF(1, 0));
    both("4. SERIAL, A engine 0 (east, then west)", 1, t);

    // 5. B pushes its own HBM to both compute chips, through A.
    t = mcycle();
    push(MEM_B, 1, RECEIVER, 2, HBM_OFF(2, 1));
    push(MEM_B, 0, DRIVER, 2, HBM_OFF(2, 0));
    both("5. B engines 1 (east) + 0 (west), both over B -> A", 2, t);

    // 6. A's engine 3: HBM -> B's SRAM, over the link between the memory chips. Timed to
    // the engine's done (its last write left A); polling B would turn that link around.
    t = mcycle();
    uint32_t id = sys_dma_engine_memcpy(MEM_A, 3, l4(MEM_B, L4_FROM_A),
                                        hbm(MEM_A, HBM_OFF(3, 0)), XFER);
    sys_dma_engine_wait(MEM_A, 3, id);
    line("6. A engine 3 -> B's SRAM (memory-chip link), to done", XFER, mcycle() - t);
    volatile uint64_t* in_b = (volatile uint64_t*)(uintptr_t)l4(MEM_B, L4_FROM_A);
    uint64_t v;
    do {  // then wait for the last word in B
        asm volatile("fence" ::: "memory");
        v = in_b[XFER / 8 - 1];
    } while (v != word(HBM_TAG_A, HBM_OFF(3, 0) + XFER - 8));
    expect("B's copy of A", count_bad(in_b, HBM_TAG_A, HBM_OFF(3, 0), XFER, REMOTE), 0);

    printf("Chip(%x, %x): [ME] %s (%d wrong)\r\n", get_current_chip_loc_x(),
           get_current_chip_loc_y(), errors ? "FAILED" : "ALL PASSED", (int)errors);
    return finish(errors ? -1 : 0);
}
