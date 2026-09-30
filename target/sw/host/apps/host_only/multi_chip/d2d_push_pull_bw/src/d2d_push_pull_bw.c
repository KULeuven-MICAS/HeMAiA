// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// PULL against PUSH over the half-duplex D2D link, for moving weights from the memory
// chiplet into this chiplet's L3.
//
//   pull  this chip's system iDMA reads HBM (or the memchip SRAM, "L4") and writes L3.
//         Every burst is a read request going out and data coming back, so the link
//         turns around for every request.
//   push  the MEMORY chiplet's system iDMA reads HBM (or L4) and writes our L3. Data
//         only flows toward us: the link answers each write at the sender.
//
// Chip 00 runs everything, once with the links in SDR and once in DDR
// (hemaia_d2d_link_ddr_on_grid). Every copy is checked: the image's word i is
// (BW_TAG << 48) | i, and each test copies from its own offset so a stale copy fails.
//
// A push is timed until its LAST word lands in L3 (polled locally, the D-cache
// invalidated by a fence per poll), not until the memchip's iDMA reports done: that
// happens when the last write leaves the memchip, before it arrives.

#include "../data/data.h"
#include "host.h"
#include "libbingo/bingo_api.h"

#define MEMCHIP_ID ((MEM_CHIP_LOC_X << 4) | MEM_CHIP_LOC_Y)
#define HOST_MHZ   ((uint64_t)HEMAIA_SIM_CLK_MHZ / HEMAIA_CORE_CLK_DIV)
#define XFER       (1024ULL * 1024)     // bytes per measured copy
#define CHUNK      (128ULL * 1024)      // a weight chunk, as the clusters stream them
#define STEP       (64ULL * 1024)       // offset between tests: every test its own words

static uint32_t errors = 0;

static uint64_t word(uint64_t byte_off) { return (BW_TAG << 48) | (byte_off / 8); }

static void check(const char* what, volatile uint64_t* dst, uint64_t src_off, uint64_t n) {
    uint32_t bad = 0;
    asm volatile("fence" ::: "memory");
    for (uint64_t i = 0; i < n / 8; i += 97)
        if (dst[i] != word(src_off + 8 * i)) bad++;
    if (dst[n / 8 - 1] != word(src_off + n - 8)) bad++;
    if (bad) {
        printf("Chip(%x, %x): [BW] %s: %d sampled words wrong, e.g. [0] = %lx, expected %lx\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), what, (int)bad,
               dst[0], word(src_off));
        errors += bad;
    }
}

static void clear_sentinels(volatile uint64_t* dst, uint64_t n, uint64_t chunk) {
    for (uint64_t off = chunk - 8; off < n; off += chunk) dst[off / 8] = 0;
    asm volatile("fence" ::: "memory");
}

// Wait until the last word of [dst, dst + n) holds `expect`.
static void wait_word(volatile uint64_t* last, uint64_t expect) {
    uint64_t v;
    do {
        asm volatile("fence" ::: "memory");
        v = *last;
    } while (v != expect);
}

static uint64_t pull(uint64_t dst, uint64_t src, uint64_t n) {
    uint64_t t0 = mcycle();
    sys_dma_blk_memcpy(get_current_chip_id(), dst, src, n);
    asm volatile("fence" ::: "memory");
    return mcycle() - t0;
}

// One push of n bytes, or n / chunk pushes of `chunk` launched back to back -- the
// pattern a prefetcher filling an L3 ring would issue.
static uint64_t push(uint64_t dst, uint64_t src, uint64_t src_off, uint64_t n, uint64_t chunk) {
    volatile uint64_t* d = (volatile uint64_t*)dst;
    clear_sentinels(d, n, chunk);
    uint64_t t0 = mcycle();
    for (uint64_t off = 0; off < n; off += chunk)
        sys_dma_memcpy(MEMCHIP_ID, dst + off, src + off, chunk);
    wait_word(&d[n / 8 - 1], word(src_off + n - 8));
    uint64_t t = mcycle() - t0;
    asm volatile("fence" ::: "memory");
    return t;
}

static void line(const char* mode, const char* what, uint64_t n, uint64_t cyc) {
    uint64_t mbps = n * HOST_MHZ / (cyc ? cyc : 1);
    printf("Chip(%x, %x): [BW] %s %s: %d B in %d cycles = %d MB/s\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), mode, what, (int)n, (int)cyc,
           (int)mbps);
}

static void run(const char* mode, uint64_t base, uint64_t l3, uint64_t hbm, uint64_t l4) {
    volatile uint64_t* d = (volatile uint64_t*)l3;
    uint64_t o;
    o = base + 0 * STEP;
    clear_sentinels(d, XFER, XFER);
    line(mode, "pull  HBM -> L3 (our iDMA)", XFER, pull(l3, hbm + o, XFER));
    check("pull HBM", d, o, XFER);
    o = base + 1 * STEP;
    line(mode, "push  HBM -> L3 (memchip iDMA)", XFER, push(l3, hbm + o, o, XFER, XFER));
    check("push HBM", d, o, XFER);
    o = base + 2 * STEP;
    line(mode, "push  HBM -> L3, 8 x 128 KiB", XFER, push(l3, hbm + o, o, XFER, CHUNK));
    check("push HBM chunked", d, o, XFER);
    o = base + 3 * STEP;
    line(mode, "push  L4 -> L3 (memchip iDMA)", XFER, push(l3, l4 + o, o, XFER, XFER));
    check("push L4", d, o, XFER);
    o = base + 4 * STEP;
    clear_sentinels(d, XFER, XFER);
    line(mode, "pull  L4 -> L3 (our iDMA)", XFER, pull(l3, l4 + o, XFER));
    check("pull L4", d, o, XFER);
}

int main() {
    uint8_t chip = get_current_chip_id();
    hemaia_d2d_link_initialize_grid(chip);
    init_uart(get_current_chip_baseaddress(), 32, 1);
    enable_sw_interrupts();
    if (bingo_hemaia_system_mmap_init() < 0) {
        printf("Chip(%x, %x): [Host] Error when initializing Allocator\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    }
    if (chip != 0) return 0;
    if (N_MEM_CHIPS == 0 || HBM_SIZE == 0) {
        printf("Chip(%x, %x): [BW] the cfg has no memory chip with an HBM\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    }
    uint64_t hbm = bingo_get_hbm_base(MEM_CHIP_LOC_X, MEM_CHIP_LOC_Y);
    uint64_t l4 = bingo_get_mempool_data_base(MEM_CHIP_LOC_X, MEM_CHIP_LOC_Y);
    uint64_t l3 = (uint64_t)bingo_l3_alloc(chip, XFER);
    if (!l3) {
        printf("Chip(%x, %x): [BW] L3 allocation failed\r\n", get_current_chip_loc_x(),
               get_current_chip_loc_y());
        return -1;
    }
    printf("Chip(%x, %x): [BW] host %d MHz, %d B per copy, image %d B\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), (int)HOST_MHZ, (int)XFER,
           (int)BW_IMAGE_BYTES);

    // Stage the whole image HBM -> L4 on the memchip: local, the link is not used.
    uint64_t t0 = mcycle();
    sys_dma_blk_memcpy(MEMCHIP_ID, l4, hbm, BW_IMAGE_BYTES);
    line("SDR", "HBM -> L4 on the memchip (no link)", BW_IMAGE_BYTES, mcycle() - t0);

    run("SDR", 0, l3, hbm, l4);
    int ddr_err = hemaia_d2d_link_ddr_on_grid(chip);
    printf("Chip(%x, %x): [BW] links switched to DDR%s\r\n", get_current_chip_loc_x(),
           get_current_chip_loc_y(), ddr_err ? " -- memchip read-back FAILED" : "");
    if (ddr_err) return -1;
    run("DDR", 8 * STEP, l3, hbm, l4);

    printf("Chip(%x, %x): [BW] %s (%d wrong words)\r\n", get_current_chip_loc_x(),
           get_current_chip_loc_y(), errors ? "FAILED" : "ALL PASSED", (int)errors);
    return errors ? -1 : 0;
}
