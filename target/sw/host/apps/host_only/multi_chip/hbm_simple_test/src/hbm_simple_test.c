// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Smoke test of the memory chip's simulated HBM
// (hw/hemaia/hemaia_mem_system/hbm).
//
// Runs on any cfg whose memory chip has an HBM (hemaia_twochiplet_16MBL3_4cluster).
// Chip 00 does all of it; any other compute chip only brings its D2D links up.
//   1. The image the testharness loaded from build/hbm/manifest.txt reads back, both
//      below and above 4 GiB (CPU loads over D2D).
//   2. CPU stores to eight pseudo-channels read back.
//   3. Load latency of the HBM next to the memchip SRAM.
//   4. Chip 00's iDMA pulls 8 KiB of the image into local L3; every word checked.
//   5. The memchip's own iDMA copies 8 KiB HBM -> HBM; pulled back and checked.

#include "../data/data.h"
#include "host.h"
#include "libbingo/bingo_api.h"

// Where the cfg puts the memory chip (occamy.h).
#define MEMCHIP_LOC_X   MEM_CHIP_LOC_X
#define MEMCHIP_LOC_Y   MEM_CHIP_LOC_Y
#define MEMCHIP_CHIP_ID ((MEM_CHIP_LOC_X << 4) | MEM_CHIP_LOC_Y)

// Bytes per channel before the default HBM moves to the next one.
#define HBM_INTERLEAVE 4096
#define STORE_TAG      0x5354ULL

static uint32_t errors = 0;

static uint64_t pattern(uint64_t tag, uint64_t i) { return (tag << 48) | i; }

static void expect(const char* what, uint64_t idx, uint64_t got, uint64_t exp) {
    if (got == exp) return;
    if (errors < 16)
        printf("Chip(%x, %x): [HBM] %s[%d] = %lx, expected %lx\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), what,
               (int)idx, got, exp);
    errors++;
}

static void report(const char* what, uint32_t errors_before) {
    printf("Chip(%x, %x): [HBM] %s: %s\r\n", get_current_chip_loc_x(),
           get_current_chip_loc_y(), what,
           errors == errors_before ? "PASS" : "FAIL");
}

// Blocking copy on `engine`'s system iDMA; returns the cycles it took. The fence
// also invalidates the write-through D-cache, so the CPU reads what the DMA wrote.
static uint64_t dma(uint8_t engine, uint64_t dst, uint64_t src, uint64_t size) {
    uint64_t t0 = mcycle();
    sys_dma_blk_memcpy(engine, dst, src, size);
    asm volatile("fence" ::: "memory");
    return mcycle() - t0;
}

// Cycles per load, loads serialised by a fence.
static uint64_t load_latency(volatile uint64_t* base) {
    uint64_t sink = 0;
    uint64_t t0 = mcycle();
    for (int k = 0; k < 8; k++) {
        sink += base[k * 1024];  // 8 KiB apart: a new line, row and channel each time
        asm volatile("fence" ::: "memory");
    }
    (void)sink;
    return (mcycle() - t0) / 8;
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
        printf("Chip(%x, %x): [HBM] the cfg has no memory chip with an HBM\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    }
    uint64_t hbm = bingo_get_hbm_base(MEMCHIP_LOC_X, MEMCHIP_LOC_Y);
    volatile uint64_t* low = (volatile uint64_t*)(hbm + HBM_LOW_OFF);
    volatile uint64_t* high = (volatile uint64_t*)(hbm + HBM_HIGH_OFF);
    volatile uint64_t* scratch = (volatile uint64_t*)(hbm + HBM_SCRATCH_OFF);
    printf("Chip(%x, %x): [HBM] %d MiB at %lx\r\n", get_current_chip_loc_x(),
           get_current_chip_loc_y(), (int)(HBM_SIZE >> 20), hbm);

    // 1. The loaded image. A prime stride touches lines all over it.
    uint32_t before = errors;
    for (uint64_t i = 0; i < HBM_LOW_WORDS; i += 509)
        expect("low", i, low[i], pattern(HBM_LOW_TAG, i));
    expect("low", HBM_LOW_WORDS - 1, low[HBM_LOW_WORDS - 1],
           pattern(HBM_LOW_TAG, HBM_LOW_WORDS - 1));
    for (uint64_t i = 0; i < HBM_HIGH_WORDS; i += 61)
        expect("high", i, high[i], pattern(HBM_HIGH_TAG, i));
    report("1. loaded image (below and above 4 GiB)", before);

    // 2. CPU stores, one per channel.
    before = errors;
    for (uint64_t k = 0; k < 8; k++)
        scratch[k * HBM_INTERLEAVE / 8] = pattern(STORE_TAG, k);
    asm volatile("fence" ::: "memory");
    for (uint64_t k = 0; k < 8; k++)
        expect("store", k, scratch[k * HBM_INTERLEAVE / 8],
               pattern(STORE_TAG, k));
    report("2. CPU stores to 8 channels", before);

    // 3. Latency.
    volatile uint64_t* sram = (volatile uint64_t*)bingo_get_mempool_data_base(
        MEMCHIP_LOC_X, MEMCHIP_LOC_Y);
    uint64_t lat_hbm = load_latency(low);
    uint64_t lat_sram = load_latency(sram);
    printf("Chip(%x, %x): [HBM] 3. load latency: HBM %d cycles, memchip SRAM %d cycles\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), (int)lat_hbm,
           (int)lat_sram);

    // 4. Chip 00's iDMA pulls the start of the image into local L3.
    uint64_t* buf = (uint64_t*)bingo_l3_alloc(chip, HBM_COPY_BYTES);
    if (buf == 0) {
        printf("Chip(%x, %x): [HBM] L3 allocation failed\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    }
    before = errors;
    uint64_t cyc = dma(chip, (uint64_t)buf, hbm + HBM_LOW_OFF, HBM_COPY_BYTES);
    for (uint64_t i = 0; i < HBM_COPY_BYTES / 8; i++)
        expect("pull", i, buf[i], pattern(HBM_LOW_TAG, i));
    printf("Chip(%x, %x): [HBM] 4. iDMA HBM -> local L3, %d B in %d cycles\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), HBM_COPY_BYTES,
           (int)cyc);
    report("4. iDMA HBM -> local L3", before);

    // 5. The memchip's iDMA copies HBM -> HBM, 64 KiB past the CPU stores; chip 00
    // pulls the copy back.
    before = errors;
    for (uint64_t i = 0; i < HBM_COPY_BYTES / 8; i++) buf[i] = 0;
    uint64_t copy = hbm + HBM_SCRATCH_OFF + 0x10000;
    cyc = dma(MEMCHIP_CHIP_ID, copy, hbm + HBM_LOW_OFF, HBM_COPY_BYTES);
    dma(chip, (uint64_t)buf, copy, HBM_COPY_BYTES);
    for (uint64_t i = 0; i < HBM_COPY_BYTES / 8; i++)
        expect("copy", i, buf[i], pattern(HBM_LOW_TAG, i));
    printf("Chip(%x, %x): [HBM] 5. memchip iDMA HBM -> HBM, %d B in %d cycles\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), HBM_COPY_BYTES,
           (int)cyc);
    report("5. memchip iDMA HBM -> HBM", before);

    bingo_l3_free(chip, (uint64_t)buf);
    printf("Chip(%x, %x): [HBM] %s (%d mismatches)\r\n", get_current_chip_loc_x(),
           get_current_chip_loc_y(), errors ? "FAILED" : "ALL PASSED",
           (int)errors);
    return errors ? -1 : 0;
}
