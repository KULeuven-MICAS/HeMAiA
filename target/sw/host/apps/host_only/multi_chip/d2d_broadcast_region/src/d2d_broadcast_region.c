// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Broadcasts (stores to chip 0xFF) on a grid with memory chips BETWEEN the compute chips,
// e.g. hemaia_sixchiplet_16MBL3_2cluster:
//     C00  M10  C20
//     C01  M11  C21
// The router forwards a broadcast over the whole rectangle; hemaia_d2d_link_initialize_grid
// programs the region: compute chips forward everywhere, memory chips forward but never
// deliver to themselves (multicast fence bit 4 = 0).
//
// Every compute chip runs ROUNDS host chip barriers over the whole rectangle -- memory chips
// inside it, which the wait skips -- each announced by ONE broadcast store. Then:
//   1. every compute chip holds every other compute chip's last checkpoint: each broadcast
//      reached every compute chip (the barriers finishing already says so);
//   2. no memory chip consumed one: chip 0 seeded a marker in each memory chip's copy of
//      the checkpoint array before any broadcast, and it is still there.

#include "host.h"

#define ROUNDS  4
#define TL      0x00
#define BR      ((uint8_t)((((N_CHIPLETS_X) - 1) << 4) | ((N_CHIPLETS_Y) - 1)))
#define MARKER  0xA5

static void say(const char* s, int v) {
    printf("Chip(%x, %x): [BC] %s %d\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(),
           s, v);
}

int main() {
    const uint8_t me = get_current_chip_id();
    volatile comm_buffer_t* cb =
        (volatile comm_buffer_t*)chiplet_addr_transform((uint64_t)&__narrow_spm_start);
    // Before any broadcast can arrive: this chip's own checkpoint array.
    for (int i = 0; i < 256; i++) cb->chip_level_checkpoint[i] = 0;
    asm volatile("fence" ::: "memory");

    // Every chip enters the DDR switch at the same moment (it is timed, see
    // hemaia_d2d_link_ddr_on_grid): nothing may run between the two calls on one chip only.
    hemaia_d2d_link_initialize_grid(me);
    int ddr_err = hemaia_d2d_link_ddr_on_grid(me);
    init_uart(get_current_chip_baseaddress(), 32, 1);
    if (ddr_err) {
        say("a memory chip's DDR register did not read back:", ddr_err);
        return -1;
    }
#if N_MEM_CHIPS > 0
    static const uint8_t mems[N_MEM_CHIPS] = MEM_CHIP_IDS;
    const uint64_t cp_off =
        (uint64_t)(uintptr_t)&cb->chip_level_checkpoint[0] & ((1ULL << 40) - 1);
    if (me == CHIPLET_ID_0) {
        // The marker in every memory chip's copy, for every compute chip's entry: a
        // broadcast it consumed would overwrite one. Done before any chip broadcasts (the
        // guard below).
        for (int m = 0; m < N_MEM_CHIPS; m++)
            for (int i = 0; i < 256; i++)
                if (HEMAIA_IS_COMPUTE_CHIP(i))
                    *(volatile uint8_t*)(uintptr_t)chiplet_addr_transform_full(mems[m],
                                                                               cp_off + i) =
                        MARKER;
        asm volatile("fence" ::: "memory");
    }
#endif
    // The guard: every chip past its DDR switch, and chip 0's markers in, before the first
    // broadcast.
    delay_cycles(20000);

    uint64_t t[ROUNDS + 1];
    t[0] = mcycle();
    for (int r = 1; r <= ROUNDS; r++) {
        chip_barrier(cb, TL, BR, (uint8_t)r);   // announce: one broadcast store
        t[r] = mcycle();
    }
    int bad = 0;
    for (int i = 0; i < 256; i++) {
        if (!HEMAIA_IS_COMPUTE_CHIP(i) || i == me) continue;
        asm volatile("fence" ::: "memory");
        if (cb->chip_level_checkpoint[i] != ROUNDS) {
            printf("Chip(%x, %x): [BC] FAIL chip %02x's checkpoint %d, expected %d\r\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y(), i,
                   (int)cb->chip_level_checkpoint[i], ROUNDS);
            bad++;
        }
    }
    printf("Chip(%x, %x): [BC] %d barriers over %02x..%02x, cycles each: %d %d %d %d\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), ROUNDS, TL, BR,
           (int)(t[1] - t[0]), (int)(t[2] - t[1]), (int)(t[3] - t[2]), (int)(t[4] - t[3]));
#if N_MEM_CHIPS > 0
    if (me == CHIPLET_ID_0) {
        for (int m = 0; m < N_MEM_CHIPS; m++)
            for (int i = 0; i < 256; i++) {
                if (!HEMAIA_IS_COMPUTE_CHIP(i)) continue;
                uint8_t v = *(volatile uint8_t*)(uintptr_t)chiplet_addr_transform_full(
                    mems[m], cp_off + i);
                if (v != MARKER) {
                    printf("Chip(%x, %x): [BC] FAIL memory chip %02x consumed chip %02x's "
                           "broadcast (%d)\r\n", get_current_chip_loc_x(),
                           get_current_chip_loc_y(), mems[m], i, (int)v);
                    bad++;
                }
            }
    }
#endif
    printf("Chip(%x, %x): [BC] %s (%d wrong)\r\n", get_current_chip_loc_x(),
           get_current_chip_loc_y(), bad ? "FAILED" : "ALL PASSED", bad);
    while (!is_transmit_done(get_current_chip_baseaddress())) {
    }
    return bad ? -1 : 0;
}
