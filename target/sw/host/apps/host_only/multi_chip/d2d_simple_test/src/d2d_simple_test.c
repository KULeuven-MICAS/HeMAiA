// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>
// This is a simple test to verify basic D2D functionality: the host CPU on one chip issues D2D writes to the mem chip, then reads them back and checks

#include "host.h"
// chip(2,0) is the "memchip": 64 MiB of external SRAM addressed via D2D.
// Its lower 32 MiB is reserved for static data,
// the upper 32 MiB is the heap region
#define MEMCHIP_LOC_X      0x2
#define MEMCHIP_LOC_Y      0x0
#define MEMCHIP_CHIP_ID    0x20  // (MEMCHIP_LOC_X << 4) | MEMCHIP_LOC_Y

int main() {
    uint8_t current_chip_id = get_current_chip_id();

    // -----------------------------------------------------------------------
    // 1. Clock-domain setup.
    //
    // The reset-time clock dividers (occamy_chip.sv defaults {16,16,1,1,1,1})
    // leave the host CPU at 500 MHz / 16. At that 16:1 host-to-D2D-PHY clock
    // ratio, the D2D narrow write path drops stores from bingoHeapInit's
    // cross-chip burst into the mem chip: the heap manager's `capacity`
    // store is written but never lands, so bingo_mempool_alloc later reads
    // capacity == 0 and every allocation fails. The framer fix removes the
    // earlier deadlock but NOT this data-loss; it is a separate, still-open
    // RTL bug in the D2D narrow path / CDC (see RESULTS.md).
    //
    // Workaround until that RTL bug is fixed: reprogram the host + cluster
    // channels to /8 (= 62.5 MHz, an 8:1 ratio that does not drop stores)
    // and re-state the four D2D PHY channels at /1.
    //
    // Channel layout (see occamy_chip.sv:143-150):
    //   ch 0                              host CPU
    //   ch 1 .. N_CLUSTERS_PER_CHIPLET    accelerator cluster clocks
    //   ch N+1 .. N+4                     East/West/North/South D2D PHY
    // -----------------------------------------------------------------------

    /////////////////////////////////////////////////////////////////////////
    // DEBUG(/16 hunt): clock workaround disabled so the test runs at the
    // /16 RTL default and the D2D narrow-write loss reproduces. Restore by
    // flipping the #if back to 1.
#if 0
    enable_clk_domain(0, 8);   // host CPU @ 500 MHz / 8 = 62.5 MHz
    for (uint8_t i = 0; i < N_CLUSTERS_PER_CHIPLET; i++) {
        enable_clk_domain(1 + i, 8);  // cluster i @ 500 MHz / 8 = 62.5 MHz
    }
    enable_clk_domain(N_CLUSTERS_PER_CHIPLET + 1, 1);  // East  D2D PHY @ 500 MHz
    enable_clk_domain(N_CLUSTERS_PER_CHIPLET + 2, 1);  // West  D2D PHY @ 500 MHz
    enable_clk_domain(N_CLUSTERS_PER_CHIPLET + 3, 1);  // North D2D PHY @ 500 MHz
    enable_clk_domain(N_CLUSTERS_PER_CHIPLET + 4, 1);  // South D2D PHY @ 500 MHz
#endif

    hemaia_d2d_link_initialize(current_chip_id);
    init_uart(get_current_chip_baseaddress(), 32, 1);
    enable_vec();
    enable_sw_interrupts();
    // if (bingo_hemaia_system_mmap_init() < 0) {
    //     printf("Chip(%x, %x): [Host] Error when initializing Allocator\r\n",
    //            get_current_chip_loc_x(), get_current_chip_loc_y());
    //     return -1;
    // }
    // printf("Chip(%x, %x): [Host] Allocator Init Success\r\n",
    //        get_current_chip_loc_x(), get_current_chip_loc_y());
        
    // -----------------------------------------------------------------------
    // Chip 0 issues a LARGE contiguous burst of plain CPU stores (mimicking
    // bingoHeapInit's volume + address) to the mem-chip heap region, fences,
    // then reads them all back. At /16 some stores in the burst are silently
    // lost; at /8 they all land. Reports the total loss count and the first
    // few lost indices so we know if loss is volume- / position- /
    // address-dependent. Remove once the RTL bug is fixed.
    //
    // PROBE_N stores of 8 bytes each = PROBE_N*8 bytes, starting at the same
    // 0x82000000 region bingoHeapInit writes.
    // -----------------------------------------------------------------------
#define PROBE_N 64
    if (current_chip_id == 0) {
        // if (bingo_mempool_init(MEMCHIP_LOC_X, MEMCHIP_LOC_Y) < 0) {
        //     printf("Chip(%x, %x): Memchip heap init failed!\r\n",
        //         get_current_chip_loc_x(), get_current_chip_loc_y());
        //     return -1;
        // } else {
        //     printf("Chip(%x, %x): Memchip heap init success!\r\n",
        //         get_current_chip_loc_x(), get_current_chip_loc_y());
        // }
        // volatile uint64_t* mc = (uint64_t*)bingo_mempool_alloc(MEMCHIP_LOC_X,
        //                                             MEMCHIP_LOC_Y,
        //                                             8192);
        volatile uint64_t* mc = (volatile uint64_t*)chiplet_addr_transform_loc(
            MEMCHIP_LOC_X, MEMCHIP_LOC_Y, 0x82000000);
        for (int i = 0; i < PROBE_N; i++) {
            mc[i] = 0xABCD000000000000ULL | (uint64_t)i;
        }
        asm volatile("fence" ::: "memory");
        int probe_fail = 0;
        int first_lost[8];
        int nfirst = 0;
        for (int i = 0; i < PROBE_N; i++) {
            uint64_t got = mc[i];
            uint64_t expect = 0xABCD000000000000ULL | (uint64_t)i;
            if (got != expect) {
                probe_fail++;
                if (nfirst < 8) first_lost[nfirst++] = i;
            }
        }
        printf("Chip(%x, %x): PROBE result = %s (%d/%d lost)\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(),
               probe_fail ? "FAIL" : "PASS", probe_fail, PROBE_N);
        for (int k = 0; k < nfirst; k++) {
            int i = first_lost[k];
            printf("Chip(%x, %x): PROBE lost[%d]: memchip[0x82000000+%d*8] got %lx expect %lx\r\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y(),
                   k, i, mc[i], 0xABCD000000000000ULL | (uint64_t)i);
        }
    }
}
