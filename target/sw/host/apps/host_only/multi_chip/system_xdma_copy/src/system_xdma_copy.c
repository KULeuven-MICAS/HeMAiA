// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>
//
// Multi-chip XDMA / IDMA copy test.
//
// The same binary runs on every compute chiplet. The test exercises three
// transfer paths through two buffers:
//   memchip_buf  — heap-allocated on the memchip at chip(2,0); shared.
//   local_l3_buf — heap-allocated on each chip's own L3 SPM; per-chip.
//
//   Stage A — chip 0 XDMA-copies its local `data` array into memchip_buf.
//   Stage B — every chip IDMA-unicasts memchip_buf -> local_l3_buf, verifies,
//             then zeroes local_l3_buf.
//   Stage C — chip(2,0)'s IDMA engine broadcasts memchip_buf to every chip's
//             local_l3_buf in one transfer; every chip re-verifies.
//
// Address model: a 64-bit address is [chip prefix (bits >=40) | 40-bit
// local address]. `chiplet_addr_transform_*` helpers OR a chip prefix into
// a local address. Chip 0 has prefix 0, chip(2,0) has prefix 0x20, and the
// broadcast prefix is 0xFF.

#include "../data/data.h"
#include "host.h"
#include "libbingo/bingo_api.h"
#include "hemaia_clk_rst_controller.h"  // enable_clk_domain()

// chip(2,0) is the "memchip": 64 MiB of external SRAM addressed via D2D.
// Its lower 32 MiB is reserved for static data (bingo_get_mempool_data_base),
// the upper 32 MiB is the heap region used by bingo_mempool_* / this test.
#define MEMCHIP_LOC_X      0x2
#define MEMCHIP_LOC_Y      0x0
#define MEMCHIP_CHIP_ID    0x20  // (MEMCHIP_LOC_X << 4) | MEMCHIP_LOC_Y

// Compute-chiplet rectangle used by chip_barrier (top-left .. bottom-right).
#define BARRIER_TL_CHIP_ID 0x00
#define BARRIER_BR_CHIP_ID 0x11

// Broadcast prefix for chiplet_addr_transform_loc().
#define BROADCAST_LOC_X    0xF
#define BROADCAST_LOC_Y    0xF

#define CHECKED_DATA_SIZE  64

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
    enable_clk_domain(0, 8);   // host CPU @ 500 MHz / 8 = 62.5 MHz
    for (uint8_t i = 0; i < N_CLUSTERS_PER_CHIPLET; i++) {
        enable_clk_domain(1 + i, 8);  // cluster i @ 500 MHz / 8 = 62.5 MHz
    }
    enable_clk_domain(N_CLUSTERS_PER_CHIPLET + 1, 1);  // East  D2D PHY @ 500 MHz
    enable_clk_domain(N_CLUSTERS_PER_CHIPLET + 2, 1);  // West  D2D PHY @ 500 MHz
    enable_clk_domain(N_CLUSTERS_PER_CHIPLET + 3, 1);  // North D2D PHY @ 500 MHz
    enable_clk_domain(N_CLUSTERS_PER_CHIPLET + 4, 1);  // South D2D PHY @ 500 MHz

    // -----------------------------------------------------------------------
    // 2. Hardware init: D2D links, UART, vector extension, SW interrupts.
    // -----------------------------------------------------------------------
    hemaia_d2d_link_initialize(current_chip_id);
    init_uart(get_current_chip_baseaddress(), 32, 1);
    enable_vec();
    enable_sw_interrupts();

    // -----------------------------------------------------------------------
    // 3. Allocator init: bring up the per-chip L2 (narrow SPM) and
    //    L3 (wide SPM) heaps. Manager structures land at the start of each
    //    heap region; their addresses are fixed by the linker layout.
    // -----------------------------------------------------------------------
    if (bingo_hemaia_system_mmap_init() < 0) {
        printf("Chip(%x, %x): [Host] Error when initializing Allocator\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    }
    printf("Chip(%x, %x): [Host] Allocator Init Success\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y());

    // Cached comm-buffer pointer for the chip_barrier calls below.
    comm_buffer_t* comm_buf =
        (comm_buffer_t*)bingo_get_l2_comm_buffer(current_chip_id);

    // -----------------------------------------------------------------------
    // 4. Memchip heap init (chip 0 only).
    //    Non-chip-0 chips don't touch the memchip heap directly — they only
    //    read memchip_buf (broadcast in stage A) — so no handshake on the
    //    manager handle is needed; the stage-A chip_barrier suffices.
    //
    //    The heap sits in the top 32 MiB of chip(2,0); the low 32 MiB
    //    (data region) is untouched by this test and reachable via
    //    bingo_get_mempool_data_base(MEMCHIP_LOC_X, MEMCHIP_LOC_Y).
    // -----------------------------------------------------------------------
    if (current_chip_id == 0){
        if (bingo_mempool_init(MEMCHIP_LOC_X, MEMCHIP_LOC_Y) < 0) {
            printf("Chip(%x, %x): Memchip heap init failed!\r\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y());
            return -1;
        } else {
            printf("Chip(%x, %x): Memchip heap init success!\r\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y());
        }
    }

    // -----------------------------------------------------------------------
    // 5. Stage A — chip 0 allocates memchip_buf on the memchip heap, fills
    //    it via XDMA, and broadcasts the pointer to every chip.
    //
    // -----------------------------------------------------------------------
    uint8_t* memchip_buf;

    if (current_chip_id == 0) {
        printf("Chip(%x, %x): Allocating memchip_buf on memchip heap...\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
        memchip_buf = (uint8_t*)bingo_mempool_alloc(MEMCHIP_LOC_X,
                                                    MEMCHIP_LOC_Y,
                                                    data_size);
        printf("Chip(%x, %x): Memchip_buf allocated at %lx\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(),
               (uintptr_t)memchip_buf);

        // Broadcast memchip_buf to every chip's stack slot.
        *(volatile uint8_t**)(chiplet_addr_transform_loc(
            BROADCAST_LOC_X, BROADCAST_LOC_Y,
            (uintptr_t)&memchip_buf)) = memchip_buf;

        // XDMA: chip 0 local `data` -> chip(2,0) memchip_buf.
        hemaia_xdma_memcpy_1d((uint8_t*)data, memchip_buf, data_size);
        uint32_t task_id = hemaia_xdma_start();
        hemaia_xdma_remote_wait(task_id);
        printf("Chip(%x, %x): XDMA copy to memchip_buf finished!\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
        asm volatile("fence" ::: "memory");
    }

    // Stage A done; every chip's memchip_buf must be set before we read it.
    chip_barrier(comm_buf, BARRIER_TL_CHIP_ID, BARRIER_BR_CHIP_ID, 1);

    // -----------------------------------------------------------------------
    // 6. Stage B — every chip allocates local_l3_buf in its own L3 SPM,
    //    IDMA-unicasts memchip_buf (chip(2,0)) into it, verifies, and
    //    zeroes local_l3_buf in preparation for stage C.
    //
    // Each chip's L3 heap was initialized identically and this is the
    // first allocation, so every chip's local_l3_buf has the same local
    // offset — important for the stage-C broadcast below.
    // -----------------------------------------------------------------------
    uint8_t* local_l3_buf =
        (uint8_t*)bingo_l3_alloc(current_chip_id, data_size);

    printf("Chip(%x, %x): local_l3_buf allocated at %lx\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(),
           (uintptr_t)local_l3_buf);

    printf("Chip(%x, %x): IDMA unicast: memchip_buf %lx -> local_l3_buf %lx\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(),
           (uintptr_t)memchip_buf, (uintptr_t)local_l3_buf);
    sys_dma_blk_memcpy(current_chip_id,
                       (uintptr_t)local_l3_buf,
                       (uintptr_t)memchip_buf,
                       data_size);
    printf("Chip(%x, %x): IDMA unicast to local_l3_buf finished!\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y());

    printf("Chip(%x, %x): Checking data correctness...\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y());
    for (uint32_t i = 0; i < CHECKED_DATA_SIZE; i++) {
        if (data[i] != local_l3_buf[i]) {
            printf("Chip(%x, %x): Data mismatch at index %d: expected %d, got %d\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y(),
                   i, data[i], local_l3_buf[i]);
            return -1;
        }
    }

    // Zero local_l3_buf before stage C overwrites it again.
    sys_dma_blk_memcpy(current_chip_id,
                       (uintptr_t)local_l3_buf,
                       (uintptr_t)WIDE_ZERO_MEM_BASE_ADDR,
                       data_size);
    printf("Chip(%x, %x): local_l3_buf cleared!\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y());

    chip_barrier(comm_buf, BARRIER_TL_CHIP_ID, BARRIER_BR_CHIP_ID, 2);

    // -----------------------------------------------------------------------
    // 7. Stage C — chip(2,0)'s IDMA engine broadcasts memchip_buf to every
    //    chip's local_l3_buf in one transfer. Only chip 0 issues the request.
    //    The destination address uses (0xF, 0xF) broadcast prefix; the same
    //    local offset reaches every chip because allocations in step 5 were
    //    deterministic.
    // -----------------------------------------------------------------------
    if (current_chip_id == 0) {
        sys_dma_blk_memcpy(
            MEMCHIP_CHIP_ID,
            chiplet_addr_transform_loc(BROADCAST_LOC_X, BROADCAST_LOC_Y,
                                        (uintptr_t)local_l3_buf),
            (uintptr_t)memchip_buf,
            data_size);
        printf("Chip(%x, %x): IDMA broadcast to local_l3_buf finished!\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
    }
    chip_barrier(comm_buf, BARRIER_TL_CHIP_ID, BARRIER_BR_CHIP_ID, 3);

    printf("Chip(%x, %x): Checking data correctness...\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y());
    for (uint32_t i = 0; i < CHECKED_DATA_SIZE; i++) {
        if (data[i] != local_l3_buf[i]) {
            printf("Chip(%x, %x): Data mismatch at index %d: expected %d, got %d\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y(),
                   i, data[i], local_l3_buf[i]);
            return -1;
        }
    }
    printf("Chip(%x, %x): Data check passed!\n",
           get_current_chip_loc_x(), get_current_chip_loc_y());

    // -----------------------------------------------------------------------
    // 8. Cleanup.
    // -----------------------------------------------------------------------
    if (current_chip_id == 0) {
        bingo_mempool_free(MEMCHIP_LOC_X, MEMCHIP_LOC_Y, (uint64_t)memchip_buf);
    }
    bingo_l3_free(current_chip_id, (uint64_t)local_l3_buf);
    return 0;
}
