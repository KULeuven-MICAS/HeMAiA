// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

// =============================================================================
// D2D (die-to-die) multi-stream test
// -----------------------------------------------------------------------------
// This host application runs on each of the four chiplets in a 2x2 HeMAiA
// topology (chip ids 0x00, 0x01, 0x10, 0x11). Every chip concurrently streams a
// local data buffer into a *shared* external DRAM, each writing to its own
// disjoint slice. The four concurrent DMA streams exercise the D2D links
// simultaneously (hence "multi-stream"). After all chips finish, chip 0x00
// reads back the four slices and checks them against the golden data.
//
// Flow:
//   1. Per-chip bring-up    : D2D link, UART, interrupts, bingo allocator.
//   2. Per-chip DMA stream  : copy local `data` into this chip's DRAM slice.
//   3. Barrier (checkpoint 2): wait until every chip has finished streaming.
//   4. Verification (chip 0): read back all four slices and compare to golden.
//   5. Barrier (checkpoint 3): all chips leave the simulation together.
// =============================================================================

#include "../data/data.h"
#include "host.h"
#include "libbingo/bingo_api.h"

// -----------------------------------------------------------------------------
// Global runtime state
// -----------------------------------------------------------------------------
// Shared comm buffer used by the cross-chip barrier, plus the bingo heap
// managers for this chip's L2/L3 scratchpad and the external DRAM.
comm_buffer_t *comm_buffer_ptr = NULL;
uint64_t local_l2_heap_manager = 0;
uint64_t local_l3_heap_manager = 0;
volatile uint64_t dram_heap_manager = 0;

// -----------------------------------------------------------------------------
// Test configuration
// -----------------------------------------------------------------------------
// Number of back-to-back DMA copies each chip issues into its DRAM slice. More
// iterations means longer, more sustained pressure on the D2D links.
#define NUM_DMA_STREAMS 8
// Number of leading bytes of each slice that chip 0x00 verifies. Kept small for
// a fast sanity check; raise it to compare more of the payload.
#define CHECKED_DATA_SIZE 64
// Base of the shared external DRAM, expressed in the destination chiplet's
// (x=0x2, y=0x0) local address space. Slices are laid out back-to-back here.
#define DRAM_DEST_X 0x2
#define DRAM_DEST_Y 0x0
#define DRAM_DEST_LOCAL_ADDR 0x80000000
// The four chiplet ids in the 2x2 mesh, ordered by their DRAM slice index.
#define CHIP_TOP_LEFT 0x00
#define CHIP_BOTTOM_RIGHT 0x11
#define NUM_CHIPS 4

// -----------------------------------------------------------------------------
// Per-chip bring-up: D2D link, UART, interrupts, and the bingo allocator.
// Returns 0 on success, -1 if the allocator fails to initialize.
// -----------------------------------------------------------------------------
static int chip_init(uint8_t current_chip_id)
{
    // Program the chiplet topology so the D2D links know their neighbours.
    hemaia_d2d_link_initialize(current_chip_id);
    // Bring up the UART so printf() works, then enable software interrupts.
    init_uart(get_current_chip_baseaddress(), 32, 1);
    enable_sw_interrupts();

    // Initialize the bingo runtime / memory allocator for this chip.
    if (bingo_hemaia_system_mmap_init() < 0)
    {
        printf("Chip(%x, %x): [Host] Error when initializing Allocator\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    }
    printf("Chip(%x, %x): [Host] Allocator Init Success\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y());

    // Cache the shared comm buffer and per-chip heap managers for later use.
    comm_buffer_ptr = (comm_buffer_t *)bingo_get_l2_comm_buffer(current_chip_id);
    local_l2_heap_manager = bingo_get_l2_heap_manager(current_chip_id);
    local_l3_heap_manager = bingo_get_l3_heap_manager(current_chip_id);
    (void)local_l2_heap_manager;

    return 0;
}

// -----------------------------------------------------------------------------
// Map a chip id to its slice index in the shared DRAM region.
// Chips 0x00 / 0x01 / 0x10 / 0x11 own slices 0 / 1 / 2 / 3 respectively.
// -----------------------------------------------------------------------------
static uint32_t chip_slice_index(uint8_t current_chip_id)
{
    switch (current_chip_id)
    {
    case 0x00:
        return 0;
    case 0x01:
        return 1;
    case 0x10:
        return 2;
    case 0x11:
        return 3;
    default:
        return 0;
    }
}

// -----------------------------------------------------------------------------
// Stream this chip's local `data` buffer into its dedicated DRAM slice.
// Issues NUM_DMA_STREAMS back-to-back blocking DMA copies so the four chips'
// streams overlap on the D2D fabric.
// -----------------------------------------------------------------------------
static void stream_data_to_dram(uint8_t current_chip_id, uint8_t *data_dest)
{
    printf("Chip(%x, %x): [Host] Data will be written to external DRAM at address %lx\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), (uintptr_t)data_dest);

    for (uint32_t i = 0; i < NUM_DMA_STREAMS; i++)
    {
        sys_dma_blk_memcpy(
            current_chip_id, (uintptr_t)data_dest, (uintptr_t)data, data_size);
        printf("%d\r\n", i);
    }
}

// -----------------------------------------------------------------------------
// Read back every chip's DRAM slice and compare the leading CHECKED_DATA_SIZE
// bytes against the golden `data`. Run by chip 0x00 only.
// Returns 0 if all slices match, -1 on the first mismatch or alloc failure.
// -----------------------------------------------------------------------------
static int verify_all_slices(uint8_t current_chip_id, uint8_t *dram_base)
{
    printf("All chips have finished DMA. Starting to check data correctness...\r\n");

    // Stage the read-back data in an L3 scratch buffer.
    uint8_t *data_to_be_checked =
        (uint8_t *)bingoHeapMalloc((uint64_t)local_l3_heap_manager, CHECKED_DATA_SIZE);
    if (data_to_be_checked == NULL)
    {
        printf("Chip(%x, %x): [Host] Error when allocating memory for data checking\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    }

    for (uint32_t slice = 0; slice < NUM_CHIPS; slice++)
    {
        // Copy the head of slice `slice` from DRAM into the scratch buffer.
        sys_dma_blk_memcpy(current_chip_id, (uintptr_t)data_to_be_checked,
                           (uintptr_t)(dram_base + data_size * slice),
                           CHECKED_DATA_SIZE);

        // Byte-wise compare against the golden data.
        for (uint32_t j = 0; j < CHECKED_DATA_SIZE; j++)
        {
            if (data_to_be_checked[j] != data[j])
            {
                printf("Data mismatch at chip %d, index %d: expected %d, got %d\n",
                       slice, j, data[j], data_to_be_checked[j]);
                return -1;
            }
        }
    }

    printf("All data is correct. \r\n");
    return 0;
}

int main()
{
    uint8_t current_chip_id = get_current_chip_id();

    // ---- 1. Per-chip bring-up ------------------------------------------------
    if (chip_init(current_chip_id) < 0)
    {
        return -1;
    }

    // ---- 2. Per-chip DMA stream ---------------------------------------------
    // Each chip targets its own disjoint slice of the shared DRAM region.
    uint8_t *dram_base = (uint8_t *)chiplet_addr_transform_loc(DRAM_DEST_X, DRAM_DEST_Y,
                                                 DRAM_DEST_LOCAL_ADDR);
    uint8_t *data_dest = dram_base + data_size * chip_slice_index(current_chip_id);
    stream_data_to_dram(current_chip_id, data_dest);

    // ---- 3. Barrier: wait for every chip to finish streaming ----------------
    chip_barrier(comm_buffer_ptr, CHIP_TOP_LEFT, CHIP_BOTTOM_RIGHT, 2);

    // ---- 4. Verification (chip 0x00 only) -----------------------------------
    if (current_chip_id == CHIP_TOP_LEFT)
    {
        if (verify_all_slices(current_chip_id, dram_base) < 0)
        {
            return -1;
        }
    }

    // ---- 5. Barrier: all chips exit the simulation together -----------------
    chip_barrier(comm_buffer_ptr, CHIP_TOP_LEFT, CHIP_BOTTOM_RIGHT, 3);

    return 0;
}
