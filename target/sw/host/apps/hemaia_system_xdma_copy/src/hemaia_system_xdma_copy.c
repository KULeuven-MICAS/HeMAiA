// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include "../data/data.h"
#include "host.h"
#include "libbingo/bingo_api.h"

comm_buffer_t* comm_buffer_ptr = NULL;
O1HeapInstance* l2_heap_manager = NULL;
O1HeapInstance* l3_heap_manager = NULL;
volatile O1HeapInstance* dram_heap_manager = NULL;

int main() {
    uint8_t current_chip_id = get_current_chip_id();
    // Program the Chiplet Topology
    hemaia_d2d_link_initialize(current_chip_id);
    // Init the uart for printf
    init_uart(get_current_chip_baseaddress(), 32, 1);
    enable_sw_interrupts();

    // Init bingo runtime
    if (bingo_hemaia_system_mmap_init() < 0) {
        printf("Chip(%x, %x): [Host] Error when initializing Allocator\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    } else {
        printf("Chip(%x, %x): [Host] Allocator Init Success\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
    }

    comm_buffer_ptr = (comm_buffer_t*)bingo_get_l2_comm_buffer(current_chip_id);
    l2_heap_manager =
        (O1HeapInstance*)bingo_get_l2_heap_manager(current_chip_id);
    l3_heap_manager =
        (O1HeapInstance*)bingo_get_l3_heap_manager(current_chip_id);

    // Init external DRAM heap
    if (get_current_chip_id() == 0) {
        dram_heap_manager = (volatile O1HeapInstance*)o1heapInit(
            (const uint64_t)chiplet_addr_transform_loc(2, 0,
                                                       SPM_WIDE_BASE_ADDR),
            64 * 1024 * 1024);  // 64 MB DRAM heap
        if (!dram_heap_manager) {
            printf("Chip(%x, %x): DRAM heap init failed!\r\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y());
            return -1;
        } else {
            printf("Chip(%x, %x): DRAM heap init succeeded at %lx\r\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y(),
                   (uintptr_t)dram_heap_manager);
        }
        *(volatile O1HeapInstance**)(chiplet_addr_transform_loc(
            0xF, 0xF, (uintptr_t)&dram_heap_manager)) = dram_heap_manager;
    } else {
        while (dram_heap_manager == NULL) {
            // Wait for chip 0 to initialize the dram_heap_manager
            asm volatile("fence" ::: "memory");
        }
        printf("Chip(%x, %x): DRAM heap manager is ready at %lx\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(),
               (uintptr_t)dram_heap_manager);
    }

    uint8_t* data_dest1;

    if (get_current_chip_id() == 0) {
        // Only chip 0 performs the XDMA copy test
        data_dest1 = (uint8_t*)o1heapAllocate((const uint64_t)dram_heap_manager,
                                              data_size);
        if (!data_dest1) {
            printf("The allocation of destination 1 at DRAM failed!\r\n");
            return -1;
        } else {
            printf("The allocation of destination 1 at DRAM succeed!\r\n");
            *(volatile uint8_t**)(chiplet_addr_transform_loc(
                0xF, 0xF, (uintptr_t)&data_dest1)) = data_dest1;
        }
        hemaia_xdma_memcpy_1d((uint8_t*)data, data_dest1, data_size);
        uint32_t task_id = hemaia_xdma_start();
        hemaia_xdma_remote_wait(task_id);
        printf("XDMA copy finished to destination 1! \r\n");
    }

    chip_barrier(comm_buffer_ptr, 0x00, 0x11, 1);

    uint8_t* data_dest2;
    data_dest2 =
        (uint8_t*)o1heapAllocate((const uint64_t)l3_heap_manager, data_size);
    if (!data_dest2) {
        printf("The allocation of destination 2 at local SRAM failed!\r\n");
        return -1;
    } else {
        printf(
            "The allocation of destination 2 at local SRAM succeed at %lx\r\n",
            (uintptr_t)data_dest2);
    }

    sys_dma_blk_memcpy(current_chip_id, (uintptr_t)data_dest2, (uintptr_t)data_dest1, data_size);
    printf("IDMA unicast copy finished to destination 2! \r\n");
    printf("Checking data correctness...\r\n");
    for (uint32_t i = 0; i < data_size; i++) {
        if (data[i] != data_dest2[i]) {
            printf("Data mismatch at index %d: expected %d, got %d\n", i,
                   data[i], data_dest2[i]);
            return -1;
        }
    }
    sys_dma_blk_memcpy(current_chip_id, (uintptr_t)data_dest2, (uintptr_t)WIDE_ZERO_MEM_BASE_ADDR, data_size);
    printf("Data at destination 2 is cleaned! \r\n");

    chip_barrier(comm_buffer_ptr, 0x00, 0x11, 2);

    if (get_current_chip_id() == 0) {
        sys_dma_blk_memcpy(
            0x20, chiplet_addr_transform_loc(0xF, 0xF, (uintptr_t)data_dest2),
            (uintptr_t)data_dest1, data_size);
        printf("IDMA broadcast copy finished to destination 2! \r\n");
    }
    chip_barrier(comm_buffer_ptr, 0x00, 0x11, 3);

    printf("Checking data correctness...\r\n");
    for (uint32_t i = 0; i < data_size; i++) {
        if (data[i] != data_dest2[i]) {
            printf("Data mismatch at index %d: expected %d, got %d\n", i,
                   data[i], data_dest2[i]);
            return -1;
        }
    }
    printf("Data check passed!\n");
    if (get_current_chip_id() == 0) {
        o1heapFree((const uint64_t)dram_heap_manager,
                   (const uint64_t)data_dest1);
    }
    o1heapFree((const uint64_t)l3_heap_manager, (const uint64_t)data_dest2);
    return 0;
}
