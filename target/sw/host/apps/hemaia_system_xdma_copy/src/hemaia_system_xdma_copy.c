// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include "../data/data.h"
#include "host.h"
#include "libbingo/bingo_api.h"

O1HeapInstance* dram_heap_manager = NULL;
comm_buffer_t* comm_buffer_ptr = NULL;

int main() {
    // Set clk manager to 1 division for a faster simulation time
    enable_clk_domain(0, 1);
    enable_clk_domain(1, 1);
    enable_clk_domain(2, 1);
    enable_clk_domain(3, 1);
    enable_clk_domain(4, 1);
    enable_clk_domain(5, 1);
    uintptr_t current_chip_address_prefix =
        (uintptr_t)get_current_chip_baseaddress();
    uint32_t current_chip_id = get_current_chip_id();

    // Init the uart for printf and enable software interrupts
    init_uart(current_chip_address_prefix, 32, 1);
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

    comm_buffer_ptr = bingo_get_l2_comm_buffer(current_chip_id);

    // Init external DRAM heap
    if (get_current_chip_id() == 0) {
        dram_heap_manager = o1heapInit(
            (void* const)(chiplet_addr_transform_loc(2, 0, SPM_WIDE_BASE_ADDR)),
            64 * 1024 * 1024);  // 64 MB DRAM heap
        if (!dram_heap_manager) {
            printf("Chip(%x, %x): DRAM heap init failed!\r\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y());
            return -1;
        } else {
            printf("Chip(%x, %x): DRAM heap init succeeded!\r\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y());
        }
        *(volatile O1HeapInstance**)(chiplet_addr_transform_loc(
            0xF, 0xF, (uintptr_t)&dram_heap_manager)) = dram_heap_manager;
    } else {
        while (dram_heap_manager == NULL) {
            // Wait for chip 0 to initialize the dram_heap_manager
            asm volatile("fence" ::: "memory");
        }
        printf("Chip(%x, %x): DRAM heap manager is ready!\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
    }

    chip_barrier(comm_buffer_ptr, 0x00, 0x10, 1);

    uint8_t* data_dest1;
    data_dest1 = (uint8_t*)o1heapAllocate(dram_heap_manager, data_size);
    if (!data_dest1) {
        printf("The allocation of destination 1 at DRAM failed!\r\n");
        return -1;
    } else {
        printf("The allocation of destination 1 at DRAM succeed!\r\n");
    }

    hemaia_xdma_memcpy_1d((uint8_t*)data, data_dest1, data_size);
    uint32_t task_id = hemaia_xdma_start();
    hemaia_xdma_remote_wait(task_id);
    printf("XDMA copy finished to destination 1! \r\n");

    uint8_t* data_dest2;
    data_dest2 = (uint8_t*)o1heapAllocate(l3_heap_manager, data_size);
    if (!data_dest2) {
        printf("The allocation of destination 2 at local SRAM failed!\r\n");
        return -1;
    } else {
        printf("The allocation of destination 2 at local SRAM succeed!\r\n");
    }

    hemaia_xdma_memcpy_1d((uint8_t*)data_dest1, data_dest2, data_size);
    task_id = hemaia_xdma_start();
    hemaia_xdma_remote_wait(task_id);
    printf("XDMA copy finished to destination 2! \r\n");

    printf("Checking data correctness...\r\n");
    for (uint32_t i = 0; i < data_size; i++) {
        if (data[i] != data_dest2[i]) {
            printf("Data mismatch at index %d: expected %d, got %d\n", i,
                   data[i], data_dest2[i]);
            return -1;
        }
    }
    printf("Data check passed!\n");
    o1heapFree(dram_heap_manager, data_dest1);
    o1heapFree(l3_heap_manager, data_dest2);
    return 0;
}
