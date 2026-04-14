// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include "../data/data.h"
#include "host.h"
#include "libbingo/bingo_api.h"

comm_buffer_t *comm_buffer_ptr = NULL;
uint64_t local_l2_heap_manager = 0;
uint64_t local_l3_heap_manager = 0;
volatile uint64_t dram_heap_manager = 0;
#define checked_data_size 64

int main()
{
    uint8_t current_chip_id = get_current_chip_id();
    // Program the Chiplet Topology
    hemaia_d2d_link_initialize(current_chip_id);
    // Init the uart for printf
    init_uart(get_current_chip_baseaddress(), 32, 1);
    enable_sw_interrupts();

    // Init bingo runtime
    if (bingo_hemaia_system_mmap_init() < 0)
    {
        printf("Chip(%x, %x): [Host] Error when initializing Allocator\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    }
    else
    {
        printf("Chip(%x, %x): [Host] Allocator Init Success\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
    }

    comm_buffer_ptr = (comm_buffer_t *)bingo_get_l2_comm_buffer(current_chip_id);
    local_l2_heap_manager = bingo_get_l2_heap_manager(current_chip_id);
    local_l3_heap_manager = bingo_get_l3_heap_manager(current_chip_id);
    (void)local_l2_heap_manager;

    // External DRAM: Chip 0 - 3 get the data from 0x200080000000 + 0 / 64 / 128 / 192KB
    uint8_t *data_dest;
    data_dest = (uint8_t *)chiplet_addr_transform_loc(0x2, 0x0, 0x80000000);
    if (current_chip_id == 0x00)
    {
        data_dest = data_dest + 0;
    }
    else if (current_chip_id == 0x01)
    {
        data_dest = data_dest + data_size;
    }
    else if (current_chip_id == 0x10)
    {
        data_dest = data_dest + data_size * 2;
    }
    else if (current_chip_id == 0x11)
    {
        data_dest = data_dest + data_size * 3;
    }
    printf("Chip(%x, %x): [Host] Data will be written to external DRAM at address %lx\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), (uintptr_t)data_dest);

    for (uint32_t i = 0; i < 16; i++)
    {
        sys_dma_blk_memcpy(
            current_chip_id, (uintptr_t)data_dest, (uintptr_t)data, data_size);

        if (i % 4 == 0)
        {
            printf("%d\r\n", i);
        }
    }

    // Wait for all chips to finish the DMA
    chip_barrier(comm_buffer_ptr, 0x00, 0x11, 2);

    if (current_chip_id == 0x00)
    {
        printf("All chips have finished DMA. Starting to check data correctness...\r\n");
        uint8_t *data_to_be_checked = (uint8_t *)bingoHeapMalloc((uint64_t)local_l3_heap_manager, checked_data_size);
        if (data_to_be_checked == NULL)
        {
            printf("Chip(%x, %x): [Host] Error when allocating memory for data checking\r\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y());
            return -1;
        }
        for (uint32_t i = 0; i < 4; i++)
        {
            sys_dma_blk_memcpy(current_chip_id, (uintptr_t)data_to_be_checked, (uintptr_t)(data_dest + data_size * i), checked_data_size);
            for (uint32_t j = 0; j < checked_data_size; j++) // Only check the first 64 bytes for quick verification, can check more if needed
            {
                if (data_to_be_checked[j] != data[j])
                {
                    printf("Data mismatch at chip %d, index %d: expected %d, got %d\n", i,
                           j, data[j], data_to_be_checked[j]);
                    return -1;
                }
            }
        }
        printf("All data is correct. \r\n");
    }

    // All chips exit the simulation at the same time
    chip_barrier(comm_buffer_ptr, 0x00, 0x11, 3);

    return 0;
}
