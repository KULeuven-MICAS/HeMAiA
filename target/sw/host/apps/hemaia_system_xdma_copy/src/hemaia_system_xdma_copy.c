// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include "../data/data.h"
#include "host.h"
#include "libhero/hero_api.h"

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

    // Heap Initialization
    // Start address is the end of the .bss section
    // Allocate half of the SPM_WIDE_SIZE to L2 heap
    extern size_t l3_heap_size;
    extern uintptr_t l3_heap_start_phy;
    extern uintptr_t l3_heap_start_virt;

    l3_heap_start_phy =
        (uintptr_t)chiplet_addr_transform((uint64_t)(&__l3_heap_start));
    l3_heap_start_virt =
        (uintptr_t)chiplet_addr_transform((uint64_t)(&__l3_heap_start));
    l3_heap_size = SPM_WIDE_SIZE / 2;

    uint8_t* data_dest;
    if (hero_dev_l3_init() == 0) {
        data_dest = (uint8_t*)hero_host_l3_malloc(data_size, NULL);
    } else {
        printf("L3 init failed!\n");
        return -1;
    }
    if (data_dest == NULL) {
        printf("L3 malloc failed!\n");
        return -1;
    }

    printf("L3 malloc succeeded! Start to call XDMA to copy the data\n");

    hemaia_xdma_memcpy_1d((uint8_t*)data, data_dest, data_size);
    uint32_t task_id = hemaia_xdma_start();
    hemaia_xdma_local_wait(task_id);
    printf("XDMA copy finished! Start to check the results\n");
    for (uint32_t i = 0; i < data_size; i++) {
        if (data[i] != data_dest[i]) {
            printf("Data mismatch at index %d: expected %d, got %d\n", i,
                   data[i], data_dest[i]);
            return -1;
        }
    }
    printf("Data check passed!\n");
    return 0;
}
