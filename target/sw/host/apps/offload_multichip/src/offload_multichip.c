// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include "host.h"

int main() {
    // Set clk manager to 1 division for a faster simulation time
    enable_clk_domain(0, 1);
    enable_clk_domain(1, 1);
    // The pointer to the communication buffer
    volatile comm_buffer_t* comm_buffer_ptr = (comm_buffer_t*)0;

    // Reset and ungate all quadrants, deisolate
    uintptr_t current_chip_address_prefix =
        (uintptr_t)get_current_chip_baseaddress();
    uint32_t current_chip_id = get_current_chip_id();
    int32_t target_chip_id = 0;

    init_uart(current_chip_address_prefix, 32, 1);
    printf("[HeMAiA] Current Chip ID is: %x%x\r\n", current_chip_id >> 4,
           current_chip_id & 0x0F);

    printf("[HeMAiA] Max X of SoP: \r\n");
    uint32_t max_x = 0;
    scanf("%x", &max_x);
    uint32_t max_y = 0;
    printf("[HeMAiA] Max Y of SoP: \r\n");
    scanf("%x", &max_y);

    for (uint32_t x = 0; x <= max_x; x++) {
        for (uint32_t y = 0; y <= max_y; y++) {
            // Reset and ungate all quadrants, deisolate
            reset_and_ungate_quadrants_all((x << 4) + y);
            deisolate_all((x << 4) + y);
        }
    }

    printf("[HeMAiA] Chip ID to execute binary: \r\n");
    scanf("%x", &target_chip_id);
    if (target_chip_id < 0) {
        printf("[HeMAiA] Invalid Chip ID. Exiting.\r\n");
        return -1;  // Invalid chip ID, exit
    }

    uintptr_t target_chip_address_prefix =
        (uintptr_t)get_chip_baseaddress(target_chip_id);

    comm_buffer_ptr = (comm_buffer_t*)(((uint64_t)&__narrow_spm_start) |
                                       target_chip_address_prefix);

    enable_sw_interrupts();
    // Program Snitch entry point and communication buffer
    comm_buffer_ptr->lock = 0;
    comm_buffer_ptr->chip_id = current_chip_id;
    program_snitches(target_chip_id, comm_buffer_ptr);

    asm volatile("fence" ::: "memory");

    printf("[HeMAiA] Calling snitch cluster on chip %d to execute the task\r\n",
           target_chip_id);

    // Start Snitches
    wakeup_snitches_cl(target_chip_id);

    int ret = wait_snitches_done(target_chip_id);

    printf("[HeMAiA] Snitch cluster done with exit code %d\r\n", ret);

    // Wait for job done and return Snitch exit code
    return ret;
}
