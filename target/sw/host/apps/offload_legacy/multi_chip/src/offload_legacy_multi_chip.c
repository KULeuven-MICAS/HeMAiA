// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include "host.h"
#include "hemaia_clk_rst_controller.h"

int main() {
    // The pointer to the communication buffer
    volatile comm_buffer_t* comm_buffer_ptr = (comm_buffer_t*)0;
    printf("[HeMAiA] Multi-chip Offload Legacy Main\r\n");
    // Reset and ungate all quadrants, deisolate
    uintptr_t current_chip_address_prefix =
        (uintptr_t)get_current_chip_baseaddress();
    uint32_t current_chip_id = get_current_chip_id();
    hemaia_d2d_link_initialize(current_chip_id);
    int32_t target_chip_id = 0;

    init_uart(current_chip_address_prefix, 32, 1);
    printf("[HeMAiA] Current Chip ID is: %x%x\r\n", current_chip_id >> 4,
           current_chip_id & 0x0F);

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
