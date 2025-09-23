// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "host.h"

// Global Variables for communication buffer
volatile comm_buffer_t* comm_buffer_ptr = (comm_buffer_t*)0;

int main() {
    // Set clk manager to 1 division for a faster simulation time
    enable_clk_domain(0, 1);
    enable_clk_domain(1, 1);
    enable_clk_domain(2, 1);
    enable_clk_domain(3, 1);
    enable_clk_domain(4, 1);
    enable_clk_domain(5, 1);

    // Reset and ungate all quadrants, deisolate
    uintptr_t current_chip_address_prefix =
        (uintptr_t)get_current_chip_baseaddress();
    uint32_t current_chip_id = get_current_chip_id();

    init_uart(current_chip_address_prefix, 32, 1);
    comm_buffer_ptr = (comm_buffer_t*)(((uint64_t)&__narrow_spm_start) |
                                       current_chip_address_prefix);
    deisolate_all(current_chip_id);
    enable_sw_interrupts();
    comm_buffer_ptr->lock = 0;
    comm_buffer_ptr->chip_id = current_chip_id;
    program_snitches(current_chip_id, comm_buffer_ptr);
    asm volatile("fence.i" ::: "memory");

    printf("[HeMAiA] Calling snitch cluster on chip %d to execute the task\r\n",
           current_chip_id);

    // Start Snitches
    wakeup_snitches_cl(current_chip_id);

    int ret = wait_snitches_done(current_chip_id);

    printf("[HeMAiA] Snitch cluster done with exit code %d\r\n", ret);

    // Wait for job done and return Snitch exit code
    return ret;
}
