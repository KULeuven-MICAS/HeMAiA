// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#include "offload_dep.h"

int main() {
    // Set clk manager to 1 division for a faster simulation time
    enable_clk_domain(0, 1);
    enable_clk_domain(1, 1);
    // Reset and ungate all quadrants, deisolate
    initialize_narrow_spm();
    uintptr_t current_chip_address_prefix =
        (uintptr_t)get_current_chip_baseaddress();
    uint32_t current_chip_id = get_current_chip_id();

    init_uart(current_chip_address_prefix, 32, 1);
    comm_buffer_ptr = (comm_buffer_t*)(((uint64_t)&__narrow_spm_start) |
                                       current_chip_address_prefix);

    // Here we init the Dev
    host_init_dev();

    // Turn on the devs
    reset_and_ungate_quadrants_all(current_chip_id);
    // Set the axi isolation
    deisolate_all(current_chip_id);
    // Clear the cluster tcdm
    initialize_all_clusters();
    // Turn on the software interrupts for host core
    enable_sw_interrupts();
    // Set the comm buffer between host and snitch cluster
    comm_buffer_ptr->lock = 0;
    comm_buffer_ptr->chip_id = current_chip_id;
    program_snitches(current_chip_id, comm_buffer_ptr);
    asm volatile("fence.i" ::: "memory");

    printf("[HeMAiA] Calling snitch cluster on chip %d to execute the task\r\n",
           current_chip_id);

    // Start Snitches
    wakeup_snitches_cl(current_chip_id);
    
    int ret = test_kernel_execution();

    printf("[HeMAiA] Snitch cluster done with exit code %d\r\n", ret);

    // Wait for job done and return Snitch exit code
    return ret;
}
