// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include "host.h"

int main() {
    // The pointer to the communication buffer
    volatile comm_buffer_t* comm_buffer_ptr = (comm_buffer_t*)0;

    comm_buffer_ptr = (comm_buffer_t*)chiplet_addr_transform(((uint64_t)&__narrow_spm_start));

    // Initialize the communication buffer
    initialize_comm_buffer((comm_buffer_t*)comm_buffer_ptr);
    // Initialize the UART
    init_uart(get_current_chip_baseaddress(), 32, 1);
    // Enable vector extension
    enable_vec();
    // Initialize the D2D link topology
    hemaia_d2d_link_initialize(get_current_chip_id());

    for (uint8_t i = 1;; i++) {
        printf("[HeMAiA] Press to run barrier %d \r\n", i);
        scan_char(get_current_chip_baseaddress());
        uint64_t cycle_num_before = mcycle();
        // Barrier
        chip_barrier(comm_buffer_ptr, 0x00, 0x11, i);
        uint64_t cycle_num_after = mcycle();
        // Print the cycle number before and after the barrier
        printf("[HeMAiA] Barrier %d done, cycle before: %x, after: %x\r\n", i,
               cycle_num_before, cycle_num_after);
    }
    // Wait for job done and return Snitch exit code
    return 0;
}
