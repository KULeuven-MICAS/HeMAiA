// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "host.h"

int main() {
    // The pointer to the communication buffer
    volatile comm_buffer_t* comm_buffer_ptr = (comm_buffer_t*)0;

    // Reset and ungate all quadrants, deisolate
    uintptr_t current_chip_address_prefix =
        (uintptr_t)get_current_chip_baseaddress();
    comm_buffer_ptr = (comm_buffer_t*)(((uint64_t)&__narrow_spm_start) |
                                       current_chip_address_prefix);
    // Initialize the communication buffer
    initialize_comm_buffer((comm_buffer_t*)comm_buffer_ptr);
    // Initialize the UART
    init_uart(current_chip_address_prefix, 32, 1);
    // print_str(current_chip_address_prefix,
    //           "[HeMAiA] The multi-chip synchronization tester \r\n");

    for (uint8_t i = 1;; i++) {
        printf("[HeMAiA] Press to run barrier %d \r\n", i);
        scan_char(current_chip_address_prefix);
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
