// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "host.c"

int main() {
    // Reset and ungate all quadrants, deisolate
    uintptr_t address_prefix = (uintptr_t)get_current_chip_baseaddress();
    uint32_t chip_id = get_current_chip_id();

    init_uart(address_prefix, 50000000, 1000000);
    print_str(address_prefix, "[Occamy] The Offload main function \r\n");
    print_str(address_prefix, "[Occamy] Current Chip ID is: ");
    print_u8(address_prefix, chip_id);
    print_str(address_prefix, "\r\n");    
    reset_and_ungate_quadrants_all(chip_id);
    print_str(address_prefix, "[Occamy] Snitch ungated. \r\n");
    deisolate_all(chip_id);
    print_str(address_prefix, "[Occamy] Snitch deisolated. \r\n");
    // Enable interrupts to receive notice of job termination
    enable_sw_interrupts();
    // Program Snitch entry point and communication buffer
    program_snitches(chip_id);
    print_str(address_prefix, "[Occamy] Snitch Jump Address Programmed. \r\n");

    // Compiler fence to ensure Snitch entry point is
    // programmed before Snitches are woken up
    asm volatile("" ::: "memory");

    print_str(address_prefix, "[Occamy] Calling snitch cluster to execute the task \r\n");

    // Start Snitches
    wakeup_snitches_cl(chip_id);

    int ret = wait_snitches_done(chip_id);

    print_str(address_prefix, "[Occamy] Snitch cluster done with exit code ");
    print_u32(address_prefix, ret);
    print_str(address_prefix, "\r\n");

    // Wait for job done and return Snitch exit code
    return ret;
}
