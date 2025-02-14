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
    uint32_t current_chip_id = get_current_chip_id();
    uint32_t target_chip_id = 0;
    char in_buf[8];

    init_uart(current_chip_address_prefix, 50000000, 1000000);
    print_str(current_chip_address_prefix,
              "[HeMAiA] The Offload main function \r\n");
    print_str(current_chip_address_prefix, "[HeMAiA] Current Chip ID is: ");
    print_u8(current_chip_address_prefix, current_chip_id);
    print_str(current_chip_address_prefix, "\r\n");
    print_str(current_chip_address_prefix, "[HeMAiA] Enter target Chip ID: ");
    scan_str(current_chip_address_prefix, in_buf);
    print_str(current_chip_address_prefix, "\r\n");

    char* cur = in_buf;

    while (*cur != '\0') {
        if (*cur >= '0' && *cur <= '9') {
            target_chip_id = (target_chip_id << 4) + *cur - '0';
        } else if (*cur >= 'A' && *cur <= 'F') {
            target_chip_id = (target_chip_id << 4) + *cur - 'A' + 10;
        } else if (*cur >= 'a' && *cur <= 'f') {
            target_chip_id = (target_chip_id << 4) + *cur - 'a' + 10;
        } else {
            print_str(current_chip_address_prefix,
                      "[HeMAiA] Invalid target chip ID. \r\n");
            scan_char(current_chip_address_prefix);
            break;
        }
        cur++;
    }

    uintptr_t target_chip_address_prefix =
        (uintptr_t)get_chip_baseaddress(target_chip_id);

    // print_str(current_chip_address_prefix, "[HeMAiA] Target Chip Address
    // Prefix is: "); print_u48(current_chip_address_prefix,
    // (uint64_t)target_chip_address_prefix);

    comm_buffer_ptr = (comm_buffer_t*)(((uint64_t)&__narrow_spm_start) |
                                       target_chip_address_prefix);

    // print_str(current_chip_address_prefix,
    //           "[HeMAiA] Snitch Communication Buffer is: ");
    // print_u48(current_chip_address_prefix, (uint64_t)comm_buffer_ptr);
    // print_str(current_chip_address_prefix, "\r\n");
    reset_and_ungate_quadrants_all(target_chip_id);
    // print_str(current_chip_address_prefix, "[HeMAiA] Snitch ungated. \r\n");
    deisolate_all(target_chip_id);
    // print_str(current_chip_address_prefix, "[HeMAiA] Snitch deisolated.
    // \r\n"); Enable interrupts to receive notice of job termination
    enable_sw_interrupts();
    // Program Snitch entry point and communication buffer
    comm_buffer_ptr->lock = 0;
    comm_buffer_ptr->chip_id = current_chip_id;
    program_snitches(target_chip_id, comm_buffer_ptr);
    // print_str(current_chip_address_prefix,
    //           "[HeMAiA] Snitch Jump Address Programmed. \r\n");

    // Compiler fence to ensure Snitch entry point is
    // programmed before Snitches are woken up
    asm volatile("fence.i" ::: "memory");

    print_str(current_chip_address_prefix,
              "[HeMAiA] Calling snitch cluster to execute the task \r\n");

    // Start Snitches
    wakeup_snitches_cl(target_chip_id);

    int ret = wait_snitches_done(target_chip_id);

    print_str(current_chip_address_prefix,
              "[HeMAiA] Snitch cluster done with exit code ");
    print_u32(current_chip_address_prefix, ret);
    print_str(current_chip_address_prefix, "\r\n");

    // Wait for job done and return Snitch exit code
    return ret;
}
