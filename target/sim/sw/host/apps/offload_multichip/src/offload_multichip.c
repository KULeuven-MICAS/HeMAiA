// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "host.h"

int32_t parse_hex(const char* str) {
    int32_t value = 0;
    const char* start = str;
    while (*str != '\0') {
        if (*str >= '0' && *str <= '9') {
            value = (value << 4) + (*str - '0');
        } else if (*str >= 'A' && *str <= 'F') {
            value = (value << 4) + (*str - 'A' + 10);
        } else if (*str >= 'a' && *str <= 'f') {
            value = (value << 4) + (*str - 'a' + 10);
        } else {
            return -1;  // Invalid character
        }
        str++;
    }
    if (str == start)
        return -2;  // No input
    else
        return value;
}

int main() {
    // The pointer to the communication buffer
    volatile comm_buffer_t* comm_buffer_ptr = (comm_buffer_t*)0;

    // Reset and ungate all quadrants, deisolate
    uintptr_t current_chip_address_prefix =
        (uintptr_t)get_current_chip_baseaddress();
    uint32_t current_chip_id = get_current_chip_id();
    int32_t target_chip_id = 0;
    char in_buf[8];

    init_uart(current_chip_address_prefix, 32, 1);
    print_str(current_chip_address_prefix, "[HeMAiA] Current Chip ID is: ");
    print_u8(current_chip_address_prefix, current_chip_id);
    print_str(current_chip_address_prefix, "\r\n");

    print_str(current_chip_address_prefix, "[HeMAiA] Max X of SoP: ");
    scan_str(current_chip_address_prefix, in_buf);
    print_str(current_chip_address_prefix, "\r\n");
    uint32_t max_x = parse_hex(in_buf);
    print_str(current_chip_address_prefix, "[HeMAiA] Max Y of SoP: ");
    scan_str(current_chip_address_prefix, in_buf);
    print_str(current_chip_address_prefix, "\r\n");
    uint32_t max_y = parse_hex(in_buf);

    for (uint32_t x = 0; x <= max_x; x++) {
        for (uint32_t y = 0; y <= max_y; y++) {
            // Reset and ungate all quadrants, deisolate
            reset_and_ungate_quadrants_all((x << 4) + y);
            deisolate_all((x << 4) + y);
        }
    }

    print_str(current_chip_address_prefix,
              "[HeMAiA] Chip ID to execute binary: ");
    scan_str(current_chip_address_prefix, in_buf);
    print_str(current_chip_address_prefix, "\r\n");

    target_chip_id = parse_hex(in_buf);
    if (target_chip_id < 0) {
        print_str(current_chip_address_prefix,
                  "[HeMAiA] Invalid Chip ID. Exiting.\r\n");
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
