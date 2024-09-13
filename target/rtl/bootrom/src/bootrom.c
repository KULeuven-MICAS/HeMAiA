// Copyright 2024 KU Leuven
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Yunhao Deng <yunhao.deng@kuleuven.be>

// Please avoid using initialized global variable and static variable (.data
// region) in C. This will lead to the loss of initial value, as they are
// allocated in spm instead of bootrom.

// For global constant, use "const var"
// For values need to share between functions, use uninitialized global variable
// + initialization function

#include "uart.h"
#include "xmodem.h"

#define bool uint8_t
#define true 1
#define false 0

void delay_cycles(uint64_t cycle) {
    uint64_t target_cycle, current_cycle;

    __asm__ volatile("csrr %0, mcycle;" : "=r"(target_cycle));
    target_cycle = target_cycle + cycle;
    while (current_cycle < target_cycle) {
    __asm__ volatile(
                    "csrr %0, mcycle;"
                    : "=r"(current_cycle)
                    );
    }
}

// Boot modes.
enum boot_mode_t { JTAG, UART, PRINTMEM, NORMAL };

void bootrom() {
    enum boot_mode_t boot_mode = JTAG;
    uint64_t start_address;
    uint64_t memory_length;

    uint32_t chip_id;
    asm volatile("csrr %0, 0xf15" : "=r"(chip_id));
    uintptr_t address_prefix = ((uintptr_t)chip_id) << 40;

    char in_buf[8];
    init_uart(address_prefix, 50000000, 1000000);

    while (1) {
        start_address = 0x80000000 | ((uint64_t)address_prefix);
        print_uart(address_prefix, "\033[2J");
        print_uart(address_prefix, "\r\n\t\t Welcome to HeMAiA Bootrom");
        print_uart(address_prefix, "\r\n");
        print_uart(address_prefix, "\r\n\t Chip ID: 0x");
        print_uart_hex(address_prefix, (char*)&chip_id, 1, false);
        print_uart(address_prefix,
                   "\r\n\t Enter the number to select the mode: ");
        print_uart(address_prefix, "\r\n\t 1. Load from JTAG");
        print_uart(address_prefix, "\r\n\t 2. Load from UART to 0x");
        print_uart_hex(address_prefix, (char*)&start_address, 8, false);
        print_uart(address_prefix, "\r\n\t 3. Print memory from 0x");
        print_uart_hex(address_prefix, (char*)&start_address, 8, false);
        print_uart(address_prefix, "\r\n\t 4. Continue to Boot from 0x");
        print_uart_hex(address_prefix, (char*)&start_address, 8, false);

        boot_mode = read_serial(address_prefix) - '0' - 1;

        switch (boot_mode) {
            case JTAG:
                print_uart(address_prefix,
                           "\r\n\t Handover to debugger... \r\n\r\n");
                __asm__ volatile(
                    "csrr a0, mhartid;"
                    "ebreak;");
                break;

            case UART:
                delay_cycles(50000000);  // Delay for 1s
                uart_xmodem(address_prefix, start_address);
                print_uart(address_prefix, "\r\n\t Load finished. \r\n\r\n");
                break;

            case PRINTMEM:
                print_uart(address_prefix,
                           "\r\n\t Enter the size of the memory in byte: ");
                scan_uart(address_prefix, in_buf);
                char* cur = in_buf;
                memory_length = 0;
                while (*cur != '\0') {
                    memory_length = memory_length * 10 + *cur - '0';
                    cur++;
                }
                print_uart(address_prefix, "\r\n\t The memory from 0x");
                print_uart_hex(address_prefix, (char*)&start_address, 8, false);
                print_uart(address_prefix, "is:");
                print_uart_hex(address_prefix, (char*)start_address,
                               memory_length, true);
                print_uart(address_prefix, "\r\n\r\n\t Print finished. ");
                read_serial(address_prefix);
                break;

            case NORMAL:
                print_uart(address_prefix, "\r\n\t Booting at 0x");
                print_uart_hex(address_prefix, (char*)&start_address, 8, false);
                print_uart(address_prefix, "...\r\n\r\n");
                return;
                break;
        }
    }
}
