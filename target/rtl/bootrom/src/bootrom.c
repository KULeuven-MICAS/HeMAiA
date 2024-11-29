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

#include "chip_id.h"
#include "sys_dma.h"
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
        __asm__ volatile("csrr %0, mcycle;" : "=r"(current_cycle));
    }
}

// Boot modes.
enum boot_mode_t {
    TARGET_CHIPID,
    UART,
    COPY_TO_REMOTE,
    COPY_FROM_REMOTE,
    PRINTMEM,
    NORMAL
};

void bootrom() {
    enum boot_mode_t boot_mode = NORMAL;
    uint64_t local_chip_mem_start_address;
    uint64_t remote_chip_mem_start_address;
    uint64_t memory_length;

    uint32_t chip_id = get_current_chip_id();
    uint32_t target_chip_id = chip_id;
    uintptr_t address_prefix = ((uintptr_t)chip_id) << 40;

    char in_buf[8];
    init_uart(address_prefix, 50000000, 1000000);

    while (1) {
        local_chip_mem_start_address = 0x80000000L | ((uint64_t)address_prefix);
        remote_chip_mem_start_address =
            0x80000000L | ((uint64_t)target_chip_id << 40);
        print_str(address_prefix, "\033[2J");
        print_str(address_prefix, "\r\n\t\t Welcome to HeMAiA Bootrom");
        print_str(address_prefix, "\r\n");
        print_str(address_prefix, "\r\n\t Chip ID: 0x");
        print_u8(address_prefix, chip_id);
        print_str(address_prefix, "\r\n\t Target Remote Chip ID: 0x");
        print_u8(address_prefix, target_chip_id);
        print_str(address_prefix,
                  "\r\n\t Enter the number to select the mode: ");
        print_str(address_prefix, "\r\n\t 1. Change the target remote Chip ID");
        print_str(address_prefix, "\r\n\t 2. Load from UART to 0x");
        print_u48(address_prefix, remote_chip_mem_start_address);
        print_str(address_prefix,
                  "\r\n\t 3. Copy memory from local chip to remote chip");
        print_str(address_prefix,
                  "\r\n\t 4. Copy memory from remote chip to local chip");
        print_str(address_prefix, "\r\n\t 5. Print memory from 0x");
        print_u48(address_prefix, remote_chip_mem_start_address);
        print_str(address_prefix, "\r\n\t 6. Continue to Boot from 0x");
        print_u48(address_prefix, local_chip_mem_start_address);
        print_str(address_prefix, "\r\n");

        boot_mode = getchar(address_prefix) - '0' - 1;

        char* cur = 0;

        switch (boot_mode) {
            case TARGET_CHIPID:
                print_str(address_prefix,
                          "\r\n\t Enter the target remote Chip ID: ");
                scan_uart(address_prefix, in_buf);
                cur = in_buf;
                target_chip_id = 0;
                while (*cur != '\0') {
                    if (*cur >= '0' || *cur <= '9') {
                        target_chip_id = (target_chip_id << 4) + *cur - '0';
                    } else if (*cur >= 'A' || *cur <= 'F') {
                        target_chip_id =
                            (target_chip_id << 4) + *cur - 'A' + 10;
                    } else if (*cur >= 'a' || *cur <= 'f') {
                        target_chip_id =
                            (target_chip_id << 4) + *cur - 'a' + 10;
                    } else {
                        print_str(address_prefix, "\r\n\t Invalid input. ");
                        getchar(address_prefix);
                        break;
                    }
                    cur++;
                }
                break;

            case COPY_TO_REMOTE:
                print_str(address_prefix,
                          "\r\n\t Enter the size of the memory in byte: ");
                scan_uart(address_prefix, in_buf);
                cur = in_buf;
                memory_length = 0;
                while (*cur != '\0') {
                    memory_length = memory_length * 10 + *cur - '0';
                    cur++;
                }
                print_str(address_prefix, "\r\n\t Copying memory from 0x");
                print_u48(address_prefix, local_chip_mem_start_address);
                print_str(address_prefix, " to 0x");
                print_u48(address_prefix, remote_chip_mem_start_address);
                print_str(address_prefix, " with size ");
                print_u48(address_prefix, memory_length);
                print_str(address_prefix, " bytes...");
                sys_dma_blk_memcpy(remote_chip_mem_start_address,
                                   local_chip_mem_start_address, memory_length);
                print_str(address_prefix, "\r\n\t Copy finished. ");
                getchar(address_prefix);
                break;

            case COPY_FROM_REMOTE:
                print_str(address_prefix,
                          "\r\n\t Enter the size of the memory in byte: ");
                scan_uart(address_prefix, in_buf);
                cur = in_buf;
                memory_length = 0;
                while (*cur != '\0') {
                    memory_length = memory_length * 10 + *cur - '0';
                    cur++;
                }
                print_str(address_prefix, "\r\n\t Copying memory from 0x");
                print_u48(address_prefix, remote_chip_mem_start_address);
                print_str(address_prefix, " to 0x");
                print_u48(address_prefix, local_chip_mem_start_address);
                print_str(address_prefix, " with size ");
                print_u48(address_prefix, memory_length);
                print_str(address_prefix, " bytes...");
                sys_dma_blk_memcpy(local_chip_mem_start_address,
                                   remote_chip_mem_start_address,
                                   memory_length);
                print_str(address_prefix, "\r\n\t Copy finished. ");
                getchar(address_prefix);
                break;

            case UART:
                delay_cycles(50000000);  // Delay for 1s
                uart_xmodem(address_prefix, remote_chip_mem_start_address);
                print_str(address_prefix, "\r\n\t Load finished. \r\n\r\n");
                break;

            case PRINTMEM:
                print_str(address_prefix,
                          "\r\n\t Enter the size of the memory in byte: ");
                scan_uart(address_prefix, in_buf);
                char* cur = in_buf;
                memory_length = 0;
                while (*cur != '\0') {
                    memory_length = memory_length * 10 + *cur - '0';
                    cur++;
                }
                print_str(address_prefix, "\r\n\t The memory from 0x");
                print_u48(address_prefix, remote_chip_mem_start_address);
                print_str(address_prefix, " is:");
                print_mem_hex(address_prefix,
                              (char*)remote_chip_mem_start_address,
                              memory_length);
                print_str(address_prefix, "\r\n\r\n\t Print finished. ");
                getchar(address_prefix);
                break;

            case NORMAL:
                print_str(address_prefix, "\033[2J");
                print_str(address_prefix, "\r\n\t Booting at 0x");
                print_u48(address_prefix, local_chip_mem_start_address);
                print_str(address_prefix, "...\r\n\r\n\r\n");
                return;
                break;
        }
    }
}
