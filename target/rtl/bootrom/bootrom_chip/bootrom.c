// Copyright 2024 KU Leuven
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Yunhao Deng <yunhao.deng@kuleuven.be>

// Simplified Bootrom with reduced printing for faster simulation

// Please avoid using initialized global variable and static variable (.data
// region) in C. This will lead to the loss of initial value, as they are
// allocated in spm instead of bootrom.

// For global constant, use "const var"
// For values need to share between functions, use uninitialized global variable
// + initialization function

#include "uart.c"
#include "chip_id.h"
#include "sys_dma_bootrom.h"
#include "xmodem.h"

#define bool uint8_t
#define true 1
#define false 0

static inline void delay_cycles(uint64_t cycle) {
    uint64_t target_cycle, current_cycle;
    __asm__ volatile("csrr %0, mcycle;" : "=r"(current_cycle));
    target_cycle = current_cycle + cycle;
    while (current_cycle < target_cycle) {
        __asm__ volatile("csrr %0, mcycle;" : "=r"(current_cycle));
    }
}

inline static void print_mem_hex(uintptr_t address_prefix, char *str,
                                 uint32_t length) {
    uint8_t lut[16] = {'0', '1', '2', '3', '4', '5', '6', '7',
                       '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
    for (uint64_t i = (uint64_t)str; i < (uint64_t)str + length; i++) {
        if (i % 16 == 0) {
            print_char(address_prefix, '\r');
            print_char(address_prefix, '\n');
            for (int j = 28; j >= 0; j = j - 4)
                print_char(address_prefix, lut[(i >> j) % 16]);
            print_char(address_prefix, ':');
            print_char(address_prefix, ' ');
        }
        char temp = *((char *)i);
        print_char(address_prefix, lut[temp / 16]);
        print_char(address_prefix, lut[temp % 16]);
        print_char(address_prefix, ' ');
    }
    while (!is_transmit_done(address_prefix));
}

// Boot modes.
enum boot_mode_t {
    HALT,
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
    init_uart(address_prefix, 32, 1);

    while (1) {
        local_chip_mem_start_address = 0x80000000L | ((uint64_t)address_prefix);
        remote_chip_mem_start_address =
            0x80000000L | ((uint64_t)target_chip_id << 40);
        printf("\033[2J\r\n\t\tBootrom\r\n\r\n\t Chip ID: 0x%02X, 0x%02X\r\n",
               chip_id, target_chip_id);
        // print_str(address_prefix, "\r\n\t 1. Halt the CVA6 Core");
        // print_str(address_prefix, "\r\n\t 2. Change the target remote Chip
        // ID"); print_str(address_prefix, "\r\n\t 3. Load from UART");
        // print_str(address_prefix,
        //           "\r\n\t 4. Copy memory from local chip to remote chip");
        // print_str(address_prefix,
        //           "\r\n\t 5. Copy memory from remote chip to local chip");
        // print_str(address_prefix, "\r\n\t 6. Print memory from 0x");
        // print_str(address_prefix, "\r\n\t 7. Continue to Boot from 0x");
        // print_str(address_prefix, "\r\n");

        boot_mode = scan_char(address_prefix) - '0' - 1;

        char* cur = 0;

        switch (boot_mode) {
            case HALT:
                printf("\r\n\t CVA6 Core is Halted. ");
                scan_char(address_prefix);
                __asm__ volatile("wfi");
                break;
            case TARGET_CHIPID:
                printf("\r\n\t Target Chip ID: ");
                scanf("%x", &target_chip_id);
                break;
            case COPY_TO_REMOTE:
                // print_str(address_prefix, "\r\n\t Memory Size: ");
                // scan_uart(address_prefix, in_buf);
                // cur = in_buf;
                memory_length = 1048576;
                // while (*cur != '\0') {
                //     memory_length = memory_length * 10 + *cur - '0';
                //     cur++;
                // }
                printf("\r\n\t Copying...");
                sys_dma_blk_memcpy(remote_chip_mem_start_address,
                                   local_chip_mem_start_address, memory_length);
                printf("\r\n\t Copy finished. ");
                scan_char(address_prefix);
                break;

            case COPY_FROM_REMOTE:
                // print_str(address_prefix, "\r\n\t Memory Size: ");
                // scan_uart(address_prefix, in_buf);
                // cur = in_buf;
                memory_length = 1048576;
                // while (*cur != '\0') {
                //     memory_length = memory_length * 10 + *cur - '0';
                //     cur++;
                // }
                printf("\r\n\t Copying...");
                sys_dma_blk_memcpy(local_chip_mem_start_address,
                                   remote_chip_mem_start_address,
                                   memory_length);
                printf("\r\n\t Copy finished. ");
                scan_char(address_prefix);
                break;

            case UART:
                delay_cycles(50000000);  // Delay for 1s
                uart_xmodem(address_prefix, remote_chip_mem_start_address);
                printf("\r\n\t Load finished. \r\n\r\n");
                break;

            case PRINTMEM:
                printf("\r\n\t Memory Size: ");
                scanf("%d", &memory_length);
                print_mem_hex(address_prefix,
                              (char*)remote_chip_mem_start_address,
                              memory_length);
                printf("\r\n\r\n\t Print finished. ");
                scan_char(address_prefix);
                break;

            case NORMAL:
                printf("\033[2J");
                printf("\r\n\t Booting...\r\n\r\n\r\n");
                return;
                break;
        }
    }
}
