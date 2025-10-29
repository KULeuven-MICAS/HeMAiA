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

#include "chip_id.h"
#include "occamy_memory_map.h"
#include "sys_dma_bootrom.h"
#include "uart.c"
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

inline static void print_mem_hex(uintptr_t address_prefix, char* str,
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
        char temp = *((char*)i);
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
    NORMAL,
    INITMEM
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
        local_chip_mem_start_address = address_prefix | SPM_WIDE_BASE_ADDR;
        remote_chip_mem_start_address =
            ((uint64_t)target_chip_id << 40) | SPM_WIDE_BASE_ADDR;
        printf(
            "\033[2J\r\n\t\t Welcome to HeMAiA Bootrom\r\n\r\n\t Chip ID: "
            "L0x%02X, R0x%02X\r\n",
            chip_id, target_chip_id);
        printf(
            "\r\n\t Enter the number to select the mode: "
            "\r\n\t 1. Halt the CVA6 Core"
            "\r\n\t 2. Change the remote Chip ID"
            "\r\n\t 3. Load from UART to Chip %02x"
            "\r\n\t 4. Copy memory from Chip %02x to Chip %02x"
            "\r\n\t 5. Copy memory from Chip %02x to Chip %02x"
            "\r\n\t 6. Print main memory in Chip %02x"
            "\r\n\t 7. Continue to boot"
            "\r\n\t 8. Initialize the main memory\r\n",
            target_chip_id, chip_id, target_chip_id, target_chip_id, chip_id,
            target_chip_id);

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
                printf("\tCopying... \r\n");
                sys_dma_blk_memcpy(remote_chip_mem_start_address,
                                   local_chip_mem_start_address, WIDE_SPM_SIZE);
                printf("\t Copy finished. \r\n");
                scan_char(address_prefix);
                break;

            case COPY_FROM_REMOTE:
                printf("\tCopying... \r\n");
                sys_dma_blk_memcpy(local_chip_mem_start_address,
                                   remote_chip_mem_start_address,
                                   WIDE_SPM_SIZE);
                printf("\t Copy finished. \r\n");
                scan_char(address_prefix);
                break;

            case UART:
                delay_cycles(50000000);  // Delay for 1s
                uart_xmodem(address_prefix, remote_chip_mem_start_address);
                printf("\r\n\t Load finished. \r\n");
                break;

            case PRINTMEM:
                printf("Enter the size of memory in byte:\r\n");
                scanf("%d", &memory_length);
                print_mem_hex(address_prefix,
                              (char*)remote_chip_mem_start_address,
                              memory_length);
                printf("\r\n\r\n\t Print finished. \r\n");
                scan_char(address_prefix);
                break;

            case NORMAL:
                printf("\033[2J");
                printf("\r\n\t Booting...\r\n\r\n\r\n");
                return;
                break;
            case INITMEM:
                sys_dma_blk_memcpy(local_chip_mem_start_address,
                                   WIDE_ZERO_MEM_BASE_ADDR | address_prefix,
                                   WIDE_SPM_SIZE);
                printf("\r\n\t Memory initialized. \r\n");
                scan_char(address_prefix);
                break;
        }
    }
}
