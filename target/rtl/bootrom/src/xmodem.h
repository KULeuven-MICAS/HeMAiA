// Copyright 2024 KU Leuven
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Yunhao Deng <yunhao.deng@kuleuven.be>

#pragma once

#include <stdint.h>
#include "uart.h"

#define SOH 0x01
#define STX 0x02
#define EOT 0x04
#define ACK 0x06
#define NAK 0x15
#define CAN 0x18
#define CTRL_Z 0x1A
#define DLE 0x10
#define size_t uint32_t

#define bool uint8_t
#define true 1
#define false 0

uint8_t compute_parity(const uint8_t *data, size_t length) {
    uint16_t parity = 0;  // Initial value
    for (size_t i = 0; i < length; i++) {
        parity += (uint16_t)data[i];  // Calculate the sum
        parity = parity & 0x00FF;
    }
    return (uint8_t)parity;
}

void *memcpy(void *dest, const void *src, size_t n) {
    uint8_t *d = (uint8_t *)dest;
    const uint8_t *s = (const uint8_t *)src;

    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }

    return dest;
}

inline void flush_cache() {
    asm volatile("fence.i" ::: "memory");
}

void uart_xmodem(uintptr_t address_prefix, uint64_t start_address) {
    uint8_t received_char;
    bool transmission_end = false;

    putchar(address_prefix, NAK);  // Request for data

    uint32_t current_offset = 0;

    while (!transmission_end) {
        uint8_t data[1024];
        uint8_t packet_number, packet_complement;
        uint8_t received_parity, calculated_parity;
        int index = 0;

        received_char = getchar(address_prefix);  // Read the header
        if (received_char == EOT) {                   // End of transmission
            putchar(address_prefix, ACK);
            print_str(address_prefix, "\r\n\t Load finished. \r\n\r\n");
            break;
        }

        if (received_char == SOH || received_char == STX) {
            packet_number = getchar(address_prefix);
            packet_complement = getchar(address_prefix);

            if (packet_number ==
                ((~packet_complement) & 0xFF)) {  // Packet number is correct
                while (index < (received_char == SOH ? 128 : 1024)) {
                    data[index++] = getchar(address_prefix);
                }

                received_parity = getchar(address_prefix);

                calculated_parity = compute_parity(data, index);

                if (received_parity == calculated_parity) {
                    // Copy data to memory
                    putchar(address_prefix, ACK);
                    memcpy((void *)(start_address + current_offset), data,
                           index);
                    current_offset += index;
                } else {
                    putchar(address_prefix,
                                 NAK);  // CRC error, request retransmission
                }
            } else {
                putchar(address_prefix, NAK);  // Packet number error
            }
        } else {
            putchar(address_prefix, CAN);  // Unexpected byte received
        }
    }
    // Flush the cache after loading the program
    flush_cache();
}
