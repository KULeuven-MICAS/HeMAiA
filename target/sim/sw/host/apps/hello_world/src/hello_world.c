// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include "chip_id.h"
#include "host.c"

// Frequency at which the UART peripheral is clocked
#define PERIPH_FREQ 50000000

int main() {
    uintptr_t address_prefix = (uintptr_t)get_current_chip_baseaddress();

    init_uart(address_prefix, PERIPH_FREQ, 1000000);
    asm volatile("fence" : : : "memory");
    print_str(address_prefix, "Hello world from Occamy in VCU128! \r\n");
    char uart_rx_buffer[512];
    char uart_tx_buffer[512];

    while (1) {
        scan_str(address_prefix, uart_rx_buffer);
        sprintf(uart_tx_buffer, "[Occamy] What you said is: %s",
                uart_rx_buffer);
        print_str(address_prefix, uart_tx_buffer);
        // Artificial delay to ensure last symbol has been transmitted
        // (just waiting for the UART TSR register to be empty is not
        // sufficient)
        for (int i = 0; i < 50000; i++) asm volatile("nop" : : : "memory");
    }

    return 0;
}
