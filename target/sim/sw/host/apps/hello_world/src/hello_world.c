// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include "host.h"

int main() {
    uintptr_t address_prefix = (uintptr_t)get_current_chip_baseaddress();

    init_uart(address_prefix, 32, 1);
    asm volatile("fence" : : : "memory");
    print_str(address_prefix, "Hello world from Occamy in VCU128! \r\n");
    char uart_rx_buffer[512];
    char uart_tx_buffer[512];

    while (1) {
        scan_str(address_prefix, uart_rx_buffer);
        sprintf(uart_tx_buffer, "[Occamy] What you said is: %s",
                uart_rx_buffer);
        print_str(address_prefix, uart_tx_buffer);
    }

    return 0;
}
