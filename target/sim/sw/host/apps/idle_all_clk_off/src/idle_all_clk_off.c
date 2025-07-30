// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include "host.c"

// Frequency at which the UART peripheral is clocked
int main() {
    init_uart(32, 1);
    asm volatile("fence" : : : "memory");
    print_uart("IDLE ALL CLK OFF \r\n");

    // Enable clock for cluster 0 and uncore
    reset_and_ungate_quadrants(0x00);
    deisolate_all();

    // Wait for a certain amount of time for simulation purposes
    for (int i = 0; i < 100000; i++) asm volatile("nop" : : : "memory");

    return 0;
}
