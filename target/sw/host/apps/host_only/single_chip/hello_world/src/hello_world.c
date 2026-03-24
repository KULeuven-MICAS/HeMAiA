// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// #include <stdio.h>
#include "host.h"

int main() {
    uintptr_t address_prefix = (uintptr_t)get_current_chip_baseaddress();
    init_uart(address_prefix, 32, 1);
    // Enable vector extension
    enable_vec();
    asm volatile("fence" : : : "memory");
    printf("Hello world from HeMAiA! \r\n");
    return 0;
}
