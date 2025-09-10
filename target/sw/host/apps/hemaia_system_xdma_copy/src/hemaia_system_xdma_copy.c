// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include <stdio.h>
#include "hemaia_clk_rst_controller.h"
#include "host.h"

int main() {
    init_uart((uintptr_t)get_current_chip_baseaddress(), 32, 1);
    asm volatile("fence" : : : "memory");
    return 0;
}
