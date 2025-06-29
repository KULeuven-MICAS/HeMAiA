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
    return -1;

    // Clock Gating the D2D links: Clock Channel 2 / 3 / 4 /5
    disable_clk_domain(2);
    disable_clk_domain(3);
    disable_clk_domain(4);
    disable_clk_domain(5);
    asm volatile("fence" : : : "memory");
    printf("The clock of D2D link is disabled\r\n");

    // Try to reset all D2D links
    reset_clk_domain(2);
    reset_clk_domain(3);
    reset_clk_domain(4);
    reset_clk_domain(5);
    asm volatile("fence" : : : "memory");
    printf("The D2D links are reset. \r\n");

    // Re-enable the clock of D2D links at halved speed
    enable_clk_domain(2, 2);  // 1: 2 division
    enable_clk_domain(3, 2);  // 1: 2 division
    enable_clk_domain(4, 2);  // 1: 2 division
    enable_clk_domain(5, 2);  // 1: 2 division
    asm volatile("fence" : : : "memory");
    printf("The clock of D2D link is enabled at halved speed. \r\n");
    asm volatile("fence" : : : "memory");
    return 0;
}
