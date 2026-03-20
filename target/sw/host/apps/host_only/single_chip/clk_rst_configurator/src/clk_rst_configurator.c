// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include <stdio.h>
#include "hemaia_clk_rst_controller.h"
#include "host.h"

int main() {
    init_uart((uintptr_t)get_current_chip_baseaddress(), 32, 1);
    // Enable vector extension
    enable_vec();
    asm volatile("fence" : : : "memory");

    // Clock Gating the D2D links
    for (int i = 1; i <= 4; i++) {
        disable_clk_domain(N_CLUSTERS_PER_CHIPLET + i);
    }
    asm volatile("fence" : : : "memory");
    printf("The clock of D2D link is disabled\r\n");

    // Try to reset all D2D links
    for (int i = 1; i <= 4; i++) {
        reset_clk_domain(N_CLUSTERS_PER_CHIPLET + i);
    }
    asm volatile("fence" : : : "memory");
    printf("The D2D links are reset. \r\n");

    // Re-enable the clock of D2D links at halved speed
    for (int i = 1; i <= 4; i++) {
        enable_clk_domain(N_CLUSTERS_PER_CHIPLET + i, 2);
    }
    asm volatile("fence" : : : "memory");
    printf("The clock of D2D link is enabled at halved speed. \r\n");
    asm volatile("fence" : : : "memory");
    return 0;
}
