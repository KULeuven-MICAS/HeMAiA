// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include <stdio.h>
#include "hemaia_d2d_link.c"
#include "host.h"

int main() {
    uintptr_t address_prefix = (uintptr_t)get_current_chip_baseaddress();

    init_uart(address_prefix, 32, 1);
    asm volatile("fence" : : : "memory");

    // Configure the East D2D link
    printf("HeMAiA D2D Link Configurator\r\n");
    printf("Configuring the East D2D link...\r\n");
    for (uint8_t i = 0; i < MAX_CFG_ROUND; i++) {
        printf("Round %d\r\n", i);
        hemaia_d2d_link_set_delay(D2D_DIRECTION_EAST);
        for (uint8_t j = 0; j < CHANNELS_PER_DIRECTION; j++) {
            hemaia_d2d_link_set_bypass_link(D2D_DIRECTION_EAST, j);
        }
        if (get_d2d_link_availability(D2D_DIRECTION_EAST)) {
            printf("East D2D link is enabled.\r\n");
            break;
        }

        if (i == MAX_CFG_ROUND - 1) {
            printf("East D2D link is disabled.\r\n");
        }
    }

    // Configure the West D2D link
    printf("Configuring the West D2D link...\r\n");
    for (uint8_t i = 0; i < MAX_CFG_ROUND; i++) {
        printf("Round %d\r\n", i);
        hemaia_d2d_link_set_delay(D2D_DIRECTION_WEST);
        for (uint8_t j = 0; j < CHANNELS_PER_DIRECTION; j++) {
            hemaia_d2d_link_set_bypass_link(D2D_DIRECTION_WEST, j);
        }
        if (get_d2d_link_availability(D2D_DIRECTION_WEST)) {
            printf("West D2D link is enabled.\r\n");
            break;
        }
        if (i == (MAX_CFG_ROUND - 1)) {
            printf("West D2D link is disabled.\r\n");
        }
    }

    // Configure the North D2D link
    printf("Configuring the North D2D link...\r\n");
    for (uint8_t i = 0; i < MAX_CFG_ROUND; i++) {
        printf("Round %d\r\n", i);
        hemaia_d2d_link_set_delay(D2D_DIRECTION_NORTH);
        for (uint8_t j = 0; j < CHANNELS_PER_DIRECTION; j++) {
            hemaia_d2d_link_set_bypass_link(D2D_DIRECTION_NORTH, j);
        }
        if (get_d2d_link_availability(D2D_DIRECTION_NORTH)) {
            printf("North D2D link is enabled.\r\n");
            break;
        }
        if (i == (MAX_CFG_ROUND - 1)) {
            printf("North D2D link is disabled.\r\n");
        }
    }

    // Configure the South D2D link
    printf("Configuring the South D2D link...\r\n");
    for (uint8_t i = 0; i < MAX_CFG_ROUND; i++) {
        printf("Round %d\r\n", i);
        hemaia_d2d_link_set_delay(D2D_DIRECTION_SOUTH);
        for (uint8_t j = 0; j < CHANNELS_PER_DIRECTION; j++) {
            hemaia_d2d_link_set_bypass_link(D2D_DIRECTION_SOUTH, j);
        }
        if (get_d2d_link_availability(D2D_DIRECTION_SOUTH)) {
            printf("South D2D link is enabled.\r\n");
            break;
        }
        if (i == (MAX_CFG_ROUND - 1)) {
            printf("South D2D link is disabled.\r\n");
        }
    }

    printf("Configuration results:\r\n");
    printf(
        "East D2D link:\r\nAvailability: %d\r\nClock delay: %d, %d, "
        "%d\r\nBypassed wires: %d, %d, %d\r\n",
        get_d2d_link_availability(D2D_DIRECTION_EAST),
        get_d2d_link_clock_delay(D2D_DIRECTION_EAST, 0),
        get_d2d_link_clock_delay(D2D_DIRECTION_EAST, 1),
        get_d2d_link_clock_delay(D2D_DIRECTION_EAST, 2),
        get_d2d_link_broken_link(D2D_DIRECTION_EAST, 0),
        get_d2d_link_broken_link(D2D_DIRECTION_EAST, 1),
        get_d2d_link_broken_link(D2D_DIRECTION_EAST, 2));
    printf(
        "West D2D link:\r\nAvailability: %d\r\nClock delay: %d, %d, "
        "%d\r\nBypassed wires: %d, %d, %d\r\n",
        get_d2d_link_availability(D2D_DIRECTION_WEST),
        get_d2d_link_clock_delay(D2D_DIRECTION_WEST, 0),
        get_d2d_link_clock_delay(D2D_DIRECTION_WEST, 1),
        get_d2d_link_clock_delay(D2D_DIRECTION_WEST, 2),
        get_d2d_link_broken_link(D2D_DIRECTION_WEST, 0),
        get_d2d_link_broken_link(D2D_DIRECTION_WEST, 1),
        get_d2d_link_broken_link(D2D_DIRECTION_WEST, 2));
    printf(
        "North D2D link:\r\nAvailability: %d\r\nClock delay: %d, %d, "
        "%d\r\nBypassed wires: %d, %d, %d\r\n",
        get_d2d_link_availability(D2D_DIRECTION_NORTH),
        get_d2d_link_clock_delay(D2D_DIRECTION_NORTH, 0),
        get_d2d_link_clock_delay(D2D_DIRECTION_NORTH, 1),
        get_d2d_link_clock_delay(D2D_DIRECTION_NORTH, 2),
        get_d2d_link_broken_link(D2D_DIRECTION_NORTH, 0),
        get_d2d_link_broken_link(D2D_DIRECTION_NORTH, 1),
        get_d2d_link_broken_link(D2D_DIRECTION_NORTH, 2));
    printf(
        "South D2D link:\r\nAvailability: %d\r\nClock delay: %d, %d, "
        "%d\r\nBypassed wires: %d, %d, %d\r\n",
        get_d2d_link_availability(D2D_DIRECTION_SOUTH),
        get_d2d_link_clock_delay(D2D_DIRECTION_SOUTH, 0),
        get_d2d_link_clock_delay(D2D_DIRECTION_SOUTH, 1),
        get_d2d_link_clock_delay(D2D_DIRECTION_SOUTH, 2),
        get_d2d_link_broken_link(D2D_DIRECTION_SOUTH, 0),
        get_d2d_link_broken_link(D2D_DIRECTION_SOUTH, 1),
        get_d2d_link_broken_link(D2D_DIRECTION_SOUTH, 2));

    return 0;
}
