// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include <stdio.h>
#include "hemaia_d2d_link.c"
#include "host.h"

#define D2D_LINK_CLOCK_DIVISION_DEFAULT 1U
#define D2D_LINK_CLOCK_DIVISION_MAX 255U
#define D2D_LINK_BYPASSED_WIRE_DEFAULT 16U
#ifndef D2D_LINK_ENABLE_DELAY_SWEEP_DEBUG
#define D2D_LINK_ENABLE_DELAY_SWEEP_DEBUG 1U
#endif

static uint8_t select_d2d_link_clock_division() {
    uint32_t clock_division = D2D_LINK_CLOCK_DIVISION_DEFAULT;

    printf("Enter D2D link clock division (1-255), then press Enter:\r\n");
    if (scanf("%u", &clock_division) != 1 || clock_division == 0 ||
        clock_division > D2D_LINK_CLOCK_DIVISION_MAX) {
        clock_division = D2D_LINK_CLOCK_DIVISION_DEFAULT;
        printf("Invalid selection. Using D2D link clock division /%d.\r\n",
               (int)clock_division);
    }

    return (uint8_t)clock_division;
}

static void set_all_d2d_link_clock_division(uint8_t clock_division) {
    for (uint8_t i = 1; i <= 4; i++) {
        enable_clk_domain(N_CLUSTERS_PER_CHIPLET + i, clock_division);
    }
    asm volatile("fence" : : : "memory");
    printf("Set D2D link clock division to /%d.\r\n", clock_division);
}

static void reset_all_d2d_link_bypassed_wires() {
    for (uint8_t direction = D2D_DIRECTION_EAST;
         direction <= D2D_DIRECTION_SOUTH; direction++) {
        for (uint8_t channel = 0; channel < CHANNELS_PER_DIRECTION; channel++) {
            set_d2d_link_broken_link((D2DDirection)direction, channel,
                                     D2D_LINK_BYPASSED_WIRE_DEFAULT);
        }
    }
    asm volatile("fence" : : : "memory");
}

static const char* get_d2d_link_direction_name(D2DDirection direction) {
    switch (direction) {
        case D2D_DIRECTION_EAST:
            return "East";
        case D2D_DIRECTION_WEST:
            return "West";
        case D2D_DIRECTION_NORTH:
            return "North";
        case D2D_DIRECTION_SOUTH:
            return "South";
        default:
            return "Unknown";
    }
}

static uint32_t get_d2d_link_selectable_error_sum(D2DDirection direction,
                                                  uint8_t channel) {
    uint32_t sum = 0;
    for (uint8_t wire = 1; wire <= 16; wire++) {
        sum += get_d2d_link_error_cycle_one_wire(direction, channel, wire);
    }
    return sum;
}

static void print_d2d_link_compact_lock_data(D2DDirection direction,
                                             uint8_t delay) {
    const char* direction_name = get_d2d_link_direction_name(direction);
    printf("%s delay=%d:", direction_name, (int)delay);
    for (uint8_t channel = 0; channel < CHANNELS_PER_DIRECTION; channel++) {
        printf(" C%d cycles=%lu W0=%u W1_16_sum=%lu", (int)channel,
               (unsigned long)get_d2d_link_tested_cycle(direction, channel),
               (unsigned int)get_d2d_link_error_cycle_one_wire(
                   direction, channel, 0),
               (unsigned long)get_d2d_link_selectable_error_sum(direction,
                                                                channel));
    }
    printf("\r\n");
}

static void restore_d2d_link_selected_clock_delays(
    D2DDirection direction, const uint8_t* selected_delay) {
    for (uint8_t channel = 0; channel < CHANNELS_PER_DIRECTION; channel++) {
        set_d2d_link_clock_delay(direction, channel, selected_delay[channel]);
    }
    asm volatile("fence" : : : "memory");
}

static void debug_sweep_d2d_link_clock_delays(D2DDirection direction) {
    if (!D2D_LINK_ENABLE_DELAY_SWEEP_DEBUG) {
        return;
    }

    uint8_t selected_delay[CHANNELS_PER_DIRECTION];
    for (uint8_t channel = 0; channel < CHANNELS_PER_DIRECTION; channel++) {
        selected_delay[channel] =
            get_d2d_link_clock_delay(direction, channel);
    }

    printf("%s delay sweep: selected C0=%d C1=%d C2=%d\r\n",
           get_d2d_link_direction_name(direction), (int)selected_delay[0],
           (int)selected_delay[1], (int)selected_delay[2]);

    for (uint8_t delay = 0; delay < HEMAIA_D2D_LINK_NUM_DELAYS; delay++) {
        set_d2d_link_clock_delay_all_channels(direction, delay);
        asm volatile("fence" : : : "memory");
        hemaia_d2d_link_require_test_one_direction(
            direction, HEMAIA_D2D_LINK_DEFAULT_TEST_CYCLES);
        print_d2d_link_compact_lock_data(direction, delay);
    }

    restore_d2d_link_selected_clock_delays(direction, selected_delay);
    hemaia_d2d_link_require_test_one_direction(
        direction, HEMAIA_D2D_LINK_DEFAULT_TEST_CYCLES);
    printf("%s selected-delay retest: ", get_d2d_link_direction_name(direction));
    for (uint8_t channel = 0; channel < CHANNELS_PER_DIRECTION; channel++) {
        printf("C%d delay=%d cycles=%lu W0=%u W1_16_sum=%lu ", (int)channel,
               (int)get_d2d_link_clock_delay(direction, channel),
               (unsigned long)get_d2d_link_tested_cycle(direction, channel),
               (unsigned int)get_d2d_link_error_cycle_one_wire(
                   direction, channel, 0),
               (unsigned long)get_d2d_link_selectable_error_sum(direction,
                                                                channel));
    }
    printf("\r\n");
}

static char scan_non_newline_char(uintptr_t address_prefix) {
    char input_char;
    do {
        input_char = scan_char(address_prefix);
    } while (input_char == '\r' || input_char == '\n');
    return input_char;
}

int main() {
    uintptr_t address_prefix = (uintptr_t)get_current_chip_baseaddress();

    init_uart(address_prefix, 32, 1);
    // Enable vector extension
    enable_vec();
    asm volatile("fence" : : : "memory");

    // Configure the D2D Topology
    hemaia_d2d_link_initialize(get_current_chip_id());

    // Configure the East D2D link
    printf("HeMAiA D2D Link Configurator\r\n");
    reset_all_d2d_link_bypassed_wires();

    uint8_t clock_division = select_d2d_link_clock_division();
    set_all_d2d_link_clock_division(clock_division);

    printf("Press 1 to set to DDR mode, 0 to set to SDR mode:\r\n");
    char mode_char = scan_non_newline_char(address_prefix);
    if (mode_char == '1') {
        set_all_d2d_link_phy_mode(D2D_PHY_MODE_DDR);
        printf("Set to DDR mode.\r\n");
    } else {
        set_all_d2d_link_phy_mode(D2D_PHY_MODE_SDR);
        printf("Set to SDR mode.\r\n");
    }
    printf("Press any key to continue\r\n");
    scan_char(address_prefix);

    printf("Calibrating the East D2D link...\r\n");
    for (uint8_t i = 0; i < MAX_CFG_ROUND; i++) {
        printf("Round %d\r\n", i);
        hemaia_d2d_link_set_delay(D2D_DIRECTION_EAST);
        debug_sweep_d2d_link_clock_delays(D2D_DIRECTION_EAST);
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
    printf("Calibrating the West D2D link...\r\n");
    for (uint8_t i = 0; i < MAX_CFG_ROUND; i++) {
        printf("Round %d\r\n", i);
        hemaia_d2d_link_set_delay(D2D_DIRECTION_WEST);
        debug_sweep_d2d_link_clock_delays(D2D_DIRECTION_WEST);
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
    printf("Calibrating the North D2D link...\r\n");
    for (uint8_t i = 0; i < MAX_CFG_ROUND; i++) {
        printf("Round %d\r\n", i);
        hemaia_d2d_link_set_delay(D2D_DIRECTION_NORTH);
        debug_sweep_d2d_link_clock_delays(D2D_DIRECTION_NORTH);
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
    printf("Calibrating the South D2D link...\r\n");
    for (uint8_t i = 0; i < MAX_CFG_ROUND; i++) {
        printf("Round %d\r\n", i);
        hemaia_d2d_link_set_delay(D2D_DIRECTION_SOUTH);
        debug_sweep_d2d_link_clock_delays(D2D_DIRECTION_SOUTH);
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
