// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include <stdio.h>
#include "hemaia_d2d_link.c"
#include "host.h"

#define D2D_LINK_CLOCK_DIVISION_DEFAULT 1U
#define D2D_LINK_CLOCK_DIVISION_MAX 255U

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
                                     D2D_LINK_NO_BYPASS);
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

static uint32_t get_d2d_link_raw_data_error_sum(D2DDirection direction,
                                                uint8_t channel) {
    uint32_t sum = 0;
    for (uint8_t wire = D2D_LINK_DEBUG_SUM_FIRST_WIRE;
         wire <= D2D_LINK_DEBUG_SUM_LAST_WIRE; wire++) {
        sum += get_d2d_link_error_cycle_one_wire(direction, channel, wire);
    }
    return sum;
}

static void print_d2d_link_direction_training_result(D2DDirection direction) {
    printf("%s selected:", get_d2d_link_direction_name(direction));
    for (uint8_t channel = 0; channel < CHANNELS_PER_DIRECTION; channel++) {
        uint32_t raw_data_errors =
            get_d2d_link_raw_data_error_sum(direction, channel);
        uint32_t counted_data_errors =
            hemaia_d2d_link_get_scored_data_error_sum(direction, channel);
        printf(
            " C%d delay=%d bypass=%d cycles=%lu W0=%u W1_16_raw=%lu "
            "W1_16_counted=%lu",
            (int)channel, (int)get_d2d_link_clock_delay(direction, channel),
            (int)get_d2d_link_broken_link(direction, channel),
            (unsigned long)get_d2d_link_tested_cycle(direction, channel),
            (unsigned int)get_d2d_link_error_cycle_one_wire(
                direction, channel, D2D_LINK_VALID_WIRE),
            (unsigned long)raw_data_errors,
            (unsigned long)counted_data_errors);
    }
    printf("\r\n");
}

static void calibrate_d2d_link_direction(D2DDirection direction) {
    const char* direction_name = get_d2d_link_direction_name(direction);
    printf("Calibrating the %s D2D link...\r\n", direction_name);
    for (uint8_t round = 0; round < MAX_CFG_ROUND; round++) {
        printf("Round %d\r\n", (int)round);
        hemaia_d2d_link_set_delay(direction);
        for (uint8_t channel = 0; channel < CHANNELS_PER_DIRECTION; channel++) {
            hemaia_d2d_link_set_bypass_link(direction, channel);
        }
        asm volatile("fence" : : : "memory");
        hemaia_d2d_link_require_test_one_direction(
            direction, HEMAIA_D2D_LINK_DEFAULT_TEST_CYCLES);

        bool available =
            hemaia_d2d_link_update_availability_from_current_errors(direction);
        print_d2d_link_direction_training_result(direction);
        if (available) {
            printf("%s D2D link is enabled.\r\n", direction_name);
            break;
        }

        if (round == MAX_CFG_ROUND - 1) {
            printf("%s D2D link is disabled.\r\n", direction_name);
        }
    }
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

    calibrate_d2d_link_direction(D2D_DIRECTION_EAST);
    calibrate_d2d_link_direction(D2D_DIRECTION_WEST);
    calibrate_d2d_link_direction(D2D_DIRECTION_NORTH);
    calibrate_d2d_link_direction(D2D_DIRECTION_SOUTH);

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
