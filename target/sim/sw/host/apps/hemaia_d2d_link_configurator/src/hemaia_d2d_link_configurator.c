// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include "hemaia_d2d_link.h"
#include "host.h"

uint8_t *hemaia_d2d_link_cfg = (uint8_t *)HEMAIA_D2D_LINK_BASE_ADDR;

uint32_t get_max_index(uint8_t *array, uint32_t size) {
    uint32_t max_index = 0;
    for (uint32_t i = 1; i < size; ++i) {
        if (array[i] > array[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}

int main() {
    uintptr_t address_prefix = (uintptr_t)get_current_chip_baseaddress();

    init_uart(address_prefix, 32, 1);
    asm volatile("fence" : : : "memory");

    // Set the East and South link into test mode
    set_d2d_link_test_mode(D2D_DIRECTION_EAST, 1);
    set_d2d_link_test_mode(D2D_DIRECTION_SOUTH, 1);
    delay_cycles(1000);
    set_d2d_link_test_mode(D2D_DIRECTION_EAST, 0);
    set_d2d_link_test_mode(D2D_DIRECTION_SOUTH, 0);

    // Evaluate the broken link on the East and South link
    uint8_t ber_counter_result[20] = {0};

    get_d2d_link_error_cycle_one_channel(D2D_DIRECTION_EAST, 0,
                                         ber_counter_result);
    printf(
        "East C0: \r\nTotal cycle: %d, wire with highest error at wire %d\r\n",
        get_d2d_link_tested_cycle(D2D_DIRECTION_EAST, 0),
        get_max_index(ber_counter_result, 20));

    get_d2d_link_error_cycle_one_channel(D2D_DIRECTION_EAST, 1,
                                         ber_counter_result);
    printf(
        "East C1: \r\nTotal cycle: %d, wire with highest error at wire %d\r\n",
        get_d2d_link_tested_cycle(D2D_DIRECTION_EAST, 1),
        get_max_index(ber_counter_result, 20));

    get_d2d_link_error_cycle_one_channel(D2D_DIRECTION_EAST, 2,
                                         ber_counter_result);
    printf(
        "East C2: \r\nTotal cycle: %d, wire with highest error at wire %d\r\n",
        get_d2d_link_tested_cycle(D2D_DIRECTION_EAST, 2),
        get_max_index(ber_counter_result, 20));

    get_d2d_link_error_cycle_one_channel(D2D_DIRECTION_SOUTH, 0,
                                         ber_counter_result);
    printf(
        "South C0: \r\nTotal cycle: %d, wire with highest error at wire %d\r\n",
        get_d2d_link_tested_cycle(D2D_DIRECTION_SOUTH, 0),
        get_max_index(ber_counter_result, 20));

    get_d2d_link_error_cycle_one_channel(D2D_DIRECTION_SOUTH, 1,
                                         ber_counter_result);
    printf(
        "South C1: \r\nTotal cycle: %d, wire with highest error at wire %d\r\n",
        get_d2d_link_tested_cycle(D2D_DIRECTION_SOUTH, 1),
        get_max_index(ber_counter_result, 20));

    get_d2d_link_error_cycle_one_channel(D2D_DIRECTION_SOUTH, 2,
                                         ber_counter_result);
    printf(
        "South C2: \r\nTotal cycle: %d, wire with highest error at wire %d\r\n",
        get_d2d_link_tested_cycle(D2D_DIRECTION_SOUTH, 2),
        get_max_index(ber_counter_result, 20));

    return 0;
}