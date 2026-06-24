#include <stdio.h>
#include "hemaia_d2d_link.h"

typedef struct {
    uint32_t error_sum;
    uint32_t cycle_count;
} D2DLinkDelaySample;

static const char* hemaia_d2d_link_direction_name(D2DDirection direction) {
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

static bool hemaia_d2d_link_is_bypassable_wire(uint8_t wire) {
    return wire >= D2D_LINK_FIRST_BYPASSABLE_WIRE &&
           wire <= D2D_LINK_LAST_BYPASSABLE_WIRE;
}

static uint32_t hemaia_d2d_link_error_sum_range(D2DDirection direction,
                                                uint8_t channel,
                                                uint8_t first_wire,
                                                uint8_t last_wire,
                                                uint8_t excluded_wire) {
    uint32_t sum = 0;
    for (uint8_t wire = first_wire; wire <= last_wire; wire++) {
        if (wire == excluded_wire) {
            continue;
        }
        sum += get_d2d_link_error_cycle_one_wire(direction, channel, wire);
    }
    return sum;
}

uint32_t hemaia_d2d_link_get_debug_data_error_sum(D2DDirection direction,
                                                  uint8_t channel) {
    return hemaia_d2d_link_error_sum_range(
        direction, channel, D2D_LINK_DEBUG_SUM_FIRST_WIRE,
        D2D_LINK_DEBUG_SUM_LAST_WIRE, HEMAIA_D2D_LINK_BROKEN_LINK_REG_SIZE);
}

uint32_t hemaia_d2d_link_get_scored_data_error_sum(D2DDirection direction,
                                                   uint8_t channel) {
    uint8_t bypassed_wire = get_d2d_link_broken_link(direction, channel);
    uint8_t excluded_wire = HEMAIA_D2D_LINK_BROKEN_LINK_REG_SIZE;
    if (hemaia_d2d_link_is_bypassable_wire(bypassed_wire)) {
        excluded_wire = bypassed_wire;
    }
    return hemaia_d2d_link_error_sum_range(
        direction, channel, D2D_LINK_DEBUG_SUM_FIRST_WIRE,
        D2D_LINK_DEBUG_SUM_LAST_WIRE, excluded_wire);
}

static void hemaia_d2d_link_print_delay_sample(D2DDirection direction,
                                               uint8_t delay) {
    printf("%s delay=%d:", hemaia_d2d_link_direction_name(direction),
           (int)delay);
    for (uint8_t channel = 0; channel < CHANNELS_PER_DIRECTION; channel++) {
        printf(" C%d cycles=%lu W0=%u W1_16_sum=%lu", (int)channel,
               (unsigned long)get_d2d_link_tested_cycle(direction, channel),
               (unsigned int)get_d2d_link_error_cycle_one_wire(
                   direction, channel, D2D_LINK_VALID_WIRE),
               (unsigned long)hemaia_d2d_link_get_debug_data_error_sum(
                   direction, channel));
    }
    printf("\r\n");
}

static bool hemaia_d2d_link_error_rate_is_better(uint32_t error,
                                                 uint32_t cycles,
                                                 uint32_t best_error,
                                                 uint32_t best_cycles) {
    if (cycles == 0) {
        return false;
    }
    if (best_cycles == 0) {
        return true;
    }
    return ((uint64_t)error * best_cycles) <
           ((uint64_t)best_error * cycles);
}

static uint8_t hemaia_d2d_link_select_delay(
    const D2DLinkDelaySample* samples) {
    uint8_t best_zero_start = 0;
    uint8_t best_zero_len = 0;
    uint8_t delay = 0;

    while (delay < HEMAIA_D2D_LINK_NUM_DELAYS) {
        if (samples[delay].cycle_count == 0 || samples[delay].error_sum != 0) {
            delay++;
            continue;
        }

        uint8_t zero_start = delay;
        while (delay < HEMAIA_D2D_LINK_NUM_DELAYS &&
               samples[delay].cycle_count > 0 &&
               samples[delay].error_sum == 0) {
            delay++;
        }

        uint8_t zero_len = delay - zero_start;
        if (zero_len > best_zero_len) {
            best_zero_start = zero_start;
            best_zero_len = zero_len;
        }
    }

    if (best_zero_len > 0) {
        return best_zero_start + ((best_zero_len - 1) / 2);
    }

    uint8_t best_delay = 0;
    for (delay = 1; delay < HEMAIA_D2D_LINK_NUM_DELAYS; delay++) {
        if (hemaia_d2d_link_error_rate_is_better(
                samples[delay].error_sum, samples[delay].cycle_count,
                samples[best_delay].error_sum,
                samples[best_delay].cycle_count)) {
            best_delay = delay;
        }
    }
    return best_delay;
}

void hemaia_d2d_link_require_test_one_direction(D2DDirection direction,
                                                uint32_t cycles) {
    while (get_d2d_link_being_tested(direction)) {
        // Wait until the link is not being tested
    }
    set_d2d_link_test_mode(direction, true);
    delay_cycles(cycles);
    set_d2d_link_test_mode(direction, false);
}

void hemaia_d2d_link_require_test_all_directions(uint32_t cycles) {
    while (get_d2d_link_being_tested(D2D_DIRECTION_EAST) ||
           get_d2d_link_being_tested(D2D_DIRECTION_WEST) ||
           get_d2d_link_being_tested(D2D_DIRECTION_NORTH) ||
           get_d2d_link_being_tested(D2D_DIRECTION_SOUTH)) {
        // Wait until the link is not being tested
    }
    set_d2d_link_test_mode(D2D_DIRECTION_EAST, true);
    set_d2d_link_test_mode(D2D_DIRECTION_WEST, true);
    set_d2d_link_test_mode(D2D_DIRECTION_NORTH, true);
    set_d2d_link_test_mode(D2D_DIRECTION_SOUTH, true);
    delay_cycles(cycles);
    set_d2d_link_test_mode(D2D_DIRECTION_EAST, false);
    set_d2d_link_test_mode(D2D_DIRECTION_WEST, false);
    set_d2d_link_test_mode(D2D_DIRECTION_NORTH, false);
    set_d2d_link_test_mode(D2D_DIRECTION_SOUTH, false);
}

// The function to set the delay for each channel in a direction
void hemaia_d2d_link_set_delay(D2DDirection direction) {
    D2DLinkDelaySample samples[CHANNELS_PER_DIRECTION]
                              [HEMAIA_D2D_LINK_NUM_DELAYS] = {0};

    for (uint8_t delay = 0; delay < HEMAIA_D2D_LINK_NUM_DELAYS; delay++) {
        set_d2d_link_clock_delay_all_channels(direction, delay);
        asm volatile("fence" : : : "memory");
        hemaia_d2d_link_require_test_one_direction(
            direction, HEMAIA_D2D_LINK_DEFAULT_TEST_CYCLES);

        hemaia_d2d_link_print_delay_sample(direction, delay);

        for (uint8_t channel = 0; channel < CHANNELS_PER_DIRECTION;
             channel++) {
            samples[channel][delay].cycle_count =
                get_d2d_link_tested_cycle(direction, channel);
            samples[channel][delay].error_sum =
                hemaia_d2d_link_get_scored_data_error_sum(direction, channel);
        }
    }

    printf("%s selected delays:", hemaia_d2d_link_direction_name(direction));
    for (uint8_t channel = 0; channel < CHANNELS_PER_DIRECTION; channel++) {
        uint8_t selected_delay =
            hemaia_d2d_link_select_delay(samples[channel]);
        set_d2d_link_clock_delay(direction, channel, selected_delay);
        printf(" C%d=%d", (int)channel, (int)selected_delay);
    }
    printf("\r\n");

    asm volatile("fence" : : : "memory");
    hemaia_d2d_link_require_test_one_direction(
        direction, HEMAIA_D2D_LINK_DEFAULT_TEST_CYCLES);
}

// The function to set the broken link for one channel in a direction
int32_t hemaia_d2d_link_set_bypass_link(D2DDirection direction,
                                        uint8_t channel) {
    uint8_t wire_error[HEMAIA_D2D_LINK_BROKEN_LINK_REG_SIZE] = {0};
    get_d2d_link_error_cycle_one_channel(direction, channel, wire_error);

    uint8_t selected_wire = D2D_LINK_NO_BYPASS;
    uint8_t bad_wire_count = 0;
    for (uint8_t wire = D2D_LINK_FIRST_BYPASSABLE_WIRE;
         wire <= D2D_LINK_LAST_BYPASSABLE_WIRE; wire++) {
        if (wire_error[wire] == 0) {
            continue;
        }
        selected_wire = wire;
        bad_wire_count++;
    }

    if (bad_wire_count > 1) {
        printf("%s C%d has multiple bad bypassable wires:",
               hemaia_d2d_link_direction_name(direction), (int)channel);
        for (uint8_t wire = D2D_LINK_FIRST_BYPASSABLE_WIRE;
             wire <= D2D_LINK_LAST_BYPASSABLE_WIRE; wire++) {
            if (wire_error[wire] != 0) {
                printf(" W%d=%u", (int)wire, (unsigned int)wire_error[wire]);
            }
        }
        printf("; selecting largest bad wire W%d.\r\n", (int)selected_wire);
    }

    set_d2d_link_broken_link(direction, channel, selected_wire);
    return selected_wire;
}

bool hemaia_d2d_link_update_availability_from_current_errors(
    D2DDirection direction) {
    bool available = true;

    for (uint8_t channel = 0; channel < CHANNELS_PER_DIRECTION; channel++) {
        uint32_t cycles = get_d2d_link_tested_cycle(direction, channel);
        uint32_t errors =
            hemaia_d2d_link_get_scored_data_error_sum(direction, channel);
        if (cycles == 0 || errors != 0) {
            available = false;
        }
    }

    set_d2d_link_availability(direction, available);
    return available;
}
