// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

#pragma once
#include <stdint.h>
#include "chip_id.h"
#include "hemaia_d2d_link_peripheral.h"
#include "occamy_memory_map.h"

typedef enum {
    D2D_DIRECTION_EAST = 0,
    D2D_DIRECTION_WEST = 1,
    D2D_DIRECTION_NORTH = 2,
    D2D_DIRECTION_SOUTH = 3
} Direction;

typedef uint8_t bool;

inline void delay_cycles(uint64_t cycle) {
    uint64_t target_cycle, current_cycle;
    __asm__ volatile("csrr %0, mcycle;" : "=r"(current_cycle));
    target_cycle = current_cycle + cycle;
    while (current_cycle < target_cycle) {
        __asm__ volatile("csrr %0, mcycle;" : "=r"(current_cycle));
    }
}

// Reset the D2D link
inline void reset_d2d_link_digital(uint32_t delay) {
    volatile uint32_t *hemaia_d2d_link_reset_addr =
        (volatile uint32_t *)(((uintptr_t)get_current_chip_baseaddress() |
                               HEMAIA_D2D_LINK_BASE_ADDR) +
                              HEMAIA_D2D_LINK_RESET_REGISTER_REG_OFFSET);
    *hemaia_d2d_link_reset_addr = 0xFFFFFFFF;  // Reset the d2d link
    delay_cycles(delay);
    *hemaia_d2d_link_reset_addr = 0x0;  // Release the reset
}

// Test mode
inline void set_d2d_link_test_mode(Direction direction, bool enable) {
    volatile uint32_t *hemaia_d2d_link_test_mode_addr =
        (volatile uint32_t *)(((uintptr_t)get_current_chip_baseaddress() |
                               HEMAIA_D2D_LINK_BASE_ADDR) +
                              HEMAIA_D2D_LINK_TEST_MODE_REGISTER_REG_OFFSET);
    if (enable) {
        *hemaia_d2d_link_test_mode_addr |= (1 << direction);
    } else {
        *hemaia_d2d_link_test_mode_addr &= ~(1 << direction);
    }
}

inline void set_all_d2d_link_test_mode(bool enable) {
    volatile uint32_t *hemaia_d2d_link_test_mode_addr =
        (volatile uint32_t *)(((uintptr_t)get_current_chip_baseaddress() |
                               HEMAIA_D2D_LINK_BASE_ADDR) +
                              HEMAIA_D2D_LINK_TEST_MODE_REGISTER_REG_OFFSET);
    if (enable) {
        *hemaia_d2d_link_test_mode_addr |= 0x0F;  // Set all test modes
    } else {
        *hemaia_d2d_link_test_mode_addr &= ~0x0F;  // Clear all test modes
    }
}

inline bool get_d2d_link_being_tested(Direction direction) {
    volatile uint32_t *hemaia_d2d_link_test_mode_addr =
        (volatile uint32_t *)(((uintptr_t)get_current_chip_baseaddress() |
                               HEMAIA_D2D_LINK_BASE_ADDR) +
                              HEMAIA_D2D_LINK_TEST_MODE_REGISTER_REG_OFFSET);
    uint32_t test_mode_status = *hemaia_d2d_link_test_mode_addr;
    return (test_mode_status & (1 << (direction + 4))) != 0;
}

// Link availability
inline void set_d2d_link_availability(Direction direction, bool avail) {
    volatile uint32_t *hemaia_d2d_link_availability_addr =
        (volatile uint32_t *)(((uintptr_t)get_current_chip_baseaddress() |
                               HEMAIA_D2D_LINK_BASE_ADDR) +
                              HEMAIA_D2D_LINK_AVAILABILITY_REGISTER_REG_OFFSET);
    if (avail) {
        *hemaia_d2d_link_availability_addr |= (1 << direction);
    } else {
        *hemaia_d2d_link_availability_addr &= ~(1 << direction);
    }
}
inline void set_all_d2d_link_availability(bool avail) {
    volatile uint32_t *hemaia_d2d_link_availability_addr =
        (volatile uint32_t *)(((uintptr_t)get_current_chip_baseaddress() |
                               HEMAIA_D2D_LINK_BASE_ADDR) +
                              HEMAIA_D2D_LINK_AVAILABILITY_REGISTER_REG_OFFSET);
    if (avail) {
        *hemaia_d2d_link_availability_addr |=
            0x0F;  // Set all link availability
    } else {
        *hemaia_d2d_link_availability_addr &=
            ~0x0F;  // Clear all link availability
    }
}

inline bool get_d2d_link_availability(Direction direction) {
    volatile uint32_t *hemaia_d2d_link_availability_addr =
        (volatile uint32_t *)(((uintptr_t)get_current_chip_baseaddress() |
                               HEMAIA_D2D_LINK_BASE_ADDR) +
                              HEMAIA_D2D_LINK_AVAILABILITY_REGISTER_REG_OFFSET);
    uint32_t availability_status = *hemaia_d2d_link_availability_addr;
    return (availability_status & (1 << direction)) != 0;
}

// Error count and total count of the D2D link BER test
inline uint32_t get_d2d_link_tested_cycle(Direction direction) {
    uintptr_t base =
        (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
    uint32_t offset = 0;
    switch (direction) {
        case D2D_DIRECTION_EAST:
            offset =
                HEMAIA_D2D_LINK_EAST_TEST_MODE_TOTAL_CYCLE_REGISTER_REG_OFFSET;
            break;
        case D2D_DIRECTION_WEST:
            offset =
                HEMAIA_D2D_LINK_WEST_TEST_MODE_TOTAL_CYCLE_REGISTER_REG_OFFSET;
            break;
            ;
            break;
        case D2D_DIRECTION_NORTH:
            offset =
                HEMAIA_D2D_LINK_NORTH_TEST_MODE_TOTAL_CYCLE_REGISTER_REG_OFFSET;
            break;
        default:
            offset =
                HEMAIA_D2D_LINK_SOUTH_TEST_MODE_TOTAL_CYCLE_REGISTER_REG_OFFSET;
            break;
    }
    return *((volatile uint32_t *)(base + offset));
}

inline uint8_t get_d2d_link_error_cycle(Direction direction, uint8_t channel) {
    uintptr_t base =
        (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
    uint8_t group = channel / 4;
    uint8_t subChannel = channel % 4;
    uint32_t offset = 0;
    switch (direction) {
        case D2D_DIRECTION_EAST:
            offset =
                HEMAIA_D2D_LINK_EAST_TEST_MODE_ERROR_REGISTER_0_REG_OFFSET +
                group * 4;
            break;
        case D2D_DIRECTION_WEST:
            offset =
                HEMAIA_D2D_LINK_WEST_TEST_MODE_ERROR_REGISTER_0_REG_OFFSET +
                group * 4;
            break;
        case D2D_DIRECTION_NORTH:
            offset =
                HEMAIA_D2D_LINK_NORTH_TEST_MODE_ERROR_REGISTER_0_REG_OFFSET +
                group * 4;
            break;
        default:
            offset =
                HEMAIA_D2D_LINK_SOUTH_TEST_MODE_ERROR_REGISTER_0_REG_OFFSET +
                group * 4;
            break;
    }
    uint32_t value = *((volatile uint32_t *)(base + offset));
    return (value >> (8 * subChannel)) & 0xFF;
}

inline void get_all_d2d_link_error_cycle(Direction direction, uint8_t *dest) {
    uintptr_t base =
        (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
    switch (direction) {
        case D2D_DIRECTION_EAST:
            base = base +
                   HEMAIA_D2D_LINK_EAST_TEST_MODE_ERROR_REGISTER_0_REG_OFFSET;
            break;
        case D2D_DIRECTION_WEST:
            base = base +
                   HEMAIA_D2D_LINK_WEST_TEST_MODE_ERROR_REGISTER_0_REG_OFFSET;
            break;
        case D2D_DIRECTION_NORTH:
            base = base +
                   HEMAIA_D2D_LINK_NORTH_TEST_MODE_ERROR_REGISTER_0_REG_OFFSET;
            break;
        default:
            base = base +
                   HEMAIA_D2D_LINK_SOUTH_TEST_MODE_ERROR_REGISTER_0_REG_OFFSET;
            break;
    }

    uint32_t *dest_addr = (uint32_t *)dest;
    for (uint8_t i = 0; i < 5; i++) {
        dest_addr[i] = ((uint32_t *)base)[i];
    }
}

// FEC unrecoverable error count
inline uint32_t get_fec_unrecoverable_error_count(Direction direction) {
    uint32_t *fec_unrecoverable_error_count_addr =
        (uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_D2D_LINK_BASE_ADDR) +
                HEMAIA_D2D_LINK_EAST_FEC_UNRECOVERABLE_ERROR_REGISTER_REG_OFFSET +
                direction * 4);
    return *fec_unrecoverable_error_count_addr;
}

// Programmable clock delay
inline void set_d2d_link_clock_delay(uint8_t delay, Direction direction) {
    uintptr_t base =
        (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
    volatile uint32_t *reg =
        (volatile uint32_t
             *)(base +
                HEMAIA_D2D_LINK_PROGRAMMABLE_CLOCK_DELAY_REGISTER_REG_OFFSET);
    uint32_t current = *reg;
    uint32_t shift = direction * 8;
    current &= ~(0xFF << shift);
    current |= ((uint32_t)delay & 0xFF) << shift;
    *reg = current;
}

inline void set_all_d2d_link_clock_delay(uint8_t delay) {
    uintptr_t base =
        (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
    volatile uint32_t *reg =
        (volatile uint32_t
             *)(base +
                HEMAIA_D2D_LINK_PROGRAMMABLE_CLOCK_DELAY_REGISTER_REG_OFFSET);
    *reg = ((uint32_t)delay << 24) | ((uint32_t)delay << 16) |
           ((uint32_t)delay << 8) | delay;
}

inline uint8_t get_d2d_link_clock_delay(Direction direction) {
    uintptr_t base =
        (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
    volatile uint32_t *reg =
        (volatile uint32_t
             *)(base +
                HEMAIA_D2D_LINK_PROGRAMMABLE_CLOCK_DELAY_REGISTER_REG_OFFSET);
    return (uint8_t)((*reg >> (direction * 8)) & 0xFF);
}

// Driving strength
inline void set_d2d_link_driving_strength(uint8_t strength,
                                          Direction direction) {
    volatile uint32_t *hemaia_d2d_link_driving_strength_addr =
        (volatile uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_D2D_LINK_BASE_ADDR) +
                HEMAIA_D2D_LINK_PROGRAMMABLE_DRIVE_STRENGTH_REGISTER_REG_OFFSET);
    *hemaia_d2d_link_driving_strength_addr =
        (*hemaia_d2d_link_driving_strength_addr & ~(0xFF << (direction * 8))) |
        (strength << (direction * 8));
}

inline void set_all_d2d_link_driving_strength(uint8_t strength) {
    volatile uint32_t *hemaia_d2d_link_driving_strength_addr =
        (volatile uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_D2D_LINK_BASE_ADDR) +
                HEMAIA_D2D_LINK_PROGRAMMABLE_DRIVE_STRENGTH_REGISTER_REG_OFFSET);
    *hemaia_d2d_link_driving_strength_addr =
        (strength << 24) | (strength << 16) | (strength << 8) | strength;
}

inline uint8_t get_d2d_link_driving_strength(Direction direction) {
    volatile uint32_t *hemaia_d2d_link_driving_strength_addr =
        (volatile uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_D2D_LINK_BASE_ADDR) +
                HEMAIA_D2D_LINK_PROGRAMMABLE_DRIVE_STRENGTH_REGISTER_REG_OFFSET);
    uint32_t strength_value = *hemaia_d2d_link_driving_strength_addr;
    return (strength_value >> (direction * 8)) & 0xFF;
}

// RX buffer threshold
inline void set_d2d_link_threshold(uint8_t threshold, Direction direction) {
    volatile uint32_t *hemaia_d2d_link_threshold_addr =
        (volatile uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_D2D_LINK_BASE_ADDR) +
                HEMAIA_D2D_LINK_RX_BUFFER_THRESHOLD_REGISTER_REG_OFFSET);
    *hemaia_d2d_link_threshold_addr =
        (*hemaia_d2d_link_threshold_addr & ~(0xFF << (direction * 8))) |
        (threshold << (direction * 8));
}

inline void set_all_d2d_link_threshold(uint8_t threshold) {
    volatile uint32_t *hemaia_d2d_link_threshold_addr =
        (volatile uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_D2D_LINK_BASE_ADDR) +
                HEMAIA_D2D_LINK_RX_BUFFER_THRESHOLD_REGISTER_REG_OFFSET);
    *hemaia_d2d_link_threshold_addr =
        (threshold << 24) | (threshold << 16) | (threshold << 8) |
        threshold;  // Set the threshold to the passed parameter
}

inline uint8_t get_d2d_link_threshold(Direction direction) {
    volatile uint32_t *hemaia_d2d_link_threshold_addr =
        (volatile uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_D2D_LINK_BASE_ADDR) +
                HEMAIA_D2D_LINK_RX_BUFFER_THRESHOLD_REGISTER_REG_OFFSET);
    uint32_t threshold_value = *hemaia_d2d_link_threshold_addr;
    return (threshold_value >> (direction * 8)) & 0xFF;
}

// Operating mode
inline bool get_d2d_link_operating_mode(Direction direction) {
    uintptr_t base =
        (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
    volatile uint32_t *reg =
        (volatile uint32_t
             *)(base + HEMAIA_D2D_LINK_TX_MODE_MONITOR_REGISTER_REG_OFFSET);
    uint32_t val = *reg;
    uint8_t shift;
    switch (direction) {
        case D2D_DIRECTION_EAST:
            shift = HEMAIA_D2D_LINK_TX_MODE_MONITOR_REGISTER_EAST_TX_MODE_BIT;
            break;
        case D2D_DIRECTION_WEST:
            shift = HEMAIA_D2D_LINK_TX_MODE_MONITOR_REGISTER_WEST_TX_MODE_BIT;
            break;
        case D2D_DIRECTION_NORTH:
            shift = HEMAIA_D2D_LINK_TX_MODE_MONITOR_REGISTER_NORTH_TX_MODE_BIT;
            break;
        default:
            shift = HEMAIA_D2D_LINK_TX_MODE_MONITOR_REGISTER_SOUTH_TX_MODE_BIT;
            break;
    }
    return ((val >> shift) & 1U) != 0U;
}

// TX mode clock frequency
inline void set_d2d_link_tx_mode_clock_frequency(uint8_t frequency,
                                                 Direction direction) {
    volatile uint32_t *hemaia_d2d_link_tx_mode_clock_frequency_addr =
        (volatile uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_D2D_LINK_BASE_ADDR) +
                HEMAIA_D2D_LINK_TX_MODE_CLOCK_FREQUENCY_REGISTER_REG_OFFSET);
    *hemaia_d2d_link_tx_mode_clock_frequency_addr =
        (*hemaia_d2d_link_tx_mode_clock_frequency_addr &
         ~(0xFF << (direction * 8))) |
        (frequency << (direction * 8));
}

inline void set_all_d2d_link_tx_mode_clock_frequency(uint8_t frequency) {
    volatile uint32_t *hemaia_d2d_link_tx_mode_clock_frequency_addr =
        (volatile uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_D2D_LINK_BASE_ADDR) +
                HEMAIA_D2D_LINK_TX_MODE_CLOCK_FREQUENCY_REGISTER_REG_OFFSET);
    *hemaia_d2d_link_tx_mode_clock_frequency_addr =
        (frequency << 24) | (frequency << 16) | (frequency << 8) | frequency;
}

inline uint8_t get_d2d_link_tx_mode_clock_frequency(Direction direction) {
    volatile uint32_t *hemaia_d2d_link_tx_mode_clock_frequency_addr =
        (volatile uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_D2D_LINK_BASE_ADDR) +
                HEMAIA_D2D_LINK_TX_MODE_CLOCK_FREQUENCY_REGISTER_REG_OFFSET);
    uint32_t frequency_value = *hemaia_d2d_link_tx_mode_clock_frequency_addr;
    return (frequency_value >> (direction * 8)) & 0xFF;
}

// Clock gating delay
inline void set_d2d_link_clock_gating_delay(uint8_t delay,
                                            Direction direction) {
    volatile uint32_t *hemaia_d2d_link_clock_gating_delay_addr =
        (volatile uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_D2D_LINK_BASE_ADDR) +
                HEMAIA_D2D_LINK_AUTOMATIC_HIGH_SPEED_CLOCK_GATING_DELAY_REGISTER_REG_OFFSET);
    *hemaia_d2d_link_clock_gating_delay_addr =
        (*hemaia_d2d_link_clock_gating_delay_addr &
         ~(0xFF << (direction * 8))) |
        (delay << (direction * 8));
}
inline void set_all_d2d_link_clock_gating_delay(uint8_t delay) {
    volatile uint32_t *hemaia_d2d_link_clock_gating_delay_addr =
        (volatile uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_D2D_LINK_BASE_ADDR) +
                HEMAIA_D2D_LINK_AUTOMATIC_HIGH_SPEED_CLOCK_GATING_DELAY_REGISTER_REG_OFFSET);
    *hemaia_d2d_link_clock_gating_delay_addr =
        (delay << 24) | (delay << 16) | (delay << 8) | delay;
}
inline uint8_t get_d2d_link_clock_gating_delay(Direction direction) {
    volatile uint32_t *hemaia_d2d_link_clock_gating_delay_addr =
        (volatile uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_D2D_LINK_BASE_ADDR) +
                HEMAIA_D2D_LINK_AUTOMATIC_HIGH_SPEED_CLOCK_GATING_DELAY_REGISTER_REG_OFFSET);
    uint32_t delay_value = *hemaia_d2d_link_clock_gating_delay_addr;
    return (delay_value >> (direction * 8)) & 0xFF;
}
