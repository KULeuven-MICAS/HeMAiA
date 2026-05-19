// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

#pragma once
#include <stdint.h>
#include "chip_id.h"
#include "hemaia_clk_rst_controller_peripheral.h"
#include "occamy_memory_map.h"

// ---------------------------------------------------------------------------
// Clock-domain layout
//
// The HeMAiA clock-reset controller exposes 32 logical clock channels per
// chip; the actual number wired up depends on the SoC configuration —
// occamy_chip.sv allocates one host-CPU channel, N_CLUSTERS_PER_CHIPLET
// cluster channels, and 4 D2D PHY channels (East/West/North/South), for
// a total of (1 + N_CLUSTERS_PER_CHIPLET + 4) channels. 
// Software can re-program any channel at runtime via
// enable_clk_domain(channel, divisor) — the new divisor takes effect on
// the next divider-cycle boundary, glitch-free, no reset needed.
//
//   Channel index   Purpose                          Wire (occamy_chip.sv)
//   --------------  -------------------------------  ---------------------
//   0               Host CPU (CVA6) clock            clk_vec[0]
//   1 .. N          Accelerator cluster clocks       clk_vec[1 +: N]
//   N+1             East  D2D TX PHY clock           clk_vec[N+1]
//   N+2             West  D2D TX PHY clock           clk_vec[N+2]
//   N+3             North D2D TX PHY clock           clk_vec[N+3]
//   N+4             South D2D TX PHY clock           clk_vec[N+4]
//
//   where N = N_CLUSTERS_PER_CHIPLET (defined by the generated
//   occamy.h). For the current 1-cluster-per-chip multichip config this
//   collapses to channels 0..5; the controller still supports up to 32.
//
// Divisor semantics: the value passed to enable_clk_domain() IS the
// integer divide ratio of the master clock, matching the RTL semantics
// in hemaia_clock_divider.sv (a value of 1 = pass-through). With a
// 500 MHz master clock:
//
//   divisor  1  → 500.00 MHz
//   divisor  6  →  83.33 MHz
//   divisor 16  →  31.25 MHz
//
// All helpers below program the LOCAL chip only — they OR
// get_current_chip_baseaddress() into the controller register address.
// To program a remote chip's divider, build the same MMIO address with
// a different chip prefix and issue a 32-bit store to it directly.
// ---------------------------------------------------------------------------

// Reset the HeMAIA domain
static inline void reset_all_clk_domain() {
    volatile uint32_t *hemaia_clk_rst_controller_reset_addr =
        (volatile uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_CLK_RST_CONTROLLER_BASE_ADDR) +
                HEMAIA_CLK_RST_CONTROLLER_RESET_REGISTER_REG_OFFSET);
    *hemaia_clk_rst_controller_reset_addr = 0xFFFFFFFF;  // Reset the clk domain
}

// Reset one specific clock domain
static inline void reset_clk_domain(uint8_t domain) {
    volatile uint32_t *hemaia_clk_rst_controller_reset_addr =
        (volatile uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_CLK_RST_CONTROLLER_BASE_ADDR) +
                HEMAIA_CLK_RST_CONTROLLER_RESET_REGISTER_REG_OFFSET);
    *hemaia_clk_rst_controller_reset_addr =
        (1 << domain);  // Reset the clk domain
}

// Disable the clock of one specific clock domain
static inline void disable_clk_domain(uint8_t domain) {
    volatile uint32_t *hemaia_clk_rst_controller_clock_valid_reg =
        (volatile uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_CLK_RST_CONTROLLER_BASE_ADDR) +
                HEMAIA_CLK_RST_CONTROLLER_CLOCK_VALID_REGISTER_REG_OFFSET);

    volatile uint32_t *hemaia_clk_rst_controller_clock_division_reg =
        (volatile uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_CLK_RST_CONTROLLER_BASE_ADDR) +
                HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C0_C3_REG_OFFSET +
                domain / 4 * 4);
    uint32_t mask = 0xFF << ((domain % 4) * 8);
    *hemaia_clk_rst_controller_clock_division_reg &=
        ~mask;  // Clear the corresponding byte
    *hemaia_clk_rst_controller_clock_valid_reg =
        1 << domain;  // Set the valid bit
}

static inline void enable_clk_domain(uint8_t domain, uint8_t division) {
    volatile uint32_t *hemaia_clk_rst_controller_clock_valid_reg =
        (volatile uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_CLK_RST_CONTROLLER_BASE_ADDR) +
                HEMAIA_CLK_RST_CONTROLLER_CLOCK_VALID_REGISTER_REG_OFFSET);

    volatile uint32_t *hemaia_clk_rst_controller_clock_division_reg =
        (volatile uint32_t
             *)(((uintptr_t)get_current_chip_baseaddress() |
                 HEMAIA_CLK_RST_CONTROLLER_BASE_ADDR) +
                HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C0_C3_REG_OFFSET +
                domain / 4 * 4);
    uint32_t division_register_val =
        *hemaia_clk_rst_controller_clock_division_reg;
    uint32_t shift = (domain % 4) * 8;
    division_register_val &= ~(0xFF << shift);  // Clear the corresponding byte
    division_register_val |=
        ((division & 0xFF) << shift);  // Set the new division value
    *hemaia_clk_rst_controller_clock_division_reg = division_register_val;
    *hemaia_clk_rst_controller_clock_valid_reg =
        (1 << domain);  // Set the valid bit
}
