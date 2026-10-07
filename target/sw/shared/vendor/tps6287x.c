// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "tps6287x.h"

#include "chip_id.h"
#include "i2c_regs.h"
#include "io.h"
#include "occamy_base_addr.h"

// TPS6287x datasheet SLVSGC5E, sections 7.3.6, 7.5 and 8:
// https://www.ti.com/lit/ds/symlink/tps62873.pdf
// Controller register definitions are generated from the actual vendored RTL.
// This polling driver supports a single controller and up to four independent
// PMICs sharing SDA/SCL (not paralleled/stacked converter outputs).
enum {
    PMIC_VSET = 0x00,
    PMIC_CONTROL1 = 0x01,
    PMIC_CONTROL2 = 0x02,
    PMIC_RESET = 0x80,
    PMIC_SSCEN = 0x40,
    PMIC_SWEN = 0x20,
    PMIC_FPWMEN = 0x10,
    PMIC_VRAMP_MASK = 0x03,
    PMIC_VRANGE_MASK = 0x0c,
    PMIC_VRANGE_5MV = 0x08,
};

#define BIT(n) (UINT32_C(1) << (n))
#define HOST_ENABLE BIT(I2C_CTRL_ENABLEHOST_BIT)
#define IDLE_EMPTY (BIT(I2C_STATUS_HOSTIDLE_BIT) | BIT(I2C_STATUS_FMTEMPTY_BIT))
#define TRANS_COMPLETE BIT(I2C_INTR_STATE_TRANS_COMPLETE_BIT)
#define NACK BIT(I2C_INTR_STATE_NAK_BIT)
#define STRETCH_TIMEOUT BIT(I2C_INTR_STATE_STRETCH_TIMEOUT_BIT)
#define FIFO_ERRORS (BIT(I2C_INTR_STATE_FMT_OVERFLOW_BIT) | \
                     BIT(I2C_INTR_STATE_RX_OVERFLOW_BIT))
#define START BIT(I2C_FDATA_START_BIT)
#define STOP BIT(I2C_FDATA_STOP_BIT)
#define READ BIT(I2C_FDATA_READ_BIT)
#define RX_FMT_RESET (BIT(I2C_FIFO_CTRL_RXRST_BIT) | BIT(I2C_FIFO_CTRL_FMTRST_BIT))
#define LINES_HIGH (UINT32_C(1) | (UINT32_C(1) << 16))

// Bounded MMIO polling, independent of CPU DVFS and CLINT interrupt ownership.
// This is a poll-count bound, not a calibrated wall-clock timeout. Hardware
// separately detects SCL stretching lasting longer than approximately 1 ms.
#ifndef TPS6287X_POLL_LIMIT
#define TPS6287X_POLL_LIMIT 100000u
#endif

static uint32_t bus_clock_hz;
static uint8_t initialized_addresses;

static uint32_t reg_read(uint32_t offset) {
    return readw((uintptr_t)(get_current_chip_baseaddress() |
                            (I2C_BASE_ADDR + offset)));
}

static void reg_write(uint32_t offset, uint32_t value) {
    writew(value, (uintptr_t)(get_current_chip_baseaddress() |
                             (I2C_BASE_ADDR + offset)));
    // Order device writes before subsequent polling/FDATA writes on CVA6.
#ifdef __riscv
    __asm__ volatile("fence iorw, iorw" ::: "memory");
#endif
}

static int valid_address(uint8_t address) {
    return address >= TPS6287X_ADDR_6K2_GND && address <= TPS6287X_ADDR_47K_VIN;
}

static uint32_t cycles(uint32_t hz, uint32_t ns) {
    return (uint32_t)(((uint64_t)hz * ns + 999999999u) / 1000000000u);
}

static int wait_idle(void) {
    for (uint32_t i = 0; i < TPS6287X_POLL_LIMIT; ++i) {
        if (reg_read(I2C_STATUS_REG_OFFSET) & BIT(I2C_STATUS_HOSTIDLE_BIT))
            return TPS6287X_OK;
    }
    return TPS6287X_ERR_TIMEOUT;
}

static int wait_lines_high(uint32_t mask) {
    for (uint32_t i = 0; i < TPS6287X_POLL_LIMIT; ++i) {
        if ((reg_read(I2C_VAL_REG_OFFSET) & mask) == mask)
            return TPS6287X_OK;
    }
    return TPS6287X_ERR_TIMEOUT;
}

static void bus_delay(uint32_t clocks) {
    // Each MMIO read crosses the peripheral clock domain and takes at least
    // one peripheral cycle. This intentionally errs on the long side.
    for (uint32_t i = 0; i < clocks; ++i)
        (void)reg_read(I2C_VAL_REG_OFFSET);
}

static int configure_bus(uint32_t hz) {
    uint32_t status = reg_read(I2C_STATUS_REG_OFFSET);
    if ((status & IDLE_EMPTY) != IDLE_EMPTY ||
        (reg_read(I2C_CTRL_REG_OFFSET) & BIT(I2C_CTRL_ENABLETARGET_BIT)))
        return TPS6287X_ERR_BUSY;

    initialized_addresses = 0;
    bus_clock_hz = 0;
    reg_write(I2C_CTRL_REG_OFFSET, 0);
    reg_write(I2C_INTR_ENABLE_REG_OFFSET, 0);
    reg_write(I2C_FIFO_CTRL_REG_OFFSET, RX_FMT_RESET);

    // Fast-mode Plus minimum timings, rounded UP to peripheral cycles.
    // In this RTL a data-bit period is TLOW + THIGH + T_R + 2*T_F, plus
    // synchronization/clock-stretch delays (i2c_fsm.sv counter_functions).
    const uint32_t rise = cycles(hz, 120);
    const uint32_t fall = cycles(hz, 120);
    const uint32_t low = cycles(hz, 500);
    uint32_t high = cycles(hz, 260);
    const uint32_t period = cycles(hz, 1000);
    if (period > low + rise + 2 * fall + high)
        high = period - low - rise - 2 * fall;
    reg_write(I2C_TIMING0_REG_OFFSET, (low << 16) | high);
    reg_write(I2C_TIMING1_REG_OFFSET, (fall << 16) | rise);
    reg_write(I2C_TIMING2_REG_OFFSET, (cycles(hz, 260) << 16) | cycles(hz, 260));
    reg_write(I2C_TIMING3_REG_OFFSET, (cycles(hz, 50) << 16) | cycles(hz, 50));
    reg_write(I2C_TIMING4_REG_OFFSET, (cycles(hz, 500) << 16) | cycles(hz, 260));
    reg_write(I2C_TIMEOUT_CTRL_REG_OFFSET,
              BIT(I2C_TIMEOUT_CTRL_EN_BIT) | cycles(hz, 1000000));

    // TI recommends a STOP after the pull-up supply comes up. The controller
    // is idle/disabled, so drive both low, release SCL, then release SDA.
    reg_write(I2C_OVRD_REG_OFFSET, BIT(I2C_OVRD_TXOVRDEN_BIT));
    bus_delay(cycles(hz, 1000));
    reg_write(I2C_OVRD_REG_OFFSET,
              BIT(I2C_OVRD_TXOVRDEN_BIT) | BIT(I2C_OVRD_SCLVAL_BIT));
    int rc = wait_lines_high(1u);
    bus_delay(cycles(hz, 1000));
    reg_write(I2C_OVRD_REG_OFFSET,
              BIT(I2C_OVRD_TXOVRDEN_BIT) | BIT(I2C_OVRD_SCLVAL_BIT) |
              BIT(I2C_OVRD_SDAVAL_BIT));
    if (rc == TPS6287X_OK) rc = wait_lines_high(LINES_HIGH);
    bus_delay(cycles(hz, 1000));
    reg_write(I2C_OVRD_REG_OFFSET, 0);
    reg_write(I2C_INTR_STATE_REG_OFFSET, UINT32_MAX);
    if (rc == TPS6287X_OK) bus_clock_hz = hz;
    return rc;
}

static int transfer(const uint32_t *commands, unsigned count, uint8_t *rx) {
    if ((reg_read(I2C_STATUS_REG_OFFSET) & IDLE_EMPTY) != IDLE_EMPTY)
        return TPS6287X_ERR_BUSY;
    if ((reg_read(I2C_VAL_REG_OFFSET) & LINES_HIGH) != LINES_HIGH)
        return TPS6287X_ERR_BUS;

    // Queue the complete short transaction before enabling the host, avoiding
    // FIFO starvation between bytes or before the repeated START of a read.
    reg_write(I2C_CTRL_REG_OFFSET, 0);
    reg_write(I2C_FIFO_CTRL_REG_OFFSET, RX_FMT_RESET);
    reg_write(I2C_INTR_STATE_REG_OFFSET, UINT32_MAX);
    for (unsigned i = 0; i < count; ++i)
        reg_write(I2C_FDATA_REG_OFFSET, commands[i]);
    reg_write(I2C_CTRL_REG_OFFSET, HOST_ENABLE);

    int rc = TPS6287X_ERR_TIMEOUT;
    uint32_t errors = 0;
    for (uint32_t i = 0; i < TPS6287X_POLL_LIMIT; ++i) {
        uint32_t irq = reg_read(I2C_INTR_STATE_REG_OFFSET);
        errors |= irq & (NACK | STRETCH_TIMEOUT | FIFO_ERRORS);
        uint32_t status = reg_read(I2C_STATUS_REG_OFFSET);
        // TRANS_COMPLETE also occurs at repeated START. Require FIFO empty
        // AND host idle so a register read cannot finish before its final STOP.
        if ((status & IDLE_EMPTY) == IDLE_EMPTY && (irq & TRANS_COMPLETE)) {
            errors |= reg_read(I2C_INTR_STATE_REG_OFFSET) &
                      (NACK | STRETCH_TIMEOUT | FIFO_ERRORS);
            rc = TPS6287X_OK;
            break;
        }
        if (errors & STRETCH_TIMEOUT) break;
    }

    // Do not classify SDA_INTERFERENCE as fatal: this RTL checks it already in
    // SetupBit/HoldStop, before released SDA has propagated through the pad and
    // the two input synchronizers. It can assert on ordinary 0->1 transitions.
    // SCL_INTERFERENCE/SDA_UNSTABLE similarly are not qualified by settled SCL.
    // This is a single-controller bus; ACK, FIFO and timeout checks are used.
    reg_write(I2C_CTRL_REG_OFFSET, 0);
    if (rc != TPS6287X_OK) {
        // Disabling host requests a STOP at the next legal FSM boundary. Do
        // not reset an in-flight FIFO: the FSM still consumes its front entry.
        if (wait_idle() == TPS6287X_OK)
            reg_write(I2C_FIFO_CTRL_REG_OFFSET, RX_FMT_RESET);
        initialized_addresses = 0;
        bus_clock_hz = 0;
        return TPS6287X_ERR_TIMEOUT;
    }
    if (errors & STRETCH_TIMEOUT) return TPS6287X_ERR_TIMEOUT;
    if (errors & NACK) return TPS6287X_ERR_NACK;
    if (errors & FIFO_ERRORS) return TPS6287X_ERR_BUS;
    if (rx != 0) {
        if (reg_read(I2C_STATUS_REG_OFFSET) & BIT(I2C_STATUS_RXEMPTY_BIT))
            return TPS6287X_ERR_BUS;
        *rx = (uint8_t)reg_read(I2C_RDATA_REG_OFFSET);
    }
    return TPS6287X_OK;
}

static int read_pmic(uint8_t address, uint8_t reg, uint8_t *value) {
    const uint32_t commands[] = {
        ((uint32_t)address << 1) | START,
        reg,
        ((uint32_t)address << 1) | 1u | START,
        1u | READ | STOP, // One byte; RCONT=0 sends NACK before STOP.
    };
    return transfer(commands, 4, value);
}

static int write_pmic(uint8_t address, uint8_t reg, uint8_t value) {
    const uint32_t commands[] = {
        ((uint32_t)address << 1) | START, reg, (uint32_t)value | STOP,
    };
    return transfer(commands, 3, 0);
}

int tps6287x_init(uint8_t address, uint32_t peripheral_clock_hz) {
    if (!valid_address(address)) return TPS6287X_ERR_ADDRESS;
    if (peripheral_clock_hz < 10000000u || peripheral_clock_hz > 1000000000u)
        return TPS6287X_ERR_CLOCK;
    const uint8_t mask = (uint8_t)(1u << (address - TPS6287X_ADDR_6K2_GND));
    initialized_addresses &= (uint8_t)~mask;
    if (bus_clock_hz != peripheral_clock_hz ||
        reg_read(I2C_TIMING0_REG_OFFSET) == 0) {
        int rc = configure_bus(peripheral_clock_hz);
        if (rc != TPS6287X_OK) return rc;
    }

    uint8_t control2, vset, control1, check;
    int rc = read_pmic(address, PMIC_CONTROL2, &control2);
    if (rc != TPS6287X_OK) return rc;
    // All standard I2C variants power up in this range. Changing the range on
    // a live SoC rail can cause an unintended voltage step, so reject it.
    if ((control2 & PMIC_VRANGE_MASK) != PMIC_VRANGE_5MV)
        return TPS6287X_ERR_CONFIG;
    rc = read_pmic(address, PMIC_VSET, &vset);
    if (rc != TPS6287X_OK) return rc;
    if (TPS6287X_MIN_MV + vset * TPS6287X_STEP_MV > TPS6287X_MAX_MV)
        return TPS6287X_ERR_CONFIG;
    rc = read_pmic(address, PMIC_CONTROL1, &control1);
    if (rc != TPS6287X_OK) return rc;
    control1 = (uint8_t)((control1 & ~(PMIC_RESET | PMIC_SSCEN | PMIC_VRAMP_MASK)) |
                         PMIC_SWEN | PMIC_FPWMEN);
    rc = write_pmic(address, PMIC_CONTROL1, control1);
    if (rc != TPS6287X_OK) return rc;
    rc = read_pmic(address, PMIC_CONTROL1, &check);
    if (rc != TPS6287X_OK) return rc;
    if (check != control1) return TPS6287X_ERR_CONFIG;
    initialized_addresses |= mask;
    return TPS6287X_OK;
}

int tps6287x_set_voltage(uint8_t address, int millivolts) {
    if (!valid_address(address)) return TPS6287X_ERR_ADDRESS;
    if (millivolts < TPS6287X_MIN_MV || millivolts > TPS6287X_MAX_MV)
        return TPS6287X_ERR_VOLTAGE;
    const uint8_t mask = (uint8_t)(1u << (address - TPS6287X_ADDR_6K2_GND));
    if (!(initialized_addresses & mask)) return TPS6287X_ERR_NOT_INITIALIZED;
    const uint8_t vset = (uint8_t)((millivolts - TPS6287X_MIN_MV + 2) /
                                  TPS6287X_STEP_MV);
    int rc = write_pmic(address, PMIC_VSET, vset);
    if (rc != TPS6287X_OK) {
        initialized_addresses &= (uint8_t)~mask;
        return rc;
    }
    return TPS6287X_MIN_MV + vset * TPS6287X_STEP_MV;
}
