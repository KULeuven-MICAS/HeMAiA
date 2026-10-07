// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef HEMAIA_TPS6287X_H_
#define HEMAIA_TPS6287X_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Unshifted 7-bit addresses, selected by VSEL at power-up. VSEL also selects
// the startup voltage; the two settings cannot be selected independently.
#define TPS6287X_ADDR_6K2_GND 0x40u
#define TPS6287X_ADDR_GND     0x41u
#define TPS6287X_ADDR_VIN     0x42u
#define TPS6287X_ADDR_47K_VIN 0x43u
#define TPS6287X_MIN_MV      400
#define TPS6287X_MAX_MV      1200
#define TPS6287X_STEP_MV     5

enum {
    TPS6287X_OK = 0,
    TPS6287X_ERR_ADDRESS = -1,
    TPS6287X_ERR_VOLTAGE = -2,
    TPS6287X_ERR_CLOCK = -3,
    TPS6287X_ERR_NOT_INITIALIZED = -4,
    TPS6287X_ERR_BUSY = -5,
    TPS6287X_ERR_NACK = -6,
    TPS6287X_ERR_TIMEOUT = -7,
    TPS6287X_ERR_BUS = -8,
    TPS6287X_ERR_CONFIG = -9,
};

/** Configure an I2C-equipped TPS62870/1/2/3 on the local chip's I2C bus.
 *
 * address: one of the four unshifted TPS6287X_ADDR_* addresses above.
 * peripheral_clock_hz: actual clk_periph_i frequency in Hz (not CPU frequency),
 * in the range 10 MHz .. 1 GHz. Use the same value for every PMIC on this bus.
 * Fast-mode Plus timing targets <= 1 MHz, allowing 120 ns bus rise/fall times;
 * clock quantization, conservative edge budgets and input synchronization can
 * make the effective SCL rate lower. External pull-ups are required.
 *
 * Selects FPWMEN=1, SSCEN=0, VRAMP=00 (10 mV/us), SWEN=1. Preserves the startup
 * voltage, discharge/hiccup settings, soft-start time and CONTROL3. Verifies the
 * reset-default VRANGE=10 (400 mV + VSET*5 mV) and a setpoint <= 1200 mV. A PMIC
 * previously configured to another range returns ERR_CONFIG without changing
 * its voltage: live range conversion is intentionally not performed here.
 *
 * Returns 0 on success, a negative TPS6287X_ERR_* on failure. Call once for EACH
 * address, and again after a PMIC/controller reset or peripheral clock change.
 * Changing the bus clock invalidates initialization of the other addresses.
 * EN and power sequencing remain the board's responsibility.
 */
int tps6287x_init(uint8_t address, uint32_t peripheral_clock_hz);

/** Program a nominal voltage, rounded to the nearest 5 mV.
 *
 * Accepts integer millivolts in [400, 1200]. Rejects out-of-range requests
 * BEFORE rounding, without any I2C traffic. E.g. 803 -> 805, 802 -> 800;
 * 1201 -> ERR_VOLTAGE. Returns the programmed millivolts on success, or a
 * negative TPS6287X_ERR_*. The address must have been successfully initialized.
 *
 * Uses one three-byte write (address, VSET register, value). Completion means
 * ACKs and STOP were observed by the controller, NOT that VOUT has settled.
 * The caller must allow board-validated settling time before raising frequency.
 * A failed write can have reached the PMIC; its voltage must not be assumed.
 *
 * These functions exclusively own the local I2C controller. Serialize all
 * calls (including between interrupts/harts); do not mix with another driver
 * or modify PMIC configuration behind this driver's back.
 *
 * Example (replace peripheral_clock_hz with the actual board clock):
 *   int rc = tps6287x_init(TPS6287X_ADDR_GND, peripheral_clock_hz);
 *   if (rc == TPS6287X_OK)
 *       rc = tps6287x_set_voltage(TPS6287X_ADDR_GND, 803); // returns 805
 */
int tps6287x_set_voltage(uint8_t address, int millivolts);

#ifdef __cplusplus
}
#endif
#endif
