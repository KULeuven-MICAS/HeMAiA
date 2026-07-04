// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// External PMIC control (over I2C) for DVFS voltage scaling.
//
// DEFERRED: the actual PMIC/I2C transaction is not implemented yet. This stub
// models the voltage-set side effect (a UART log + a settle delay) so the DVFS
// control mechanism can be exercised end-to-end in simulation. When the board /
// PMIC is available, replace the body of pmic_set_voltage() with a real OpenTitan
// I2C master transaction to the PMIC (the I2C controller is mapped at
// I2C_BASE_ADDR = 0x0200_4000 and pinned out at the chip level).
#pragma once

#include <stdint.h>

// Program the external PMIC to the voltage corresponding to `level`
// (the same power/clock-division code the PM published in DVFS_REQUEST).
static inline void pmic_set_voltage(uint8_t level) {
    // TODO(deferred): drive the OpenTitan I2C master at I2C_BASE_ADDR to write the
    // PMIC voltage register. For now, only log and model the regulator settle time.
    printf("[dvfs] pmic_set_voltage(level=%u) [STUB]\n", (unsigned)level);
    for (volatile int i = 0; i < 100; i++) {
        asm volatile("nop");
    }
}
