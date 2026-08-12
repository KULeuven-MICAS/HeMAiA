// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Exercises the IO_DRIVE_STRENGTH register in the SoC controller and the
// set_io_drive_strength() helpers in host.h.
//
// This only checks the configuration path -- that every per-domain field is
// reachable, independently writable and reads back what was written. The actual
// pad drive strength is an analog property and is not observable from software;
// in RTL simulation the behavioural tc_digital_io ignores its DS input entirely,
// so this test behaves identically with and without the pads present.
//
// Note it does drive every group, including UART and JTAG, down to the weakest
// setting for one step. That is harmless here (the weakest code still drives,
// and at UART baud rates the edge rate is irrelevant), but it is the reason this
// is a bring-up/simulation test rather than something to run behind a marginal
// serial link.

#include <stdio.h>
#include "host.h"

static const struct {
    const char *name;
    io_drv_group_t group;
} io_drv_groups[] = {
    {"MISC", IO_DRV_MISC}, {"D2D", IO_DRV_D2D},   {"UART", IO_DRV_UART},
    {"GPIO", IO_DRV_GPIO}, {"SPIM", IO_DRV_SPIM}, {"SPIS", IO_DRV_SPIS},
    {"I2C", IO_DRV_I2C},   {"JTAG", IO_DRV_JTAG},
};

#define N_IO_DRV_GROUPS ARRAY_ELEM_COUNT(io_drv_groups)

// Compare one group against an expected value, reporting and counting mismatches.
static int check_group(uint8_t chip_id, unsigned idx, uint8_t expected) {
    uint8_t got = get_io_drive_strength(chip_id, io_drv_groups[idx].group);
    if (got != expected) {
        printf("  %s: got 0x%x, expected 0x%x\r\n", io_drv_groups[idx].name, got,
               expected);
        return 1;
    }
    return 0;
}

int main() {
    uint8_t chip_id = get_current_chip_id();
    init_uart((uintptr_t)get_current_chip_baseaddress(), 32, 1);
    asm volatile("fence" : : : "memory");

    int errors = 0;

    printf("[io_drv] IO drive strength configuration test\r\n");

    // 1. Reset value: every group must come up at IO_DRV_RESET, otherwise the
    //    boot-critical pins are not at the drive strength the RTL intends.
    uint32_t reset_raw = get_io_drive_strength_raw(chip_id);
    printf("[io_drv] reset value: 0x%x\r\n", reset_raw);
    for (unsigned i = 0; i < N_IO_DRV_GROUPS; i++)
        errors += check_group(chip_id, i, IO_DRV_RESET);
    printf("[io_drv] reset value      : %s\r\n", errors ? "FAIL" : "PASS");

    // 2. Field independence: give every group a distinct value and read them all
    //    back. Catches overlapping or aliased fields, which a uniform pattern
    //    would hide.
    int step_errors = errors;
    for (unsigned i = 0; i < N_IO_DRV_GROUPS; i++)
        set_io_drive_strength(chip_id, io_drv_groups[i].group, (uint8_t)i);
    for (unsigned i = 0; i < N_IO_DRV_GROUPS; i++)
        errors += check_group(chip_id, i, (uint8_t)i);
    printf("[io_drv] distinct per group: %s (0x%x)\r\n",
           (errors == step_errors) ? "PASS" : "FAIL",
           get_io_drive_strength_raw(chip_id));

    // 3. Writing one group must not disturb the others.
    step_errors = errors;
    set_io_drive_strength(chip_id, IO_DRV_SPIM, 0xf);
    for (unsigned i = 0; i < N_IO_DRV_GROUPS; i++)
        errors += check_group(
            chip_id, i,
            (io_drv_groups[i].group == IO_DRV_SPIM) ? 0xf : (uint8_t)i);
    printf("[io_drv] single-group write: %s\r\n",
           (errors == step_errors) ? "PASS" : "FAIL");

    // 4. The all-groups helper, at both ends of the range.
    step_errors = errors;
    for (uint8_t v = 0; v <= 0xf; v += 0xf) {
        set_io_drive_strength_all(chip_id, v);
        for (unsigned i = 0; i < N_IO_DRV_GROUPS; i++)
            errors += check_group(chip_id, i, v);
    }
    printf("[io_drv] set-all helper    : %s\r\n",
           (errors == step_errors) ? "PASS" : "FAIL");

    // 5. Values wider than a field must not bleed into the neighbouring group.
    step_errors = errors;
    set_io_drive_strength_all(chip_id, IO_DRV_RESET);
    set_io_drive_strength(chip_id, IO_DRV_MISC, 0xff);
    errors += check_group(chip_id, 0, 0xf);  // MISC, masked down
    for (unsigned i = 1; i < N_IO_DRV_GROUPS; i++)
        errors += check_group(chip_id, i, IO_DRV_RESET);
    printf("[io_drv] out-of-range mask : %s\r\n",
           (errors == step_errors) ? "PASS" : "FAIL");

    // 6. Leave the pads as we found them.
    set_io_drive_strength_all(chip_id, IO_DRV_RESET);
    asm volatile("fence" : : : "memory");
    if (get_io_drive_strength_raw(chip_id) != reset_raw) {
        printf("  restore: got 0x%x, expected 0x%x\r\n",
               get_io_drive_strength_raw(chip_id), reset_raw);
        errors++;
    }

    printf("[io_drv] %s (%d error(s))\r\n", errors ? "FAIL" : "PASS", errors);
    return errors;
}
