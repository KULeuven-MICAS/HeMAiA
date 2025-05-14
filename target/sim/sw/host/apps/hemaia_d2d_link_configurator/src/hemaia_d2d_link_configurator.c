// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include "host.h"
#include "hemaia_d2d_link_peripheral.h"

uint8_t *hemaia_d2d_link_cfg = (uint8_t *)HEMAIA_D2D_LINK_BASE_ADDR;

int main() {
    uintptr_t address_prefix = (uintptr_t)get_current_chip_baseaddress();

    init_uart(address_prefix, 32, 1);
    // Try to reset the D2D link
    asm volatile("fence" : : : "memory");
    print_str(address_prefix, "Press enter to reset the d2d_link \r\n");
    scan_char(address_prefix);
    *((uint16_t*)(hemaia_d2d_link_cfg + HEMAIA_D2D_LINK_RESET_REGISTER_REG_OFFSET)) = 0xFFFF; // Reset the d2d link
    asm volatile("fence" : : : "memory");
    *((uint16_t*)(hemaia_d2d_link_cfg + HEMAIA_D2D_LINK_RESET_REGISTER_REG_OFFSET)) = 0x0; // Release the reset
    asm volatile("fence" : : : "memory");
    print_str(address_prefix, "Press enter to set the threshold to 6 \r\n");
    scan_char(address_prefix);
    *((uint32_t*)(hemaia_d2d_link_cfg + HEMAIA_D2D_LINK_RX_BUFFER_THRESHOLD_REGISTER_REG_OFFSET)) = 0x06060606; // Set the threshold to 6
    asm volatile("fence" : : : "memory");
    print_str(address_prefix, "All operation is finished. \r\n");
    return 0;
}
