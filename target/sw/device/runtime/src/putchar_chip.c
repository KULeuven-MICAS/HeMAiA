// Copyright 2021 ETH Zurich, University of Bologna; Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "uart.h"

void _putchar(char character) {
    // Print to UART of local chip
    print_char((uintptr_t)0, character);
}
