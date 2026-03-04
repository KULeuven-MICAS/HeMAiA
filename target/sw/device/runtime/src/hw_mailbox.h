// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

// Device-side hardware mailbox helper API
// Provides blocking and non-blocking accessors for H2C (host->cluster) and
// C2H (cluster->host) mailboxes from cluster-side runtime code.

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "hw_mailbox_decls.h"
#include "mailbox.h"

// Status bit semantics (mirrors host-side comments):
// Bit0: EMPTY (1 means read FIFO empty)
// Bit1: FULL  (1 means write FIFO full)

#ifndef BINGO_EXTRACT_BIT
#define BINGO_EXTRACT_BIT(val, bit) (((val) >> (bit)) & 0x1u)
#endif

// Return conventions (aligned with host runtime):
// try_* : 1 = success, 0 = would block, <0 = error
// blocking variants: 0 = success, <0 = error (no timeout variant here)

int h2c_mailbox_try_read(uint32_t *buffer);
int h2c_mailbox_read(uint32_t *buffer);
int c2h_mailbox_try_write(uint32_t word);
int c2h_mailbox_write(uint32_t word);

