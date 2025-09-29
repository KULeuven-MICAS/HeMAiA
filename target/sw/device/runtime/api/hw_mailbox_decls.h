// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

#define MBOX_DEVICE_READY (0x01U)
#define MBOX_DEVICE_START (0x02U)
#define MBOX_DEVICE_BUSY (0x03U)
#define MBOX_DEVICE_DONE (0x04U)
#define MBOX_DEVICE_STOP (0x0FU)

int h2c_mailbox_try_read(uint32_t *buffer);
int h2c_mailbox_read(uint32_t *buffer);
int c2h_mailbox_try_write(uint32_t word);
int c2h_mailbox_write(uint32_t word);