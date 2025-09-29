// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Device-side mailbox access implementation.
// Implements non-blocking (try_*) and blocking primitives for:
//   H2C (Host -> Cluster): cluster reads words the host enqueued.
//   C2H (Cluster -> Host): cluster writes words consumed by the host.
//
// Status register bit semantics (32-bit status registers):
//   Bit 0: EMPTY  (1 => Read FIFO empty, no data to read)
//   Bit 1: FULL   (1 => Write FIFO full, cannot write)
//   Bit 2: Write FIFO level > write interrupt threshold (unused here)
//   Bit 3: Read  FIFO level > read  interrupt threshold (unused here)
//
// Return conventions:
//   try_read / try_write: 1 success, 0 would block, <0 error
//   read / write (blocking): 0 success, <0 error

#include "hw_mailbox.h"

#ifndef BINGO_EXTRACT_BIT
#define BINGO_EXTRACT_BIT(v,b) (((v) >> (b)) & 0x1u)
#endif


// -------------------------
// H2C  (Host -> Cluster)   : cluster reads
// -------------------------

int h2c_mailbox_try_read(uint32_t *buffer) {
	if (!buffer) return -1; // param error
	uint8_t cluster_id = snrt_cluster_idx();
	volatile uint32_t read_addr = h2c_mailbox_read_address(cluster_id);
	volatile uint32_t status_addr = (uint32_t)h2c_mailbox_status_flag_address(cluster_id);
	uint32_t status = readw((uintptr_t)status_addr);
	if (BINGO_EXTRACT_BIT(status, 0)) {
		// EMPTY
		return 0; // would block
	}
	*buffer = readw((uintptr_t)read_addr);
	return 1; // success
}

int h2c_mailbox_read(uint32_t *buffer) {
	if (!buffer) return -1;
	while (1) {
		int r = h2c_mailbox_try_read(buffer);
		if (r == 1) return 0; // success
	}
}

// -------------------------
// C2H (Cluster -> Host)    : cluster writes
// -------------------------

int c2h_mailbox_try_write(uint32_t word) {
	volatile uint32_t write_addr = c2h_mailbox_write_address();
	volatile uint32_t status_addr = c2h_mailbox_status_flag_address();
	uint32_t status = readw((uintptr_t)status_addr);
	if (BINGO_EXTRACT_BIT(status, 1)) {
		// FULL
		return 0; // would block
	}
	writew(word, (uintptr_t)write_addr);
	return 1; // success
}

int c2h_mailbox_write(uint32_t word) {
	while (1) {
		int r = c2h_mailbox_try_write(word);
		if (r == 1) return 0; // success
	}
}

