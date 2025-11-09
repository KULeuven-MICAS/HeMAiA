// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include <stdint.h>
#include "occamy_base_addr.h"
#include "occamy_memory_map.h"
#define H2C_MAILBOX_BASE_ADDR QUAD_0_CFG_BASE_ADDR

#define MAILBOX_WRITE_DATA_ADDR_OFFSET           0
#define MAILBOX_READ_DATA_ADDR_OFFSET            1
#define MAILBOX_STATUS_FLAG_OFFSET               2
#define MAILBOX_ERROR_FLAG_OFFSET                3
#define MAILBOX_WRITE_DATA_INTR_THRESHOLD_OFFSET 4
#define MAILBOX_READ_DATA_INTR_THRESHOLD_OFFSET  5
#define MAILBOX_INTR_STATUS_OFFSET               6
#define MAILBOX_INTR_ENABLE_OFFSET               7
#define MAILBOX_INTR_PENDING_OFFSET              8
#define MAILBOX_CONTROL_REG_OFFSET               9

inline uint64_t h2h_mailbox_base_address(){
    return (uint64_t)H2H_MAILBOX_BASE_ADDR;
}

inline uint32_t c2h_mailbox_base_address(){
    return (uint32_t)C2H_MAILBOX_BASE_ADDR;
}

inline uint64_t h2c_mailbox_base_address(uint8_t cluster_idx){
    return (uint64_t)(H2C_MAILBOX_BASE_ADDR + cluster_idx * MAILBOX_ADDR_SPACE);
}

inline uint64_t h2h_mailbox_write_address(){
    // The host 2 host mailbox data width is 64bit
    // Hence we multiply the offset by 8
    return h2h_mailbox_base_address() + MAILBOX_WRITE_DATA_ADDR_OFFSET * 8;
}

inline uint64_t h2h_mailbox_read_address(){
    // The host 2 host mailbox data width is 64bit
    // Hence we multiply the offset by 8
    return h2h_mailbox_base_address() + MAILBOX_READ_DATA_ADDR_OFFSET * 8;
}

inline uint64_t h2h_mailbox_status_flag_address(){
    // The host 2 host mailbox data width is 64bit
    // Hence we multiply the offset by 8
    // The status flag is a 64bit register
    // Bit 0: Empty flag: Read FIFO is empty
    // Bit 1: Full flag: Write FIFO is full and subsequent writes to mailbox are ignored
    // Bit 2: Write FIFO level is higher than the threshold set in WIRQT
    // Bit 3: Read FIFO level is higher than the threshold set in RIRQT
    return h2h_mailbox_base_address() + MAILBOX_STATUS_FLAG_OFFSET * 8;
}

inline uint64_t h2c_mailbox_write_address(uint8_t cluster_idx){
    // The host 2 cluster mailbox data width is 32bit
    // Hence we multiply the offset by 4
    return h2c_mailbox_base_address(cluster_idx) + MAILBOX_WRITE_DATA_ADDR_OFFSET * 4;
}

// only the cluster can read h2c mailbox data,
// hence the address is uint32_t
inline uint32_t h2c_mailbox_read_address(uint8_t cluster_idx){
    // The host 2 cluster mailbox data width is 32bit
    // Hence we multiply the offset by 4
    return (uint32_t)(h2c_mailbox_base_address(cluster_idx) + MAILBOX_READ_DATA_ADDR_OFFSET * 4);
}

inline uint64_t h2c_mailbox_status_flag_address(uint8_t cluster_idx){
    // The host 2 cluster mailbox data width is 32bit
    // Hence we multiply the offset by 4
    // The status flag is a 32bit register
    // Bit 0: Empty flag: Read FIFO is empty
    // Bit 1: Full flag: Write FIFO is full and subsequent writes to mailbox are ignored
    // Bit 2: Write FIFO level is higher than the threshold set in WIRQT
    // Bit 3: Read FIFO level is higher than the threshold set in RIRQT
    return h2c_mailbox_base_address(cluster_idx) + MAILBOX_STATUS_FLAG_OFFSET * 4;
}


inline uint32_t c2h_mailbox_write_address(){
    // The cluster 2 host mailbox data width is 32bit
    // Hence we multiply the offset by 4
    return (uint32_t)(c2h_mailbox_base_address() + MAILBOX_WRITE_DATA_ADDR_OFFSET * 4);
}

inline uint64_t c2h_mailbox_read_address(){
    // The cluster 2 host mailbox data width is 32bit
    // Hence we multiply the offset by 4
    return c2h_mailbox_base_address() + MAILBOX_READ_DATA_ADDR_OFFSET * 4;
}

inline uint64_t c2h_mailbox_status_flag_address(){
    // The cluster 2 host mailbox data width is 32bit
    // Hence we multiply the offset by 4
    // The status flag is a 32bit register
    // Bit 0: Empty flag: Read FIFO is empty
    // Bit 1: Full flag: Write FIFO is full and subsequent writes to mailbox are ignored
    // Bit 2: Write FIFO level is higher than the threshold set in WIRQT
    // Bit 3: Read FIFO level is higher than the threshold set in RIRQT
    return c2h_mailbox_base_address() + MAILBOX_STATUS_FLAG_OFFSET * 4;
}