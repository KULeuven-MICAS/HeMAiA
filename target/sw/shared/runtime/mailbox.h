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

inline void* h2h_mailbox_address(){
    return (void*)H2H_MAILBOX_BASE_ADDR;
}

inline void* c2h_mailbox_address(){
    return (void*)C2H_MAILBOX_BASE_ADDR;
}

inline void* h2c_mailbox_address(uint32_t cluster_idx){
    return (void*)(H2C_MAILBOX_BASE_ADDR + cluster_idx * MAILBOX_ADDR_SPACE);
}