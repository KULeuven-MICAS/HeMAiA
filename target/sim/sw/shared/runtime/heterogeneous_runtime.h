// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "chip_id.h"
#include "uart.h"
#include "occamy.h"
#include "occamy_memory_map.h"

// *Note*: to ensure that the usr_data field is at the same offset
// in the host and device (resp. 64b and 32b architectures)
// usr_data is an explicitly-sized integer field instead of a pointer
typedef struct {
    volatile uint32_t lock;
    volatile uint32_t usr_data_ptr;
} comm_buffer_t;

/**************/
/* Interrupts */
/**************/

inline static void set_host_sw_interrupt(uint8_t chip_id) {
    uint32_t* msip_ptr =
        (uint32_t*)(((uintptr_t)clint_msip_ptr(0)) |
                    ((uintptr_t)get_chip_baseaddress(chip_id)));
    print_u32(0, (uintptr_t)msip_ptr);
    *msip_ptr = 1;
}

inline void clear_host_sw_interrupt_unsafe(uint8_t chip_id) {
    uint32_t* msip_ptr =
        (uint32_t*)(((uintptr_t)clint_msip_ptr(0)) |
                    ((uintptr_t)get_chip_baseaddress(chip_id)));

    *msip_ptr = 0;
}

inline void wait_host_sw_interrupt_clear(uint8_t chip_id) {
    uint32_t* msip_ptr =
        (uint32_t*)(((uintptr_t)clint_msip_ptr(0)) |
                    ((uintptr_t)get_chip_baseaddress(chip_id)));

    while (*msip_ptr);
}

static inline void clear_host_sw_interrupt(uint8_t chip_id) {
    clear_host_sw_interrupt_unsafe(chip_id);
    wait_host_sw_interrupt_clear(chip_id);
}

/**************************/
/* Quadrant configuration */
/**************************/

// Configure RO cache address range
inline void configure_read_only_cache_addr_rule(uint8_t chip_id,
                                                uint32_t quad_idx,
                                                uint32_t rule_idx,
                                                uint64_t start_addr,
                                                uint64_t end_addr) {
    volatile uint64_t* rule_ptr =
        (uint64_t*)(((uintptr_t)quad_cfg_ro_cache_addr_rule_ptr(quad_idx,
                                                                rule_idx)) |
                    ((uintptr_t)get_chip_baseaddress(chip_id)));
    *(rule_ptr) = start_addr;
    *(rule_ptr + 1) = end_addr;
}

// Enable RO cache
inline void enable_read_only_cache(uint8_t chip_id, uint32_t quad_idx) {
    volatile uint32_t* enable_ptr =
        (uint32_t*)(((uintptr_t)quad_cfg_ro_cache_enable_ptr(quad_idx)) |
                    ((uintptr_t)get_chip_baseaddress(chip_id)));
    *enable_ptr = 1;
}
