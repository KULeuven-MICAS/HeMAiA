// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#include "chip_id.h"
#include "occamy.h"
#include "occamy_memory_map.h"

// *Note*: to ensure that the usr_data field is at the same offset
// in the host and device (resp. 64b and 32b architectures)
// usr_data is an explicitly-sized integer field instead of a pointer
typedef struct __attribute__((aligned(8))){
    volatile uint32_t lock;
    volatile uint32_t chip_id;
    volatile uint32_t usr_data_ptr;
    volatile uint32_t chip_barrier;
    // Chip Level synchronization mechanism: 16x16 chip matrix
    volatile uint8_t chip_level_checkpoint[256];
} comm_buffer_t;

/**************/
/* Interrupts */
/**************/

inline static void set_host_sw_interrupt(uint8_t chip_id) {
#if __riscv_xlen == 64
    volatile uint32_t* msip_ptr =
        (uint32_t*)(((uintptr_t)clint_msip_ptr(0)) |
                    ((uintptr_t)get_chip_baseaddress(chip_id)));
    *msip_ptr = 1;
#elif __riscv_xlen == 32
    volatile uint32_t* msip_ptr = clint_msip_ptr(0);
    uint32_t target_addrh = get_chip_baseaddress_value(chip_id) >> 32;
    uint32_t current_addrh = get_current_chip_baseaddress_value() >> 32;

    register uint32_t reg_target_addrh asm("t0") = target_addrh;
    register uint32_t reg_return_value asm("t1") = 1;
    register uint32_t reg_msip_ptr asm("t2") = (uint32_t)msip_ptr;
    register uint32_t reg_current_addrh asm("t3") = current_addrh;

    asm volatile(
        "csrw 0xbc0, t0;"
        "sw t1, 0(t2);"
        "csrw 0xbc0, t3;"
        :
        : "r"(reg_target_addrh), "r"(reg_return_value), "r"(reg_msip_ptr),
          "r"(reg_current_addrh)
        : "memory");
#endif
}

inline void clear_host_sw_interrupt_unsafe(uint8_t chip_id) {
#if __riscv_xlen == 64
    volatile uint32_t* msip_ptr =
        (uint32_t*)(((uintptr_t)clint_msip_ptr(0)) |
                    ((uintptr_t)get_chip_baseaddress(chip_id)));
    *msip_ptr = 0;
#elif __riscv_xlen == 32
    volatile uint32_t* msip_ptr = clint_msip_ptr(0);
    uint32_t target_addrh = get_chip_baseaddress_value(chip_id) >> 32;
    uint32_t current_addrh = get_current_chip_baseaddress_value() >> 32;

    register uint32_t reg_target_addrh asm("t0") = target_addrh;
    register uint32_t reg_return_value asm("t1") = 0;
    register uint32_t reg_msip_ptr asm("t2") = (uint32_t)msip_ptr;
    register uint32_t reg_current_addrh asm("t3") = current_addrh;

    asm volatile(
        "csrw 0xbc0, t0;"
        "sw t1, 0(t2);"
        "csrw 0xbc0, t3;"
        :
        : "r"(reg_target_addrh), "r"(reg_return_value), "r"(reg_msip_ptr),
          "r"(reg_current_addrh)
        : "memory");
#endif
}

inline void wait_host_sw_interrupt_clear(uint8_t chip_id) {
#if __riscv_xlen == 64
    volatile uint32_t* msip_ptr =
        (uint32_t*)(((uintptr_t)clint_msip_ptr(0)) |
                    ((uintptr_t)get_chip_baseaddress(chip_id)));
    while (*msip_ptr);
#elif __riscv_xlen == 32
    volatile uint32_t* msip_ptr = clint_msip_ptr(0);
    uint32_t target_addrh = get_chip_baseaddress_value(chip_id) >> 32;
    uint32_t current_addrh = get_current_chip_baseaddress_value() >> 32;

    register uint32_t reg_target_addrh asm("t0") = target_addrh;
    register uint32_t reg_value asm("t1");
    register uint32_t reg_msip_ptr asm("t2") = (uint32_t)msip_ptr;
    register uint32_t reg_current_addrh asm("t3") = current_addrh;

    do {
        asm volatile(
            "csrw 0xbc0, t0;"
            "lw t1, 0(t2);"
            "csrw 0xbc0, t3;"
            : "=r"(reg_value)
            : "r"(reg_target_addrh), "r"(reg_msip_ptr), "r"(reg_current_addrh)
            : "memory");
    } while (reg_value);
#endif
}

static inline void clear_host_sw_interrupt(uint8_t chip_id) {
    clear_host_sw_interrupt_unsafe(chip_id);
    wait_host_sw_interrupt_clear(chip_id);
}

static inline volatile uint32_t* __attribute__((const)) get_mutex_ptr_host() {
    #if __riscv_xlen == 64
    uint64_t comm_buffer_ptr_addr = chiplet_addr_transform((uint64_t)soc_ctrl_scratch_ptr(2));
    comm_buffer_t* comm_buffer_ptr;
    comm_buffer_ptr = (comm_buffer_t*)readd((uintptr_t)comm_buffer_ptr_addr);
    return &(comm_buffer_ptr->lock);
    #elif __riscv_xlen == 32
    return 0;
    #endif
}

static inline volatile uint32_t* __attribute__((const)) get_mutex_ptr_device() {
    #if __riscv_xlen == 64
    return 0;
    #elif __riscv_xlen == 32
    uint32_t comm_buffer_ptr_addr = (uint32_t)soc_ctrl_scratch_ptr(2);
    comm_buffer_t* comm_buffer_ptr;
    comm_buffer_ptr = (comm_buffer_t*)readw((uintptr_t)comm_buffer_ptr_addr);
    return &(comm_buffer_ptr->lock);
    #endif
}
