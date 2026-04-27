// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#include "chip_id.h"
#include "occamy.h"
#include "occamy_memory_map.h"
#include "io.h"

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

// ============================================================================
// Per-Kernel Scratchpad (64 bytes = 16 x uint32_t)
// ============================================================================
// Shared between host (64-bit CVA6) and device (32-bit Snitch) cores.
// All fields are uint32_t to ensure identical layout on both architectures.
// The struct is aligned to 4 bytes (natural alignment for uint32_t arrays).
//
// return_value + num_return_values are FUNCTIONAL (always written by kernels).
// Timing fields are PROFILING (gated by BINGO_SCRATCHPAD_PROFILING).

#define BINGO_KERNEL_SCRATCHPAD_SIZE  64
#define BINGO_KERNEL_SCRATCHPAD_WORDS 16

typedef struct __attribute__((packed, aligned(4))) {
    uint32_t return_value;      // [0]  Result OR pointer to output data (32-bit)
    uint32_t num_return_values; // [1]  Element count at return_value ptr (0 = scalar)
    uint32_t start_time;        // [2]  mcycle at kernel start
    uint32_t end_time;          // [3]  mcycle at kernel end
    uint32_t arg_parse_time;    // [4]  Cycles for arg parsing
    uint32_t cfg_time;          // [5]  Cycles for accelerator config
    uint32_t kernel_run_time;   // [6]  Cycles for actual computation
    uint32_t reserved[9];       // [7-15] Future use
} bingo_kernel_scratchpad_t;

#ifdef BINGO_SCRATCHPAD_PROFILING
#define BINGO_SP_PROFILE(sp, field, val) ((sp)->field = (val))
#else
#define BINGO_SP_PROFILE(sp, field, val) ((void)0)
#endif

// Extract scratchpad from kernel args struct (architecture-aware).
// scratchpad_ptr is always the LAST field in the args struct.
#include <stddef.h>
#if __riscv_xlen == 64
// Host (64-bit): scratchpad_ptr is uint64_t
#define BINGO_GET_SCRATCHPAD(arg, args_type) \
    ((bingo_kernel_scratchpad_t*)(uintptr_t)(*(uint64_t*)((char*)(arg) + offsetof(args_type, scratchpad_ptr))))
#elif __riscv_xlen == 32
// Device (32-bit): scratchpad_ptr is uint32_t
#define BINGO_GET_SCRATCHPAD(arg, args_type) \
    ((bingo_kernel_scratchpad_t*)(uintptr_t)(*(uint32_t*)((char*)(arg) + offsetof(args_type, scratchpad_ptr))))
#endif

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

static inline void clear_host_sw_interrupt_unsafe(uint8_t chip_id) {
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

// Cross-chiplet byte read from a Snitch dev core (RV32). The chiplet prefix
// goes into Mseg (CSR 0xbc0); the inner `lbu` carries the low 32 bits of the
// remote address; Mseg is restored to the current chiplet immediately after.
// On RV64 (host) Mseg is unused — the address can be dereferenced directly.
#if __riscv_xlen == 32
static inline uint8_t bingo_remote_readb(uint8_t chip_id, uint32_t addr_lo) {
    uint32_t target_addrh  = get_chip_baseaddress_value(chip_id) >> 32;
    uint32_t current_addrh = get_current_chip_baseaddress_value() >> 32;

    register uint32_t reg_target_addrh  asm("t0") = target_addrh;
    register uint32_t reg_value         asm("t1");
    register uint32_t reg_addr          asm("t2") = addr_lo;
    register uint32_t reg_current_addrh asm("t3") = current_addrh;

    asm volatile(
        "csrw 0xbc0, t0;"
        "lbu  t1, 0(t2);"
        "csrw 0xbc0, t3;"
        : "=r"(reg_value)
        : "r"(reg_target_addrh), "r"(reg_addr), "r"(reg_current_addrh)
        : "memory");
    return (uint8_t)reg_value;
}
#elif __riscv_xlen == 64
static inline uint8_t bingo_remote_readb(uint8_t chip_id, uint32_t addr_lo) {
    volatile uint8_t* p = (uint8_t*)(((uintptr_t)addr_lo) | (uintptr_t)get_chip_baseaddress(chip_id));
    return *p;
}
#endif

static inline void wait_host_sw_interrupt_clear(uint8_t chip_id) {
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
/**
 * @brief lock a mutex, blocking
 * @details test-and-set (tas) implementation of a lock.
 *          Declare mutex with `static volatile uint32_t mtx = 0;`
 */
static inline void mutex_tas_acquire(volatile uint32_t* pmtx) {
    asm volatile(
        "li            x5,1          # x5 = 1\n"
        "1:\n"
        "  amoswap.w.aq  x5,x5,(%0)   # x5 = oldlock & lock = 1\n"
        "  bnez          x5,1b      # Retry if previously set)\n"
        : "+r"(pmtx)
        :
        : "x5");
}

/**
 * @brief lock a mutex, blocking
 * @details test-and-test-and-set (ttas) implementation of a lock.
 *          Declare mutex with `static volatile uint32_t mtx = 0;`
 */
static inline void mutex_ttas_acquire(volatile uint32_t* pmtx) {
    asm volatile(
        "1:\n"
        "  lw x5, 0(%0)\n"
        "  bnez x5, 1b\n"
        "  li x5,1          # x5 = 1\n"
        "2:\n"
        "  amoswap.w.aq  x5,x5,(%0)   # x5 = oldlock & lock = 1\n"
        "  bnez          x5,2b      # Retry if previously set)\n"
        : "+r"(pmtx)
        :
        : "x5");
}

/**
 * @brief Release the mutex
 */
static inline void mutex_release(volatile uint32_t* pmtx) {
    asm volatile("amoswap.w.rl  x0,x0,(%0)   # Release lock by storing 0\n"
                 : "+r"(pmtx));
}

static inline void clear_host_sw_interrupt(uint8_t chip_id) {
    clear_host_sw_interrupt_unsafe(chip_id);
    wait_host_sw_interrupt_clear(chip_id);
}
static inline volatile uint32_t* get_shared_lock() {
#if __riscv_xlen == 64
    // For the host side, the comm buffer is located at the soc_ctrl_sractch(1)
    // Notice this address must be aligned to 8B
    // A little hacky way to get the comm buffer pointer
    // See bootrom.ld
    // the soc_ctrl_scratch_addr_base = 0x2000_0014
    // So we pick the comm_buffer_ptr to be at offset 0x2000_0014 + 4 = 0x2000_0018 and it is 8B aligned with 8B size, otherwise the host core cannot access this address
    uint32_t comm_buffer_ptr = readw(chiplet_addr_transform((uintptr_t)soc_ctrl_scratch_addr(1)));
    uint64_t comm_buffer_ptr_64 = chiplet_addr_transform((uint64_t)comm_buffer_ptr);
    return &(((comm_buffer_t *)comm_buffer_ptr_64)->lock);
#elif __riscv_xlen == 32
    uint32_t comm_buffer_ptr = readw(soc_ctrl_scratch_addr(1));
    return &(((comm_buffer_t *)comm_buffer_ptr)->lock);
#endif    
}
