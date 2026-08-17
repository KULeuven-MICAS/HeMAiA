// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

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

inline void set_host_sw_interrupt() { *clint_msip_ptr(0) = 1; }

inline void clear_host_sw_interrupt_unsafe() { *clint_msip_ptr(0) = 0; }

inline void wait_host_sw_interrupt_clear() {
    while (*clint_msip_ptr(0))
        ;
}

inline void clear_host_sw_interrupt() {
    clear_host_sw_interrupt_unsafe();
    wait_host_sw_interrupt_clear();
}

/*********/
/* Mutex */
/*********/

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

// Thread-safe printf, see src/printf_safe.c for the implementation.
// Uses the comm_buffer_t.lock above (via get_shared_lock() in
// occamy_device.h) so that concurrent cores don't interleave their output.
int printf_safe(const char *fmt, ...);

/**************************/
/* Quadrant configuration */
/**************************/

// Configure RO cache address range
inline void configure_read_only_cache_addr_rule(uint32_t quad_idx,
                                                uint32_t rule_idx,
                                                uint64_t start_addr,
                                                uint64_t end_addr) {
    volatile uint64_t* rule_ptr =
        quad_cfg_ro_cache_addr_rule_ptr(quad_idx, rule_idx);
    *(rule_ptr) = start_addr;
    *(rule_ptr + 1) = end_addr;
}

// Enable RO cache
inline void enable_read_only_cache(uint32_t quad_idx) {
    *(quad_cfg_ro_cache_enable_ptr(quad_idx)) = 1;
}
