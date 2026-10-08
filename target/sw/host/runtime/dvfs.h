// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>
#include "tps6287x.h"

// Include after host.h. Defaults match the PLL testbench; validate for the board.
#define DVFS_PMIC_ADDR  0x41u     // Unshifted 7-bit address: 0x40 through 0x43.
#define DVFS_PERIPH_HZ  32000000u // I2C peripheral clock in Hz, independent of CPU DVFS.
#define DVFS_NORMAL_MV  1000      // Normal voltage in mV: 400 through 1200.
#define DVFS_IDLE_MV    800       // Idle voltage in mV: 400 through DVFS_NORMAL_MV.
#define DVFS_SETTLE_US  50u       // Settling after boot/RAISE, in us.
#define DVFS_MASTER_HZ  UINT64_C(4000000000) // PLL output before per-domain division.

#define HOST_DVFS_MSIP_BIT HW_MANAGER_DVFS_MSIP_BIT

#define DVFS_REQ_PENDING_MASK (1u << 0)
#define DVFS_REQ_DIR_MASK     (1u << 1)  // 1 = raise (chip busy), 0 = lower (idle)
#define DVFS_REQ_LEVEL_SHIFT  8
#define DVFS_REQ_LEVEL_MASK   0xFFu

typedef struct {
    uint8_t pmic_address;
    uint8_t normal_level; // PM levels are clock divisors, not voltage codes.
    uint8_t idle_level;
    uint32_t peripheral_clock_hz;
    int normal_mv;
    int idle_mv;
    uint32_t settle_us;
} dvfs_config_t;

static uint8_t _dvfs_chip_id;
static uint8_t _dvfs_num_domains;
static dvfs_config_t _dvfs_config;
static uint8_t _dvfs_ready;

static inline uint8_t dvfs_host_divisor(void) {
    uintptr_t address = (uintptr_t)get_current_chip_baseaddress() |
                        HEMAIA_CLK_RST_CONTROLLER_BASE_ADDR;
    uint8_t divisor = (uint8_t)readw(address +
        HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C0_C3_REG_OFFSET);
    // Channel 0 is the host. Zero at reset falls back to a conservative /1 wait.
    return divisor ? divisor : 1;
}

static inline void dvfs_wait_cycles(uint64_t start, uint64_t cycles) {
    // Unsigned elapsed time handles counter wrap; mcycle must remain enabled.
    while ((uint64_t)(mcycle() - start) < cycles) {}
}

static inline void dvfs_wait_settle(void) {
    uint64_t start = mcycle();
    uint64_t denominator = (uint64_t)dvfs_host_divisor() * 1000000u;
    uint64_t cycles = (DVFS_MASTER_HZ * _dvfs_config.settle_us +
                       denominator - 1u) / denominator;
    // Round up to CPU cycles; the wait includes the read/calculation overhead.
    dvfs_wait_cycles(start, cycles);
}

static inline void dvfs_set_clock_level(uint8_t level) {
    for (uint8_t domain = 0; domain < _dvfs_num_domains; ++domain)
        enable_clk_domain(domain, level);
    fence();
    // On LOWER/boot, the following I2C transaction covers divider/CDC latency.
}

static inline int dvfs_apply_request(uint32_t req) {
    if (!_dvfs_ready) return TPS6287X_ERR_NOT_INITIALIZED;
    uint8_t level = (uint8_t)((req >> DVFS_REQ_LEVEL_SHIFT) & DVFS_REQ_LEVEL_MASK);
    int raise = (req & DVFS_REQ_DIR_MASK) != 0;
    if (level != (raise ? _dvfs_config.normal_level : _dvfs_config.idle_level))
        return TPS6287X_ERR_CONFIG;

    // LOWER: F first, then V. RAISE: V first, settle, then F.
    if (!raise) dvfs_set_clock_level(level);
    int actual_mv = tps6287x_set_voltage(_dvfs_config.pmic_address,
        raise ? _dvfs_config.normal_mv : _dvfs_config.idle_mv);
    if (actual_mv < 0) return actual_mv;
    if (raise) {
        dvfs_wait_settle();
        dvfs_set_clock_level(level);
    }
    // LOWER needs no analog wait: the reduced frequency is safe during ramp-down.
    return actual_mv;
}

#define DVFS_LOG_MAX 64
// Keep UART outside the ISR; print this bounded log after the workload.
static volatile uint32_t _dvfs_log[DVFS_LOG_MAX];
static volatile int _dvfs_log_result[DVFS_LOG_MAX];
static volatile uint32_t _dvfs_log_count;

static inline int dvfs_service_request(void) {
    // Clear before ACK so a re-armed doorbell is not lost.
    clear_sw_interrupt_unsafe(_dvfs_chip_id, HOST_DVFS_MSIP_BIT);
    int result = TPS6287X_OK;
    uint32_t req = readw(
        (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_dvfs_request_addr()));

    if (req & DVFS_REQ_PENDING_MASK) {
        uint8_t level = (uint8_t)((req >> DVFS_REQ_LEVEL_SHIFT) & DVFS_REQ_LEVEL_MASK);
        result = dvfs_apply_request(req);
        if (result >= 0) {
            writew(level,
                   (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_dvfs_ack_addr()));
        }
        if (_dvfs_log_count < DVFS_LOG_MAX) {
            _dvfs_log[_dvfs_log_count] = req;
            _dvfs_log_result[_dvfs_log_count] = result;
        }
        _dvfs_log_count++;
    }

    return result < 0 ? result : TPS6287X_OK;
}

static inline void dvfs_dump_log(void) {
    uint32_t n = _dvfs_log_count;
    printf("[dvfs][chip%u] serviced %u DVFS doorbell event(s) across %u clk domain(s):\n",
           (unsigned)_dvfs_chip_id, (unsigned)n, (unsigned)_dvfs_num_domains);
    for (uint32_t i = 0; i < n && i < DVFS_LOG_MAX; i++) {
        uint32_t r = _dvfs_log[i];
        uint8_t level = (uint8_t)((r >> DVFS_REQ_LEVEL_SHIFT) & DVFS_REQ_LEVEL_MASK);
        if (_dvfs_log_result[i] < 0) {
            printf("  #%u FAILED level=%u error=%d; not acknowledged\n",
                   (unsigned)i, (unsigned)level, _dvfs_log_result[i]);
            continue;
        }
        if (r & DVFS_REQ_DIR_MASK) {
            printf("  #%u RAISE level=%u -> PMIC %d mV + settle wait, then clk freq up; acked\n",
                   (unsigned)i, (unsigned)level, _dvfs_log_result[i]);
        } else {
            printf("  #%u LOWER level=%u -> clk freq down, then PMIC %d mV; acked\n",
                   (unsigned)i, (unsigned)level, _dvfs_log_result[i]);
        }
    }
}

#ifdef __riscv
__attribute__((interrupt("machine"), aligned(4)))
void dvfs_trap_handler(void) {
    uint64_t mcause;
    asm volatile("csrr %0, mcause" : "=r"(mcause));
    if (mcause == ((UINT64_C(1) << 63) | 3u)) {
        clear_host_sw_interrupt_unsafe(_dvfs_chip_id);
        int rc = dvfs_service_request();
        if (rc < 0) host_abort((uint64_t)-rc);
    } else host_abort((mcause & 0xffu) + 1u);
}
#endif

// Call before EN_IDLE_PM/workload start. Domains include host + clusters, not D2D.
static inline int dvfs_init(uint8_t num_domains, const dvfs_config_t *config) {
    if (config == 0 || num_domains == 0 || num_domains > N_CLUSTERS_PER_CHIPLET + 1 ||
        config->normal_level == 0 || config->normal_level >= config->idle_level ||
        config->normal_mv < TPS6287X_MIN_MV || config->normal_mv > TPS6287X_MAX_MV ||
        config->idle_mv < TPS6287X_MIN_MV || config->idle_mv > config->normal_mv ||
        config->settle_us == 0 || config->settle_us > 1000000u)
        return TPS6287X_ERR_CONFIG;
    _dvfs_chip_id     = get_current_chip_id();
    _dvfs_num_domains = num_domains;
    _dvfs_config = *config;
    _dvfs_ready = 0;
    _dvfs_log_count = 0;
    int rc = tps6287x_init(config->pmic_address, config->peripheral_clock_hz);
    if (rc < 0) return rc;
    // Start at idle; the first busy request raises voltage before frequency.
    dvfs_set_clock_level(config->idle_level);
    rc = tps6287x_set_voltage(config->pmic_address, config->idle_mv);
    if (rc < 0) return rc;
    dvfs_wait_settle();
    _dvfs_ready = 1;

    uint64_t clint_msip_word =
        (uintptr_t)get_current_chip_baseaddress() | clint_msip_base;
    writew((uint32_t)(clint_msip_word >> 32),
           (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_dvfs_clint_msip_hi_addr()));
    writew((uint32_t)(clint_msip_word),
           (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_dvfs_clint_msip_lo_addr()));

    // Seed the applied level before enabling the hardware PM and its interrupt.
    writew(config->idle_level,
           (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_dvfs_ack_addr()));

    writew(1, (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_pm_mode_addr()));

#ifdef __riscv
    uint64_t tvec = (uint64_t)(uintptr_t)&dvfs_trap_handler;
    asm volatile("csrw mtvec, %0" : : "r"(tvec));
    enable_sw_interrupts();      // mie.MSIE
    enable_global_interrupts();  // mstatus.MIE
#endif
    return TPS6287X_OK;
}
