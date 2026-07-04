// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Host-side DVFS runtime.
//
// In DVFS mode the bingo hw manager PM only monitors chip-wide idle/busy and, on
// a transition, publishes the requested action in the quad_periph DVFS_REQUEST
// register and rings a doorbell by setting the host's dedicated CLINT MSIP bit.
// The host takes a machine software interrupt, reads DVFS_REQUEST, drives the
// PMIC (voltage) and the clk/rst controller (frequency) in the safe order,
// acknowledges the PM, and clears the doorbell. The PM re-arm handshake converges
// on the latest desired state.
//
// SAFETY CONTRACT. The chip must never
// clock fast at low Vdd, so the actuation order is not optional:
//   RAISE (busy):  pmic_set_voltage(hi)  -- BLOCKS until Vdd confirmed settled --
//                  then raise frequency.
//   LOWER (idle):  lower frequency (ensure applied), then pmic_set_voltage(lo).
// The RTL has no voltage feedback, so this ordering + the settle/confirm wait
// (pmic_set_voltage() does not return until Vdd has reached target) are enforced
// ONLY here in software. Write DVFS_ACK only after the full V+F transition is
// applied. The actuation below is mimicked (records + prints); the real PMIC-over
// -I2C + clk/rst writes, keeping this order, are the deferred SW effort.
//
// This header depends on helpers defined earlier in host.h (interrupt/CLINT/clk
// helpers, chip-id + quad_ctrl address helpers), so it is included at the end of
// host.h.
#pragma once

#include <stdint.h>
#include "pmic.h"

// Dedicated host DVFS doorbell bit inside the packed CLINT MSIP word: the HW-manager
// IPI target occamy.py appends after the harts, routed into the host's ipi_i in
// occamy_soc.sv.tpl. Sourced from the generated occamy.h (via host.h) so it is never
// hardcoded and tracks the hart count automatically. Must match HOST_DVFS_MSIP_BIT in
// bingo_hw_manager_pm.sv.
#define HOST_DVFS_MSIP_BIT HW_MANAGER_DVFS_MSIP_BIT

// DVFS_REQUEST field layout (must match occamy_quad_periph DVFS_REQUEST register).
#define DVFS_REQ_PENDING_MASK (1u << 0)
#define DVFS_REQ_DIR_MASK     (1u << 1)  // 1 = raise (chip busy), 0 = lower (idle)
#define DVFS_REQ_LEVEL_SHIFT  8
#define DVFS_REQ_LEVEL_MASK   0xFFu

// Set by dvfs_init(): the local chip id and how many clock domains scale together
// (DVFS is chiplet-level, so all active domains take the same division).
static uint8_t _dvfs_chip_id;
static uint8_t _dvfs_num_domains;

// -------------------------------------------------------------------------
// DVFS event log
// -------------------------------------------------------------------------
// The trap handler must stay reentrant-safe: it can fire while the host is
// mid-printf (printf is not reentrant / holds a UART lock), so it must NOT print.
// Instead it records each serviced DVFS_REQUEST here; main() narrates the
// (mimicked) voltage/frequency control sequence afterwards via dvfs_dump_log().
#define DVFS_LOG_MAX 64
static volatile uint32_t _dvfs_log[DVFS_LOG_MAX];
static volatile uint32_t _dvfs_log_count;

// How many doorbells to print live from inside the ISR (each line proves the host
// actually took + serviced the interrupt). Capped so a burst of transitions during
// scheduler startup does not flood the UART / slow the sim; the rest are counted
// and summarized by dvfs_dump_log().
#define DVFS_ISR_PRINT_MAX 24

// Reentrant-safe UART writes for use INSIDE the trap handler. print_char() (from
// uart.c, pulled in via host.h) only polls the UART TX-empty flag and writes THR --
// no lock and no static format buffer -- so it is safe to call while the host may be
// mid-printf. (base_address is uart.c's UART base, set by init_uart().)
static inline void dvfs_isr_puts(const char *s) {
    while (*s) print_char(base_address, *s++);
}
static inline void dvfs_isr_putdec(uint32_t v) {
    char buf[12];
    int i = 0;
    if (v == 0) { print_char(base_address, '0'); return; }
    while (v) { buf[i++] = (char)('0' + (v % 10)); v /= 10; }
    while (i) print_char(base_address, buf[--i]);
}

// Service one DVFS request published by the PM: read it, print a live "handled"
// line (reentrant-safe), record it, acknowledge the applied level so the PM updates
// its shadow and re-arms, then clear the doorbell.
static inline void dvfs_service_request(void) {
    uint32_t req = readw(
        (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_dvfs_request_addr()));

    if (req & DVFS_REQ_PENDING_MASK) {
        uint8_t level = (uint8_t)((req >> DVFS_REQ_LEVEL_SHIFT) & DVFS_REQ_LEVEL_MASK);
        if (_dvfs_log_count < DVFS_LOG_MAX) {
            _dvfs_log[_dvfs_log_count] = req;
        }
        _dvfs_log_count++;
        // Live proof that the host took + is servicing this interrupt.
        if (_dvfs_log_count <= DVFS_ISR_PRINT_MAX) {
            dvfs_isr_puts("[dvfs][isr] chip");
            dvfs_isr_putdec(_dvfs_chip_id);
            dvfs_isr_puts(" host handled DVFS doorbell #");
            dvfs_isr_putdec(_dvfs_log_count);
            dvfs_isr_puts((req & DVFS_REQ_DIR_MASK) ? ": RAISE (V up,F up) level="
                                                    : ": LOWER (F down,V down) level=");
            dvfs_isr_putdec(level);
            dvfs_isr_puts("\r\n");
        }
        writew(level,
               (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_dvfs_ack_addr()));
    }

    // Clear our dedicated doorbell bit (read-modify-write of the packed MSIP word).
    clear_sw_interrupt_unsafe(_dvfs_chip_id, HOST_DVFS_MSIP_BIT);
}

// Narrate the DVFS control sequence the ISR serviced. Call from main (NOT the ISR).
// For each doorbell the host would drive the PMIC (voltage) and clk/rst controller
// (frequency) in the safe order; the actuation is mimicked by these prints (the real
// PMIC-over-I2C + enable_clk_domain() calls are the deferred SW effort).
static inline void dvfs_dump_log(void) {
    uint32_t n = _dvfs_log_count;
    printf("[dvfs][chip%u] serviced %u DVFS doorbell event(s) across %u clk domain(s):\n",
           (unsigned)_dvfs_chip_id, (unsigned)n, (unsigned)_dvfs_num_domains);
    for (uint32_t i = 0; i < n && i < DVFS_LOG_MAX; i++) {
        uint32_t r = _dvfs_log[i];
        uint8_t level = (uint8_t)((r >> DVFS_REQ_LEVEL_SHIFT) & DVFS_REQ_LEVEL_MASK);
        if (r & DVFS_REQ_DIR_MASK) {
            printf("  #%u RAISE (chip busy) level=%u -> [mimic] PMIC Vdd up, then clk freq up; acked %u\n",
                   (unsigned)i, (unsigned)level, (unsigned)level);
        } else {
            printf("  #%u LOWER (chip idle) level=%u -> [mimic] clk freq down, then PMIC Vdd down; acked %u\n",
                   (unsigned)i, (unsigned)level, (unsigned)level);
        }
    }
}

// Machine-mode trap vector. The interrupt attribute makes GCC emit the context
// save/restore and mret. Only the machine software interrupt (mcause low bits
// == 3, i.e. MSIP -> ipi_i) is expected here.
__attribute__((interrupt("machine"), aligned(4)))
void dvfs_trap_handler(void) {
    uint64_t mcause;
    asm volatile("csrr %0, mcause" : "=r"(mcause));
    if ((mcause & 0xff) == 3) {  // machine software interrupt
        dvfs_service_request();
        // The DVFS doorbell (msip bit 3) and the host IPI (msip bit 0) share this
        // machine-software-interrupt cause. The device sets msip[0] on cluster
        // start/exit but the host never blocks on it (offload sync is via the
        // mailbox queues), so ack it here too to avoid re-trapping on it.
        clear_sw_interrupt_unsafe(_dvfs_chip_id, 0);
    }
}

// Enable DVFS mode. `num_domains` is the number of clock domains to scale together;
// `boot_level` is the power level the chip is at when armed (= the normal level) — it
// seeds DVFS_ACK so the PM's `pending = (desired != acked)` starts consistent. Passing
// it explicitly (rather than reading NORM_POWER_LEVEL) means this can be called before
// the PM registers are written, and avoids a spurious initial RAISE from a stale 0 ack.
static inline void dvfs_init(uint8_t num_domains, uint8_t boot_level) {
    _dvfs_chip_id     = get_current_chip_id();
    _dvfs_num_domains = num_domains;

    // Tell the PM where this chiplet's CLINT MSIP word lives (doorbell target).
    uint64_t clint_msip_word =
        (uintptr_t)get_current_chip_baseaddress() | clint_msip_base;
    writew((uint32_t)(clint_msip_word >> 32),
           (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_dvfs_clint_msip_hi_addr()));
    writew((uint32_t)(clint_msip_word),
           (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_dvfs_clint_msip_lo_addr()));

    // Seed DVFS_ACK with the boot (= normal) level so the PM starts at "normal" and does
    // not report a spurious RAISE before any real work: while the chip is busy at startup
    // desired==acked==boot_level, so it only rings once it actually goes idle (LOWER).
    writew(boot_level,
           (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_dvfs_ack_addr()));

    // Select DVFS mode (0 = DFS, 1 = DVFS). Done before EN_IDLE_PM is set (in
    // bingo_hw_scheduler_init_pm) so the PM never briefly runs autonomous DFS.
    writew(1, (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_pm_mode_addr()));

    // Install the trap vector (direct mode) and unmask the software interrupt.
    uint64_t tvec = (uint64_t)(uintptr_t)&dvfs_trap_handler;
    asm volatile("csrw mtvec, %0" : : "r"(tvec));
    enable_sw_interrupts();      // mie.MSIE
    enable_global_interrupts();  // mstatus.MIE
}
