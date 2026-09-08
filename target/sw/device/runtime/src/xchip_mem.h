// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Cross-chiplet scalar memory access from a Snitch device core.
//
// THE PROBLEM. HeMAiA's address space is 48 bits wide and bits [47:40] hold the chiplet
// id, but a Snitch dev core is RV32: a remote address simply does not fit in a pointer.
// The host (RV64) has no such trouble and just ORs the prefix in, which is why these
// helpers are device-only and live here rather than in the shared runtime header.
//
// THE MECHANISM. The Mseg CSR (0xbc0) supplies the high address bits for the next
// load/store, so every access is the same three-instruction sandwich:
//
//     csrw 0xbc0, <target chiplet prefix>
//     lw / sw / lbu   <the low 32 bits of the address>
//     csrw 0xbc0, <this chiplet's prefix>      // always restore
//
// This is the same sequence announce_chip_checkpoint() uses, except that it aims at ONE
// specific chiplet instead of the 0xFF broadcast prefix.
//
// THE PRECONDITION, and it is the easy one to get wrong: `addr_lo` must be a LOCAL
// address that means the same thing on the target chiplet as it does here. That holds
// for TCDM and for the comm buffer, because every chiplet places them at identical local
// addresses and only the prefix differs -- so "&my_thing" is also "&its_thing". It does
// NOT hold for anything whose placement is per-chip.
//
// WRITES ARE FIRE-AND-FORGET. The D2D framer drops the remote B response and
// synthesizes a local one, so snrt_xchip_writew() retiring does NOT prove the data
// landed on the far side. Never infer progress from a store completing; infer it from
// reading back a value someone else wrote. A reader is the only real acknowledgement.
//
// NO ATOMICS ACROSS CHIPLETS. A remote AMO is resolved on the ISSUING chip by
// i_soc_narrow_wide_amo_adapter and emitted downstream as a plain read + write, so two
// chiplets updating one remote location lose updates. Build cross-chip protocols out of
// single-writer locations, never out of a shared counter.
#pragma once

#include <stdint.h>

#include "chip_id.h"

// A pointer to a structure that every chiplet places at the same local address, reduced
// to the prefix-free 32-bit offset these helpers take. On RV32 the pointer is already
// local; the cast is a no-op. On RV64 it drops the chiplet prefix in bits [47:40].
#define SNRT_XCHIP_LOCAL_OFF(ptr) ((uint32_t)(uintptr_t)(ptr))

// Store a 32-bit word into `chip_id`'s copy of `addr_lo`.
inline void snrt_xchip_writew(uint8_t chip_id, uint32_t addr_lo, uint32_t val) {
    uint32_t target_addrh = get_chip_baseaddress_value(chip_id) >> 32;
    uint32_t current_addrh = get_current_chip_baseaddress_value() >> 32;

    register uint32_t reg_target_addrh asm("t0") = target_addrh;
    register uint32_t reg_value asm("t1") = val;
    register uint32_t reg_addr asm("t2") = addr_lo;
    register uint32_t reg_current_addrh asm("t3") = current_addrh;

    asm volatile(
        "csrw 0xbc0, t0;"
        "sw   t1, 0(t2);"
        "csrw 0xbc0, t3;"
        :
        : "r"(reg_target_addrh), "r"(reg_value), "r"(reg_addr),
          "r"(reg_current_addrh)
        : "memory");
}

// Load a 32-bit word from `chip_id`'s copy of `addr_lo`.
inline uint32_t snrt_xchip_readw(uint8_t chip_id, uint32_t addr_lo) {
    uint32_t target_addrh = get_chip_baseaddress_value(chip_id) >> 32;
    uint32_t current_addrh = get_current_chip_baseaddress_value() >> 32;

    register uint32_t reg_target_addrh asm("t0") = target_addrh;
    register uint32_t reg_value asm("t1");
    register uint32_t reg_addr asm("t2") = addr_lo;
    register uint32_t reg_current_addrh asm("t3") = current_addrh;

    asm volatile(
        "csrw 0xbc0, t0;"
        "lw   t1, 0(t2);"
        "csrw 0xbc0, t3;"
        : "=r"(reg_value)
        : "r"(reg_target_addrh), "r"(reg_addr), "r"(reg_current_addrh)
        : "memory");
    return reg_value;
}

// Store a byte into `chip_id`'s copy of `addr_lo`.
inline void snrt_xchip_writeb(uint8_t chip_id, uint32_t addr_lo, uint8_t val) {
    uint32_t target_addrh = get_chip_baseaddress_value(chip_id) >> 32;
    uint32_t current_addrh = get_current_chip_baseaddress_value() >> 32;

    register uint32_t reg_target_addrh asm("t0") = target_addrh;
    register uint32_t reg_value asm("t1") = (uint32_t)val;
    register uint32_t reg_addr asm("t2") = addr_lo;
    register uint32_t reg_current_addrh asm("t3") = current_addrh;

    asm volatile(
        "csrw 0xbc0, t0;"
        "sb   t1, 0(t2);"
        "csrw 0xbc0, t3;"
        :
        : "r"(reg_target_addrh), "r"(reg_value), "r"(reg_addr),
          "r"(reg_current_addrh)
        : "memory");
}

// Load a byte from `chip_id`'s copy of `addr_lo`.
inline uint8_t snrt_xchip_readb(uint8_t chip_id, uint32_t addr_lo) {
    uint32_t target_addrh = get_chip_baseaddress_value(chip_id) >> 32;
    uint32_t current_addrh = get_current_chip_baseaddress_value() >> 32;

    register uint32_t reg_target_addrh asm("t0") = target_addrh;
    register uint32_t reg_value asm("t1");
    register uint32_t reg_addr asm("t2") = addr_lo;
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
