// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

#pragma once
#include <stdint.h>

inline uint8_t get_current_chip_id() {
    uint32_t chip_id;
# if __riscv_xlen == 64
    // 64-bit system (CVA6), get chip_id from 0xf15
    asm volatile("csrr %0, 0xf15" : "=r"(chip_id));
# else
    // 32-bit system, get chip_id from 0xbc2 (base_addrh)
    // and shift it to the right by 8 bits
    asm volatile ("csrr %0, 0xbc2" : "=r"(chip_id));
    chip_id = chip_id >> 8;
# endif
    return (uint8_t)chip_id;
}

inline uint8_t get_current_chip_loc_x() {
    uint8_t chip_id = get_current_chip_id();
    return chip_id >> 4;
}

inline uint8_t get_current_chip_loc_y() {
    uint8_t chip_id = get_current_chip_id();
    return chip_id & 0x0F;
}

inline uint8_t chip_loc_to_chip_id(uint8_t x, uint8_t y) {
    return (x << 4) | (y & 0x0F);
} 

inline uint8_t *get_current_chip_baseaddress() {
#if __riscv_xlen == 64
    // 64-bit system (CVA6), get chip_id from 0xf15
    uint32_t chip_id;
    asm volatile("csrr %0, 0xf15" : "=r"(chip_id));
    return (uint8_t *)((uintptr_t)chip_id << 40);
#else
    // 32-bit system, return 0 (not supported)
    return (uint8_t *)0;
#endif
}

inline uint8_t *get_chip_baseaddress(uint8_t chip_id) {
#if __riscv_xlen == 64
    // 64-bit system, perform the shift and return the base address
    return (uint8_t *)((uintptr_t)chip_id << 40);
#else
    // 32-bit system, return 0 (not supported)
    return (uint8_t *)0;
#endif
}

inline uint64_t get_chip_baseaddress_value(uint8_t chip_id) {
    return (((uint64_t)chip_id) << 40);
}

inline uint64_t get_current_chip_baseaddress_value() {
    uint32_t chip_id = get_current_chip_id();
    return get_chip_baseaddress_value(chip_id);
}


inline uint64_t chiplet_addr_transform(uint64_t addr){
    // Append the chiplet prefix to the current address
    // The addr is assumed to be a 40-bit address
    // The chiplet prefix is obtained from the current chip id
    return (uint64_t)get_current_chip_baseaddress() | addr;
}


inline uint64_t chiplet_addr_transform_full(uint8_t chip_id, uint64_t addr){
    // Append the chiplet prefix to the current address
    // The addr is assumed to be a 40-bit address
    // The chiplet prefix is obtained from the input chip id
    return (uint64_t)get_chip_baseaddress(chip_id) | addr;
}

inline uint64_t chiplet_addr_transform_loc(uint8_t chip_loc_x, uint8_t chip_loc_y, uint64_t addr){
    // Append the chiplet prefix to the current address
    // The addr is assumed to be a 40-bit address
    // The chiplet prefix is obtained from the input chip loc
    uint8_t chip_id = chip_loc_to_chip_id(chip_loc_x, chip_loc_y);
    return (uint64_t)get_chip_baseaddress(chip_id) | addr;
}
