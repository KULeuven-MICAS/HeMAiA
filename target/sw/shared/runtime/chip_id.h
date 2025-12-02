// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

#pragma once
#include <stdint.h>

inline uint8_t get_current_chip_id() {
    
# if __riscv_xlen == 64
    uint64_t chip_id;
    asm volatile("csrr %0, 0xf16" : "=r"(chip_id));
# else
    // 32-bit system, get base_addrh from 0xbc2
    // and shift it to the right by 8 bits
    uint32_t base_addr_h;
    uint32_t chip_id;
    asm volatile ("csrr %0, 0xbc2" : "=r"(base_addr_h));
    chip_id = base_addr_h >> 8;
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

inline uint64_t get_current_chip_baseaddress() {
    
#if __riscv_xlen == 64
    uint64_t chip_id;
    asm volatile("csrr %0, 0xf16" : "=r"(chip_id));
    return (chip_id << 40);
#else
    // 32-bit system, get chip_id from 0xbc2 (base_addrh)
    uint32_t base_addr_h;
    asm volatile ("csrr %0, 0xbc2" : "=r"(base_addr_h));
    return ((uint64_t)base_addr_h << 32);
#endif
    
}

inline uint64_t get_chip_baseaddress(uint8_t chip_id) {
    return ((uint64_t)chip_id << 40);
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
    return get_current_chip_baseaddress() | addr;
}


inline uint64_t chiplet_addr_transform_full(uint8_t chip_id, uint64_t addr){
    // Append the chiplet prefix to the current address
    // The addr is assumed to be a 40-bit address
    // The chiplet prefix is obtained from the input chip id
    return get_chip_baseaddress(chip_id) | addr;
}

inline uint64_t chiplet_addr_transform_loc(uint8_t chip_loc_x, uint8_t chip_loc_y, uint64_t addr){
    // Append the chiplet prefix to the current address
    // The addr is assumed to be a 40-bit address
    // The chiplet prefix is obtained from the input chip loc
    uint8_t chip_id = chip_loc_to_chip_id(chip_loc_x, chip_loc_y);
    return get_chip_baseaddress(chip_id) | addr;
}
