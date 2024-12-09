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

inline uint32_t get_chip_baseaddress_h(uint8_t chip_id) {
    return (uint32_t)(chip_id << 8);
}

inline uint32_t get_current_chip_baseaddress_h() {
    uint32_t chip_id = get_current_chip_id();
    return (uint32_t)(chip_id << 8);
}