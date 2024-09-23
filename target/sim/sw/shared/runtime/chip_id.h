# pragma once
# include <stdint.h>

uint8_t get_current_chip_id() {
    uint32_t chip_id;
    asm volatile("csrr %0, 0xf15" : "=r"(chip_id));
    return (uint8_t)chip_id;
}

uint8_t *get_current_chip_baseaddress() {
    uint32_t chip_id;
    asm volatile("csrr %0, 0xf15" : "=r"(chip_id));
    return (uint8_t *)((uintptr_t)chip_id << 40);
}

uint8_t *get_chip_baseaddress(uint8_t chip_id) {
    return (uint8_t *)((uintptr_t)chip_id << 40);
}
