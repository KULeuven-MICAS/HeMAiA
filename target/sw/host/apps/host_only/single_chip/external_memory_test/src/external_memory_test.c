// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "host.h"

#if !defined(EXTERNAL_MEMORY_BASE_ADDR) || !defined(EXTERNAL_MEMORY_SIZE)
#error "external_memory_test requires external_memory in the SoC configuration"
#endif

#define DMA_WORDS 32
static uint64_t dma_source[DMA_WORDS] __attribute__((aligned(64)));
static volatile uint64_t dma_destination[DMA_WORDS] __attribute__((aligned(64)));

static int dma_copy(uint8_t chip_id, uint64_t dst, uint64_t src) {
    asm volatile("fence" ::: "memory");
    uint64_t transfer_id = sys_dma_memcpy(chip_id, dst, src, sizeof(dma_source));
    for (unsigned wait = 0; wait < 100000; ++wait) {
        if (*sys_dma_done_ptr(chip_id) == transfer_id) {
            asm volatile("fence" ::: "memory");
            return 0;
        }
    }
    return 1;
}

int main(void) {
    init_uart(get_current_chip_baseaddress(), 32, 1);
    uint64_t base = chiplet_addr_transform(EXTERNAL_MEMORY_BASE_ADDR);
    volatile uint64_t *first = (volatile uint64_t *)(uintptr_t)base;
    volatile uint64_t *last =
        (volatile uint64_t *)(uintptr_t)(base + EXTERNAL_MEMORY_SIZE - sizeof(uint64_t));

    if (*first != 0 || *last != 0) {
        printf("External memory: unwritten locations did not return zero\r\n");
        return 1;
    }

    *first = UINT64_C(0x0123456789abcdef);
    *last = UINT64_C(0xfedcba9876543210);
    ((volatile uint8_t *)first)[3] = 0x3c;
    asm volatile("fence" ::: "memory");
    if (*first != UINT64_C(0x012345673cabcdef) ||
        *last != UINT64_C(0xfedcba9876543210)) {
        printf("External memory: window boundary or byte strobe check failed\r\n");
        return 1;
    }

    for (unsigned i = 0; i < DMA_WORDS; ++i) {
        dma_source[i] = UINT64_C(0x1f2e3d4c5b6a7988) * (i + 1);
    }
    uint8_t chip_id = get_current_chip_id();
    // Four full 512-bit beats ending at the final byte of the mapped window.
    uint64_t burst_address = base + EXTERNAL_MEMORY_SIZE - sizeof(dma_source);
    if (dma_copy(chip_id, burst_address, chiplet_addr_transform((uintptr_t)dma_source)) ||
        dma_copy(chip_id, chiplet_addr_transform((uintptr_t)dma_destination), burst_address)) {
        printf("External memory: system iDMA timed out\r\n");
        return 1;
    }

    const volatile uint64_t *external_data = (const volatile uint64_t *)(uintptr_t)burst_address;
    for (unsigned i = 0; i < DMA_WORDS; ++i) {
        if (external_data[i] != dma_source[i] || dma_destination[i] != dma_source[i]) {
            printf("External memory: iDMA round trip failed at word %u\r\n", i);
            return 1;
        }
    }
    if (*first != UINT64_C(0x012345673cabcdef)) {
        printf("External memory: the start and end of the window alias\r\n");
        return 1;
    }
    printf("External memory: boundary, byte strobes, and 512-bit iDMA round trip passed\r\n");
    return 0;
}
