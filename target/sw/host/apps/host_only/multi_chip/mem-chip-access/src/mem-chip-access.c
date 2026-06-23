// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>

#include "../data/data.h"
#include <stddef.h>

#include "chip_id.h"
#include "hemaia_d2d_link.h"
#include "libbingo/bingo_api.h"
#include "sys_dma.h"
#include "uart.c"

#define ARRAY_ELEM_COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define MIE_MSIE_OFFSET 3
#define MSTATUS_FS_OFFSET 13

#define MEMCHIP_LOC_X 0x2
#define MEMCHIP_LOC_Y 0x0

// Compute-chiplet rectangle used by chip_barrier (top-left .. bottom-right).
#define BARRIER_TL_CHIP_ID 0x00
#define BARRIER_BR_CHIP_ID 0x11

static const uint8_t TEST_CHIPS[4] = {0x00, 0x01, 0x10, 0x11};

void initialize_bss(void) {
    extern volatile uint64_t __bss_start;
    extern volatile uint64_t __bss_end;

    uint8_t *bss = (uint8_t *)&__bss_start;
    size_t bss_size = (size_t)(&__bss_end) - (size_t)(&__bss_start);
    for (size_t i = 0; i < bss_size; i++) {
        bss[i] = 0;
    }
    asm volatile("fence" ::: "memory");
}

void enable_fpu(void) {
    uint64_t mstatus;

    asm volatile("csrr %[mstatus], mstatus" : [mstatus] "=r"(mstatus));
    mstatus |= (1 << MSTATUS_FS_OFFSET);
    asm volatile("csrw mstatus, %[mstatus]" : : [mstatus] "r"(mstatus));
}

void *memset(void *dst, int value, size_t size) {
    uint8_t *bytes = (uint8_t *)dst;
    for (size_t i = 0; i < size; i++) {
        bytes[i] = (uint8_t)value;
    }
    return dst;
}

size_t strlen(const char *str) {
    size_t len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

static inline void enable_vec(void) {
    asm volatile("csrs mstatus, %[bits];"
                 :
                 : [bits] "r"(0x00000600 & (0x00000600 >> 1)));
}

static inline void enable_sw_interrupts(void) {
    uint64_t mie;

    asm volatile("csrr %[mie], mie" : [mie] "=r"(mie));
    mie |= (1 << MIE_MSIE_OFFSET);
    asm volatile("csrw mie, %[mie]" : : [mie] "r"(mie));
}

static int32_t chip_index(uint8_t chip_id) {
    for (uint32_t i = 0; i < ARRAY_ELEM_COUNT(TEST_CHIPS); i++) {
        if (TEST_CHIPS[i] == chip_id) {
            return (int32_t)i;
        }
    }
    return -1;
}

static void announce_chip_checkpoint(volatile comm_buffer_t *comm_buf,
                                     uint8_t checkpoint) {
    volatile uint8_t *this_chip_checkpoint =
        &comm_buf->chip_level_checkpoint[get_current_chip_id()];
    this_chip_checkpoint =
        (uint8_t *)(((uint64_t)this_chip_checkpoint) | (((uint64_t)0xFF) << 40));
    *this_chip_checkpoint = checkpoint;
}

static void wait_chips_checkpoint(volatile comm_buffer_t *comm_buf,
                                  uint8_t top_left_chip_id,
                                  uint8_t bottom_right_chip_id,
                                  uint8_t checkpoint) {
    volatile uint8_t *chip_level_checkpoint = &comm_buf->chip_level_checkpoint[0];
    uint8_t current_chip_id = get_current_chip_id();
    uint8_t continue_loop = 1;

    while (continue_loop) {
        continue_loop = 0;
        asm volatile("fence" ::: "memory");
        for (uint8_t i = top_left_chip_id >> 4; i <= (bottom_right_chip_id >> 4);
             i++) {
            for (uint8_t j = top_left_chip_id & 0xF;
                 j <= (bottom_right_chip_id & 0xF); j++) {
                if ((*(chip_level_checkpoint + ((i << 4) + j)) < checkpoint) &&
                    (current_chip_id != ((i << 4) + j))) {
                    continue_loop = 1;
                    break;
                }
            }
        }
    }
}

static void chip_barrier(volatile comm_buffer_t *comm_buf,
                         uint8_t top_left_chip_id,
                         uint8_t bottom_right_chip_id, uint8_t checkpoint) {
    announce_chip_checkpoint(comm_buf, checkpoint);
    wait_chips_checkpoint(comm_buf, top_left_chip_id, bottom_right_chip_id,
                          checkpoint);
}

static void next_barrier(comm_buffer_t *comm_buf, uint8_t *checkpoint) {
    chip_barrier(comm_buf, BARRIER_TL_CHIP_ID, BARRIER_BR_CHIP_ID,
                 (*checkpoint)++);
}

static int32_t run_idma_1d(const char *label, uint8_t engine_chip_id,
                           void *dst, const void *src, uint32_t size) {
    printf("Chip(%x, %x): %s IDMA engine %x %lx -> %lx (%d bytes)\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), label,
           engine_chip_id, (uintptr_t)src, (uintptr_t)dst, size);

    uint64_t transfer_id = sys_dma_memcpy(engine_chip_id,
                                          (uint64_t)(uintptr_t)dst,
                                          (uint64_t)(uintptr_t)src,
                                          size);
    if (transfer_id == 0) {
        printf("Chip(%x, %x): %s IDMA launch failed\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), label);
        return -1;
    }

    while (*(sys_dma_done_ptr(engine_chip_id)) != transfer_id) {
        asm volatile("nop");
    }
    asm volatile("fence" ::: "memory");

    printf("Chip(%x, %x): %s IDMA finished!\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), label);
    return 0;
}

static void clear_result(uint8_t current_chip_id) {
    sys_dma_blk_memcpy(current_chip_id,
                       (uintptr_t)mem_chip_access_result_l3,
                       (uintptr_t)WIDE_ZERO_MEM_BASE_ADDR,
                       mem_chip_access_data_bytes);
    asm volatile("fence" ::: "memory");
}

static int32_t check_result(uint32_t chunk_offset) {
    const uint8_t *golden = mem_chip_access_golden_l3 + chunk_offset;

    for (uint32_t i = 0; i < mem_chip_access_data_bytes; i++) {
        if (mem_chip_access_result_l3[i] != golden[i]) {
            printf("Chip(%x, %x): mismatch at byte %d: expected %d, got %d\r\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y(), i,
                   golden[i], mem_chip_access_result_l3[i]);
            return -1;
        }
    }

    printf("Chip(%x, %x): mem-chip-access check passed\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y());
    return 0;
}

int main(void) {
    uint8_t current_chip_id = get_current_chip_id();

    hemaia_d2d_link_initialize(current_chip_id);
    init_uart(get_current_chip_baseaddress(), 32, 1);
    enable_vec();
    enable_sw_interrupts();
    asm volatile("fence" ::: "memory");

    if (bingo_hemaia_system_mmap_init() < 0) {
        printf("Chip(%x, %x): [Host] Error when initializing Allocator\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    }
    printf("Chip(%x, %x): [Host] Allocator Init Success\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y());

    comm_buffer_t *comm_buf =
        (comm_buffer_t *)bingo_get_l2_comm_buffer(current_chip_id);

    int32_t chunk_index = chip_index(current_chip_id);
    if (chunk_index < 0) {
        printf("Chip(%x, %x): unsupported compute chip id 0x%x\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(),
               current_chip_id);
        return -1;
    }
    if ((uint32_t)chunk_index >= mem_chip_access_num_chunks) {
        printf("Chip(%x, %x): chunk index %d exceeds num_chunks %d\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(),
               chunk_index, mem_chip_access_num_chunks);
        return -1;
    }

    clear_result(current_chip_id);

    uint8_t checkpoint = 1;
    next_barrier(comm_buf, &checkpoint);

    uint32_t chunk_offset = (uint32_t)chunk_index * mem_chip_access_data_bytes;
    uint8_t *mem_chip_src = (uint8_t *)(uintptr_t)chiplet_addr_transform_loc(
        MEMCHIP_LOC_X, MEMCHIP_LOC_Y, mem_chip_access_local_base + chunk_offset);
    uint8_t *local_l3_dst = mem_chip_access_result_l3;

    printf("Chip(%x, %x): Starting mem-chip-access static read\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y());
    next_barrier(comm_buf, &checkpoint);
    if (run_idma_1d("mem-chip-access static read", current_chip_id,
                    local_l3_dst, mem_chip_src,
                    mem_chip_access_data_bytes) != 0) {
        return -1;
    }
    next_barrier(comm_buf, &checkpoint);

    int32_t ret = check_result(chunk_offset);
    next_barrier(comm_buf, &checkpoint);
    return ret;
}
