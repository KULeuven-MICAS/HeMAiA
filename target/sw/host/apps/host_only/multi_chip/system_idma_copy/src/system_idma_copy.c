// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>
//
// Multi-chip IDMA copy test.
//
// The same binary runs on every compute chiplet. Chip 00 initializes a shared
// memchip buffer at chip(2,0) with IDMA. Then every compute chip IDMA-unicasts
// from that memchip buffer into its local L3 destination buffer. Chip 00 also
// asks the memchip IDMA engine to broadcast the memchip buffer to every chip's
// local L3 destination buffer. After those memchip tests, each compute chiplet
// uses its own IDMA engine for the chip-to-chip concurrency phases.
//
// Test matrix:
//   Topology: 00 top-left, 01 bottom-left, 10 top-right, 11 bottom-right,
//             20 memory chip at the right side of 10.
//   Read notation "dst<-src": dst chiplet issues the IDMA copy into its local
//                             destination buffer.
//   Write notation "src->dst": src chiplet issues the IDMA copy into dst's
//                              destination buffer.
//
//   0. Memchip seed:        00 IDMA-writes data to chip 20 heap buffer
//   0. Memchip unicast:     00/01/10/11 read chip 20 heap buffer
//   0. Memchip broadcast:   chip 20 IDMA-writes to 00/01/10/11
//   1. Clockwise read:      00<-10, 01<-00, 11<-01, 10<-11
//   2. Anticlockwise read:  00<-01, 10<-00, 11<-10, 01<-11
//   3. Clockwise write:     00->01, 01->11, 11->10, 10->00
//   4. Anticlockwise write: 00->10, 10->11, 11->01, 01->00
//   5. 3-to-1 read:         01<-00, 10<-00, 11<-00
//   6. 3-to-1 write:        01->00, 10->00, 11->00
//   7. X-style read:        00<-11, 11<-00, 10<-01, 01<-10
//   8. X-style write:       00->11, 11->00, 10->01, 01->10

#include "../data/data.h"
#include "host.h"
#include "libbingo/bingo_api.h"
#include "hemaia_clk_rst_controller.h"

// chip(2,0) is the "memchip": 64 MiB of external SRAM addressed via D2D.
// Its lower 32 MiB is reserved for static data, and the upper 32 MiB is the
// heap region used by bingo_mempool_* / this test.
#define MEMCHIP_LOC_X   0x2
#define MEMCHIP_LOC_Y   0x0
#define MEMCHIP_CHIP_ID 0x20

// Compute-chiplet rectangle used by chip_barrier (top-left .. bottom-right).
#define BARRIER_TL_CHIP_ID 0x00
#define BARRIER_BR_CHIP_ID 0x11

// Broadcast prefix for chiplet_addr_transform_loc().
#define BROADCAST_LOC_X 0xF
#define BROADCAST_LOC_Y 0xF

#define CHECKED_DATA_SIZE 64
#define CHIP_NONE         0xFF
#define CHIP_LOCAL_MASK   ((1ULL << 40) - 1)

static const uint8_t TEST_CHIPS[4] = {0x00, 0x01, 0x10, 0x11};

static int32_t chip_index(uint8_t chip_id) {
    for (uint32_t i = 0; i < ARRAY_ELEM_COUNT(TEST_CHIPS); i++) {
        if (TEST_CHIPS[i] == chip_id) {
            return (int32_t)i;
        }
    }
    return -1;
}

static uint8_t* addr_on_chip(uint8_t chip_id, const void* local_addr) {
    uint64_t local_offset = (uint64_t)((uintptr_t)local_addr & CHIP_LOCAL_MASK);
    return (uint8_t*)(uintptr_t)chiplet_addr_transform_full(chip_id,
                                                            local_offset);
}

static uint8_t* broadcast_addr(const void* local_addr) {
    uint64_t local_offset = (uint64_t)((uintptr_t)local_addr & CHIP_LOCAL_MASK);
    return (uint8_t*)(uintptr_t)chiplet_addr_transform_loc(
        BROADCAST_LOC_X, BROADCAST_LOC_Y, local_offset);
}

static void next_barrier(comm_buffer_t* comm_buf, uint8_t* checkpoint) {
    chip_barrier(comm_buf, BARRIER_TL_CHIP_ID, BARRIER_BR_CHIP_ID,
                 (*checkpoint)++);
}

static int32_t run_idma_1d(const char* label, uint8_t engine_chip_id,
                           void* dst, const void* src, uint32_t size) {
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

static int32_t check_dst(const char* label, const uint8_t* local_l3_dst_buf) {
    printf("Chip(%x, %x): Checking %s result...\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), label);

    for (uint32_t i = 0; i < CHECKED_DATA_SIZE; i++) {
        if (data[i] != local_l3_dst_buf[i]) {
            printf("Chip(%x, %x): %s mismatch at index %d: expected %d, got %d\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y(), label,
                   i, data[i], local_l3_dst_buf[i]);
            return -1;
        }
    }

    printf("Chip(%x, %x): %s check passed!\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), label);
    return 0;
}

static void clear_dst(uint8_t current_chip_id, uint8_t* local_l3_dst_buf,
                      uint32_t size) {
    sys_dma_blk_memcpy(current_chip_id,
                       (uintptr_t)local_l3_dst_buf,
                       (uintptr_t)WIDE_ZERO_MEM_BASE_ADDR,
                       size);
    asm volatile("fence" ::: "memory");
}

static int32_t run_idma_read_phase(const char* label, uint8_t current_chip_id,
                                   comm_buffer_t* comm_buf,
                                   uint8_t* local_l3_src_buf,
                                   uint8_t* local_l3_dst_buf,
                                   const uint8_t src_chip_by_index[4],
                                   const uint8_t check_by_index[4],
                                   uint8_t* checkpoint) {
    int32_t idx = chip_index(current_chip_id);
    if (idx < 0) {
        printf("Chip(%x, %x): %s unknown chip id %x\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), label,
               current_chip_id);
        return -1;
    }

    printf("Chip(%x, %x): Starting %s\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), label);
    next_barrier(comm_buf, checkpoint);

    uint8_t src_chip = src_chip_by_index[idx];
    if (src_chip != CHIP_NONE) {
        uint8_t* remote_src = addr_on_chip(src_chip, local_l3_src_buf);
        int32_t ret = run_idma_1d(label, current_chip_id, local_l3_dst_buf,
                                  remote_src, data_size);
        if (ret != 0) {
            return ret;
        }
    }

    next_barrier(comm_buf, checkpoint);

    if (check_by_index[idx]) {
        int32_t ret = check_dst(label, local_l3_dst_buf);
        if (ret != 0) {
            return ret;
        }
    }

    clear_dst(current_chip_id, local_l3_dst_buf, data_size);
    next_barrier(comm_buf, checkpoint);
    return 0;
}

static int32_t run_idma_write_phase(const char* label, uint8_t current_chip_id,
                                    comm_buffer_t* comm_buf,
                                    uint8_t* local_l3_src_buf,
                                    uint8_t* local_l3_dst_buf,
                                    const uint8_t dst_chip_by_index[4],
                                    const uint8_t check_by_index[4],
                                    uint8_t* checkpoint) {
    int32_t idx = chip_index(current_chip_id);
    if (idx < 0) {
        printf("Chip(%x, %x): %s unknown chip id %x\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), label,
               current_chip_id);
        return -1;
    }

    printf("Chip(%x, %x): Starting %s\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), label);
    next_barrier(comm_buf, checkpoint);

    uint8_t dst_chip = dst_chip_by_index[idx];
    if (dst_chip != CHIP_NONE) {
        uint8_t* remote_dst = addr_on_chip(dst_chip, local_l3_dst_buf);
        int32_t ret = run_idma_1d(label, current_chip_id, remote_dst,
                                  local_l3_src_buf, data_size);
        if (ret != 0) {
            return ret;
        }
    }

    next_barrier(comm_buf, checkpoint);

    if (check_by_index[idx]) {
        int32_t ret = check_dst(label, local_l3_dst_buf);
        if (ret != 0) {
            return ret;
        }
    }

    clear_dst(current_chip_id, local_l3_dst_buf, data_size);
    next_barrier(comm_buf, checkpoint);
    return 0;
}

int main() {
    uint8_t current_chip_id = get_current_chip_id();

    hemaia_d2d_link_initialize_4c1m(current_chip_id);
    init_uart(get_current_chip_baseaddress(), 32, 1);
    enable_sw_interrupts();

    if (bingo_hemaia_system_mmap_init() < 0) {
        printf("Chip(%x, %x): [Host] Error when initializing Allocator\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    }
    printf("Chip(%x, %x): [Host] Allocator Init Success\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y());

    comm_buffer_t* comm_buf =
        (comm_buffer_t*)bingo_get_l2_comm_buffer(current_chip_id);

    if (current_chip_id == 0) {
        if (bingo_mempool_init(MEMCHIP_LOC_X, MEMCHIP_LOC_Y) < 0) {
            printf("Chip(%x, %x): Memchip heap init failed!\r\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y());
            return -1;
        }
        printf("Chip(%x, %x): Memchip heap init success!\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
    }

    uint8_t* local_l3_src_buf =
        (uint8_t*)bingo_l3_alloc(current_chip_id, data_size);
    uint8_t* local_l3_dst_buf =
        (uint8_t*)bingo_l3_alloc(current_chip_id, data_size);
    printf("Chip(%x, %x): local_l3_src_buf allocated at %lx\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(),
           (uintptr_t)local_l3_src_buf);
    printf("Chip(%x, %x): local_l3_dst_buf allocated at %lx\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(),
           (uintptr_t)local_l3_dst_buf);
    if (local_l3_src_buf == 0 || local_l3_dst_buf == 0) {
        printf("Chip(%x, %x): L3 buffer allocation failed!\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    }

    if (run_idma_1d("local source seed", current_chip_id, local_l3_src_buf,
                    data, data_size) != 0) {
        return -1;
    }
    clear_dst(current_chip_id, local_l3_dst_buf, data_size);

    uint8_t* volatile memchip_buf = 0;
    if (current_chip_id == 0) {
        memchip_buf = (uint8_t*)bingo_mempool_alloc(MEMCHIP_LOC_X,
                                                    MEMCHIP_LOC_Y,
                                                    data_size);
        if (memchip_buf == 0) {
            printf("Chip(%x, %x): memchip_buf allocation failed!\r\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y());
            return -1;
        }
        printf("Chip(%x, %x): memchip_buf allocated at %lx\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(),
               (uintptr_t)memchip_buf);

        *(volatile uint8_t**)(chiplet_addr_transform_loc(
            BROADCAST_LOC_X, BROADCAST_LOC_Y,
            (uintptr_t)&memchip_buf)) = memchip_buf;

        if (run_idma_1d("memchip seed", current_chip_id, memchip_buf,
                        data, data_size) != 0) {
            return -1;
        }
    }

    uint8_t checkpoint = 1;
    next_barrier(comm_buf, &checkpoint);

    printf("Chip(%x, %x): Starting memchip unicast\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y());
    next_barrier(comm_buf, &checkpoint);
    if (run_idma_1d("memchip unicast", current_chip_id, local_l3_dst_buf,
                    memchip_buf, data_size) != 0) {
        return -1;
    }
    next_barrier(comm_buf, &checkpoint);
    if (check_dst("memchip unicast", local_l3_dst_buf) != 0) {
        return -1;
    }
    clear_dst(current_chip_id, local_l3_dst_buf, data_size);
    next_barrier(comm_buf, &checkpoint);

    printf("Chip(%x, %x): Starting memchip broadcast\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y());
    next_barrier(comm_buf, &checkpoint);
    if (current_chip_id == 0) {
        if (run_idma_1d("memchip broadcast", MEMCHIP_CHIP_ID,
                        broadcast_addr(local_l3_dst_buf), memchip_buf,
                        data_size) != 0) {
            return -1;
        }
    }
    next_barrier(comm_buf, &checkpoint);
    if (check_dst("memchip broadcast", local_l3_dst_buf) != 0) {
        return -1;
    }
    clear_dst(current_chip_id, local_l3_dst_buf, data_size);
    next_barrier(comm_buf, &checkpoint);

    const uint8_t check_all[4] = {1, 1, 1, 1};
    const uint8_t check_three_to_one_read[4] = {0, 1, 1, 1};
    const uint8_t check_three_to_one_write[4] = {1, 0, 0, 0};

    const uint8_t clockwise_read_src[4] = {0x10, 0x00, 0x11, 0x01};
    const uint8_t anticlockwise_read_src[4] = {0x01, 0x11, 0x00, 0x10};
    const uint8_t clockwise_write_dst[4] = {0x01, 0x11, 0x00, 0x10};
    const uint8_t anticlockwise_write_dst[4] = {0x10, 0x00, 0x11, 0x01};
    const uint8_t three_to_one_src[4] = {CHIP_NONE, 0x00, 0x00, 0x00};
    const uint8_t three_to_one_dst[4] = {CHIP_NONE, 0x00, 0x00, 0x00};
    const uint8_t x_style_peer[4] = {0x11, 0x10, 0x01, 0x00};

    int32_t idma_ret = run_idma_read_phase(
        "clockwise read", current_chip_id, comm_buf, local_l3_src_buf,
        local_l3_dst_buf, clockwise_read_src, check_all, &checkpoint);
    if (idma_ret != 0) {
        return -1;
    }

    idma_ret = run_idma_read_phase(
        "anticlockwise read", current_chip_id, comm_buf, local_l3_src_buf,
        local_l3_dst_buf, anticlockwise_read_src, check_all, &checkpoint);
    if (idma_ret != 0) {
        return -1;
    }

    idma_ret = run_idma_write_phase(
        "clockwise write", current_chip_id, comm_buf, local_l3_src_buf,
        local_l3_dst_buf, clockwise_write_dst, check_all, &checkpoint);
    if (idma_ret != 0) {
        return -1;
    }

    idma_ret = run_idma_write_phase(
        "anticlockwise write", current_chip_id, comm_buf, local_l3_src_buf,
        local_l3_dst_buf, anticlockwise_write_dst, check_all, &checkpoint);
    if (idma_ret != 0) {
        return -1;
    }

    idma_ret = run_idma_read_phase(
        "3-to-1 read", current_chip_id, comm_buf, local_l3_src_buf,
        local_l3_dst_buf, three_to_one_src, check_three_to_one_read,
        &checkpoint);
    if (idma_ret != 0) {
        return -1;
    }

    idma_ret = run_idma_write_phase(
        "3-to-1 write", current_chip_id, comm_buf, local_l3_src_buf,
        local_l3_dst_buf, three_to_one_dst, check_three_to_one_write,
        &checkpoint);
    if (idma_ret != 0) {
        return -1;
    }

    idma_ret = run_idma_read_phase(
        "X-style read", current_chip_id, comm_buf, local_l3_src_buf,
        local_l3_dst_buf, x_style_peer, check_all, &checkpoint);
    if (idma_ret != 0) {
        return -1;
    }

    idma_ret = run_idma_write_phase(
        "X-style write", current_chip_id, comm_buf, local_l3_src_buf,
        local_l3_dst_buf, x_style_peer, check_all, &checkpoint);
    if (idma_ret != 0) {
        return -1;
    }

    printf("Chip(%x, %x): All multi-chip IDMA tests passed!\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y());

    if (current_chip_id == 0) {
        bingo_mempool_free(MEMCHIP_LOC_X, MEMCHIP_LOC_Y, (uint64_t)memchip_buf);
    }
    bingo_l3_free(current_chip_id, (uint64_t)local_l3_dst_buf);
    bingo_l3_free(current_chip_id, (uint64_t)local_l3_src_buf);
    return 0;
}
