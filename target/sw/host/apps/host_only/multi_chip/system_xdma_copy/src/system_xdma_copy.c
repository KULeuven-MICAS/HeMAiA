// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>
//
// Multi-chip XDMA copy stress test.
//
// The same binary runs on every compute chiplet. Each chip owns one read-only
// L3 source buffer and one L3 destination buffer. Every test phase starts all
// chiplets together, waits for all issued XDMA tasks to finish, verifies the
// expected destination buffers, then clears the local destination buffer before
// the next phase.
//
// Test matrix:
//   Topology: 00 top-left, 01 bottom-left, 10 top-right, 11 bottom-right.
//   Read notation "dst<-src": dst chiplet issues the copy into its local dst.
//   Write notation "src->dst": src chiplet issues the copy into dst's dst.
//
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

// Compute-chiplet rectangle used by chip_barrier (top-left .. bottom-right).
#define BARRIER_TL_CHIP_ID 0x00
#define BARRIER_BR_CHIP_ID 0x11

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

static void copy_data_to_buf(uint8_t* dst, const uint8_t* src, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        dst[i] = src[i];
    }
    asm volatile("fence" ::: "memory");
}

static void clear_buf(uint8_t* buf, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        buf[i] = 0;
    }
    asm volatile("fence" ::: "memory");
}

static int32_t run_xdma_1d(const char* label, const void* src, void* dst,
                           uint32_t size) {
    printf("Chip(%x, %x): %s XDMA %lx -> %lx (%d bytes)\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), label,
           (uintptr_t)src, (uintptr_t)dst, size);

    int32_t cfg_ret = hemaia_xdma_memcpy_1d(src, dst, size);
    if (cfg_ret != 0) {
        printf("Chip(%x, %x): %s XDMA config failed: %d\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), label,
               cfg_ret);
        return cfg_ret;
    }

    uint32_t task_id = hemaia_xdma_start();
    hemaia_xdma_remote_wait(task_id);
    asm volatile("fence" ::: "memory");

    printf("Chip(%x, %x): %s XDMA finished!\r\n",
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

static void next_barrier(comm_buffer_t* comm_buf, uint8_t* checkpoint) {
    chip_barrier(comm_buf, BARRIER_TL_CHIP_ID, BARRIER_BR_CHIP_ID,
                 (*checkpoint)++);
}

static int32_t run_read_phase(const char* label, uint8_t current_chip_id,
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
        int32_t ret = run_xdma_1d(label, remote_src, local_l3_dst_buf,
                                  data_size);
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

    clear_buf(local_l3_dst_buf, data_size);
    next_barrier(comm_buf, checkpoint);
    return 0;
}

static int32_t run_write_phase(const char* label, uint8_t current_chip_id,
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
        int32_t ret = run_xdma_1d(label, local_l3_src_buf, remote_dst,
                                  data_size);
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

    clear_buf(local_l3_dst_buf, data_size);
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

    copy_data_to_buf(local_l3_src_buf, data, data_size);
    clear_buf(local_l3_dst_buf, data_size);

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

    uint8_t checkpoint = 1;
    next_barrier(comm_buf, &checkpoint);

    int32_t xdma_ret = run_read_phase(
        "clockwise read", current_chip_id, comm_buf, local_l3_src_buf,
        local_l3_dst_buf, clockwise_read_src, check_all, &checkpoint);
    if (xdma_ret != 0) {
        return -1;
    }

    xdma_ret = run_read_phase(
        "anticlockwise read", current_chip_id, comm_buf, local_l3_src_buf,
        local_l3_dst_buf, anticlockwise_read_src, check_all, &checkpoint);
    if (xdma_ret != 0) {
        return -1;
    }

    xdma_ret = run_write_phase(
        "clockwise write", current_chip_id, comm_buf, local_l3_src_buf,
        local_l3_dst_buf, clockwise_write_dst, check_all, &checkpoint);
    if (xdma_ret != 0) {
        return -1;
    }

    xdma_ret = run_write_phase(
        "anticlockwise write", current_chip_id, comm_buf, local_l3_src_buf,
        local_l3_dst_buf, anticlockwise_write_dst, check_all, &checkpoint);
    if (xdma_ret != 0) {
        return -1;
    }

    xdma_ret = run_read_phase(
        "3-to-1 read", current_chip_id, comm_buf, local_l3_src_buf,
        local_l3_dst_buf, three_to_one_src, check_three_to_one_read,
        &checkpoint);
    if (xdma_ret != 0) {
        return -1;
    }

    xdma_ret = run_write_phase(
        "3-to-1 write", current_chip_id, comm_buf, local_l3_src_buf,
        local_l3_dst_buf, three_to_one_dst, check_three_to_one_write,
        &checkpoint);
    if (xdma_ret != 0) {
        return -1;
    }

    xdma_ret = run_read_phase(
        "X-style read", current_chip_id, comm_buf, local_l3_src_buf,
        local_l3_dst_buf, x_style_peer, check_all, &checkpoint);
    if (xdma_ret != 0) {
        return -1;
    }

    xdma_ret = run_write_phase(
        "X-style write", current_chip_id, comm_buf, local_l3_src_buf,
        local_l3_dst_buf, x_style_peer, check_all, &checkpoint);
    if (xdma_ret != 0) {
        return -1;
    }

    printf("Chip(%x, %x): All multi-chip XDMA tests passed!\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y());

    bingo_l3_free(current_chip_id, (uint64_t)local_l3_dst_buf);
    bingo_l3_free(current_chip_id, (uint64_t)local_l3_src_buf);
    return 0;
}
