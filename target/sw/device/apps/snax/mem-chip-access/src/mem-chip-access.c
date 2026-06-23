// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>

#include <stdint.h>

#include "chip_id.h"
#include "data.h"
#include "snrt.h"
#include "sys_dma.h"

#define MEM_CHIP_LOC_X 0x2
#define MEM_CHIP_LOC_Y 0x0

static int32_t chip_chunk_index(uint8_t chip_id) {
    switch (chip_id) {
        case 0x00:
            return 0;
        case 0x01:
            return 1;
        case 0x10:
            return 2;
        case 0x11:
            return 3;
        default:
            return -1;
    }
}

static int32_t system_idma_copy(uint8_t engine_chip_id, uint64_t dst,
                                uint64_t src, uint32_t size) {
    uint64_t transfer_id = sys_dma_memcpy(engine_chip_id, dst, src, size);
    if (transfer_id == 0) {
        printf_safe("Chip(%x, %x): system iDMA launch failed\r\n",
                    get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    }

    while (*(sys_dma_done_ptr(engine_chip_id)) != transfer_id) {
        asm volatile("nop");
    }
    asm volatile("fence" ::: "memory");
    return 0;
}

static int32_t check_result(uint32_t chunk_offset) {
    const uint8_t *golden = mem_chip_access_golden_l3 + chunk_offset;

    for (uint32_t i = 0; i < mem_chip_access_data_bytes; i++) {
        if (mem_chip_access_result_l3[i] != golden[i]) {
            printf_safe("Chip(%x, %x): mismatch at byte %d: expected %d, got %d\r\n",
                        get_current_chip_loc_x(), get_current_chip_loc_y(), i,
                        golden[i], mem_chip_access_result_l3[i]);
            return -1;
        }
    }

    return 0;
}

int main() {
    if (snrt_cluster_idx() != 0 || !snrt_is_dm_core()) {
        return 0;
    }

    uint8_t current_chip_id = get_current_chip_id();
    int32_t chunk_index = chip_chunk_index(current_chip_id);
    if (chunk_index < 0) {
        printf_safe("Chip(%x, %x): unsupported compute chip id 0x%x\r\n",
                    get_current_chip_loc_x(), get_current_chip_loc_y(),
                    current_chip_id);
        return -1;
    }

    if ((uint32_t)chunk_index >= mem_chip_access_num_chunks) {
        printf_safe("Chip(%x, %x): chunk index %d exceeds num_chunks %d\r\n",
                    get_current_chip_loc_x(), get_current_chip_loc_y(),
                    chunk_index, mem_chip_access_num_chunks);
        return -1;
    }

    uint32_t chunk_offset = (uint32_t)chunk_index * mem_chip_access_data_bytes;
    uint64_t mem_chip_src = chiplet_addr_transform_loc(
        MEM_CHIP_LOC_X, MEM_CHIP_LOC_Y,
        mem_chip_access_local_base + chunk_offset);
    uint64_t local_l3_dst = chiplet_addr_transform(
        (uint64_t)(uintptr_t)mem_chip_access_result_l3);

    printf_safe("Chip(%x, %x): system iDMA read memchip 0x%lx -> local L3 0x%lx (%d bytes)\r\n",
                get_current_chip_loc_x(), get_current_chip_loc_y(),
                mem_chip_src, local_l3_dst, mem_chip_access_data_bytes);

    if (system_idma_copy(current_chip_id, local_l3_dst, mem_chip_src,
                         mem_chip_access_data_bytes) != 0) {
        return -1;
    }

    if (check_result(chunk_offset) != 0) {
        return -1;
    }

    printf_safe("Chip(%x, %x): mem-chip-access check passed for chunk %d\r\n",
                get_current_chip_loc_x(), get_current_chip_loc_y(),
                chunk_index);
    return 0;
}
