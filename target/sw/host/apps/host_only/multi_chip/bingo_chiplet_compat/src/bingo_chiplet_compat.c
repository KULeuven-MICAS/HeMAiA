// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "host.h"
#include "libbingo/bingo_api.h"

// The FPGA is at (0,0). Its east link connects to chip_brugge at (1,0).
#define FPGA_CHIP_ID 0x00
#define BRUGGE_CHIP_ID 0x10
#define BRUGGE_HOST_CORE_ID 2
#define BRUGGE_SNITCH_SOURCE_ID 1
#define BRUGGE_DEP_QUEUE 0x0a000000
#define FPGA_HOST_DEP_TASK_ID 11
#define FPGA_SNITCH_DEP_TASK_ID 12
#define BRUGGE_FIRST_TASK_ID 17
#define BRUGGE_FOLLOW_UP_TASK_ID 18
#define BRUGGE_TO_FPGA_SNITCH_TAG 5
#define BRUGGE_TO_FPGA_HOST_TAG 6
#define FPGA_TO_BRUGGE_TAG 9
#define FPGA_ARMED UINT64_C(0x42494e474f)

// chip_brugge 2C descriptor with source chip 0x10/core 1 and a dependency
// set for chip 0x00/cluster 0/core 2, tag 5.
#define BRUGGE_SNITCH_COMPLETION UINT64_C(0x0160010042000e00)

static uint64_t task_desc[2];

static void configure_link(bool is_fpga) {
    D2DDirection peer = is_fpga ? D2D_DIRECTION_EAST : D2D_DIRECTION_WEST;
    unsigned physical_clusters = is_fpga ? 0 : 2;
    enable_clk_domain(0, 7);
    enable_clk_domain(physical_clusters + 1 + peer, 1);
    set_all_d2d_link_tx_turnaround_silence_period(0);
    set_all_d2d_link_availability(false);
    set_d2d_link_availability(peer, true);
}

static int execute_task(uint32_t expected_task_id, uint8_t chip) {
    uint32_t ready_task_id = bingo_hw_scheduler_get_global_task_id();
    if (ready_task_id != expected_task_id) {
        printf("Bingo chiplet compatibility: expected task %u, got %u on chip %x\r\n",
               expected_task_id, ready_task_id, chip);
        return 1;
    }
    bingo_hw_scheduler_write_done_queue(ready_task_id);
    return 0;
}

int main(void) {
    uint8_t chip = get_current_chip_id();
    bool is_fpga = chip == FPGA_CHIP_ID;
    init_uart(get_current_chip_baseaddress(), 32, 1);
    if (!is_fpga && chip != BRUGGE_CHIP_ID) {
        printf("Bingo chiplet compatibility: unexpected chip ID %x\r\n", chip);
        return 1;
    }
    configure_link(is_fpga);

    const bingo_hw_manager_task_desc_t old_completion = {
        .task_id = 7,
        .assigned_chiplet_id = BRUGGE_CHIP_ID,
        .assigned_core_id = BRUGGE_SNITCH_SOURCE_ID,
        .dep_set_enabled = true,
        .dep_set_chiplet_id = FPGA_CHIP_ID,
        .dep_set_code = 1U << BRUGGE_HOST_CORE_ID,
        .dep_set_tag = BRUGGE_TO_FPGA_SNITCH_TAG,
    };
    if (encode_bingo_hw_manager_task_desc(old_completion) != BRUGGE_SNITCH_COMPLETION) {
        printf("Bingo chiplet compatibility: chip_brugge descriptor mismatch\r\n");
        return 1;
    }

    const bingo_hw_manager_task_desc_t first_task = {
        .task_id = is_fpga ? FPGA_HOST_DEP_TASK_ID : BRUGGE_FIRST_TASK_ID,
        .assigned_chiplet_id = chip,
        .assigned_core_id = BRUGGE_HOST_CORE_ID,
        .dep_check_enabled = is_fpga,
        .dep_check_code = 1U << BRUGGE_HOST_CORE_ID,
        .dep_check_tag = BRUGGE_TO_FPGA_HOST_TAG,
        .dep_set_enabled = !is_fpga,
        .dep_set_chiplet_id = FPGA_CHIP_ID,
        .dep_set_code = 1U << BRUGGE_HOST_CORE_ID,
        .dep_set_tag = BRUGGE_TO_FPGA_HOST_TAG,
    };
    const bingo_hw_manager_task_desc_t second_task = {
        .task_id = is_fpga ? FPGA_SNITCH_DEP_TASK_ID : BRUGGE_FOLLOW_UP_TASK_ID,
        .assigned_chiplet_id = chip,
        .assigned_core_id = BRUGGE_HOST_CORE_ID,
        .dep_check_enabled = true,
        .dep_check_code = 1U << (is_fpga ? BRUGGE_SNITCH_SOURCE_ID : BRUGGE_HOST_CORE_ID),
        .dep_check_tag = is_fpga ? BRUGGE_TO_FPGA_SNITCH_TAG : FPGA_TO_BRUGGE_TAG,
        .dep_set_enabled = is_fpga,
        .dep_set_chiplet_id = BRUGGE_CHIP_ID,
        .dep_set_code = 1U << BRUGGE_HOST_CORE_ID,
        .dep_set_tag = FPGA_TO_BRUGGE_TAG,
    };
    task_desc[0] = encode_bingo_hw_manager_task_desc(first_task);
    task_desc[1] = encode_bingo_hw_manager_task_desc(second_task);
    uint64_t local_desc_addr = chiplet_addr_transform((uintptr_t)task_desc);
    if (is_fpga) {
        bingo_hw_scheduler_init(0, 0, 0, 0, 0, local_desc_addr, 2);
        printf("Bingo chiplet compatibility: FPGA waiting for Brugge dependencies\r\n");
        bingo_write_h2h_mailbox(BRUGGE_CHIP_ID, FPGA_ARMED, 0, NULL);
    } else {
        bingo_hw_scheduler_init_host(local_desc_addr, 2);
        uint64_t ready = 0;
        const uint64_t timeout_cycles = 1000000;
        if (bingo_read_h2h_mailbox(&ready, timeout_cycles, NULL) != BINGO_MB_OK ||
            ready != FPGA_ARMED) {
            printf("Bingo chiplet compatibility: FPGA readiness timed out\r\n");
            return 1;
        }

    }

    if (execute_task(first_task.task_id, chip)) {
        return 1;
    }
    if (is_fpga) {
        printf("Bingo chiplet compatibility: Brugge manager released FPGA CVA6 task\r\n");
    } else {
        printf("Bingo chiplet compatibility: Brugge sending legacy Snitch dependency\r\n");
        writed(BRUGGE_SNITCH_COMPLETION,
               chiplet_addr_transform_full(FPGA_CHIP_ID, BRUGGE_DEP_QUEUE));
    }

    if (execute_task(second_task.task_id, chip)) {
        return 1;
    }
    if (is_fpga) {
        printf("Bingo chiplet compatibility: FPGA CVA6 task completed after Snitch dependency\r\n");
    } else {
        printf("Bingo chiplet compatibility: Brugge follow-up released\r\n");
    }
    return 0;
}
