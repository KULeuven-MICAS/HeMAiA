// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include "libbingo/bingo_api.h"
#include "host.h"
#include "hemaia_clk_rst_controller.h"
#include "dummy_workload.h"
// Devices
#define MAX_TASKS  64
int kernel_execution(){
    // Since the main memory will be copied exactly to all the chiplets
    // The lower address of the dev_array will be the same
    // Only the upper 48-40 chiplet id prefix will be different

    // Set up the tasks list
    bingo_task_t **task_list = (bingo_task_t **)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(bingo_task_t) * MAX_TASKS);
    if (!task_list) {
        printf("Chip(%x, %x): [Host] Error: Cannot allocate task list.\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    }
    uint32_t num_tasks = 0;
    // printf("Chip(%x, %x): [Host] Task list @ 0x%lx\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), (uintptr_t)task_list);
    /////////////////////////
    // User defined workload
    /////////////////////////
    // Normally the user will define their own workload here
    num_tasks = __workload_multichip_dummy(task_list);

    ////////////////////////////
    // End user defined workload
    ////////////////////////////
    // Call bingo runtime
    // printf("Chip(%x, %x): [Host] %d tasks created.\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), num_tasks);
    bingo_runtime_schedule(
        task_list, 
        num_tasks
    );
    printf("Chip(%x, %x): [Host] %d tasks done.\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), num_tasks);
    // Close all the clusters
    bingo_close_all_clusters(task_list, num_tasks);
    return 0;
}