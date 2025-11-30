// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include "host.h"
#include "libbingo/bingo_api.h"
#include "gemm_intra_chiplet_workload.h"

// Kernel Execution
int kernel_execution() {
    // Set up the tasks list
    // We set a maximum number of tasks to 64
    // Can be changed if needed
    #define MAX_TASKS 64
    bingo_task_t **task_list = (bingo_task_t **)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(bingo_task_t) * MAX_TASKS);
    if (!task_list) {
        printf("Chip(%x, %x): [Host] Error: Cannot allocate task list.\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    }
    uint32_t num_tasks = 0;

    /////////////////////////
    // User defined workload
    /////////////////////////
    num_tasks = __workload_versacore_intra_chiplet(task_list);

    ////////////////////////////
    // End user defined workload
    ////////////////////////////

    // Call bingo runtime
    bingo_runtime_schedule(task_list, num_tasks);
    printf("Chip(%x, %x): [Host] All tasks done.\n", get_current_chip_loc_x(),
           get_current_chip_loc_y());
    // Close all the clusters
    bingo_close_all_clusters(task_list, num_tasks);
    // Free the output data

    return 0;
}
