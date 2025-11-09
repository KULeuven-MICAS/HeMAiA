// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include "libbingo/bingo_api.h"
#include "host.h"
#include "dummy_workload.h"
// #include "xdma_1d_copy_workload.h"
// #include "versacore_workload.h"

// Kernel Execution
int kernel_execution(){

    // Set up the tasks list
    // We set a maximum number of tasks to 64
    // Can be changed if needed
    bingo_task_t *task_list[64] = {0};
    uint32_t num_tasks = 0;

    /////////////////////////
    // User defined workload
    /////////////////////////

    num_tasks = __workload_dummy(task_list);
    // uintptr_t output_data_ptr = 0;
    // __workload_versacore(task_list, &num_tasks, &output_data_ptr);
    // __workload_xdma_1d_copy(task_list, &num_tasks, &output_data_ptr);

    ////////////////////////////
    // End user defined workload
    ////////////////////////////

    // Call bingo runtime
    bingo_runtime_schedule(
        task_list, 
        num_tasks
    );
    printf("Chip(%x, %x): [Host] All tasks done.\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    // Close all the clusters
    bingo_close_all_clusters(task_list, num_tasks);
    // Free the output data

    return 0;
}

