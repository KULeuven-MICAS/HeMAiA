// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include "dummy_data.h"
#include "libbingo/bingo_api.h"
#include "host.h"
// Since the main memory will be copied exactly to all the chiplets
// The lower address of the dev_array will be the same
// Only the upper 48-40 chiplet id prefix will be different
uint32_t __workload_multichip_dummy(bingo_task_t **task_list) {
    // ---------------------------------------------------------------------
    // Multichip Dummy Workload
    // ---------------------------------------------------------------------
    // Constructs the following distributed DAG across 4 chiplets.
    // Notation: Tx((chip_x,chip_y),cluster)
    //
    //          T0((0,0),0)                                              |
    //            | \                                                    |
    //            |  \                                                   |
    //        T1((0,0),1) T2((0,0),2)                                    | Running on chiplet (0,0)
    //            \   /                                                  |
    //             T3((0,0),3)                                           |
    //             /    \
    //            /      \
    //       T4((1,0),0) T8((0,1),0)  ------------                       |
    //         |            |    \               \                       |
    //         v            |     \               \                      |
    //       T5((1,0),1)    |      \               \                     | Running in parallel on chiplet (1,0) and (0,1)
    //         |            |       \               \                    |
    //         v            |        \               \                   |
    //       T6((1,0),2)    |         \               \                  |
    //         |            v          v               v                 |
    //         v          T9((0,1),1)  T10((0,1),2)  T11((0,1),3)        |
    //       T7((1,0),3)      \          |              /
    //          \              \         |             /
    //           \              \        |            /
    //            \              \       |           / 
    //             \              \      |          /
    //              \              \     |         /
    //                -------------- T12((1,1),0)                        | Running on chiplet (1,1)
    //
    // Dependency list (successor <- predecessors):
    //   T1 <- T0
    //   T2 <- T0
    //   T3 <- T1, T2
    //   T4 <- T3
    //   T5 <- T4
    //   T6 <- T5
    //   T7 <- T6
    //   T8 <- T3
    //   T9 <- T8
    //   T10 <- T8
    //   T11 <- T8
    //   T12 <- T7, T9, T10, T11
    //
    // Total tasks: 13 (T0..T12). This exercises both local and remote
    // predecessor counters in the Bingo runtime across chips 0..3.
    // ------------------------------------------------------------------

    if (!task_list) {
        printf("[MULTI-DUMMY] Error: Null input pointers.\r\n");
        return 0;
    }
    check_kernel_tab_ready();
    // printf("Chip(%x, %x): [Host] Get kernel function...\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    uint32_t dummy_func_addr = get_device_function("__snax_kernel_dummy");
    if (dummy_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf("[MULTI-DUMMY] Error: Kernel symbol lookup failed!\r\n");
        return 0;
    }

    // Create tasks in ID order so that logical name Tn == task_id n

    bingo_task_t *T0  = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_0), chip_loc_to_chip_id(0,0), 0);
    bingo_task_t *T1  = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_1), chip_loc_to_chip_id(0,0), 1);
    bingo_task_t *T2  = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_2), chip_loc_to_chip_id(0,0), 2);
    bingo_task_t *T3  = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_3), chip_loc_to_chip_id(0,0), 3);
    bingo_task_t *T4  = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_4), chip_loc_to_chip_id(1,0), 0);
    bingo_task_t *T5  = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_5), chip_loc_to_chip_id(1,0), 1);
    bingo_task_t *T6  = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_6), chip_loc_to_chip_id(1,0), 2);
    bingo_task_t *T7  = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_7), chip_loc_to_chip_id(1,0), 3);
    bingo_task_t *T8  = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_8), chip_loc_to_chip_id(0,1), 0);
    bingo_task_t *T9  = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_9), chip_loc_to_chip_id(0,1), 1);
    bingo_task_t *T10 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_10), chip_loc_to_chip_id(0,1), 2);
    bingo_task_t *T11 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_11), chip_loc_to_chip_id(0,1), 3);
    bingo_task_t *T12 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_12), chip_loc_to_chip_id(1,1), 0); // final aggregation node

    // Sanity check allocation (minimal)
    if (!(T0 && T12)) {
        printf("[MULTI-DUMMY] Error: Task allocation failure.\n");

        return 0;
    }

    // Apply dependencies (successor, predecessor)
    // Chip0 subgraph
    bingo_task_add_depend(T1,  T0);
    bingo_task_add_depend(T2,  T0);
    bingo_task_add_depend(T3,  T1);
    bingo_task_add_depend(T3,  T2);
    // Chip1 chain
    bingo_task_add_depend(T4,  T3);
    bingo_task_add_depend(T5,  T4);
    bingo_task_add_depend(T6,  T5);
    bingo_task_add_depend(T7,  T6);
    // Chip2 fanout from T3
    bingo_task_add_depend(T8,  T3);
    bingo_task_add_depend(T9,  T8);
    bingo_task_add_depend(T10, T8);
    bingo_task_add_depend(T11, T8);
    // Final aggregation on chip3
    bingo_task_add_depend(T12, T7);
    bingo_task_add_depend(T12, T9);
    bingo_task_add_depend(T12, T10);
    bingo_task_add_depend(T12, T11);

    // Export tasks in ID order
    task_list[0]  = T0;
    task_list[1]  = T1;
    task_list[2]  = T2;
    task_list[3]  = T3;
    task_list[4]  = T4;
    task_list[5]  = T5;
    task_list[6]  = T6;
    task_list[7]  = T7;
    task_list[8]  = T8;
    task_list[9]  = T9;
    task_list[10] = T10;
    task_list[11] = T11;
    task_list[12] = T12;
    uint32_t num_tasks = 13; // T0..T12 inclusive
    // Export only up to T12
    return num_tasks;
}