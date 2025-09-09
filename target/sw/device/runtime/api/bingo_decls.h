// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
typedef struct {
    // Here are the fields that are used by the communication between the host and the cluster
    volatile struct l2_layout *l2l_ptr;       // pointer to the L2 layout
    volatile struct ring_buf *g_a2h_rb;               // pointer to the ring buffer for acc->host for system calls
    volatile struct ring_buf *g_a2h_mbox;             // pointer to the mailbox for acc->host for cmds
    volatile struct ring_buf *g_h2a_mbox;             // pointer to the mailbox for host ->acc for cmds
    uint32_t cmd;                    // cmd from host
    uint32_t task_id;                // task id from host
    uint32_t (*offloadFn)(uint32_t); // offloaded function pointer
    uint32_t offloadArgs;            // offloaded function arguments
    // Here are the fields that are used by the offload unit to manage the workers
    uint32_t exit_flag;              // terminate the offload unit
    uint32_t start_flag;             // start the workers
    uint32_t workers_in_loop;        // number of workers in the loop
    uint32_t workers_wfi;            // number of workers in WFI
    uint32_t fini_count;             // number of workers that finished the offloaded function
    // Here are the fields for performance monitoring
    uint32_t start_cycles;           // start cycles of the offloaded function
    uint32_t end_cycles;             // end cycles of the offloaded function
} bingo_offload_unit_t;

/**
 * @brief Initialize the bingo offload unit
 */
inline void bingo_offload_init();

/**
 * @brief send all workers in loop to exit()
 */
inline void bingo_offload_exit();

/**
 * @brief Enter the event unit loop, never exits
 *
 * @param cluster_core_idx cluster-local core index
 */
inline void bingo_offload_loop(uint32_t cluster_core_idx);

/**
 * @brief Set function to execute by `nthreads` number of threads
 * @details
 *
 * @param offloadFn pointer to worker function to be executed
 * @param offloadArg pointer to function arguments
 */
inline void bingo_offload_dispatch(uint32_t (*offloadFn)(uint32_t), uint32_t offloadArg);


/**
 * @brief Debugging info to printf
 * @details
 */
inline void bingo_offload_print_status();
