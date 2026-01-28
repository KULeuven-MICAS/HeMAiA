// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

typedef struct {
  uint32_t dev_arg_list_ptr;
  uint32_t dev_kernel_list_ptr;
  uint32_t gid_to_dev_tid_list_ptr;
} bingo_hw_offload_unit_t;


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
} bingo_sw_offload_unit_t;

#define BINGO_C2H_FLAG_MASK        0xFu
#define BINGO_C2H_CLUSTER_MASK     0xFFu
#define BINGO_C2H_TASK_MASK        0xFFFFu
#define BINGO_C2H_FLAG_SHIFT       0
#define BINGO_C2H_CLUSTER_SHIFT    4
#define BINGO_C2H_TASK_SHIFT       12
#define BINGO_C2H_RESERVED_SHIFT   28

typedef struct {
  uint16_t task_id;   // 16-bit task identifier
  uint8_t  cluster_id;// Source cluster id
  uint8_t  flag;      // Lower 4 bits meaningful (MBOX_DEVICE_*)
  uint8_t  reserved;  // Upper 4 bits of the original packed word (kept for future)
} bingo_c2h_msg_fields_t;

static inline uint32_t bingo_c2h_msg_encode(bingo_c2h_msg_fields_t f) {
  return ((uint32_t)(f.reserved & 0xF)    << BINGO_C2H_RESERVED_SHIFT) |
         ((uint32_t)(f.task_id  & BINGO_C2H_TASK_MASK)    << BINGO_C2H_TASK_SHIFT) |
         ((uint32_t)(f.cluster_id & BINGO_C2H_CLUSTER_MASK) << BINGO_C2H_CLUSTER_SHIFT) |
         ((uint32_t)(f.flag & BINGO_C2H_FLAG_MASK) << BINGO_C2H_FLAG_SHIFT);
}

/**
 * @brief Initialize the bingo offload unit
 */
inline void bingo_sw_offload_init();

/**
 * @brief send all workers in loop to exit()
 */
inline void bingo_sw_offload_exit();

/**
 * @brief Set function to execute by `nthreads` number of threads
 * @details
 *
 * @param offloadFn pointer to worker function to be executed
 * @param offloadArg pointer to function arguments
 */
inline void bingo_sw_offload_dispatch(uint32_t (*offloadFn)(uint32_t), uint32_t offloadArg);


/**
 * @brief Debugging info to printf
 * @details
 */
inline void bingo_sw_offload_print_status();
