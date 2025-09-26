// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

#pragma once

#include <stdint.h>
#include <stdbool.h>
// Hero lib for sw mailbox and malloc
#include "libhero/hero_api.h"
// HW mailbox
#include "mailbox.h"
// Chip id
#include "chip_id.h"
// Bingo utils for sleep and cycle count
#include "bingo_utils.h"
extern struct O1HeapInstance *l2_heap_manager;
extern uint64_t l2_heap_start_phy, l2_heap_start_virt, l2_heap_size;
extern struct O1HeapInstance *l3_heap_manager;
extern uint64_t l3_heap_start_phy, l3_heap_start_virt, l3_heap_size;

#define HOST_SLEEP_CYCLES 100
#define MAX_SUCCESSORS 8
#define DEV_START (0x02U)
///////////////////////////////
///// Data Structure      /////
///////////////////////////////
// The task is running on the device 
// Let us fix the width of the pointers
typedef struct task {
  uint32_t task_id;            // unique task id
  uint32_t fn_ptr;             // function pointer
  uint32_t args_ptr;           // function args
  uint8_t num_predecessors;    // remaining deps count
  uint8_t num_successors;      // num of child tasks
  struct task *successors[MAX_SUCCESSORS];
  bool offloaded;              // flag to indicate the task has been offloaded or not
  bool completed;              // flag to indicate whether this task is finished or not
  uint8_t assigned_chip_id;    // assigned chip id
  uint8_t assigned_cluster_id; // assigned cluster id
} bingo_task_t;


////////////////////
///// API      /////
////////////////////

// Mailbox read/write functions

// H2H mailbox status/error codes
#define BINGO_MB_OK            0
#define BINGO_MB_ERR_PARAM    -1
#define BINGO_MB_ERR_TIMEOUT  -2

// Blocking write: waits until space available or timeout
// timeout_cycles = 0 means wait forever
// retry_hint allows caller to observe loop iterations (can pass NULL)
int bingo_write_h2h_mailbox(uint8_t chip_id, uint64_t dword,
                            uint64_t timeout_cycles, uint32_t *retry_hint);

// Blocking read: waits until data available or timeout
// timeout_cycles = 0 means wait forever
int bingo_read_h2h_mailbox(uint64_t *buffer,
                           uint64_t timeout_cycles, uint32_t *retry_hint);

// Non-blocking attempts
// Return: 1 = success (read/write performed), 0 = would block, <0 = error
int bingo_try_write_h2h_mailbox(uint8_t chip_id, uint64_t dword);
int bingo_try_read_h2h_mailbox(uint64_t *buffer);

// H2C mailbox
// Host can write to any chip's H2C mailbox
void bingo_write_h2c_mailbox(HeroDev *dev, uint32_t word);

// C2H mailbox
// Host can only read from its own chip's C2H mailbox
uint32_t bingo_read_c2h_mailbox(HeroDev *dev, uint32_t *buffer); 


// Create a new task
bingo_task_t *bingo_task_create(uint32_t fn_ptr, uint32_t args_ptr);

// Declare dependency on the current task
void bingo_task_add_depend(bingo_task_t *task, bingo_task_t *dep_task);

// Offload the task to dev
void bingo_task_offload(bingo_task_t *task, HeroDev *dev);

// The main scheduling function
void bingo_runtime_schedule(bingo_task_t **task_list, uint32_t num_tasks, HeroDev **dev_list_2d[], uint32_t num_chips, uint32_t num_devs_per_chip);

