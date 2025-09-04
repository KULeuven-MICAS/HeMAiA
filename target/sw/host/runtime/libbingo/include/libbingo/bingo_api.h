// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "libhero/hero_api.h"
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

// Create a new task
bingo_task_t *bingo_task_create(uint32_t fn_ptr, uint32_t args_ptr);

// Declare dependency on the current task
void bingo_task_add_depend(bingo_task_t *task, bingo_task_t *dep_task);

// Offload the task to dev
void bingo_task_offload(bingo_task_t *task, HeroDev *dev);

void bingo_runtime_schedule(bingo_task_t **task_list, uint32_t num_tasks, HeroDev **dev_list, uint32_t num_devs);

void bingo_runtime_schedule_multichip(bingo_task_t **task_list, uint32_t num_tasks, HeroDev **dev_list_2d[], uint32_t num_chips, uint32_t num_devs_per_chip);