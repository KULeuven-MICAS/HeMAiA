// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

#include "libbingo/bingo_api.h"

uint32_t global_task_id = 0;

bingo_task_t *bingo_task_create(uint32_t fn_ptr, uint32_t args_ptr) {
    bingo_task_t *task = o1heapAllocate(l2_heap_manager, sizeof(bingo_task_t));
    task->task_id = global_task_id;
    task->fn_ptr = fn_ptr;
    task->args_ptr = args_ptr;
    task->num_predecessors = 0;
    task->num_successors = 0;
    for (int i = 0; i < MAX_SUCCESSORS; i++) {
        task->successors[i] = NULL;
    }
    task->offloaded = false;
    task->completed = false;
    task->assigned_cid = 0;
    global_task_id += 1;
    return task;
}

void bingo_task_add_depend(bingo_task_t *task, bingo_task_t *dep_task) {
    task->num_predecessors++;
    dep_task->successors[dep_task->num_successors] = task;
    dep_task->num_successors++;
}

void bingo_task_offload(bingo_task_t *task, HeroDev *dev) {
    // The pre-defined sequence is:
    // 1. Write the Start CMD to Dev
    // 2. Write the Task Id to Dev
    // 3. Write the kernel function address to Dev
    // 4. Write the kernel function args address to Dev
    printf("[Host] Offloading task %d to cluster %d with function pointer 0x%x and args pointer 0x%x\n",
            task->task_id, task->assigned_cid, task->fn_ptr, task->args_ptr);
    hero_dev_mbox_write(dev, (uint32_t)DEV_START);
    hero_dev_mbox_write(dev, (uint32_t)(task->task_id));
    hero_dev_mbox_write(dev, (uint32_t)(task->fn_ptr));
    hero_dev_mbox_write(dev, (uint32_t)(task->args_ptr));

    task->offloaded = true;
}

void bingo_runtime_schedule(bingo_task_t **task_list, uint32_t num_tasks, HeroDev **dev_list, uint32_t num_devs) {
    uint32_t num_completed_task = 0;

    while (num_completed_task != num_tasks) {
        // (1) Poll all devs for completed tasks
        for (uint32_t d = 0; d < num_devs; d++) {
            HeroDev *dev = dev_list[d];
            uint32_t dev_return_flag;
            uint32_t dev_completed_task_id;
            uint32_t dev_kernel_cc;
            uint32_t ret = hero_dev_mbox_try_read(dev, &dev_return_flag);
            if(ret){
                printf("[Host] Dev %d returned with flag %d\n", dev->dev_id, dev_return_flag);
                if (dev_return_flag == MBOX_DEVICE_DONE) {
                    hero_dev_mbox_read(dev, &dev_completed_task_id, 1);
                    hero_dev_mbox_read(dev, &dev_kernel_cc, 1);

                    bingo_task_t *complete_task = task_list[dev_completed_task_id];
                    complete_task->completed = true;
                    num_completed_task++;

                    for (uint32_t s = 0; s < complete_task->num_successors; s++) {
                        bingo_task_t *suc = complete_task->successors[s];
                        suc->num_predecessors--;
                    }
                }                
            }
        }

        // (2) Offload tasks with no predecessors and not yet offloaded
        for (uint32_t i = 0; i < num_tasks; i++) {
            bingo_task_t *t = task_list[i];
            if (!t->offloaded && t->num_predecessors == 0) {
                bingo_task_offload(t, dev_list[t->assigned_cid]);
            }
        }

        // (3) Sleep to reduce CPU usage
        bingo_csleep(HOST_SLEEP_CYCLES);
    }
    // All tasks are completed, close the devs
    for (uint32_t d = 0; d < num_devs; d++) {
        HeroDev *dev = dev_list[d];
        hero_dev_mbox_write(dev, MBOX_DEVICE_STOP);
        printf("[Host] Dev %d completed all tasks\n", dev->dev_id);
    }
}