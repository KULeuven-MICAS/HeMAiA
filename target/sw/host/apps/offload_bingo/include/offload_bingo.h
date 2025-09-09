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
// #include "xdma_1d_copy_workload.h"
// Devices
#define NUM_DEV N_CLUSTERS
#define NUM_CHIP 1
HeroDev dev_array[NUM_DEV];
int libhero_log_level = LOG_WARN;
void host_init_local_dev() {
    // printf("[Host] Start to Init the Devs\n");
    // We first reset the devs
    for (uint8_t i = 0; i < NUM_DEV; i++) {
        HeroDev *dev = &dev_array[i];
        dev->dev = NULL;
        dev->dev_id = i;
        dev->chip_id = 0;
        dev->global_mems = NULL;

        char *alias = dev->alias;
        alias[0] = 's';
        alias[1] = 'n';
        alias[2] = 'a';
        alias[3] = 'x';
        alias[4] = '_';
        alias[5] = 'c';
        alias[6] = 'o';
        alias[7] = 'r';
        alias[8] = 'e';
        alias[9] = '_';
        alias[10] = '0' + i; 
        alias[11] = '\0';
        dev->mboxes.a2h_mbox = NULL;
        dev->mboxes.a2h_mbox_mem.v_addr = NULL;
        dev->mboxes.a2h_mbox_mem.p_addr = (size_t)(0xBAADF00D);
        dev->mboxes.a2h_mbox_mem.size = (unsigned)0;
        dev->mboxes.a2h_mbox_mem.alias = "a2h_mbox";
        dev->mboxes.a2h_mbox_mem.next = NULL;
        dev->mboxes.h2a_mbox = NULL;
        dev->mboxes.h2a_mbox_mem.v_addr = NULL;
        dev->mboxes.h2a_mbox_mem.p_addr = (size_t)(0xBAADF00D);
        dev->mboxes.h2a_mbox_mem.size = (unsigned)0;
        dev->mboxes.h2a_mbox_mem.alias = "h2a_mbox";
        dev->mboxes.h2a_mbox_mem.next = NULL;        
        dev->mboxes.rb_mbox = NULL;
        dev->mboxes.rb_mbox_mem.v_addr = NULL;
        dev->mboxes.rb_mbox_mem.p_addr = (size_t)(0xBAADF00D);
        dev->mboxes.rb_mbox_mem.size = (unsigned)0;
        dev->mboxes.rb_mbox_mem.alias = "rb_mbox";
        dev->mboxes.rb_mbox_mem.next = NULL; 
    }
    // Init the L2/L3 Heap Allocator
    // printf("[Host] Init the Allocators\n");
    hero_dev_mmap_init();
    // Init the devices
    for (uint8_t i = 0; i < NUM_DEV; i++) {
        // This step will 
        // 1. Allocate the local and global mem
        // 2. Allocate the sw mailboxes
        // 3. Write the mail box pointers to the soc ctrl
        hero_dev_init(&dev_array[i]);
    }
}

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
    
    __workload_dummy(task_list, &num_tasks);
    // uintptr_t output_data_ptr = 0;
    // __workload_xdma_1d_copy(task_list, &num_tasks, &output_data_ptr);

    ////////////////////////////
    // End user defined workload
    ////////////////////////////

    // Call bingo runtime
    HeroDev global_dev[NUM_CHIP][NUM_DEV];
    sys_dma_blk_memcpy(
        (uint64_t)(uintptr_t)(&global_dev[0]),
        chiplet_addr_transform_full(0, (uint64_t)dev_array),
        sizeof(HeroDev) * NUM_DEV);
    asm volatile("fence" ::: "memory");
    HeroDev *dev_ptr_2d[NUM_CHIP][NUM_DEV];
    HeroDev **dev_ptrs[NUM_CHIP];

    for (int c = 0; c < NUM_CHIP; c++) {
        for (int i = 0; i < NUM_DEV; i++) {
            dev_ptr_2d[c][i] = &global_dev[c][i];
        }
        dev_ptrs[c] = dev_ptr_2d[c];
    }
    bingo_runtime_schedule(
        task_list, 
        num_tasks, 
        dev_ptrs, 
        NUM_CHIP, 
        NUM_DEV
    );
    // hero_host_l3_free((void*)output_data_ptr);
    return 0;
}