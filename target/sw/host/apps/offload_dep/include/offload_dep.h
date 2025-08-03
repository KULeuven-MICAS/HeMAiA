// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include "libbingo/bingo_api.h"
#include "host.h"
#include "hemaia_clk_rst_controller.h"

// Global Variables for communication buffer
volatile comm_buffer_t* comm_buffer_ptr = (comm_buffer_t*)0;

// Devices
#define NUM_DEV N_CLUSTERS
HeroDev dev_array[NUM_DEV];
HeroDev *dev_list[NUM_DEV];
int libhero_log_level = LOG_WARN;
void host_init_dev() {
    printf("[Host] Start to Init the Devs\n");
    // We first reset the devs
    for (uint8_t i = 0; i < NUM_DEV; i++) {
        HeroDev *dev = &dev_array[i];
        dev->dev = NULL;
        dev->dev_id = i;
        dev->local_mems = NULL;
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
    printf("[Host] Init the Allocators\n");
    hero_dev_mmap_init();
    // Init the devices
    for (uint8_t i = 0; i < NUM_DEV; i++) {
        // This step will 
        // 1. Allocate the local and global mem
        // 2. Allocate the sw mailboxes
        // 3. Write the mail box pointers to the soc ctrl
        printf("[Host] Init the Dev %d\n", i);
        hero_dev_init(&dev_array[i]);
    }
    // printf("[Host] Mailbox h2a[0] address = %x\n", dev_array[0].mboxes.h2a_mbox_mem.p_addr);
    // printf("[Host] Mailbox h2a[1] address = %x\n", dev_array[1].mboxes.h2a_mbox_mem.p_addr);
}

// Kernel Execution

int test_kernel_execution(){
    for (int i = 0; i < NUM_DEV; i++) {
        dev_list[i] = &dev_array[i];
    }
    // Set up the kernel args
    uint32_t dummy_args[] = {42};
    uint32_t test_data_size = 8;
    uintptr_t dummy_input_addr = hero_dev_l2_malloc(&dev_array[0], test_data_size*sizeof(uint32_t), NULL);
    uintptr_t dummy_output_addr = hero_dev_l2_malloc(&dev_array[0], test_data_size*sizeof(uint32_t), NULL);
    uint32_t* dummy_input_array = (uint32_t*)dummy_input_addr;
    uint32_t* dummy_output_array = (uint32_t*)dummy_output_addr;
    for (uint32_t i = 0; i < test_data_size; i++)
    {
        dummy_input_array[i] = i;
    }
    uint32_t load_compute_store_args[] = {
        (uint32_t)dummy_input_addr, // Input data address
        test_data_size * sizeof(uint32_t), // Input data size in bytes
        (uint32_t)dummy_output_addr, // Output data address
        test_data_size * sizeof(uint32_t) // Output data size in bytes
    };
    // Get the kernel function address
    check_kernel_tab_ready();
    uint32_t dummy_func_addr = get_device_function("__snax_kernel_dummy");
    uint32_t load_compute_store_addr = get_device_function("__snax_kernel_load_compute_store");
    if (dummy_func_addr == 0xBAADF00D || load_compute_store_addr == 0xBAADF00D) {
        printf("Error: Kernel symbol lookup failed!\n");
        return -1;
    }
    // printf("[Host] Dummy Function Address: 0x%x\n", dummy_func_addr);
    // printf("[Host] CSR Function Address: 0x%x\n", csr_func_addr);
    // printf("[Host] Dummy Args Address: 0x%x\n", dummy_args);
    // printf("[Host] CSR Args Address: 0x%x\n", csr_args);
    // Create two tasks
    bingo_task_t *task_dummy = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)dummy_args);
    bingo_task_t *task_load_compute_store   = bingo_task_create(load_compute_store_addr, (uint32_t)(uintptr_t)load_compute_store_args);

    // Set the task dependency
    // load_compute_store task depends on the dummy task
    bingo_task_add_depend(task_dummy, task_load_compute_store);

    // Set the assigned cluster id
    task_dummy->assigned_cid = 1;
    task_load_compute_store->assigned_cid = 0;

    // Set up the bingo scheduler
    bingo_task_t *task_list[] = {task_dummy, task_load_compute_store};
    bingo_runtime_schedule(task_list, 2, dev_list, NUM_DEV);
    return 0;
}