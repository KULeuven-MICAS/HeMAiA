// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include "libbingo/bingo_api.h"
#include "host.h"
#include "hemaia_clk_rst_controller.h"
#include "data.h"
// Devices
#define NUM_DEV N_CLUSTERS
#define NUM_CHIP N_CHIPLETS
HeroDev dev_array[NUM_DEV];
int libhero_log_level = LOG_WARN;
void host_init_local_dev(){
    // Here we init all the Mailboxs for host<->dev communication
    // All the chiplet only perform local initilization
    // printf("[Host] Start to Init the Local Devs\r\n");
    // We first reset the local devs
    for (uint8_t i = 0; i < NUM_DEV; i++) {
        HeroDev *dev = (HeroDev*)chiplet_addr_transform((uint64_t)(&dev_array[i]));
        dev->dev = NULL;
        dev->dev_id = i;
        dev->chip_id = get_current_chip_id();
        dev->global_mems = NULL;

        char *alias = dev->alias;
        alias[0]  = 's';
        alias[1]  = 'n';
        alias[2]  = 'a';
        alias[3]  = 'x';
        alias[4]  = '_';
        alias[5]  = 'c';
        alias[6]  = 'l';
        alias[7]  = 'u';
        alias[8]  = 's';
        alias[9]  = 't';
        alias[10] = 'e';
        alias[11] = 'r';
        alias[12] = '_';
        alias[13] = '0' + i; 
        alias[14] = '_';
        alias[15] = 'c';
        alias[16] = 'h';
        alias[17] = 'i';
        alias[18] = 'p';
        alias[19] = '_';
        alias[20] = '0' + get_current_chip_id();
        alias[21] = '\0';
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
    // printf("[Host] Init the Allocators\r\n");
    hero_dev_mmap_init();
    // Init the devices
    for (uint8_t i = 0; i < NUM_DEV; i++) {
        HeroDev *dev = (HeroDev*)chiplet_addr_transform((uint64_t)(&dev_array[i]));
        // This step will 
        // 1. Allocate the local and global mem
        // 2. Allocate the sw mailboxes
        // 3. Write the mail box pointers to the soc ctrl
        // printf("[Host] Init the Dev %d\r\n", i);
        hero_dev_init(dev);
    }
}

int kernel_execution(){
    // Only the main chiplet will run this function
    // Since the main memory will be copied exactly to all the chiplets
    // The lower address of the dev_array will be the same
    // Only the upper 48-40 chiplet id prefix will be different


    // Get the kernel function address
    check_kernel_tab_ready();
    // printf("[Host] Init the Kernel Symbol Table\r\n");
    uint32_t dummy_func_addr = get_device_function("__snax_kernel_dummy");
    if (dummy_func_addr == 0xBAADF00D) {
        printf("Error: Kernel symbol lookup failed!\r\n");
        return -1;
    }

    // Register the tasks
    // printf("[Host] Start to Register tasks\r\n");
    bingo_task_t *task_dummy_chip0_cluster0 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_chip0_cluster0));
    bingo_task_t *task_dummy_chip0_cluster1 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_chip0_cluster1));
    bingo_task_t *task_dummy_chip1_cluster0 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_chip1_cluster0));
    bingo_task_t *task_dummy_chip1_cluster1 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_chip1_cluster1));

    // Set the task dependency
    //      c0_cl0
    //           |
    //           v
    //  ----c1_cl1
    //  |        |
    //  v        v
    // c0_cl1   c1_cl0
    // So the c0_cl0 will start to run
    // Then the c1_cl1 will run after
    // Then the c0_cl0 and  c1_cl0 will run in parallel
    // We set the dependency from the child to the parent
    bingo_task_add_depend(task_dummy_chip0_cluster1, task_dummy_chip1_cluster1);
    bingo_task_add_depend(task_dummy_chip1_cluster0, task_dummy_chip1_cluster1);
    bingo_task_add_depend(task_dummy_chip1_cluster1, task_dummy_chip0_cluster0);

    //Set the assigned chiplet id and cluster id
    task_dummy_chip0_cluster0->assigned_chip_id = 0;
    task_dummy_chip0_cluster0->assigned_cluster_id = 0;
    task_dummy_chip0_cluster1->assigned_chip_id = 0;
    task_dummy_chip0_cluster1->assigned_cluster_id = 1;
    task_dummy_chip1_cluster0->assigned_chip_id = 1;
    task_dummy_chip1_cluster0->assigned_cluster_id = 0;
    task_dummy_chip1_cluster1->assigned_chip_id = 1;
    task_dummy_chip1_cluster1->assigned_cluster_id = 1;

    // Set up the bingo scheduler
    bingo_task_t *task_list[] = {task_dummy_chip0_cluster0, 
                                 task_dummy_chip0_cluster1,
                                 task_dummy_chip1_cluster0,
                                 task_dummy_chip1_cluster1};
    // Call bingo runtime
    HeroDev global_dev[NUM_CHIP][NUM_DEV];
    // printf("[Host] The local dev array addr is at %lx\r\n", (uint64_t)(uintptr_t)(chiplet_addr_transform_full(0, (uint64_t)dev_array)));
    sys_dma_blk_memcpy(
        (uint64_t)(uintptr_t)(&global_dev[0]),
        chiplet_addr_transform_full(0, (uint64_t)dev_array),
        sizeof(HeroDev) * NUM_DEV);
    asm volatile("fence" ::: "memory");
    // printf("[Host] The remote dev array addr is at %lx\r\n", (uint64_t)(uintptr_t)(chiplet_addr_transform_full(16, (uint64_t)dev_array)));
    sys_dma_blk_memcpy(
        (uint64_t)(uintptr_t)(&global_dev[1]),
        chiplet_addr_transform_loc(1, 0, (uint64_t)dev_array),
        sizeof(HeroDev) * NUM_DEV);
    asm volatile("fence" ::: "memory");


    // for (int c = 0; c < NUM_CHIP; c++){
    //     printf("[Host] Init the global dev @ chip %d\r\n", c);
    //     printf("[Host] The global dev array addr is at %lx\r\n", (uint64_t)(uintptr_t)(&global_dev[c]));
    //     printf("[Host] The dev array addr is at %lx\r\n", (uint64_t)(uintptr_t)(chiplet_addr_transform_full(c, (uint64_t)dev_array)));
    //     sys_dma_blk_memcpy(
    //         (uint64_t)(uintptr_t)(&global_dev[c]),
    //         chiplet_addr_transform_full(c, (uint64_t)dev_array),
    //         sizeof(HeroDev) * NUM_DEV);
    //     asm volatile("fence" ::: "memory");
    // }
    // printf("[Host] Init the global dev @%lx\r\n", (uint64_t)(uintptr_t)(&global_dev));
    // printf("[Host] The h2a mailbox of chip 0 cluster 0 is at 0x%lx\r\n", global_dev[0][0].mboxes.h2a_mbox);
    // printf("[Host] The h2a mailbox of chip 0 cluster 1 is at 0x%lx\r\n", global_dev[0][1].mboxes.h2a_mbox);
    // printf("[Host] The h2a mailbox of chip 1 cluster 0 is at 0x%lx\r\n", global_dev[1][0].mboxes.h2a_mbox);
    // printf("[Host] The h2a mailbox of chip 1 cluster 1 is at 0x%lx\r\n", global_dev[1][1].mboxes.h2a_mbox);



    HeroDev *dev_ptr_2d[NUM_CHIP][NUM_DEV];
    for (int c = 0; c < NUM_CHIP; c++) {
        for (int i = 0; i < NUM_DEV; i++) {
            dev_ptr_2d[c][i] = &global_dev[c][i];
        }
    }
    HeroDev **dev_ptrs[NUM_CHIP];
    for (int c = 0; c < NUM_CHIP; c++) {
        dev_ptrs[c] = dev_ptr_2d[c];
    }
    bingo_runtime_schedule_multichip(
        task_list, 
        sizeof(task_list)/sizeof(task_list[0]), 
        dev_ptrs, 
        NUM_CHIP, 
        NUM_DEV
    );
    return 0;
}