// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Cyril Koenig <cykoenig@iis.ee.ethz.ch>

#include <stdint.h>

#include "libhero/debug.h"
#include "libhero/hero_api.h"
#include "libhero/io.h"
#include "libhero/utils.h"

#include "allocators.h"
#include "snitch_cluster.h"


// static void occamy_set_isolation(int iso) {
//     uint32_t mask, val;
//     val = iso ? 1U : 0U;
//     mask = (val << QCTL_ISOLATE_NARROW_IN_BIT) | (val << QCTL_ISOLATE_NARROW_OUT_BIT) |
//            (val << QCTL_ISOLATE_WIDE_IN_BIT) | (val << QCTL_ISOLATE_WIDE_OUT_BIT);
//     writew(mask, occ_quad_ctrl + QCTL_ISOLATE_REG_OFFSET);
//     fence();
// }

// void clint_set_irq(uint32_t irq) {
//     int val = readw((uint32_t *)occ_clint);
//     writew(val | irq, (uint32_t *)occ_clint);
// }

// void clint_clear_irq(uint32_t irq) {
//     int val = readw((uint32_t *)occ_clint);
//     writew(val & ~irq, (uint32_t *)occ_clint);
// }

// Set up transparent TLB
static int occamy_tlb_write(uint32_t idx, uint64_t addr_begin, uint64_t addr_end, uint32_t flags) {
    // uint64_t page_num_base = addr_begin >> 12;
    // uint64_t page_num_first = addr_begin >> 12;
    // uint64_t page_num_last = addr_end >> 12;

    // writew((uint32_t)  page_num_first        , (uint32_t *) (occ_quad_ctrl + QCTL_TLB_NARROW_REG_OFFSET + QCTL_TLB_REG_STRIDE * idx + 0x00) );
    // writew((uint32_t) (page_num_first >> 32) , (uint32_t *) (occ_quad_ctrl + QCTL_TLB_NARROW_REG_OFFSET + QCTL_TLB_REG_STRIDE * idx + 0x04) );
    // writew((uint32_t)  page_num_last         , (uint32_t *) (occ_quad_ctrl + QCTL_TLB_NARROW_REG_OFFSET + QCTL_TLB_REG_STRIDE * idx + 0x08) );
    // writew((uint32_t) (page_num_last >> 32)  , (uint32_t *) (occ_quad_ctrl + QCTL_TLB_NARROW_REG_OFFSET + QCTL_TLB_REG_STRIDE * idx + 0x0c) );
    // writew((uint32_t)  page_num_base         , (uint32_t *) (occ_quad_ctrl + QCTL_TLB_NARROW_REG_OFFSET + QCTL_TLB_REG_STRIDE * idx + 0x10) );
    // writew((uint32_t) (page_num_base >> 32)  , (uint32_t *) (occ_quad_ctrl + QCTL_TLB_NARROW_REG_OFFSET + QCTL_TLB_REG_STRIDE * idx + 0x14) );
    // writew((uint32_t)  flags                 , (uint32_t *) (occ_quad_ctrl + QCTL_TLB_NARROW_REG_OFFSET + QCTL_TLB_REG_STRIDE * idx + 0x18) );
    // writew((uint32_t)  page_num_first        , (uint32_t *) (occ_quad_ctrl + QCTL_TLB_WIDE_REG_OFFSET   + QCTL_TLB_REG_STRIDE * idx + 0x00) );
    // writew((uint32_t) (page_num_first >> 32) , (uint32_t *) (occ_quad_ctrl + QCTL_TLB_WIDE_REG_OFFSET   + QCTL_TLB_REG_STRIDE * idx + 0x04) );
    // writew((uint32_t)  page_num_last         , (uint32_t *) (occ_quad_ctrl + QCTL_TLB_WIDE_REG_OFFSET   + QCTL_TLB_REG_STRIDE * idx + 0x08) );
    // writew((uint32_t) (page_num_last >> 32)  , (uint32_t *) (occ_quad_ctrl + QCTL_TLB_WIDE_REG_OFFSET   + QCTL_TLB_REG_STRIDE * idx + 0x0c) );
    // writew((uint32_t)  page_num_base         , (uint32_t *) (occ_quad_ctrl + QCTL_TLB_WIDE_REG_OFFSET   + QCTL_TLB_REG_STRIDE * idx + 0x10) );
    // writew((uint32_t) (page_num_base >> 32)  , (uint32_t *) (occ_quad_ctrl + QCTL_TLB_WIDE_REG_OFFSET   + QCTL_TLB_REG_STRIDE * idx + 0x14) );
    // writew((uint32_t)  flags                 , (uint32_t *) (occ_quad_ctrl + QCTL_TLB_WIDE_REG_OFFSET   + QCTL_TLB_REG_STRIDE * idx + 0x18) );
    pr_warn("%s unimplemented\n", __func__);
    return 0;
}

// void hero_dev_reset(HeroDev *dev, unsigned full) {
//     pr_trace("%s snitch_cluster\n", __func__);
//     // Isolate
//     occamy_set_isolation(1);
//     // Reset
//     writew(0, occ_quad_ctrl + QCTL_RESET_N_REG_OFFSET);
//     clint_clear_irq(0x1FF << 1);
//     fence();
//     for (volatile int i = 0; i < 16; i++)
// 	;
//     // De-reset
//     writew(1, occ_quad_ctrl + QCTL_RESET_N_REG_OFFSET);
//     fence();
//     // De-isolate
//     occamy_set_isolation(0);
// }


int hero_dev_mmap_init(){
    int err = 0;
    // We use the physical address
    occ_soc_ctrl = (void *)SOC_CTRL_BASE_ADDR;
    occ_l3_heap = (void *)(__l3_heap_start);
    occ_l2_heap = (void *)(SPM_NARROW_BASE_ADDR + L2_OFFSET); // Heap starts after the spm_base+mailbox_size
    // Assign the memory info to the L2
    // 1. The mailbox region at the beginning of L2
    l2_mailbox_size = NARROW_SPM_MAILBOX_SIZE;
    l2_mailbox_start_phy = (uintptr_t)SPM_NARROW_BASE_ADDR;
    l2_mailbox_start_virt = (uintptr_t)SPM_NARROW_BASE_ADDR;
    
    // 2. Put half of the l2 memory map to heap allocator
    l2_heap_size = SPM_NARROW_SIZE / 2;
    l2_heap_start_phy = (uintptr_t)occ_l2_heap;
    l2_heap_start_virt = (uintptr_t)occ_l2_heap;
    err = hero_dev_l2_init();
    if(err) {
        pr_error("Error when initializing L2 mem.\n");
        return err;
    }
    // Put half of the l3 memory map to heap allocator
    uint64_t occ_l3_heap_size = SPM_WIDE_SIZE / 2;
    l3_heap_size = ALIGN_UP(occ_l3_heap_size, O1HEAP_ALIGNMENT);
    l3_heap_start_phy = (uintptr_t)(&__l3_heap_start);
    l3_heap_start_virt = (uintptr_t)(&__l3_heap_start);
    // printf("[Init] L3 heap start addr = %x, size = %x\n", l3_heap_start_phy, l3_heap_size);
    err = hero_dev_l3_init();
    if(err) {
        pr_error("Error when initializing L3 mem.\n");
        return err; 
    }   
}


// In HeMAiA we use the physical address
// So we map the physical address to the virtual address
// Besides we currently do not need to support the full openmp
// We use a minimal task dep management runtime instead
int hero_dev_mmap(HeroDev *dev) {
    // The L3 shared mem
    int64_t global_mems_phy;
    HeroSubDev_t *global_mems_tail = (HeroSubDev_t *)hero_dev_l3_malloc(dev, sizeof(HeroSubDev_t), &global_mems_phy);
    if(!global_mems_tail){
        pr_error("Error when allocating global_mems_tail.\n");
        return 1;
    }
    global_mems_tail->v_addr = (unsigned *)occ_l3_heap;
    global_mems_tail->p_addr = (size_t)occ_l3_heap;
    global_mems_tail->alias = "l3";
    global_mems_tail->next = NULL;
    dev->global_mems = global_mems_tail;
    return 0;
}

int hero_dev_init(HeroDev *dev){
    pr_trace("%p\n", dev);
    // Allocate the local and global mem
    hero_dev_mmap(dev);
    // Allocate sw mailboxes
    hero_dev_alloc_mboxes(dev);
    // Point to the mailboxes
    uint64_t mbox_ptrs_phy;
    struct l2_layout *mbox_ptrs = (struct l2_layout *)hero_dev_l2_mailbox_malloc(dev, sizeof(struct l2_layout), &mbox_ptrs_phy);
    mbox_ptrs->h2a_mbox = (uint32_t) dev->mboxes.h2a_mbox_mem.p_addr;
    mbox_ptrs->a2h_mbox = (uint32_t) dev->mboxes.a2h_mbox_mem.p_addr;
    mbox_ptrs->a2h_rb = (uint32_t) dev->mboxes.rb_mbox_mem.p_addr;
    mbox_ptrs->heap = l3_heap_start_phy;
    // Give the poiter to the mailboxes to the device
    writew(mbox_ptrs_phy, soc_ctrl_mailbox_scratch_addr(dev->dev_id));
    pr_trace("Write Mbox Address %x to %x\n",mbox_ptrs_phy,soc_ctrl_mailbox_scratch_addr(dev->dev_id));
    return 0;
}

// TODO: Removed numerical constants and share some header file
// with the driver
// int hero_dev_mmap(HeroDev *dev) {
//     int err = 0;

//     char* env_libhero_log = getenv("LIBHERO_LOG");
//     if(env_libhero_log)
//         libhero_log_level = strtol(env_libhero_log, NULL, 10);

//     pr_trace("\n");

//     device_fd = open("/dev/occamydev--1", O_RDWR | O_SYNC);
//     CHECK_ASSERT(-1, device_fd > 0, "Can't open driver chardev\n");

//     // Call card_mmap from the driver map address spaces
//     err |= driver_lookup_mmap(device_fd, SNITCH_CLUSTER_MMAP_ID, &occ_snitch_cluster);
//     err |= driver_lookup_mmap(device_fd, QUADRANT_CTRL_MMAP_ID, &occ_quad_ctrl);
//     err |= driver_lookup_mmap(device_fd, SOC_CTRL_MMAP_ID, &occ_soc_ctrl);
//     err |= driver_lookup_mmap(device_fd, L3_MMAP_ID, &occ_l3);
//     err |= driver_lookup_mmap(device_fd, SCRATCHPAD_WIDE_MMAP_ID, &occ_l2);
//     err |= driver_lookup_mmap(device_fd, CLINT_MMAP_ID, &occ_clint);

//     fflush(stdout);

//     if (err)
//         goto error_driver;
    
//     // Put the occamy tcdm in the local_mems list for OpenMP to use
//     HeroSubDev_t *local_mems_tail = malloc(sizeof(HeroSubDev_t));
//     size_t occ_snitch_cluster_size; 
//     uintptr_t occ_snitch_cluster_phys;
//     driver_lookup_mem(device_fd, SNITCH_CLUSTER_MMAP_ID, &occ_snitch_cluster_size, &occ_snitch_cluster_phys);
//     if(!local_mems_tail){
//         pr_error("Error when allocating local_mems_tail.\n");
//         goto error_driver;
//     }
//     local_mems_tail->v_addr = occ_snitch_cluster;
//     // TODO: Split lookup between device and host phy addr
//     local_mems_tail->p_addr = 0xFFFFFFFF & occ_snitch_cluster_phys;
//     local_mems_tail->size   = occ_snitch_cluster_size;
//     local_mems_tail->alias  = "l1_snitch_cluster";
//     dev->local_mems = local_mems_tail;

//     // Put half of the l2 memory map in the global_mems list
//     // (We give the first half to openmp and the top half to o1heap)
//     size_t occ_l2_size; 
//     uintptr_t occ_l2_phys; 
//     driver_lookup_mem(device_fd, SCRATCHPAD_WIDE_MMAP_ID, &occ_l2_size, &occ_l2_phys);

//     // Use the upper half of the device L2 mem for the heap allocator
//     l2_heap_start_phy = occ_l2_phys + ALIGN_UP(occ_l2_size / 2, O1HEAP_ALIGNMENT);
//     l2_heap_start_virt = occ_l2 + ALIGN_UP(occ_l2_size / 2, O1HEAP_ALIGNMENT);
//     l2_heap_size = occ_l2_size / 2;
//     err = hero_dev_l2_init(dev);
//     if(err) {
//         pr_error("Error when initializing L2 mem.\n");
//         goto end;
//     }

//     // Use the L3 mem for heap allocator
//     size_t occ_l3_size; 
//     uintptr_t occ_l3_phys; 
//     driver_lookup_mem(device_fd, L3_MMAP_ID, &occ_l3_size, &occ_l3_phys);
//     l3_heap_start_phy = occ_l3_phys + ALIGN_UP(occ_l3_size / 2, O1HEAP_ALIGNMENT);
//     l3_heap_start_virt = occ_l3 + ALIGN_UP(occ_l3_size / 2, O1HEAP_ALIGNMENT);
//     l3_heap_size = occ_l3_size / 2;
//     err = hero_dev_l3_init(dev);
//     if(err) {
//         pr_error("Error when initializing L3 mem.\n");
//         goto end;
//     }

//     // Give the rest of L3 to openmp
//     HeroSubDev_t *l2_mems_tail = malloc(sizeof(HeroSubDev_t));
//     if(!l2_mems_tail){
//         pr_error("Error when allocating l2_mems_tail.\n");
//         goto error_driver;
//     }
//     l2_mems_tail->v_addr = occ_l3;
//     // TODO: Split lookup between device and host phy addr
//     l2_mems_tail->p_addr = 0xFFFFFFFF & occ_l3_phys;
//     l2_mems_tail->size   = occ_l3_size / 2;
//     l2_mems_tail->alias  = "l3";
//     l2_mems_tail->next   = NULL;
//     dev->global_mems = l2_mems_tail;
    
//     goto end;
// error_driver:
//     err = -1;
//     pr_error("Error when communicating with the Carfield driver.\n");
// end:
//     return err;
// }
// int hero_dev_init(HeroDev *dev) {
//     pr_trace("%p\n", dev);

//     // Allocate sw mailboxes
//     hero_dev_alloc_mboxes(dev);
//     // Point to the mailboxes
//     int64_t mbox_ptrs_phy;
//     struct l3_layout *mbox_ptrs = hero_dev_l3_malloc(dev, sizeof(struct l3_layout), &mbox_ptrs_phy);

//     mbox_ptrs->h2a_mbox = (uint32_t) dev->mboxes.h2a_mbox_mem.p_addr;
//     mbox_ptrs->a2h_mbox = (uint32_t) dev->mboxes.a2h_mbox_mem.p_addr;
//     mbox_ptrs->a2h_rb = (uint32_t) dev->mboxes.rb_mbox_mem.p_addr;
//     mbox_ptrs->heap = l3_heap_start_phy;
//     // Give the poiter to the mailboxes to the device
//     writew(mbox_ptrs_phy, occ_soc_ctrl + SCTL_SCRATCH_2_REG_OFFSET);

//     // Setup the TLBs
//     occamy_tlb_write(0, 0x01000000, 0x0101ffff, 0x3);  // BOOTROM
//     occamy_tlb_write(1, 0x02000000, 0x02000fff, 0x3);  // SoC Control
//     occamy_tlb_write(2, 0x04000000, 0x040fffff, 0x1);  // CLINT
//     occamy_tlb_write(3, 0x10000000, 0x105fffff, 0x1);  // Quadrants
//     occamy_tlb_write(4, 0xC0000000, 0xffffffff, 0x1);  // HBM0/1
//     occamy_tlb_write(5, 0x71000000, 0x71100000, 0x1);  // SPM wide

//     // Enable the TLBs
//     writew(1, occ_quad_ctrl + 0x18);
//     writew(1, occ_quad_ctrl + 0x1c);

//     return 0;
// }

// void hero_dev_exe_start(HeroDev *dev) {
//     pr_trace("%p\n", dev);

//     // Set entry-point, bootrom pointer and l3 layout struct pointer
//     writew((uint32_t) 0xc0000000             , occ_soc_ctrl + SCTL_SCRATCH_0_REG_OFFSET);
//     writew((uint32_t)(uint64_t)0             , occ_soc_ctrl + SCTL_SCRATCH_1_REG_OFFSET);

//     fence();

//     clint_set_irq(0x1FF << 1);
// }

int hero_dev_munmap(HeroDev *dev) {
    int err = 0;
    pr_trace("%p\n", dev);
    hero_dev_free_mboxes(dev);
    // close(device_fd);
}