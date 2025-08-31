// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Cyril Koenig <cykoenig@iis.ee.ethz.ch>
#include "libhero/debug.h"
#include "libhero/hero_api.h"
#include "libhero/io.h"
#include "libhero/ringbuf.h"
//////////////////////////////
///// MEMORY MANAGEMENT //////
//////////////////////////////

// Stucture containing the *device* L2 and L3 allocator
struct O1HeapInstance *l2_heap_manager, *l2_mailbox_manager, *l3_heap_manager;

uintptr_t l2_mailbox_start_phy, l2_mailbox_start_virt;
size_t l2_mailbox_size;

uintptr_t l2_heap_start_phy, l2_heap_start_virt;
size_t l2_heap_size;

uintptr_t l3_heap_start_phy, l3_heap_start_virt;
size_t l3_heap_size;

//////////////////////////////
///// MAILBOXES         //////
//////////////////////////////

int hero_dev_alloc_mboxes(HeroDev *dev) {
    pr_trace("%s default\n", __func__);

    // Alloc ringbuf structure
    dev->mboxes.h2a_mbox = (struct ring_buf *)hero_dev_l2_mailbox_malloc(dev, sizeof(struct ring_buf), &dev->mboxes.h2a_mbox_mem.p_addr);
    // Alloc data array for ringbuf
    dev->mboxes.h2a_mbox->data_v = hero_dev_l2_mailbox_malloc(dev, sizeof(uint32_t)*16, (uintptr_t *)&dev->mboxes.h2a_mbox->data_p);

    // Same for the acc2host mailbox
    dev->mboxes.a2h_mbox = (struct ring_buf *)hero_dev_l2_mailbox_malloc(dev, sizeof(struct ring_buf), &dev->mboxes.a2h_mbox_mem.p_addr);
    dev->mboxes.a2h_mbox->data_v = hero_dev_l2_mailbox_malloc(dev, sizeof(uint32_t)*16, (uintptr_t *)&dev->mboxes.a2h_mbox->data_p);

    // Same for the acc2host ringbug mailbox
    dev->mboxes.rb_mbox = (struct ring_buf *)hero_dev_l2_mailbox_malloc(dev, sizeof(struct ring_buf), &dev->mboxes.rb_mbox_mem.p_addr);
    dev->mboxes.rb_mbox->data_v = hero_dev_l2_mailbox_malloc(dev, sizeof(uint32_t)*16, (uintptr_t *)&dev->mboxes.rb_mbox->data_p);

    if(!dev->mboxes.a2h_mbox || !dev->mboxes.h2a_mbox || !dev->mboxes.rb_mbox) {
        return -1;
    }

    rb_init(dev->mboxes.a2h_mbox, 16, sizeof(uint32_t));
    rb_init(dev->mboxes.h2a_mbox, 16, sizeof(uint32_t));
    rb_init(dev->mboxes.rb_mbox, 16, sizeof(uint32_t));

    return 0;
}


void hero_dev_l2_mailbox_free(HeroDev *dev, uintptr_t v_addr, uintptr_t p_addr) {
    pr_trace("%p - %p\n", l2_mailbox_manager, v_addr);
    o1heapFree(l2_mailbox_manager, (void*)v_addr);
}

void hero_dev_l2_free(HeroDev *dev, uintptr_t v_addr, uintptr_t p_addr) {
    pr_trace("%p - %p\n", l2_heap_manager, v_addr);
    o1heapFree(l2_heap_manager, (void*)v_addr);
}

void hero_dev_l3_free(HeroDev *dev, uintptr_t v_addr, uintptr_t p_addr) {
    pr_trace("%p - %p\n", l3_heap_manager, v_addr);
    o1heapFree(l3_heap_manager, (void*)v_addr);
    // pr_warn("%s unimplemented\n", __func__);
    // return 0;    
}

int hero_dev_free_mboxes(HeroDev *dev) {
    pr_trace("%s default\n", __func__);

    hero_dev_l2_mailbox_free(dev, (uintptr_t)dev->mboxes.h2a_mbox->data_v, 0);
    hero_dev_l2_mailbox_free(dev, (uintptr_t)dev->mboxes.h2a_mbox, 0);

    hero_dev_l2_mailbox_free(dev, (uintptr_t)dev->mboxes.a2h_mbox->data_v, 0);
    hero_dev_l2_mailbox_free(dev, (uintptr_t)dev->mboxes.a2h_mbox, 0);

    hero_dev_l2_mailbox_free(dev, (uintptr_t)dev->mboxes.rb_mbox->data_v, 0);
    hero_dev_l2_mailbox_free(dev, (uintptr_t)dev->mboxes.rb_mbox, 0);

    return 0;
}

int hero_dev_mbox_try_read(const HeroDev *dev, uint32_t *buffer){
    pr_trace("%s default\n", __func__);
    int ret;
    // If this region is cached and no cache coherency, need a fence
    // asm volatile("fence" : : : "memory");
    ret = rb_host_get(dev->mboxes.a2h_mbox, buffer) == 0 ? 1 : 0;
    return ret;
}


int hero_dev_mbox_read(const HeroDev *dev, uint32_t *buffer, size_t n_words) {
    pr_trace("%s default\n", __func__);
    int ret, retry = 0;
    while (n_words--) {
        do {
        // If this region is cached and no cache coherency, need a fence
        // asm volatile("fence" : : : "memory");
            ret = rb_host_get(dev->mboxes.a2h_mbox, &buffer[n_words]);
            if (ret) {
                if (++retry == 10)
                    pr_warn("high retry on mbox read()\n");
                // For now avoid sleep that creates context switch
                for(int i = 0; i < HOST_MBOX_CYCLES; i++) {
                    asm volatile ("nop");
                }
            }
        } while (ret);
    }

    return 0;
}

int hero_dev_mbox_write(HeroDev *dev, uint32_t word) {
    pr_trace("%s Mbox write to dev %d @ h2a_mbox = %x with %x\n", __func__, dev->dev_id, dev->mboxes.h2a_mbox, word);
    int ret, retry = 0;
    do {
        // If this region is cached and no cache coherency, need a fence
        // asm volatile("fence" : : : "memory");
        ret = rb_host_put(dev->mboxes.h2a_mbox, &word);
        if (ret) {
            if (++retry == 100)
                pr_warn("high retry on mbox write()\n");
            // For now avoid sleep that creates context switch
            for(int i = 0; i < HOST_MBOX_CYCLES; i++) {
                asm volatile ("nop");
            }
        }
    } while (ret);
    return ret;
}

//////////////////////////////
///// INIT              //////
//////////////////////////////

__attribute__((weak)) int hero_dev_mmap(HeroDev *dev) {
    pr_warn("%s unimplemented\n", __func__);
    return 0;
}

__attribute__((weak)) int hero_dev_munmap(HeroDev *dev) {
    pr_warn("%s unimplemented\n", __func__);
    return 0;
}

__attribute__((weak)) int hero_dev_init(HeroDev *dev) {

    pr_warn("%s unimplemented\n", __func__);
}

__attribute__((weak)) void hero_dev_reset(HeroDev *dev, bool full) {
    pr_warn("%s unimplemented\n", __func__);
}

//////////////////////////////
///// EXECUTION         //////
//////////////////////////////

int hero_dev_load_bin_from_mem(HeroDev *dev, void *ptr, uint32_t size) {
    pr_warn("%s unimplemented\n", __func__);
    return 0;
}

__attribute__((weak)) void hero_dev_exe_start(HeroDev *dev) {
    pr_warn("%s unimplemented\n", __func__);
    while(1) {}
}

void hero_dev_exe_stop(HeroDev *dev) {
    pr_warn("%s unimplemented\n", __func__);
}

int hero_dev_exe_wait(const HeroDev *dev, uint32_t timeout_s) {
    pr_warn("%s unimplemented\n", __func__);
    return 0;
}

//////////////////////////////
///// MEMORY MANAGEMENT //////
//////////////////////////////


int hero_dev_l2_init() {
    pr_trace("%s default\n", __func__);
    // Initialize L2 mailbox manager in the beginning of the L2
    if (!l2_mailbox_manager) {
        if(!l2_mailbox_start_phy || !l2_mailbox_size) {
            pr_error("%s does not know where to put the heap manager\n", __func__);
            return -1;
        }
        pr_trace("Initializing o1heap at %p (%p) size %x\n", (void *) l2_mailbox_start_phy, (void *) l2_mailbox_start_virt, l2_mailbox_size);
        l2_mailbox_manager = o1heapInit((void *) l2_mailbox_start_virt, l2_mailbox_size);
        if (l2_mailbox_manager == NULL) {
            pr_error("Failed to initialize L2 mailbox manager.\n");
            return -1;
        } else {
            pr_debug("Allocated L2 mailbox manager at %p.\n", l2_mailbox_manager);
        }
    } else {
        pr_warn("%s is already initialized\n", __func__);
    }


    // Initialize L2 heap manager in the middle of the reserved memory range
    if (!l2_heap_manager) {
        if(!l2_heap_start_phy || !l2_heap_size) {
            pr_error("%s does not know where to put the heap manager\n", __func__);
            return -1;
        }
        pr_trace("Initializing o1heap at %p (%p) size %x\n", (void *) l2_heap_start_phy, (void *) l2_heap_start_virt, l2_heap_size);
        l2_heap_manager = o1heapInit((void *) l2_heap_start_virt, l2_heap_size);
        if (l2_heap_manager == NULL) {
            pr_error("Failed to initialize L2 heap manager.\n");
            return -1;
        } else {
            pr_debug("Allocated L2 heap manager at %p.\n", l2_heap_manager);
        }
    } else {
        pr_warn("%s is already initialized\n", __func__);
    }
    return 0;
}

int hero_dev_l3_init() {
    pr_trace("%s default\n", __func__);
    // Initialize L2 heap manager in the middle of the reserved memory range
    if (!l3_heap_manager) {
        if(!l3_heap_start_phy || !l3_heap_size) {
            pr_error("%s does not know where to put the heap manager\n", __func__);
            return -1;
        }
        pr_trace("Initializing o1heap at %p (%p) size %x\n", (void *)(l3_heap_start_phy), (void *)(l3_heap_start_virt), l3_heap_size);
        l3_heap_manager = o1heapInit((void *)(l3_heap_start_virt), l3_heap_size);
        if (l3_heap_manager == NULL) {
            pr_error("Failed to initialize L3 heap manager.\n");
            return -1;
        } else {
            pr_debug("Allocated L3 heap manager at %p.\n", l3_heap_manager);
        }
    } else {
        pr_warn("%s is already initialized\n", __func__);
    }
    return 0;
}

uintptr_t hero_dev_l2_mailbox_malloc(HeroDev *dev, uint32_t size_b, uintptr_t *p_addr) {
    pr_trace("%p %x\n", l2_mailbox_manager, size_b);
    void *result = o1heapAllocate(l2_mailbox_manager, size_b);
    *p_addr = (uintptr_t)((void *) result - l2_mailbox_start_virt + l2_mailbox_start_phy);
    pr_trace("%s Allocated %u bytes at %x (%p)\n", __func__, size_b, (void *) result - l2_mailbox_start_virt + l2_mailbox_start_phy, result);
    return (uintptr_t)result;
}


uintptr_t hero_dev_l2_malloc(HeroDev *dev, uint32_t size_b, uintptr_t *p_addr) {
    pr_trace("%p %x\n", l2_heap_manager, size_b);
    void *result = o1heapAllocate(l2_heap_manager, size_b);
    *p_addr = (uintptr_t)((void *) result - l2_heap_start_virt + l2_heap_start_phy);
    pr_trace("%s Allocated %u bytes at %x (%p)\n", __func__, size_b, (void *) result - l2_heap_start_virt + l2_heap_start_phy, result);
    return (uintptr_t)result;
}

uintptr_t hero_dev_l3_malloc(HeroDev *dev, uint32_t size_b, uintptr_t *p_addr) {
    pr_trace("%s default\n", __func__);
    void *result = o1heapAllocate(l3_heap_manager, size_b);
    *p_addr = (uintptr_t)((void *) result - l3_heap_start_virt + l3_heap_start_phy);
    pr_trace("%s Allocated %u bytes at %x (%p)\n", __func__, size_b, (void *) result - l3_heap_start_virt + l3_heap_start_phy, result);
    return (uintptr_t)result;
}



int hero_dev_dma_xfer(const HeroDev *dev, uintptr_t addr_l3,
                      uintptr_t addr_pulp, size_t size_b, bool host_read) {
    pr_warn("%s unimplemented\n", __func__);
    return 0;
}

__attribute__((weak)) uintptr_t hero_host_l3_malloc(HeroDev *dev, uint32_t size_b, uintptr_t *p_addr) {
    pr_warn("%s unimplemented\n", __func__);
    return (uintptr_t)NULL;
}

__attribute__((weak)) uintptr_t hero_iommu_map_virt(HeroDev *dev, uint32_t size_b, void *v_addr) {
    pr_warn("%s unimplemented\n", __func__);
    return (uintptr_t)0;
}

__attribute__((weak)) int hero_iommu_map_virt_to_phys(HeroDev *dev, uint32_t size_b, void *v_addr, uintptr_t p_addr) {
    pr_warn("%s unimplemented\n", __func__);
    return 0;
}