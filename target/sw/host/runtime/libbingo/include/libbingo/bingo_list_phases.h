// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// The host half of the paged task lists (shared/runtime/bingo_paged_lists.h): the mini compiler's generated code
// includes this file only when the DFG has list_phases, so no other build compiles it.

#pragma once

#include "libbingo/bingo_api.h"
#include "bingo_paged_lists.h"

// One cluster's part of a chiplet's generated tables, in words
typedef struct {
    uint32_t hdr_off;   // its header in the header table
    uint32_t hdr_words;
    uint32_t p0_off;    // its phase-0 entries in the list table
    uint32_t p0_words;
} bingo_lp_cluster_t;

// The init of the paged lists, in place of bingo_hw_scheduler_init_gid_lists: each cluster's header gets the
// kernel addresses and is copied into its L1, its phase-0 entries into its window 0 (the later phases come by the
// graph's list loads), and the quad ctrl points the cluster at its header. Then the start bingo_api.c's inits
// share: the power manager, the first launch, the release of the clusters.
static inline void bingo_hw_scheduler_init_list_phases(uint32_t *hdrs, const bingo_lp_cluster_t *cl,
                                                       const uint64_t *win0, const uint32_t *lists,
                                                       const uint32_t *fn_addrs, uint32_t n_fn,
                                                       uint64_t bingo_hw_scheduler_task_desc_list_base,
                                                       uint32_t bingo_hw_scheduler_num_task_desc) {
    const uint8_t chip = get_current_chip_id();
    writew(2, (uintptr_t)chiplet_addr_transform((uint64_t)soc_ctrl_kernel_tab_scratch_addr(3)));
    for (uint32_t i = 0; i < N_CLUSTERS_PER_CHIPLET; i++) {
        uint32_t *h = hdrs + cl[i].hdr_off;
        if (h[0] != BINGO_GID_LIST_PAGED_MAGIC || h[3] > n_fn) host_abort(BINGO_EXIT_LAYOUT);
        uint32_t *fn = h + BINGO_PAGED_FN_OFF(h[1], h[2]);
        for (uint32_t j = 0; j < h[3]; j++) fn[j] = fn_addrs[j];
    }
    // the DMA reads the headers from memory, after the host's stores of the kernel addresses
    asm volatile("fence" ::: "memory");
    for (uint32_t i = 0; i < N_CLUSTERS_PER_CHIPLET; i++) {
        const uint32_t bytes = cl[i].hdr_words * (uint32_t)sizeof(uint32_t);
        uint64_t ptr_hdr = bingo_l1_alloc(chip, i, bytes);
        sys_dma_blk_memcpy(chip, ptr_hdr,
                           (uint64_t)chiplet_addr_transform_full(chip, (uint64_t)(uintptr_t)(hdrs + cl[i].hdr_off)),
                           bytes);
        if (cl[i].p0_words) {
            sys_dma_blk_memcpy(chip, win0[i],
                               (uint64_t)chiplet_addr_transform_full(chip, (uint64_t)(uintptr_t)(lists + cl[i].p0_off)),
                               cl[i].p0_words * (uint32_t)sizeof(uint32_t));
        }
        writew((uint32_t)win0[i], (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_arg_ptr_addr(i)));
        writew((uint32_t)win0[i], (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_kernel_ptr_addr(i)));
        writew((uint32_t)ptr_hdr, (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_global_id_to_dev_id_addr(i)));
    }
    bingo_hw_scheduler_init_pm();
    bingo_hw_scheduler_launch(bingo_hw_scheduler_task_desc_list_base, bingo_hw_scheduler_num_task_desc);
    // Tell the device that the host init is done and make sure the write has landed.
    writew(1, (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_host_init_done_addr()));
    asm volatile("fence" ::: "memory");
    while (readw((uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_host_init_done_addr())) == 0) {
    }
    // The read-back above proves the quadrant is programmed; now release the clusters, which poll this word in the
    // L2 comm buffer (device bingo.h).
    ((volatile comm_buffer_t *)(uintptr_t)bingo_get_l2_comm_buffer(get_current_chip_id()))->host_init_done = 1;
    asm volatile("fence" ::: "memory");
}
