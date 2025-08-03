// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "snrt.h"
extern uint32_t __snax_symtab_start;
extern uint32_t __snax_symtab_end;
uintptr_t l2l_addr[NUM_DEV] = {0};
struct l2_layout *l2l_ptr[NUM_DEV] = {NULL};
uint32_t cmd[NUM_DEV] = {0};
uint32_t data[NUM_DEV] = {0};
uint32_t task_id[NUM_DEV] = {0};
uint32_t offloadArgs[NUM_DEV] = {0};
void (*offloadFn[NUM_DEV])(uint32_t) = {NULL};

typedef struct {
    uint32_t exit_flag;
    uint32_t start_flag;
    uint32_t workers_in_loop;
    uint32_t workers_wfi;
    void (*offloadFn)(uint32_t);
    uint32_t offloadArgs;
    uint32_t fini_count;
} bingo_offload_unit_t;

__thread volatile bingo_offload_unit_t *offload_unit_ptr;
volatile bingo_offload_unit_t *volatile offload_unit_ptr_global;


inline void bingo_wait_worker_wfi(void) {
    uint32_t scratch = offload_unit_ptr->workers_in_loop;
    while (__atomic_load_n(&offload_unit_ptr->workers_wfi, __ATOMIC_RELAXED) != scratch)
        ;
}

inline void bingo_wake_workers(){
    // Guard to wake only if all workers are wfi
    bingo_wait_worker_wfi();
    // Wake the cluster cores. We do this with cluster relative hart IDs and do
    // not wake the last hart since this is the main thread
    uint32_t numcores = snrt_cluster_core_num();
    // (1 << numcores) - 1 will  create a bitmask with all bits set to 1
    // e.g. 4 cores will be (1<<4)-1 = 0b1111
    // We clear the MSB by ~(1 << (numcores-1)) to not wake the manager core
    // e.g. 4 cores will be ~(1 << (4-1)) = 0b0111
    snrt_int_cluster_set((~(1 << (numcores-1))) & ((1 << numcores) - 1));
}


inline uint32_t bingo_get_workers_in_wfi() {
    return __atomic_load_n(&offload_unit_ptr->workers_wfi, __ATOMIC_RELAXED);
}


inline void bingo_worker_wfi(uint32_t cluster_core_idx) {
    __atomic_add_fetch(&offload_unit_ptr->workers_wfi, 1, __ATOMIC_RELAXED);
    snrt_wfi();
    snrt_int_cluster_clr(1 << cluster_core_idx);
    __atomic_add_fetch(&offload_unit_ptr->workers_wfi, -1, __ATOMIC_RELAXED);
}



inline void bingo_offload_init() {
    // Initialize the offload manager
    // This will let the dm to manage the offload
    if (snrt_is_dm_core()) {
        offload_unit_ptr = snrt_l1alloc(sizeof(bingo_offload_unit_t));
        snrt_memset((void *)offload_unit_ptr, 0, sizeof(bingo_offload_unit_t));
        offload_unit_ptr_global = offload_unit_ptr;
    } else {
        while (!offload_unit_ptr_global);
        offload_unit_ptr = offload_unit_ptr_global;
    }
}

inline void bingo_offload_event_loop(uint32_t cluster_core_idx) {
    // This function is called by the non-manager cores

    // count number of workers in loop
    __atomic_add_fetch(&offload_unit_ptr->workers_in_loop, 1, __ATOMIC_RELAXED);
    // Enable the interrupt for the core
    snrt_interrupt_enable(IRQ_M_CLUSTER);
    // Set the core to WFI
    
    while (1)
    {
        if (offload_unit_ptr->exit_flag) {
            snrt_interrupt_disable(IRQ_M_CLUSTER);
            break;
        }
        if (offload_unit_ptr->start_flag) {
            offload_unit_ptr->offloadFn(offload_unit_ptr->offloadArgs);

        }
        // enter wait for interrupt
        __atomic_add_fetch(&offload_unit_ptr->fini_count, 1, __ATOMIC_RELAXED);
        bingo_worker_wfi(cluster_core_idx);
    }
}

inline void bingo_offload_dispatch(void (*offloadFn)(uint32_t), uint32_t offloadArg) {
    // This function is called by the manager core to dispatch the offload
    // function to the workers
    uint32_t scratch;
    bingo_wait_worker_wfi();
    offload_unit_ptr->offloadFn = offloadFn;
    offload_unit_ptr->offloadArgs = offloadArgs;
    offload_unit_ptr->start_flag = 1;
    offload_unit_ptr->exit_flag = 0;
    offload_unit_ptr->fini_count = 0;
    // Wake up the workers ro run the offload function
    bingo_wake_workers();
    // When the workers are done, they will add the fini_count by 1 and set itself to wfi again

    // The Manager core will also execute the offload function
    offloadFn(offloadArgs);
    // When the Manager core is done, check the state of the fini_count
    // if other workers are not done, wait for them to finish
    while (__atomic_load_n(&offload_unit_ptr->fini_count, __ATOMIC_RELAXED) != bingo_get_workers_in_loop())
        ;
    // stop workers from re-executing the task
    offload_unit_ptr->start_flag = 0;
}


uint32_t bingo_offload_manager() {

    bingo_offload_init();
    if(snrt_is_dm_core()) {
        // Wait for all cores to be ready
        while (bingo_get_workers_in_wfi() != (snrt_cluster_core_num() - 1)) {
            // Wait for all cores to be in WFI
        }
    } else {
        // We set other cores to WFI and wait for the manager to start
        bingo_offload_event_loop(snrt_cluster_core_idx());
        return 0;
    }

    // The DM core is the manager, it will handle the offload
    uint32_t core_idx = snrt_global_core_idx();
    // read memory layout from soc_ctrl
    l2l_addr[core_idx] =
        *((volatile uint32_t *)soc_ctrl_mailbox_scratch_addr(core_idx));
    // convert to struct pointer
    l2l_ptr[core_idx] =
        (volatile struct l2_layout *)(uintptr_t)l2l_addr[core_idx];
    g_a2h_rb[core_idx] =
        (struct ring_buf *)(uintptr_t)l2l_ptr[core_idx]->a2h_rb;
    g_a2h_mbox[core_idx] =
        (struct ring_buf *)(uintptr_t)l2l_ptr[core_idx]->a2h_mbox;
    g_h2a_mbox[core_idx] =
        (struct ring_buf *)(uintptr_t)l2l_ptr[core_idx]->h2a_mbox;
    uint32_t start_cycles = 0;
    uint32_t end_cycles = 0;
    uint32_t kernel_cycles = 0;
    // Init the manager (handshake btw host and accelerator is here)
    // gomp_init_offload_manager();

    // FIXME For the momenent we are not using the cmd sended as trigger.
    // It should be used to perform the deactivation of the accelerator,
    // as well as other operations, like local data allocation or movement.
    // FIXME Note that the offload at the moment use several time the
    // mailbox. We should compact the offload descriptor and just sent a
    // pointer to that descriptor.

    while (1) {
        // if (DEBUG_LEVEL_OFFLOAD_MANAGER > 0)
        //   snrt_trace("Waiting for command...\n");

        // (1) Wait for the offload trigger cmd == MBOX_DEVICE_START
        // if(core_idx==0){
        //   printf("[Core0] Waiting for command...\n");
        //   dump_mbox(g_h2a_mbox[core_idx]);
        // }
        mailbox_read(core_idx, (unsigned int *)&cmd[core_idx], 1);
        // if(core_idx==0) {
        //     printf("[Core0] Get the command %d from Host!\n",
        //     cmd[core_idx]);
        // }
        if (MBOX_DEVICE_STOP == cmd[core_idx]) {
            // if (DEBUG_LEVEL_OFFLOAD_MANAGER > 0)
            //   snrt_trace("Got MBOX_DEVICE_STOP from host, stopping
            //   execution now.\n");
            // Set the exit flag to stop the workers
            offload_unit_ptr->exit_flag = 1;
            bingo_wake_workers();
            break;
        } else if (MBOX_DEVICE_LOGLVL == cmd[core_idx]) {
            // if (DEBUG_LEVEL_OFFLOAD_MANAGER > 0)
            //   snrt_trace("Got command 0x%x, setting log level.\n", cmd);
            mailbox_read(core_idx, (unsigned int *)&data[core_idx], 1);
            // snrt_debug_set_loglevel(data);
            continue;
        } else if (MBOX_DEVICE_START != cmd[core_idx]) {
            // if (DEBUG_LEVEL_OFFLOAD_MANAGER > 0)
            //   snrt_trace("Got unexpected command 0x%x, stopping execution
            //   now.\n", cmd);
            break;
        }
        // (2) The host sends the task id
        mailbox_read(core_idx, (uint32_t *)&task_id[core_idx], 1);
        // if(core_idx==0) printf("[Core0] Get the task_id %d from Host!\n",
        // task_id[core_idx]);
        //  (2) The host sends through the mailbox the pointer to the
        //  function that should be executed on the accelerator.

        mailbox_read(core_idx, (uint32_t *)&offloadFn[core_idx], 1);
        // if(core_idx==0) printf("[Core0] Get the offloadFn 0x%x from
        // Host!\n", offloadFn[core_idx]); if (DEBUG_LEVEL_OFFLOAD_MANAGER >
        // 0)
        //   snrt_trace("tgt_fn @ 0x%x\n", (unsigned int)offloadFn);

        // (3) The host sends through the mailbox the pointer to the
        // arguments that should be used.
        mailbox_read(core_idx, (uint32_t *)&offloadArgs[core_idx], 1);
        // if(core_idx==0) printf("[Core0] Get the offloadArgs 0x%x from
        // Host!\n", offloadArgs[core_idx]); if (DEBUG_LEVEL_OFFLOAD_MANAGER
        // > 0)
        //   snrt_trace("tgt_vars @ 0x%x\n", (unsigned int)offloadArgs);

        // if (DEBUG_LEVEL_OFFLOAD_MANAGER > 0)
        //   snrt_trace("nbOffloadRabMissHandlers %d/%d\n",
        //   nbOffloadRabMissHandlers, active_pe);
        printf("[Core %d] Task %d start running\n", core_idx,
               task_id[core_idx]);
        // if (DEBUG_LEVEL_OFFLOAD_MANAGER > 0)
        //   snrt_trace("begin offloading\n");
        //  reset_timer();
        //  start_timer();

        // for (unsigned i = 0; i < 16; i += 2) {
        //   snrt_trace(" %2d: 0x%08x = ... ; %2d: 0x%08x = ...\n", i,
        //   ((uint32_t *)offloadArgs)[i],
        //              /* *((uint32_t *)(((uint32_t *)offloadArgs)[i])) ,*/
        //              i + 1,
        //              ((uint32_t *)offloadArgs)[i + 1] /*, *((uint32_t
        //              *)(((uint32_t *)offloadArgs)[i + 1]))*/ );
        // }

        // (5) Execute the offloaded function.
        start_cycles = snrt_mcycle();
        bingo_offload_dispatch(offloadFn[core_idx], offloadArgs[core_idx]);
        

        // if (DEBUG_LEVEL_OFFLOAD_MANAGER > 0)
        //   snrt_trace("end offloading\n");

        // (6) Report EOC and profiling
        // snrt_info("cycles: %d\r\n", cycles);
        end_cycles = snrt_mcycle();
        kernel_cycles = end_cycles - start_cycles;
        printf("[Core %d] Task %d finish with CC=%d \n", core_idx,
               task_id[core_idx], kernel_cycles);
        // return
        // (1) Done
        // (2) Task id
        // (3) Cycles
        mailbox_write(core_idx, MBOX_DEVICE_DONE);
        printf("[Core %d] Send MBOX_DEVICE_DONE=%d\n", core_idx,
               MBOX_DEVICE_DONE);
        mailbox_write(core_idx, task_id[core_idx]);
        printf("[Core %d] Send task_id=%d\n", core_idx, task_id[core_idx]);
        mailbox_write(core_idx, kernel_cycles);
        printf("[Core %d] Send kernel_cycles=%d\n", core_idx, kernel_cycles);
        // if (DEBUG_LEVEL_OFFLOAD_MANAGER > 0)
        //   snrt_trace("Kernel execution time [Snitch cycles] = %d\n",
        //   cycles);
    }

    return 0;
}