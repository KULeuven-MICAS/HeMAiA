// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

//================================================================================
// Debug
//================================================================================

#ifdef BINGO_DEBUG_LEVEL
#include "printf.h"
#define _BINGO_PRINTF(...)             \
    if (1) {                        \
        printf("[bingo] "__VA_ARGS__); \
    }
#define BINGO_PRINTF(d, ...)        \
    if (BINGO_DEBUG_LEVEL >= d) {   \
        _BINGO_PRINTF(__VA_ARGS__); \
    }
#else
#define BINGO_PRINTF(d, ...)
#endif

inline bingo_offload_unit_t *get_bingo_offload_unit() {
    return (bingo_offload_unit_t *)&(cls()->bingo_offload_unit);
}


//================================================================================
// Data
//================================================================================
/**
 * @brief Pointer to the snax kernel table
 *
 */
extern uint32_t __snax_symtab_start;
extern uint32_t __snax_symtab_end;

//================================================================================
// Functions
//================================================================================

inline void bingo_wait_worker_wfi() {
    uint32_t scratch = get_bingo_offload_unit()->workers_in_loop;
    while (__atomic_load_n(&get_bingo_offload_unit()->workers_wfi, __ATOMIC_RELAXED) != scratch)
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

inline void bingo_worker_wfi(uint32_t cluster_core_idx) {
    __atomic_add_fetch(&get_bingo_offload_unit()->workers_wfi, 1, __ATOMIC_RELAXED);
    snrt_wfi();
    snrt_int_cluster_clr(1 << cluster_core_idx);
    __atomic_add_fetch(&get_bingo_offload_unit()->workers_wfi, -1, __ATOMIC_RELAXED);
}


/**
 * @brief Debugging info to printf
 * @details
 */
inline void bingo_print_status() {
    BINGO_PRINTF(0, "workers_in_loop=%d\n", get_bingo_offload_unit()->workers_in_loop);
}

/**
 * Getters
 */
inline uint32_t bingo_get_workers_in_loop() {
    return __atomic_load_n(&get_bingo_offload_unit()->workers_in_loop, __ATOMIC_RELAXED);
}
inline uint32_t bingo_get_workers_in_wfi() {
    return __atomic_load_n(&get_bingo_offload_unit()->workers_wfi, __ATOMIC_RELAXED);
}


/**
 * @brief Initialize the bingo offload unit
 */
inline void bingo_offload_init() {
    // Initialize the offload manager
    // This will let the dm to manage the offload
    if (snrt_is_dm_core()) {
        snrt_memset((void *)get_bingo_offload_unit(), 0, sizeof(bingo_offload_unit_t));
    }
    // Make sure the bingo offload unit is reset by the DM core
    snrt_cluster_hw_barrier();
}

/**
 * @brief send all workers in loop to exit()
 * @param core_idx cluster-local core index
 */
inline void bingo_offload_exit() {
    // make sure queue is empty
    // set exit flag and wake cores
    bingo_wait_worker_wfi();
    get_bingo_offload_unit()->exit_flag = 1;
    bingo_wake_workers();
}

/**
 * @brief Enter the event unit loop, never exits
 *
 * @param cluster_core_idx cluster-local core index
 */
inline void bingo_offload_event_loop(uint32_t cluster_core_idx) {
    // This function is called by the non-manager cores
    // count number of workers in loop
    __atomic_add_fetch(&get_bingo_offload_unit()->workers_in_loop, 1, __ATOMIC_RELAXED);
    // Enable the interrupt for the core
    snrt_interrupt_enable(IRQ_M_CLUSTER);
    // Set the core to WFI
    
    while (1)
    {
        if (get_bingo_offload_unit()->exit_flag) {
            // If exit flag is set, we stop the event loop
            snrt_interrupt_disable(IRQ_M_CLUSTER);
            break;
        }
        if (get_bingo_offload_unit()->start_flag) {
            // If start flag is enabled, we execute the offload function
            get_bingo_offload_unit()->offloadFn(get_bingo_offload_unit()->offloadArgs);

        }
        // enter wait for interrupt
        __atomic_add_fetch(&get_bingo_offload_unit()->fini_count, 1, __ATOMIC_RELAXED);
        bingo_worker_wfi(cluster_core_idx);
    }
}

/**
 * @brief Set function to execute by each cores
 * @details
 *
 * @param offloadFn pointer to worker function to be executed
 * @param offloadArg pointer to function arguments
 */
inline void bingo_offload_dispatch(uint32_t (*offloadFn)(uint32_t), uint32_t offloadArgs) {
    // This function is called by the manager core to dispatch the offload
    // function to the workers
    uint32_t scratch;
    bingo_wait_worker_wfi();
    get_bingo_offload_unit()->offloadFn = offloadFn;
    get_bingo_offload_unit()->offloadArgs = offloadArgs;
    get_bingo_offload_unit()->start_flag = 1;
    get_bingo_offload_unit()->exit_flag = 0;
    get_bingo_offload_unit()->fini_count = 0;
    // Wake up the workers ro run the offload function
    bingo_wake_workers();
    // When the workers are done, they will add the fini_count by 1 and set itself to wfi again

    // The Manager core will also execute the offload function
    offloadFn(offloadArgs);
    // When the Manager core is done, check the state of the fini_count
    // if other workers are not done, wait for them to finish
    while (__atomic_load_n(&get_bingo_offload_unit()->fini_count, __ATOMIC_RELAXED) != bingo_get_workers_in_loop())
        ;
    // stop workers from re-executing the task
    get_bingo_offload_unit()->start_flag = 0;
}

inline uint32_t bingo_offload_manager(){
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

    while(1){
        // (1) Wait for the offload trigger cmd == MBOX_DEVICE_START
        h2c_mailbox_read((uint32_t *)&(get_bingo_offload_unit()->cmd));
        // printf("[Cluster %d] Received command: 0x%x\n", snrt_cluster_idx(), bingo_offload_unit_ptr->cmd);
        if (MBOX_DEVICE_STOP == (get_bingo_offload_unit()->cmd)) {
            // Got MBOX_DEVICE_STOP from host, stopping execution now.
            // Set the exit flag to stop the workers
            bingo_offload_exit();
            break;
        } else if (MBOX_DEVICE_START != (get_bingo_offload_unit()->cmd)) {
            // Got unexpected command 0x%x, stopping execution now.
            return -1;
        }
        // (2) The host sends the task id
        h2c_mailbox_read((uint32_t *)&(get_bingo_offload_unit()->task_id));
        // printf("[Cluster %d] Received task id: 0x%x\n", snrt_cluster_idx(), get_bingo_offload_unit()->task_id);
        // (3) The host sends through the mailbox the pointer to the function that should be executed on the accelerator.
        h2c_mailbox_read((uint32_t *)&(get_bingo_offload_unit()->offloadFn));
        // printf("[Cluster %d] Received function pointer: 0x%x\n", snrt_cluster_idx(), (uint32_t)(get_bingo_offload_unit()->offloadFn));
        // (4) The host sends through the mailbox the pointer to the arguments that should be used.
        h2c_mailbox_read((uint32_t *)&(get_bingo_offload_unit()->offloadArgs));
        // printf("[Cluster %d] Received function args: 0x%x\n", snrt_cluster_idx(), get_bingo_offload_unit()->offloadArgs);
        // Bookkeeping the start cycles
        get_bingo_offload_unit()->start_cycles = snrt_mcycle();
        // (5) The manager core will execute the offload function
        bingo_offload_dispatch(get_bingo_offload_unit()->offloadFn, get_bingo_offload_unit()->offloadArgs);
        // Bookkeeping the end cycles
        get_bingo_offload_unit()->end_cycles = snrt_mcycle();
        // printf("[Cluster %d] Task %d finish with CC=%d \n", 
                // snrt_cluster_idx(),
                // get_bingo_offload_unit()->task_id,
                // get_bingo_offload_unit()->end_cycles-get_bingo_offload_unit()->start_cycles);
        // (6) return the result through the mailbox to the host
        bingo_c2h_msg_fields_t c2h_msg;
        c2h_msg.cluster_id = snrt_cluster_idx();
        c2h_msg.flag = MBOX_DEVICE_DONE;
        c2h_msg.task_id = (uint16_t)get_bingo_offload_unit()->task_id;
        c2h_msg.reserved = 0;
        c2h_mailbox_write(bingo_c2h_msg_encode(c2h_msg));
    }
    return 0;
}