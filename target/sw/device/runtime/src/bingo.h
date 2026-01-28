// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>


#define BINGO_RET_SUCC 0
#define BINGO_RET_EXIT 1
#define BINGO_RET_FAIL 2

//================================================================================
// Debug
//================================================================================

#ifdef BINGO_DEBUG_LEVEL
#define _BINGO_PRINTF(...)             \
    if (1) {                        \
        printf_safe("[Bingo Dev] "__VA_ARGS__); \
    }
#define BINGO_PRINTF(d, ...)        \
    if (BINGO_DEBUG_LEVEL >= d) {   \
        _BINGO_PRINTF(__VA_ARGS__); \
    }
#else
#define BINGO_PRINTF(d, ...)
#endif

inline bingo_sw_offload_unit_t *get_bingo_sw_offload_unit() {
    return (bingo_sw_offload_unit_t *)&(cls()->bingo_sw_offload_unit);
}

inline bingo_hw_offload_unit_t *get_bingo_hw_offload_unit() {
    return (bingo_hw_offload_unit_t *)&(cls()->bingo_hw_offload_unit);
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

inline void bingo_sw_offload_wait_worker_wfi() {
    uint32_t scratch = get_bingo_sw_offload_unit()->workers_in_loop;
    while (__atomic_load_n(&get_bingo_sw_offload_unit()->workers_wfi, __ATOMIC_RELAXED) != scratch)
        ;
}

inline void bingo_sw_offload_wake_workers(){
    // Guard to wake only if all workers are wfi
    bingo_sw_offload_wait_worker_wfi();
    // Wake the cluster cores. We do this with cluster relative hart IDs and do
    // not wake the last hart since this is the main thread
    uint32_t numcores = snrt_cluster_core_num();
    // (1 << numcores) - 1 will  create a bitmask with all bits set to 1
    // e.g. 4 cores will be (1<<4)-1 = 0b1111
    // We clear the MSB by ~(1 << (numcores-1)) to not wake the manager core
    // e.g. 4 cores will be ~(1 << (4-1)) = 0b0111
    snrt_int_cluster_set((~(1 << (numcores-1))) & ((1 << numcores) - 1));
}

inline void bingo_sw_offload_worker_wfi(uint32_t cluster_core_idx) {
    __atomic_add_fetch(&get_bingo_sw_offload_unit()->workers_wfi, 1, __ATOMIC_RELAXED);
    snrt_wfi();
    snrt_int_cluster_clr(1 << cluster_core_idx);
    __atomic_add_fetch(&get_bingo_sw_offload_unit()->workers_wfi, -1, __ATOMIC_RELAXED);
}


/**
 * @brief Debugging info to printf
 * @details
 */
inline void bingo_sw_offload_print_status() {
    BINGO_PRINTF(1, "workers_in_loop=%d\n", get_bingo_sw_offload_unit()->workers_in_loop);
}

/**
 * Getters
 */
inline uint32_t bingo_sw_offload_get_workers_in_loop() {
    return __atomic_load_n(&get_bingo_sw_offload_unit()->workers_in_loop, __ATOMIC_RELAXED);
}
inline uint32_t bingo_sw_offload_get_workers_in_wfi() {
    return __atomic_load_n(&get_bingo_sw_offload_unit()->workers_wfi, __ATOMIC_RELAXED);
}


/**
 * @brief Initialize the bingo offload unit
 */
inline void bingo_sw_offload_init() {
    // Initialize the offload manager
    // This will let the dm to manage the offload
    if (snrt_is_dm_core()) {
        snrt_memset((void *)get_bingo_sw_offload_unit(), 0, sizeof(bingo_sw_offload_unit_t));
    }
    // Make sure the bingo offload unit is reset by the DM core
    snrt_cluster_hw_barrier();
}

/**
 * @brief send all workers in loop to exit()
 * @param core_idx cluster-local core index
 */
inline void bingo_sw_offload_exit() {
    // make sure queue is empty
    // set exit flag and wake cores
    bingo_sw_offload_wait_worker_wfi();
    get_bingo_sw_offload_unit()->exit_flag = 1;
    bingo_sw_offload_wake_workers();
}

/**
 * @brief Enter the event unit loop, never exits
 *
 * @param cluster_core_idx cluster-local core index
 */
inline void bingo_sw_offload_event_loop(uint32_t cluster_core_idx) {
    // This function is called by the non-manager cores
    // count number of workers in loop
    __atomic_add_fetch(&get_bingo_sw_offload_unit()->workers_in_loop, 1, __ATOMIC_RELAXED);
    // Enable the interrupt for the core
    snrt_interrupt_enable(IRQ_M_CLUSTER);
    // Set the core to WFI
    
    while (1)
    {
        if (get_bingo_sw_offload_unit()->exit_flag) {
            // If exit flag is set, we stop the event loop
            snrt_interrupt_disable(IRQ_M_CLUSTER);
            break;
        }
        if (get_bingo_sw_offload_unit()->start_flag) {
            // If start flag is enabled, we execute the offload function
            get_bingo_sw_offload_unit()->offloadFn(get_bingo_sw_offload_unit()->offloadArgs);

        }
        // enter wait for interrupt
        __atomic_add_fetch(&get_bingo_sw_offload_unit()->fini_count, 1, __ATOMIC_RELAXED);
        bingo_sw_offload_worker_wfi(cluster_core_idx);
    }
}

/**
 * @brief Set function to execute by each cores
 * @details
 *
 * @param offloadFn pointer to worker function to be executed
 * @param offloadArg pointer to function arguments
 */
inline void bingo_sw_offload_dispatch(uint32_t (*offloadFn)(uint32_t), uint32_t offloadArgs) {
    // This function is called by the manager core to dispatch the offload
    // function to the workers
    uint32_t scratch;
    bingo_sw_offload_wait_worker_wfi();
    get_bingo_sw_offload_unit()->offloadFn = offloadFn;
    get_bingo_sw_offload_unit()->offloadArgs = offloadArgs;
    get_bingo_sw_offload_unit()->start_flag = 1;
    get_bingo_sw_offload_unit()->exit_flag = 0;
    get_bingo_sw_offload_unit()->fini_count = 0;
    // Wake up the workers ro run the offload function
    bingo_sw_offload_wake_workers();
    // When the workers are done, they will add the fini_count by 1 and set itself to wfi again

    // The Manager core will also execute the offload function
    offloadFn(offloadArgs);
    // When the Manager core is done, check the state of the fini_count
    // if other workers are not done, wait for them to finish
    while (__atomic_load_n(&get_bingo_sw_offload_unit()->fini_count, __ATOMIC_RELAXED) != bingo_sw_offload_get_workers_in_loop())
        ;
    // stop workers from re-executing the task
    get_bingo_sw_offload_unit()->start_flag = 0;
}

inline int32_t bingo_sw_offload_manager(){
    bingo_sw_offload_init();

    if(snrt_is_dm_core()) {
        // Wait for all cores to be ready
        while (bingo_sw_offload_get_workers_in_wfi() != (snrt_cluster_core_num() - 1)) {
            // Wait for all cores to be in WFI
        }
    } else {
        // We set other cores to WFI and wait for the manager to start
        bingo_sw_offload_event_loop(snrt_cluster_core_idx());
        return 0;
    }

    while(1){
        // (1) Wait for the offload trigger cmd == MBOX_DEVICE_START
        h2c_mailbox_read((uint32_t *)&(get_bingo_sw_offload_unit()->cmd));
        // printf("[Cluster %d] Received command: 0x%x\n", snrt_cluster_idx(), bingo_sw_offload_unit_ptr->cmd);
        if (MBOX_DEVICE_STOP == (get_bingo_sw_offload_unit()->cmd)) {
            // Got MBOX_DEVICE_STOP from host, stopping execution now.
            // Set the exit flag to stop the workers
            bingo_sw_offload_exit();
            break;
        } else if (MBOX_DEVICE_START != (get_bingo_sw_offload_unit()->cmd)) {
            // Got unexpected command 0x%x, stopping execution now.
            return -1;
        }
        // (2) The host sends the task id
        h2c_mailbox_read((uint32_t *)&(get_bingo_sw_offload_unit()->task_id));
        // printf("[Cluster %d] Received task id: 0x%x\n", snrt_cluster_idx(), get_bingo_sw_offload_unit()->task_id);
        // (3) The host sends through the mailbox the pointer to the function that should be executed on the accelerator.
        h2c_mailbox_read((uint32_t *)&(get_bingo_sw_offload_unit()->offloadFn));
        // printf("[Cluster %d] Received function pointer: 0x%x\n", snrt_cluster_idx(), (uint32_t)(get_bingo_sw_offload_unit()->offloadFn));
        // (4) The host sends through the mailbox the pointer to the arguments that should be used.
        h2c_mailbox_read((uint32_t *)&(get_bingo_sw_offload_unit()->offloadArgs));
        // printf("[Cluster %d] Received function args: 0x%x\n", snrt_cluster_idx(), get_bingo_sw_offload_unit()->offloadArgs);
        // Bookkeeping the start cycles
        get_bingo_sw_offload_unit()->start_cycles = snrt_mcycle();
        // (5) The manager core will execute the offload function
        bingo_sw_offload_dispatch(get_bingo_sw_offload_unit()->offloadFn, get_bingo_sw_offload_unit()->offloadArgs);
        // Bookkeeping the end cycles
        get_bingo_sw_offload_unit()->end_cycles = snrt_mcycle();
        // printf("[Cluster %d] Task %d finish with CC=%d \n", 
                // snrt_cluster_idx(),
                // get_bingo_sw_offload_unit()->task_id,
                // get_bingo_sw_offload_unit()->end_cycles-get_bingo_sw_offload_unit()->start_cycles);
        // (6) return the result through the mailbox to the host
        bingo_c2h_msg_fields_t c2h_msg;
        c2h_msg.cluster_id = snrt_cluster_idx();
        c2h_msg.flag = MBOX_DEVICE_DONE;
        c2h_msg.task_id = (uint16_t)get_bingo_sw_offload_unit()->task_id;
        c2h_msg.reserved = 0;
        c2h_mailbox_write(bingo_c2h_msg_encode(c2h_msg));
    }
    return 0;
}

// This is the bingo hw offload manager function
// It will be used together with the bingo hw manager at the Quad Ctrl
// By using the hw manager, the granularity of offloading is down to the core level
// So each core will run the following function

// Before running this device function, the host will write two ptrs to the SoC CTRL
// One is the ptr to the arg_list, another is the ptr to the fn_list
// In the beginning of the bingo_hw_offload_manager(), it will read these two ptrs and store them locally
// Later when the device core obtains the task_id, it will check the two lists to get the function ptr and arg ptr

inline uint32_t read_bingo_hw_manager_ready_queue(){
    uint32_t task_id;
    asm volatile("csrr %0, 0x5fe" : "=r"(task_id));
    // Add a dummy dependency to force the CPU to wait for the read to finish
    asm volatile("beq %0, %0, 1f; 1:" :: "r"(task_id));
    return task_id;
}

inline void write_bingo_hw_manager_done_queue(uint32_t task_id){
    asm volatile("csrw 0x5ff, %0" : : "r"(task_id));
}

/**
 * @brief Initialize the bingo HW offload unit
 */
inline void bingo_hw_offload_init() {
    // Initialize the HW offload manager
    if (snrt_is_dm_core()) {
        // Poll the host init done
        while(readw(quad_ctrl_host_init_done_addr()) == 0){
            // wait
        }
        // Now we can read the ptrs
        get_bingo_hw_offload_unit()->dev_arg_list_ptr = readw(quad_ctrl_arg_ptr_addr(snrt_cluster_idx()));
        get_bingo_hw_offload_unit()->dev_kernel_list_ptr = readw(quad_ctrl_kernel_ptr_addr(snrt_cluster_idx()));
        get_bingo_hw_offload_unit()->gid_to_dev_tid_list_ptr = readw(quad_ctrl_global_id_to_dev_id_addr(snrt_cluster_idx()));
        BINGO_PRINTF(1, "[Cluster %d Core %d]: HW offload unit initialized with arg ptr=0x%x, kernel ptr=0x%x, gid to dev tid ptr=0x%x\r\n",
               snrt_cluster_idx(), snrt_cluster_core_idx(),
               get_bingo_hw_offload_unit()->dev_arg_list_ptr,
               get_bingo_hw_offload_unit()->dev_kernel_list_ptr,
               get_bingo_hw_offload_unit()->gid_to_dev_tid_list_ptr);

    }
    // Make sure the bingo offload unit is reset by the DM core
    snrt_cluster_hw_barrier();
}

inline int32_t bingo_hw_offload_get_dev_task_id(uint32_t global_task_id){
    // Get the dev task id from the global task id
    return (int32_t)readw((uintptr_t)(get_bingo_hw_offload_unit()->gid_to_dev_tid_list_ptr + global_task_id * sizeof(uint32_t)));
}

inline uint32_t bingo_hw_offload_get_kernel_ptr(uint32_t dev_task_id){
    // Get the kernel ptr from the dev task id
    return readw((uintptr_t)(get_bingo_hw_offload_unit()->dev_kernel_list_ptr + dev_task_id * sizeof(uint32_t)));
}

inline uint32_t bingo_hw_offload_get_arg_ptr(uint32_t dev_task_id){
    // Get the arg ptr from the dev task id
    return readw((uintptr_t)(get_bingo_hw_offload_unit()->dev_arg_list_ptr + dev_task_id * sizeof(uint32_t))); 
}

inline int32_t bingo_hw_offload_manager(){
    // Step 1: Get the arg_list and fn_list ptrs from the quad ctrl CFG
    uint32_t cur_arg_ptr;
    uint32_t cur_kernel_ptr;
    uint32_t cur_global_task_id;
    int32_t cur_dev_task_id;
    uint32_t kernel_return_value;
    uint32_t err = 0;
    bingo_hw_offload_init();
    while(1){
        // Step 2: Reading the Ready Queue to get the task id
        // Using CSR to read the ready queue
        // 1. Blocking read
        BINGO_TRACE_MARKER(BINGO_TRACE_MGR_GET_READY_START);
        cur_global_task_id = read_bingo_hw_manager_ready_queue();
        BINGO_TRACE_MARKER(BINGO_TRACE_MGR_GET_READY_END);

        BINGO_TRACE_MARKER(BINGO_TRACE_MGR_PREP_START);
        // 2. Then we get the dev task id from the global task id
        cur_dev_task_id = bingo_hw_offload_get_dev_task_id(cur_global_task_id);
        if (cur_dev_task_id == -1){
            BINGO_PRINTF(2, "[Cluster %d Core %d]: Error: Invalid dev task id for global task id %d\r\n",
               snrt_cluster_idx(), snrt_cluster_core_idx(),
               cur_global_task_id);
            err=1;
            break;
        }
        cur_arg_ptr = bingo_hw_offload_get_arg_ptr(cur_dev_task_id);
        cur_kernel_ptr = bingo_hw_offload_get_kernel_ptr(cur_dev_task_id);
        BINGO_TRACE_MARKER(BINGO_TRACE_MGR_PREP_END);

        BINGO_PRINTF(1, "[Cluster %d Core %d]: Task %d Info: Dev tid=%d, arg ptr=0x%x, kernel ptr=0x%x\r\n",
               snrt_cluster_idx(), snrt_cluster_core_idx(),
               cur_global_task_id,
               cur_dev_task_id,
               cur_arg_ptr,
               cur_kernel_ptr);
        BINGO_PRINTF(2, "[Cluster %d Core %d]: Task %d Info: Running Dev Kernel ...\r\n",
               snrt_cluster_idx(),
               snrt_cluster_core_idx(),
               cur_global_task_id);
        // 3. Execute the function
        BINGO_TRACE_MARKER(BINGO_TRACE_MGR_RUN_KERNEL_START);
        kernel_return_value = ((uint32_t (*)(uint32_t))cur_kernel_ptr)(cur_arg_ptr);
        BINGO_TRACE_MARKER(BINGO_TRACE_MGR_RUN_KERNEL_END);

        // 4. Write the Done queue to notify the bingo hw scheduler
        BINGO_TRACE_MARKER(BINGO_TRACE_MGR_WRITE_DONE_START);
        if (kernel_return_value == BINGO_RET_SUCC){
            BINGO_PRINTF(2, "[Cluster %d Core %d]: Task %d Info: Succ!\r\n",
                   snrt_cluster_idx(),
                   snrt_cluster_core_idx(),
                   cur_global_task_id);
            write_bingo_hw_manager_done_queue(cur_global_task_id);
            BINGO_TRACE_MARKER(BINGO_TRACE_MGR_WRITE_DONE_END);
            // THen keep reading the next ready task id
        } else if (kernel_return_value == BINGO_RET_EXIT){
            // Exit signal received from the kernel
            // There will be special kernel to return a 1 to exit the hw offload manager
            // Access the clint mutex to printf the error message
            BINGO_PRINTF(2, "[Cluster %d Core %d]: Task %d Info: Exiting ...\r\n",
                   snrt_cluster_idx(),
                   snrt_cluster_core_idx(),
                   cur_global_task_id);
            write_bingo_hw_manager_done_queue(cur_global_task_id);
            BINGO_TRACE_MARKER(BINGO_TRACE_MGR_WRITE_DONE_END);
            break;
        } else if (kernel_return_value == BINGO_RET_FAIL){
            // Other error code
            err = kernel_return_value;
            // Access the clint mutex to printf the error message
            BINGO_PRINTF(1, "[Cluster %d Core %d]: Task %d Error: Error!!!, error code=%d\r\n",
                   snrt_cluster_idx(),
                   snrt_cluster_core_idx(),
                   cur_global_task_id,
                   kernel_return_value);
            write_bingo_hw_manager_done_queue(cur_global_task_id);
            BINGO_TRACE_MARKER(BINGO_TRACE_MGR_WRITE_DONE_END);
            break;
        } else {
            // Unknown error code
            err = kernel_return_value;
            // Access the clint mutex to printf the error message
            BINGO_PRINTF(1, "[Cluster %d Core %d]: Task %d Error: Unknown return code=%d\r\n",
                   snrt_cluster_idx(),
                   snrt_cluster_core_idx(),
                   cur_global_task_id,
                   kernel_return_value);
            write_bingo_hw_manager_done_queue(cur_global_task_id);
            BINGO_TRACE_MARKER(BINGO_TRACE_MGR_WRITE_DONE_END);
            break;
        }
    }
    return err;
}

inline int32_t bingo_offload_manager(){
    // we use the 4th scratch register to indicate the offload type
    // Not set yet = 0
    // SW offload = 1
    // HW offload = 2
    while(readw(soc_ctrl_kernel_tab_scratch_addr(3))==0){
        // wait for the host to set the offload type
    }
    if (readw(soc_ctrl_kernel_tab_scratch_addr(3))==1){
        // Software offload
        return bingo_sw_offload_manager();
    } else if (readw(soc_ctrl_kernel_tab_scratch_addr(3))==2){
        // Hardware offload
        return bingo_hw_offload_manager();
    } else {
        // Invalid offload type
        printf_safe("[Cluster %d] Error: Invalid offload type %d\r\n", snrt_cluster_idx(), readw(soc_ctrl_kernel_tab_scratch_addr(3)));
        return -1;
    }   
}