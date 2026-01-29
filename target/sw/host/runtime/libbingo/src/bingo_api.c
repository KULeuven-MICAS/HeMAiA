// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

// This is the host-side runtime for offloading to the clusters
#include "libbingo/bingo_api.h"
#include <stddef.h>  // size_t
#include <stdlib.h>  // NULL
#include <stdio.h>
#include "heterogeneous_runtime.h"
uint64_t global_task_id = 0; // Internal monotonically increasing id source, notice this is used in each chiplet
///////////////////////////
// Memory Allocator API  //
///////////////////////////

int bingo_hemaia_system_mmap_init(){
    // comm_buffer assignment: Begin from SPM_NARROW_BASE_ADDR and initialize to zero
    uint64_t comm_buffer = chiplet_addr_transform(SPM_NARROW_BASE_ADDR);
    // printf("Chip(%x, %x): [Host] Comm buffer addr: %lx\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), comm_buffer);
    // L2 heap init
    // Start addr is the spm narrow
    // Size is half of the narrow spm
    // The rest is leave to stack
    uint64_t l2_heap_start = ALIGN_UP(comm_buffer + sizeof(comm_buffer_t), O1HEAP_ALIGNMENT);
    uint64_t l2_heap_size = ALIGN_UP(NARROW_SPM_SIZE / 2, O1HEAP_ALIGNMENT);
    uint64_t l2_heap_manager = o1heapInit(l2_heap_start, l2_heap_size);
    // printf("Chip(%x, %x): [Host] L2 heap start: %lx, size(kB): %d\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), l2_heap_manager, l2_heap_size>>10);
    // printf("Chip(%x, %x): [Host] L2 heap start: %lx, size: %lx, heap manager: 0x%lx\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), l2_heap_start, l2_heap_size, l2_heap_manager);
    if(l2_heap_manager==0UL) {
        printf("Error when initializing L2 heap. \r\n");
        return -1;
    }
    // L3 heap init
    // Start addr is the l3 heap start symbol aligned to 1KB
    // Size is from l3 heap start to wide spm end aligned down 1KB
    uint64_t l3_heap_start = ALIGN_UP(chiplet_addr_transform((uint64_t)(&__l3_heap_start)), SPM_WIDE_ALIGNMENT);
    uint64_t l3_heap_size = ALIGN_DOWN(chiplet_addr_transform((uint64_t)(&__wide_spm_end)-SPM_WIDE_ALIGNMENT), SPM_WIDE_ALIGNMENT) - l3_heap_start;
    uint64_t l3_heap_manager = o1heapInit(l3_heap_start, l3_heap_size);
    // printf("Chip(%x, %x): [Host] L3 heap start: %lx, size(kB): %d\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), l3_heap_manager, l3_heap_size>>10);
    // printf("Chip(%x, %x): [Host] L3 heap start: %lx, size: %lx, heap manager: 0x%lx\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), l3_heap_start, l3_heap_size, l3_heap_manager);
    if(l3_heap_manager==0UL) {
        printf("Error when initializing L3 heap.\r\n");
        return -1;
    }
    return 0;
}

uint64_t bingo_get_l1_heap_manager(uint8_t chip_id, uint32_t cluster_id){
    // Notice the l1 heap is init by the cluster at the start of the TCDM of each cluster
    // Here we hack the o1heap sw to force a 32bit version of heap manager so that the host can access to the correct
    // heap information from the O1HeapInstanc32 structure
    return chiplet_addr_transform_full(chip_id, (uint64_t)cluster_tcdm_start_addr(cluster_id));
}


uint64_t bingo_get_l2_comm_buffer(uint8_t chip_id){
    return chiplet_addr_transform_full(chip_id, (uintptr_t)SPM_NARROW_BASE_ADDR);
}

uint64_t bingo_get_l3_heap_manager(uint8_t chip_id){
    return chiplet_addr_transform_full(chip_id, ALIGN_UP(chiplet_addr_transform((uint64_t)(&__l3_heap_start)), SPM_WIDE_ALIGNMENT));
}

uint64_t bingo_get_l2_heap_manager(uint8_t chip_id){
    return chiplet_addr_transform_full(chip_id, ALIGN_UP((uintptr_t)SPM_NARROW_BASE_ADDR + sizeof(comm_buffer_t), O1HEAP_ALIGNMENT));
}

uint64_t bingo_l1_alloc(uint8_t chip_id, uint32_t cluster_id, uint64_t size){
    uint64_t results = o1heapAllocate(bingo_get_l1_heap_manager(chip_id, cluster_id), size);
    if (results==0UL) {
        printf_safe("Chip(%x, %x): [Host] L1 malloc failed for size %d on cluster %d\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), size, cluster_id);
    }
    BINGO_PRINTF(3, "Chip(%x, %x): [Host] L1 malloc on cluster %d: ptr=0x%lx, size=%d\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(),
           cluster_id,
           results,
           size);
    return results;
}

uint64_t bingo_l2_alloc(uint8_t chip_id, uint64_t size){
    uint64_t results = o1heapAllocate(bingo_get_l2_heap_manager(chip_id), size);
    if (results==0UL) {
        printf_safe("Chip(%x, %x): [Host] L2 malloc failed for size %d\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), size);
    }
    BINGO_PRINTF(3, "Chip(%x, %x): [Host] L2 malloc: ptr=0x%lx, size=%d\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(),
           results,
           size);
    return results;
}

uint64_t bingo_l3_alloc(uint8_t chip_id, uint64_t size){
    uint64_t results = o1heapAllocate(bingo_get_l3_heap_manager(chip_id), size);
    if (results==0UL) {
        printf_safe("Chip(%x, %x): [Host] L3 malloc failed for size %d\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), size);
    }
    BINGO_PRINTF(3, "Chip(%x, %x): [Host] L3 malloc: ptr=0x%lx, size=%d\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(),
           results,
           size);
    return results;
}

void bingo_l1_free(uint8_t chip_id, uint32_t cluster_id, uint64_t ptr){
    o1heapFree(bingo_get_l1_heap_manager(chip_id, cluster_id), ptr);
}

void bingo_l2_free(uint8_t chip_id, uint64_t ptr){
    o1heapFree(bingo_get_l2_heap_manager(chip_id), ptr);
}

void bingo_l3_free(uint8_t chip_id, uint64_t ptr){
    o1heapFree(bingo_get_l3_heap_manager(chip_id), ptr);
}


//////////////////////////
// Mailbox API          //
//////////////////////////

int bingo_try_write_h2h_mailbox(uint8_t chip_id, uint64_t dword) {
    uint8_t current_chip_id = get_current_chip_id();
    if (chip_id == current_chip_id) {
        return BINGO_MB_ERR_PARAM;
    }
    volatile uint64_t target_h2h_mailbox_write_addr = chiplet_addr_transform_full(chip_id, h2h_mailbox_write_address());
    volatile uint64_t target_h2h_mailbox_status_addr = chiplet_addr_transform_full(chip_id, h2h_mailbox_status_flag_address());
    uint64_t status = readd((uintptr_t)target_h2h_mailbox_status_addr);
    if (BINGO_EXTRACT_BIT(status, 1)) {
        return 0; // would block
    }
    writed(dword, (uintptr_t)target_h2h_mailbox_write_addr);
    return 1; // success
}

int bingo_try_read_h2h_mailbox(uint64_t *buffer) {
    if (!buffer) return BINGO_MB_ERR_PARAM;
    volatile uint64_t local_h2h_mailbox_read_addr = chiplet_addr_transform(h2h_mailbox_read_address());
    volatile uint64_t local_h2h_mailbox_status_addr = chiplet_addr_transform(h2h_mailbox_status_flag_address());
    uint64_t status = readd((uintptr_t)local_h2h_mailbox_status_addr);
    if (BINGO_EXTRACT_BIT(status, 0)) {
        return 0; // would block
    }
    *buffer = readd((uintptr_t)local_h2h_mailbox_read_addr);
    return 1; // success
}

int bingo_write_h2h_mailbox(uint8_t chip_id, uint64_t dword,
                            uint64_t timeout_cycles, uint32_t *retry_hint) {
    uint8_t current_chip_id = get_current_chip_id();
    if (chip_id == current_chip_id) {
        printf("Chip(%x, %x): [Host] Error: Cannot write to its own H2H mailbox!\\r\\n", get_current_chip_loc_x(), get_current_chip_loc_y());
        return BINGO_MB_ERR_PARAM;
    }
    volatile uint64_t target_h2h_mailbox_write_addr = chiplet_addr_transform_full(chip_id, h2h_mailbox_write_address());
    volatile uint64_t target_h2h_mailbox_status_addr = chiplet_addr_transform_full(chip_id, h2h_mailbox_status_flag_address());
    // Do not check status, just write directly
    writed(dword, (uintptr_t)target_h2h_mailbox_write_addr);
}

int bingo_read_h2h_mailbox(uint64_t *buffer,
                           uint64_t timeout_cycles, uint32_t *retry_hint) {
    if (!buffer) return BINGO_MB_ERR_PARAM;
    volatile uint64_t local_h2h_mailbox_read_addr = chiplet_addr_transform(h2h_mailbox_read_address());
    volatile uint64_t local_h2h_mailbox_status_addr = chiplet_addr_transform(h2h_mailbox_status_flag_address());
    uint64_t start_cycle = bingo_mcycle();
    uint32_t retries = 0;
    while (1) {
        uint64_t status = readd((uintptr_t)local_h2h_mailbox_status_addr);
        if (!BINGO_EXTRACT_BIT(status, 0)) {
            *buffer = readd((uintptr_t)local_h2h_mailbox_read_addr);
            if (retry_hint) *retry_hint = retries;
            return BINGO_MB_OK;
        }
        // FIFO empty
        retries++;
        if (timeout_cycles && ((bingo_mcycle() - start_cycle) > timeout_cycles)) {
            if (retry_hint) *retry_hint = retries;
            return BINGO_MB_ERR_TIMEOUT;
        }
        bingo_csleep(HOST_SLEEP_CYCLES);
    }
}

int bingo_try_write_h2c_mailbox(uint8_t cluster_id, uint32_t word){
    volatile uint64_t h2c_mailbox_write_addr = chiplet_addr_transform(h2c_mailbox_write_address(cluster_id));
    volatile uint64_t h2c_mailbox_status_addr = chiplet_addr_transform(h2c_mailbox_status_flag_address(cluster_id));
    uint32_t status = readw((uintptr_t)h2c_mailbox_status_addr);
    if (BINGO_EXTRACT_BIT(status, 1)) {
        return 0; // would block
    }
    writew(word, (uintptr_t)h2c_mailbox_write_addr);
    return 1; // success
}

int bingo_write_h2c_mailbox(uint8_t cluster_id, uint32_t word, uint64_t timeout_cycles, uint32_t *retry_hint){
    volatile uint64_t h2c_mailbox_write_addr = chiplet_addr_transform(h2c_mailbox_write_address(cluster_id));
    volatile uint64_t h2c_mailbox_status_addr = chiplet_addr_transform(h2c_mailbox_status_flag_address(cluster_id));

    uint64_t start_cycle = bingo_mcycle();
    uint32_t retries = 0;
    while (1) {
        uint32_t status = readw((uintptr_t)h2c_mailbox_status_addr);
        if (!BINGO_EXTRACT_BIT(status, 1)) {
            writew(word, (uintptr_t)h2c_mailbox_write_addr);
            if (retry_hint) *retry_hint = retries;
            return BINGO_MB_OK;
        }
        // FIFO full
        retries++;
        if (timeout_cycles && ((bingo_mcycle() - start_cycle) > timeout_cycles)) {
            if (retry_hint) *retry_hint = retries;
            return BINGO_MB_ERR_TIMEOUT;
        }
        bingo_csleep(HOST_SLEEP_CYCLES);
    }
}

int bingo_try_read_c2h_mailbox(uint32_t *buffer){
    if (!buffer) return BINGO_MB_ERR_PARAM;
    volatile uint64_t c2h_mailbox_read_addr = chiplet_addr_transform(c2h_mailbox_read_address());
    volatile uint64_t c2h_mailbox_status_addr = chiplet_addr_transform(c2h_mailbox_status_flag_address());
    uint32_t status = readw((uintptr_t)c2h_mailbox_status_addr);
    // Bit0 == 1 means EMPTY (following H2H semantics), so data available when bit0 == 0
    if (!BINGO_EXTRACT_BIT(status, 0)) {
        *buffer = readw((uintptr_t)c2h_mailbox_read_addr);
        return 1; // success
    }
    return 0; // would block
}


int bingo_read_c2h_mailbox(uint32_t *buffer, uint64_t timeout_cycles, uint32_t *retry_hint){
    if (!buffer) return BINGO_MB_ERR_PARAM;
    volatile uint64_t c2h_mailbox_read_addr = chiplet_addr_transform(c2h_mailbox_read_address());
    volatile uint64_t c2h_mailbox_status_addr = chiplet_addr_transform(c2h_mailbox_status_flag_address());
    uint64_t start_cycle = bingo_mcycle();
    uint32_t retries = 0;
    while (1) {
        uint32_t status = readw((uintptr_t)c2h_mailbox_status_addr);
        if (!BINGO_EXTRACT_BIT(status, 0)) {
            *buffer = readw((uintptr_t)c2h_mailbox_read_addr);
            if (retry_hint) *retry_hint = retries;
            return BINGO_MB_OK;
        }
        // FIFO empty
        retries++;
        if (timeout_cycles && ((bingo_mcycle() - start_cycle) > timeout_cycles)) {
            if (retry_hint) *retry_hint = retries;
            return BINGO_MB_ERR_TIMEOUT;
        }
        bingo_csleep(HOST_SLEEP_CYCLES);
    }
}

/////////////////////////////
// Task Management API    //
/////////////////////////////
// Helper function to get the task from the task id
bingo_task_t* bingo_get_task_from_id(bingo_task_t **task_list, uint32_t num_tasks, uint8_t task_id) {
    if (!task_list || num_tasks == 0) {
        printf("[BINGO] Error: Invalid task list or number of tasks.\r\n");
        return NULL;
    }

    for (uint32_t i = 0; i < num_tasks; i++) {
        if (task_list[i] && task_list[i]->task_id == task_id) {
            return task_list[i]; // Found the task
        }
    }

    // Task not found
    printf("[BINGO] Warning: Task with ID %d not found in the task list.\r\n", task_id);
    return NULL;
}

// Helper inline to test if a task is globally ready (both local and remote deps satisfied)
inline bool bingo_task_is_globally_ready(const bingo_task_t *t) {
  return (t->local_pred_remaining == 0) && (t->remote_pred_remaining == 0);
}

// Reset live dependency counters from their initial snapshots (used for replays / diagnostics)
inline void bingo_task_reset_deps(bingo_task_t *t) {
  t->local_pred_remaining  = t->local_pred_initial;
  t->remote_pred_remaining = t->remote_pred_initial;
}

// Task creation initializes counters & clears successor lists.
bingo_task_t *bingo_task_create(uint32_t fn_ptr, uint32_t args_ptr, uint8_t assigned_chip_id, uint8_t assigned_cluster_id) {

    bingo_task_t *task = (bingo_task_t *)(uintptr_t)o1heapAllocate(bingo_get_l2_heap_manager(get_current_chip_id()), sizeof(bingo_task_t));
    if (!task){
        printf("[BINGO] Error: Failed to allocate memory for new task.\r\n");
        return NULL;
    }
    task->task_id = readh((uintptr_t)chiplet_addr_transform((uint64_t)&global_task_id));
    task->fn_ptr = fn_ptr;
    task->args_ptr = args_ptr;
    task->assigned_chip_id = assigned_chip_id;
    task->assigned_cluster_id = assigned_cluster_id;

    task->local_pred_initial = 0;
    task->local_pred_remaining = 0;
    task->remote_pred_initial = 0;
    task->remote_pred_remaining = 0;

    task->num_local_successors = 0;
    for (int i = 0; i < MAX_SUCCESSORS; i++) {
        task->local_successors[i] = NULL;
    }
    task->num_remote_successors = 0;
    for (int i = 0; i < BINGO_MAX_REMOTE_SUCC; i++) {
        task->remote_successors[i].chip_id = 0;
        task->remote_successors[i].task_id = 0;
    }

    task->offloaded = false;
    task->completed = false;
    task->enqueued_ready = false;
    task->completion_notified = false;
    task->debug_seq_issue = 0;
    task->debug_seq_complete = 0;
    // Increment global task id by 1
    writeh(
        readh((uintptr_t)chiplet_addr_transform((uint64_t)&global_task_id)) + 1,
        (uintptr_t)chiplet_addr_transform((uint64_t)&global_task_id)
    );
    return task;
}

// Add dependency: dep_task (predecessor) -> task (successor)
// Decide if successor is local or remote relative to predecessor's chip assignment.
void bingo_task_add_depend(bingo_task_t *task, bingo_task_t *dep_task) {
    if (!task || !dep_task){
        printf("[BINGO] Error: Null task pointer in bingo_task_add_depend.\r\n");
        return;
    }
    if (dep_task->assigned_chip_id == task->assigned_chip_id) {
        // Local edge
        task->local_pred_initial++;
        task->local_pred_remaining++;
        if (dep_task->num_local_successors < MAX_SUCCESSORS) {
            dep_task->local_successors[dep_task->num_local_successors++] = task;
        } else {
            printf("[BINGO] Error: MAX_SUCCESSORS exceeded for task %u\r\n", dep_task->task_id);
        }
    } else {
        // Remote edge
        task->remote_pred_initial++;
        task->remote_pred_remaining++;
        if (dep_task->num_remote_successors < BINGO_MAX_REMOTE_SUCC) {
            dep_task->remote_successors[dep_task->num_remote_successors].chip_id = task->assigned_chip_id;
            dep_task->remote_successors[dep_task->num_remote_successors].task_id = task->task_id;
            dep_task->num_remote_successors++;
        } else {
            printf("[BINGO] Error: BINGO_MAX_REMOTE_SUCC exceeded for task %u\r\n", dep_task->task_id);
        }
    }
}

void bingo_task_offload(bingo_task_t *task) {
    if (!task){
        printf_safe("[BINGO] Error: Null task pointer in bingo_task_offload.\r\n");
        return;
    }
    if (task->offloaded){
        printf_safe("[BINGO] Warning: Task %u already offloaded.\r\n", task->task_id);
        return;
    }
    BINGO_PRINTF(1, "Chip(%x, %x): [Host] Offloaded task %x -> chip %x cluster %d fn=0x%x args=0x%x\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(),
           (uint32_t)task->task_id,
           (uint32_t)task->assigned_chip_id,
           (uint32_t)task->assigned_cluster_id, (uint32_t)task->fn_ptr, (uint32_t)task->args_ptr); 
    // Send start command + task info via H2C mailbox
    uint32_t retry_hint_start, retry_hint_id, retry_hint_fn, retry_hint_args;
    uint32_t total_retries = 0;
    bingo_write_h2c_mailbox(task->assigned_cluster_id, (uint32_t)MBOX_DEVICE_START, 0, &retry_hint_start);
    bingo_write_h2c_mailbox(task->assigned_cluster_id, (uint32_t)task->task_id, 0, &retry_hint_id);
    bingo_write_h2c_mailbox(task->assigned_cluster_id, (uint32_t)task->fn_ptr, 0, &retry_hint_fn);
    bingo_write_h2c_mailbox(task->assigned_cluster_id, (uint32_t)task->args_ptr, 0, &retry_hint_args);
    total_retries += retry_hint_start + retry_hint_id + retry_hint_fn + retry_hint_args;
    if(total_retries > 100) {
        printf_safe("Chip(%x, %x): [Host] Warning: High mailbox write retries (%u) when offloading task %u to chip %u cluster %u\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), total_retries,
               task->task_id, task->assigned_chip_id, task->assigned_cluster_id);
    }
    task->offloaded = true;
   
}


static void bingo_broadcast_completion(const bingo_task_t *t, uint8_t src_chip) {
    if (t->completion_notified) return;
    // Iterate remote successors and send messages to the corresponding chips
    // The message is the successor task id
    for (uint8_t i = 0; i < t->num_remote_successors; i++) {
        BINGO_PRINTF(1, "Chip(%x, %x): [Host] Broadcasting completion to remote successor with chip %x and task id %u\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), t->remote_successors[i].chip_id, t->remote_successors[i].task_id);
        uint8_t dst_chip = t->remote_successors[i].chip_id;
        uint16_t dst_task = t->remote_successors[i].task_id;
        bingo_msg_fields_t mf = {
            .type = BINGO_MSG_TASK_COMPLETE,
            .src_chip = src_chip,
            .seq = 0, // sequence optional; could be added later
            .payload = dst_task
        };
        uint64_t word = bingo_msg_encode(mf);
        // Blocking write (timeout 0 => infinite)
        bingo_write_h2h_mailbox(dst_chip, word, 0, NULL);
    }
}

// ------------------------------
// Ready Queue Helpers
// ------------------------------
static uint16_t bingo_next_pow2(uint16_t v){
    if (v < 2) return 2;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    return (uint16_t)(v + 1);
}

static inline bool sched_is_empty(const bingo_chip_sched_t *s){
    return s->ready_head == s->ready_tail;
}
static inline bool sched_is_full(const bingo_chip_sched_t *s){
    return ((uint16_t)(s->ready_tail + 1) & s->ready_mask) == s->ready_head;
}
static inline void sched_enqueue(bingo_chip_sched_t *s, bingo_task_t *t){
    if (sched_is_full(s)) {
        // Simple strategy: drop enqueue attempt (should not happen with sufficient cap)
        // Could log warning here.
        return;
    }
    s->ready_ring[s->ready_tail] = t;
    s->ready_tail = (s->ready_tail + 1) & s->ready_mask;
}
static inline bingo_task_t *sched_dequeue(bingo_chip_sched_t *s){
    if (sched_is_empty(s)) return NULL;
    bingo_task_t *t = s->ready_ring[s->ready_head];
    s->ready_head = (s->ready_head + 1) & s->ready_mask;
    return t;
}

void bingo_runtime_schedule(bingo_task_t **task_list, uint32_t num_tasks) {
    // We need to set the soc_ctrl_kernel_tab_scratch_addr(3) to 1 to indicate we are using the SW scheduler
    // See target/sw/device/runtime/src/bingo.h for details
    writew(1,      (uintptr_t)chiplet_addr_transform((uint64_t)soc_ctrl_kernel_tab_scratch_addr(3)));
    // Tell the device that the host init is done
    writew(1,      (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_host_init_done_addr()));
    asm volatile("fence" ::: "memory");
    uint8_t current_chip = get_current_chip_id();
    // Count local tasks
    uint16_t local_total = 0;
    for (uint32_t i = 0; i < num_tasks; i++) {
        if (task_list[i]->assigned_chip_id == current_chip){
            local_total++;
        }
        
    }
    printf_safe("Chip(%x, %x): [Host] Starting runtime schedule for %u local tasks\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), local_total);
    if (local_total == 0) return; // Nothing to do

    // Initialize scheduler instance (stack-local for now; could be static per chip)
    BINGO_TRACE_MARKER(BINGO_TRACE_SW_MGR_INIT_TASK_QUEUE_START);
    uint16_t cap = bingo_next_pow2(local_total * 2); // room for burst readiness
    bingo_task_t **ring = (bingo_task_t **)(uintptr_t)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(bingo_task_t*) * cap);
    if (!ring) {
        printf_safe("[BINGO] Error: Failed to allocate ready ring.\r\n");
        return;
    }
    bingo_chip_sched_t *sched = (bingo_chip_sched_t *)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(bingo_chip_sched_t));
    if (!sched) {
        printf_safe("[BINGO] Error: Failed to allocate scheduler instance.\r\n");
        return;
    }
    sched->chip_id = current_chip;
    sched->ready_cap = cap;
    sched->ready_mask = (uint16_t)(cap - 1);
    sched->ready_head = 0;
    sched->ready_tail = 0;
    sched->ready_ring = ring;
    sched->inflight = 0;
    sched->completed = 0;
    sched->tx_msgs = 0;
    sched->rx_msgs = 0;
    sched->seq_counter = 0;

    // Seed initial ready tasks (those with zero local & remote predecessors)
    for (uint32_t i = 0; i < num_tasks; i++) {
        bingo_task_t *t = task_list[i];
        if (t->assigned_chip_id != current_chip) continue;
        if (bingo_task_is_globally_ready(t)) {
            sched_enqueue(sched, t);
        }
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_SW_MGR_INIT_TASK_QUEUE_END);
    uint16_t local_completed = 0;
    while (local_completed < local_total) {
        // 1. Poll device C2H events (task completions etc.)
        while (1) {
            uint32_t word;
            int r = bingo_try_read_c2h_mailbox(&word);
            if (r <= 0) break; // no info yet

            bingo_c2h_msg_fields_t msg = bingo_c2h_msg_decode(word);
            switch (msg.flag & 0xF) {
                case MBOX_DEVICE_DONE: {
                    BINGO_TRACE_MARKER(BINGO_TRACE_SW_MGR_ENQUEUE_LOCAL_READY_TASKS_START);
                    uint16_t tid = msg.task_id;
                    BINGO_PRINTF(1, "Chip(%x, %x): [Host] Task %x completed on chip %x cluster %d \r\n",
                           get_current_chip_loc_x(), get_current_chip_loc_y(),
                           (uint32_t)tid, current_chip, (uint32_t)msg.cluster_id);
                    if (tid < num_tasks) {
                        // find the completed task

                        bingo_task_t* completed_task = bingo_get_task_from_id(task_list, num_tasks, tid);
                        if (!completed_task) {
                            printf_safe("Chip(%x, %x): [Host] Error: Completed task %x not found in task list!\r\n",
                                   get_current_chip_loc_x(), get_current_chip_loc_y(), (uint32_t)tid);
                            break;
                        }
                        if(completed_task->assigned_chip_id != current_chip) {
                            printf_safe("Chip(%x, %x): [Host] Error: Completed task %d assigned to chip %x, but current chip is %x!\r\n",
                                   get_current_chip_loc_x(), get_current_chip_loc_y(),
                                   (uint32_t)tid,
                                   (uint32_t)completed_task->assigned_chip_id,
                                   (uint32_t)current_chip);
                            break;
                        }
                        if (!completed_task->completed && completed_task->assigned_chip_id == current_chip) {
                            completed_task->completed = true;
                            local_completed++;
                            // Broadcast remote completion once
                            if (!completed_task->completion_notified && completed_task->num_remote_successors > 0) {
                                printf("Chip(%x, %x): [Host] Broadcasting completion of task %u to %u remote successors\r\n",
                                       get_current_chip_loc_x(), get_current_chip_loc_y(), completed_task->task_id, completed_task->num_remote_successors);
                                bingo_broadcast_completion(completed_task, current_chip);
                                completed_task->completion_notified = true;
                            }
                            // Decrement local successor counters & enqueue those now globally ready
                            for (uint8_t s = 0; s < completed_task->num_local_successors; s++) {
                                bingo_task_t *suc = completed_task->local_successors[s];
                                if (suc->local_pred_remaining) {
                                    suc->local_pred_remaining--;
                                    if (bingo_task_is_globally_ready(suc) && !suc->offloaded) {
                                        sched_enqueue(sched, suc);
                                    }
                                }
                            }

                        }
                        
                    }
                    BINGO_TRACE_MARKER(BINGO_TRACE_SW_MGR_ENQUEUE_LOCAL_READY_TASKS_END);
                    break; }
                default:
                    break; // Ignore other flags for now
            }
        }

        // 2. Drain H2H remote completion notifications and enqueue tasks whose remote preds reach zero
        while (1) {
            uint64_t word;
            int rr = bingo_try_read_h2h_mailbox(&word);
            if (rr <= 0) break;
            BINGO_TRACE_MARKER(BINGO_TRACE_SW_MGR_ENQUEUE_REMOTE_READY_TASKS_START);
            bingo_msg_fields_t f = bingo_msg_decode(word);
            if (f.type == BINGO_MSG_TASK_COMPLETE) {
                uint16_t tid = f.payload;
                // The Tid is the suc task id of the previously completed remote task
                if (tid < num_tasks) {
                    bingo_task_t* local_task = bingo_get_task_from_id(task_list, num_tasks, tid);
                    local_task->remote_pred_remaining--;
                    if (bingo_task_is_globally_ready(local_task) && !local_task->offloaded) {
                        sched_enqueue(sched, local_task);
                    }                            
                }
            }
            BINGO_TRACE_MARKER(BINGO_TRACE_SW_MGR_ENQUEUE_REMOTE_READY_TASKS_END);
        }

        // 3. Offload as many ready tasks as possible (could add throttle / inflight cap later)
        
        while (1) {
            bingo_task_t *t = sched_dequeue(sched);
            if (!t) break;
            BINGO_TRACE_MARKER(BINGO_TRACE_SW_MGR_SCHED_READY_TASKS_START);
            if (!t->offloaded) {
                bingo_task_offload(t);
                sched->inflight++;
            }
            BINGO_TRACE_MARKER(BINGO_TRACE_SW_MGR_SCHED_READY_TASKS_END);
        }
        

        // 4. Sleep a bit to reduce host busy waiting
        if (local_completed < local_total) {
            bingo_csleep(HOST_SLEEP_CYCLES);
        }
    }
    // Destroy the allocated ready ring
    o1heapFree(bingo_get_l2_heap_manager(get_current_chip_id()), (uint64_t)sched->ready_ring);
    // Desctroy the scheduler instance
    o1heapFree(bingo_get_l2_heap_manager(get_current_chip_id()), (uint64_t)sched);
    printf("Chip(%x, %x): [Host] Runtime schedule completed for all %u local tasks\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), local_total);
}

void bingo_close_all_clusters(bingo_task_t **task_list, uint32_t num_tasks){
    uint8_t current_chip = get_current_chip_id();
    // Issue STOP to each cluster used
    bool cluster_stopped[N_CLUSTERS_PER_CHIPLET] = {0};
    for (uint32_t i = 0; i < num_tasks; i++) {
        bingo_task_t *t = task_list[i];
        if (t->assigned_chip_id != current_chip) continue;
        uint8_t cid = t->assigned_cluster_id;
        if (!cluster_stopped[cid]) {
            bingo_write_h2c_mailbox(cid, MBOX_DEVICE_STOP, 0, NULL);
            cluster_stopped[cid] = true;
        }
    }
}

/////////////////////////
// For the HW scheduler
/////////////////////////
uint32_t bingo_hw_scheduler_get_global_task_id(){
    return readw((uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_host_ready_done_queue_addr()));
}
void bingo_hw_scheduler_write_done_queue(uint32_t global_task_id){
    writew(global_task_id, (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_host_ready_done_queue_addr()));
}


int32_t bingo_hw_scheduler_get_host_task_id(int32_t* global_task_id_to_host_task_id, uint32_t global_task_id){
    return global_task_id_to_host_task_id[global_task_id];
}

int64_t* bingo_hw_scheduler_get_host_arg(int64_t* host_arg_list_base, uint32_t host_task_id){
    return &host_arg_list_base[host_task_id];
}

int64_t* bingo_hw_scheduler_get_host_kernel(int64_t* host_kernel_list_base, uint32_t host_task_id){
    return &host_kernel_list_base[host_task_id];
}

void bingo_hw_scheduler_init_pm(){
    BINGO_PRINTF(1, "Chip(%x, %x): [Host] Initializing HW Scheduler Power Manager\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    // Init the power manager
    // We need to prepare the following registers:
    // 1. quad_ctrl_idle_power_level_addr: set to the desired power level for idle state
    // For simulation we can set to roughly 1/4 of the normal speed (6)
    writew(25,                           (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_idle_power_level_addr()));
    // 2. quad_ctrl_norm_power_level_addr: set to the desired power level for normal state
    // For simulation the default value is 6
    // This is due to the 4Ghz PLL divided by 16 gives
    // For chip testing, we should choose another value derived from the 4Ghz PLL
    writew(6,                             (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_norm_power_level_addr()));
    // 3. quad_ctrl_pm_base_hi_addr: set to the high 32 bits of the power manager base address
    uint64_t CLK_CONTROLLER_ADDR = chiplet_addr_transform(HEMAIA_CLK_RST_CONTROLLER_BASE_ADDR);
    writew((uint32_t)(CLK_CONTROLLER_ADDR>>32),       (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_pm_base_hi_addr()));
    // 4. quad_ctrl_pm_base_lo_addr: set to the low 32 bits of the power manager base address
    writew((uint32_t)(CLK_CONTROLLER_ADDR),           (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_pm_base_lo_addr()));
    // 5. quad_ctrl_core_power_domain_addr: set to the core power domain
    // First iterate the cores and then clusters
    // cluster 0 core0, cluster 0 core1 cluster 0 core2, cluster 1 core 1
    // From the hardware design, we have the following power domain mapping:
    // Domain 0: Host core
    // Domain 1: Cluster 0
    // Domain 2: Cluster 1
    // ...
    // At most we have 32 domains (0-31)
    // However, we put the host core to the core 2 of cluster 0
    // and we not want to change the host core domain
    // So we need to special case the host core here
    uint32_t core_power_domain;
    uint32_t idx;
    for (uint32_t cluster = 0; cluster < N_CLUSTERS_PER_CHIPLET; cluster++){
        for (uint32_t core = 0; core < N_CORES_PER_CLUSTER + 1; core++){ // +1 for the host core
            idx = cluster * (N_CORES_PER_CLUSTER + 1) + core;
            if (cluster == 0 && core == 2){
                // Host core, do not change its power domain
                // In Bingo HW scheduler, it will read this value and compare it with the 32
                // to decide whether this is a valid domain or not
                // We set to 99 to indicate it is invalid
                core_power_domain = 99; 
            } else {
                core_power_domain = cluster + 1; // Cluster i
            }
            writew(core_power_domain, (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_core_power_domain_addr(idx)));
        }
    }
    // 6. quad_ctrl_enable_idle_pm_addr: set to 1 to enable power manager
    writew(0,                             (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_enable_idle_pm_addr()));
    asm volatile("fence" ::: "memory");
}


// The task will be initized directly on a .h file generated from the mini compiler
// So the whole scheduling process will be handled by the hardware scheduler
// The host-side work is just in the begining to write the task_list_ptr and #num_tasks to the quad ctrl reg
// Then the HW scheduler will read the task list from the memory and schedule the tasks
// The Host core will also hooked with th ARA core for the simd
// So its work is just to read the ready queue and write to the done queue when the simd is done
void bingo_hw_scheduler_init(uint64_t dev_arg_base_addr, uint64_t dev_kernel_base_addr, uint32_t num_dev_tasks, uint64_t global_task_id_to_dev_task_id_base_addr, uint64_t task_desc_list_base, uint32_t num_tasks){
    // We need to set the soc_ctrl_kernel_tab_scratch_addr(3) to 2 to indicate we are using the HW scheduler
    // See target/sw/device/runtime/src/bingo.h for details
    writew(2, (uintptr_t)chiplet_addr_transform((uint64_t)soc_ctrl_kernel_tab_scratch_addr(3)));
    // For the dev_arg_base_addr, dev_kernel_base_addr, global_task_id_to_dev_task_id_base_addr
    // We use those list when we get a globak task id and need to get the arg/kernel ptr for the device
    // The host core will write those list to the clusters' TCDM to let the device access them directly
    // instead of going to the main memory
    // Essentially we are doing the SW Cache here for the device to reduce the main memory access

    // Assign the space in the clusters TCDM for those lists
    for (uint32_t i = 0; i < N_CLUSTERS_PER_CHIPLET; i++)
    {
        // Allocate space in the cluster's L1 for those lists
        uint64_t ptr_dev_arg_base = bingo_l1_alloc(get_current_chip_id(), i, num_dev_tasks * sizeof(uint32_t));
        uint64_t ptr_dev_kernel_base = bingo_l1_alloc(get_current_chip_id(), i, num_dev_tasks * sizeof(uint32_t));
        uint64_t ptr_global_id_to_dev_id_base = bingo_l1_alloc(get_current_chip_id(), i, num_tasks * sizeof(int32_t));
        // Copy the data using soc dma
        // Copy dev arg list
        sys_dma_blk_memcpy(get_current_chip_id(),
                            ptr_dev_arg_base,
                            (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), dev_arg_base_addr),
                            num_dev_tasks * sizeof(uint32_t));
        // Copy dev kernel list
        sys_dma_blk_memcpy(get_current_chip_id(),
                            ptr_dev_kernel_base,
                            (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), dev_kernel_base_addr),
                            num_dev_tasks * sizeof(uint32_t));
        // Copy global task id to dev task id list
        sys_dma_blk_memcpy(get_current_chip_id(),
                            ptr_global_id_to_dev_id_base,
                            (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), global_task_id_to_dev_task_id_base_addr),
                            num_tasks * sizeof(int32_t));
        writew(ptr_dev_arg_base,                 (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_arg_ptr_addr(i)));
        writew(ptr_dev_kernel_base,              (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_kernel_ptr_addr(i)));
        writew(ptr_global_id_to_dev_id_base, (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_global_id_to_dev_id_addr(i)));
    }
    // Init the power manager
    bingo_hw_scheduler_init_pm();
    // Init the task desc list base and num tasks
    writew(task_desc_list_base>>32,       (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_task_desc_base_hi_addr()));
    writew((uint32_t)task_desc_list_base, (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_task_desc_base_lo_addr()));
    writew(num_tasks,                     (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_num_task_addr()));
    // Start the HW scheduler to load the task list
    writew(1,                             (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_start_bingo_hw_manager_addr()));
    // Tell the device that the host init is done
    writew(1,                             (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_host_init_done_addr()));
    asm volatile("fence" ::: "memory");
}

uint32_t bingo_hw_scheduler(uint64_t* host_arg_list, uint64_t* host_kernel_list, int32_t* global_task_id_to_host_task_id){
    uint32_t current_global_task_id;
    int32_t current_host_task_id;
    uint64_t current_arg_ptr;
    uint64_t current_kernel_ptr;
    uint64_t kernel_return_value;
    uint32_t err=0;
    while (1) {
        // 1. First read the ready queue
        BINGO_TRACE_MARKER(BINGO_TRACE_MGR_GET_READY_START);
        current_global_task_id = bingo_hw_scheduler_get_global_task_id();
        BINGO_TRACE_MARKER(BINGO_TRACE_MGR_GET_READY_END);
        // 2. Then we get the host task id from the global task id
        BINGO_TRACE_MARKER(BINGO_TRACE_MGR_PREP_START);
        current_host_task_id = bingo_hw_scheduler_get_host_task_id(global_task_id_to_host_task_id, current_global_task_id);
        if (current_host_task_id == -1){
            BINGO_PRINTF(1, "Chip(%x, %x): [Host]: Error: Invalid host task id for global task id %d\r\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y(),
                   current_global_task_id);
            err=1;
            break;
        }
        // Then get the task arg and fn ptr from the task list
        current_arg_ptr = *bingo_hw_scheduler_get_host_arg(host_arg_list, current_host_task_id);
        current_kernel_ptr = *bingo_hw_scheduler_get_host_kernel(host_kernel_list, current_host_task_id);
        BINGO_TRACE_MARKER(BINGO_TRACE_MGR_PREP_END);
        BINGO_PRINTF(1, "Chip(%x, %x): [Host]: Task %d Info: Host tid=%d, arg ptr=0x%lx, kernel ptr=0x%lx\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(),
               current_global_task_id, current_host_task_id,
               current_arg_ptr, current_kernel_ptr);
        BINGO_PRINTF(2, "Chip(%x, %x): [Host]: Task %d Info: Running Host kernel ...\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(),
               current_global_task_id);
        // 3. Then run the host kernel
        BINGO_TRACE_MARKER(BINGO_TRACE_MGR_RUN_KERNEL_START);
        kernel_return_value = ((uint64_t (*)(uint64_t))current_kernel_ptr)(current_arg_ptr);
        BINGO_TRACE_MARKER(BINGO_TRACE_MGR_RUN_KERNEL_END);

        // 4. Finally write back to the done queue
        BINGO_TRACE_MARKER(BINGO_TRACE_MGR_WRITE_DONE_START);
        if (kernel_return_value == BINGO_RET_SUCC){
            BINGO_PRINTF(2, "Chip(%x, %x): [Host]: Task %d Info: Succ!\r\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y(),
                   current_global_task_id);
            // The normal case, write back to the done queue
            bingo_hw_scheduler_write_done_queue(current_global_task_id);
            BINGO_TRACE_MARKER(BINGO_TRACE_MGR_WRITE_DONE_END);
        } else if (kernel_return_value == BINGO_RET_EXIT){
            BINGO_PRINTF(2, "Chip(%x, %x): [Host]: Task %d Info: Exiting ...\r\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y(),
                   current_global_task_id);
            // The host exit task should be the last task
            BINGO_PRINTF(2, "Chip(%x, %x): [Host]: All Task Done!\r\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y());
            BINGO_TRACE_MARKER(BINGO_TRACE_MGR_WRITE_DONE_END);
            break;
        } else if (kernel_return_value == BINGO_RET_FAIL){
            // Error case
            BINGO_PRINTF(1, "Chip(%x, %x): [Host]: Task %d Info: Error!!!, error code=%d\r\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y(),
                   current_global_task_id, kernel_return_value);
            err = kernel_return_value;
            BINGO_TRACE_MARKER(BINGO_TRACE_MGR_WRITE_DONE_END);
            break;
        } else {
            // Unknown return value
            BINGO_PRINTF(1, "Chip(%x, %x): [Host]: Task %d Info: Unknown return value from host kernel: %d\r\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y(),
                   current_global_task_id, kernel_return_value);
            err = kernel_return_value;
            BINGO_TRACE_MARKER(BINGO_TRACE_MGR_WRITE_DONE_END);
            break;
        }
    }
    return err;
}