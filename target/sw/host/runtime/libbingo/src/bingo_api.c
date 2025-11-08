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
    comm_buffer_t* comm_buffer = (comm_buffer_t *)chiplet_addr_transform(SPM_NARROW_BASE_ADDR);
    printf("Chip(%x, %x): [Host] Comm buffer addr: %lx\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), (uintptr_t)comm_buffer);
    // L2 heap init
    // Start addr is the spm narrow
    // Size is half of the narrow spm
    // The rest is leave to stack
    uintptr_t l2_heap_start = ALIGN_UP((uintptr_t)comm_buffer + sizeof(comm_buffer_t), O1HEAP_ALIGNMENT);
    size_t l2_heap_size = ALIGN_UP(NARROW_SPM_SIZE / 2, O1HEAP_ALIGNMENT);
    O1HeapInstance *l2_heap_manager = o1heapInit((void *)l2_heap_start, l2_heap_size);
    printf("Chip(%x, %x): [Host] L2 heap start: %lx\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), l2_heap_start);
    // printf("Chip(%x, %x): [Host] L2 heap start: %lx, size: %lx, heap manager: 0x%lx\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), l2_heap_start, l2_heap_size, l2_heap_manager);
    if(!l2_heap_manager) {
        printf("Error when initializing L2 heap. \r\n");
        return -1;
    }
    // L3 heap init
    // Start addr is the l3 heap start symbol
    // Size is half of the wide spm
    uintptr_t l3_heap_start = ALIGN_UP(chiplet_addr_transform((uint64_t)(&__l3_heap_start)), O1HEAP_ALIGNMENT);
    size_t l3_heap_size = ALIGN_DOWN(chiplet_addr_transform((uint64_t)(&__wide_spm_end)) - 1, O1HEAP_ALIGNMENT) - l3_heap_start;
    O1HeapInstance *l3_heap_manager = o1heapInit((void *)l3_heap_start, l3_heap_size);
    printf("Chip(%x, %x): [Host] L3 heap start: %lx\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), l3_heap_start);
    // printf("Chip(%x, %x): [Host] L3 heap start: %lx, size: %lx, heap manager: 0x%lx\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), l3_heap_start, l3_heap_size, l3_heap_manager);
    if(!l3_heap_manager) {
        printf("Error when initializing L3 heap.\r\n");
        return -1;
    }
    return 0;
}

O1HeapInstance32 *bingo_get_l1_heap_manager(uint8_t chip_id, uint32_t cluster_id){
    // Notice the l1 heap is init by the cluster at the start of the TCDM of each cluster
    // Here we hack the o1heap sw to force a 32bit version of heap manager so that the host can access to the correct
    // heap information from the O1HeapInstanc32 structure
    return (O1HeapInstance32 *)chiplet_addr_transform_full(chip_id, (uint64_t)cluster_tcdm_start_addr(cluster_id));
}


comm_buffer_t *bingo_get_l2_comm_buffer(uint8_t chip_id){
    return (comm_buffer_t *)chiplet_addr_transform_full(chip_id, (uintptr_t)SPM_NARROW_BASE_ADDR);
}

O1HeapInstance *bingo_get_l2_heap_manager(uint8_t chip_id){
    return (O1HeapInstance *)chiplet_addr_transform_full(chip_id, ALIGN_UP((uintptr_t)SPM_NARROW_BASE_ADDR + sizeof(comm_buffer_t), O1HEAP_ALIGNMENT));
}

O1HeapInstance *bingo_get_l3_heap_manager(uint8_t chip_id){
    return (O1HeapInstance *)chiplet_addr_transform_full(chip_id, (uintptr_t)chiplet_addr_transform((uint64_t)(&__l3_heap_start)));
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

    uint64_t start_cycle = bingo_mcycle();
    uint32_t retries = 0;
    while (1) {
        uint64_t status = readd((uintptr_t)target_h2h_mailbox_status_addr);
        if (!BINGO_EXTRACT_BIT(status, 1)) {
            writed(dword, (uintptr_t)target_h2h_mailbox_write_addr);
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

// Task creation initializes counters & clears successor lists.
bingo_task_t *bingo_task_create(uint32_t fn_ptr, uint32_t args_ptr, uint8_t assigned_chip_id, uint8_t assigned_cluster_id) {

    bingo_task_t *task = o1heapAllocate(bingo_get_l2_heap_manager(get_current_chip_id()), sizeof(bingo_task_t));
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
        printf("[BINGO] Error: Null task pointer in bingo_task_offload.\r\n");
        return;
    }
    if (task->offloaded){
        printf("[BINGO] Warning: Task %u already offloaded.\r\n", task->task_id);
        return;
    }
    printf("Chip(%x, %x): [Host] Offloaded task %u -> chip %u cluster %u fn=0x%x args=0x%x\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), task->task_id,
           task->assigned_chip_id, task->assigned_cluster_id, task->fn_ptr, task->args_ptr); 
    // Send start command + task info via H2C mailbox
    uint32_t retry_hint_start, retry_hint_id, retry_hint_fn, retry_hint_args;
    uint32_t total_retries = 0;
    bingo_write_h2c_mailbox(task->assigned_cluster_id, (uint32_t)MBOX_DEVICE_START, 0, &retry_hint_start);
    bingo_write_h2c_mailbox(task->assigned_cluster_id, (uint32_t)task->task_id, 0, &retry_hint_id);
    bingo_write_h2c_mailbox(task->assigned_cluster_id, (uint32_t)task->fn_ptr, 0, &retry_hint_fn);
    bingo_write_h2c_mailbox(task->assigned_cluster_id, (uint32_t)task->args_ptr, 0, &retry_hint_args);
    total_retries += retry_hint_start + retry_hint_id + retry_hint_fn + retry_hint_args;
    if(total_retries > 100) {
        printf("Chip(%x, %x): [Host] Warning: High mailbox write retries (%u) when offloading task %u to chip %u cluster %u\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(), total_retries,
               task->task_id, task->assigned_chip_id, task->assigned_cluster_id);
    }
    task->offloaded = true;
   
}

// ------------------------------
// Decentralized (per-chip) runtime loop using H2H mailbox notifications.
// This host instance manages ONLY tasks whose assigned_chip_id == current chip.
// ------------------------------
static void bingo_process_mailbox_events(bingo_task_t **task_list, uint32_t num_tasks) {
    while (1) {
        uint64_t word;
        int r = bingo_try_read_h2h_mailbox(&word);
        if (r <= 0) break; // 0 would block, <0 error
        bingo_msg_fields_t f = bingo_msg_decode(word);
        if (f.type == BINGO_MSG_TASK_COMPLETE) {
            uint16_t tid = f.payload;
            if (tid < num_tasks) {
                bingo_task_t *t = task_list[tid];
                if (t->remote_pred_remaining) {
                    t->remote_pred_remaining--;
                }
            }
        }
    }
}

static void bingo_broadcast_completion(const bingo_task_t *t, uint8_t src_chip) {
    if (t->completion_notified) return;
    // Iterate remote successors and send message per unique chip (one per successor; future optimization: coalesce)
    for (uint8_t i = 0; i < t->num_remote_successors; i++) {
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

    uint8_t current_chip = get_current_chip_id();

    // Count local tasks
    uint16_t local_total = 0;
    for (uint32_t i = 0; i < num_tasks; i++) {
        if (task_list[i]->assigned_chip_id == current_chip) local_total++;
    }
    printf("Chip(%x, %x): [Host] Starting runtime schedule for %u local tasks\r\n",
           get_current_chip_loc_x(), get_current_chip_loc_y(), local_total);
    if (local_total == 0) return; // Nothing to do

    // Initialize scheduler instance (stack-local for now; could be static per chip)
    uint16_t cap = bingo_next_pow2(local_total * 2); // room for burst readiness
    bingo_task_t **ring = (bingo_task_t **)o1heapAllocate(bingo_get_l2_heap_manager(get_current_chip_id()), sizeof(bingo_task_t*) * cap);
    if (!ring) {
        printf("[BINGO] Error: Failed to allocate ready ring.\r\n");
        return;
    }
    bingo_chip_sched_t sched = {
        .chip_id = current_chip,
        .ready_cap = cap,
        .ready_mask = (uint16_t)(cap - 1),
        .ready_head = 0,
        .ready_tail = 0,
        .ready_ring = ring,
        .inflight = 0,
        .completed = 0,
        .tx_msgs = 0,
        .rx_msgs = 0,
        .seq_counter = 0
    };

    // Seed initial ready tasks (those with zero local & remote predecessors)
    for (uint32_t i = 0; i < num_tasks; i++) {
        bingo_task_t *t = task_list[i];
        if (t->assigned_chip_id != current_chip) continue;
        if (bingo_task_is_globally_ready(t)) {
            sched_enqueue(&sched, t);
        }
    }

    uint16_t local_completed = 0;
    while (local_completed < local_total) {
        // 1. Poll device C2H events (task completions etc.)
        while (1) {
            uint32_t word;
            int r = bingo_try_read_c2h_mailbox(&word);
            if (r <= 0) break; // no more
            bingo_c2h_msg_fields_t msg = bingo_c2h_msg_decode(word);
            switch (msg.flag & 0xF) {
                case MBOX_DEVICE_DONE: {
                    uint16_t tid = msg.task_id;
                    printf("Chip(%x, %x): [Host] Task %u completed on chip %u cluster %u \r\n",
                           get_current_chip_loc_x(), get_current_chip_loc_y(),
                           tid, current_chip, msg.cluster_id);
                    if (tid < num_tasks) {
                        bingo_task_t *completed_task = task_list[tid];
                        if (!completed_task->completed && completed_task->assigned_chip_id == current_chip) {
                            completed_task->completed = true;
                            local_completed++;
                            // Decrement local successor counters & enqueue those now globally ready
                            for (uint8_t s = 0; s < completed_task->num_local_successors; s++) {
                                bingo_task_t *suc = completed_task->local_successors[s];
                                if (suc->local_pred_remaining) {
                                    suc->local_pred_remaining--;
                                    if (bingo_task_is_globally_ready(suc) && !suc->offloaded) {
                                        sched_enqueue(&sched, suc);
                                    }
                                }
                            }
                            // Broadcast remote completion once
                            if (!completed_task->completion_notified) {
                                bingo_broadcast_completion(completed_task, current_chip);
                                completed_task->completion_notified = true;
                            }
                        }
                    }
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
            bingo_msg_fields_t f = bingo_msg_decode(word);
            if (f.type == BINGO_MSG_TASK_COMPLETE) {
                uint16_t tid = f.payload;
                if (tid < num_tasks) {
                    bingo_task_t *t = task_list[tid];
                    if (t->assigned_chip_id == current_chip && t->remote_pred_remaining) {
                        t->remote_pred_remaining--;
                        if (bingo_task_is_globally_ready(t) && !t->offloaded) {
                            sched_enqueue(&sched, t);
                        }
                    }
                }
            }
        }

        // 3. Offload as many ready tasks as possible (could add throttle / inflight cap later)
        while (1) {
            bingo_task_t *t = sched_dequeue(&sched);
            if (!t) break;
            if (!t->offloaded) {
                bingo_task_offload(t);
                sched.inflight++;
            }
        }

        // 4. Sleep a bit to reduce host busy waiting
        if (local_completed < local_total) {
            bingo_csleep(HOST_SLEEP_CYCLES);
        }
    }
    // Destroy the allocated ready ring
    o1heapFree(bingo_get_l2_heap_manager(get_current_chip_id()), sched.ready_ring);
}

void bingo_close_all_clusters(bingo_task_t **task_list, uint32_t num_tasks){
    uint8_t current_chip = get_current_chip_id();
    // Issue STOP to each cluster used
    bool cluster_stopped[N_CLUSTERS] = {0};
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
