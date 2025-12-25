// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

// The bingo host-side runtime API
// Mainly three parts:
// 1) Memory allocation (L2 and L3)
// 2) Mailbox communication
// 3) Task management

#pragma once

#include <stdint.h>
#include <stdbool.h>
// Memory allocators
#include "allocators.h"
// HW mailbox
#include "mailbox.h"
// Chip id
#include "chip_id.h"
// Bingo utils
#include "bingo_utils.h"
// Generic I/O
#include "io.h"
// printf
#include "uart.h"
// Occamy
#include "occamy.h"
// Heterogeneous runtime
#include "heterogeneous_runtime.h"
// kernel args
#include "device_kernel_args.h"
#include "host_kernel_args.h"
#define HOST_SLEEP_CYCLES 100
#define MAX_SUCCESSORS 8                 // Maximum number of (local) successor tasks tracked directly
#define BINGO_MAX_REMOTE_SUCC 8          // Maximum number of remote successors (fan-out messages)

// Device control commands
#define MBOX_DEVICE_READY (0x01U)
#define MBOX_DEVICE_START (0x02U)
#define MBOX_DEVICE_BUSY (0x03U)
#define MBOX_DEVICE_DONE (0x04U)
#define MBOX_DEVICE_STOP (0x0FU)

///////////////////////////////
///// Data Structure      /////
///////////////////////////////

//////////////////////////////////
// The BINGO SW TASk Descriptor///
//////////////////////////////////
// The task is running on the cluster
// Forward declaration for remote successor struct (defined below)
struct bingo_remote_succ;

// Distributed runtime task descriptor.
// NOTE: We keep original fields for backward compatibility and extend with
// explicit local/remote dependency tracking required for decentralized scheduling.
typedef struct task {
  // ---- Identity & user-specified execution info ----
  uint16_t task_id;             // Globally unique task id (stable across chips)
  uint32_t fn_ptr;              // Device-visible function pointer (kept 32-bit: device address space)
  uint32_t args_ptr;            // Pointer to argument struct in device-visible memory
  uint8_t  assigned_chip_id;    // Target chip for execution
  uint8_t  assigned_cluster_id; // Optional intra-chip cluster / core group

  // ---- Dependency bookkeeping (decentralized) ----

  // Initial counts (for debugging / reset) and live remaining counters for local and remote predecessors.
  uint8_t  local_pred_initial;      // Number of predecessors that reside on the same chip
  uint8_t  local_pred_remaining;    // Remaining local predecessors (decremented by local completions)
  uint8_t  remote_pred_initial;     // Number of predecessors that reside on other chips
  uint8_t  remote_pred_remaining;   // Remaining remote predecessors (decremented by mailbox notifications)

  // ---- Successor fan-out (LOCAL) ----
  uint8_t  num_local_successors;           // Number of local successor tasks in 'local_successors'
  struct task *local_successors[MAX_SUCCESSORS]; // Direct pointers to local successors (fast in-process decrement)

  // ---- Successor fan-out (REMOTE) ----
  uint8_t  num_remote_successors;   // Number of remote successors in 'remote_successors'
  struct bingo_remote_succ {
    uint8_t  chip_id;               // Destination chip id
    uint16_t task_id;               // Remote task id (no pointer available cross-process)
  } remote_successors[BINGO_MAX_REMOTE_SUCC];

  // ---- Runtime state flags ----
  bool offloaded;                   // Task has been submitted to device on its assigned chip
  bool completed;                   // Device signalled completion locally
  bool enqueued_ready;              // Placed into ready queue (prevents double insertion)
  bool completion_notified;         // Remote completion notification already broadcast (only once)

  // ---- Optional diagnostics / profiling ----
  uint32_t debug_seq_issue;         // Sequence number when issued (monotonic per chip)
  uint32_t debug_seq_complete;      // Sequence number when completed
} bingo_task_t;



// ---------------------------------------------------------------
// Intra-chip (C2H) mailbox message format (32-bit) from cluster to host
// ---------------------------------------------------------------
// Unified with inter-chip style: provide encode/decode helpers instead of C bitfields
// to avoid implementation-defined layout/packing.
// Layout (MSB -> LSB):
//  [31:28]  reserved (future use: version, severity, etc.)
//  [27:12]  task_id   (16 bits)
//  [11:4]   cluster_id (8 bits)
//  [3:0]    flag (4 bits) -> uses MBOX_DEVICE_* constants (fits in 4 bits)
// A single 32-bit word is enough for most events. Extra payload (e.g., kernel cycles)
// follows in subsequent words when the flag semantics require it (e.g., DONE => next
// word = kernel cycles). This mirrors the pattern already used but formalizes the first
// word structure.

#define BINGO_C2H_FLAG_MASK        0xFu
#define BINGO_C2H_CLUSTER_MASK     0xFFu
#define BINGO_C2H_TASK_MASK        0xFFFFu
#define BINGO_C2H_FLAG_SHIFT       0
#define BINGO_C2H_CLUSTER_SHIFT    4
#define BINGO_C2H_TASK_SHIFT       12
#define BINGO_C2H_RESERVED_SHIFT   28

typedef struct {
  uint16_t task_id;   // 16-bit task identifier
  uint8_t  cluster_id;// Source cluster id
  uint8_t  flag;      // Lower 4 bits meaningful (MBOX_DEVICE_*)
  uint8_t  reserved;  // Upper 4 bits of the original packed word (kept for future)
} bingo_c2h_msg_fields_t;

static inline uint32_t bingo_c2h_msg_encode(bingo_c2h_msg_fields_t f) {
  return ((uint32_t)(f.reserved & 0xF)    << BINGO_C2H_RESERVED_SHIFT) |
         ((uint32_t)(f.task_id  & BINGO_C2H_TASK_MASK)    << BINGO_C2H_TASK_SHIFT) |
         ((uint32_t)(f.cluster_id & BINGO_C2H_CLUSTER_MASK) << BINGO_C2H_CLUSTER_SHIFT) |
         ((uint32_t)(f.flag & BINGO_C2H_FLAG_MASK) << BINGO_C2H_FLAG_SHIFT);
}

static inline bingo_c2h_msg_fields_t bingo_c2h_msg_decode(uint32_t w) {
  bingo_c2h_msg_fields_t f;
  f.flag       = (uint8_t)((w >> BINGO_C2H_FLAG_SHIFT) & BINGO_C2H_FLAG_MASK);
  f.cluster_id = (uint8_t)((w >> BINGO_C2H_CLUSTER_SHIFT) & BINGO_C2H_CLUSTER_MASK);
  f.task_id    = (uint16_t)((w >> BINGO_C2H_TASK_SHIFT) & BINGO_C2H_TASK_MASK);
  f.reserved   = (uint8_t)((w >> BINGO_C2H_RESERVED_SHIFT) & 0xF);
  return f;
}


// ------------------------------
// Inter-chip mailbox message format (64-bit)
// ------------------------------
// We encode control information into a single 64-bit dword for efficient mailbox usage.
// Layout (MSB -> LSB):
//  [63:56] message type
//  [55:48] source chip id
//  [47:16] sequence (optional tracing / wrap-around) OR reserved (can be zero)
//  [15:0]  payload (task id or other type-specific data)
// Rationale: keeps task id as full 32-bit field (scales to large DAGs) while providing
// compact type + origin metadata; allows simple mask/shift ops (no branching) and
// future extensibility via sequence for ordering / debugging.

typedef enum {
  BINGO_MSG_TASK_COMPLETE = 0x01,  // Payload = completed task id
  BINGO_MSG_TASK_READY    = 0x02,  // (Optional future) Payload = task id becoming ready remotely
  BINGO_MSG_HEARTBEAT     = 0x7E,  // Liveness / watchdog
  BINGO_MSG_DIAG          = 0x7F   // Diagnostic / test message
} bingo_msg_type_t;

typedef struct {
  uint8_t  type;       // bingo_msg_type_t
  uint8_t  src_chip;   // Originating chip id
  uint32_t seq;        // Sequence (debug / ordering) - wrap allowed
  uint16_t payload;    // Task id or type-specific value
} bingo_msg_fields_t;

// Pack fields into 64-bit word
static inline uint64_t bingo_msg_encode(bingo_msg_fields_t f) {
  return ((uint64_t)f.type     << 56) |
         ((uint64_t)f.src_chip << 48) |
         ((uint64_t)f.seq      << 16) |
         ((uint64_t)f.payload);
}

// Unpack 64-bit word into structured fields
static inline bingo_msg_fields_t bingo_msg_decode(uint64_t w) {
  bingo_msg_fields_t f;
  f.type     = (uint8_t)(w >> 56);
  f.src_chip = (uint8_t)(w >> 48);
  f.seq      = (uint32_t)(w >> 16);
  f.payload  = (uint16_t)(w & 0xFFFFu);
  return f;
}

// ------------------------------
// Per-chip scheduler runtime state (host-side)
// ------------------------------
// Maintains ready queue and accounting; declared here for visibility to tests / tools.
typedef struct {
  uint8_t  chip_id;          // This scheduler's chip
  uint16_t ready_cap;        // Capacity of ring buffer (power-of-two recommended)
  uint16_t ready_mask;       // Mask for fast modulo (ready_cap - 1) when power-of-two
  uint16_t ready_head;       // Dequeue index
  uint16_t ready_tail;       // Enqueue index
  bingo_task_t **ready_ring; // Array of task pointers (allocated at init)
  uint32_t inflight;         // Tasks currently offloaded and not yet completed
  uint32_t completed;        // Cumulative completions (local)
  uint64_t tx_msgs;          // Messages transmitted
  uint64_t rx_msgs;          // Messages received
  uint32_t seq_counter;      // Monotonic sequence for message stamping
} bingo_chip_sched_t;

//////////////////////////////////////
// BINGO HW Manager Task Descriptor //
//////////////////////////////////////

typedef struct hw_manager_task {
    bool     task_type;            // Task Type 0: Normal Task, 1: Dummy Task 
    uint16_t task_id;              // Task ID
    uint8_t  assigned_chiplet_id;  // Target chiplet for execution
    uint8_t  assigned_cluster_id;  // Target cluster for execution
    uint8_t  assigned_core_id;     // Target core for execution
    bool     dep_check_enabled;    // Whether dependency checking is enabled
    uint8_t  dep_check_code;       // Dependency check code
    bool     dep_set_enabled;      // Whether dependency setting is enabled
    bool     dep_set_all_chiplet;  // Whether to set dependency on all chiplets
    uint8_t  dep_set_code;         // Chiplet ID to set dependency
    uint8_t  dep_set_chiplet_id;   // Chiplet ID to set dependency
    uint8_t  dep_set_cluster_id;   // Cluster ID to set dependency
} bingo_hw_manager_task_desc_t;

static inline uint64_t encode_bingo_hw_manager_task_desc(bingo_hw_manager_task_desc_t desc) {
    uint64_t encoded = 0;

    // Task type (1 bit) at encoded[0]
    encoded |= ENCODE_BITFIELD(desc.task_type, TASK_TYPE_WIDTH, TASK_TYPE_SHIFT);

    // Task ID (12 bits) at encoded[12:1]
    encoded |= ENCODE_BITFIELD(desc.task_id, TASK_ID_WIDTH, TASK_ID_SHIFT);

    // Assigned chiplet ID (8 bits)
    encoded |= ENCODE_BITFIELD(desc.assigned_chiplet_id, ASSIGNED_CHIPLET_ID_WIDTH, ASSIGNED_CHIPLET_ID_SHIFT);

    // Assigned cluster ID (N_CLUSTERS_WIDTH bits)
    encoded |= ENCODE_BITFIELD(desc.assigned_cluster_id, ASSIGNED_CLUSTER_ID_WIDTH, ASSIGNED_CLUSTER_ID_SHIFT);

    // Assigned core ID (N_CORES_WIDTH bits)
    encoded |= ENCODE_BITFIELD(desc.assigned_core_id, ASSIGNED_CORE_ID_WIDTH, ASSIGNED_CORE_ID_SHIFT);

    // Dep check enabled (1 bit)
    encoded |= ENCODE_BITFIELD(desc.dep_check_enabled, DEP_CHECK_ENABLED_WIDTH, DEP_CHECK_ENABLED_SHIFT);

    // Dep check code (N_CORES_PER_CLUSTER bits)
    encoded |= ENCODE_BITFIELD(desc.dep_check_code, DEP_CHECK_CODE_WIDTH, DEP_CHECK_CODE_SHIFT);

    // Dep set enabled (1 bit)
    encoded |= ENCODE_BITFIELD(desc.dep_set_enabled, DEP_SET_ENABLED_WIDTH, DEP_SET_ENABLED_SHIFT);

    // Dep set all chiplet (1 bit)
    encoded |= ENCODE_BITFIELD(desc.dep_set_all_chiplet, DEP_SET_ALL_CHIPLET_WIDTH, DEP_SET_ALL_CHIPLET_SHIFT);

    // Dep set code (N_CORES_PER_CLUSTER bits)
    encoded |= ENCODE_BITFIELD(desc.dep_set_code, DEP_SET_CODE_WIDTH, DEP_SET_CODE_SHIFT);

    // Dep set chiplet ID (N_CHIPLETS_WIDTH bits)
    encoded |= ENCODE_BITFIELD(desc.dep_set_chiplet_id, DEP_SET_CHIPLET_ID_WIDTH, DEP_SET_CHIPLET_ID_SHIFT);

    // Dep set cluster ID (N_CLUSTERS_WIDTH bits)
    encoded |= ENCODE_BITFIELD(desc.dep_set_cluster_id, DEP_SET_CLUSTER_ID_WIDTH, DEP_SET_CLUSTER_ID_SHIFT);

    return encoded;
}

////////////////////
///// API      /////
////////////////////

// Allocator init (L2 and L3)
int bingo_hemaia_system_mmap_init();

// Getters for heap managers
uint64_t bingo_get_l2_comm_buffer(uint8_t chip_id);
uint64_t bingo_get_l2_heap_manager(uint8_t chip_id);
uint64_t bingo_get_l3_heap_manager(uint8_t chip_id);
uint64_t bingo_get_l1_heap_manager(uint8_t chip_id, uint32_t cluster_id);

// Allocator API
uint64_t bingo_l1_alloc(uint8_t chip_id, uint32_t cluster_id, uint64_t size);
uint64_t bingo_l2_alloc(uint8_t chip_id, uint64_t size);
uint64_t bingo_l3_alloc(uint8_t chip_id, uint64_t size);
void bingo_l1_free(uint8_t chip_id, uint32_t cluster_id, uint64_t ptr);
void bingo_l2_free(uint8_t chip_id, uint64_t ptr);
void bingo_l3_free(uint8_t chip_id, uint64_t ptr);


// Mailbox read/write functions
/////////////////////////////////////
///// Host to Host Mailbox      /////
/////////////////////////////////////
// H2H mailbox status/error codes
#define BINGO_MB_OK            0
#define BINGO_MB_ERR_PARAM    -1
#define BINGO_MB_ERR_TIMEOUT  -2

// Blocking write: waits until space available or timeout
// timeout_cycles = 0 means wait forever
// retry_hint allows caller to observe loop iterations (can pass NULL)
int bingo_write_h2h_mailbox(uint8_t chip_id, uint64_t dword,
                            uint64_t timeout_cycles, uint32_t *retry_hint);

// Blocking read: waits until data available or timeout
// timeout_cycles = 0 means wait forever
// retry_hint allows caller to observe loop iterations (can pass NULL)
int bingo_read_h2h_mailbox(uint64_t *buffer,
                           uint64_t timeout_cycles, uint32_t *retry_hint);

// Non-blocking attempts
// Return: 1 = success (read/write performed), 0 = would block, <0 = error
int bingo_try_write_h2h_mailbox(uint8_t chip_id, uint64_t dword);
int bingo_try_read_h2h_mailbox(uint64_t *buffer);
////////////////////////////////////////
///// Host to Cluster Mailbox      /////
////////////////////////////////////////
// H2C mailbox
// There are one H2C mailbox for each cluster in a chip
// For example, if a chip has 4 clusters, it has 4 H2C mailboxes
// Host can only write to its own chip's H2C mailbox
// And since clusters are 32bit, the messages are 32bit wide
// Blocking write: waits until space available or timeout
// timeout_cycles = 0 means wait forever
// retry_hint allows caller to observe loop iterations (can pass NULL)
int bingo_write_h2c_mailbox(uint8_t cluster_id, uint32_t word, uint64_t timeout_cycles, uint32_t *retry_hint);
// Non-blocking attempts
// Return: 1 = success (read/write performed), 0 = would block, <0 = error
int bingo_try_write_h2c_mailbox(uint8_t cluster_id, uint32_t word);
////////////////////////////////////////
///// Cluster to Host Mailbox      /////
////////////////////////////////////////
// C2H mailbox
// There is only one C2H mailbox per chip
// Since clusters are 32bit, the messages are 32bit wide
// Host can only read from its own chip's C2H mailbox
// Blocking read: waits until data available or timeout
// timeout_cycles = 0 means wait forever
// retry_hint allows caller to observe loop iterations (can pass NULL)
int bingo_read_c2h_mailbox(uint32_t *buffer, uint64_t timeout_cycles, uint32_t *retry_hint);
// Non-blocking attempts
// Return: 1 = success (read/write performed), 0 = would block, <0 = error
int bingo_try_read_c2h_mailbox(uint32_t *buffer);


// Core Bingo Runtime Dependency Scheduling
// Create a new task
bingo_task_t *bingo_task_create(uint32_t fn_ptr, uint32_t args_ptr, uint8_t assigned_chip_id, uint8_t assigned_cluster_id);

// Declare dependency on the current task
void bingo_task_add_depend(bingo_task_t *task, bingo_task_t *dep_task);

// Offload the task (writes command + metadata to H2C mailbox of its assigned cluster)
void bingo_task_offload(bingo_task_t *task);

// Local (per-chip) decentralized scheduling loop; executes only tasks whose
// assigned_chip_id equals the current chip id. Handles offload, completion,
// inter-chip dependency notifications.
void bingo_runtime_schedule(bingo_task_t **task_list, uint32_t num_tasks);

void bingo_close_all_clusters(bingo_task_t **task_list, uint32_t num_tasks);