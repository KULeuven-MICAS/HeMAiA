# Libbingo Runtime

Libbingo is a lightweight bare‑metal task dependency runtime for the **HeMAiA** (Heterogeneous Multiple AI Accelerator) platform. It automates:

* Task registration & dependency tracking (local + cross‑chip)
* Event‑driven scheduling with an O(1) ready queue
* Inter‑chip task completion notification using hardware mailboxes
* Host ↔ Device mailbox/message encoding (H2H, H2C, C2H)

Its design philosophy borrows ideas from LLVM `libomptarget`, PULP `libhero`, and modern decentralized schedulers, while remaining small enough for bring‑up and experimentation.

---
## 1. Key Features
* Decentralized multi‑chip task graph execution
* Blocking & non‑blocking mailbox API with timeouts
* O(1) ring‑buffer based ready queue (no full rescans)
* Explicit local vs. remote dependency bookkeeping
* Compact inter‑chip 64‑bit message format + 32‑bit C2H encoded return
---
## 2. High‑Level Architecture

```
	+----------------------- Host (Chip X) ----------------------+
	|                                                            |
	|  Task Array  + Scheduler (ready ring)                      |
	|      |             |                                       |
	|      v             | enqueue/dequeue                      |
	|  +----------+  +--------+                                 |
	|  |  Task T  |  | Ready  |  <---- mailbox events ---------- | <--- Messages from remote chips
	|  +----------+  | Queue  |                                 |
	|        |        +--------+                                |
	|        v             | dispatch                           |
	|   HW Offload / Device Mailbox                             |
	+------------------------------------------------------------+
```

Core concepts:
* `bingo_task_t`: metadata (function pointer / device entry, dependency counters, successors, placement).
* `bingo_chip_sched_t`: per‑chip scheduler state (ready ring buffer, mask, indices).
* Mailboxes: memory‑mapped registers with status bits (bit0 empty, bit1 full) enforcing single‑producer/single‑consumer discipline per direction.
* Messages: fixed layout words encoding type, src chip, sequence, payload (task id or other control).

---
## 3. Programming Model (Host Side)
1. Allocate / initialize communication structures (heaps, comm buffer).
2. Discover device kernels (symbol table exchange) and obtain function addresses.
3. Create tasks using `bingo_task_create(func_addr, arg_addr, chip_id, cluster_id)`.
4. Establish dependencies with `bingo_task_add_depend(child, parent)`.
5. Start / enter scheduling loop `bingo_runtime_schedule(...)` (or a top‑level convenience wrapper).
6. Tasks become ready when their local and remote predecessor counters reach zero. Remote completions arrive via mailbox messages.
7. Scheduler dequeues ready tasks and offloads them to their destination chip/cluster.
8. Completion events (locally or via messages) decrement successor counters and possibly enqueue new ready tasks.

---
## 4. Core API Quick Reference
(Refer to `include/libbingo/bingo_api.h` for authoritative signatures.)

| Category        | Function (summary) |
|-----------------|--------------------|
| Task            | `bingo_task_create`, `bingo_task_add_depend` |
| Scheduling      | `bingo_runtime_schedule`, `bingo_close_all_clusters` |
| Mailbox (H2H)   | `bingo_write_h2h_mailbox`, `bingo_read_h2h_mailbox`, `bingo_try_write_h2h_mailbox`, `bingo_try_read_h2h_mailbox` |
| Mailbox (H2C/C2H)| Device ↔ host variants (see `mailbox.h`, `hw_mailbox.c`) |
| Message encode  | `bingo_encode_ic_msg`, `bingo_decode_ic_msg`, `bingo_encode_c2h_msg` |
| Heaps           | `bingo_get_l2_heap_manager`, `bingo_get_l3_heap_manager` |

Error codes (typical):
```
BINGO_MB_OK = 0
BINGO_MB_ERR_PARAM
BINGO_MB_ERR_TIMEOUT
```

Mailbox timeout API contract: `timeout_cycles == 0` → wait indefinitely; else abort after elapsed cycles.

---
## 5. Mailbox Semantics
Each mailbox has a status register with defined bit usage:
* bit0 (`EMPTY`): 1 when slot holds no valid data
* bit1 (`FULL`): 1 when slot contains valid data

Write sequence (producer):
1. Poll until `EMPTY==1 && FULL==0` (or timeout)
2. Write data payload register
3. Set FULL (HW or store ordering) — consumer observes FULL=1

Read sequence (consumer):
1. Poll until `FULL==1` (blocking API) or return `TRY_AGAIN`
2. Read payload
3. Clear FULL / set EMPTY so producer can write next

The implementation wraps these with cycle‑based timeout and optional relaxed spinning (`bingo_csleep`) to reduce contention.

---
## 6. Message Formats
### Inter‑Chip (64‑bit)
```
|63        56|55      48|47                 32|31                 0|
|  msg_type  | src_chip |   sequence / ext    |      payload       |
```
* `msg_type`: enumeration (task done, wake, control...)
* `src_chip`: physical chip ID
* `sequence`: monotonic counter (future; currently placeholder)
* `payload`: task identifier or command argument

### C2H Return (32‑bit packed)
```
|31    24|23      16|15        8|7        0|
| flags  | cluster  |  task_id  |  rsvd    |
```
Encoding/decoding done via shift/mask helpers (no C bitfields to keep ABI stable across compilers).

---
## 7. Scheduler & Ready Queue

The ready queue is a power‑of‑two ring buffer with `(head, tail, mask)` enabling:
* O(1) enqueue/dequeue
* Fast fullness/emptiness checks
* Minimal branches in the hot path

Transitions:
```
 Task dep counter -> 0  ---> enqueue (if not already queued) ---> dispatcher issues offload
							  ^
					    mailbox message / local completion
```

This replaces earlier O(N) scans per iteration, lowering latency and scaling with active rather than total tasks.

---
## 8. Dependency Model
Each task tracks:
* `local_remaining`: # of predecessors executing on the same chip
* `remote_remaining`: # of predecessors whose completion arrives via inter‑chip messages
* Successor lists: local successor array + remote successor descriptors `(chip_id, task_id)`

When a task finishes locally, the runtime decrements local successors directly; for remote successors it emits one inter‑chip message per remote chip grouping.

---
## 9. Memory Management
Libbingo uses **O1Heap** arenas per chip & per level:
* L2 heap: low‑latency small allocations (metadata, task descriptors)
* L3 heap: larger buffers / user data

Helper accessors expose arena managers. Alignments follow O1Heap requirements (usually 64 B). A failed allocation should be instrumented (roadmap includes diagnostics for capacity & fragmentation stats).

---
## 10. Multi‑Chip Execution Flow
1. Host initializes per‑chip heaps & communication buffer
2. Device publishes symbol table; host reads kernel addresses
3. Host creates full task graph (possibly spanning chips)
4. Initial ready tasks enqueued (those with zero preds)
5. Scheduler loop:
   * Dequeue ready task
   * Offload to target chip/cluster (writes kernel addr + args + interrupt)
   * Process inbound mailbox messages (mark remote completions)
   * Newly satisfied tasks enqueued
6. Termination when all tasks completed (tracked count or explicit final task)

---
## 11. Example (Simplified)
```c

// Task list
bingo_task_t *task_list[2] = NULL;
// Acquire device kernel addresses
uint32_t kernel_fn = get_device_function("__snax_kernel_foo");

// Prepare args structure in L3
foo_args_t *args = o1heapAllocate(bingo_get_l3_heap_manager(chip_id), sizeof(*args));
args->param = 42;

// Create tasks
bingo_task_t *t0 = bingo_task_create(kernel_fn, (uint32_t)(uintptr_t)args, chip_id, 0);
bingo_task_t *t1 = bingo_task_create(kernel_fn, (uint32_t)(uintptr_t)args, chip_id, 0);

// Dependency: t0 -> t1
bingo_task_add_depend(t1, t0);
// Add the task to task list
task_list[0] = t0;
task_list[1] = t1;
// Run scheduler (blocking until completion)
bingo_runtime_schedule(task_list, 2 /*num_tasks*/);
```