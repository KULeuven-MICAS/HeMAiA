#pragma once

// ============================================================================
// Performance Tracing for Bingo
// ============================================================================
// This mechanism uses "Magic NOPs" to inject markers into the instruction trace
// without affecting the architectural state of the processor.
//
// The markers are implemented as 'xori x0, x0, IMM' instructions.
// - They execute as a valid NOP (writing to zero register).
// - They carry a 12-bit immediate payload (IMM) visible in the trace dump.
//
// These markers are parsed by post-processing scripts to generate timelines
// (e.g., for Perfetto).

// Enable/Disable Tracing
// Define BINGO_PERF_TRACING before including this header or via compiler flags.
#ifdef BINGO_PERF_TRACING

// The Magic NOP Macro
// Uses the immediate value %0 (limited to 12 bits: 0-4095)
// We use xori instead of ori to avoid spike-dasm decoding it as prefetch
#define BINGO_TRACE_MARKER(id) asm volatile("xori x0, x0, %0" :: "i"(id))

#else

// If tracing is disabled, these compile to nothing.
#define BINGO_TRACE_MARKER(id) ((void)0)

#endif

// ============================================================================
// Trace Marker IDs (Events)
// ============================================================================
// Hierarchical ID scheme (max 12 bits):
// 0x0xx: BINGO SW Manager Events 
// 0x1XX: BINGO HW Manager Events
// 0x2XX: Kernel Configuration Events
// 0x3XX: Accelerator Execution Events
// 0x4XX: Cross-chip synchronization events


// // --- BINGO SW Manager Events ---
// Marks the lifespan of a task within the SW Manager loop
#define BINGO_TRACE_SW_MGR_INIT_TASK_QUEUE_START     0x010 // Start processing
#define BINGO_TRACE_SW_MGR_INIT_TASK_QUEUE_END       0x011 // End processing
#define BINGO_TRACE_SW_MGR_ENQUEUE_LOCAL_READY_TASKS_START   0x012
#define BINGO_TRACE_SW_MGR_ENQUEUE_LOCAL_READY_TASKS_END     0x013
#define BINGO_TRACE_SW_MGR_ENQUEUE_REMOTE_READY_TASKS_START  0x014
#define BINGO_TRACE_SW_MGR_ENQUEUE_REMOTE_READY_TASKS_END    0x015
#define BINGO_TRACE_SW_MGR_SCHED_READY_TASKS_START          0x016
#define BINGO_TRACE_SW_MGR_SCHED_READY_TASKS_END  0x017

// --- Hardware Manager Events ---
// Marks the lifespan of a task within the HW Manager loop
#define BINGO_TRACE_MGR_GET_READY_START        0x110 // Start reading Ready Queue
#define BINGO_TRACE_MGR_GET_READY_END          0x111 // End reading Ready Queue
#define BINGO_TRACE_MGR_PREP_START             0x112 // Start preparing kernel (get args, ptrs)
#define BINGO_TRACE_MGR_PREP_END               0x113 // End preparing kernel
#define BINGO_TRACE_MGR_RUN_KERNEL_START       0x114 // Start running kernel
#define BINGO_TRACE_MGR_RUN_KERNEL_END         0x115 // End running kernel
#define BINGO_TRACE_MGR_WRITE_DONE_START       0x116 // Start writing Done Queue
#define BINGO_TRACE_MGR_WRITE_DONE_END         0x117 // End writing Done Queue
#define BINGO_TRACE_KERNEL_ARG_PARSE_START     0x118 // Parsing kernel arguments
#define BINGO_TRACE_KERNEL_ARG_PARSE_END       0x119 // Finished parsing kernel arguments

// --- Kernel Internal Phases ---
// These are used inside individual kernels
// Non-computation kernels (Dummy, Exit)
#define BINGO_TRACE_DUMMY_KERNEL_START  0x200
#define BINGO_TRACE_DUMMY_KERNEL_END    0x201

// Computation Kernels: Configuration Phase
// IDMA
#define BINGO_TRACE_IDMA_CFG_START         0x210
#define BINGO_TRACE_IDMA_CFG_END           0x211
// XDMA
#define BINGO_TRACE_XDMA_CFG_START         0x220
#define BINGO_TRACE_XDMA_CFG_END           0x221
// GEMM FULL
#define BINGO_TRACE_GEMM_FULL_CFG_START    0x230
#define BINGO_TRACE_GEMM_FULL_CFG_END      0x231
// Minimal GEMM
#define BINGO_TRACE_GEMM_MIN_CFG_START     0x240
#define BINGO_TRACE_GEMM_MIN_CFG_END       0x241
// SIMD
#define BINGO_TRACE_SIMD_CFG_START         0x250
#define BINGO_TRACE_SIMD_CFG_END           0x251
// HOST IDMA
#define BINGO_TRACE_HOST_IDMA_CFG_START    0x260
#define BINGO_TRACE_HOST_IDMA_CFG_END      0x261

// Computation Kernels: Compute/Run Phase
// IDMA
#define BINGO_TRACE_IDMA_RUN_START        0x310
#define BINGO_TRACE_IDMA_RUN_END          0x311
// XDMA
#define BINGO_TRACE_XDMA_RUN_START        0x320
#define BINGO_TRACE_XDMA_RUN_END          0x321
// GEMM FULL
#define BINGO_TRACE_GEMM_FULL_RUN_START   0x330
#define BINGO_TRACE_GEMM_FULL_RUN_END     0x331
// Minimal GEMM
#define BINGO_TRACE_GEMM_MIN_RUN_START    0x340
#define BINGO_TRACE_GEMM_MIN_RUN_END      0x341
// SIMD (generic host RVV kernels: reduce/silu/softmax/rmsnorm/add)
#define BINGO_TRACE_SIMD_RUN_START        0x350
#define BINGO_TRACE_SIMD_RUN_END          0x351
// HOST QUANTIZE (fp16/fp32 activation -> int8 GEMM-operand requant on the CVA6)
#define BINGO_TRACE_QUANT_RUN_START       0x352
#define BINGO_TRACE_QUANT_RUN_END         0x353
// HOST SCALAR_BCAST (per-row sqrt/recip/neg special-functions the xDMA can't do)
#define BINGO_TRACE_SCALAR_RUN_START      0x354
#define BINGO_TRACE_SCALAR_RUN_END        0x355
// HOST IDMA
#define BINGO_TRACE_HOST_IDMA_RUN_START   0x360
#define BINGO_TRACE_HOST_IDMA_RUN_END     0x361

// ============================================================================
// --- Cross-chip synchronization (0x4XX) ---
// ============================================================================
// Emitted by the chip barriers in target/sw/device/runtime/src/chip_sync.h.
//
// The two mechanisms get DISTINCT ids on purpose: the unified
// snrt_chip_global_barrier(use_sw) lets one application use both, and a trace that
// conflated them could not say which one paid for a given interval.
//
// START/END bracket the whole barrier and are emitted by EVERY core, so the pair also
// shows the intra-chip rendezvous. ANNOUNCE and WAIT are emitted only by the chip's
// representative core and split the cross-chip cost into its network term and its skew
// term -- the decomposition that says whether a barrier is slow because the fabric is
// slow or because one chip arrived late.
//
// 0x400-0x40F are left free for the min-SFR benchmark's own kernel markers.

// --- Broadcast (in-router) chip barrier ---
#define BINGO_TRACE_HW_CHIP_BARRIER_START           0x410
#define BINGO_TRACE_HW_CHIP_BARRIER_END             0x411
#define BINGO_TRACE_HW_CHIP_BARRIER_ANNOUNCE_START  0x412
#define BINGO_TRACE_HW_CHIP_BARRIER_ANNOUNCE_END    0x413
#define BINGO_TRACE_HW_CHIP_BARRIER_WAIT_START      0x414
#define BINGO_TRACE_HW_CHIP_BARRIER_WAIT_END        0x415

// --- Pure-software chip barrier ---
#define BINGO_TRACE_SW_CHIP_BARRIER_START           0x420
#define BINGO_TRACE_SW_CHIP_BARRIER_END             0x421
#define BINGO_TRACE_SW_CHIP_BARRIER_ANNOUNCE_START  0x422
#define BINGO_TRACE_SW_CHIP_BARRIER_ANNOUNCE_END    0x423
#define BINGO_TRACE_SW_CHIP_BARRIER_WAIT_START      0x424
#define BINGO_TRACE_SW_CHIP_BARRIER_WAIT_END        0x425

// --- Sync-probe kernel (cross-chip synchronization latency measurement) ---
// The task body itself; the interval between two probes is the quantity of interest.
#define BINGO_TRACE_SYNC_PROBE_START                0x430
#define BINGO_TRACE_SYNC_PROBE_END                  0x431
