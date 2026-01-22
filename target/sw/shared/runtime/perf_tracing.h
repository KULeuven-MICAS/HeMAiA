#pragma once

// ============================================================================
// Performance Tracing for Bingo
// ============================================================================
// This mechanism uses "Magic NOPs" to inject markers into the instruction trace
// without affecting the architectural state of the processor.
//
// The markers are implemented as 'ori x0, x0, IMM' instructions.
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
#define BINGO_TRACE_MARKER(id) asm volatile("ori x0, x0, %0" :: "i"(id))

#else

// If tracing is disabled, these compile to nothing.
#define BINGO_TRACE_MARKER(id) ((void)0)

#endif

// ============================================================================
// Trace Marker IDs (Events)
// ============================================================================
// Hierarchical ID scheme (max 12 bits):
// 0x1XX: Hardware Manager Events
// 0x2XX: Kernel Configuration Events
// 0x3XX: Accelerator Execution Events

// --- Hardware Manager Events ---
// Marks the lifespan of a task within the HW Manager loop
#define BINGO_TRACE_MGR_GET_READY_START  0x110 // Start reading Ready Queue
#define BINGO_TRACE_MGR_GET_READY_END    0x111 // End reading Ready Queue
#define BINGO_TRACE_MGR_PREP_START       0x112 // Start preparing kernel (get args, ptrs)
#define BINGO_TRACE_MGR_PREP_END         0x113 // End preparing kernel
#define BINGO_TRACE_MGR_RUN_KERNEL_START 0x114 // Start running kernel
#define BINGO_TRACE_MGR_RUN_KERNEL_END   0x115 // End running kernel
#define BINGO_TRACE_MGR_WRITE_DONE_START 0x116 // Start writing Done Queue
#define BINGO_TRACE_MGR_WRITE_DONE_END   0x117 // End writing Done Queue

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
// SIMD
#define BINGO_TRACE_SIMD_RUN_START        0x350
#define BINGO_TRACE_SIMD_RUN_END          0x351
// HOST IDMA
#define BINGO_TRACE_HOST_IDMA_RUN_START   0x360
#define BINGO_TRACE_HOST_IDMA_RUN_END     0x361