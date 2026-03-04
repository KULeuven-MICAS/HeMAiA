# Software Bingo Offload Applications

This directory contains applications using the **Bingo Software Manager**, a pure-software framework for task offloading.

## Concept: Software Task Orchestration
The Bingo SW framework introduces a structured way to handle Data-Flow Graphs (DFG):
- **DFG Model**: Users define workloads as a collection of tasks (kernels) with dependencies.
- **Worker Dispatch**: The manager uses software mechanisms to wake up worker cores and dispatch specific function pointers for execution.
- **Pure SW**: No special hardware scheduling logic is required, making it highly portable.

## Workloads
- **double_buffer**: Implementation of a double-buffering scheme using Bingo tasks.
- **xdma_1d_copy**: Simple 1D buffer copy verification using the Bingo SW manager.
- **gemm_merge_load_and_compute**: A single task where the core handles both data fetching and computation.
- **gemm_seperate_load_and_compute**: Separates data movement (XDMA) and compute into different tasks, allowing for software-managed overlapping.
- **gemm_stacked**: Repeated execution of GEMM kernels, 1) D1 = A1 * B1 2) E1 = D1 * B2.
- **gemm_tiled**: Double-Bufferd Gemm implementation.

## Build Command
```bash
make single-sw HOST_APP_TYPE=offload_bingo_sw CHIP_TYPE=single_chip WORKLOAD=gemm_tiled DEV_APP=snax-bingo-offload
```
