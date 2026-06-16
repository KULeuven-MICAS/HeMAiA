# Software Bingo Offload Applications

This directory contains applications using the **Bingo Software Manager**, a pure-software framework for task offloading.

## Concept: Software Task Orchestration
The Bingo SW framework introduces a structured way to handle Data-Flow Graphs (DFG):
- **DFG Model**: Users define workloads as a collection of tasks (kernels) with dependencies.
- **Worker Dispatch**: The manager uses software mechanisms to wake up worker cores and dispatch specific function pointers for execution.
- **Pure SW**: No special hardware scheduling logic is required, making it highly portable.

## Workloads
The workload directories are split by chip topology. Each workload emits a
separate ELF through the corresponding `single_chip` or `multi_chip` Makefile.

### Single-chip workloads
- **dummy_1cluster**: Minimal smoke test that dispatches a dummy kernel through the SW manager.
- **xdma_1d_copy_1cluster**: Simple 1D buffer copy verification using the Bingo SW manager.
- **gemm_merge_load_and_compute_1cluster**: A single task where the core handles both data fetching and computation.
- **gemm_seperate_load_and_compute_1cluster**: Separates data movement (xDMA) and compute into different tasks, allowing for software-managed overlapping.
- **gemm_stacked_1cluster**: Repeated execution of GEMM kernels, 1) D1 = A1 * B1 2) E1 = D1 * B2.
- **gemm_tiled_1cluster**: Double-buffered tiled GEMM implementation.

### Multi-chip workloads
- **gemm_broadcast_4chiplet_1cluster**: Four-chiplet GEMM that broadcasts shared operands across chiplets.
- **xdma_mempool_4chiplet_1cluster**: Four-chiplet xDMA test moving data across chiplet memory pools.

## Build Command
```bash
make single-sw HOST_APP_TYPE=offload_bingo_sw CHIP_TYPE=single_chip WORKLOAD=gemm_tiled_1cluster DEV_APP=snax-bingo-offload
```
