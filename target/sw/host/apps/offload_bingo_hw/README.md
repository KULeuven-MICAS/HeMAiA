# Hardware Bingo Offload Applications

This directory contains applications using the **Bingo Hardware Manager**, a hardware-assisted framework for high-performance task scheduling.

## Concept: Hardware-Managed Ready Queues
The Bingo HW framework offloads the "Manager" logic from software to dedicated hardware logic at the Quad/SoC level:
- **HW Scheduler**: A hardware unit manages the ready and done queues for all cores in a cluster or quad.
- **CSR Access**: Instead of waiting for a software manager to dispatch a function pointer, each core independently polls a hardware "Ready Queue" via a specialized CSR.
- **Zero-Overhead Dispatch**: When a core is free, it reads the next task ID from the hardware. The hardware ensures task atomicity and distribution across available cores.
- **Fine-Grained Scheduling**: Enables much finer task granularity than the software manager by removing the interrupt/polling overhead.

## Workloads
- **gemm_double_buffer**: Matrix multiplication with hardware-synchronized double buffering.
- **gemm_seperate_load_and_compute_parallel**: Separate data/compute tasks scheduled in parallel by the HW manager.
- **gemm_seperate_load_and_compute_serial**: Baseline workloads for comparing HW scheduling efficiency.
- **gemm_stacked**: Highly parallel execution of GEMM stacks across multiple cores.
- **gemm_broadcast**: (Multi-chip) Workload demonstrating hardware-assisted broadcasting of weights/data across chiplets.

## Build Command
```bash
make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip WORKLOAD=gemm_double_buffer DEV_APP=snax-bingo-offload
```
