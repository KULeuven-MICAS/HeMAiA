# Hardware Bingo Offload Applications

This directory contains applications using the **Bingo Hardware Manager**, a hardware-assisted framework for high-performance task scheduling.

## Concept: Hardware-Managed Ready Queues
The Bingo HW framework offloads the "Manager" logic from software to dedicated hardware logic at the Quad/SoC level:
- **HW Scheduler**: A hardware unit manages the ready and done queues for all cores in a cluster or quad.
- **CSR Access**: Instead of waiting for a software manager to dispatch a function pointer, each core independently polls a hardware "Ready Queue" via a specialized CSR.
- **Zero-Overhead Dispatch**: When a core is free, it reads the next task ID from the hardware. The hardware ensures task atomicity and distribution across available cores.
- **Fine-Grained Scheduling**: Enables much finer task granularity than the software manager by removing the interrupt/polling overhead.

## Workloads
The workload directories are split by chip topology. Each workload emits a
separate ELF through the corresponding `single_chip` or `multi_chip` Makefile.

### Single-chip workloads
- **concurrent_dma_1cluster**: Exercises concurrent DMA operations within one cluster.
- **gemm_double_buffer_1cluster**: Tiled GEMM pipeline using double-buffered DMA and compute stages.
- **gemm_serial_dma_1cluster**: Vanilla GEMM baseline with serialized load tasks with data loaded from L3.
- **gemm_parallel_dma_1cluster**: Vanilla GEMM variant with two load tasks scheduled in parallel with data loaded from L3.
- **gemm_mem_chip_1cluster**: Vanilla GEMM with data loaded from the memory chip into the compute chiplet before execution.
- **gemm_stacked_1cluster**: Stacked GEMM tasks for stressing hardware-managed task dispatch.
- **gemm_sweep_1cluster**: GEMM characterization sweep across multiple `(M, K, N, array_shape)` configurations.
- **xdma_ci_ops_1cluster**: Runs all xDMA operator types in one workload.
- **xdma_1d_1cluster**: Minimal single xDMA 1D-copy functional test (load → copy → store → check).
- **moe4_4cluster**: Four-cluster mixture-of-experts workload.
- **basic_quantize_host**: Host-kernel smoke test for quantization.
- **basic_dequantize_host**: Host-kernel smoke test for dequantization.
- **basic_int32_add_host**: Host-kernel smoke test for int32 addition and reduction chains.
- **basic_softmax_host**: Host-kernel smoke test for softmax.
- **gemm_ksplit_4cluster**: Four-cluster GEMM with K-dimension splitting and reduction.
- **attn_reference_4cluster**: Four-cluster reference attention graph covering GEMM, softmax, and post-processing kernels.

### Multi-chip workloads
- **gemm_msplit_4chiplet_1cluster**: Four-chiplet GEMM with M-dimension output partitioning.
- **gemm_ksplit_4chiplet_1cluster_basic**: Four-chiplet GEMM with K-dimension splitting and cross-chiplet reduction.
- **gemm_parallel_dma_4chiplet_1cluster**: Four-chiplet vanilla GEMM with parallel host and cluster iDMA load.
- **dma_for_gemm_msplit_4chiplet_1cluster**: Four-chiplet DMA-only M-split data movement test for evaluating the proposed data-movement techniques without running GEMM. The ablation cases gradually introduce optimizations: (0) no broadcast, no dual-DMA, no half-duplex; each chiplet loads its own A tile and B directly from the memory chip and checks both buffers. (1) broadcast; chiplet `00` loads B once and broadcasts it while each chiplet loads and checks its own A tile. (2) broadcast with dual-DMA. (3) broadcast with dual-DMA and half-duplex.
- **host_int32_add_4chiplet_1cluster**: Four-chiplet int32 cross chip addition workload.
- **dma_read_from_chip00_4chiplet_1cluster**: Cross-chiplet DMA read test. The workload first loads four input chunks from the memory chip into the local L3 memories of chiplets `00`, `01`, `10`, and `11`. After each chiplet verifies its local chunk, chiplet `00` issues xDMA reads that pull the chunks from chiplets `01`, `10`, and `11` into chiplet `00` L3, then checks the received data against the golden buffers.
- **dma_write_from_other_chip_4chiplet_1cluster**: Cross-chiplet DMA write test. The workload first loads and verifies one local L3 input chunk on each chiplet from the memory chip. Then chiplets `01`, `10`, and `11` each issue xDMA writes that push their local chunks into receive buffers in chiplet `00` L3, where chiplet `00` validates the received data.

## Build Command
```bash
make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip WORKLOAD=gemm_double_buffer_1cluster DEV_APP=snax-bingo-offload
```
