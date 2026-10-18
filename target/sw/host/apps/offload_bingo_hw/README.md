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
Each workload's `main_bingo.py` includes a workload description and task
dependency graph comment block.

### Single-chip workloads
- **concurrent_dma_1cluster**: Exercises concurrent DMA operations within one cluster.
- **gemm_double_buffer_1cluster**: Tiled GEMM pipeline using double-buffered DMA and compute stages.
- **gemm_serial_dma_1cluster**: Vanilla GEMM baseline with serialized load tasks with data loaded from L3.
- **gemm_parallel_dma_1cluster**: Vanilla GEMM variant with two load tasks scheduled in parallel with data loaded from L3.
- **gemm_mem_chip_1cluster**: Vanilla GEMM with data loaded from the memory chip into the compute chiplet before execution.
- **gemm_stacked_1cluster**: Stacked GEMM tasks for stressing hardware-managed task dispatch.
- **gemm_sweep_1cluster**: GEMM characterization sweep across multiple `(M, K, N, array_shape)` configurations.
- **gemm_psweep_{i4i4_i32,i8i4_f16,i8i4_i32,i8i8_f16,i8i8_i32,i8i8_i8}_1cluster**: GEMM precision sweep, one workload per precision, over the shared `(M, K, N, array_shape)` grid of `util/sim/gemm/gemm_psweep_lib.py` (run by `sweep/unit_test/gemm`).
- **xdma_ci_ops_1cluster**: Runs all xDMA operator types in one workload.
- **xdma_1d_1cluster**: Minimal single xDMA 1D-copy functional test (load → copy → store → check).
- **idma_swap_1cluster**: Per-op sweep for the iDMA adjacent-element-pair swap (the data half of RoPE's `rotate_half`); see its README.
- **simd_softmax_1cluster**: Fused FP16 row-wise softmax — the whole reduce(MAX) → EXP → normalize pipeline in one SIMD-core kernel, checked against an FP16 golden.
- **simd_softmax_2cluster**: Minimal 2-cluster SIMD (`hemaia_tapeout_2c_simd`) version of the above — one small fused softmax per cluster (`[1,64]` and `[2,64]`), run concurrently against per-cluster L1/L3 buffers. Kept tiny on purpose: the full rows × cols sweep does not fit this cfg's wide-SPM L3 heap; take the cycle LUT from the 1-cluster workload.
- **simd_softmax_t_1cluster**: Softmax A/B on the SIMD core: the same tile reduced row-major and transposed, against the same golden.
- **simd_rmsnorm_1cluster**: Fused FP16 RMSNorm in one SIMD-core kernel (sum-of-squares reduce → StreamMap RSQRT → normalize), checked against an FP16 golden.
- **simd_rmsnorm_t_1cluster**: RMSNorm A/B on the SIMD core: the same tile reduced row-major and transposed, against the same golden.
- **simd_rope_1cluster**: FP16 rotary position embedding (RoPE) as two tasks — the adjacent-pair swap of x on the DM core (iDMA) and the whole rotation as one SIMD-core task.
- **simd_silu_1cluster**: FP16 SiLU activation as one SIMD-core kernel (a single StreamMap pass).
- **simd_swiglu_1cluster**: FP16 SwiGLU (SiLU(gate) · up) as one SIMD-core kernel (StreamMap SiLU, then a StreamElementwise multiply).
- **simd_quant_1cluster**: The SIMD FP16 → INT8 quantiser armed on its own, for a per-operator cost curve.
- **xdma_transpose_1cluster**: xDMA 2D transpose.
- **xdma_submatrix_1cluster**: xDMA submatrix (tile) extraction.
- **xdma_expand_1cluster**: xDMA stride-0 broadcast / expand.
- **xdma_concat_1cluster**: xDMA 2D concatenation.
- **xdma_pad_1cluster**: xDMA padding.
- **xdma_gather_1cluster**: xDMA gather.
- **xdma_6d_1cluster**: xDMA 6D strided copy.
- **xdma_elementwise_add_1cluster**: xDMA elementwise add (the K-split int32 partial-sum reduce).
- **xdma_{row_to_a,a_to_row,row_to_b,b_to_row,row_to_d,d_to_row}_1cluster**: Blocked-layout converters between row-major and the GEMM operand (A / B / D) layouts, swept across array shapes and element widths.
- **moe4_4cluster**: Four-cluster mixture-of-experts workload.
- **fa_decode_1cluster**, **fa_prefill_1cluster**: FlashAttention on one cluster as a BINGO task graph — the two matmuls on the GEMM core, the online softmax on the SIMD core, overlapped through the graph. The decode workload takes one query position per token; the prefill workload has many query tiles attend the same resident K/V.
- **fa_decode_4cluster**, **fa_prefill_4cluster**: The same, KV-sharded over four clusters; the cross-cluster online-softmax merge is a ChainGather over the softmax-statistics junction.
- **basic_quantize_host**: Host-kernel smoke test for quantization.
- **basic_dequantize_host**: Host-kernel smoke test for dequantization.
- **basic_int32_add_host**: Host-kernel smoke test for int32 addition and reduction chains.
- **basic_softmax_host**: Host-kernel smoke test for softmax.
- **gemm_ksplit_4cluster**: Four-cluster GEMM with K-dimension splitting and reduction.
- **gemm_nsplit_accumprevc_4cluster**: Four-cluster GEMM split along N; each cluster reduces K on-chip with the VersaCore `accumPrevC` feature, so the host sums nothing.
- **attn_reference_4cluster**: Four-cluster reference attention graph covering GEMM, softmax, and post-processing kernels.
- **l3_bw_4cluster**: Main-memory bandwidth in three phases (read-only, write-only, concurrent read and write) across four clusters.
- **device_host_idma_check_1cluster**: Dependency-tag regression: the host check must wait for the host iDMA golden load, not only for the earlier device iDMA copy.
- **dummy_2cluster**: Placeholder 2-cluster workload for the `hemaia_tapeout_2c` CI suite: each cluster iDMA-copies one L3 vector into its own L1 and the host checks the result.
- **dummy_simd_2cluster**: The same placeholder for the `hemaia_tapeout_2c_simd` CI suite.

### Multi-chip workloads
- **gemm_msplit_4chiplet_1cluster**: Four-chiplet GEMM with M-dimension output partitioning.
- **gemm_ksplit_4chiplet_1cluster_basic**: Four-chiplet GEMM with K-dimension splitting and cross-chiplet reduction using host.
- **gemm_ksplit_4chiplet_1cluster_xdma_add**: Four-chiplet GEMM with K-dimension splitting and cross-chiplet reduction using xdma.
- **gemm_ksplit_4chiplet_1cluster_xdma_add_err_check**: Regression shape of the workload above, with a running-sum check after each xDMA add; `error_record.md` records the ordering bug it exposes.
- **gemm_ksplit_4chiplet_2cluster**: Four-chiplet, two-cluster GEMM with K split into eight chunks, one per cluster, reduced on chiplet `00` cluster 0.
- **gemm_msplit_4chiplet_2cluster**: Four-chiplet, two-cluster GEMM with M-dimension output partitioning; B is broadcast once to the remote chiplets.
- **gemm_msplit_4chiplet_2cluster_basic**: The same M-split GEMM with B fanned out by point-to-point D2D copies instead of a broadcast.
- **gemm_parallel_dma_4chiplet_1cluster**: Four-chiplet vanilla GEMM with parallel host and cluster iDMA load.
- **dma_for_gemm_msplit_4chiplet_1cluster**: Four-chiplet DMA-only M-split data movement test for evaluating the proposed data-movement techniques without running GEMM. The ablation cases gradually introduce optimizations: (0) no broadcast, no dual-DMA, no half-duplex; each chiplet loads its own A tile and B directly from the memory chip and checks both buffers. (1) broadcast; chiplet `00` loads B once and broadcasts it while each chiplet loads and checks its own A tile. (2) broadcast with dual-DMA. Cases 0-2 are implemented (case 3, with half-duplex, is an empty stub); the build runs case 1.
- **host_int32_add_4chiplet_1cluster**: Four-chiplet int32 cross chip addition workload using host.
- **xdma_int32_add_4chiplet_1cluster**: Four-chiplet int32 cross chip addition workload using xdma.
- **dma_read_from_chip00_4chiplet_1cluster**: Cross-chiplet DMA read test. The workload first loads four input chunks from the memory chip into the local L3 memories of chiplets `00`, `01`, `10`, and `11`. After each chiplet verifies its local chunk, chiplet `00` issues xDMA reads that pull the chunks from chiplets `01`, `10`, and `11` into chiplet `00` L3, then checks the received data against the golden buffers.
- **idma_read_from_chip00_4chiplet_2cluster**: Cross-chiplet system-iDMA read test with two active clusters per chiplet: chiplet `00` pulls the other chiplets' per-cluster L3 staging chunks with the system iDMA and checks them.
- **xdma_read_from_chip00_4chiplet_2cluster**: The same test with xDMA reads.
- **dma_write_from_other_chip_4chiplet_1cluster**: Cross-chiplet DMA write test. The workload first loads and verifies one local L3 input chunk on each chiplet from the memory chip. Then chiplets `01`, `10`, and `11` each issue xDMA writes that push their local chunks into receive buffers in chiplet `00` L3, where chiplet `00` validates the received data.
- **xdma_remote_write_4chiplet_1cluster**: Smallest test of a cross-chiplet **cluster**-xDMA remote write — chiplet `00` cluster 0 pushes 1 KiB straight out of its own L1 into chiplet `01` cluster 0's L1, and chiplet `01` checks what landed. Unlike the `dma_*` tests above (host xDMA/iDMA on L3), this exercises the xDMA cross-cluster cfg/grant handshake across the D2D link; a hang means the handshake never closed. The receive buffer is pre-poisoned and the golden lives in the checker's own L3, so a pass cannot come from stale L1 or a confounding cross-chip read.
- **xdma_remote_write_4chiplet_2cluster**: The same, into **both** clusters of the remote chiplet (chiplet `00` cluster 0 → chiplet `10` clusters 0 and 1). The destination cluster is implied by the destination address, so each write exercises a different cluster's control-MMIO window; both destinations are poisoned and checked, so an address that resolves onto the wrong cluster leaves the other poisoned and fails.
- **dvfs_cross_trigger_4chiplet_1cluster**: DVFS cross-trigger: chiplet `00` runs a six-GEMM sequence, then broadcasts B to trigger shorter GEMM sequences on chiplets `01`, `10` and `11`.
- **sync_latency_sweep_1cluster**: Cross-chip synchronisation latency of the BINGO hardware manager: round trips anchored on chiplet `00`, one-to-one and one-to-all over a series of chiplet rectangles, plus a local control phase on chiplet `00`.
- **dsv2**: DeepSeek-V2-Lite layer 1 (MLA attention + MoE) stage by stage on the two-, six- and sixteen-chiplet platforms; see [multi_chip/workloads/dsv2/README.md](multi_chip/workloads/dsv2/README.md).
- **ffn**: DeepSeek-V2-Lite's dense FFN (layer 0) on 16 compute + 4 memory chiplets, decode and 128-token prefill, with the broadcast / reduce / sync ablation; see [multi_chip/workloads/ffn/README.md](multi_chip/workloads/ffn/README.md).
- **fa**: DeepSeek-V2-Lite's layer-1 MLA attention on 16 compute + 4 memory chiplets, decode and prefill, with the same ablation; see [multi_chip/workloads/fa/README.md](multi_chip/workloads/fa/README.md).
- **moe**: DeepSeek-V2-Lite's layer-1 MoE on 16 compute + 4 memory chiplets; the decode ablation variants are in `moe/dsv2_l1/` and the sweep is described in `target/sim/automation/sweep/llm_blocks/moe/README.md`.
- **common**: the code ffn, moe and fa share (`multi_chip/workloads/common/`): `fabric.py` (the mesh, the fold order and the topology and node helpers of their multi-chiplet build) and `driver.py` (the command line and the `run()` body of their drivers).

## Build Command
```bash
make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip WORKLOAD=gemm_double_buffer_1cluster DEV_APP=snax-bingo-offload
```
