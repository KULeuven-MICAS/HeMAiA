# Core-Level BINGO Kernel Catalog

This document lists the currently implemented core-level BINGO kernel entries for the hardware offload flow. It intentionally excludes cluster-level software-offload kernels such as `__snax_kernel_*`.

Sources of truth:

- Device kernel implementations: `target/sw/device/apps/snax/snax-bingo-offload/libsnaxkernel/offload_hw_kernels/`
- Device kernel argument ABI: `target/sw/host/runtime/libbingo/include/libbingo/device_kernel_args.h`
- Host kernel implementations: `target/sw/host/runtime/host_kernel_lib.h`
- Host kernel argument ABI: `target/sw/host/runtime/libbingo/include/libbingo/host_kernel_args.h`

## GEMM Core Kernels

| Kernel name | Function description |
|-------------|----------------------|
| `__snax_bingo_kernel_gemm_full` | Configures the VersaCore streamer and GEMM accelerator, then runs and waits for a GEMM task. Supports shape lookup through `array_shape_idx`, transposed A/B inputs, previous-C accumulation, INT8 quantization, INT32-to-FP16 output conversion, and INT4 A/B input modes. |
| `__snax_bingo_kernel_gemm_minimal` | Updates only the A/B/C/D base addresses, then starts and waits for a preconfigured VersaCore streamer/GEMM run. Intended for repeated GEMMs that reuse an existing CSR configuration. |

## DMA Core Kernels

| Kernel name | Function description |
|-------------|----------------------|
| `__snax_bingo_kernel_idma_1d_copy` | Runs a 1D iDMA copy from source to destination and waits for completion. |
| `__snax_bingo_kernel_idma_broadcast` | Runs an iDMA-backed broadcast copy by transforming the destination address to the broadcast address space before starting the transfer. |
| `__snax_bingo_kernel_xdma_1d_copy` | Runs a full-address 1D xDMA copy and waits for the xDMA task to complete. |
| `__snax_bingo_kernel_xdma_6d` | Runs a generic xDMA transfer with explicit spatial stride plus up to five temporal stride/bound dimensions. |
| `__snax_bingo_kernel_xdma_transpose_2d` | Transposes a 2D matrix from `[M, N]` to `[N, M]`; uses the xDMA transposer extension when available and otherwise falls back to a byte-wise transpose. |
| `__snax_bingo_kernel_xdma_submatrix_2d` | Extracts a rectangular 2D slice from a source matrix into a compact destination matrix. |
| `__snax_bingo_kernel_xdma_expand_2d` | Broadcasts one row shaped `[1, N]` into an output matrix shaped `[M, N]`. |
| `__snax_bingo_kernel_xdma_concat_2d` | Copies a 2D input chunk into a row or column offset inside a larger output tensor. |
| `__snax_bingo_kernel_xdma_pad_2d` | Zero-fills a padded 2D output tensor and copies the source matrix into the padded region. |
| `__snax_bingo_kernel_xdma_gather_2d` | Gathers rows from a source matrix using an arithmetic row index pattern defined by start and stride. |
| `__snax_bingo_kernel_xdma_d_to_row_major` | Converts VersaCore D-layout tiles `[M_T, N_T, meshRow, meshCol]` to logical row-major layout. |
| `__snax_bingo_kernel_xdma_row_major_to_a` | Converts logical row-major data to VersaCore A-layout tiles `[M_T, K_T, meshRow, tileSize]`. |
| `__snax_bingo_kernel_xdma_row_major_to_b` | Converts logical row-major data to VersaCore B-layout tiles `[N_T, K_T, meshCol, tileSize]`. |
| `__snax_bingo_kernel_xdma_a_to_row_major` | Converts VersaCore A-layout tiles back to logical row-major layout. |
| `__snax_bingo_kernel_xdma_b_to_row_major` | Converts VersaCore B-layout tiles back to logical row-major layout. |
| `__snax_bingo_kernel_xdma_row_major_to_d` | Converts logical row-major data to VersaCore D-layout tiles `[M_T, N_T, meshRow, meshCol]`. |

## Host Kernels

The typed `__host_bingo_kernel_<op>_f32` / `_i32` kernels (plus `quantize_f32i8` /
`dequantize_i32f32`) below are the single-precision entry points. The elementwise /
reduce / softmax / rmsnorm / silu_mul ops also have a runtime-typed multi-precision
**dispatcher** `__host_bingo_kernel_<op>` that picks fp32/fp16/int8/int16 from a
`precision` arg (`BINGO_PREC_*`) and delegates the fp32 case to the typed kernel. From
the mini-compiler, drive these via the unified `HostBingoKernelAra*Args` classes (one
per op, sharing the `ara_binary` / `ara_unary` / `ara_softmax` / `ara_rmsnorm` /
`ara_convert` arg structs — all carrying `precision` at arg index [4] and
`scratchpad_ptr` at [5]).

The `precision` selectors are `BINGO_PREC_FP32`/`FP16`/`INT8`/`INT16`, plus
`BINGO_PREC_INT32` (=4) for elementwise add: `HostBingoKernelAraAddArgs(...,
precision=BINGO_PREC_INT32)` routes to the distinct `__host_bingo_kernel_add_i32`
kernel (K-split partial-D accumulation) instead of the multi-precision `add`
dispatcher. The two fixed-type conversions are driven by precision-in-name classes
`HostBingoKernelAraQuantizeF32I8Args` (FP32→INT8) and
`HostBingoKernelAraDequantizeI32F32Args` (INT32→FP32); their `precision` field is a
no-op passthrough.

| Kernel name | Function description |
|-------------|----------------------|
| `__host_bingo_kernel_entry` | Marks host-side task-graph entry, records the start cycle counter in the task scratchpad, and returns success. |
| `__host_bingo_kernel_exit` | Marks host-side task-graph exit and returns the BINGO success exit code. |
| `__host_bingo_kernel_dummy` | Prints a dummy host task value and publishes a zero return value. |
| `__host_bingo_kernel_check_result` | Compares output data against golden data. Supports byte-exact, FP32 tolerance, and FP16 tolerance modes. |
| `__host_bingo_kernel_idma` | Runs a host-side system DMA copy through `sys_dma_memcpy` and waits for the chip DMA completion counter. |
| `__host_bingo_kernel_xdma_1d_copy` | Runs a host-side 1D xDMA copy through the HeMAiA xDMA library and waits for remote xDMA completion. |
| `__host_bingo_kernel_rmsnorm_f32` | Computes FP32 RMSNorm over `num_tokens` rows with Ara/RVV vectorization. |
| `__host_bingo_kernel_softmax_f32` | Computes FP32 row-wise softmax with max subtraction, exponential approximation, and Ara/RVV vectorization. |
| `__host_bingo_kernel_silu_mul_f32` | Computes fused FP32 SiLU-and-multiply: `out = silu(gate) * up`. |
| `__host_bingo_kernel_add_f32` | Computes elementwise FP32 addition. |
| `__host_bingo_kernel_sub_f32` | Computes elementwise FP32 subtraction. |
| `__host_bingo_kernel_mul_f32` | Computes elementwise FP32 multiplication. |
| `__host_bingo_kernel_div_f32` | Computes elementwise FP32 division. |
| `__host_bingo_kernel_max_f32` | Computes elementwise FP32 maximum. |
| `__host_bingo_kernel_min_f32` | Computes elementwise FP32 minimum. |
| `__host_bingo_kernel_add_i32` | Computes elementwise INT32 addition, mainly for accumulating K-split GEMM partial results. |
| `__host_bingo_kernel_relu_f32` | Computes elementwise FP32 ReLU. |
| `__host_bingo_kernel_neg_f32` | Computes elementwise FP32 negation. |
| `__host_bingo_kernel_abs_f32` | Computes elementwise FP32 absolute value. |
| `__host_bingo_kernel_exp_f32` | Computes elementwise FP32 exponential using the local Cephes-style approximation. |
| `__host_bingo_kernel_sigmoid_f32` | Computes elementwise FP32 sigmoid. |
| `__host_bingo_kernel_tanh_f32` | Computes elementwise FP32 tanh using the exponential approximation. |
| `__host_bingo_kernel_sqrt_f32` | Computes elementwise FP32 square root. |
| `__host_bingo_kernel_reciprocal_f32` | Computes elementwise FP32 reciprocal. |
| `__host_bingo_kernel_silu_f32` | Computes elementwise FP32 SiLU. |
| `__host_bingo_kernel_gelu_f32` | Computes elementwise FP32 fast GELU approximation. |
| `__host_bingo_kernel_reduce_sum_f32` | Reduces an FP32 array to one sum value. |
| `__host_bingo_kernel_reduce_max_f32` | Reduces an FP32 array to one maximum value. |
| `__host_bingo_kernel_reduce_mean_f32` | Reduces an FP32 array to one mean value. |
| `__host_bingo_kernel_quantize_f32i8` | Quantizes FP32 input to INT8 with per-tensor symmetric scaling and writes the computed scale. |
| `__host_bingo_kernel_dequantize_i32f32` | Dequantizes INT32 GEMM accumulator data to FP32 using a provided combined scale. |
| `__host_bingo_kernel_cerf_gating` | Computes CERF gating masks for top-k, threshold, or static modes and writes optional per-expert activation flags for software guards. |
