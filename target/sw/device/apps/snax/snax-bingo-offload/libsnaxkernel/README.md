# libsnaxkernel — device-side kernel library for `snax-bingo-offload`

Header-only kernel library compiled into every device build of
`snax-bingo-offload`. Each kernel is a `void` (cluster-level / SW flow) or
`uint32_t` (core-level / HW flow) function the host runtime can look up
by name and dispatch onto the snitch cluster.

## Layout

```
libsnaxkernel/
├── macros.h                — common scaffolding: SNAX_LIB_DEFINE, debug prints,
│                              BINGO_SW_GUARD_CHECK, BINGO_L1_ALLOC_OR_RETURN,
│                              BINGO_GET_SP, BINGO_XDMA_TRY,
│                              BINGO_XDMA_EXT_UNSUPPORTED, make_u64, EXTRACT_BIT
├── snax_kernel_lib.h       — top-level aggregator: `#include`s every partial
│                              below and defines the SNAX_SYMTAB the host
│                              dispatcher reads at offload time
├── Makefile                — generates runtime/snax/versacore/gemm_shapes.h from
│                              the active cluster cfg (validate_shapes.py --emit);
│                              exposes DATA_H so every kernel TU depends on it
├── offload_sw_kernels/     — cluster-level kernels (bingo-sw flow)
│   ├── basic.h             — dummy / sync_probe / csr / check_results /
│   │                          check_results_full / load_compute_store /
│   │                          double_buffer
│   ├── idma.h              — idma_1d_copy
│   ├── xdma.h              — xdma_1d_copy
│   └── gemm.h              — versacore_load_compute_store /
│                              minimal_cfg_start_gemm_and_wait
└── offload_hw_kernels/     — core-level kernels (bingo-hw flow)
    ├── basic.h             — dummy / sync_probe / sync_mark / chip_barrier / exit
    ├── idma.h              — iDMA 1d/2d copies, broadcast, flagged put/fetch with
    │                          wait_flags, pairwise swap, slot copy, ring load
    ├── xdma.h              — xDMA data movement: 1d/6d copies, memset, crest
    │                          expand, multicast, chain_gather, pack_fa_partial,
    │                          the layout converter (layout_convert: row-major
    │                          ↔ A/B/D), 2d helpers (transpose, submatrix, expand,
    │                          concat, pad, gather), and the writer-side
    │                          ElementwiseAdd accumulate (elementwise_add /
    │                          elementwise_add_ab)
    ├── simd.h              — SIMD-core (hart 1) kernels: the FP16 stream
    │                          operators and the fused whole-ops (softmax,
    │                          rmsnorm, silu, swiglu, rope, moe_combine,
    │                          fa_softmax)
    ├── gemm.h              — GEMM full / start / wait / shape_init / minimal and
    │                          the six per-precision wrappers (gemm_i8i8_i32,
    │                          gemm_i8i4_i32, gemm_i4i4_i32, gemm_i8i8_i8,
    │                          gemm_i8i4_f16, gemm_i8i8_f16)
    ├── gemm_fa.h           — the FlashAttention matmuls (qk / pv, plus the
    │                          queued _start forms) and gemm_perf_report
    ├── gemv.h              — the one-token GEMV on VersaCore's (1, 4, 32) shape
    ├── simd_row.h          — one token's row: RMSNorm, quantise into A row 0,
    │                          MLA normalise, row softmax, SwiGLU
    ├── moe_route.h         — the router's top k as an expert-slot record
    │                          (moe_route / moe_router_pack)
    └── moe_prefill.h       — the MoE prefill: route table, row lists, row gather,
                               and the SIMD row-tile passes
```

The two flows mirror the host-side split under
`HeMAiA/target/sw/host/apps/{offload_bingo_sw, offload_bingo_hw}/`:

- **bingo-sw (cluster-level)**: each kernel runs across the whole cluster,
  coordinating the DM core and compute cores via `snrt_cluster_hw_barrier()`
  and the per-cluster scratchpad (`get_cls_shared_ptrs()`). Kernels return
  `void`.
- **bingo-hw (core-level)**: each kernel runs on a single core (typically
  core 0) under bingo HW scheduling. Kernels return `uint32_t`
  (`BINGO_RET_SUCC` / `BINGO_RET_FAIL`) and write into a per-kernel
  scratchpad obtained via `BINGO_GET_SP(arg, <kernel>_args_t)`.

## GEMM shape table (`runtime/snax/versacore/gemm_shapes.h`)

Both `offload_sw_kernels/gemm.h` and `offload_hw_kernels/gemm.h` index a
shared `bingo_gemm_shape_params[array_shape_idx]` table that lives at
`HeMAiA/target/sw/device/runtime/snax/versacore/gemm_shapes.h`. The table
is generated from the active cluster cfg (`clusters[0]` of
`target/rtl/cfg/lru.hjson`) by `validate_shapes.py --emit` (next to it) on
every build; it is not committed. The formula block at the top of the
generated header shows how each field derives from
`meshRow`/`tileSize`/`meshCol`.

To add a shape: add it to the cluster cfg's array shapes and rebuild;
`gemm_shapes.h` is regenerated.

Host workloads under `offload_bingo_sw/**/` include `<gemm_shapes.h>`
directly and read the mesh dims from that same table, so the table is the
one place `meshRow`/`tileSize`/`meshCol` are defined.

## Adding a new kernel

1. Pick the partial header that matches what the kernel programs:
   `basic.h` for utilities (the SW flow's `basic.h` also holds the iDMA
   demo kernels `check_results_full`, `load_compute_store` and
   `double_buffer`), `idma.h` / `xdma.h` for kernels that program the
   corresponding engine, `gemm.h` / `gemm_fa.h` / `gemv.h` for versacore
   work, `simd.h` / `simd_row.h` for SIMD-core work, `moe_route.h` /
   `moe_prefill.h` for the MoE kernels.
2. Define the kernel with `SNAX_LIB_DEFINE` and the right return type for
   its flow (`void` for SW, `uint32_t` for HW).
3. For HW kernels with the SW guard pattern (CERF group sharing across
   experts), call `BINGO_SW_GUARD_CHECK(arg, <kernel>_args_t)` first and read
   the scratchpad via `BINGO_GET_SP(arg, <kernel>_args_t)`.
4. Add a matching `__SNAX_KERNEL_ARGS_DEFINE` struct under
   `host/runtime/libbingo/include/libbingo/device_kernel_args.h` so the
   host can construct args with the same layout the kernel parses.
5. Add the matching Python arg class in the engine's file under
   `host/runtime/libbingo/mini_compiler/libs/kernels/`
   (`kernel_<engine>.py`; `bingo_kernel_args.py` re-exports it) — its
   `get_struct_name()` returns the `…_args_t` typedef and its emitted
   fields must match the struct order — so host DFGs / the mini-compiler
   can build the args for this kernel.
6. Register the kernel in the `__snax_symtab[]` block of
   `snax_kernel_lib.h` so `get_device_function("…")` can resolve it.

The kernel name string (e.g. `__snax_bingo_kernel_xdma_elementwise_add`) is
the single source of truth tying these five sites together — keep it
identical across the C function, the `…_args_t` struct, the Python class's
`get_struct_name()`, the symtab export, and the host `kernel_name=`.

## Common helpers (`macros.h`)

| macro / inline                            | use                                                                         |
| ----------------------------------------- | --------------------------------------------------------------------------- |
| `SNAX_LIB_DEFINE`                         | `__attribute__((used))` — keeps the kernel symbol after dead-code stripping |
| `SNAX_EXPORT_FUNC(name)`                  | symtab entry: `{#name, (uint32_t)name}`                                     |
| `BINGO_L1_ALLOC_OR_RETURN(var, n, label)` | `snrt_l1_malloc` + null-check + debug print, `return` on failure            |
| `BINGO_SW_GUARD_CHECK(arg, args_t)`       | per-expert SW guard for CERF group sharing (see comment block in `macros.h`)|
| `BINGO_GET_SP(arg, args_t)`               | grab `bingo_kernel_scratchpad_t*` from the args struct's last field, `scratchpad_ptr` (`offsetof(args_t, scratchpad_ptr)`) |
| `BINGO_XDMA_TRY(call, kname)`             | issue an xDMA library call; on a non-zero return print the decoded reason and `return BINGO_RET_FAIL` |
| `BINGO_XDMA_EXT_UNSUPPORTED(kname, exts)` | refuse a kernel whose xDMA extension the RTL cfg did not generate: print and `return BINGO_RET_FAIL` |
| `IDMA_DEBUG_PRINT` / `XDMA_DEBUG_PRINT` / `VERSACORE_DEBUG_PRINT` | conditional `printf_safe` (gated on `IDMA_DEBUG`/`XDMA_DEBUG`/`VERSACORE_DEBUG`) |
| `make_u64(hi, lo)`                        | combine two `uint32_t` halves into a `uint64_t` address                     |

## Build flow

```
make sw HOST_APP_TYPE=offload_bingo_sw CHIP_TYPE=single_chip \
        WORKLOAD=<workload> DEV_APP=snax-bingo-offload
```

`Makefile` (this directory) is `include`d by the per-app Makefile and
contributes:

- A rule that regenerates `gemm_shapes.h` whenever the hwcfg or
  `validate_shapes.py` changes; the `.validate.stamp` that depends on it
  is exposed via `DATA_H`, which the outer build wires as a prereq of
  every kernel object file.
- A `clean-data` target that removes the stamp and the generated
  `gemm_shapes.h`.

The kernel headers are hand-written C. `gemm_shapes.h` is generated by
`validate_shapes.py --emit` and is not committed.
