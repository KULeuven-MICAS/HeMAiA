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
│                              BINGO_GET_SP, make_u64, EXTRACT_BIT
├── snax_kernel_lib.h       — top-level aggregator: `#include`s every partial
│                              below and defines the SNAX_SYMTAB the host
│                              dispatcher reads at offload time
├── Makefile                — drives validate_shapes.py against the active
│                              hwcfg; exposes DATA_H so every kernel TU
│                              depends on the validation stamp
├── offload_sw_kernels/     — cluster-level kernels (bingo-sw flow)
│   ├── basic.h             — dummy / csr / check_results
│   ├── idma.h              — idma_1d_copy / check_results_full /
│   │                          load_compute_store / double_buffer
│   ├── xdma.h              — xdma_1d_copy
│   └── gemm.h              — versacore_load_compute_store /
│                              minimal_cfg_start_gemm_and_wait
└── offload_hw_kernels/     — core-level kernels (bingo-hw flow)
    ├── basic.h             — bingo dummy / entry_point / exit
    ├── idma.h              — bingo iDMA copy + broadcast
    ├── xdma.h              — bingo xDMA: 1d/6d copies, layout transforms,
    │                          2d helpers (transpose, submatrix, expand,
    │                          concat, pad, gather)
    └── gemm.h              — bingo GEMM full / minimal
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
  scratchpad obtained via `BINGO_GET_SP(arg, nfields)`.

## GEMM shape table (`runtime/snax/versacore/gemm_shapes.h`)

Both `offload_sw_kernels/gemm.h` and `offload_hw_kernels/gemm.h` index a
shared `bingo_gemm_shape_params[array_shape_idx]` table that lives at
`HeMAiA/target/sw/device/runtime/snax/versacore/gemm_shapes.h`. The table
is hand-maintained C — see the formula block at the top of the header for
how to derive each field — and `validate_shapes.py` (next to it)
cross-checks every entry against the active hwcfg
(`snax_versacore_to_cluster.hjson`) on every build. Any drift fails
`make sw` with a field-level error.

To add a shape:

1. Bump `BINGO_NUM_ARRAY_SHAPES` in `gemm_shapes.h`.
2. Append a `[N] = { … }` initializer (use the formula block to derive
   `Ctlbound0`, `Ctlstride0`, `D32tlbound0`, `D32tlstride0`, and the
   `channel_en_*` masks from `meshRow`/`tileSize`/`meshCol`).
3. Update the hwcfg to match.
4. `make sw` — the validator will confirm the numbers.

Host workloads under `offload_bingo_sw/**/` include `<gemm_shapes.h>`
directly and read mesh dims via the same table; `gemm_datagen.py` no
longer emits per-workload `meshRow`/`tileSize`/`meshCol` globals.

## Adding a new kernel

1. Pick the partial header that matches what the kernel programs:
   `basic.h` for non-DMA utilities, `idma.h` / `xdma.h` for kernels that
   program the corresponding engine, `gemm.h` for versacore work.
2. Define the kernel with `SNAX_LIB_DEFINE` and the right return type for
   its flow (`void` for SW, `uint32_t` for HW).
3. For HW kernels with the SW guard pattern (CERF group sharing across
   experts), call `BINGO_SW_GUARD_CHECK(arg, nfields)` first and read the
   scratchpad via `BINGO_GET_SP(arg, nfields)`.
4. Add a matching `__SNAX_KERNEL_ARGS_DEFINE` struct under
   `host/runtime/libbingo/include/libbingo/device_kernel_args.h` so the
   host can construct args with the same layout the kernel parses.
5. Register the kernel in the `__snax_symtab[]` block of
   `snax_kernel_lib.h` so `get_device_function("…")` can resolve it.

## Common helpers (`macros.h`)

| macro / inline                            | use                                                                         |
| ----------------------------------------- | --------------------------------------------------------------------------- |
| `SNAX_LIB_DEFINE`                         | `__attribute__((used))` — keeps the kernel symbol after dead-code stripping |
| `SNAX_EXPORT_FUNC(name)`                  | symtab entry: `{#name, (uint32_t)name}`                                     |
| `BINGO_L1_ALLOC_OR_RETURN(var, n, label)` | `snrt_l1_malloc` + null-check + debug print, `return` on failure            |
| `BINGO_SW_GUARD_CHECK(arg, nfields)`      | per-expert SW guard for CERF group sharing (see comment block in `macros.h`)|
| `BINGO_GET_SP(arg, nfields)`              | grab `bingo_kernel_scratchpad_t*` from the last arg slot                    |
| `IDMA_DEBUG_PRINT` / `XDMA_DEBUG_PRINT` / `VERSACORE_DEBUG_PRINT` | conditional `printf_safe` (gated on `IDMA_DEBUG`/`XDMA_DEBUG`/`VERSACORE_DEBUG`) |
| `make_u64(hi, lo)`                        | combine two `uint32_t` halves into a `uint64_t` address                     |

## Build flow

```
make sw HOST_APP_TYPE=offload_bingo_sw CHIP_TYPE=single_chip \
        WORKLOAD=<workload> DEV_APP=snax-bingo-offload
```

`Makefile` (this directory) is `include`d by the per-app Makefile and
contributes:

- A validation rule that runs `validate_shapes.py` whenever
  `gemm_shapes.h` or the hwcfg changes; the resulting `.validate.stamp`
  is exposed via `DATA_H`, which the outer build wires as a prereq of
  every kernel object file.
- A `clean-data` target that removes the stamp.

Nothing in this directory is generated. Both `offload_*_kernels/gemm.h`
and the shared `gemm_shapes.h` are committed plain C; `validate_shapes.py`
only reads them.
