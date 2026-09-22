# kernels/ — the kernel ABI

One args class per device or host kernel, mirroring the C structs in
`libbingo/include/libbingo/*_kernel_args.h`. An args class validates its arguments in
Python and then emits the C field assignments that initialise the struct.

This is where a bad argument should be caught, because the alternative is a struct that
initialises cleanly and a kernel that does the wrong thing without failing.

| file | engine |
|---|---|
| `bingo_kernel_args.py` | **Facade.** Re-exports everything below, so `from bingo_kernel_args import …` keeps working for the ~60 workloads that do it. |
| `kernel_base.py` | The `BingoKernelArgs` ABC: address resolution across the four handle types, C field emission, and `PLACEMENT_ORDER` for the few kernels that compute one buffer's address as an offset from another. |
| `kernel_misc.py` | Dummy, sync probe, sync report — kernels that drive no accelerator — and the cluster iDMA. |
| `kernel_gemm.py` | VersaCore GEMM: the general descriptor, the minimal one, and the typed variants. |
| `kernel_gemm_fa.py` | The FlashAttention QK/PV pair and the array's performance counters. |
| `kernel_xdma.py` | xDMA transfers, the 6-D AGU, 2-D shape ops, and the in-fabric junction CSR builders. |
| `kernel_layout.py` | The dedicated layout converters: D↔row-major, row-major↔A, row-major↔B. |
| `kernel_simd.py` | SIMD streaming primitives and the fused whole-operators. |
| `kernel_host.py` | Host transfers and the per-precision result checks. |
| `kernel_ara.py` | Ara (RVV) host kernels, typed by precision. |

## Adding a kernel

Put the class in the engine's file. The facade picks it up through its star import; nothing
there needs editing unless you are adding a whole engine.

The class name is not free: `BingoNode` infers `kernel_name` from `KERNEL_NAME` on the args,
and that string is the contract with the device-side `SNAX_EXPORT_FUNC` registry. A name
that does not exist there links fine and dispatches to nothing.

## Three things these classes exist to catch

- **The typed GEMM and layout classes are spelled out, not generated.** A generated name
  that has no `SNAX_EXPORT_FUNC` behind it fails silently, so the list is explicit.
- **`row_major_to_b` and `b_to_row_major` drive the hardware transposer**, which is correct
  only at `elem_bytes=1`. At 2 and 4 it returns wrong data without failing.
- **`check_result` is one class per precision.** The kernel selects its comparison with an
  integer, and that integer silently couples three decisions — which comparison runs, what
  `num_elements` counts, and whether `tolerance` is a distance or a ratio. Getting any of
  them wrong passes rather than faults, so the precision picks the class instead.
