# libs/kernels/ — the kernel ABI

One args class per device or host kernel, mirroring the C structs in
`libbingo/include/libbingo/*_kernel_args.h`. An args class validates its arguments in
Python and then emits the C field assignments that initialise the struct.

This is where a bad argument should be caught, because the alternative is a struct that
initialises cleanly and a kernel that does the wrong thing without failing.

| file | engine |
|---|---|
| `bingo_kernel_args.py` | **Facade.** Re-exports everything below, so `from bingo_kernel_args import …` keeps working for the ~60 workloads that do it. |
| `kernel_base.py` | The `BingoKernelArgs` ABC: address resolution across the four handle types, C field emission, and `PLACEMENT_ORDER` for the few kernels that compute one buffer's address as an offset from another. |
| `kernel_misc.py` | Dummy, stamp, chip barrier and stamp report — kernels that drive no accelerator; the cluster iDMA transfers; the MoE route and router pack. |
| `kernel_gemm.py` | VersaCore GEMM: the general descriptor, the minimal one, and the typed variants. |
| `kernel_gemm_fa.py` | The FlashAttention QK/PV pair and the array's performance counters. |
| `kernel_xdma.py` | xDMA transfers, the 6-D AGU, 2-D shape ops, and the in-fabric junction CSR builders. |
| `kernel_layout.py` | One layout-convert class for row_major↔A/B/D (six directions via `xdma_conv_args`), emitted as a derived `xdma_6d` nest or the `xdma_layout_convert` element loop. |
| `kernel_simd.py` | SIMD streaming primitives and the fused whole-operators. |
| `kernel_moe.py` | The MoE prefill: route table, row lists, row gather, row-tile SIMD passes. |
| `blocked_nest.py` | Not an args class: the loop nest `simd_rmsnorm` needs to write A/B blocks for a given mesh, derived per (tile, mesh, precision) and checked against the layout's index map. `kernel_simd.py` calls it while building the args, and the result goes into the struct's `blk_*` fields. |
| `kernel_host.py` | Host transfers and the per-precision result checks. |
| `kernel_ara.py` | Ara (RVV) host kernels, typed by precision. |

## Adding a kernel

Put the class in the engine's file. The facade picks it up through its star import; nothing
there needs editing unless you are adding a whole engine.

The class name is not free: `BingoNode` infers `kernel_name` from `KERNEL_NAME` on the args,
and that string is the contract with the device-side `SNAX_EXPORT_FUNC` registry. A name
that does not exist there links fine and dispatches to nothing.

Every class declares `accesses()`: for each field that holds an address, what the kernel does
there and how many bytes from it (`kernel_base.py`: `Read`, `Write`, `ReadWrite` with the byte
count, `None` for to the end of the buffer; `Sync` for memory a runtime protocol orders, such as a
flag; `Hold` for a buffer kept live but not touched). The byte counts come from the args
(`size`, M x K x N with the array shape, rows x cols, beats). There is no default: the compiler
uses the declarations to check that the graph orders every two tasks of which one writes bytes
the other touches (`passes/bingo_dfg_hazards.py`), and refuses a task whose class declares
none or leaves an address field out.

A kernel driven through a CSR manager and a streamer may also declare how it runs in parts:
`split()` returns a `Split` (`kernel_base.py`) with its configure, start and wait tasks, a start
that drains itself, how completion is tracked (a retired counter the compiler numbers, or the
part's own state), how many works may be in flight and whether another kernel may run
meanwhile. `dispatch_accel()` names the accelerator whose counter counts a whole task,
`config_key()` the configuration a task leaves in the CSRs and `config_reuse()` its cheaper form
when the CSRs hold it already. The compiler's asynchronous execution (`dfg.async_exec`) and CSR
reuse work from these declarations alone (`passes/bingo_dfg_schedule.py`).

## Three things these classes exist to catch

- **The typed GEMM classes are spelled out, not generated.** A generated name
  that has no `SNAX_EXPORT_FUNC` behind it fails silently, so the list is explicit.
- **row_major↔B uses the 8×8 byte transposer** only at `elem_bytes=1` with `tileSize` and
  `meshCol` multiples of 8; other widths run the xDMA core's element loop, which is correct
  but slow.
- **`check_result` is one class per precision.** The kernel selects its comparison with an
  integer, and that integer silently couples three decisions — which comparison runs, what
  `num_elements` counts, and whether `tolerance` is a distance or a ratio. Getting any of
  them wrong passes rather than faults, so the precision picks the class instead.
