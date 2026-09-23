# comm/ — common machinery

Everything a block is built out of, and nothing specific to one. Two halves:

## What a block declares, and how two are joined

| file | |
|---|---|
| `ports.py` | `PortSpec` (layout, precision, shape, mem_level), `Port` (a bound spec plus the nodes at the block's edge), `Block`, `BlockResult`, and `at_offset`. **The layout names are worked through there with real index maps** — what `row_major`, `col_major`, `A`, `B`, `D`, `d32` and `monoid` each do to element `[row][col]`, and why `d32` is not `D` — which offsets any of the four handle types, because a staged array is a *symbol* on the host path and a *fixed address* on the memchip path, and only `BingoMemAlloc` has `.view()`. |
| `ctx.py` | The node and handle factory, bound to one cluster and one name prefix. The prefix is what lets one block be instantiated twice: handle names must be unique per chip, because the emitter dedups by identity and two same-named handles emit two C declarations that do not compile. |
| `link.py` | `Pipeline`, `link`, and `check_contract`. The pipeline runs in three phases — **connect** (`add` records, builds nothing), **resolve** (`run` picks one realisation per block so every boundary agrees), **build** (each block emits its own sub-DFG, in add order). `raw()` and `source()` put the application's own work into that same order, because node creation order is dispatch order. `check_contract` matches EXACTLY: a block that converts says so with a variant, which is a promise its `build()` has to keep. |
| `variant.py` | `Cost` and `Variant` — what a block could equally well have been. A layout a block does not pin is a knob it owns; `variants()` lists the realisations and `respec()` builds one. Legality is by construction: a variant is realised in order to be priced, so an illegal one raises in the block's own constructor and drops out of the search — one copy of the rules, in the constructor. `Cost` compares lexicographically in engine order (cross-lane folds, SIMD passes, xDMA, iDMA) — counts derived from the shape, never measured cycles, so it cannot rot. |
| `paths.py` | Reaching `util/sim`. Delegates to `_bingo_paths.repo_root()`; there is deliberately one implementation of that search. |

**The vocabulary is a closed set.** `Layout`, `DType` and `MemLevel` were tuples of bare
strings checked with `in`. They are now StrEnums, so a member *is* its string — it compares
equal, hashes and formats as one, and every call site that passes `"A"` or `"L3"` keeps
working — but `Layout.` lists the options, each carries a `doc`, and a typo is refused at
construction with the valid values named rather than reaching a kernel as a layout nobody
implements. `PortSpec` coerces, so a spec always holds the member however it was written.

**An input port usually names no memory level.** `mem_level=None` means *"wherever you
have it — I will fetch it"*, which is the honest declaration: a block knows which level its
own engines need (L1, always — the GEMM and SIMD have no AXI port) but has no business
dictating where the caller keeps the tensor. The block states the level its transfers read
from in `needs`, and the hoist happens inside its own build. A bound `Port` reads the level
off the handle, so binding a block's level-less spec directly just works. An **output** is
never None: a produced buffer is somewhere definite.

`Port.ends` is what makes the join mechanical: for an output it is the nodes that finish
writing the buffer, for an input the nodes that first read it. The linker joins one to the
other with a real RAW dependency, and nothing else.

## How a mismatch gets closed

| file | |
|---|---|
| `transfer.py` | Decides what a mismatch costs and emits the nodes. `plan()` builds nothing and can be inspected; `bring_in()` emits — and emits **nothing** when the bound port already matches, which is what lets a composed graph be identical to a hand-written one. |
| `nest.py` | Derives the strided xDMA transfer that performs a layout change, and the hand-written D→A nest it is checked against. |

**The closures, and why each is the engine it is.** `L4 → L3` is the host iDMA: the
memory-chiplet pool is off-die, so it is hoisted once instead of fetched per tile. A
**layout** change is one xDMA pass fused into the load — the AGU takes independent strides
on each side, so it converts while it transports, costing nothing beyond the load that had
to happen anyway. **Precision** is not done here at all.

## Nothing in nest.py is trusted

`convert_args` derives the strides and then walks them in numpy against ground-truth index
maps of **both** layouts, at graph-build time, on the real bounds. A nest that is wrong by
one dimension moves the right *number* of bytes to the wrong *offsets* — which no byte
count catches and which passes on random data. The generic derivation reproduces
`d_to_a_args`, the RTL-validated hand-written nest, byte-pair for byte-pair.

Two hardware limits it enforces rather than discovers at runtime:

- **The run must be 8 bytes, contiguous on both sides.** At fp16 an A-layout `tileSize` run
  is 4×2 = 8 B and works; at int8 it is 4 B and falls onto the CPU fallback. So **a reshape
  into or out of A-layout is an fp16 operation** — quantise *after* it.
- **Anything involving B-layout is a transpose**, not a reshape, and needs the transposer
  kernel (which is correct only at `elem_bytes=1`).
