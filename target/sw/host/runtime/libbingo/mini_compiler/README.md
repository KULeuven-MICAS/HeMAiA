# mini_compiler

Turns a task graph into the C that runs it. A workload assembles a `BingoDFG` -- nodes on
cores, edges between them, buffers they read and write -- and `bingo_compile_dfg` emits
`offload_bingo_hw.h`, plus the reports and the checks that say the graph is safe to run.

## Layout

| | |
|---|---|
| [`graph/`](graph/) | The DFG data model: nodes, memory handles, the graph itself. Knows nothing about passes. |
| [`kernels/`](kernels/) | The kernel ABI. One args class per device/host kernel, matching the C structs in `libbingo/include`, one file per engine. |
| [`passes/`](passes/) | What runs OVER an assembled graph, in `bingo_compile_dfg` order, plus the analyses those passes use. |
| [`platform/`](platform/) | The machine: which core carries which engine, how many clusters, what transfer sizes the datapath accepts. |
| [`libs/`](libs/) | Reusable blocks -- attention, its fold, the MoE FFN -- and the linker that assembles several into one graph. |
| [`tests/`](tests/) | Runnable checks. No framework: `pixi run python3 tests/test_libs.py`. |

Each directory has its own README describing what is in it and why.

## What a compile actually does

`bingo_compile_dfg` (`graph/bingo_dfg.py`) runs them in a fixed order, and the order
matters because each depends on the last:

1. **entry and exit nodes** -- one source, and a per-core exit chain.
2. **conditional regions** -- the CERF gating, and its two validations. This sits BETWEEN
   the two halves of the transform because it adds nodes, and the core-sequencing edges
   below have to cover them.
3. **core sequencing and dummy nodes** -- the ordering the manager needs.
4. **dependency tags** -- set/check info per node, then a minimum chain cover to fit them
   into the available tags.
5. **hang check** -- are those tags sufficient for every edge?
6. **static L1** -- place the cluster buffers, reusing memory between buffers whose
   lifetimes cannot overlap.
7. **emit** -- write the C. Placement and handle validation run HERE, at the point where a
   wrong answer would be written out, rather than as a separate pass up front.

Alongside the hang check there is the **sim check**, which replays the graph against a
model of the hardware task manager. It is a deadlock oracle, not a performance model.

## Importing it

Modules are imported FLAT -- `from bingo_dfg import BingoDFG` -- and reached by putting
this directory on `sys.path` and importing the bootstrap first:

```python
sys.path.append(f"{ROOT}/target/sw/host/runtime/libbingo/mini_compiler")
import _bingo_paths          # appends graph/, kernels/, passes/, platform/
from bingo_dfg import BingoDFG
```

That is the same arrangement, for the same reason, as `util/sim/_usg_paths.py`: grouping
the files should not cost sixty workloads their import lines. `libs` is a real package and
bootstraps itself, so `import libs` needs no preamble.

## Two things to know before changing anything here

**Never locate a file by counting `..`.** A fixed number of parent steps is right for
exactly one directory depth and SILENT when it is wrong -- the computed path is simply a
directory that does not exist, so the failure surfaces much later and somewhere else. When
these files were grouped into subdirectories, two such counts in `bingo_platform.py` broke,
and 29 workloads reported *"generated core-role map is missing -- run `make snax-sw-gen`"*.
The file was there the whole time. Use `_bingo_paths.repo_root()`, which searches for a
marker.

**Node creation order is dispatch order.** The manager issues tasks in the order they were
added, so reordering node construction silently reschedules the graph. This is why the
linker in `libs/` adds edges but never inserts a node.
