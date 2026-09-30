# mem/ — where data lives

A node's operands are memory HANDLES, and where a workload's arrays go is decided by the
platform. Both are here, so the one place that turns "an array" into "the handle a kernel
reads" is in one directory.

| file | what it is |
|---|---|
| `bingo_mem_handle.py` | How a node names memory: `BingoMemAlloc` (an allocation), `BingoMemAllocView` (a byte offset into one), `BingoMemSymbol` (a C variable, optionally another chip's copy), `BingoMemFixedAddr` (an absolute address, optionally tagged with its pool). |
| `bingo_data_staging.py` | `DataStaging`: places a workload's baked inputs and goldens and returns each one's handle -- in the host image as C arrays (`BingoMemSymbol`), on the memory chiplet in `mempool.bin` (`BingoMemFixedAddr`), or in its HBM (`put_hbm`, `build/hbm/`); then `emit` writes `data.h` and the images. |

## Why there are four handle types

Because where a buffer lives is decided by the *platform*, not the workload. A config with
a memory chiplet stages its arrays into `mempool.bin` and addresses them absolutely; a
config without one emits them as C arrays and addresses them by symbol. Code that offsets
a handle has to handle all four or it works on one platform and silently reads the wrong
tile on the other — see `at_offset` in `libs/comm/ports.py`.

Getting the placement wrong does not fault: a memory-chiplet address on a config without
one is simply unmapped, and the checks compare one piece of garbage against another.
`DataStaging` reads the platform so a workload never has to choose.

Both are imported flat, like the rest of the compiler: `import _bingo_paths` first, then
`from bingo_data_staging import DataStaging`.
