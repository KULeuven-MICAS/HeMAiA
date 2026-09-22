# graph/ — the DFG data model

What a workload builds. No pass logic lives here; the passes are mixed into `BingoDFG`
from [`../passes/`](../passes/), which is why this directory can be read on its own.

| file | what it is |
|---|---|
| `bingo_dfg.py` | `BingoDFG` itself: nodes, edges, handles, and `bingo_compile_dfg`, which runs the passes in order. The class is assembled from mixins, so this file is mostly the graph API and the compile driver. |
| `bingo_node.py` | `BingoNode`: a kernel, its args, and the (chiplet, cluster, core) it is placed on. Also the dependency-tag fields the descriptor pass fills in. |
| `bingo_mem_handle.py` | How a node names memory: `BingoMemAlloc` (an allocation), `BingoMemAllocView` (a byte offset into one), `BingoMemSymbol` (a C variable), `BingoMemFixedAddr` (an absolute address). |

## Why there are four handle types

Because where a buffer lives is decided by the *platform*, not the workload. A config with
a memory chiplet stages its arrays into `mempool.bin` and addresses them absolutely; a
config without one emits them as C arrays and addresses them by symbol. Code that offsets
a handle has to handle all four or it works on one platform and silently reads the wrong
tile on the other — see `at_offset` in `libs/comm/ports.py`.

## The mixin split

`BingoDFG` is built by inheriting `BingoDFGTransformsMixin`, `…ConditionalMixin`,
`…DescriptorMixin`, `…ValidateMixin` and `…EmitMixin`. The mixins may not import
`bingo_dfg` — it imports them — so anything they share has to live somewhere neither
imports. There used to be a `bingo_dfg_common.py` holding exactly that; its contents moved
to where they are actually used (the kernel→engine map beside the core-role map it is
checked against, the task-list word size beside the descriptor measured in those words),
and `bingo_dfg` re-exports them so old import sites keep working.
