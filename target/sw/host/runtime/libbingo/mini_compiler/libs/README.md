# libs/ — reusable blocks, and the linker that assembles them

A block is a sub-DFG with a declared interface. It states what it consumes and produces;
the linker binds one block's outputs to the next block's inputs. The mini-compiler still
sees ONE assembled graph and runs every pass on it exactly as before — this layer only does
assembly.

It is a function call whose body happens to be a graph.

| | |
|---|---|
| [`comm/`](comm/) | **Common** machinery: ports, ctx, the linker, and how an operand that does not match gets moved or reshaped. |
| [`block/`](block/) | The blocks: FlashAttention, its fold, the MoE FFN. |
| [`verify/`](verify/) | Host-side readbacks and comparisons, named by precision. |

## Every port carries four things

Shape, **layout**, **precision** and **mem_level** — and all four are part of the signature
rather than an assumption, because none of them faults when it is wrong:

- a mismatched **layout** is a permutation with the same flat length, so every byte is read
  and written and the arithmetic runs — and the answer is a scrambled tensor. On random
  test data the golden is scrambled identically and it *passes*.
- a mismatched **precision** reads two fp16 elements as one int8 pair. Nothing is out of range.
- a **mem_level** naming a memory pool the platform does not have is simply unmapped. The
  loads return whatever the fabric gives back and the checks compare garbage against
  garbage.

An input port may leave `mem_level` unset, meaning *"wherever you have it"* — the block
fetches it. That is a promise, and the block keeps it by overriding `needs` with the level
its own transfers read from. A port that names no level and a block that adds none in
`needs` is refused: nothing would ever resolve where the operand is read.

## Two rules that shape everything here

**The linker never inserts a node.** A block loads its own operands, because only the block
knows it needs them in its own cluster's L1 — the GEMM and SIMD have no AXI port. So the
loads a block already builds *are* the transport, and the linker only adds edges. This is
not fastidiousness: node creation order is dispatch order on this machine, so a linker that
injected nodes would silently reschedule the graph. It is also why a block that closes its
own layout or location gaps says so by overriding `needs` — the conversion happens in the
*block's* build, in the block's own order, and the contract is checked against `needs`
rather than against a flag that could disagree with what build() does.

**Lossless is automatic, lossy is explicit.** A layout change is a permutation with exactly
one right answer, and a buffer in the wrong memory only has to be moved, so a block may
close either on its own. A precision change needs a *scale*, and there is no correct
default — the right one depends on the range of the data, which the linker cannot see.
Choosing one silently is how a tensor ends up 83% saturated and agreeing with its golden
anyway. It is refused, by name.

## Using it

```python
from libs import Ctx, Pipeline, Port
from libs.block import FlashAttention, fa_gather

blk = FlashAttention(bc=32, br=32, dhead=128, nkv=8, clusters=4, decomp="kvsplit")
r = Pipeline(ctx).add(blk, name="attn", bind={"q": ..., "k": ..., "v": ...}).result
fa_gather(ctx.scope("attn"), blk.cfg, r.extra["shards"])   # the fold is a choice
```
