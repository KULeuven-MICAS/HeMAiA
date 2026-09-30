# blocks/ — the blocks

Each is a sub-DFG with a declared interface, built once, in pipeline order.

| file | |
|---|---|
| `flash_attention.py` | `FlashAttention` over one or more clusters, and `FaCfg`, which holds every parameter it has. |
| `gather.py` | `fa_gather`, the in-fabric fold of attention's per-cluster partials. A function, not a block. |
| `moe.py` | `MoeFFN` — a mixture-of-experts feed-forward layer with CERF-skipped losers — and `MoeCfg`. |
| `linear.py` | `Linear` — one INT8 GEMM, or a one-token GEMV (`gemv=True`, VersaCore's (1, 4, 32) shape; `groups=G` for per-head GEMVs), with its operand loads, and `LoadStream`, a cluster's shared weight slabs. A weight larger than one chunk is streamed in column chunks through `w_buffers` L1 buffers (double buffering by default), from main memory or from the memory chiplet's HBM. Every weight matrix in a layer is this block with different shapes. |
| `simd/` | The per-row and per-element operators a layer is glued together with, split on whether layout is load-bearing: `simd/pointwise.py` (`Quantize`, `Dequantize`, `Residual` — elementwise, any layout), `simd/norm.py` (`RMSNorm` — reduces along a row, so orientation is worth 3x), `simd/rope.py` (`RoPE` — rotates along a row), `simd/row.py` (one token's row around a GEMV: `RMSNormRow`, `QuantizeARow` into row 0 of the GEMV's A operand -- with `segs`, every head's operand of a per-head GEMV in one task --, `ScaleCols` — the per-column dequantisation --, `SoftmaxRow` (the router's), `SwigluARow` (straight into the down GEMV's operand, a static or a slot's scale), `ScaleRowBySlot` (a routed weight chosen at run time) and `AddRow`), `simd/common.py` (the tile and the beat). |
| `mla.py` | Multi-head latent attention (DeepSeek-V2), one token per pass: `RopeRows` (RoPE over the heads' q_pe and the k_pe, gathered from the projections' outputs), `CacheAppend` (the token's row into the key and value copies of the latent cache), `QAssemble` (the score operand Q8 from every head's q~ and q_pe, two scales) and `MlaAttention` (the online softmax over key tiles on one cluster, the last tile out as FP16, o~ = O / l, transposed). |
| `route.py` | A mixture of experts whose experts the router picks at run time, as data: `MoeRoute` (the top k as an expert-slot record: ids, weights and each expert's table entry) and `SlotLoad` (a slot's tensor from the address the record names). A slot's GEMVs are `Linear(w_slot=...)`. |
| `move.py` | `Broadcast`, `Pull` (L1 to L1 across clusters), `Fetch` (L3/HBM to L1), `Stash` (a checked buffer copied to L3 by its own cluster as soon as it exists) and the zero-cost views `Slice`, `TransposeView`, `View`. |
| `reshape.py` | `Reshape` — an explicit layout change as a stage, for the gap between two blocks that both live in L1. |
| `shard.py` | `Scatter` and `Gather` — the two ends of a row split, so a layer can run a row-independent operator on every cluster. |

`multi_chip/workloads/dsv2/two_chiplet` builds DeepSeek-V2-Lite's layer 1 from them, one stage
at a time and as the full layer, its weights in the HBM.

## Streamed weights, and the HBM

A weight is the one operand that is read once per pass and is usually larger than L1, so
`Linear` streams it: the B layout is n-major, so a run of output columns is one contiguous
run of bytes and a complete B operand. The chunks rotate through `w_buffers` L1 buffers
(`w_chunk_bytes` each, 128 KiB by default) on the DM core's iDMA while the array computes
the chunk before. The order is carried by edges -- each task on its chunk's load and on the
task before it, each load on the task that last read its buffer -- because creation order is
dispatch order only where the compiler's topological sort cannot reorder.

With a `LoadStream` (one per cluster), every streamed operand on the cluster -- every
projection's chunks, and MlaAttention's K/V tiles -- goes through the same few slabs in
build order: a load goes to the next slab and waits only for whatever last read it, which
may belong to the previous block. The next weight then starts streaming while the last
chunks of this one compute, and a layer of N projections holds one set of slabs. The
stream's loads are not reported as block sources, so gating sources (for static L1 reuse)
never serialises them behind a producer.

A slot of a routed expert streams the same way from an address chosen at run time:
`Linear(w_slot=(slot, "gu"), rec=...)` reads the expert-slot record `MoeRoute` wrote, and
its loads form a chain after the record. A fan-out of many loads from the record's producer
would need one dependency tag per edge in one cell (the compiler allows 32).

`MemLevel.HBM` is the memory chiplet's HBM (`hemaia_twochiplet_16MBL3_4cluster` only).
`DataStaging.put_hbm()` places an array there and returns a fixed address that records its
pool, and `emit()` writes the `build/hbm/manifest.txt` the testharness loads. An HBM operand
is loaded straight into L1 by the block that reads it, never hoisted to L3 first: it is
streamed once, so a hoist would only read it twice.

## One block, one cluster

Every block here runs where its cfg says, and its ports declare it, so the pipeline refuses
a binding that would have a kernel read another cluster's TCDM. Spreading an operator over
the machine is therefore something the **layer writes** — `Scatter`, one block per cluster,
`Gather` — which keeps the decomposition visible in the layer's own source and every node's
placement on a port that is checked.

Two blocks do span clusters, and each states why it is one construct rather than several:
`FlashAttention` interleaves its per-cluster pipelines (node creation order is dispatch
order, so emitting them back to back would be a different schedule), and `MoeFFN`'s expert
lanes are branches of one conditional fork. Both declare **per-cluster ports**, so what
crosses their boundary is still explicit.

A `Gather` costs one local copy on the root that a hand-written split would not: a block
allocates its own output and cannot be handed someone else's buffer to write into. It is
one iDMA move on the otherwise idle DM core.

## Which ops care about layout

**Elementwise** — `Quantize`, `Residual` — touch each value independently, so any layout
works: a permutation of the inputs is the same permutation of the output.

**Per-row** — `RMSNorm`, `RoPE` — reduce or rotate *along* a row, so the row must be
contiguous. In D-layout `(m, n, r, c)` a matrix row is **not** contiguous, so handing
D-layout to `RMSNorm` normalises groups that are not rows. It does not fault and it does
not go out of range; the answer is a well-formed tensor of wrong numbers. Their ports say
`row_major` and mean it.

**And among the per-row ops, only the REDUCING one cares about orientation.** The SIMD
block reduces along beats for free (one FP32 accumulator per lane) and across the lanes of
a beat only through a serialised fold that stalls the reader once per row. Store the tile
transposed and one lane *is* one token, so the fold disappears and the scale rides back in
as a sticky operand instead of a replicated plane: `RMSNorm` measures 3,135 → 1,073 cc at
[32, 128]. `RoPE` has no reduction, so there is nothing for it to win. `NormCfg` carries `in_layout`
/ `out_layout`; left unset each is a knob the block owns and the resolver fills from the
neighbours, and pinned it is a boundary the layer asked for.

That forces the order a layer has to use, and it is hardware, not taste:

```
GEMM (D/f16) -> Reshape to row_major (f16) -> RMSNorm -> Reshape to A (f16) -> Quantize -> GEMM
```

Both reshapes are **FP16** and the quantise comes **after**, because a conversion into or
out of A-layout needs an 8-byte run contiguous on both sides — at int8 an A-layout
`tileSize` run is 4 bytes and falls off the hardware path. Quantising first would make the
reshape impossible; `comm/nest.py` refuses it by name.

## FlashAttention takes three ports, whatever the placement

`q`, `k`, `v`. `FaCfg.clusters` names the clusters it runs on; how the operands are
split across them is the block's business:

| decomposition | `q` | `k` | `v` |
|---|---|---|---|
| `headpar` | `[clusters·Br, d]`, one query head per cluster, stacked | `[Bc, d]`, shared (that IS the GQA relation) | `[Bc, d]`, shared |
| `kvsplit` | `[Br, d]`, shared | `[clusters·Bc, d]`, disjoint KV shards, stacked | `[Bc, d]`, shared |

The block slices them with byte offsets. Its per-cluster tables are indexed by position,
so `validate()` refuses a placement that is not contiguous from 0 — lifting that is an
audit of every such index, not a parameter change.

It also names **no memory level** on those ports. Where the caller keeps Q, K and V is the
caller's business — main memory, the memory-chiplet pool, or an L1 buffer a previous block
just wrote. What the block knows is where its own transfers read from, and that is
`FlashAttention.needs` (L3). Declaring L3 on `inputs` made a true statement about this block's
loads into a false demand on everyone upstream.

## Every parameter comes from FaCfg

There are no module-level knobs. `FaCfg` has four labelled groups — shape, topology,
**internal performance tuning** (24 fields, every default the measured best on this
quadrant), and hardware, which is *checked* rather than chosen.

Two of them come from the split cluster's FA datapath extensions (snax
`hw/chisel/doc/fa_datapath_extensions.md`):

- **`score_shift`** (k) — the QK D port writes the score tile as RNE(S · 2^-k), and the
  softmax is handed `exp_scale` = a · 2^k, so `score_scale` keeps meaning *a on the raw
  INT32 score*. A full-range INT8 score over d = 128 is 31× past FP16's range and becomes
  Inf without the shift. A power of two only moves the exponent, so the shift loses no
  precision and changes no result (`test_libs.py` checks l and O are bit-identical). Unset,
  k is the smallest that keeps every INT8 score finite (6 at d = 128). The running m the
  kernels leave is in the shifted domain; the gather folds `exp_scale · m`, which is a·m.
- **`p8_pitch`, `p8_nest`** — P8's 64 B blocks 160 B apart, and the two ping-pong P
  buffers nested in one region. At the dense 64 B PV's two operand streams step round the
  TCDM banks at different rates and collide (1.44 cycles a pass on the snax reference,
  1.15 at 160). Nesting costs 8 KiB at Bc = 512 instead of 48.

`validate()` refuses a tuning knob the chosen decomposition never reads. That is not
pedantry: all fourteen pull/broadcast/push knobs are `headpar`-only (under kvsplit the
shards are disjoint, so there is nothing to share) and `stagger_first_v` is `kvsplit`-only.
Unchecked, setting one on the wrong decomposition builds, runs, and reports an unchanged
cycle count — which reads as *"the optimisation does not help"* rather than *"it was never
applied"*.

## The output stays where it is

`o_c{c}` is that cluster's accumulator, INT32, in the D port's **scatter** layout (`d32` —
a different bijection from `D`), in that cluster's L1. Under `kvsplit` with more than one
cluster there are also `m_c{c}` and `l_c{c}`, the partial running max and sum.

Those are two ports rather than one because that is what they physically are: the max lives
in the softmax arena and the sum in the last score buffer, written by different kernels
into different allocations. The single contiguous FP32 `(m, l)` the junction folds does not
exist until the gather's pack builds it.

## What the gather folds

A kvsplit shard's partial is the **triple** `(m, l, O)`. The complete combine is

```
m* = max_c m_c
l* = sum_c exp(m_c - m*) * l_c
O* = sum_c exp(m_c - m*) * O_c
```

`fa_gather` does the **first two**, in the fabric, as the partials cross the monoid
junction. `O` is left as `o_c{c}`, still scaled by that cluster's own `m_c`; applying the
third line is the caller's, using the `m*` the fold returns. Reading `o_c{c}` as final
gives an answer wrong by a per-cluster exponential — and where the maxima are close, wrong
by little enough to look plausible.

## Why the gather is not a block

Two reasons, and the second is the binding one.

The fold is a **choice**, not a consequence — in the fabric, on the host, or not at all,
since a next stage sharded the same way would pay for a gather and then re-split what it
gathered. And it **cannot be expressed as a Block**: a Block owns its operands through
ports, and a port names a buffer with a layout and a producer. What the fold reads is each
cluster's softmax *arena*, a buffer the previous block is still updating in place — a
running state, not a produced tensor. Declaring it as a port would promise something the
port model cannot honour, so it takes the shards directly and says so.

Under `headpar`, or with one cluster, it is a no-op: there is nothing to fold.

## Not yet here

**Causal masking.** There is none — in this file or anywhere in the tree. Every query
tile attends to every key tile, which is right for a decode step and wrong for a real
prefill. `NQ` sets how many query tiles run, and that is the *only* difference between
`fa_decode_4cluster` (NQ=1) and `fa_prefill_4cluster` (NQ=2), so "prefill" here means
multiple query tiles over full attention. The mask is a SIMD-kernel change — applied to
the score tile between QK and the softmax — not a parameter. See the TODO on the class.

**Plain attention.** `flash_attention.py` is named for what it is; a straightforward
non-flash implementation would land beside it as `attention.py`.
