# block/ — the blocks

Each is a sub-DFG with a declared interface, built once, in pipeline order.

| file | |
|---|---|
| `flash_attention.py` | `FlashAttention` over one or more clusters, and `FaCfg`, which holds every parameter it has. |
| `gather.py` | `fa_gather`, the in-fabric fold of attention's per-cluster partials. A function, not a block. |
| `moe.py` | `MoeFFN` — a mixture-of-experts feed-forward layer with CERF-skipped losers — and `MoeCfg`. |

## FlashAttention takes three ports, whatever the cluster count

`q`, `k`, `v`. How they are split across clusters is the block's business:

| decomposition | `q` | `k` | `v` |
|---|---|---|---|
| `headpar` | `[clusters·Br, d]`, one query head per cluster, stacked | `[Bc, d]`, shared (that IS the GQA relation) | `[Bc, d]`, shared |
| `kvsplit` | `[Br, d]`, shared | `[clusters·Bc, d]`, disjoint KV shards, stacked | `[Bc, d]`, shared |

The block slices them with byte offsets. Declaring nine ports instead made the caller know
about clusters for no reason.

It also names **no memory level** on those ports. Where the caller keeps Q, K and V is the
caller's business — main memory, the memory-chiplet pool, or an L1 buffer a previous block
just wrote. What the block knows is where its own transfers read from, and that is
`FlashAttention.needs` (L3). Declaring L3 on `inputs` made a true statement about this block's
loads into a false demand on everyone upstream.

## Every parameter comes from FaCfg

There are no module-level knobs. `FaCfg` has four labelled groups — shape, topology,
**internal performance tuning** (22 fields, every default the measured best on this
quadrant), and hardware, which is *checked* rather than chosen.

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
