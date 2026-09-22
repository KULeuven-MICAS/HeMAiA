# block/ — the blocks

Each is a sub-DFG with a declared interface, built once, in pipeline order.

| file | |
|---|---|
| `attention.py` | `Attention` — FlashAttention over one or more clusters — and `FaCfg`, which holds every parameter it has. |
| `gather.py` | `fa_gather`, the in-fabric fold of attention's per-cluster partials. A function, not a block. |
| `moe.py` | `MoeFFN` — a mixture-of-experts feed-forward layer with CERF-skipped losers — and `MoeCfg`. |

## Attention takes three ports, whatever the cluster count

`q`, `k`, `v`. How they are split across clusters is the block's business:

| decomposition | `q` | `k` | `v` |
|---|---|---|---|
| `headpar` | `[clusters·Br, d]`, one query head per cluster, stacked | `[Bc, d]`, shared (that IS the GQA relation) | `[Bc, d]`, shared |
| `kvsplit` | `[Br, d]`, shared | `[clusters·Bc, d]`, disjoint KV shards, stacked | `[Bc, d]`, shared |

The block slices them with byte offsets. Declaring nine ports instead made the caller know
about clusters for no reason.

## Every parameter comes from FaCfg

There are no module-level knobs. `FaCfg` has four labelled groups — shape, topology,
**internal performance tuning** (22 fields, every default the measured best on this
quadrant), and hardware, which is *checked* rather than chosen.

`validate()` refuses a tuning knob the chosen decomposition never reads. That is not
pedantry: all fourteen pull/broadcast/push knobs are `headpar`-only (under kvsplit the
shards are disjoint, so there is nothing to share), `stagger_first_v` is `kvsplit`-only,
and setting one on the wrong decomposition used to build, run, and report an unchanged
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
