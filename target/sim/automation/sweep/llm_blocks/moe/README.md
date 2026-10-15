# L2 MoE: DeepSeek-V2-Lite's layer-1 MoE on 16 compute + 4 memory chiplets, with an ablation

One button: `./run.sh all` runs every variant (the weights are `sweep/common/weights`' cache).
- `./run.sh run dsv2_l1/full dsv2_l1/base` runs a subset.
- `./run.sh parse` re-parses into `results/moe.csv`; `../ffn/plot_l2.py` draws the figure.

## Question

The paper's three mechanisms, each against its conventional counterpart on the same hardware and
the same compute:
- **broadcast**: one D2D router multicast into every cluster's L1 and a landed flag, against
  every cluster pulling the root's copy with its iDMA;
- **reduce**: the xDMA ChainGather folding in transit, against the root pulling every partial and
  folding it on its SIMD;
- **sync**: the BINGO hardware scheduler resolving every cross-chip dependency per edge, against
  barriers: the vanilla HeMAiA chip barrier at every cross-chip sync point.

How much does each buy inside a real LLM layer whose weights stream from the memory chiplets,
alone and together?

## Workload

DeepSeek-V2-Lite's layer-1 MoE at its real size, one token: 64 routed experts (inter 1408), top-6,
and the 2 shared experts as one MLP of 2816. **The weights are the checkpoint's own** (`sweep/common/weights`'
cache, the FFN's pinned commit): experts INT4 per output column, the router INT8 per column, the
RMSNorm gain applied to the input before it is quantised (as the FFN's). The token is a seeded
normal draw.
- **Route:** experts [20, 8, 56, 18, 26, 55]; both arms pick the same, with the same FP16 weights
  on this token.

## Mapping (`workloads/moe/common/moe_mc.py`)

| step | where | what |
|---|---|---|
| broadcast | root -> every cluster | x8 (2 KiB) |
| router | every chip, cluster 0 | its 4 experts' logits: an INT8 GEMV over 32 columns (28 zero: one SIMD beat), dequantised |
| route | the chips -> root -> every chip | see below |
| shared expert | the x = 0 column (00..03), 8 clusters | 352 intermediate columns each, streamed before the route |
| routed slots | 6 chip pairs (10/11, 30/31, 40/41, 12/13, 32/33, 42/43) | each slot on 4 clusters, 352 columns each; the slot's weights, factors and SwiGLU scale come from the chip's route RECORD; its prefetcher pushes them once the record is in its L3 |
| fold, reduce | as the FFN's | every cluster's weighted partial over all 2048 outputs, added on its chip, then folded into the root |

**The route, two arms** (`route: fold | gather`, the collectives switch with `bcast` and `reduce`):
- **fold** (in fabric). Each chip packs its 4 (logit, id) pairs into a TopK beat and its softmax
  partial (m_k, l_k) into a softmax-statistics group (`moe_router_pack`: FP16 m and l at lane 0
  of a key beat and a value beat). Two ChainGathers fold them across the chips on the root in
  parallel: the TopK junction on cluster 0's xDMA (the 8 largest logits with their ids), the
  softmax-statistics junction on cluster 1's ((m*, l*)). The root applies them to the 8
  candidates (`simd_softmax_row_ext`).
- **gather** (baseline). The root pulls every chip's 4 logits into one row of 64 and runs the
  softmax over it.
- Then the route (probabilities, ids) goes to every chip's cluster 0 (multicast or pulls), where
  `moe_route` picks the top 6 against the chip's own expert table. The record goes to the chip's
  L3 (its prefetcher) and to cluster 1.

**The route goes first.** Every DM core runs its tasks in order and a ring load blocks until its
chunk lands. On the shared-expert chips the router's tasks (and on the root the route's pulls and
broadcast) would otherwise sort between the shared expert's loads and wait for the shared stream,
while every routed weight waits for the route. So cluster 0's stream on each shared chip waits for
the chip's last task of the route, the root's for the route's arrival (`moe_mc._route_first`); the
prefetcher still pushes the shared chunks into L3 meanwhile.

**Golden.** Every device step's arithmetic model, the softmax-statistics junction's included (its
exponential is a 128-entry table, `moe_data.ssj_exp`, and its combine an FP32 FMA rounded to FP16
at every hop, `ssj_fold`), so both
arms are checked byte for byte: y, one slot chip's record, one far chip's partial. FP16 addition
does not associate, so the golden adds in each arm's order: the root's pulls start from the
root's own partial; a ChainGather (the reduce, the TopK and softmax-statistics folds) starts from the chain's
first chip and adds the root's own last (`chiplet_mc.fold_order`, `chain_order`).

## Variants (the ablation)

Every combination of the three mechanisms: what is broadcast is x (2 KiB) and the route; what is reduced is the router's logits (TopK and softmax-statistics ChainGathers on the root) and every chip's partial.

| variant | rung | broadcast | reduce | sync |
|---|---|---|---|---|
| `base` | R0 | iDMA pulls | root gathers | chip barriers |
| `bcast` | R1 | multicast | root | chip barriers |
| `bcast_reduce` | R2 | multicast | ChainGather | chip barriers |
| `all` | R3 | multicast | ChainGather | hardware, per edge |
| `reduce_sync` | R3 without broadcast | pulls | ChainGather | hardware |
| `bcast_sync` | R3 without reduce | multicast | root | hardware |
| `reduce` | reduce alone | pulls | ChainGather | chip barriers |
| `sync` | sync alone | pulls | root | hardware |

**Barrier sync, the baseline** (`sync: barrier`, `barrier_mode: bcast`). Inside the window no
task depends on another chip's. The vanilla HeMAiA chip barrier (`snrt_hw_chip_global_barrier`,
`chip_sync.h`) stands in for those dependencies: one barrier task on every chip's cluster-0 DM
core, which announces its arrival with one 0xFF broadcast store and polls the others' locally.
Every chip waits for every chip, which is what the per-edge hardware sync removes. Barriers sit
after the start anchor, before a pull broadcast of a value made inside the window, and before the
reduce; a multicast needs none (its landed flag says the data is there). Barriers are chained per
chip, so every chip runs them in program order.
**Hardware sync** (`sync: edge`): every cross-chip dependency is a BINGO hardware edge, so a task
waits for exactly its producers.

Weights stream plain (`weight_crest: false`): CREST is not part of the paper, and the paper's
cluster has no decompressor. The first round, a CREST ladder on the 16 MiB-L3 platform, is in
`results/round1/` (its runs and variants: `isca_archive/2026-10-07_l2_round1`).

## Measurement

As the FFN's (`sweep/llm_blocks/ffn/README.md`). **Phases** add the route: the last slot chip's record.

## Files

- `run.sh`, `parse_moe.py` (with `../ffn/parse_ffn.py`).
- **Workloads:** `target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/moe/`:
  `common/moe_mc.py` (the driver, on `ffn/common/chiplet_mc.py`), `common/moe_data.py`
  (weights, route, golden), `common/moe.mk`, `dsv2_l1/<variant>/params.hjson`.
