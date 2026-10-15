# L2 FA: MLA decode over a real 8K-token cache on 16 compute + 4 memory chiplets, with an ablation

One button: `./run.sh all` runs every variant (the cache must exist: `real_kv_cache.py` once).
- `./run.sh run dsv2_l1/full dsv2_l1/base` runs a subset.
- `./run.sh parse` re-parses into `results/fa.csv`; `../ffn/plot_l2.py` draws the figure.

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

DeepSeek-V2-Lite's layer-1 MLA attention in absorbed form (16 heads, d_qk = 576, d_v = 512), one
query at position 8191 over keys 0..8191.
- **The cache is real.** `real_kv_cache.py` runs 8,193 tokens of a public-domain book through
  the checkpoint's tokenizer, embedding, layer 0 and layer 1's KV projection (float32). Per
  token: the normalised latent `c_unit[512]` and the rotated `k_pe[64]`, plus the hidden state
  `x1` entering layer 1 (`cache/dsv2_l1_kv_8193.npz`, `cache/manifest.json`).
- **The device cache is INT8**, one static scale per part (amax onto 127, as `util/calib.py`):
  the key copy (A layout of [keys, 576]) and the value copy (V^T = c8^T).
- **The query is real too:** the last token's `x1` through layer 1's input RMSNorm and `q_proj`,
  RoPE at its position, `q_nope` absorbed through `W_UK` (the kv_a_layernorm gain moved onto the
  query), quantised with calib's tied scales (`s_qt s_c = s_qpe s_kpe`, so a score is one INT32
  dot product) calibrated on the last 64 tokens' queries. The 16 heads fill lanes 0..15 and
  again 16..31 (the kernels' 32 lanes).
- **Device vs the float attention over the float cache:** cosine 0.9955.

## Mapping (split by key, `workloads/fa/common/fa_mc.py`)

| step | where | what |
|---|---|---|
| broadcast | root -> every cluster | Q8 (18 KiB) into the cluster's symmetric buffer |
| pass 1 | every cluster | its 4 K tiles (64 keys x 576 B) through its ring; per tile S^T = K Q^T and the FA softmax with its own seed: the shard's row max |
| chip max | each chip | cluster 1's max copied to cluster 0, the softmax kernel run over the two beats as a lanewise max |
| m* | 16 chips -> root -> every cluster | the chips' maxima met on the root, then broadcast into every cluster's pass-2 arena |
| pass 2 | every cluster | its 4 V tiles through its ring; the softmax SEEDED at m* (no correction across clusters) and PV; the last tile leaves O^T as FP16 |
| local fold, reduce | as the FFN's | each cluster's (O^T, row sums) added on its chip's cluster 0, then folded into the root in the fold order |
| normalise | root | o~ = O c / l (rsqrt of l^2 / c^2), then the transpose: one head per row |

- **The cache streams like a weight.** Every cluster's shard lives in its memory chip's HBM,
  tile-major: its K tiles, then its V tiles. Each tile is one ring chunk (K0..K3 for pass 1,
  then V0..V3 for pass 2), pushed by the memory chip's engine under the chip's prefetcher.
- **The V tiles stream during the exchange of the maxima.** Every DM core runs its tasks in
  order and a ring load blocks until its chunk lands. So the V loads follow the cluster's copy of
  its maximum, V0 and V1 come before m*'s arrival, V2 and V3 after it. On the root, the
  exchange's DM tasks also come after V1.
- **m* lands in place.** Pass 2's softmax arena sits in the symmetric buffer, so m* (the
  multicast or each cluster's pull) writes straight into its `mrun`; `lrun` is zeroed with the
  buffer at set-up.
- **Golden:** `dsv2_datagen.split_attention_golden` (mode key), with the device's arithmetic
  models, adding the chips' partials in the order the reduce does: the root's pulls start from
  the root's own partial, the ChainGather from the chain's first chip with the root's own last
  (`chiplet_mc.fold_order`). The max is exact in any order. Every variant is checked byte for
  byte.

## Variants (the ablation)

Every combination of the three mechanisms: what is broadcast is the query (18 KiB) and m*; what is reduced is the chips' maxima (FP16 MAX) and every cluster's (O^T, l) partial.

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

As the FFN's (`sweep/llm_blocks/ffn/README.md`): two anchors on chip 0x00 bracket the attention (the
prefetchers start after the first, so the whole cache stream counts); SRAM accesses from the
probe's marks. **Checks**, byte-exact: the output `xt` [32 x 512] FP16, m*, and one far chip's
folded (O^T, l).

**Phases** (`../ffn/trace_ffn.py`, each the latest end over the clusters): q on every cluster,
pass 1, m* made on the root, m* in every cluster, pass 2, the chip fold, the reduce, the
normalised output.

## Files

- `real_kv_cache.py`: the cache into `cache/`.
- `run.sh`: the runs, through `common/launch.sh` (watchdog, sim slots).
- `parse_fa.py`: `results/fa.csv`, with `../ffn/parse_ffn.py`.
- **Workloads:** `target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/fa/`.
  - `common/fa_mc.py`: the driver, on `ffn/common/chiplet_mc.py`.
  - `common/fa_data.py`: the cache, the query, scales, golden.
  - `common/fa.mk`: the build.
  - `dsv2_l1/<variant>/params.hjson`: one directory per variant.
