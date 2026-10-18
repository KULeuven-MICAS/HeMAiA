# L2 MoE: DeepSeek-V2-Lite's layer-1 MoE on 16 compute + 4 memory chiplets, with an ablation

One button: `./run.sh all` runs every variant (the weights are `sweep/common/weights`' cache).
- `./run.sh run dsv2_l1/all dsv2_l1/base` runs a subset.
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

## Prefill (`stages.sh`)

DeepSeek-V2-Lite's layer-1 MoE over a prompt (`workloads/moe/common/moe_prefill_mc.py`), one stage a
run: `./stages.sh poc<v>` (tiny: every expert's first 384 intermediate columns, the shared MLP's first
1024), `./stages.sh full<v>` (the real size), `./stages.sh gtb_<variant>` (the reduce on its own,
`common/gather_tb_mc.py`). Each variant's `params.hjson` lists its levers.

| stage | tokens | what it adds | result |
|---|---|---|---|
| `full14` | 128 | register-launched weight pushes | PASS, 1,810.6 us, 29.9 % MAC |
| `full17` | 128 | queued weight pushes (memory chip descriptor queue) | PASS, 1,483.2 us, 36.5 % MAC |
| `poc18s` | 256 | full17's levers, serial ChainGathers | PASS, 1,036.3 us |
| `poc18c` | 256 | the two clusters' ChainGathers at the same time | PASS, 1,008.3 us |
| `poc19` | 256 | poc18c with the real size's 8 X blocks | PASS, 1,034.3 us |
| `poc24` / `poc25` / `poc27` | 256 | 64-column slices, half slabs, the route on every chiplet; + RX yield; serial chains | all stop at `y_chain_c0j5` (64-column ChainGathers under the down stream) |
| `full29` | 256 | the real size: shared MLP split evenly | head measured (route end 313 us, rings full from 180 us); stopped |
| `poc35` | 256 | prefetchers after the X multicast, the route tile buffer, the shared MLP chunk by chunk, X in the expert buffers, 128-column slices (`inter_sh` 2816) | PASS, 1,146.6 us |
| `full35` | 256 | poc35 at the real size | head measured (route end 204 us); stopped |
| `full37` | 256 | full35 + the route collection and fetches early, X in one multicast | PASS, 1,867.7 us, 58.0 % MAC (down phase 1,015-1,867 us: the root finishes every slice 55-62 us after the other chips) |
| `poc40` | 256 | full37 + expert slots smallest-largest-alternating, second shared pass after slot 2, concurrent chains, X put by both root clusters, per-slice GEMM phases | PASS, 1,089.6 us (the root's lag stays ~50 us a slice) |
| `gtb_serial_quiet` / `gtb_concurrent_quiet` | - | 16 ChainGathers of 32 KiB, serial / two at a time, no weight stream | PASS, 238.5 / 182.7 us |
| `gtb_serial_stream` | - | the same, serial, under the queued down stream | first chain 32 us; stopped by the xDMA stall report (now a warning) |
| `poc41_1` | 256 | poc40 + the root's store of slice j after slice j+1's loads (`store_lag` 1) | PASS, 888.2 us; root lag a slice -0.3..+12.4 us (poc40 ~50) |
| `poc42` | 256 | poc40 + `store_lag` 2 | hung at slice 3: the ChainGather finish-vs-push deadlock (concurrent chains); `stg_poc42_dbg` replays it on the debug simv |
| `poc43` | 256 | poc41_1 without holding the pushes for X | PASS, 972.6 us; X lands at 122.9 us instead of 47.8: the hold stays |
| `gtb_serial_v41` / `gtb_concurrent_v41` / `gtb_concurrent_v41_drift` | - | 8 ChainGathers of 64 KiB a cluster under the stream with v41's traffic (2 slabs, 9-slot rings); serial / concurrent / concurrent with drifting start offsets | PASS all |
| `full45` | 256 | full37 + `store_lag` 1, `slot_order` [1, 2, 3, 0], `shared_after` [0], serial chains (`chain_concurrent` false) | PASS, 1,587.7 us, 68.2 % MAC; the serial chains (28-39 us each under the stream) set the down phase's pace |

Kept variants: `dsv2_l1_prefill/{v17, v37, v45}`, `dsv2_l1_prefill_tiny/{v40, v41, v42, v45}`, `gather_tb/{serial_quiet,
concurrent_quiet, serial_v41, concurrent_v41, concurrent_v41_drift}`. Every other variant of the table is archived (params
only) in `isca_archive/2026-10-15_moe_prefill_variants` and `isca_archive/2026-10-16_moe_prefill_variants`; copy its
directory back to rerun it.

Concurrent ChainGathers run the same way round the ring: two in opposite directions deadlock on the
D2D links. Earlier variants (tiny all..v11, full all..v16 but v14, the gather testbench's HBM-partial
pair) are in `isca_archive/2026-10-15_moe_prefill_variants`.

## Files

- `run.sh`, `parse_moe.py` (with `../ffn/parse_ffn.py`): the ablation.
- `stages.sh`: the prefill; `tools/timeline.py` (a stage run's task spans), `tools/gen_check.sh`.
- **Workloads:** `target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/moe/`:
  `common/moe_mc.py` (the driver, on `ffn/common/chiplet_mc.py`), `common/moe_data.py`
  (weights, route, golden), `common/moe.mk`, `dsv2_l1/<variant>/params.hjson`.
