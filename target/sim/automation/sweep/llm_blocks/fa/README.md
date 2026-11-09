# L2 FA: the attention block's ablation on 16 compute + 4 memory chiplets

Each run is one workload of `workloads/fa/` through `stages.sh <stage>` (its watchdog profile, tags `stg_<stage>`):
- **The decode ladder** (the paper's): `stages.sh llama3dec_<rung>` for `base`, `bcast`, `bcast_reduce`,
  `all` (`workloads/fa/llama3_decode_full/`; the paper's runs are `stg_llama3dec_<rung>_w`), Llama-3-8B's whole decode attention block, one token over an
  8,192-position cache. `exp/eval6_fa/decode.csv` in the paper and the FA tracker's decode section hold the windows.
- **The prefill ladder** (the paper's): `stages.sh llama3abl_base`, `llama3abl_bcast`, `llama3abl_all`
  (`workloads/fa/llama3_prefill_ablation/`; `all` = `llama3_prefill/t1024`, whose run `stg_llama3_t1024_1` is the paper's), Llama-3-8B at
  1,024 tokens; `exp/eval6_fa/data.csv`.
- **The stage apps** (debugging): `fresh_<variant>` (DSv2-Lite, `workloads/fa/dsv2_l1_prefill/`) and
  `llama3_<variant>` (`workloads/fa/llama3_prefill/`), each prefill stage alone and chained; `fresh_t1024` is the
  DSv2-Lite block at 1,024 tokens the paper quotes.

`./run.sh run [variant ...]` launches a batch (by default the decode ladder) and parses it into `results/fa.csv`;
`./run.sh parse` re-parses. `fa_trace.py` reads a run's window from its scheduler events.

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

## The rungs

Each rung adds one mechanism on the same mapping and buffers; what is broadcast and reduced is the block's own
(decode: x, the q halves, m\* and o, the softmax statistics and the partial outputs; prefill: X and the gathered head
outputs, and no partials to reduce, since every head's softmax completes on one cluster).

| rung | broadcast | reduce | sync |
|---|---|---|---|
| `base` | copies one chip at a time, or pulls | the root pulls and folds | a chip barrier after every kernel |
| `bcast` | multicast | the root | barriers |
| `bcast_reduce` | multicast | ChainGather (the statistics through the softmax-statistics junction) | barriers |
| `all` | multicast | ChainGather | hardware, per edge |

**Barrier sync, the baseline** (`sync: barrier`, `barrier_mode: sw`): inside the window no task depends on another
chip's; a point-to-point software chip barrier on every chip's cluster-0 xDMA core follows every kernel and fences
every core of the chip, so no two kernels overlap. **Hardware sync** (`sync: edge`): every cross-chip dependency is a
BINGO hardware edge, so a task waits for exactly its producers, and the levers that overlap kernels become possible.

Weights stream plain.

## Measurement

Two anchors on chip 0x00 bracket the block (the prefetchers start after the first, so the whole weight and cache
stream counts); the window runs from the start anchor's end to the end anchor's. **Checks**, byte-exact: the decode
ladder's nine (the q halves, a far cluster's q, the root's statistics, a far chip's partial, y, and the appended K
and V^T tiles, the last five copied from HBM into the root's L3 first); the prefill's y and per-stage stashes.

## The prefill stage apps (DSv2-Lite and Llama-3-8B, for debugging)

The layer as a prefill over a fresh 256-token prompt (empty cache, causal), INT4 weights
(`workloads/fa/common/fa_prefill_mc.py` and `fa_stages_mc.py` generate it): head parallel (chip = head, cluster =
half of a chunk's tokens), two chunks of 128, the keys made in the window (stages 1-3) and swept by causal tiles.
`workloads/fa/dsv2_l1_prefill/` holds one optimal workload per stage (stage0 .. stage3, stage0123, stage4,
stage5, stage678) and the whole block (`full_n`; `t1024` at 1,024 tokens); `results/optimal_mappings.json` names each one's run and window.
`workloads/fa/llama3_prefill` is the Llama-3-8B variant (GQA attention, the driver in `workloads/llama3/common`).
Every run with its finding is in `results/prefill_stages.md`; the FA Prefill Tracker artifact has the figures.

## Files

- `stages.sh`: one run with its watchdog (`common/sim_watchdog.py`, profiles `fa_prefill` and `decode`).
- `run.sh`: a batch of runs through `common/launch.sh`, then `parse_fa.py` into `results/fa.csv` with
  `../ffn/parse_ffn.py`.
- `fa_trace.py`: a prefill run's window: MAC, engine shares, phases, the ring pushed per memory chip;
  `results/<run>/{summary.txt, series.json, window.png, chips.png}`.
- `real_kv_cache.py`: DeepSeek-V2-Lite's real latent cache into `cache/`.
- **Workloads:** `target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/fa/` (its README lists them and their
  drivers).
