# DeepSeek-V2-Lite's layer-1 attention (MLA, absorbed) on 16 compute + 4 memory chiplets

One attention layer, 16 heads over a latent cache of 8,192 tokens (576-wide K rows, 512-wide V), INT8, the cache
streamed from the memory chiplets' HBM, on `hemaia_16c4m_8MBL3_2cluster`. The paper ablates three mechanisms
against their conventional counterparts: **bcast** (one D2D multicast against pulls or per-chip pushes),
**reduce** (xDMA ChainGather folding in transit against the root pulling every partial), **sync** (per-edge BINGO
hardware dependencies against chip barriers). Each directory below is one point of that ablation; a variant's
switches are its `params.hjson`.

```
fa/
  common/                shared code
    fa_mc.py               driver of the decode workloads (one token, split by key)
    fa_prefill_mc.py       driver and generator of the prefill workloads (the mapping below); its docstring
                           lists the parameters
    fa_prefill_data.py     the prefill's data, layouts and golden (the device arithmetic)
    fa_data.py             the cache, the query, the scales and the golden (decode)
    fa.mk                  the build every variant's Makefile includes
    (chiplet_mc.py is ../ffn/common's: platform, weight rings, anchors, barriers, fences;
     fabric.py and driver.py are ../../common's)
  dsv2_l1/               one decode token, 8 variants: base bcast reduce sync bcast_reduce bcast_sync
                         reduce_sync all (the three mechanisms in every combination)
  dsv2_l1_prefill/       256-token prefill, the ladder base -> bcast -> bcast_reduce -> all, each rung
                         adding one mechanism (base..bcast_reduce 4 chunks of 64, all 2 chunks of 128)
  dsv2_l1_prefill_tiny/  reduced-cache copies (1,024 keys): all (the twin of dsv2_l1_prefill/all), v19 (all
                         without per-chiplet images and gid lists), v21 (v19 + per-chiplet images)
```

Numbered variants (`v<N>`) are local iterations and are not tracked; the named variants are. Every variant
directory holds exactly three files: `Makefile` (names its path and includes `fa.mk`),
`main_bingo.py` (calls the driver), `params.hjson` (the switches). Everything else is generated.

## The decode workloads (`dsv2_l1/`)

One query token at position 8,191 split by key over the 32 clusters (`common/fa_mc.py`): the 16 heads fill the
softmax's 16 lanes, every cluster sweeps its shard of 4 key tiles, the clusters' row maxima meet on the root and
are broadcast (m\*), the partials are folded into the root. The checks are byte-exact; the design is in
`target/sim/automation/sweep/llm_blocks/fa/README.md`.

## The prefill mapping (`dsv2_l1_prefill/`)

DeepSeek-V2-Lite at its real size: a **256-token prefill** (the MoE's length) over the 8,192-token cache, as
4 chunks of 64 tokens, **head parallel**.

| | |
|---|---|
| chip k | head k (16 chips, 16 heads); the heads share one latent cache, so every chip sweeps all of it |
| cluster c | half of each chunk's tokens: 32 query rows of one head, as two 16-row query tiles (the SIMD beat) |
| chunk | x (64 x 2048) in, q = x W_q and the W_UK absorb, the sweep over the cache, W_UV and W_O, the partial y (32 x 2048 FP16 a cluster) reduced into C00 |
| sweep | 32 K/V tiles of 256 keys (K 144 KiB, V 128 KiB); one resident tile serves both query tiles; online softmax per cluster, no exchange |
| tasks | QK 2,304 passes, PV 2,048 passes: longer than the next task's software set-up, so `gemm_async` hides it |
| cache | in each memory chip's HBM, through the weight ring like a weight; bcast: one multicast into the row |
| weights | the head's W_q, W_UK, W_UV, W_O (0.79 MB) through the same ring every chunk |

| variant | bcast | reduce | sync |
|---|---|---|---|
| `base` | pull | root (iDMA load-all, SIMD fold) | barrier (vanilla chip barrier, full fence) |
| `bcast` | bcast_l1 | root | barrier |
| `bcast_reduce` | bcast_l1 | chain | barrier |
| `all` | bcast_l1 | chain | edge |

The other four combinations (`reduce_sync`, `bcast_sync`, `reduce`, `sync`) are not built.

`all` takes the same mapping with 2 chunks of 128 tokens (four query tiles a cluster): it halves the weight and
cache stream, which binds 64-token chunks at the real size. Its partial y goes through a half-size symmetric
buffer, one W_O block and its ChainGather at a time (`y_halves`): only so do four query tiles fit the 990,000 B
L1. `y_halves` needs the in-transit reduce and per-edge sync, so the other rungs keep 64-token chunks.

**Task tables.** At 256 tokens the graph has 113k (all) to 150k (4 x 64) descriptors, and one host image holding
every chiplet's tables does not fit the 8 MiB L3 (11.5 MiB and ~29 MiB). Every full-size variant therefore
builds one host image per chiplet (`per_chip_image`: target/sw/host/apps/per_chip_image.py, with the host.ld
every host image uses), each holding the shared code and data and that chiplet's own tables; the build checks
that every shared symbol sits at one address in all of them. Each cluster's L1 holds its own tasks' global ids
(`gid_lists`), not a 2-byte entry for every task of the graph.

### Why this mapping

- **Task size.** A tile of 256 keys makes the QK and PV tasks long enough that, with `gemm_async`
  (`gemm_fa_qk_start` / `gemm_fa_pv_start` and `gemm_wait`), the software set-up of the next GEMM runs under the
  previous one.
- **Chunk size.** With 32-token chunks a cluster has one query tile, so every K/V tile the iDMA fetches serves
  16 MACs per byte and the iDMA binds. At 64 tokens (two query tiles) one resident tile serves both.
- **Stream.** Every chunk streams its head's weights and the whole cache again, so with small chunks the real
  size is bound by that stream rather than by the arrays.

`sweep/llm_blocks/fa/estimate_prefill.py` estimates the ladder; the runs and their findings are in
`target/sim/automation/sweep/llm_blocks/fa/README.md`.

## Generating, building, running

The directories hold sources only. `make` in a variant (or the runner) generates `fa_data.h`,
`offload_bingo_hw.h`, `final_dfg.csv`, `static_l1_layout.*` and `build/hbm/` (the cache images); `make clean`
removes them. The cache is `target/sim/automation/sweep/llm_blocks/fa/cache/` (`real_kv_cache.py`), the query's
weights `target/sim/automation/sweep/common/weights/cache`.

- Decode variants: `target/sim/automation/sweep/llm_blocks/fa/run.sh run dsv2_l1/all ...`
- Prefill variants: `target/sim/automation/sweep/llm_blocks/fa/stages.sh poc_<variant>` (tiny) or
  `full_<variant>` (real size), each with its watchdog; `fa_trace.py` reads a run's window.
