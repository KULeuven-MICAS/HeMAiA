# FlashAttention on 16 compute + 4 memory chiplets: Llama-3-8B and DeepSeek-V2-Lite

The attention of one layer on `hemaia_16c4m_8MBL3_2cluster`: Llama-3-8B's grouped-query attention as one decode
token over an 8,192-position KV cache in HBM (the whole block, from x to y and the new key and value written back) and
as a prefill over a fresh prompt, and DeepSeek-V2-Lite's layer-1 MLA attention as a prefill over a fresh prompt. The
paper ablates three mechanisms against their conventional counterparts: **bcast** (one D2D multicast against pulls
or per-chip pushes), **reduce** (xDMA ChainGather folding in transit against the root pulling every partial),
**sync** (per-edge BINGO hardware dependencies against chip barriers). A variant's switches are its `params.hjson`.

```
fa/
  common/                shared code
    fa_mc.py               the key-split decode (one token, the cache split by key over the clusters): the base
                           of fa_llama_mc.py and fa_decode_ks_mc.py
    fa_llama_mc.py         its Llama form (the 32 heads as two halves of the softmax's lanes), the base of
                           ../llama3/common/llama3_decode_ks_mc.py
    fa_llama_data.py       its cache, query, scales and golden (synthetic INT4 weights, llama3_data's draw), and
                           the softmax-statistics junction's arithmetic the decode ladder's golden uses
    fa_decode_mc.py,       DSv2-Lite's MLA decode mapped as the prefill (head parallel) or split by key, used by
    fa_decode_ks_mc.py     ../dsv2/layer1/decode/
    fa_prefill_mc.py       driver and generator of the prefill workloads (the mapping below); its docstring
                           lists the parameters
    fa_stages_mc.py        the prefill split into stages (stage0 .. stage678) and chained again (full)
    fa_prefill_data.py     the prefill's data, layouts and golden (the device arithmetic)
    fa_data.py             fa_mc.py's cache, query, scales and golden (DSv2-Lite)
    fa.mk                  the build every variant's Makefile includes
    (chiplet_mc.py: platform, weight rings, anchors, barriers, fences; fabric.py: both util/sim/multichip's;
     driver.py: util/sim/common's; model_data.py, the golden's modules and the BF16 reader: util/sim/llm's)
  llama3_decode_full/    Llama-3-8B's decode, the whole attention block, one token over 8,192 positions: the paper's
                         ladder base -> bcast -> bcast_reduce -> all (each rung adds one mechanism); its
                         driver is ../llama3/common/llama3_decode_ks_mc.py
  llama3_prefill/        Llama-3-8B's grouped-query attention over a fresh prompt: at 1,024 tokens the whole block
                         (t1024, the paper's all-mechanism rung); at 256 tokens one optimal workload per stage
                         (stage1234, stage5, stage678) and the stages chained (full), kept for debugging; its driver
                         is ../llama3/common/llama3_stages_mc.py
  llama3_prefill_ablation/ the Llama prefill's ladder at 1,024 tokens: base and bcast (t1024's mapping without its
                         overlap levers, a chip barrier after every kernel) and all (t1024's switches); driver
                         ../llama3/common/llama3_attn_mc.py
  dsv2_l1_prefill/       DeepSeek-V2-Lite's prefill over a fresh prompt: at 1,024 tokens the whole block (t1024); at
                         256 tokens one optimal workload per stage (stage0, stage1, stage2, stage3, stage0123,
                         stage01234, stage4, stage5, stage678) and the stages chained (full_n), kept for debugging
```

Numbered variants (`v<N>`) are local iterations and are not tracked; the named variants are. Every variant
directory holds exactly three files: `Makefile` (names its path and includes `fa.mk`),
`main_bingo.py` (calls the driver), `params.hjson` (the switches). Everything else is generated.

## The decode ladder (`llama3_decode_full/`)

One decode token of Llama-3-8B's attention (32 query and 8 key/value heads of 128, synthetic INT4 weights) over an
8,192-position INT8 cache, the whole block: x from HBM, the q / k / v GEMVs, the cache split by key over the 32
clusters with the one-pass online-softmax merge (the softmax statistics merged by the softmax-statistics junction,
m\* back to every cluster, each partial re-based and summed), W_O split by outputs, and y and the token's key and
value written back into HBM. `llama3/common/llama3_decode_ks_mc.py` (THE ABLATION in its params) builds every rung
on the same mapping and buffers. Every rung also orders its memory traffic the same way (`cache_order after_q`:
each cache copy after q is in every cluster; `wo_push before_chains`: W_O pushed during the sweep), since a running
push from a memory chip holds up other traffic through it, and copies y and the appended tiles into the root's L3
before the checks (`hbm_checks stash`).

## The prefill ladder (`llama3_prefill_ablation/`)

Llama-3-8B's prefill at 1,024 tokens, head parallel, so there are no partials to reduce and the reduction rung is
the broadcast rung. `base` and `bcast` run `t1024`'s mapping with `sync: barrier` (a chip barrier after every
kernel, so none of the levers that overlap two kernels) and, for `base`, `x_cast: unicast` (x copied into one chip
at a time); `all` is the rung with all three, `llama3_prefill/t1024`'s switches (the paper's run of it is
`stg_llama3_t1024_1`, from `llama3_prefill/t1024`).

## The prefill (`dsv2_l1_prefill/`)

DeepSeek-V2-Lite's layer-1 MLA attention as a **prefill over a fresh prompt** (empty cache, causal) of 256 or
1,024 tokens,
INT4 weights streamed from the memory chiplets' HBM through the weight ring, **head parallel**: chip k = head k,
cluster c = half of each 128-token chunk's tokens (four 16-row query tiles). The block in stages:

| stage | what it does |
|---|---|
| 0 | x from HBM into every chip's L3 (one multicast by a memory chiplet's push engine) |
| 1 | the latent keys by columns: each chip its slice of W_KVA over every tile of x |
| 2 | each 16-token piece of keys and values gathered on its owner, then on the root C00 |
| 3 | the root multicasts each chunk's keys to every chip |
| 4 | the queries: x W_q, the W_UK absorb, RoPE, quantised to INT8 |
| 5 | the causal sweep: QK, online softmax, PV over the chunk's key tiles |
| 6-8 | W_UV, W_O, and the 16 heads' partials of y folded into C00 by reduce chains, a W_O block at a time |

`stageN` runs one stage alone (its inputs preloaded before the window), `stage0123` and `stage01234` chain stages
0-3 and 0-4, `full_n` the whole block at 256 tokens. `t1024` is the whole block at 1,024 tokens (eight chunks): each
chunk's keys made just ahead of its queries, each W_O block's reduce chain ending on a collector chip in turn instead
of C00 (`y_roots`), and the order changes its `params.hjson` lists, each proven first on the block cut to two chunks. Each workload is the best measured mapping of its stage (`params.hjson` says how it got there);
`target/sim/automation/sweep/llm_blocks/fa/results/optimal_mappings.json` names each one's run and window.

**Task tables.** At 256 tokens the graph has over 100k descriptors, and one host image holding every chiplet's
tables does not fit the 8 MiB L3. Every full-size variant therefore builds one host image per chiplet
(`per_chip_image`: target/sw/host/apps/per_chip_image.py, with the host.ld every host image uses), each holding the
shared code and data and that chiplet's own tables; the build checks that every shared symbol sits at one address
in all of them. Each cluster's L1 holds its own tasks' global ids (`gid_lists`), not a 2-byte entry for every task
of the graph.

## The Llama-3-8B attention (`llama3_prefill/`)

Llama-3-8B's grouped-query attention (32 query and 8 key/value heads of 128, hidden 4096, synthetic INT4 weights
of the model's shapes) over the same fresh prompt: a query head a cluster, a key/value group on a chip pair,
passes of 64 tokens (4 at 256 tokens, 16 in `t1024`), the same causal `fa_sweep`. Its driver is
`../llama3/common/llama3_stages_mc.py` (on `llama3_attn_mc.py`, shared with the whole Llama layer).

The runs and their findings are in `target/sim/automation/sweep/llm_blocks/fa/results/prefill_stages.md`.

## Generating, building, running

The directories hold sources only. `make` in a variant (or the runner) generates `fa_data.h`,
`offload_bingo_hw.h`, `final_dfg.csv`, `static_l1_layout.*` and `build/hbm/` (the cache images); `make clean`
removes them. The caches are in `target/sim/automation/sweep/llm_blocks/fa/cache/` (DSv2: `real_kv_cache.py`, its
query's weights `target/sim/automation/sweep/common/weights/cache`; Llama: made by `fa_llama_data.py` on first use).

- Every variant: `target/sim/automation/sweep/llm_blocks/fa/stages.sh <stage>`, with its watchdog: `llama3dec_<rung>`
  (the decode ladder, e.g. `llama3dec_all`), `llama3abl_<rung>` (the prefill ladder), `llama3_<variant>` (Llama
  prefill), `fresh_<variant>` (DSv2 prefill); `fa_trace.py` reads a run's window. `run.sh run [variant ...]` launches
  a batch and parses it into `results/fa.csv` (by default the decode ladder).
