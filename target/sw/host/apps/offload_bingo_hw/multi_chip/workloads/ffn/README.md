# Dense SwiGLU FFNs on 16 compute + 4 memory chiplets

One LLM feed-forward layer, `x[d] -> gate, up (d -> f) -> SwiGLU -> down -> y[d]`, INT4 weights streamed from the
memory chiplets' HBM, tensor-parallel over every cluster of `hemaia_16c4m_8MBL3_2cluster`: DeepSeek-V2-Lite's
layer 0 (d = 2,048, f = 10,944, the checkpoint's weights) and Llama-3-8B (d = 4,096, f = 14,336, synthetic weights of
the model's shapes). The paper ablates three mechanisms against their conventional counterparts on DeepSeek-V2-Lite's
FFN at 64 tokens on 16 chips (`dsv2_l0_prefill_ablation`): **broadcast** (one
D2D multicast of X from HBM against the memory chip writing one copy a chip), **reduce** (xDMA ChainGather folding in
transit against the root pulling every partial and folding it), **sync** (per-edge BINGO hardware dependencies against
software chip barriers after every kernel); and it scales the same FFN from 4 to 16 chips (`scaling`).

```
ffn/
  common/                the block's code (common/README.md maps it)
    ffn_prefill_mc/        the prefill block (its README.md explains the mapping and every parameter); run() is the
                           entry point of the whole-block workloads, and the layers run it inside theirs
    ffn_decode_mc/         one decode token on the prefill's mapping, every weight multiply on the GEMV (the dsv2
                           and llama3 decode FFNs)
    ffn_data.py            the golden and the data staging (checkpoint weights from the sweep cache, or a
                           registered synthetic model such as ../llama3/common/llama3_data.py's)
    ffn.mk                 the build every DeepSeek variant's Makefile includes
    (chiplet_mc.py, the multi-chiplet base every block builds on, and fabric.py are util/sim/multichip's, driver.py
     util/sim/common's, model_data.py util/sim/llm's, the GEMV helpers util/sim/gemm/gemv_utils.py's: see util/sim/multichip/README.md)
  dsv2_l0_prefill/       DeepSeek-V2-Lite over a fresh 256-token prompt: stage0, stage123, stage45, full
  dsv2_l0_prefill_ablation/  the paper's ablation at 64 tokens on 16 chips: base, bcast, bcast_reduce, all
  llama3_prefill/        Llama-3-8B over a fresh 256-token prompt at the model's size: stage0, stage123, stage45,
                         full; t64, the paper's one-pass Llama run; built with ../llama3/common/llama3.mk
  scaling/               the strong-scaling sweep at 64 tokens: base and all on 4, 8, 12 and 16 chips
```

Every variant directory holds exactly three files: `Makefile` (names its path and includes `ffn.mk` or
`llama3.mk`), `main_bingo.py` (calls the driver), `params.hjson` (the switches). Everything else is generated.

## The prefill workloads

Both models use the one mapping of `common/ffn_prefill_mc/`: the tokens in passes of `rows`; X of a pass in every
cluster's L1; each weight chunk, as it streams in, used for all the pass's rows in one GEMM; the down projection split
by output columns over a chip's two clusters and reduced slice by slice by ChainGathers into the root, under the next
slices' GEMMs. **X comes from HBM**: every pass's X sits in the HBM of the root's memory chip (0x20), and the root's
host queues that chip's push engine to multicast it into every chip's L3, a 64 KiB tile and its flag at a time, ahead
of the root's weight-ring posts in that engine's queue. Each cluster posts its weight ring's runs to its memory chip's
push engine from its DM core (`WeightRings.push_from`), so a ring's pushes are graph tasks. Rows a pass: an L1 holds a pass's X (256 KiB) beside the weight slabs and the A16/A8 buffers, so 128 rows for
DeepSeek (hidden 2,048) and 64 for Llama (hidden 4,096).

Each model's block in stage apps (`<model>_prefill/stage*/main_bingo.py`, each a standalone program of its stage on
`chiplet_mc` and `ffn_data`, the same tasks as the whole block's), each app's inputs in place before its window (in L1,
copied from an HBM image), so a window is the stage alone:

| app | what it runs |
|---|---|
| `stage0` | X of every pass from HBM into every chip's L3, the first pass's X into every cluster's L1 |
| `stage123` | one pass (tokens = rows), X in L1 before the window: gate\|up over the weight stream, dequantise and SwiGLU, the reshape D -> A, the quantise and the A8 exchange between a chip's clusters |
| `stage45` | one pass, the chip's A8 in L1 before the window: the down slices over the weight stream, the ChainGathers into the root, the stores |
| `full` | the whole 256-token block |

`dsv2_l0_prefill`: two passes of 128 rows; the second copies each chunk from the L3 ring slot it landed in (the ring
holds all of a cluster's chunks), so the weights stream from HBM once. `llama3_prefill`: four passes of 64 rows; every
pass streams the weights from the memory chips again (`restream`: the model's 84 MiB of INT4 weights do not stay in the
rings); 8 output slots, 4 slices a ChainGather (`chain_group`). The stage apps take their full app's params (stage123
and stage45 with tokens = rows). Each `params.hjson` lists every option it uses, and `common/ffn_prefill_mc/README.md`
explains each one (`params.py` where it reads it). The runs and the stage-by-stage traces are on the FFN Prefill Tracker artifact.

## The ablation (`dsv2_l0_prefill_ablation`)

Four rungs of DeepSeek-V2-Lite's FFN at 64 tokens (one pass of 64 rows, X from HBM, the GEMMs queued: below) on 16
chips, each adding one mechanism, all on the same chunks, slices and buffers (`common/ffn_prefill_mc/params.py`, the
ablation's arms, explains the switches). `base` and `all` carry the params of the scaling sweep's 16-chip points (`scaling/base_n16`,
`scaling/all_n16`), whose runs (`stg_sweep64_base_n16`, `stg_poc64_g8`) are the paper's R0 and R3:

| rung | workload | `x_cast` | `reduce` | `sync` | |
|---|---|---|---|---|---|
| `base` | `dsv2_l0_prefill_ablation/base` | unicast | root | barrier | the conventional baseline |
| `bcast` | `dsv2_l0_prefill_ablation/bcast` | multicast | root | barrier | + broadcast |
| `bcast_reduce` | `dsv2_l0_prefill_ablation/bcast_reduce` | multicast | chain | barrier | + in-transit reduce |
| `all` | `dsv2_l0_prefill_ablation/all` | multicast | chain | edge | + per-edge sync |

The barrier is the vanilla point-to-point software chip barrier (`barrier_mode: sw`). Kernel by kernel, a barrier
follows every kernel (gate|up with its SwiGLU, the A8, the down slices, the reduce), fencing every core of the chip, so
those rungs have no overlap inside the pass. `bcast_reduce` runs one ChainGather a cluster, as `all` does (`chain_group`
also works kernel by kernel), so the step to `all` is the synchronisation alone. Stage: `./stages.sh abl64_<rung>`; runs and
results: `target/sim/automation/sweep/llm_blocks/ffn/README.md`.

## Asynchronous dispatch

The driver makes every GEMM one blocking `gemm_full` task and every SIMD and xDMA task whole; the compiler orders each
core's program (`mini_compiler/passes/bingo_dfg_schedule.py`). It splits the GEMMs, the SIMD's staged kernels and the
xDMA's transfers into configure / start / wait parts where its schedule gains (params `async_exec: false` keeps every
task whole: `llama3_prefill/stage123` and `stage45` beside their `_async` twins): the core programs the
next dispatch while the accelerator runs, a task's consumers and the buffers it reads follow its wait, and each
cluster's GEMM core reads the array shape's parameters from an L1 copy. A dispatch that repeats its core's previous
configuration writes only its addresses (the compiler's CSR reuse).

## The strong-scaling sweep at 64 tokens (`scaling`)

DeepSeek-V2-Lite's dense FFN, 64 tokens in one pass, on 4, 8, 12 and 16 chips (`chip_rows`: whole rows, each with its memory
chip; `all_n16` takes every chip), `base` and `all`. `all` runs one ChainGather a cluster (`chain_group` = `out_slots` = every
slice); at 4 chips that output does not fit L1, so it runs 2 slices a ChainGather through 4 output slots, with `dn_cols` 64,
`blocks` 2 and an 88 KiB slab for the longer down chunks. The root stays about 60 columns lighter at every size (`root_cols`).
Stage: `./stages.sh sweep64_<arm>_n<N>` (tags `stg_sweep64_<arm>_n<N>`; `all_n16` ran as `stg_poc64_g8`).

## `llama3_prefill/t64`

Llama-3-8B's FFN, one pass of 64 tokens, every mechanism, X from HBM, the GEMMs queued: the paper's Llama number. Stage:
`./stages.sh llama3_t64`.

## Generating, building, running

The directories hold sources only. `make` in a variant (or the runner) generates the data header
(`ffn_data.h`, Llama `llama3_data.h`), `offload_bingo_hw.h`, `final_dfg.csv`, `static_l1_layout.*` and
`build/hbm/` (the HBM images of the weights and X, 34-90 MB an app); `make clean` removes the data, the headers and the
HBM images. The DeepSeek weights come from `target/sim/automation/sweep/llm_blocks/ffn/cache/` (`fetch_weights.py`).

- Generator alone (no simulation; the DFG, the hazard check and the manager-model hang check run here):
  `target/sim/automation/sweep/llm_blocks/ffn/tools/gen_one.sh <variant under ffn/> <log>`
- Simulate a prefill app with the hang guard: `target/sim/automation/sweep/llm_blocks/ffn/stages.sh <model>_<app>`
  (`dsv2_<app>` = `dsv2_l0_prefill/<app>`, `llama3_<app>` = `llama3_prefill/<app>`, `abl64_<rung>` =
  `dsv2_l0_prefill_ablation/<rung>`, `sweep64_<arm>_n<N>` = `scaling/<arm>_n<N>`, `llama3_t64`)
- Runs, parsers and plots of the sweep: `target/sim/automation/sweep/llm_blocks/ffn/README.md`
