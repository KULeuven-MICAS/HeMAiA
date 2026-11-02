# Dense SwiGLU FFNs on 16 compute + 4 memory chiplets

One LLM feed-forward layer, `x[d] -> gate, up (d -> f) -> SwiGLU -> down -> y[d]`, INT4 weights streamed from the
memory chiplets' HBM, tensor-parallel over every cluster of `hemaia_16c4m_8MBL3_2cluster`: DeepSeek-V2-Lite's
layer 0 (d = 2,048, f = 10,944, the checkpoint's weights) and Llama-3-8B (d = 4,096, f = 14,336, synthetic weights of
the model's shapes). The paper ablates three mechanisms against their conventional counterparts on the DeepSeek
prefill (`dsv2_l0_prefill_ablation`): **broadcast** (one D2D multicast of X from HBM against the memory chip writing
one copy a chip), **reduce** (xDMA ChainGather folding in transit against the root pulling every partial and folding
it), **sync** (per-edge BINGO hardware dependencies against software chip barriers, kernel by kernel and fused).

```
ffn/
  common/                shared code
    ffn_prefill_mc.py      the prefill block (its docstring explains the mapping; every parameter is explained
                           where it is read); llama3/common/llama3_prefill_mc.py runs it inside the Llama layer
    ffn_stages_mc.py       the prefill split into stage apps (params stage: 0, 123, 45, full), the method of
                           fa/common/fa_stages_mc.py and moe/common/moe_stages_mc.py
    chiplet_mc.py          the multi-chiplet base the driver uses: platform, weight rings, anchors,
                           barriers, fences. SHARED with ../moe and ../fa, which import it from here
    ffn_data.py            the golden and the data staging (checkpoint weights from the sweep cache, or a
                           registered synthetic model such as ../llama3/common/llama3_data.py's)
    ffn.mk                 the build every DeepSeek variant's Makefile includes
    (fabric.py and driver.py are ../../common's: the mesh, the topology and node helpers, and the command
     line and `run()` body of every driver in ffn, moe and fa)
  dsv2_l0_prefill/       DeepSeek-V2-Lite over a fresh 256-token prompt: stage0, stage123, stage45, full
  dsv2_l0_prefill_ablation/  the ablation ladder on dsv2_l0_prefill/full: base, bcast, bcast_reduce, fused, all
  llama3_prefill/        Llama-3-8B over a fresh 256-token prompt at the model's size: stage0, stage123, stage45,
                         full; built with ../llama3/common/llama3.mk
```

Every variant directory holds exactly three files: `Makefile` (names its path and includes `ffn.mk` or
`llama3.mk`), `main_bingo.py` (calls the driver), `params.hjson` (the switches). Everything else is generated.

## The prefill workloads

Both models use the one mapping of `common/ffn_prefill_mc.py`: the tokens in passes of `rows`; X of a pass in every
cluster's L1; each weight chunk, as it streams in, used for all the pass's rows in one GEMM; the down projection split
by output columns over a chip's two clusters and reduced slice by slice by ChainGathers into the root, under the next
slices' GEMMs. **X comes from HBM**: every pass's X sits in the HBM of the root's memory chip (0x20), and the root's
host queues that chip's push engine to multicast it into every chip's L3, a 64 KiB tile and its flag at a time, before
the root's weight prefetcher starts (`x_hbm`, the default; `x_device` is the Llama layer's, whose X is made on the
device). Rows a pass: an L1 holds a pass's X (256 KiB) beside the weight slabs and the A16/A8 buffers, so 128 rows for
DeepSeek (hidden 2,048) and 64 for Llama (hidden 4,096).

Each model's block in stage apps (`common/ffn_stages_mc.py`), each app's inputs in place before its window (in L1,
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
and stage45 with tokens = rows). Each `params.hjson` lists every option it uses, and `common/ffn_prefill_mc.py` explains
each one where it reads it. The runs and the stage-by-stage traces are on the FFN Prefill Tracker artifact.

## The ablation ladder (`dsv2_l0_prefill_ablation`)

Five rungs of `dsv2_l0_prefill/full` (256 tokens, two passes of 128 rows, X from HBM), each adding one mechanism, all
on the same chunks, slices and buffers (`common/ffn_prefill_mc.py`, THE ABLATION, explains the switches):

| rung | `x_cast` | `reduce` | `sync` | |
|---|---|---|---|---|
| `base` | unicast | root | barrier, `barrier_at: kernel` | the conventional baseline |
| `bcast` | multicast | root | barrier, kernel | + broadcast |
| `bcast_reduce` | multicast | chain | barrier, kernel | + in-transit reduce |
| `fused` | multicast | chain | barrier, `barrier_at: cross` | + fused kernels: a barrier only where data crosses chips |
| `all` | multicast | chain | edge | + per-edge sync (byte for byte `dsv2_l0_prefill/full`) |

The barrier is the vanilla point-to-point software chip barrier (`barrier_mode: sw`). Kernel by kernel, a barrier
follows every kernel (gate|up with its SwiGLU, the A8, the down slices, the reduce), fencing every core of the chip;
fused, one precedes each slice's gathers and one sits between a pass's last gathers and the next pass's A8 copies,
and only the tasks it names wait, so the reduce runs under the next GEMMs. Runs and results:
`target/sim/automation/sweep/llm_blocks/ffn/README.md`.

## Generating, building, running

The directories hold sources only. `make` in a variant (or the runner) generates the data header
(`ffn_data.h`, Llama `llama3_data.h`), `offload_bingo_hw.h`, `final_dfg.csv`, `static_l1_layout.*` and
`build/hbm/` (the HBM images of the weights and X, 34-90 MB an app); `make clean` removes the data, the headers and the
HBM images. The DeepSeek weights come from `target/sim/automation/sweep/llm_blocks/ffn/cache/` (`fetch_weights.py`).

- Generator alone (no simulation; the DFG, the hazard check and the manager-model hang check run here):
  `target/sim/automation/sweep/llm_blocks/ffn/tools/gen_one.sh <variant under ffn/> <log>`
- Simulate a prefill app with the hang guard: `target/sim/automation/sweep/llm_blocks/ffn/stages.sh <model>_<app>`
  (`dsv2_<app>` = `dsv2_l0_prefill/<app>`, `llama3_<app>` = `llama3_prefill/<app>`, `abl_<rung>` =
  `dsv2_l0_prefill_ablation/<rung>`)
- Runs, parsers and plots of the sweep: `target/sim/automation/sweep/llm_blocks/ffn/README.md`
