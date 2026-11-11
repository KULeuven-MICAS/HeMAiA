# ffn/common: the dense SwiGLU FFN block

The code of one LLM feed-forward layer, `x[d] -> gate, up (d -> f) -> SwiGLU -> down -> y[d]`, with INT4 weights
streamed from the memory chiplets' HBM and tensor-parallel over every cluster of `hemaia_16c4m_8MBL3_2cluster`. The
workload directories under `../` (and the decode FFNs under `../../dsv2` and `../../llama3`) hold only a `Makefile`, a
`main_bingo.py` and a `params.hjson`; everything they generate comes from here.

```
ffn/common/
  ffn_prefill_mc.py   PBuild: the prefill block, T tokens in passes of `rows`. Its docstring explains the mapping;
                      every parameter is explained where __init__ reads it. run() is the entry point of the
                      whole-block workloads under ../ (the stage apps' main_bingo.py are standalone)
  ffn_decode_mc.py    DBuild(PBuild): one token on the prefill's mapping, one 16-row tile (GEMM or GEMV forms).
                      run() is the entry point of the decode FFN workloads
  ffn_data.py         FFNData: the weights packed for the device, the static scales and the golden of the device's
                      arithmetic; load_weights and the SYNTHETIC registry of models without a checkpoint
  ffn.mk              the build a DeepSeek FFN variant's Makefile includes (Llama variants: ../../llama3/common/llama3.mk)
```

What the FFN shares with the other blocks lives in `util/sim/multichip/` (its README maps it), the drivers' common
body in `util/sim/common/driver.py`, its data helpers and
golden in `util/sim/llm/model_data.py` (with `util/sim/common/` and `util/sim/simd/`), the GEMV helpers in
`util/sim/gemm/gemv_utils.py`:

```
util/sim/multichip/
  fabric.py           FabricBase: the mesh, remote addresses, the chips' fold order
  chiplet_mc.py       ChipletBuild(FabricBase): the platform, weight rings, stamps, barriers, broadcast,
                      reduce, checks and finish() that every block builds on (ffn, fa, moe, the llama3 and dsv2 layers)
```

## The class stack

```
fabric.FabricBase                       util/sim/multichip/fabric.py
  chiplet_mc.ChipletBuild               util/sim/multichip/chiplet_mc.py
    ffn_prefill_mc.PBuild               the prefill block
      ffn_decode_mc.DBuild              one token
    ../<model>_prefill/stage*/          one stage alone (Stage0, Stage123, Stage45 in each main_bingo.py)
```

A build runs `__init__` (params, data, golden, staging, rings, buffers), then `build()` (the task graph), then
`finish(app_name)` (stamps, checks, emission). `PBuild` is grouped in that order:

| section of `ffn_prefill_mc.py` | what it makes |
|---|---|
| `__init__` | the params, `FFNData` and its golden, the column split over the clusters (`cols`, `kcols`), the down slices |
| the plan | the gate\|up chunks of a cluster (`_gu_chunks`, `_gu_cols`) |
| data | the HBM images of the weights and of X, the scales, every cluster's L1 buffers (`_stage`, `_buffers`) |
| helpers | per-core program order, one GEMM (blocking or queued), one ChainGather, one weight-chunk load |
| X | X from HBM: the multicast into every chip's L3 and each cluster's copy into L1 |
| gate\|up and the A8 | the GEMMs over the weight stream, dequantise and SwiGLU, the D -> A reshape, the quantise and the A8 exchange |
| the prefill | `build()`: the passes, the down slices, the ChainGathers into the root and the stores; `_passes_kernel` for the ablation's kernel-by-kernel rung |
| checks | the window, the exchange buffers, the golden checks (`_checks`), the hazard check |

## Who uses it

| user | through |
|---|---|
| `../<model>_prefill/full`, `../llama3_prefill/t64`, `../dsv2_l0_prefill_ablation/`, `../scaling/` | `ffn_prefill_mc.run` |
| `../<model>_prefill/stage*/` | their own standalone `main_bingo.py` (`ChipletBuild` and `FFNData`) |
| the Llama prefill layer (`../../llama3/full_layer_prefill/common/llama3_layer_mc.py`) and DeepSeek layer 0 (`../../dsv2/common/dsv2_layer_prefill_mc.py`) | `PBuild` inside the layer, `FFNData` on the layer's rows |
| the decode layers (`../../dsv2/common/dsv2_decode_mc.py`, `../../llama3/common/llama3_decode_layer_mc.py`) | `DBuild` |
| `../../llama3/common/llama3_data.py` | registers its synthetic FFNs in `ffn_data.SYNTHETIC` (`register_ffn`, `register_ffn_shape`) |

The weights: DeepSeek-V2-Lite's layer 0 from the sweep cache (`target/sim/automation/sweep/llm_blocks/ffn/cache/`,
filled by `fetch_weights.py`, params `model`, `weights`); Llama-3 shapes are synthetic (params `model` names a
registered builder).

## Parameters

Each is explained where it is read:

- `util/sim/multichip/chiplet_mc.py`, `ChipletBuild.__init__`: the platform and the emission (`tokens`, `wbits`, `bcast`,
  `reduce`, `sync`, `barrier_mode`, `chip_rows`, `weight_ring`, `w_chunk_bytes`, `w_buffers`, `gid_lists`,
  `list_phases`, `per_chip_image`, `static_l1`, `l1_capacity`, `async_exec`, ...).
- `ffn_prefill_mc.py`, `PBuild.__init__`: the mapping and the ablation switches (`rows`, `blocks`, `dn_cols`, `x_cast`,
  `chain_group`, `out_slots`, `root_cols`, `restream`, ...).
- `ffn_decode_mc.py`: the decode's own (`x_row`, `dn_gemv`, `gu_gemv`, ...).

## Generating and building

`make` in a variant runs its `main_bingo.py` (the generator: the data header, `offload_bingo_hw.h`, `final_dfg.csv`,
`static_l1_layout.*` and the HBM image `build/hbm/`) and compiles the host image. The generator reruns whenever one of
its sources is newer than its outputs: `mini_compiler`, the kernel-args headers, this directory, `util/sim/multichip/`,
`util/sim/gemm/gemv_utils.py` and the weight manifest (`ffn.mk`, `BINGO_GEN_DEPS`). The fa, moe, llama3 and dsv2 builds also list `ffn/common/*.py`, so an
edit here makes their workloads regenerate too. Never run `make` in a workload whose simulation is running: the
generator rewrites `build/hbm/` under it.

- The generator alone, outputs in another directory:
  `target/sim/automation/sweep/common/gen_check.sh ffn/<variant> <out dir> <log>`
- Simulate with the hang guard: `target/sim/automation/sweep/llm_blocks/ffn/stages.sh <tag>` (the tags are listed in
  `../README.md`)
- The generator's log reports the weight rings' overwrite races (`ring races`: the runs the prefetcher's byte guard
  holds back)

The workloads, the ladders and the sweeps are described in `../README.md`; the runs and results in
`target/sim/automation/sweep/llm_blocks/ffn/README.md`.
