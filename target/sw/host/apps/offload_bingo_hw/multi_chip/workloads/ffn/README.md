# DeepSeek-V2-Lite's dense FFN (layer 0) on 16 compute + 4 memory chiplets

One LLM feed-forward layer, `x[2048] -> gate, up (2048 -> 10,944) -> SwiGLU -> down -> y[2048]`, INT4 weights
streamed from the memory chiplets' HBM, tensor-parallel over every cluster of `hemaia_16c4m_8MBL3_2cluster`.
The paper ablates three mechanisms against their conventional counterparts: **bcast** (one D2D multicast
against every cluster pulling the root's copy), **reduce** (xDMA ChainGather folding in transit against the
root pulling every partial), **sync** (per-edge BINGO hardware dependencies against chip barriers). Each
directory below is one point of that ablation; a variant's switches are its `params.hjson`.

```
ffn/
  common/                shared code
    ffn_mc.py              driver of the T = 1 (decode) workloads
    ffn_prefill_mc.py      driver of the 128-token prefill workloads (its docstring explains the mapping
                           and every parameter)
    chiplet_mc.py          the multi-chiplet base both drivers use: platform, weight rings, anchors,
                           barriers, fences. SHARED with ../moe and ../fa, which import it from here
    ffn_data.py            the golden and the data staging (checkpoint weights from the sweep cache)
    ffn.mk                 the build every variant's Makefile includes
  dsv2_l0/               T = 1 decode, 8 variants: base bcast reduce sync bcast_reduce bcast_sync
                         reduce_sync all (the three mechanisms in every combination)
  dsv2_l0_prefill/       128-token prefill, the ladder base -> bcast -> bcast_reduce -> all, each rung
                         adding one mechanism; all is the single-pass mapping (82.0 % of the MACs)
  dsv2_l0_prefill_tiny/  reduced-size copies (3,072 intermediate columns): all and the GEMM bench
```

Every variant directory holds exactly three files: `Makefile` (names its path and includes `ffn.mk`),
`main_bingo.py` (calls the driver), `params.hjson` (the switches). Everything else is generated.

## The prefill ladder (`dsv2_l0_prefill/`)

| variant | bcast | reduce | sync | mapping |
|---|---|---|---|---|
| `base` | pull | root (iDMA load-all, local fold) | barrier (vanilla chip barrier, full fence) | 4 iterations of 32 rows |
| `bcast` | bcast_l1 | root | barrier | 4 iterations of 32 rows |
| `bcast_reduce` | bcast_l1 | chain | barrier | 4 iterations of 32 rows |
| `all` | bcast_l1 | chain | edge | single pass |

`all` uses `mapping: single_pass`: each weight chunk used for all 128 rows as it streams in, the down
projection split by output columns over a chip's two clusters, the output reduced slice by slice under the
next slices' GEMMs. Its `params.hjson` lists every option it uses, and `common/ffn_prefill_mc.py` explains
each one. The 4-iteration `all` (1,365.4 us, 38.5 % of the MACs) is archived with the other versions.

## The reduced-size variants (`dsv2_l0_prefill_tiny/`)

The first 3,072 intermediate columns, so a run takes about 40 minutes instead of 2-3 hours; every full-size
change is tried here first. All use the same 16 + 4 chiplet platform.

| variant | purpose |
|---|---|
| `all` | `dsv2_l0_prefill/all` on 3,072 columns |
| `gemm_bench_paired` | GEMM shapes alone on one cluster, INT4 weights in the B-layout and the paired layout and INT8 (`common/gemm_bench_mc.py`) |

The earlier versions of `all` (the 4-iteration one and the single-pass ones before it), the debug variants
and the stage workloads that bisected the `all` hang are archived outside the repository
(`isca_archive/2026-10-13_ffn_variants`).

## Generating, building, running

The directories hold sources only. `make` in a variant (or the runner) generates `ffn_data.h`,
`offload_bingo_hw.h`, `final_dfg.csv`, `static_l1_layout.*` and `build/hbm/` (the HBM images of the real
weights, 35-180 MB a variant); `make clean` removes the data, the headers and the HBM images. The weights come from
`target/sim/automation/sweep/ffn/cache/` (`fetch_weights.py`).

- Generator alone (no simulation; the DFG, the hazard check and the manager-model hang check run here):
  `target/sim/automation/sweep/ffn/tools/gen_one.sh <variant under ffn/> <log>`
- Simulate variants: `target/sim/automation/sweep/ffn/run.sh run dsv2_l0_prefill/base ...`
- Simulate a prefill variant with the hang guard: `target/sim/automation/sweep/ffn/stages.sh <stage>`
  (`full` = `dsv2_l0_prefill/all`, `poc` = `dsv2_l0_prefill_tiny/all`, `bench_paired`)
- Status and the state of the ladder: `target/sim/automation/sweep/ffn/HANDOFF.md`
