# Sweeps

Every simulation experiment of HeMAiA, grouped by what it measures. Each experiment directory has
its own README or runner docstring; this page is the map.

| group | experiment | what it measures | how it runs |
|---|---|---|---|
| `unit_test/` | `xdma` | the xDMA's op-cost LUT (copies, layout changes, extensions) | `make sweep-xdma` |
| | `ara` | the Ara FP32 kernels' op-cost LUT, one app per kernel | `make sweep-ara` |
| | `simd` | the SIMD operators' cost LUT (`task_simd.yaml`, `task_simd_fused.yaml`) | `make sweep-simd` |
| | `gemm` | the VersaCore GEMM precision sweep, one task per precision | `make sweep-gemm` |
| | `versacore` | VersaCore GEMM tests from a workload table (`testing_launch.py`) | `python3 testing_launch.py` |
| `sop_ubench/` | `reduce` | E2: the chain reduce against trees of chains and the software reductions, in a standalone hardware testbench | `./run.sh build`, `./run.sh sweep` |
| | `sync` | E3: one cross-chip synchronisation, four mechanisms | `./run.sh launch`, `./run.sh parse`, `./run.sh` |
| `llm_blocks/` | `ffn` | DeepSeek-V2-Lite's layer-0 FFN on 16 + 4 chiplets: the ablation ladder at 64 tokens and the strong-scaling sweep | `./stages.sh <stage>` |
| | `moe` | DeepSeek-V2-Lite's layer-1 MoE, a 512-token prefill, with the ablation ladder | `./stages.sh abl_<rung>` |
| | `fa` | the attention block: Llama-3-8B's decode and 1,024-token prefill ladders, and the prefill stage apps | `./stages.sh <stage>`, `./run.sh run` |
| `llm_layer/` | `dsv2_prefill` | DeepSeek-V2-Lite's layers 0 and 1 as one graph each, a fresh 1,024-token prompt on 16 + 4 chiplets | `./stages.sh layer{0,1}_prefill_full` |
| | `llama3_prefill` | a Llama-3-8B decoder layer as one graph, a fresh 1,024-token prompt on 16 + 4 chiplets | `./stages.sh fl_full` |
| `misc/` | `cerf_moe` | the cycles to change K tasks of every chiplet's graph, by the CERF or by the hosts | `./stages.sh <arm>_k<K>` |
| | `cerf_node` | the decode control path of one node | `common/launch.sh run misc/cerf_node <tag> points.yaml` |
| | `fgp_bench` | VCS fine-grained parallelism: simulated time a minute at 1, 4 and 8 threads | (results only) |
| `common/` | | the hsweep runner, the launcher, the watchdog, the run tools and the model weights the others share | |

`sop_ubench` stands for the system-of-packages microbenchmarks: the chiplet package's own
mechanisms, measured apart from any workload.

## How an experiment runs

**The sim runner's task flow** (`unit_test/`). A `run_*_sweep.py` (or
`testing_launch.py`) builds and runs the tasks of a task yaml through
`util/automation_scripts/hemaia_sim_runner.py`. Each task lands in `task_<n>_<workload>/bin/`
next to the script, with a `simulation_summary_*.md`; the `gather_*` / `*_report.py` scripts turn
the tasks into the LUTs. Without `--sw-only` they start with `make clean` and rebuild the hardware.

**hsweep** (`sop_ubench/sync`, `llm_blocks/`, `llm_layer/`, `misc/`). The experiment's `run.sh` or `stages.sh`
writes a points file and starts one runner through `common/launch.sh`:

    common/launch.sh build <cfg>                            # a cfg's simv; the runs reuse it
    common/launch.sh run <group>/<exp> <tag> <points.yaml>  # a run, detached, with its event log
    common/watchdog.sh <run dir> <profile> [key=value ...]  # the hang guard (sim_watchdog.py)

A run lands in `<group>/<exp>/runs/<tag>/`, with `<tag>.events` (the event log), `<tag>.log`
and `<tag>.watchdog.json` beside it. The parse scripts turn the runs into `results/` and the
plot scripts into `figs/`. The scripts find `sweep/` and their experiment name (its path under
`sweep/`, e.g. `llm_blocks/moe`) from where they sit, so the directories can be moved.

**A standalone testbench** (`sop_ubench/reduce`): its `run.sh` builds and runs its own VCS testbench of the
package's reduction path; no software, no runner.

Shared by the LLM experiments: the weights of DeepSeek-V2-Lite's layer 1 in `common/weights/cache/`
(`common/weights/fetch_weights.py`), the FFN's layer 0 in `llm_blocks/ffn/cache/`, and the FA's
KV cache in `llm_blocks/fa/cache/` (`llm_blocks/fa/real_kv_cache.py`).

## What git keeps

Scripts, task and point files, and READMEs (`../.gitignore` lists them). Runs, `task_<n>/` dirs,
summaries, caches, `results/` and `figs/` stay out; `misc/` holds no tracked file.

## Cleaning

`make clean-stale` (also part of the repository's `make clean`) runs `../clean_stale.py`:
old `task_<n>/` dirs and summaries, the core traces of finished hsweep runs (after their task spans
are taken into `tasks.csv`), staged images and simulator copies, old build logs and the workloads'
generated `build/` dirs, but only what has been left unchanged for 24 hours (`STALE_AGE_H`).
`STALE_DRY=1` lists what would go. A run directory holding a file named `KEEP` is never touched.
