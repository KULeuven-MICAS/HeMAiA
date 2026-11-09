# L2 FFN: a real LLM FFN on 16 compute chiplets + 4 memory chiplets, with an ablation

## Question

The paper's three mechanisms, each against its conventional counterpart on the same hardware, the same mapping and
the same compute:
- **broadcast**: one D2D router multicast of X from the memory chip into every chip's L3, against the memory chip's push
  engine writing one copy into each chip in turn;
- **reduce**: the xDMA ChainGather folding in transit, against the root pulling every chip's partial and folding it on
  its SIMD;
- **sync**: the BINGO hardware scheduler resolving every cross-chip dependency per edge, against software chip
  barriers after every kernel.

How much does each buy inside a real LLM layer whose weights stream from the memory chiplets, and how does the layer
scale from 4 to 16 chips?

## Workload

DeepSeek-V2-Lite's dense FFN (layer 0): `x -> gate, up (2,048 -> 10,944) -> SwiGLU -> down (10,944 -> 2,048)` over a
fresh 64-token prompt in one pass of 64 rows, INT4 weights (the checkpoint's own, `fetch_weights.py` caches them in
`cache/`), X from the root memory chip's HBM, every GEMM queued and configured ahead. The mapping (each weight chunk
meets all 64 rows in one GEMM; each chip's down projection split by output columns between its two clusters; the
outputs reduced into the root) is `workloads/ffn/README.md`'s and the driver's,
`workloads/ffn/common/ffn_prefill_mc.py` (THE ABLATION). The same block on Llama-3-8B's FFN (`4,096 -> 14,336 -> 4,096`,
synthetic INT4 weights of the model's shapes), one pass of 64 tokens with every mechanism: `workloads/ffn/llama3_prefill/t64`.

## The ladder (16 chips)

| rung | workload | X to every chip (`x_cast`) | reduce (`reduce`) | sync (`sync`) |
|---|---|---|---|---|
| R0 `base` | `dsv2_l0_prefill_ablation/base` | unicast: one copy a chip | root fold | point-to-point chip barrier after every kernel |
| R1 `bcast` | `dsv2_l0_prefill_ablation/bcast` | multicast | root fold | barrier after every kernel |
| R2 `bcast_reduce` | `dsv2_l0_prefill_ablation/bcast_reduce` | multicast | ChainGather, one a cluster | barrier after every kernel |
| R3 `all` | `dsv2_l0_prefill_ablation/all` | multicast | ChainGather, one a cluster | BINGO per-edge hardware sync, no barrier |

(workload paths under `target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/ffn/`; `base` and `all` carry the params
of the scaling sweep's 16-chip points, `scaling/base_n16` and `scaling/all_n16`, whose runs are the measured R0 and R3.)

- **The barrier** is the vanilla HeMAiA software chip barrier in point-to-point mode (`barrier_mode: sw`, chip_sync.h:
  every chip's arrival goes to the top-left chip, which releases them in turn), run as a BINGO task
  (`__snax_bingo_kernel_chip_barrier`) on each chip's cluster-0 DM core. Inside the window no dependency edge crosses
  chips in R0 to R2 (the generator checks it); only the barriers synchronise the chips.
- **Kernel by kernel** (R0 to R2): gate|up (its dequantise and SwiGLU the GEMM's epilogue), the A8 (reshape, quantise,
  the copy into the other cluster), the down slices, the reduce and stores, a barrier after each, fencing every core of
  the chip. R2 runs one ChainGather a cluster, as R3 does, so the step to R3 is the synchronisation alone.

## Running

- One rung: `PARALLEL=1 ./stages.sh abl64_<rung>` (rung base, bcast, bcast_reduce or all; tag `stg_abl64_<rung>`); one core a simulator (SIM_SLOTS, default 8);
  each run gets its sim_watchdog and `common/trim_live.py`.
- The strong-scaling sweep: `PARALLEL=1 ./stages.sh sweep64_<arm>_n<N>` (`workloads/ffn/scaling/<arm>_n<N>`, arm base
  or all, N = 4, 8, 12, 16, tag `stg_sweep64_<arm>_n<N>`, bound 2.5 ms, TIMEOUT_H 9; `all_n16` ran as `stg_poc64_g8`); the
  one-pass Llama run: `./stages.sh llama3_t64` (`workloads/ffn/llama3_prefill/t64`).
- Progress every 30 min: `setsid nohup ../../common/monitor.sh 30 <run tag> ... &` appends one line a run to
  `runs/monitor.log` (`common/runtime_mac.py`: simulated time, the window, tasks done, the GEMMs' MAC as a share
  of peak over the window and the last interval, UART errors; report only); `EXP=llm_blocks/ffn
  ../../common/wait_sample.sh` waits for the next sample (`MON_NAME=ablation_monitor` reads the older runs' log).
- Results: `./ablation_report.py` writes `results/ablation_t64.{csv,md}` (verdict and checks, the window from the root's
  SYNC line in cycles of the 500 MHz cluster clock, the MAC-bound time over it, each rung's step; each rung's newest
  `stg_abl64_<rung>` run, base and all else the sweep's 16-chip runs; `rung=tag` names another run). The sweep's table is the paper's `exp/eval4_ffn/scaling.csv`.

## Measurement

- **Anchors.** Two DM-core tasks on chip 0x00 stamp mcycle and store an SRAM-probe mark: start (after every chip's
  setup and, with barriers, the first barrier) and end (after the root's last store of Y_raw). X's transfer and the
  weight stream are inside the window.
- **MAC utilisation**: the FFN's MACs at 32 arrays x 1,024 MAC a cycle over the window: DeepSeek at 64 tokens 4.30 G
  (131,328 cycles, 262.7 us), Llama at 64 tokens 11.27 G (344,064 cycles, 688.1 us); a sweep point on N chips has 2 N arrays.
- **Checks.** The root compares Y_raw (the first and the last 2 KiB of each pass) and its own A8 with the golden, byte for
  byte; the golden follows each rung's fold order.

## Files

- `stages.sh`: one prefill app (the ablation's rungs: `abl64_<rung>`; the sweep: `sweep64_<arm>_n<N>`; the stage apps:
  `dsv2_<app>`, `llama3_<app>`; `llama3_t64`) with its watchdog.
- `ablation_report.py`: the ladder's table. `tools/`: `sp_report.py` (where a prefill run's time went), `replay_pf.py`
  (replay the task graph with measured durations). The run tools every experiment shares (the monitor, the waiters,
  the trace trimming, `gen_check.sh`, `stop_all.sh`) are in `../../common/` (its README).
- `parse_ffn.py`, `trace_ffn.py`: the level-2 parsers the MoE and FA ladders share.
  `fetch_weights.py`: the weights into `cache/`.
- `runs/stg_full_v20`: the DSv2 prefill reference run the FFN Prefill Tracker reads (T = 128, X preloaded in 0x00's L3).
