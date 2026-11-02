# L2 FFN: a real LLM FFN on 16 compute chiplets + 4 memory chiplets, with an ablation

## Question

The paper's three mechanisms, each against its conventional counterpart on the same hardware, the same mapping and
the same compute:
- **broadcast**: one D2D router multicast of X from the memory chip into every chip's L3, against the memory chip's push
  engine writing one copy into each chip in turn;
- **reduce**: the xDMA ChainGather folding in transit, against the root pulling every chip's partial and folding it on
  its SIMD;
- **sync**: the BINGO hardware scheduler resolving every cross-chip dependency per edge, against software chip
  barriers: a barrier after every kernel (kernel by kernel), and the fused kernel that keeps a barrier only where data
  crosses chips.

How much does each buy inside a real LLM layer whose weights stream from the memory chiplets?

## Workload

DeepSeek-V2-Lite's dense FFN (layer 0): `x -> gate, up (2,048 -> 10,944) -> SwiGLU -> down (10,944 -> 2,048)` over a
fresh 256-token prompt in two passes of 128 rows, INT4 weights (the checkpoint's own, `fetch_weights.py` caches them
in `cache/`), X from the root memory chip's HBM. The mapping is `workloads/ffn/dsv2_l0_prefill/full`'s (each weight
chunk meets all of a pass's rows in one GEMM; each chip's down projection split by output columns between its two
clusters; the outputs reduced slice by slice into the root): `workloads/ffn/README.md` and the driver,
`workloads/ffn/common/ffn_prefill_mc.py` (THE ABLATION).

## The ladder (`workloads/ffn/dsv2_l0_prefill_ablation/<rung>`)

| rung | X to every chip (`x_cast`) | reduce (`reduce`) | sync (`sync`, `barrier_at`) |
|---|---|---|---|
| R0 `base` | unicast: one copy a chip | root fold | point-to-point chip barrier after every kernel |
| R1 `bcast` | multicast | root fold | barrier after every kernel |
| R2 `bcast_reduce` | multicast | ChainGather | barrier after every kernel |
| R3 `fused` | multicast | ChainGather | barrier only before each slice's gathers and between passes, gathers asynchronous |
| R4 `all` | multicast | ChainGather | BINGO per-edge hardware sync, no barrier (`dsv2_l0_prefill/full`) |

- **The barrier** is the vanilla HeMAiA software chip barrier in point-to-point mode (`barrier_mode: sw`, chip_sync.h:
  every chip's arrival goes to the top-left chip, which releases them in turn), run as a BINGO task
  (`__snax_bingo_kernel_chip_barrier`) on each chip's cluster-0 DM core. Inside the window no dependency edge crosses
  chips in R0 to R3 (the generator checks it); only the barriers synchronise the chips.
- **Kernel by kernel** (R0 to R2): per pass, gate|up (its dequantise and SwiGLU the GEMM's epilogue), the A8 (reshape,
  quantise, the copy into the other cluster), the down slices, the reduce and stores, a barrier after each, fencing
  every core of the chip. The down slices wait for the reduce in their output slots, the symmetric buffer's two and
  then the X buffer's, which nothing reads until the next pass's X.
- **Fused** (R3): the tuned schedule of R4 with every cross-chip edge replaced by a barrier: one before each slice's
  gathers (every chip's down GEMMs of the slice, the root's previous gathers) and one a pass between the last gathers
  and the next pass's A8 copies. Only the tasks a barrier names wait for it, so the gathers run under the next GEMMs
  (the reduce issued asynchronously, as collectives overlapped with compute) and the passes overlap.

## Running

- One rung: `PARALLEL=1 ./stages.sh abl_<rung>` (one core a simulator, at most 8 at once; bound 6.5 ms simulated and
  TIMEOUT_H 48 for `base` and `bcast`, 4 ms and 30 h for the others); each run gets its sim_watchdog and `trim_live.py`.
- Progress every 30 min: `setsid nohup tools/ablation_watch.sh 30 stg_abl_base ... &` appends one line a run to
  `runs/ablation_monitor.log` (`tools/runtime_mac.py`: simulated time, the window, tasks done, the GEMMs' MAC as a share
  of peak over the window and the last interval, UART errors; report only); `tools/wait_sample.sh` waits for the next
  sample.
- Results: `./ablation_report.py` writes `results/ablation.csv` and `results/ablation.md` (verdict and checks, the window
  from the root's SYNC line in cycles of the 500 MHz cluster clock, the MAC-bound time over it, each rung's step).

## Measurement

- **Anchors.** Two DM-core tasks on chip 0x00 stamp mcycle and store an SRAM-probe mark: start (after every chip's
  setup and, with barriers, the first barrier) and end (after the root's last store of Y_raw). X's transfer and the
  weight stream are inside the window.
- **MAC utilisation**: the FFN's 17.2 G MACs at 32 arrays x 1,024 MAC a cycle (525,312 cycles, 1,050.6 us) over the
  window.
- **Checks.** The root compares Y_raw (the first and the last 2 KiB of each pass) and its own A8 with the golden, byte for
  byte; the golden follows each rung's fold order.

## Files

- `stages.sh`: one prefill app (the ladder's rungs: `abl_<rung>`; the stage apps: `dsv2_<app>`, `llama3_<app>`) with its
  watchdog; `trim_live.py`, `trim_traces.py`: keep a running simulation inside the disk quota.
- `ablation_report.py`: the ladder's table. `tools/`: `runtime_mac.py`, `ablation_watch.sh`, `wait_sample.sh` (the
  monitor), `gen_one.sh` and `gen_check.sh` (the generator alone), `stage_tick.sh` and `multi_tick.sh` (waiters over
  stage runs), `stop_all.sh`, `sp_report.py` (where a prefill run's time went), `replay_pf.py` (replay the task graph with
  measured durations).
- `parse_ffn.py`, `trace_ffn.py`, `plot_l2.py`: the level-2 parsers and figures the MoE and FA ladders share.
  `fetch_weights.py`: the weights into `cache/`.
- `runs/stg_full_v20`: the DSv2 prefill reference run the FFN Prefill Tracker reads (T = 128, X preloaded in 0x00's L3).
  `results/round1/`: the first decode round (a CREST ladder on the 16 MiB-L3 platform); its workloads (`ffn/dsv2_l0`)
  and runner were removed on 1 November 2026 (archive: `~/no_backup/hemaia_cleanup_2026-11-01/`).
