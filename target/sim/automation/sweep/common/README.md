# sweep/common: one button per experiment

The `sop_ubench` and `llm_blocks` experiments run through `hsweep.py`.
It wraps `HeMAiASimRunner` and adds what a sweep needs around it.

| command | does |
|---|---|
| `launch.sh build <cfg> [tree]` | builds the RTL, bootrom and simv for a cfg, detached, with a watchdog. Writes `target/sim/bin/.hsweep_stamp.json` (the cfg's content hash, the snax commit) |
| `launch.sh run <exp> <tag> <points.yaml> [--tree T] [--jobs N]` | builds the apps and runs the points, detached, with a watchdog. Output goes to `<tree>/target/sim/automation/sweep/<exp>/runs/<tag>/` |
| `hsweep.py verdict <run dir>` | the verdict for each task. PASS needs all of: the end-of-computation marker, no failed check on any UART, no live watchdog, no timeout. Writes `verdict.json` |
| `hsweep.py prune <run dir>` | deletes the traces once a run has been parsed (it keeps the host trace and the UARTs) |
| `watchdog.sh <run dir> [profile] [key=value ...]` | starts the hang guard `sim_watchdog.py` for one run, detached; run it again to bring a dead one back |
| `sim_watchdog.py status <run dir>` | one line: simulated time, rate, chiplets busy, tasks fetched/released/started/done, quiet time against its limit; `STALE` if the watchdog died |
| `sim_watchdog.py params [--profile P]` | every parameter, its value and what it does |
| `sim_watchdog.py analyse --log bingo_progress.log` | replays a finished log through the detectors and prints when, and why, the first would have fired |

**Rules this enforces:**
- `run` refuses to start when the standing simv was built for another cfg. Rebuild first.
- App builds take a lock on the tree, and take `.hsweep_chisel.lock` in the tree's snax checkout
  (`deps/snitch_cluster`, or the path `Bender.local` overrides `snitch_cluster` to) SHARED, since they
  only read the snax tree. RTL builds take it exclusively, and so must anything else that runs sbt
  there (`flock` does by default), so a generation never overlaps an app build in any tree that
  builds against the same checkout.
- At most `SIM_SLOTS` (default 3) of this session's simulations run at once, across trees. A run waits for a free slot.
  Taking a slot happens under `common/.sim_slots.lock` until the new simulation process exists, so
  runners that wait together do not overshoot.
- Every launch arms `watch.sh`. Events go to `<run dir>.events` and to `common/all.events`:
  ARMED, STAGED, SIM START, HEARTBEAT, CHECK FAIL, L1 OOM, LIVE WATCHDOG, FROZEN, EXITED, DONE, GONE.
  Liveness is the host trace's simulation time, not trace growth.

**A points file:**

```yaml
cfg: target/rtl/cfg/hemaia_16c4m_8MBL3_2cluster.hjson
memchip: true          # with_macro / with_d2d for cfgs with a memory chiplet
timeout_h: 3
points:
  - {name: allreduce, host_app_type: offload_legacy, chip_type: multi_chip,
     workload: None, dev_app: snax-coll-allreduce}
```

## The hang guard

`sim_watchdog.py` watches one simulator and stops it (SIGINT, then SIGKILL) when the run is hung or runs out
of room. `SIM_WATCHDOG=<profile> launch.sh run ...` arms it for a run; `watchdog.sh` does so by hand. Its
liveness signal is the **scheduler monitor**: the testbench (`target/sim/testharness/soc_probes.sv`,
`snax_bingo_progress_probe`, bound into every `bingo_hw_manager_top`) writes `bingo_progress.log` next to the
simulator, one line per task event (`F` fetched, `R` released, `S` started, `D` done) and, every few
microseconds of simulated time, each chiplet's counters and queue heads. A workload moves exactly when those
counters move; a core's instruction trace does not show that, because a core parks in a blocking read whenever
its next task waits for a dependency. `+bingo_quiet` drops the per-task lines (the progress lines stay),
`+bingo_prog_cyc=N` and `+bingo_stall_cyc=N` set the progress period and the per-chiplet quiet report.

It stops the run on: no task fetched, released, started or done on any chiplet for `max(stall_floor_us,
stall_factor x the longest task so far)` of simulated time while a chiplet has work outstanding (`HANG-SCHED`);
a real `[AXI-STALL]` line while the scheduler is also quiet (`HANG-AXI`); a frozen clock and logs (`WEDGE`); an
assertion; disk headroom or log size; the time bound; no chiplet started its list (`NOT-ARMED`); too few
simulator threads; the simulator still up long after every list drained (`HANG-EXIT`). Parameters and named
profiles (`sim_watchdog.profiles.json`) are listed by `params`. Output: `<run dir>.events` (a HEARTBEAT every 10
min, WARN and STOP lines, the same file `watch.sh` writes), `<run dir>.hang.txt` (the evidence of a stop: every
busy chiplet's counters and queue heads with task names, the stalled AXI links), `<run dir>.watchdog.json` (the live
status). `test_sim_watchdog.py` tests it with scripted logs and a stand-in process (no simulator needed).

**Hardware build:** `hemaia_16c4m_8MBL3_2cluster`, the paper's platform: a 5 x 4 package, 16 compute
chiplets (2 split clusters each, 1 MiB TCDM per cluster, 8 MiB L3 per chip) and 4 memory chiplets in
the middle column. The collectives (`sop_ubench/collective`) and the LLM blocks (`llm_blocks/ffn`, `llm_blocks/moe`,
`llm_blocks/fa`) run on it; `sop_ubench/sync` re-plots measured data and needs no build.

**Weights:** `weights/` holds DeepSeek-V2-Lite's layer 1 (`weights/fetch_weights.py`, the cache in
`weights/cache/` with its manifest), which the MoE and FA workloads read; the FFN fetches its own
layer into `llm_blocks/ffn/cache/` with the same HTTP helpers.

**Pitfalls behind the design:**
- A failing simulation exits 0.
- The stock stamp never checks the cfg.
- `make clean` wipes every build product, and two builds in one tree race.
- `/tmp` is RAM, so prune after parsing.
- Never `pkill -f`: kill by PID.
