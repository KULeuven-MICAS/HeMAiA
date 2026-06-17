# HeMAiA simulation automation

Everything under `target/sim/automation` drives the HeMAiA RTL simulation flows
from Python. All of them share one backbone — `HeMAiASimRunner` in
[`util/automation_scripts/hemaia_sim_runner.py`](../../../util/automation_scripts/hemaia_sim_runner.py) —
which holds the reset / init / build / compile / run / summary logic. The
sub-directories only differ in the task list they feed it and the engine /
waveform / vendor-module knobs they set.

## Layout

| Path          | Role                                                                                   |
| ------------- | -------------------------------------------------------------------------------------- |
| `test/`       | Interactive single-task drivers (`0/1/2_start_*_sim.py`) — vsim + waveform, for debug. |
| `ci/`         | Regression suites: `local_ci` (host VCS) and `git_ci` (containerised Verilator).       |
| `sweep/`      | Op-cost LUT sweeps: `xdma` and `ara` (many tasks, VCS, no waveform).                    |
| `Makefile`    | Convenience targets (`local-ci`, `git-ci`, `sweep-xdma`, `sweep-ara`).                  |

Run any batch flow through the Makefile (`JOBS` overrides the parallel-sim count):

```bash
cd HeMAiA/target/sim/automation
make local-ci   JOBS=8     # host VCS regression
make git-ci                # Verilator regression (inside the hemaia container)
make sweep-xdma            # xDMA op-cost LUT sweep
make sweep-ara             # Ara FP32 op-cost sweep
```

> **VCS license note.** The batch flows launch up to `JOBS` parallel VCS sims.
> The launcher passes `+vcs+lic+wait` so a sim that loses the runtime-license
> race **queues** for a seat instead of exiting immediately (which used to be
> mis-reported as a FAIL in a few seconds). If you have fewer seats than `JOBS`,
> lower `JOBS` to avoid queue thrashing. Each run's full stdout is saved to
> `task_<n>/bin/sim_run.log` for post-mortem triage.

---

# Test drivers (`test/`)

End-to-end Python drivers for the three HeMAiA simulation flows used during
tapeout verification. Each script collapses the legacy shell sequence
(`0_reset_private_modules.sh` → `1_git_pull_private_modules.sh` → build inside
docker → host-side compile) into a single command. They default to the Questasim
(`vsim`) engine with a recorded waveform/log for easy debugging; pass
`--engine vcs --waveform 0` for a fast batch run.

## Layout

| File                                | Role                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------ |
| `test/0_start_single_chiplet_sim.py`| Single-chiplet RTL sim (open-source only).                                     |
| `test/1_start_multi_chiplet_sim.py` | Multi-chiplet RTL sim (TSMC16 macros + D2D, no vendor PLL).                    |
| `test/2_start_tapeout_sim.py`       | Full tapeout RTL sim (TSMC16 macros + D2D + vendor PLL).                       |

The three entrypoints are thin wrappers around `HeMAiASimRunner` in
[`util/automation_scripts/hemaia_sim_runner.py`](../../../util/automation_scripts/hemaia_sim_runner.py),
the common class shared by the CI (`ci/`), sweep (`sweep/`), and test flows.
It holds the reset / init / build / compile / run logic. To add a new flow, copy
any of the existing scripts and flip the `with_macro`, `with_d2d`, `with_pll`
flags, the cfg/sim_cfg, and the workload knobs.

## Flows

| Script                              | RTL cfg                       | Sim cfg                  | Vendor repos pulled                                |
| ----------------------------------- | ----------------------------- | ------------------------ | -------------------------------------------------- |
| `test/0_start_single_chiplet_sim.py`| `hemaia_singlechip.hjson`     | `sim_rtl.hjson`          | none (open-source `tech_cells_generic` only)       |
| `test/1_start_multi_chiplet_sim.py` | `hemaia_multichip_ci.hjson`   | `sim_rtl.hjson`          | `tech_cells_tsmc16`, `hemaia_d2d_link`             |
| `test/2_start_tapeout_sim.py`       | `hemaia_tapeout.hjson`        | `sim_rtl_with_pll.hjson` | `tech_cells_tsmc16`, `hemaia_d2d_link`, `hemaia_clk_rst_controller` |

The vendor-repo selection mirrors the `--macro / --d2d / --pll` flags of
[`../../tapeout/1_git_pull_private_modules.sh`](../../tapeout/1_git_pull_private_modules.sh):

* **`0` — single-chiplet, open-source.** No TSMC16 macros, no D2D link, no
  vendor PLL. Useful for fast iteration on cluster-level kernels.
  Simulation time: ~5–10 min.
* **`1` — multi-chiplet (no PLL).** 2×2 compute + 1 memory chip on the
  virtual interposer. Pulls the TSMC16 macros and the D2D link, but keeps
  the open-source clock/reset controller. The only delta from flow `2` is
  the PLL, so this is a good intermediate stop before enabling it.
  Simulation time: ~15–40 min.
* **`2` — full tapeout.** Same multi-chiplet setup as flow `1` plus the
  vendor PLL controller (`use_vendor_pll: true` in `hemaia_tapeout.hjson`,
  `sim_with_pll: true` in `sim_rtl_with_pll.hjson`). Pulls all three
  private repos. Simulation time: >2 h.

Recommended ladder: validate the SW with flow `0`, promote to flow `1` to
exercise the cross-chiplet path, and only run flow `2` for the final
pre-silicon sign-off pass.

## Flow inside each script

All three scripts run the same `HeMAiASimRunner` backbone:

* **Step 1 — Clean + init.** Runs `make clean` in the container, then the
  canonical `../../tapeout/0_reset_private_modules.sh` and
  `1_git_pull_private_modules.sh --macro=.. --d2d=.. --pll=..` to reset and
  clone only the vendor repos enabled by the per-flow flags.
* **Step 2 — Build HW/SW.** Runs `make sw`, `make bootrom`, `make rtl` inside
  `ghcr.io/kuleuven-micas/hemaia:main` via `podman`.
* **Step 3 — Build apps + stage.** Builds the app hex and copies it into
  `test/task_0/bin/`.
* **Step 4 — Prepare + compile.** Runs `make hemaia_system_<engine>_preparation`
  (container) and `make hemaia_system_<engine>` (host), then stages the
  simulation artefacts into the task dir. `SIM_WITH_WAVEFORM` follows
  `--waveform`.
* **Step 5 — Run** the compiled `occamy_chip.<engine>` binary in
  `test/task_0/bin/` (transcript + waveform land there) and report `PASS`/`FAIL`.
* **Step 6 — Summary.** Writes a `simulation_summary_*.md` in `test/`.

## Prerequisites

* **`vsim` on `PATH`** — source your Questasim/EDA setup before running.
  Each script aborts early if `vsim` is missing.
* **`podman`** — used to pull and run the HeMAiA build container.
* **SSH access to `github.com:IveanEx`** — required for flows `1` and `2`
  so the private vendor repos can be cloned over SSH.

## Configuring the workload

The HW configuration is fixed by the cfg files listed above. The SW workload
knobs are exposed two ways — pick whichever fits:

**1. CLI flags (no editing).** Every entrypoint accepts `--host-app-type`,
`--chip-type`, `--workload`, and `--dev-app`. Any flag you omit falls back to
the script's default. Run `--help` to see the flags and their current defaults:

```bash
python3 test/0_start_single_chiplet_sim.py --help

python3 test/0_start_single_chiplet_sim.py \
    --host-app-type offload_bingo_hw \
    --workload gemm_stacked_1cluster \
    --dev-app snax-bingo-offload
```

**2. Editing the defaults.** The defaults live as constants at the top of each
entrypoint and feed `parse_workload_args` → `HeMAiASimRunner`:

```python
HOST_APP_TYPE     = "offload_legacy"   # host_only | offload_legacy | offload_bingo_hw | offload_bingo_sw
CHIP_TYPE         = "single_chip"       # or "multi_chip"
WORKLOAD          = "None"
DEV_APP           = "snax-versacore-matmul-profile"
ENGINE            = "vsim"              # vsim | vcs | vlt
SIM_WITH_WAVEFORM = 1                   # 0 | 1
```

Editing a constant changes the default used when the matching CLI flag is
omitted. The sim-flavor knobs (`cfg`, `with_macro` / `with_d2d` / `with_pll`)
are fixed per script. `ENGINE` and `SIM_WITH_WAVEFORM` set the defaults for the
`--engine` / `--waveform` flags, so any flow can still be flipped on the CLI
(e.g. `--engine vcs --waveform 0` for a fast batch run).

## Usage

```bash
# Source the EDA environment first (vsim on PATH).
cd HeMAiA/target/sim/automation/test

# Run a flow with its built-in defaults:
python3 0_start_single_chiplet_sim.py    # open-source single-chip
python3 1_start_multi_chiplet_sim.py     # multi-chiplet, no PLL
python3 2_start_tapeout_sim.py           # full tapeout with PLL

# Or override the SW workload on the fly (omitted flags keep their defaults):
python3 0_start_single_chiplet_sim.py \
    --host-app-type offload_bingo_hw --workload gemm_stacked_1cluster \
    --dev-app snax-bingo-offload
```

Each script is idempotent — re-running it triggers a full reset → re-init
→ rebuild → re-run cycle, so it is safe to switch between flows without
manually cleaning the workspace in between.

## Iterating on SW only

If a run fails inside the device app and you only need to tweak SW, you
can skip steps 0–2 and rebuild apps in place:

```bash
# 1. Edit the SW source.
# 2. From the repo root:
make clean-sw
make apps HOST_APP_TYPE=<host app> CHIP_TYPE=<chip type> \
          WORKLOAD=<workload>      DEV_APP=<dev app>
# 3. Re-launch the existing simulation binary:
cd target/sim/bin && ./occamy_chip.vsim
```

This skips the RTL/bootrom/vsim recompile and keeps the vendor-repo and
`Bender.local` state untouched.

---

# Regression suites (`ci/`)

* `ci/local_ci` — the local CI suite (`run_local_ci.py`, `task_local_ci.yaml`),
  VCS engine, no waveform, full orchestration on a host with the EDA tools. Run
  via `make local-ci` (or `make -C target/sim/automation local-ci`).
* `ci/git_ci` — the GitHub Actions Verilator CI (`run_git_ci.py`,
  `task_git_ci.yaml`). It runs inside the `hemaia:main` container with the HW/sim
  pre-built by the workflow, so it drives the common class in its lightweight
  mode (`in_container` + `skip_setup/build/compile`). Triggered by
  `.github/workflows/ci.yml`; run manually with `make git-ci`.

# Op-cost sweeps (`sweep/`)

* `sweep/xdma` — the xDMA op-cost LUT sweep (`run_xdma_sweep.py`,
  `task_xdma.yaml` + the `gather_xdma_luts.py` / `xdma_lut_report.py`
  post-processing). Run via `make sweep-xdma`.
* `sweep/ara` — the Ara FP32 op-cost sweep (`run_ara_sweep.py`, `task_ara.yaml`
  + `gather_ara_luts.py` / `ara_lut_report.py`). One task per decomposed
  `ara_<kernel>` app (each a self-contained `src/main.c`; test data generated by
  `util/sim/ara_lib.py`). Run via `make sweep-ara`.

Both sweeps use the VCS engine with no waveform and run up to `JOBS` tasks in
parallel; see the VCS license note at the top. Each task lands in
`sweep/<name>/task_<n>/bin/` and a `simulation_summary_*.md` is written next to
the runner.
