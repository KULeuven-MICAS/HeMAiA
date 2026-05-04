# HeMAiA tapeout sim drivers

End-to-end Python drivers for the three Questasim-based simulation flows used
during HeMAiA tapeout verification. Each script collapses the legacy shell
sequence (`0_reset_private_modules.sh` → `1_git_pull_private_modules.sh` → `2_init_inside_docker.sh`
→ host-side `make hemaia_system_vsim`) into a single command.

## Layout

| File                                | Role                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------ |
| [`test_util.py`](test_util.py)      | Shared helpers + the `run_sim_flow` driver. No entrypoint of its own.          |
| `0_start_single_chiplet_sim.py`     | Single-chiplet RTL sim (open-source only).                                     |
| `1_start_multi_chiplet_sim.py`      | Multi-chiplet RTL sim (TSMC16 macros + D2D, no vendor PLL).                    |
| `2_start_tapeout_sim.py`            | Full tapeout RTL sim (TSMC16 macros + D2D + vendor PLL).                       |

The three entrypoints are thin wrappers around `run_sim_flow` in `test_util.py`,
which holds the shared reset / init / build / compile / run logic. To add a
new flow, copy any of the existing scripts and flip the `with_macro`,
`with_d2d`, `with_pll` flags and the workload knobs.

## Flows

| Script                              | RTL cfg                       | Sim cfg                  | Vendor repos pulled                                |
| ----------------------------------- | ----------------------------- | ------------------------ | -------------------------------------------------- |
| `0_start_single_chiplet_sim.py`     | `hemaia_singlechip.hjson`     | `sim_rtl.hjson`          | none (open-source `tech_cells_generic` only)       |
| `1_start_multi_chiplet_sim.py`      | `hemaia_multichip_ci.hjson`   | `sim_rtl.hjson`          | `tech_cells_tsmc16`, `hemaia_d2d_link`             |
| `2_start_tapeout_sim.py`            | `hemaia_tapeout.hjson`        | `sim_rtl_with_pll.hjson` | `tech_cells_tsmc16`, `hemaia_d2d_link`, `hemaia_clk_rst_controller` |

The vendor-repo selection mirrors the `--macro / --d2d / --pll` flags of
[`../tapeout/1_git_pull_private_modules.sh`](../tapeout/1_git_pull_private_modules.sh):

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

All three scripts share the same five-step backbone:

* **Step 0 — Reset** (mirrors `../tapeout/0_reset_private_modules.sh`). Removes any of
  `hw/hemaia/{tech_cells_tsmc16,hemaia_d2d_link,hemaia_clk_rst_controller}`
  that exist, drops their `Bender.local` entries, restores the open-source
  `tech_cells_generic` line, deletes `Bender.lock`, and runs `make clean`.
* **Step 1 — Init outside docker** (mirrors `../tapeout/1_git_pull_private_modules.sh`).
  Clones / pulls only the vendor repos enabled by the per-flow `with_macro`,
  `with_d2d`, `with_pll` flags, and upserts the matching `Bender.local`
  entries.
* **Step 2 — Init inside docker** (mirrors `../2_init_inside_docker.sh`).
  Runs `make apps`, `make bootrom`, `make rtl`, and
  `make hemaia_system_vsim_preparation` inside
  `ghcr.io/kuleuven-micas/hemaia:main` via `podman`.
* **Step 3 — Compile vsim** on the host (Questasim is host-only).
* **Step 4 — Run** the compiled `target/sim/bin/occamy_chip.vsim` binary
  and print `PASS`/`FAIL` plus the last 10 lines of output.

## Prerequisites

* **`vsim` on `PATH`** — source your Questasim/EDA setup before running.
  Each script aborts early if `vsim` is missing.
* **`podman`** — used to pull and run the HeMAiA build container.
* **SSH access to `github.com:IveanEx`** — required for flows `1` and `2`
  so the private vendor repos can be cloned over SSH.

## Configuring the workload

The HW configuration is fixed by the cfg files listed above. The SW
workload knobs sit at the top of each entrypoint and feed straight into
`run_sim_flow`:

```python
HOST_APP_TYPE = "offload_bingo_hw"  # host_only | offload_legacy | offload_bingo_hw | offload_bingo_sw
CHIP_TYPE     = "single_chip"       # or "multi_chip"
WORKLOAD      = "gemm_stacked_1cluster"
DEV_APP       = "snax-bingo-offload"

def main() -> None:
    run_sim_flow(
        cfg_name="hemaia_singlechip.hjson",
        sim_cfg_name="sim_rtl.hjson",
        with_macro=False,
        with_d2d=False,
        with_pll=False,
        host_app_type=HOST_APP_TYPE,
        chip_type=CHIP_TYPE,
        workload=WORKLOAD,
        dev_app=DEV_APP,
        script_path=Path(__file__),
    )
```

Edit those constants in the relevant script before launching to point at a
different workload or host app.

## Usage

```bash
# Source the EDA environment first (vsim on PATH).
cd HeMAiA/target/testing

python3 0_start_single_chiplet_sim.py    # open-source single-chip
python3 1_start_multi_chiplet_sim.py     # multi-chiplet, no PLL
python3 2_start_tapeout_sim.py           # full tapeout with PLL
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

## Full local CI
It is encouraged to look around the HeMAiA/target/sim/automation/ci/start_vsim.py and the HeMAiA/target/sim/automation/ci/task_vsim.yaml on the full local CI.
