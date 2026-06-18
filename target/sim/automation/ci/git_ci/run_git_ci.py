#!/usr/bin/env python3
"""
HeMAiA git (GitHub Actions) CI runner
=====================================

Runs the ``task_git_ci.yaml`` Verilator regression on a GitHub runner.  Like the
local CI, this drives the *whole* flow end-to-end (build sw/bootrom/rtl, compile
the Verilator binary, build the per-task apps, run) so the workflow only has to
check out the repo and call this one script -- no duplicated build steps in
``ci.yml``.

The GitHub runner imposes two constraints that the shared :class:`HeMAiASimRunner`
handles via switches:

  * ``in_container`` -- the job already runs *inside* the ``hemaia:main``
    container, so the "container" make steps run directly (no podman-in-podman).
  * ``skip_setup``   -- no reset / private-repo clone: GH runners have no SSH for
    the private vendor repos, and the Verilator CI cfg is open-source single-chip
    (no tsmc16 / D2D / PLL), so there is nothing to clone.

    python3 run_git_ci.py [-j JOBS] [-f task.yaml] [--cfg CFG]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT.parents[5]  # target/sim/automation/ci/git_ci -> repo root
sys.path.insert(0, str(_REPO_ROOT / "util" / "automation_scripts"))

from hemaia_sim_runner import (  # noqa: E402
    DEFAULT_MAX_SIM_JOBS, HeMAiASimRunner, parse_tasks, resolve_task_yaml,
)

# RTL/SW config the whole flow is built with (the NoC job passes
# --cfg target/rtl/cfg/hemaia_ci_noc.hjson).
GIT_CI_CFG = "target/rtl/cfg/hemaia_ci.hjson"
SIM_CFG = "target/sim/cfg/sim_rtl.hjson"
DEFAULT_TASK_YAML = "task_git_ci.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HeMAiA Verilator CI on a GitHub runner.")
    parser.add_argument(
        "-j", "--max-sim-jobs", type=int, default=DEFAULT_MAX_SIM_JOBS,
        help=f"max simulations to run in parallel (default: {DEFAULT_MAX_SIM_JOBS})")
    parser.add_argument(
        "-f", "--task-yaml", default=_SCRIPT.parent / DEFAULT_TASK_YAML,
        help=f"task list YAML (default: {DEFAULT_TASK_YAML} next to this script).")
    parser.add_argument(
        "--cfg", default=GIT_CI_CFG,
        help="RTL/SW config (CFG_OVERRIDE) the apps are built with (default: %(default)s).")
    args = parser.parse_args()
    if args.max_sim_jobs < 1:
        parser.error("--max-sim-jobs must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    task_yaml = resolve_task_yaml(args.task_yaml)
    if not task_yaml.exists():
        raise FileNotFoundError(f"Task YAML file {task_yaml} does not exist")
    print(f"Using task list: {task_yaml}")

    runner = HeMAiASimRunner(
        repo_root=_REPO_ROOT,
        output_dir=_SCRIPT.parent,
        engine="vlt",
        with_waveform=False,
        cfg=args.cfg,
        sim_cfg=SIM_CFG,
        with_macro=False,
        with_d2d=False,
        with_pll=False,
        max_jobs=args.max_sim_jobs,
        # GitHub runner: already inside the build container, but with no SSH for
        # the private vendor repos.  Build/compile are driven here (not skipped),
        # parallelising the Verilator compile across all available cores.
        in_container=True,
        skip_setup=True,
        compile_jobs=os.cpu_count(),
    )
    runner.run(parse_tasks(task_yaml))


if __name__ == "__main__":
    main()
