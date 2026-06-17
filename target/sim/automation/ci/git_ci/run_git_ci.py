#!/usr/bin/env python3
"""
HeMAiA git (GitHub Actions) CI runner
=====================================

Runs the ``task_vlt.yaml`` Verilator regression on a GitHub runner.  Unlike the
local CI, this executes *inside* the ``hemaia:main`` container with the HW/sim
already built by the workflow steps (``make sw/bootrom/rtl/hemaia_system_vlt``).

It therefore drives the shared :class:`HeMAiASimRunner` in its lightweight mode:

  * ``in_container``  -- run make directly (no podman-in-container)
  * ``skip_setup``    -- no reset / private-repo clone (no SSH on GH runners)
  * ``skip_build``    -- the workflow already built sw/bootrom/rtl
  * ``skip_compile``  -- the workflow already built the Verilator binary; just
                         build the per-task apps, stage the binary, and run.

    python3 run_git_ci.py [-j JOBS] [-f task.yaml] [--cfg CFG]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT.parents[5]  # target/sim/automation/ci/git_ci -> repo root
sys.path.insert(0, str(_REPO_ROOT / "util" / "automation_scripts"))

from hemaia_sim_runner import (  # noqa: E402
    DEFAULT_MAX_SIM_JOBS, HeMAiASimRunner, parse_tasks,
)

# Must match the CFG_OVERRIDE the workflow built sw/rtl with (the NoC job passes
# --cfg target/rtl/cfg/hemaia_ci_noc.hjson).
GIT_CI_CFG = "target/rtl/cfg/hemaia_ci.hjson"
SIM_CFG = "target/sim/cfg/sim_rtl.hjson"  # unused (skip_compile), kept for completeness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HeMAiA Verilator CI on a GitHub runner.")
    parser.add_argument(
        "-j", "--max-sim-jobs", type=int, default=DEFAULT_MAX_SIM_JOBS,
        help=f"max simulations to run in parallel (default: {DEFAULT_MAX_SIM_JOBS})")
    parser.add_argument(
        "-f", "--task-yaml", default=None,
        help="task list YAML (default: task_vlt.yaml next to this script).")
    parser.add_argument(
        "--cfg", default=GIT_CI_CFG,
        help="RTL/SW config (CFG_OVERRIDE) the apps are built with (default: %(default)s).")
    args = parser.parse_args()
    if args.max_sim_jobs < 1:
        parser.error("--max-sim-jobs must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    if args.task_yaml:
        candidate = Path(args.task_yaml)
        task_yaml = candidate if candidate.is_absolute() else _SCRIPT.parent / candidate
    else:
        task_yaml = _SCRIPT.parent / "task_vlt.yaml"
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
        # GitHub runner: already inside the container with HW/sim pre-built.
        in_container=True,
        skip_setup=True,
        skip_build=True,
        skip_compile=True,
    )
    runner.run(parse_tasks(task_yaml))


if __name__ == "__main__":
    main()
