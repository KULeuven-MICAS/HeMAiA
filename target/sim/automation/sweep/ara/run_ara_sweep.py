#!/usr/bin/env python3
"""
HeMAiA Ara FP32 op-cost sweep runner
====================================

Builds and runs ``task_ara.yaml`` (one task per decomposed ara_<kernel> app)
through the shared :class:`HeMAiASimRunner`, on the single-chip config (the
kernels run on the CVA6 host + Ara vector unit).  Defaults to the VCS engine
with no waveform recording.

After this finishes, post-process the per-task UART logs into a cycle CSV:

    python3 gather_ara_luts.py     # -> ara_cycles.csv
    python3 ara_lut_report.py      # human-readable summary of the measured cycles

    python3 run_ara_sweep.py [-j JOBS] [-f task.yaml] [--engine vcs|vsim] [--waveform 0|1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT.parents[5]  # target/sim/automation/sweep/ara -> repo root
sys.path.insert(0, str(_REPO_ROOT / "util" / "automation_scripts"))

from hemaia_sim_runner import (  # noqa: E402
    DEFAULT_MAX_SIM_JOBS, ENGINES, HeMAiASimRunner, parse_tasks,
)

# host_only single-chip; the kernels run on CVA6+Ara, no D2D/macro.
ARA_CFG = "target/rtl/cfg/hemaia_singlechip.hjson"
SIM_CFG = "target/sim/cfg/sim_rtl.hjson"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run the HeMAiA Ara FP32 op-cost sweep.")
    parser.add_argument(
        "-j", "--max-sim-jobs", type=int, default=DEFAULT_MAX_SIM_JOBS,
        help=f"max simulations to run in parallel (default: {DEFAULT_MAX_SIM_JOBS})")
    parser.add_argument(
        "-f", "--task-yaml", default=None,
        help="task list YAML (default: task_ara.yaml next to this script).")
    parser.add_argument(
        "--engine", choices=sorted(ENGINES), default="vcs",
        help="simulation engine (default: %(default)s)")
    parser.add_argument(
        "--waveform", type=int, choices=(0, 1), default=0,
        help="SIM_WITH_WAVEFORM (default: %(default)s)")
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
        task_yaml = _SCRIPT.parent / "task_ara.yaml"
    if not task_yaml.exists():
        raise FileNotFoundError(f"Task YAML file {task_yaml} does not exist")
    print(f"Using task list: {task_yaml}")

    runner = HeMAiASimRunner(
        repo_root=_REPO_ROOT,
        output_dir=_SCRIPT.parent,
        engine=args.engine,
        with_waveform=bool(args.waveform),
        cfg=ARA_CFG,
        sim_cfg=SIM_CFG,
        with_macro=False,
        with_d2d=False,
        with_pll=False,
        max_jobs=args.max_sim_jobs,
    )
    runner.run(parse_tasks(task_yaml))


if __name__ == "__main__":
    main()
