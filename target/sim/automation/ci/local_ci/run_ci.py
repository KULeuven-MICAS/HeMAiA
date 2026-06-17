#!/usr/bin/env python3
"""
HeMAiA CI runner
================

Builds and runs the full ``task_local_ci.yaml`` CI suite through the shared
:class:`HeMAiASimRunner`.  Defaults to the VCS engine with no waveform recording
(fast batch runs); pass ``--engine vsim --waveform 1`` to reproduce a failing
task under Questasim with a waveform for debugging.

    python3 run_ci.py [-j JOBS] [-f task.yaml] [--cfg CFG] [--engine vcs|vsim]
                      [--waveform 0|1] [--no-d2d] [--no-macro]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT.parents[5]  # target/sim/automation/ci/local_ci -> repo root
sys.path.insert(0, str(_REPO_ROOT / "util" / "automation_scripts"))

from hemaia_sim_runner import (  # noqa: E402
    DEFAULT_MAX_SIM_JOBS, ENGINES, HeMAiASimRunner, parse_tasks,
)

CI_CFG = "target/rtl/cfg/hemaia_multichip_ci.hjson"
SIM_CFG = "target/sim/cfg/sim_rtl.hjson"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run the HeMAiA CI suite.")
    parser.add_argument(
        "-j", "--max-sim-jobs", type=int, default=DEFAULT_MAX_SIM_JOBS,
        help=f"max simulations to run in parallel (default: {DEFAULT_MAX_SIM_JOBS})")
    parser.add_argument(
        "-f", "--task-yaml", default=None,
        help="task list YAML (default: task_local_ci.yaml next to this script). A "
             "relative path is resolved against the local_ci dir, then cwd.")
    parser.add_argument(
        "--cfg", default=CI_CFG,
        help="RTL/SW config (CFG_OVERRIDE) for the whole flow (default: %(default)s).")
    parser.add_argument(
        "--engine", choices=sorted(ENGINES), default="vcs",
        help="simulation engine (default: %(default)s)")
    parser.add_argument(
        "--waveform", type=int, choices=(0, 1), default=0,
        help="SIM_WITH_WAVEFORM (default: %(default)s)")
    parser.add_argument(
        "--d2d", action=argparse.BooleanOptionalAction, default=True,
        help="init the hemaia_d2d_link private module (use --no-d2d for single-chip flow).")
    parser.add_argument(
        "--macro", action=argparse.BooleanOptionalAction, default=True,
        help="init the tech_cells_tsmc16 private module (use --no-macro for the single-chip flow).")
    args = parser.parse_args()
    if args.max_sim_jobs < 1:
        parser.error("--max-sim-jobs must be >= 1")
    return args


def _resolve_task_yaml(arg: str | None) -> Path:
    if not arg:
        return _SCRIPT.parent / "task_local_ci.yaml"
    candidate = Path(arg)
    if candidate.is_absolute():
        return candidate
    if (_SCRIPT.parent / candidate).exists():
        return _SCRIPT.parent / candidate
    return (Path.cwd() / candidate).resolve()


def main() -> None:
    args = parse_args()
    task_yaml = _resolve_task_yaml(args.task_yaml)
    if not task_yaml.exists():
        raise FileNotFoundError(f"Task YAML file {task_yaml} does not exist")
    print(f"Using task list: {task_yaml}")

    runner = HeMAiASimRunner(
        repo_root=_REPO_ROOT,
        output_dir=_SCRIPT.parent,
        engine=args.engine,
        with_waveform=bool(args.waveform),
        cfg=args.cfg,
        sim_cfg=SIM_CFG,
        with_macro=args.macro,
        with_d2d=args.d2d,
        with_pll=False,
        max_jobs=args.max_sim_jobs,
    )
    runner.run(parse_tasks(task_yaml))


if __name__ == "__main__":
    main()
