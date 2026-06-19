#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""
HeMAiA VersaCore GEMM precision LUT sweep runner.

Builds and runs ``task_gemm.yaml`` (one task per precision) through the shared
:class:`HeMAiASimRunner`, using the single-chip config — same pattern as the
xDMA/ara sweeps. Each task is a ``gemm_psweep_<prec>_1cluster`` workload that
sweeps the shared (M,K,N,array_shape) grid in ONE sim (precision can't be looped
in one binary), so the 5 precisions run in parallel.

After this finishes, post-process the per-task traces into a cycle CSV:

    python3 gather_gemm_luts.py    # -> gemm_cycles.csv  (per precision x shape)

    python3 run_gemm_sweep.py [-j JOBS] [-f task.yaml] [--engine vcs|vsim] [--waveform 0|1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT.parents[5]  # target/sim/automation/sweep/gemm -> repo root
sys.path.insert(0, str(_REPO_ROOT / "util" / "automation_scripts"))

from hemaia_sim_runner import (  # noqa: E402
    DEFAULT_MAX_SIM_JOBS, ENGINES, HeMAiASimRunner, parse_tasks, resolve_task_yaml,
)

# Single-chip: 16 MiB host WIDE_SPM holds the baked per-config arrays; no D2D/macro.
SWEEP_CFG = "target/rtl/cfg/hemaia_singlechip.hjson"
SIM_CFG = "target/sim/cfg/sim_rtl.hjson"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run the HeMAiA GEMM precision LUT sweep.")
    parser.add_argument(
        "-j", "--max-sim-jobs", type=int, default=DEFAULT_MAX_SIM_JOBS,
        help=f"max simulations to run in parallel (default: {DEFAULT_MAX_SIM_JOBS})")
    parser.add_argument(
        "-f", "--task-yaml", default=_SCRIPT.parent / "task_gemm.yaml",
        help="task list YAML (default: task_gemm.yaml next to this script).")
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
    task_yaml = resolve_task_yaml(args.task_yaml)
    if not task_yaml.exists():
        raise FileNotFoundError(f"Task YAML file {task_yaml} does not exist")
    print(f"Using task list: {task_yaml}")

    runner = HeMAiASimRunner(
        repo_root=_REPO_ROOT,
        output_dir=_SCRIPT.parent,
        engine=args.engine,
        with_waveform=bool(args.waveform),
        cfg=SWEEP_CFG,
        sim_cfg=SIM_CFG,
        with_macro=False,
        with_d2d=False,
        with_pll=False,
        max_jobs=args.max_sim_jobs,
    )
    runner.run(parse_tasks(task_yaml))


if __name__ == "__main__":
    main()
