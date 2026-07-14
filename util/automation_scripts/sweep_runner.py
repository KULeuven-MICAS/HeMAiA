#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Shared CLI for the per-domain LUT sweep runners.

``run_{ara,gemm,xdma}_sweep.py`` only differ in their default task YAML name and
help text: they all build and run a task list through :class:`HeMAiASimRunner` on
the single-chip config (host CVA6 + accelerators, no D2D/macro/PLL). This wraps
that identical argparse + runner construction so each runner is a one-line call.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hemaia_sim_runner import (  # noqa: E402
    DEFAULT_MAX_SIM_JOBS, ENGINES, HeMAiASimRunner, parse_tasks, resolve_task_yaml,
)

# Single-chip: host WIDE_SPM holds the baked per-config arrays; no D2D/macro/PLL.
DEFAULT_CFG = "target/rtl/cfg/hemaia_singlechiplet_1cluster.hjson"
DEFAULT_SIM_CFG = "target/sim/cfg/sim_rtl.hjson"


def run_sweep_cli(script_file, *, description, default_task_name,
                  cfg=DEFAULT_CFG, sim_cfg=DEFAULT_SIM_CFG):
    """Parse args and run the sweep described by *script_file*'s task YAML.

    *script_file* is the caller's ``__file__``; its directory is the run output
    dir and the home of the default ``default_task_name`` task list.
    """
    script = Path(script_file).resolve()
    repo_root = Path(__file__).resolve().parents[2]  # util/automation_scripts -> repo root

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-j", "--max-sim-jobs", type=int, default=DEFAULT_MAX_SIM_JOBS,
        help=f"max simulations to run in parallel (default: {DEFAULT_MAX_SIM_JOBS})")
    parser.add_argument(
        "-f", "--task-yaml", default=script.parent / default_task_name,
        help=f"task list YAML (default: {default_task_name} next to the runner).")
    parser.add_argument(
        "--engine", choices=sorted(ENGINES), default="vcs",
        help="simulation engine (default: %(default)s)")
    parser.add_argument(
        "--waveform", type=int, choices=(0, 1), default=0,
        help="SIM_WITH_WAVEFORM (default: %(default)s)")
    args = parser.parse_args()
    if args.max_sim_jobs < 1:
        parser.error("--max-sim-jobs must be >= 1")

    task_yaml = resolve_task_yaml(args.task_yaml)
    if not task_yaml.exists():
        raise FileNotFoundError(f"Task YAML file {task_yaml} does not exist")
    print(f"Using task list: {task_yaml}")

    runner = HeMAiASimRunner(
        repo_root=repo_root,
        output_dir=script.parent,
        engine=args.engine,
        with_waveform=bool(args.waveform),
        cfg=cfg,
        sim_cfg=sim_cfg,
        with_macro=False,
        with_d2d=False,
        with_pll=False,
        max_jobs=args.max_sim_jobs,
    )
    runner.run(parse_tasks(task_yaml))
