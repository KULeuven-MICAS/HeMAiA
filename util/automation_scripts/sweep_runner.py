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

import os
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hemaia_sim_runner import (  # noqa: E402
    DEFAULT_MAX_SIM_JOBS, ENGINES, HeMAiASimRunner, parse_tasks, resolve_task_yaml,
)

# Single-chip: host WIDE_SPM holds the baked per-config arrays; no D2D/macro/PLL.
# 16 MiB of wide SPM, so a sweep's golden arrays and its per-config task descriptors
# both fit -- the 128 KiB tapeout L3 is what forces the CI suites to stage on a memchip.
DEFAULT_CFG = "target/rtl/cfg/hemaia_singlechiplet_16MB_1cluster.hjson"
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
    parser.add_argument(
        "--cfg", default=cfg,
        help="RTL/SW config (CFG_OVERRIDE) for the whole flow (default: %(default)s).")
    parser.add_argument(
        "--sw-only", action="store_true",
        help="fast SW-only re-run: reuse the already-built RTL and compiled simulation, "
             "rebuild ONLY the per-task app binaries, and re-run. Requires a prior full "
             "run of this sweep at the same --cfg.")
    parser.add_argument(
        "--reuse-build", action="store_true",
        help="reuse the already-built SW/bootrom/RTL but still COMPILE the simulation. "
             "What --sw-only cannot do: recover from a failed or absent EDA compile "
             "without paying for the ~30-40 min RTL generation again. Also skips the "
             "repo reset, so hand edits to generated RTL survive.")
    parser.add_argument(
        "--sim-cores", type=int, default=1, metavar="N",
        help="VCS cores per SIMULATION (fine-grained parallelism). 1 = single-core, the "
             "default. >1 passes -fgp=num_threads:N-1 with -fgp=allow_less_cores, so a "
             "busy backend degrades instead of aborting. FGP needs its own VCS license "
             "feature and this flow already has seat pressure, so verify a known-good "
             "workload before trusting a measurement taken with it.")
    args = parser.parse_args()
    if args.max_sim_jobs < 1:
        parser.error("--max-sim-jobs must be >= 1")
    if args.sim_cores < 1:
        parser.error("--sim-cores must be >= 1")

    task_yaml = resolve_task_yaml(args.task_yaml)
    if not task_yaml.exists():
        raise FileNotFoundError(f"Task YAML file {task_yaml} does not exist")
    print(f"Using task list: {task_yaml}")

    runner = HeMAiASimRunner(
        repo_root=repo_root,
        output_dir=script.parent,
        engine=args.engine,
        with_waveform=bool(args.waveform),
        cfg=args.cfg,
        sim_cfg=sim_cfg,
        with_macro=False,
        with_d2d=False,
        with_pll=False,
        # The SW fleet build (~96 host apps) dominates a sweep's setup and is -j-safe;
        # `rtl`/`bootrom` are not, and stay serial.
        build_jobs=os.cpu_count(),
        max_jobs=args.max_sim_jobs,
        vcs_sim_cores_per_job=args.sim_cores,
        skip_setup=args.sw_only or args.reuse_build,
        skip_build=args.sw_only or args.reuse_build,
        skip_compile=args.sw_only,
    )
    runner.run(parse_tasks(task_yaml))
