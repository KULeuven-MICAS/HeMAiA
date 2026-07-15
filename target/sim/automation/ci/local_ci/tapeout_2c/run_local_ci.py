#!/usr/bin/env python3
"""
HeMAiA CI runner -- ``hemaia_tapeout_2c`` suite
===============================================

Builds and runs the ``task_local_ci.yaml`` next to this script through the shared
:class:`HeMAiASimRunner`, against ``target/rtl/cfg/hemaia_tapeout_2c.hjson``:
4 chiplets, 2x ``snax_versacore_to_256KB_cluster`` per chiplet.

One of four per-cfg suites under ``ci/local_ci/``. Each subdirectory pairs one
tapeout RTL cfg with the task list valid for it, and owns its own ``task_<idx>/``
run directories, so the suites do not clobber each other:

    tapeout/          hemaia_tapeout.hjson           1x versacore
    tapeout_1c_simd/  hemaia_tapeout_1c_simd.hjson   1x versacore+SIMD
    tapeout_2c/       hemaia_tapeout_2c.hjson        2x versacore 256KB
    tapeout_2c_simd/  hemaia_tapeout_2c_simd.hjson   2x versacore 256KB+SIMD

The runner sets ``use_vendor_pll: false`` and a 16 MiB ``spm_wide`` in the cfg
before building (see ``resolve_rtl_cfg``); the patched copy lands in
``target/rtl/cfg/generated/``.

Defaults to the VCS engine with no waveform (fast batch runs); pass
``--engine vsim --waveform 1`` to reproduce a failing task under Questasim.

    python3 run_local_ci.py [-j JOBS] [-f task.yaml] [--cfg CFG] [--engine vcs|vsim]
                            [--waveform 0|1] [--no-d2d] [--no-macro]
                            [--spm-wide-size BYTES]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT.parents[6]  # target/sim/automation/ci/local_ci/<suite> -> repo root
sys.path.insert(0, str(_REPO_ROOT / "util" / "automation_scripts"))

from hemaia_sim_runner import (  # noqa: E402
    DEFAULT_MAX_SIM_JOBS, ENGINES, HeMAiASimRunner, parse_tasks, resolve_task_yaml,
)

CI_CFG = "target/rtl/cfg/hemaia_tapeout_2c.hjson"
SIM_CFG = "target/sim/cfg/sim_rtl.hjson"
DEFAULT_TASK_YAML = "task_local_ci.yaml"
# The tapeout's real 128 kiB spm_wide is too small for parts of the SW tree
# (a ~2 MiB .devicebin overflows WIDE_SPM at link time). Give the simulation a
# 16 MiB scratchpad instead; pass --spm-wide-size 0 to run the real 128 kiB.
DEFAULT_SPM_WIDE_SIZE = 16 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run the HeMAiA CI suite.")
    parser.add_argument(
        "-j", "--max-sim-jobs", type=int, default=DEFAULT_MAX_SIM_JOBS,
        help=f"max simulations to run in parallel (default: {DEFAULT_MAX_SIM_JOBS})")
    parser.add_argument(
        "-f", "--task-yaml", default=_SCRIPT.parent / DEFAULT_TASK_YAML,
        help=f"task list YAML (default: {DEFAULT_TASK_YAML} next to this script).")
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
    parser.add_argument(
        "--spm-wide-size", type=int, default=DEFAULT_SPM_WIDE_SIZE,
        help="spm_wide.length in bytes, patched into the cfg (default: %(default)s = 16 MiB). "
             "Pass 0 to keep the cfg's own value (the tapeout's real 128 kiB).")
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
        cfg=args.cfg,
        sim_cfg=SIM_CFG,
        with_macro=args.macro,
        with_d2d=args.d2d,
        with_pll=False,
        spm_wide_size=args.spm_wide_size or None,
        max_jobs=args.max_sim_jobs,
    )
    runner.run(parse_tasks(task_yaml))


if __name__ == "__main__":
    main()
