#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""
HeMAiA VersaCore GEMM precision LUT sweep runner (one sim per config).

Mirrors the xDMA sweep, but GEMM precision changes the generated data layout
(int4 packing / quantized / fp16 output), so precision cannot be looped inside
one binary -- each (precision, shape) config is built and simulated on its own,
exactly like target/sim/automation/test/versacore/testing_launch.py (whose
build/sim machinery this reuses). Each run's instruction traces + config are
staged into gemm/task_<idx>/; afterwards:

    python3 gather_gemm_luts.py        # -> gemm_cycles.csv (GEMM_FULL_RUN dur_cc)

The (precision, shape) configs come from the GEMM_PRECISIONS contract in
testing_workload_gen.py (all 5 suites by default). Tracing is on by default
(-DBINGO_PERF_TRACING via `make apps`), so the GEMM_FULL_RUN markers are emitted.

Usage:
    python3 run_gemm_sweep.py --first-run            # one-time RTL+VCS build
    python3 run_gemm_sweep.py [--max-cases N] [--suites base_int8 int4_b ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT.parents[5]  # target/sim/automation/sweep/gemm -> repo root
_VERSACORE = _REPO_ROOT / "target/sim/automation/test/versacore"
sys.path.insert(0, str(_VERSACORE))

# Reuse the per-config build/sim machinery + the precision contract.
import testing_launch as tl            # noqa: E402
import testing_workload_gen as twg     # noqa: E402

STAGED_LOGS = _REPO_ROOT / "target/sim/bin/logs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-cases", type=int, default=16,
                   help="random sample size across all suites (0 = no cap).")
    p.add_argument("--suites", nargs="*", choices=list(twg.GEMM_PRECISIONS),
                   default=None, help="precision suites (default: all).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--timeout", type=float, default=None,
                   help="per-sim timeout in seconds.")
    p.add_argument("--first-run", action="store_true",
                   help="run the one-time RTL+VCS build before the sweep.")
    p.add_argument("--out-dir", type=Path, default=_SCRIPT.parent,
                   help="dir to hold task_<idx>/ run dirs (default: this dir).")
    return p.parse_args()


def select_configs(suites, max_cases, seed):
    hwcfg = twg.load_hwcfg(twg.find_default_hwcfg())
    if max_cases:
        rows, total = twg.sample_workloads(
            hwcfg, suites=suites,
            l1_memory_limit=twg.L1_MEMORY_LIMIT_BYTES,
            l3_memory_limit=twg.L3_MEMORY_LIMIT_BYTES,
            max_cases=max_cases, seed=seed)
        print(f"Sampled {len(rows)} of {total} (precision, shape) configs")
        return rows
    rows = list(twg.iter_workloads(
        hwcfg, suites=suites,
        l1_memory_limit=twg.L1_MEMORY_LIMIT_BYTES,
        l3_memory_limit=twg.L3_MEMORY_LIMIT_BYTES))
    print(f"Selected {len(rows)} (precision, shape) configs")
    return rows


def main() -> None:
    args = parse_args()
    if args.first_run:
        tl.first_run_setup()

    configs = select_configs(args.suites, args.max_cases, args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(configs):
        task = out_dir / f"task_{idx}"
        task.mkdir(parents=True, exist_ok=True)
        params = tl.row_to_params(row)
        print(f"\n=== task_{idx}: {row.get('test_name')} ({row.get('suite')}) ===")

        status, rc, timed_out, elapsed = tl.run_one_case(
            params, log_path=task / "run.log", timeout=args.timeout)
        print(f"  status={status} rc={rc} timed_out={timed_out} {elapsed:.1f}s")

        # Stage this run's instruction traces so gather_gemm_luts.py can read them.
        dst_logs = task / "bin" / "logs"
        if dst_logs.exists():
            shutil.rmtree(dst_logs)
        if STAGED_LOGS.exists():
            shutil.copytree(STAGED_LOGS, dst_logs)
        else:
            print(f"  ! no trace logs at {STAGED_LOGS} (was PERF_TRACING on?)")
        # Save the config so the gather step can pair cycles with (suite, M,K,N).
        with (task / "config.json").open("w") as f:
            json.dump({**row, "status": status}, f, indent=2)

    print(f"\nRan {len(configs)} configs into {out_dir}/task_*/. "
          f"Now: python3 {_SCRIPT.parent / 'gather_gemm_luts.py'}")


if __name__ == "__main__":
    main()
