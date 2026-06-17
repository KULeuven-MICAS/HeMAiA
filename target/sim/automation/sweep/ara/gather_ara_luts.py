#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Gather per-kernel Ara (CVA6+Ara FP32) cycle measurements into a single CSV.
#
# After `target/sim/automation/sweep/ara/run_ara_sweep.py` finishes, each
# ara_<kernel> workload has run in its own ara/task_<idx>/bin/ dir and printed
#   CYCLES,<kernel>,<N>,<rep>,<cycles>
# over UART (uart_chip_0_0.log).  This script:
#   1. parses those CYCLES lines from each task's UART log,
#   2. averages cycles per (kernel, N) over reps,
#   3. writes one consolidated CSV (op_name,op_node,n,cycles).
#
# This stays at the measurement level only — curve fitting lives in the bingo
# framework, which consumes this CSV.  op_node is the HeMAiA kernel symbol that
# produced each measurement (adjust OP_SPEC if a kernel is renamed).

import argparse
import csv
import glob
import os
import re
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, "../../../../../"))
_ARA_DIR = _THIS
_DEFAULT_OUT = os.path.join(_ARA_DIR, "ara_cycles.csv")

# kernel name (as printed in the CYCLES line) -> bingo op_node fn name.
# All but dequantize are __host_bingo_kernel_fp32_<name>.
_FP32 = [
    "add", "sub", "mul", "div", "max", "min", "silu_mul",
    "exp", "sigmoid", "sqrt", "relu", "neg", "abs", "tanh", "reciprocal",
    "silu", "gelu", "reduce_sum", "reduce_max", "reduce_mean",
    "quantize", "softmax", "rmsnorm",
]
OP_SPEC = {k: f"__host_bingo_kernel_fp32_{k}" for k in _FP32}
OP_SPEC["dequantize"] = "__host_bingo_kernel_int32_dequantize"

_CYCLES_RE = re.compile(r"CYCLES,([^,]+),(\d+),(\d+),(\d+)")


def _parse_task_order(task_yaml):
    """Return the ordered workload names from the task YAML (= task_<idx>)."""
    sys.path.insert(0, os.path.join(_ROOT, "util/automation_scripts"))
    from hemaia_sim_runner import parse_tasks  # noqa E402  (no PyYAML needed)
    from pathlib import Path
    return [t["workload"] for t in parse_tasks(Path(task_yaml))]


def _parse_cycles(uart_log):
    """kernel -> {N: [cycles, ...]} from the CYCLES lines in a UART log."""
    out = {}
    with open(uart_log, errors="replace") as f:
        for line in f:
            m = _CYCLES_RE.search(line)
            if not m:
                continue
            kernel, n, _rep, cyc = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
            out.setdefault(kernel, {}).setdefault(n, []).append(cyc)
    return out


def gather_all(ci_dir, out_csv, only=None, verbose=True):
    # Aggregate CYCLES across every task UART log, keyed by the kernel name in
    # the CSV itself (so the result is independent of task ordering).
    agg = {}  # kernel -> {N: [cycles,...]}
    logs = sorted(glob.glob(os.path.join(ci_dir, "task_*", "bin", "uart_chip_0_0.log")))
    if not logs:
        print(f"No UART logs found under {ci_dir}/task_*/bin/uart_chip_0_0.log")
        return 0
    for log in logs:
        for kernel, by_n in _parse_cycles(log).items():
            dst = agg.setdefault(kernel, {})
            for n, cycs in by_n.items():
                dst.setdefault(n, []).extend(cycs)

    rows = []
    for kernel in sorted(agg):
        if only and kernel not in only:
            continue
        if kernel not in OP_SPEC:
            print(f"{kernel}: no OP_SPEC entry (skipped)")
            continue
        op_node = OP_SPEC[kernel]
        # Average cycles per N once, then reuse for both the CSV rows and the log.
        points = [(n, round(sum(cycs) / len(cycs)))
                  for n, cycs in sorted(agg[kernel].items())]
        for n, cyc in points:
            rows.append({"op_name": kernel, "op_node": op_node, "n": n, "cycles": cyc})
        if verbose:
            pts = ", ".join(f"n={n}:{cyc}" for n, cyc in points)
            print(f"{kernel}: {len(points)} points  [{pts}]")

    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["op_name", "op_node", "n", "cycles"])
            w.writeheader()
            w.writerows(rows)

    missing = [k for k in OP_SPEC if k not in agg]
    if missing:
        print(f"\nNo CYCLES data for: {', '.join(sorted(missing))}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-yaml", default=os.path.join(_ARA_DIR, "task_ara.yaml"),
                    help="only used to report the expected task order")
    ap.add_argument("--ci-dir", default=_ARA_DIR,
                    help="dir holding the task_<idx> run dirs (default: ara sweep dir)")
    ap.add_argument("--out", default=_DEFAULT_OUT,
                    help="output CSV path (default: %(default)s)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these kernel names")
    args = ap.parse_args()

    if os.path.exists(args.task_yaml):
        order = _parse_task_order(args.task_yaml)
        print(f"Expected {len(order)} ara tasks from {os.path.basename(args.task_yaml)}")

    n = gather_all(args.ci_dir, args.out, only=args.only)
    print(f"\nWrote {n} row(s) to {args.out}")


if __name__ == "__main__":
    main()
