#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Gather per-(precision, shape) VersaCore GEMM cycle measurements into a CSV.
#
# Unlike the Ara sweep (which prints CYCLES over UART), the device GEMM kernel
# carries no UART timing -- it brackets its compute with the magic-NOP markers
# BINGO_TRACE_GEMM_FULL_RUN_START/END (perf_tracing.h, enabled by default via
# -DBINGO_PERF_TRACING).  run_gemm_sweep.py runs ONE sim per config and stages
# that run's instruction traces + the config into gemm/task_<idx>/.  For each
# task this script (mirroring gather_xdma_luts.py):
#   1. converts the cluster-core traces (.dasm -> .txt via spike-dasm |
#      util/trace/gen_trace.py --permissive),
#   2. runs util/bingo_trace/bingo_trace.py to emit bingo_trace.json,
#   3. reads the GEMM_FULL_RUN dur_cc (one compute per config; summed if >1),
#   4. pairs it with task_<idx>/config.json (the precision suite + M/K/N/shape),
#   5. writes one CSV: op_name,op_node,prec,array_shape,M,K,N,cycles.
#
# The `prec` column is the GEMM_PRECISIONS suite name (base_int8/int4_b/int4_ab/
# quantized/int4_b_int32_to_fp16) -- the same precision axis the Ara LUT carries,
# so the co-design / curve-fitting step can pair the two engines.

import argparse
import csv
import glob
import json
import os
import subprocess

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, "../../../../../"))
_GEN_TRACE = os.path.join(_ROOT, "util/trace/gen_trace.py")
_BINGO_TRACE = os.path.join(_ROOT, "util/bingo_trace/bingo_trace.py")
_PERF_HEADER = os.path.join(_ROOT, "target/sw/shared/runtime/perf_tracing.h")
_DEFAULT_OUT = os.path.join(_THIS, "gemm_cycles.csv")

OP_NAME = "gemm_full"
OP_NODE = "__snax_bingo_kernel_gemm_full"
SHAPE_FIELDS = ["array_shape", "M", "K", "N"]


def _convert_traces(logs_dir, verbose=True):
    """spike-dasm | gen_trace.py --permissive  for every .dasm lacking a .txt."""
    made = 0
    for dasm in sorted(glob.glob(os.path.join(logs_dir, "trace_chip_*_hart_*.dasm"))):
        txt = dasm[:-len(".dasm")] + ".txt"
        if os.path.exists(txt) and os.path.getmtime(txt) >= os.path.getmtime(dasm):
            continue
        cmd = f"spike-dasm < {dasm!r} | python3 {_GEN_TRACE!r} --permissive > {txt!r}"
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ! gen_trace failed on {os.path.basename(dasm)}: "
                  f"{r.stderr.strip()[:200]}")
            continue
        made += 1
    if verbose:
        print(f"  converted {made} .dasm -> .txt in {logs_dir}")


def _run_bingo_trace(logs_dir):
    out = os.path.join(logs_dir, "bingo_trace.json")
    r = subprocess.run(
        ["python3", _BINGO_TRACE, "--trace-header", _PERF_HEADER,
         "--log-dir", logs_dir, "--output", out],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ! bingo_trace failed: {r.stderr.strip()[:300]}")
        return None
    return out


def gemm_run_cycles(bingo_json):
    """Sum of GEMM_FULL_RUN dur_cc events (one GEMM compute per config sim)."""
    with open(bingo_json) as f:
        trace = json.load(f)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    runs = [e for e in events
            if e.get("ph") == "X" and "GEMM_FULL_RUN" in str(e.get("name", ""))]
    return sum(int(e.get("args", {}).get("dur_cc", 0)) for e in runs), len(runs)


def gather_all(sweep_dir, out_csv, verbose=True):
    rows = []
    task_dirs = sorted(glob.glob(os.path.join(sweep_dir, "task_*")))
    if not task_dirs:
        print(f"No task_* dirs under {sweep_dir}")
        return 0
    for task in task_dirs:
        idx = os.path.basename(task)
        logs_dir = os.path.join(task, "bin", "logs")
        cfg_path = os.path.join(task, "config.json")
        if not os.path.isdir(logs_dir) or not os.path.exists(cfg_path):
            print(f"{idx}: missing logs dir or config.json (skipped)")
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        _convert_traces(logs_dir, verbose)
        bingo_json = _run_bingo_trace(logs_dir)
        if not bingo_json:
            continue
        cyc, n_events = gemm_run_cycles(bingo_json)
        if n_events == 0:
            print(f"{idx} [{cfg.get('suite')}]: no GEMM_FULL_RUN event (skipped)")
            continue
        if n_events > 1:
            print(f"{idx} [{cfg.get('suite')}]: {n_events} GEMM_FULL_RUN events (summed)")
        row = {"op_name": OP_NAME, "op_node": OP_NODE,
               "prec": cfg.get("suite", ""), "cycles": cyc}
        for k in SHAPE_FIELDS:
            row[k] = cfg.get(k, "")
        rows.append(row)
        if verbose:
            shp = " ".join(f"{k}={cfg.get(k)}" for k in SHAPE_FIELDS)
            print(f"{idx} [{row['prec']}] {shp}: {cyc} cc")

    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        fields = ["op_name", "op_node", "prec"] + SHAPE_FIELDS + ["cycles"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", default=_THIS,
                    help="dir holding the task_<idx> run dirs (default: this dir)")
    ap.add_argument("--out", default=_DEFAULT_OUT, help="output CSV (default: %(default)s)")
    args = ap.parse_args()
    n = gather_all(args.sweep_dir, args.out)
    print(f"\nWrote {n} row(s) to {args.out}")


if __name__ == "__main__":
    main()
