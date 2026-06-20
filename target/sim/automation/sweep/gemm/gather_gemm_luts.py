#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Gather per-(precision, shape) VersaCore GEMM cycle measurements into a CSV.
#
# After run_gemm_sweep.py finishes, each gemm_psweep_<prec>_1cluster task has run
# in its own gemm/task_<idx>/bin/ dir. The device GEMM kernels carry no UART
# timing -- they bracket compute with the magic-NOP markers
# BINGO_TRACE_GEMM_FULL_RUN_START/END (perf_tracing.h, on by default). Each task
# sweeps the shared (M,K,N,array_shape) grid in one sim, emitting one
# GEMM_FULL_RUN event per config (in CONFIGS order). For every task this script
# (mirroring gather_xdma_luts.py):
#   1. converts the cluster-core traces (.dasm -> .txt via spike-dasm |
#      util/trace/gen_trace.py --permissive),
#   2. runs util/bingo_trace/bingo_trace.py to emit bingo_trace.json,
#   3. reads the ordered GEMM_FULL_RUN dur_cc values,
#   4. pairs the Nth value with the workload's configs.json[N] = (M,K,N,shape)
#      and the precision (from configs.json),
#   5. writes one CSV: op_name,op_node,prec,array_shape,M,K,N,cycles.
#
# The `prec` column is the GEMM precision (i8i8_i32 / i8i4_i32 / i4i4_i32 /
# i8i8_i8 / i8i4_f16) -- the axis the co-design / curve-fitting step pairs with
# the Ara LUT. task_<idx> -> workload mapping follows task_gemm.yaml order.

import argparse
import csv
import json
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, "../../../../../"))
sys.path.insert(0, os.path.join(_ROOT, "util/automation_scripts"))
from bingo_trace_gather import (  # noqa: E402
    convert_traces, extract_run_cycles, parse_task_order, run_bingo_trace,
)

_WORKLOADS = os.path.join(
    _ROOT, "target/sw/host/apps/offload_bingo_hw/single_chip/workloads")
_DEFAULT_OUT = os.path.join(_THIS, "gemm_cycles.csv")

OP_NAME = "gemm_full"
# precision -> device wrapper symbol that produced the measurement.
PREC_KERNEL = {
    "i8i8_i32": "__snax_bingo_kernel_gemm_i8i8_i32",
    "i8i4_i32": "__snax_bingo_kernel_gemm_i8i4_i32",
    "i4i4_i32": "__snax_bingo_kernel_gemm_i4i4_i32",
    "i8i8_i8":  "__snax_bingo_kernel_gemm_i8i8_i8",
    "i8i4_f16": "__snax_bingo_kernel_gemm_i8i4_f16",
}


def gather_one(workload, idx, ci_dir, verbose=True):
    logs_dir = os.path.join(ci_dir, f"task_{idx}", "bin", "logs")
    cfg_path = os.path.join(_WORKLOADS, workload, "configs.json")
    if not os.path.isdir(logs_dir):
        print(f"task_{idx} {workload}: MISSING logs dir {logs_dir}")
        return []
    if not os.path.exists(cfg_path):
        print(f"task_{idx} {workload}: MISSING configs.json {cfg_path}")
        return []
    with open(cfg_path) as f:
        cj = json.load(f)
    prec = cj.get("precision", "?")
    configs = cj["configs"]
    op_node = PREC_KERNEL.get(prec, f"__snax_bingo_kernel_gemm_{prec}")

    convert_traces(logs_dir, verbose)
    bingo_json = run_bingo_trace(logs_dir)
    if not bingo_json:
        return []
    cycles = extract_run_cycles(bingo_json, "GEMM_FULL_RUN")
    if len(cycles) != len(configs):
        print(f"task_{idx} {workload}: {len(cycles)} GEMM_FULL_RUN events vs "
              f"{len(configs)} configs (pairing first {min(len(cycles), len(configs))})")

    rows = []
    for (M, K, N, shape), cyc in zip(configs, cycles):
        rows.append({"op_name": OP_NAME, "op_node": op_node, "prec": prec,
                     "array_shape": shape, "M": M, "K": K, "N": N, "cycles": cyc})
    if verbose:
        pts = ", ".join(f"({r['M']},{r['K']},{r['N']},s{r['array_shape']}):{r['cycles']}"
                        for r in rows)
        print(f"task_{idx} {workload} [{prec}]: {len(rows)} points  [{pts}]")
    return rows


def gather_all(task_yaml, ci_dir, out_csv, verbose=True):
    order = parse_task_order(task_yaml)
    print(f"Expected {len(order)} gemm tasks from {os.path.basename(task_yaml)}")
    rows = []
    for idx, workload in enumerate(order):
        rows.extend(gather_one(workload, idx, ci_dir, verbose))
    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        fields = ["op_name", "op_node", "prec", "array_shape", "M", "K", "N", "cycles"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-yaml", default=os.path.join(_THIS, "task_gemm.yaml"))
    ap.add_argument("--ci-dir", default=_THIS,
                    help="dir holding the task_<idx> run dirs (default: this dir)")
    ap.add_argument("--out", default=_DEFAULT_OUT, help="output CSV (default: %(default)s)")
    args = ap.parse_args()
    n = gather_all(args.task_yaml, args.ci_dir, args.out)
    print(f"\nWrote {n} row(s) to {args.out}")


if __name__ == "__main__":
    main()
