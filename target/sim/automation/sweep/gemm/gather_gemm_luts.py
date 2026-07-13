#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Gather the VersaCore GEMM cycle measurements into a CSV, ONE ROW PER RUNNABLE KERNEL.
#
# A GEMM kernel id is a precision MODE plus a runtime ARRAY SHAPE, and each of those has a
# REUSE ("_minimal") sibling: 6 modes x 3 shapes x 2 = 36 -- exactly the 36 op files in the
# framework's lut/gemm/. This script emits that id, not a single generic "gemm_full", so a
# measurement can never be attributed to the wrong shape or to the wrong side of the
# configure/reuse split (the whole point of baking the shape into the kernel id).
#
# Each gemm_psweep_<mode>_1cluster task sweeps the shared (M,K,N,array_shape) grid in one
# sim. Per config the DFG runs the CONFIGURE kernel and then its REUSE kernel on the same
# operands, and the device kernels bracket themselves with magic-NOP markers
# (perf_tracing.h):
#
#     configure : BINGO_TRACE_GEMM_FULL_CFG_*   BINGO_TRACE_GEMM_FULL_RUN_*
#     reuse     : BINGO_TRACE_GEMM_MIN_CFG_*    BINGO_TRACE_GEMM_MIN_RUN_*
#
# RUN is the size-dependent compute -> the LUT's op_latency_lut_points(m,k,n).
# CFG is the fixed per-call programming overhead -> the LUT's op_latency_fixed. That split
# is the reason the reuse kernel exists at all (it skips the mode/shape CSR writes), so the
# sweep has to report both or the reuse curve is meaningless.
#
# Per task: convert the cluster-core traces, run bingo_trace.py, read the ORDERED marker
# streams, and pair the Nth value of each with configs.json[N] (which carries the mesh and
# the op ids, written by gemm_psweep_lib so the sweep and the LUT cannot disagree).
#
# Output CSV: op_name,op_node,prec,array_shape,mesh,variant,M,K,N,cycles,cfg_cycles

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

MINIMAL_KERNEL = "__snax_bingo_kernel_gemm_minimal"
# precision -> device wrapper symbol that produced the CONFIGURE measurement.
PREC_KERNEL = {
    "i8i8_i32": "__snax_bingo_kernel_gemm_i8i8_i32",
    "i8i4_i32": "__snax_bingo_kernel_gemm_i8i4_i32",
    "i4i4_i32": "__snax_bingo_kernel_gemm_i4i4_i32",
    "i8i8_i8":  "__snax_bingo_kernel_gemm_i8i8_i8",
    "i8i4_f16": "__snax_bingo_kernel_gemm_i8i4_f16",
    "i8i8_f16": "__snax_bingo_kernel_gemm_i8i8_f16",
}

FIELDS = ["op_name", "op_node", "prec", "array_shape", "mesh", "variant",
          "M", "K", "N", "cycles", "cfg_cycles"]


def _pair(stream, configs, what, workload, verbose):
    """Zip an ordered marker stream with the config list, complaining on a mismatch."""
    if len(stream) != len(configs):
        print(f"  !! {workload}: {len(stream)} {what} events vs {len(configs)} configs "
              f"(pairing the first {min(len(stream), len(configs))})")
    return stream


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
    meshes = {int(k): tuple(v) for k, v in cj.get("meshes", {}).items()}
    op_ids = cj.get("op_ids")
    op_ids_min = cj.get("op_ids_minimal")
    if not op_ids or not op_ids_min:
        print(f"task_{idx} {workload}: configs.json has no op_ids "
              f"(stale -- rebuild the workload); skipping")
        return []

    convert_traces(logs_dir, verbose)
    bingo_json = run_bingo_trace(logs_dir)
    if not bingo_json:
        return []

    full_run = _pair(extract_run_cycles(bingo_json, "GEMM_FULL_RUN"), configs,
                     "GEMM_FULL_RUN", workload, verbose)
    full_cfg = _pair(extract_run_cycles(bingo_json, "GEMM_FULL_CFG"), configs,
                     "GEMM_FULL_CFG", workload, verbose)
    min_run = _pair(extract_run_cycles(bingo_json, "GEMM_MIN_RUN"), configs,
                    "GEMM_MIN_RUN", workload, verbose)
    min_cfg = _pair(extract_run_cycles(bingo_json, "GEMM_MIN_CFG"), configs,
                    "GEMM_MIN_CFG", workload, verbose)

    rows = []
    for i, (M, K, N, shape) in enumerate(configs):
        mesh = meshes.get(shape, ())
        mesh_s = "x".join(str(x) for x in mesh)
        if i < len(full_run):
            rows.append({
                "op_name": op_ids[i], "op_node": PREC_KERNEL.get(prec, "?"), "prec": prec,
                "array_shape": shape, "mesh": mesh_s, "variant": "configure",
                "M": M, "K": K, "N": N,
                "cycles": full_run[i],
                "cfg_cycles": full_cfg[i] if i < len(full_cfg) else "",
            })
        if i < len(min_run):
            rows.append({
                "op_name": op_ids_min[i], "op_node": MINIMAL_KERNEL, "prec": prec,
                "array_shape": shape, "mesh": mesh_s, "variant": "minimal",
                "M": M, "K": K, "N": N,
                "cycles": min_run[i],
                "cfg_cycles": min_cfg[i] if i < len(min_cfg) else "",
            })
    if verbose:
        ids = sorted({r["op_name"] for r in rows})
        print(f"task_{idx} {workload} [{prec}]: {len(rows)} rows over {len(ids)} op id(s)")
        for oid in ids:
            pts = [r for r in rows if r["op_name"] == oid]
            cfgs = {r["cfg_cycles"] for r in pts}
            rng = f"{min(int(p['cycles']) for p in pts)}..{max(int(p['cycles']) for p in pts)}"
            print(f"    {oid:34} {len(pts):2} pts  run={rng:14} cfg={sorted(cfgs)}")
    return rows


def gather_all(task_yaml, ci_dir, out_csv, verbose=True):
    order = parse_task_order(task_yaml)
    print(f"Expected {len(order)} gemm tasks from {os.path.basename(task_yaml)}")
    rows = []
    for idx, workload in enumerate(order):
        rows.extend(gather_one(workload, idx, ci_dir, verbose))
    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    ids = sorted({r["op_name"] for r in rows})
    print(f"\nWrote {len(rows)} row(s) over {len(ids)}/36 op ids to {out_csv}")
    missing = 36 - len(ids)
    if missing:
        print(f"NOTE: {missing} of the 36 kernel ids have no measurement -- see the "
              f"D-extension note in gemm_psweep_lib (narrow-output modes cannot emit the "
              f"meshRow=1 shape).")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-yaml", default=os.path.join(_THIS, "task_gemm.yaml"))
    ap.add_argument("--ci-dir", default=_THIS,
                    help="dir holding the task_<idx> run dirs (default: this dir)")
    ap.add_argument("--out", default=_DEFAULT_OUT, help="output CSV (default: %(default)s)")
    args = ap.parse_args()
    gather_all(args.task_yaml, args.ci_dir, args.out)


if __name__ == "__main__":
    main()
