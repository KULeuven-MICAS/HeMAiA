#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Gather per-op xDMA cycle measurements into a single CSV from a parallel run.
#
# After `target/sim/automation/sweep/xdma/run_xdma_sweep.py` finishes, each xDMA
# sweep workload has run in its own xdma/task_<idx>/bin/ dir.
# For every such task this script:
#   1. converts the cluster-core instruction traces (.dasm -> .txt via
#      spike-dasm | util/trace/gen_trace.py --permissive),
#   2. runs util/bingo_trace/bingo_trace.py to emit bingo_trace.json,
#   3. reads the ordered BINGO_TRACE_XDMA_RUN dur_cc values and pairs them with
#      the workload's configs.json (concat emits 2 events/config -> summed),
#   4. derives each point's params (bytes | n | rows,cols) from the config,
#   5. writes one consolidated CSV: op_name,op_node,<union params>,cycles
#      (blank cells for params an op doesn't use).
#
# This stays at the measurement level only — curve fitting lives in the bingo
# framework, which consumes this CSV.
#
# The task_<idx> -> workload mapping is the order of task_vsim_xdma_lut.yaml
# (the same order run_xdma_sweep.py assigns task dirs).

import argparse
import csv
import json
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, "../../../../../"))
sys.path.insert(0, os.path.join(_ROOT, "util/automation_scripts"))
from bingo_trace_gather import (  # noqa: E402
    convert_traces, extract_run_cycles, parse_task_order, run_bingo_trace, task_dir,
)

_XDMA_DIR = _THIS
_WORKLOADS = os.path.join(
    _ROOT, "target/sw/host/apps/offload_bingo_hw/single_chip/workloads")
_DEFAULT_OUT = os.path.join(_XDMA_DIR, "xdma_cycles.csv")

# Mesh dims for the layout ops (snax_versacore_spatial_unrolling[array_shape=0]
# = [meshRow, tileSize, meshCol]). Override via --mesh if a workload uses a
# different array_shape.
DEFAULT_MESH = (32, 2, 32)


# --- per-op spec: workload dir -> LUT metadata + how a config maps to points ---
# point(cfg, mesh) returns the dict of op_parameter_list values for that config.
def _layout_a(cfg, mesh):   # row<->a : rows = M_T*meshRow, cols = K_T*tileSize
    mR, tS, mC = mesh
    return {"rows": cfg["M_T"] * mR, "cols": cfg["K_T"] * tS}


def _layout_b(cfg, mesh):   # row<->b : rows = K_T*tileSize, cols = N_T*meshCol
    mR, tS, mC = mesh
    return {"rows": cfg["K_T"] * tS, "cols": cfg["N_T"] * mC}


def _layout_d(cfg, mesh):   # row<->d : rows = M_T*meshRow, cols = N_T*meshCol
    mR, tS, mC = mesh
    return {"rows": cfg["M_T"] * mR, "cols": cfg["N_T"] * mC}


OP_SPEC = {
    # workload dir: (op_name, op_node, op_fit, params, events_per_cfg, point_fn)
    "xdma_6d_1cluster": (
        "xdma_6d", "__snax_bingo_kernel_xdma_6d", "linear", ["bytes"], 1,
        lambda c, m: {"bytes": c["rows"] * c["cols"] * c["elem_bytes"]}),
    "xdma_transpose_1cluster": (
        "xdma_transpose_2d", "__snax_bingo_kernel_xdma_transpose_2d",
        "bilinear", ["rows", "cols"], 1,
        lambda c, m: {"rows": c["rows"], "cols": c["cols"]}),
    "xdma_submatrix_1cluster": (
        "xdma_submatrix_2d", "__snax_bingo_kernel_xdma_submatrix_2d",
        "bilinear", ["rows", "cols"], 1,
        lambda c, m: {"rows": c["re"] - c["rs"], "cols": c["ce"] - c["cs"]}),
    "xdma_expand_1cluster": (
        "xdma_expand_2d", "__snax_bingo_kernel_xdma_expand_2d",
        "bilinear", ["rows", "cols"], 1,
        lambda c, m: {"rows": c["rows"], "cols": c["cols"]}),
    "xdma_concat_1cluster": (
        "xdma_concat_2d", "__snax_bingo_kernel_xdma_concat_2d",
        "bilinear", ["rows", "cols"], 2,   # top+bottom -> summed
        lambda c, m: {"rows": c["rows"], "cols": c["cols"]}),
    "xdma_pad_1cluster": (
        "xdma_pad_2d", "__snax_bingo_kernel_xdma_pad_2d",
        "bilinear", ["rows", "cols"], 1,
        lambda c, m: {"rows": c["rows"] + c.get("pt", 8) + c.get("pb", 0),
                      "cols": c["cols"] + c.get("pl", 0) + c.get("pr", 0)}),
    "xdma_gather_1cluster": (
        "xdma_gather_2d", "__snax_bingo_kernel_xdma_gather_2d",
        "bilinear", ["rows", "cols"], 1,
        lambda c, m: {"rows": c["rows"] // c.get("stride", 2), "cols": c["cols"]}),
    "xdma_elementwise_add_1cluster": (
        "xdma_elementwise_add", "__snax_bingo_kernel_xdma_elementwise_add",
        "linear", ["n"], 1,
        lambda c, m: {"n": c["per_op"]}),
    # layout conversions
    "xdma_row_to_a_1cluster": (
        "xdma_row_major_to_a", "__snax_bingo_kernel_xdma_row_major_to_a",
        "bilinear", ["rows", "cols"], 1, _layout_a),
    "xdma_a_to_row_1cluster": (
        "xdma_a_to_row_major", "__snax_bingo_kernel_xdma_a_to_row_major",
        "bilinear", ["rows", "cols"], 1, _layout_a),
    "xdma_row_to_b_1cluster": (
        "xdma_row_major_to_b", "__snax_bingo_kernel_xdma_row_major_to_b",
        "bilinear", ["rows", "cols"], 1, _layout_b),
    "xdma_b_to_row_1cluster": (
        "xdma_b_to_row_major", "__snax_bingo_kernel_xdma_b_to_row_major",
        "bilinear", ["rows", "cols"], 1, _layout_b),
    "xdma_row_to_d_1cluster": (
        "xdma_row_major_to_d", "__snax_bingo_kernel_xdma_row_major_to_d",
        "bilinear", ["rows", "cols"], 1, _layout_d),
    "xdma_d_to_row_1cluster": (
        "xdma_d_to_row_major", "__snax_bingo_kernel_xdma_d_to_row_major",
        "bilinear", ["rows", "cols"], 1, _layout_d),
}


def gather_one(workload, idx, ci_dir, mesh, drop_warmup=True, verbose=True):
    """Return (op_name, op_node, params, points) for *workload*, or None.

    *points* is a list of dicts, each the op's params plus a "cycles" value.
    (op_fit in OP_SPEC is left as in-code documentation; the CSV carries only the
    measured cycles -- curve fitting lives in the bingo framework.)
    """
    if workload not in OP_SPEC:
        if verbose:
            print(f"task_{idx} {workload}: no LUT spec (skipped)")
        return None
    op_name, op_node, _op_fit, params, epc, point_fn = OP_SPEC[workload]
    tdir = task_dir(ci_dir, idx)
    if tdir is None:
        print(f"task_{idx} {workload}: MISSING run dir under {ci_dir}")
        return None
    logs_dir = os.path.join(tdir, "bin", "logs")
    cfg_path = os.path.join(_WORKLOADS, workload, "configs.json")
    if not os.path.isdir(logs_dir):
        print(f"task_{idx} {workload}: MISSING logs dir {logs_dir}")
        return None
    if not os.path.exists(cfg_path):
        print(f"task_{idx} {workload}: MISSING configs.json {cfg_path}")
        return None

    convert_traces(logs_dir, verbose)
    bingo_json = run_bingo_trace(logs_dir)
    if not bingo_json:
        return None
    cycles = extract_run_cycles(bingo_json, "XDMA_RUN")
    with open(cfg_path) as f:
        configs = json.load(f)["configs"]

    # The first kernel invocation of a sweep pays a one-time cold-start penalty
    # (i-cache fill + first xDMA config) that is ~10x the steady-state cost and
    # wrecks the linear/bilinear fit -- drop that first config as a warm-up.
    if drop_warmup and len(configs) > 1 and len(cycles) >= 2 * epc:
        configs = configs[1:]
        cycles = cycles[epc:]

    need = len(configs) * epc
    if len(cycles) != need:
        print(f"task_{idx} {workload}: WARNING {len(cycles)} XDMA_RUN events vs "
              f"{need} expected ({len(configs)} cfgs x {epc}) — pairing first "
              f"{min(len(cycles), need)}")

    points = []
    for ci, cfg in enumerate(configs):
        grp = cycles[ci * epc:(ci + 1) * epc]
        if len(grp) < epc:
            break
        pt = point_fn(cfg, mesh)
        pt["cycles"] = sum(grp)         # concat sums top+bottom; others 1 value
        points.append(pt)

    if not points:
        print(f"task_{idx} {workload}: no usable points")
        return None

    print(f"task_{idx} {workload}: {len(points)} points")
    for pt in points:
        print("    " + ", ".join(f"{k}={pt[k]}" for k in (params + ["cycles"])))
    return op_name, op_node, params, points


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-yaml",
                    default=os.path.join(_XDMA_DIR, "task_xdma.yaml"))
    ap.add_argument("--ci-dir", default=_XDMA_DIR,
                    help="dir holding the task_<idx> run dirs (default: xdma sweep dir)")
    ap.add_argument("--out", default=_DEFAULT_OUT,
                    help="output CSV path (default: %(default)s)")
    ap.add_argument("--mesh", default=",".join(map(str, DEFAULT_MESH)),
                    help="meshRow,tileSize,meshCol (array_shape=0 default 32,2,32)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these workload dir names")
    ap.add_argument("--keep-warmup", action="store_true",
                    help="keep the cold-start first config (default: drop it)")
    args = ap.parse_args()
    mesh = tuple(int(x) for x in args.mesh.split(","))

    order = parse_task_order(args.task_yaml)
    rows = []
    param_cols = []  # union of param names, in first-seen order
    for idx, workload in enumerate(order):
        if args.only and workload not in args.only:
            continue
        res = gather_one(workload, idx, args.ci_dir, mesh,
                         drop_warmup=not args.keep_warmup)
        if not res:
            continue
        op_name, op_node, params, points = res
        for p in params:
            if p not in param_cols:
                param_cols.append(p)
        for pt in points:
            row = {"op_name": op_name, "op_node": op_node, "cycles": pt["cycles"]}
            row.update({p: pt[p] for p in params})
            rows.append(row)

    if rows:
        fieldnames = ["op_name", "op_node"] + param_cols + ["cycles"]
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            w.writeheader()
            w.writerows(rows)
    print(f"\nWrote {len(rows)} row(s) to {args.out}")


if __name__ == "__main__":
    main()
