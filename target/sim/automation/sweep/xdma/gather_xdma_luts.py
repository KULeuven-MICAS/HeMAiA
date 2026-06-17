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
import subprocess
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, "../../../../../"))
_XDMA_DIR = _THIS
_WORKLOADS = os.path.join(
    _ROOT, "target/sw/host/apps/offload_bingo_hw/single_chip/workloads")
_GEN_TRACE = os.path.join(_ROOT, "util/trace/gen_trace.py")
_BINGO_TRACE = os.path.join(_ROOT, "util/bingo_trace/bingo_trace.py")
_PERF_HEADER = os.path.join(_ROOT, "target/sw/shared/runtime/perf_tracing.h")
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


def _parse_task_order(task_yaml):
    """Return the ordered list of workload names from the task YAML (= task_<idx>)."""
    sys.path.insert(0, os.path.join(_ROOT, "util/automation_scripts"))
    from hemaia_sim_runner import parse_tasks  # noqa E402  (no PyYAML needed)
    from pathlib import Path
    tasks = parse_tasks(Path(task_yaml))
    return [t["workload"] for t in tasks]


def _convert_traces(logs_dir, verbose=True):
    """spike-dasm | gen_trace.py --permissive  for every .dasm lacking a .txt."""
    import glob
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


def _xdma_run_cycles(bingo_json):
    """Ordered list of XDMA_RUN dur_cc values (by timestamp)."""
    with open(bingo_json) as f:
        trace = json.load(f)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    runs = [e for e in events
            if e.get("ph") == "X" and "XDMA_RUN" in str(e.get("name", ""))]
    runs.sort(key=lambda e: e.get("ts", 0))
    return [int(e.get("args", {}).get("dur_cc", 0)) for e in runs]


def gather_one(workload, idx, ci_dir, mesh, drop_warmup=True, verbose=True):
    """Return (op_name, op_node, op_fit, params, points) for *workload*, or None.

    *points* is a list of dicts, each the op's params plus a "cycles" value.
    """
    if workload not in OP_SPEC:
        if verbose:
            print(f"task_{idx} {workload}: no LUT spec (skipped)")
        return None
    op_name, op_node, op_fit, params, epc, point_fn = OP_SPEC[workload]
    logs_dir = os.path.join(ci_dir, f"task_{idx}", "bin", "logs")
    cfg_path = os.path.join(_WORKLOADS, workload, "configs.json")
    if not os.path.isdir(logs_dir):
        print(f"task_{idx} {workload}: MISSING logs dir {logs_dir}")
        return None
    if not os.path.exists(cfg_path):
        print(f"task_{idx} {workload}: MISSING configs.json {cfg_path}")
        return None

    _convert_traces(logs_dir, verbose)
    bingo_json = _run_bingo_trace(logs_dir)
    if not bingo_json:
        return None
    cycles = _xdma_run_cycles(bingo_json)
    configs = json.load(open(cfg_path))["configs"]

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
    return op_name, op_node, op_fit, params, points


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-yaml",
                    default=os.path.join(_XDMA_DIR, "task_vsim_xdma_lut.yaml"))
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

    order = _parse_task_order(args.task_yaml)
    rows = []
    param_cols = []  # union of param names, in first-seen order
    for idx, workload in enumerate(order):
        if args.only and workload not in args.only:
            continue
        res = gather_one(workload, idx, args.ci_dir, mesh,
                         drop_warmup=not args.keep_warmup)
        if not res:
            continue
        op_name, op_node, _op_fit, params, points = res  # fit type is bingo's job
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
