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


# --- Fused SIMD kernels: whole-kernel (MGR_RUN_KERNEL) cost keyed by (rows, cols) ---
# Unlike the single-pass shape ops (exactly one XDMA_RUN per config), the fused SIMD kernels
# run a VARIABLE number of xDMA passes per config (rows==1 fast path vs rows>1 general path),
# so their per-op cost is the manager's whole-kernel span, not a pass. Each config runs BOTH
# precisions (f16 then i8), and we emit BOTH as separate LUT ops so the int8-output cost is
# measured too (xdma_softmax gets the f16; xdma_softmax_i8 is measured for reference -- the
# emitter only fills an op that has its own LUT file, so the i8 row lands in the CSV either way).
# A spec is (op_name, op_node, span_index[, key_mode]); len(list) == XDMA-passes/config (kpc).
# key_mode "rows_cols" (default) -> bilinear [rows, cols]; "n" -> linear [n] (n = rows*cols), for
# the generic streaming passes whose LUTs are element-count keyed. Ops with no LUT file (the _i8 /
# _ref rows) still land in the CSV; the emitter discards them loudly (measured for reference).
SIMD_OP_SPEC = {
    "xdma_softmax_1cluster": [
        ("xdma_softmax",    "__snax_bingo_kernel_xdma_softmax_f16_f16", 0),
        ("xdma_softmax_i8", "__snax_bingo_kernel_xdma_softmax_f16_i8",  1),
    ],
    "xdma_rmsnorm_1cluster": [
        ("xdma_rmsnorm",    "__snax_bingo_kernel_xdma_rmsnorm_f16_f16", 0),
        ("xdma_rmsnorm_i8", "__snax_bingo_kernel_xdma_rmsnorm_f16_i8",  1),
    ],
    # rope: one fused whole-op kernel -> bilinear [rows, cols], like softmax/rmsnorm.
    "xdma_rope_1cluster": [
        ("xdma_rope",       "__snax_bingo_kernel_xdma_rope", 0),
    ],
    # silu / swiglu: one fused whole-op kernel each -> bilinear [rows, cols], like softmax/rmsnorm/rope.
    "xdma_silu_1cluster": [
        ("xdma_silu",   "__snax_bingo_kernel_xdma_silu_f16_f16", 0),
    ],
    "xdma_swiglu_1cluster": [
        ("xdma_swiglu", "__snax_bingo_kernel_xdma_swiglu_f16_f16", 0),
    ],
}


def _simd_kernel_mgr_cycles(bingo_json):
    """Ordered dur_cc of the SIMD-kernel MGR_RUN_KERNEL spans: DM-core (Cluster 0 Core 1)
    manager spans that CONTAIN an XDMA_RUN pass. Load (idma) shows IDMA_RUN, Store/Check run on
    the host core -- only the fused compute kernels wrap XDMA_RUN passes, so containment picks
    them out unambiguously regardless of how many passes each config runs."""
    with open(bingo_json) as f:
        trace = json.load(f)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    DM = "Cluster 0 Core 1"
    mgr = sorted((e for e in events if e.get("ph") == "X"
                  and "MGR_RUN_KERNEL" in str(e.get("name", "")) and e.get("tid") == DM),
                 key=lambda e: e.get("ts", 0))
    xrun = [e for e in events if e.get("ph") == "X"
            and "XDMA_RUN" in str(e.get("name", "")) and e.get("tid") == DM]
    out = []
    for m in mgr:
        t0, t1 = m.get("ts", 0), m.get("ts", 0) + m.get("dur", 0)
        if any(t0 <= x.get("ts", 0) < t1 for x in xrun):
            out.append(int(m.get("args", {}).get("dur_cc", 0)))
    return out


def gather_simd_one(workload, idx, ci_dir, drop_warmup=True, verbose=True):
    """(op_name, op_node, ['rows','cols'], points) groups for a fused SIMD workload -- one group
    per kernel invocation per config (f16 and i8)."""
    if workload not in SIMD_OP_SPEC:
        return []
    specs = SIMD_OP_SPEC[workload]
    kpc = len(specs)                      # kernel invocations per config
    tdir = task_dir(ci_dir, idx)
    if tdir is None:
        print(f"task_{idx} {workload}: MISSING run dir under {ci_dir}")
        return []
    logs_dir = os.path.join(tdir, "bin", "logs")
    cfg_path = os.path.join(_WORKLOADS, workload, "configs.json")
    if not os.path.isdir(logs_dir) or not os.path.exists(cfg_path):
        print(f"task_{idx} {workload}: MISSING logs/configs.json")
        return []
    convert_traces(logs_dir, verbose)
    bingo_json = run_bingo_trace(logs_dir)
    if not bingo_json:
        return []
    cycles = _simd_kernel_mgr_cycles(bingo_json)
    with open(cfg_path) as f:
        configs = json.load(f)["configs"]

    # Drop the cold-start first config (icache fill + first xDMA config, ~2-3x steady state).
    if drop_warmup and len(configs) > 1 and len(cycles) >= 2 * kpc:
        configs = configs[1:]
        cycles = cycles[kpc:]

    need = len(configs) * kpc
    if len(cycles) != need:
        print(f"task_{idx} {workload}: ERROR {len(cycles)} SIMD-kernel MGR spans vs {need} "
              f"expected ({len(configs)} cfgs x {kpc}). Refusing to pair.")
        return []

    out = []
    for spec in specs:
        op_name, op_node, si = spec[0], spec[1], spec[2]
        key_mode = spec[3] if len(spec) > 3 else "rows_cols"
        if key_mode == "n":
            params = ["n"]
            points = [{"n": cfg["rows"] * cfg["cols"],
                       "cycles": cycles[ci * kpc + si]} for ci, cfg in enumerate(configs)]
            desc = lambda pt: f"n={pt['n']}, cycles={pt['cycles']}"
        else:
            params = ["rows", "cols"]
            points = [{"rows": cfg["rows"], "cols": cfg["cols"],
                       "cycles": cycles[ci * kpc + si]} for ci, cfg in enumerate(configs)]
            desc = lambda pt: f"rows={pt['rows']}, cols={pt['cols']}, cycles={pt['cycles']}"
        print(f"task_{idx} {workload}: {op_name}: {len(points)} points")
        for pt in points:
            print(f"      {desc(pt)}")
        out.append((op_name, op_node, params, points))
    return out


def gather_one(workload, idx, ci_dir, mesh, drop_warmup=True, verbose=True):
    """Return a list of (op_name, op_node, params, points) groups for *workload*.

    One workload can feed SEVERAL LUTs. A converter sweeps 3 array shapes x 3 element
    widths, and each (shape, width) pair is a distinct runnable kernel with its own
    LUT -- so its 81 configs split into 9 groups. The non-layout ops yield exactly one
    group.

    The variant is read from configs.json's `op_ids` (written by xdma_ops_lib), NOT
    reconstructed here: the sweep decides which kernel each config runs, so letting
    it name the LUT is what stops the two from ever disagreeing. `meshes` likewise
    gives the per-config (meshRow, tileSize, meshCol); the single global --mesh is
    only correct for a workload that sweeps ONE shape.

    *points* is a list of dicts, each the op's params plus a "cycles" value.
    """
    if workload not in OP_SPEC:
        if verbose:
            print(f"task_{idx} {workload}: no LUT spec (skipped)")
        return []
    op_name, op_node, _op_fit, params, epc, point_fn = OP_SPEC[workload]
    tdir = task_dir(ci_dir, idx)
    if tdir is None:
        print(f"task_{idx} {workload}: MISSING run dir under {ci_dir}")
        return []
    logs_dir = os.path.join(tdir, "bin", "logs")
    cfg_path = os.path.join(_WORKLOADS, workload, "configs.json")
    if not os.path.isdir(logs_dir):
        print(f"task_{idx} {workload}: MISSING logs dir {logs_dir}")
        return []
    if not os.path.exists(cfg_path):
        print(f"task_{idx} {workload}: MISSING configs.json {cfg_path}")
        return []

    convert_traces(logs_dir, verbose)
    bingo_json = run_bingo_trace(logs_dir)
    if not bingo_json:
        return []
    cycles = extract_run_cycles(bingo_json, "XDMA_RUN")
    with open(cfg_path) as f:
        cj = json.load(f)
    configs = cj["configs"]
    op_ids = cj.get("op_ids")                      # layout workloads only
    meshes = {int(k): tuple(v) for k, v in cj.get("meshes", {}).items()}

    # The first kernel invocation of a sweep pays a one-time cold-start penalty
    # (i-cache fill + first xDMA config) that is ~10x the steady-state cost and
    # wrecks the linear/bilinear fit -- drop that first config as a warm-up.
    if drop_warmup and len(configs) > 1 and len(cycles) >= 2 * epc:
        configs = configs[1:]
        cycles = cycles[epc:]
        if op_ids:
            op_ids = op_ids[1:]

    need = len(configs) * epc
    if len(cycles) != need:
        # Events pair with configs POSITIONALLY: every config emits exactly `epc` XDMA_RUN
        # events (CPU fallbacks mark themselves too), so the counts must match exactly. A
        # mismatch means the pairing has SHIFTED and every later point is attributed to the
        # wrong config -- refuse rather than emit silently-wrong LUTs.
        print(f"task_{idx} {workload}: ERROR {len(cycles)} XDMA_RUN events vs {need} "
              f"expected ({len(configs)} cfgs x {epc}). Refusing to pair -- the "
              f"config<->event alignment cannot be trusted.")
        return []

    # group -> points
    groups = {}
    for ci, cfg in enumerate(configs):
        grp = cycles[ci * epc:(ci + 1) * epc]
        this_op = op_ids[ci] if op_ids else op_name
        this_node = f"__snax_bingo_kernel_{this_op}" if op_ids else op_node
        this_mesh = meshes.get(int(cfg.get("array_shape", 0)), mesh) if meshes else mesh
        pt = point_fn(cfg, this_mesh)
        pt["cycles"] = sum(grp)         # concat sums top+bottom; others 1 value
        groups.setdefault((this_op, this_node), []).append(pt)

    if not groups:
        print(f"task_{idx} {workload}: no usable points")
        return []

    print(f"task_{idx} {workload}: {len(configs)} configs -> {len(groups)} LUT(s)")
    out = []
    for (nm, node), pts in sorted(groups.items()):
        print(f"    {nm}: {len(pts)} points")
        for pt in pts:
            print("      " + ", ".join(f"{k}={pt[k]}" for k in (params + ["cycles"])))
        out.append((nm, node, params, pts))
    return out


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
        groups = gather_one(workload, idx, args.ci_dir, mesh,
                            drop_warmup=not args.keep_warmup)
        groups += gather_simd_one(workload, idx, args.ci_dir,
                                  drop_warmup=not args.keep_warmup)
        for op_name, op_node, params, points in groups:
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
