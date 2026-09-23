#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Gather per-op SIMD cycle measurements into a single CSV from a parallel run.
#
# After `target/sim/automation/sweep/simd/run_simd_sweep.py` finishes, each SIMD
# sweep workload has run in its own simd/task_<idx>/bin/ dir. For every such task
# this script:
#   1. converts the cluster-core instruction traces (.dasm -> .txt via
#      spike-dasm | util/trace/gen_trace.py --permissive),
#   2. runs util/bingo_trace/bingo_trace.py to emit bingo_trace.json,
#   3. reads the SIMD core's spans and pairs them with the workload's configs.json,
#   4. derives each point's params (rows,cols | n) from the config,
#   5. writes one consolidated CSV: op_name,op_node,<union params>,cycles.
#
# TWO cost levels come out of the same run, and they answer different questions:
#
#   KERNEL  the BINGO_TRACE_MGR_RUN_KERNEL span around one kernel dispatch. This is
#           what a Bingo node costs -- argument parse, every pass, and the scalar
#           work between them. It is the number the scheduler needs.
#   PASS    the BINGO_TRACE_SIMD_RUN spans inside it, one per armed chain. This is
#           what the hardware costs, and it is what tells you whether a fused kernel
#           is limited by the operator chain or by the scalar code around it.
#
# A fused op emits a VARIABLE number of passes per config (the rows==1 fast path
# arms fewer stages than the general path), so its LUT has to be keyed on the kernel
# span; pairing passes positionally across configs would shift every later point onto
# the wrong config. The primitive ops arm exactly one chain, so for them the two
# levels measure the same work and the difference between them IS the software
# overhead.
#
# This stays at the measurement level only -- curve fitting lives in the bingo
# framework, which consumes this CSV.
#
# The task_<idx> -> workload mapping is the order of task_simd.yaml (the same order
# run_simd_sweep.py assigns task dirs).

import argparse
import csv
import json
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, "../../../../../"))
sys.path.insert(0, os.path.join(_ROOT, "util/automation_scripts"))
from bingo_trace_gather import (  # noqa: E402
    convert_traces, parse_task_order, run_bingo_trace, task_dir,
)

_SIMD_DIR = _THIS
_WORKLOADS = os.path.join(
    _ROOT, "target/sw/host/apps/offload_bingo_hw/single_chip/workloads")
_DEFAULT_OUT = os.path.join(_SIMD_DIR, "simd_cycles.csv")


# Which hart carries the SIMD operator chain -- read from the SAME generated map the
# kernels and the mini-compiler use, so this gatherer cannot name a different core than
# the one the work actually ran on and silently report an empty span. The map is
# snax-core-roles-defs.h, derived upstream from the cluster hjson and mirrored into the
# device tree by `make snax-sw-gen` (target/sw/Makefile).
#
# Falls back to the split cluster's 1 when the mirror is absent: this script is a
# post-hoc trace reader that is legitimately run against traces collected elsewhere,
# without a configured build tree. --simd-core overrides either way.
def _default_simd_core():
    header = os.path.join(
        _ROOT, "target/sw/device/runtime/snax/snax_core_roles_defs.h")
    try:
        with open(header) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and parts[0] == "#define" and parts[1] == "SNAX_CORE_SIMD":
                    return int(parts[2], 0)
    except OSError:
        pass
    return 1


DEFAULT_SIMD_CORE = _default_simd_core()
DEFAULT_CLUSTER = 0


# --- per-op spec ------------------------------------------------------------
#
# (op_name, op_node, kernels_per_config_index, key_mode)
#
# `kernels_per_config_index` is the position of this kernel within one config's
# dispatches: a workload that runs both the f16 and the int8 variant of an op emits
# two kernel spans per config, and index 0/1 picks them apart. len(list) is therefore
# the number of SIMD kernel dispatches each config makes, and the gatherer refuses to
# pair if the trace disagrees.
#
# key_mode "rows_cols" -> bilinear [rows, cols]; "n" -> linear [n], n = rows*cols.
OP_SPEC = {
    # ONE kernel per config, not two: both workloads dropped their int8 chain, because
    # keeping the f16 and i8 check chains for every config needs 48 host nodes and the
    # scheduler metadata for those does not fit the tapeout L3. The int8 variants are
    # exercised by the standalone snax apps instead.
    "simd_softmax_1cluster": [
        ("simd_softmax",     "__snax_bingo_kernel_simd_softmax_f16_f16", 0, "rows_cols"),
    ],
    "simd_rmsnorm_1cluster": [
        ("simd_rmsnorm",     "__snax_bingo_kernel_simd_rmsnorm_f16_f16", 0, "rows_cols"),
    ],
    "simd_rope_1cluster": [
        # The SIMD half only. RoPE is two nodes: the adjacent-pair swap runs on the DM
        # core (an 8-byte-word AGU cannot express a 2-byte reorder) and the rotation is
        # one fused task here. Measuring the SIMD node alone is what a SIMD cost model
        # wants -- the swap is iDMA time on another engine.
        ("simd_rope",        "__snax_bingo_kernel_simd_rope",      0, "rows_cols"),
    ],
    "simd_silu_1cluster": [
        ("simd_silu",        "__snax_bingo_kernel_simd_silu_f16_f16",    0, "n"),
    ],
    "simd_swiglu_1cluster": [
        ("simd_swiglu",      "__snax_bingo_kernel_simd_swiglu_f16_f16",  0, "n"),
    ],
    # The quantiser, element-count keyed. It is its own workload because every layer
    # that narrows to int8 dispatches it, so its cost is wanted on its own terms.
    "simd_quant_1cluster": [
        ("simd_fp16_to_int8",       "__snax_bingo_kernel_simd_fp16_to_int8",       0, "n"),
    ],
}


def _spans(bingo_json, tid):
    """(kernel_spans, pass_spans) for *tid*, each ordered by timestamp.

    A kernel span is kept only when it CONTAINS at least one SIMD_RUN pass. The SIMD
    core also runs the manager loop for anything else the DFG places on it, and on a
    four-engine cluster that is a real possibility -- containment is what identifies a
    span as a SIMD kernel without having to hardcode which node names exist.
    """
    with open(bingo_json) as f:
        trace = json.load(f)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace

    def pick(needle):
        return sorted((e for e in events
                       if e.get("ph") == "X"
                       and needle in str(e.get("name", ""))
                       and e.get("tid") == tid),
                      key=lambda e: e.get("ts", 0))

    runs = pick("SIMD_RUN")
    kernels = []
    for m in pick("MGR_RUN_KERNEL"):
        t0 = m.get("ts", 0)
        t1 = t0 + m.get("dur", 0)
        inner = [r for r in runs if t0 <= r.get("ts", 0) < t1]
        if inner:
            kernels.append((int(m.get("args", {}).get("dur_cc", 0)),
                            [int(r.get("args", {}).get("dur_cc", 0)) for r in inner]))
    return kernels


def gather_one(workload, idx, ci_dir, simd_tid, drop_warmup=True, verbose=True):
    """Return (op_name, op_node, params, points) groups for *workload*."""
    if workload not in OP_SPEC:
        if verbose:
            print(f"task_{idx} {workload}: no LUT spec (skipped)")
        return []
    specs = OP_SPEC[workload]
    kpc = len(specs)                      # SIMD kernel dispatches per config
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
    kernels = _spans(bingo_json, simd_tid)
    with open(cfg_path) as f:
        configs = json.load(f)["configs"]

    # The first kernel dispatch of a sweep pays a one-time cold start (i-cache fill +
    # the first CSR configuration) that runs ~2-3x steady state and wrecks the fit.
    if drop_warmup and len(configs) > 1 and len(kernels) >= 2 * kpc:
        configs = configs[1:]
        kernels = kernels[kpc:]

    need = len(configs) * kpc
    if len(kernels) != need:
        # Spans pair with configs POSITIONALLY. A mismatch means the pairing has
        # SHIFTED and every later point is attributed to the wrong config -- refuse
        # rather than emit silently-wrong LUTs.
        print(f"task_{idx} {workload}: ERROR {len(kernels)} SIMD kernel spans vs {need} "
              f"expected ({len(configs)} cfgs x {kpc}). Refusing to pair -- the "
              f"config<->event alignment cannot be trusted.")
        return []

    out = []
    for op_name, op_node, si, key_mode in specs:
        if key_mode == "n":
            params = ["n"]
            def key(cfg):
                return {"n": cfg["rows"] * cfg["cols"]}
        else:
            params = ["rows", "cols"]
            def key(cfg):
                return {"rows": cfg["rows"], "cols": cfg["cols"]}
        points = []
        for ci, cfg in enumerate(configs):
            kern_cc, pass_cc = kernels[ci * kpc + si]
            pt = key(cfg)
            pt["cycles"] = kern_cc
            pt["passes"] = len(pass_cc)
            pt["pass_cycles"] = sum(pass_cc)
            points.append(pt)
        print(f"task_{idx} {workload}: {op_name}: {len(points)} points")
        for pt in points:
            body = ", ".join(f"{p}={pt[p]}" for p in params)
            print(f"      {body}, cycles={pt['cycles']} "
                  f"({pt['passes']} pass(es), {pt['pass_cycles']} cc in the chain)")
        out.append((op_name, op_node, params, points))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-yaml", default=os.path.join(_SIMD_DIR, "task_simd.yaml"))
    ap.add_argument("--ci-dir", default=_SIMD_DIR,
                    help="dir holding the task_<idx> run dirs (default: simd sweep dir)")
    ap.add_argument("--out", default=_DEFAULT_OUT,
                    help="output CSV path (default: %(default)s)")
    ap.add_argument("--simd-core", type=int, default=DEFAULT_SIMD_CORE,
                    help="cluster core carrying the SIMD chain (default: %(default)s)")
    ap.add_argument("--cluster", type=int, default=DEFAULT_CLUSTER,
                    help="cluster to read (default: %(default)s)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these workload dir names")
    ap.add_argument("--keep-warmup", action="store_true",
                    help="keep the cold-start first config (default: drop it)")
    args = ap.parse_args()
    simd_tid = f"Cluster {args.cluster} Core {args.simd_core}"
    print(f"Reading SIMD spans from tid {simd_tid!r}")

    order = parse_task_order(args.task_yaml)
    rows = []
    param_cols = []  # union of param names, in first-seen order
    for idx, workload in enumerate(order):
        if args.only and workload not in args.only:
            continue
        for op_name, op_node, params, points in gather_one(
                workload, idx, args.ci_dir, simd_tid,
                drop_warmup=not args.keep_warmup):
            for p in params:
                if p not in param_cols:
                    param_cols.append(p)
            for pt in points:
                row = {"op_name": op_name, "op_node": op_node,
                       "cycles": pt["cycles"], "passes": pt["passes"],
                       "pass_cycles": pt["pass_cycles"]}
                row.update({p: pt[p] for p in params})
                rows.append(row)

    if rows:
        fieldnames = (["op_name", "op_node"] + param_cols
                      + ["cycles", "passes", "pass_cycles"])
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            w.writeheader()
            w.writerows(rows)
    print(f"\nWrote {len(rows)} row(s) to {args.out}")


if __name__ == "__main__":
    main()
