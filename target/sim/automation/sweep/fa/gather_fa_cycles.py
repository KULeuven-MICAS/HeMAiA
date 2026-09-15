#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Read the FlashAttention run's trace and report the same numbers the reference
# snax-flashattn app prints for itself, so the two are comparable line for line.
#
# WHAT COMES OUT, and why each one:
#
#   S^T = K.Q^T        GEMM cycles for the score matmul, and cycles per ARRAY PASS.
#   O^T = V^T.P^T      the same for the O matmul.
#
#       One pass is Mu*Ku*Nu = 1024 MAC and a dispatch is M*N*K of them, so cycles/pass
#       is the engine's own efficiency with the tile shape and the streamer factored out.
#       The arithmetic floor is 1.00. The reference measures 2.00 on both shapes and
#       attributes it to the array's `muls_out_data -|> tree.io.in`, a 1-entry queue with
#       no pipe bypass, which accepts on alternate cycles however ready the consumer is.
#       Reading 2.00 here too means the port reproduces the reference's behaviour; a
#       LOWER number would mean a dispatch was measured while it had not started.
#
#   softmax            SIMD cycles per KV tile: all eleven tasks plus the drain.
#   pipeline           first GEMM start to last GEMM end.
#   occupancy          (GEMM busy + SIMD busy) / pipeline, out of 200%.
#
#       Two engines run concurrently, so this is out of 200, not 100: above 100 means both
#       were busy at once for part of the run, which is the entire point of the design. If
#       it sits at ~100 the two engines are serialising and the skewed WAR edges in the
#       task graph are not doing their job.
#
# The BINGO port has one structural difference from the reference and it shows up here:
# the reference's GEMM core spins on a counter in TCDM, so its "peer-wait" is time inside
# the core's own loop, whereas here the manager simply does not dispatch a node until its
# edges are satisfied. Idle time is therefore between kernel spans rather than inside
# them, and the pipeline span minus the busy time is what corresponds to the reference's
# stall columns.

import argparse
import json
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, "../../../../../"))
sys.path.insert(0, os.path.join(_ROOT, "util/automation_scripts"))
from bingo_trace_gather import (  # noqa: E402
    convert_traces, parse_task_order, run_bingo_trace, task_dir,
)

_WORKLOADS = os.path.join(
    _ROOT, "target/sw/host/apps/offload_bingo_hw/single_chip/workloads")

GEMM_TID = "Cluster 0 Core 0"
SIMD_TID = "Cluster 0 Core 1"


def _events(bingo_json):
    with open(bingo_json) as f:
        trace = json.load(f)
    return trace["traceEvents"] if isinstance(trace, dict) else trace


def _spans(events, tid, needle):
    out = [e for e in events
           if e.get("ph") == "X" and needle in str(e.get("name", ""))
           and e.get("tid") == tid]
    out.sort(key=lambda e: e.get("ts", 0))
    return out


def _cc(e):
    return int(e.get("args", {}).get("dur_cc", 0))


def report(task_idx, ci_dir, cfg, verbose=True):
    tdir = task_dir(ci_dir, task_idx)
    if tdir is None:
        sys.exit(f"no run dir for task {task_idx} under {ci_dir}")
    logs = os.path.join(tdir, "bin", "logs")
    if not os.path.isdir(logs):
        sys.exit(f"no logs dir at {logs}")
    convert_traces(logs, verbose)
    bj = run_bingo_trace(logs)
    if not bj:
        sys.exit("bingo_trace failed")
    ev = _events(bj)

    nkv = cfg["nkv"]
    gemm_runs = _spans(ev, GEMM_TID, "GEMM_FULL_RUN")
    simd_runs = _spans(ev, SIMD_TID, "SIMD_RUN")
    # Configuration is reported separately from the dispatch. The reference reads the
    # accelerator's OWN performance counter, which sees neither; the RUN span here is the
    # array time plus the few cycles of the busy handshake around it, and the CFG span is
    # the ~60 constant-address csrw that program the streamer. Quoting both is what makes
    # the RUN column comparable with the reference's gemm_cycles.
    gemm_cfgs = _spans(ev, GEMM_TID, "GEMM_FULL_CFG")
    simd_cfgs = _spans(ev, SIMD_TID, "SIMD_CFG")

    if len(gemm_runs) != 2 * nkv:
        print(f"WARNING: {len(gemm_runs)} GEMM dispatches, expected {2 * nkv} "
              f"({nkv} KV tiles x 2 matmuls). The split below cannot be trusted.")
    if len(simd_runs) != nkv:
        print(f"WARNING: {len(simd_runs)} SIMD spans, expected {nkv}.")

    # QK and PV alternate on the GEMM core only after the pipeline fills; identify them by
    # DURATION instead, which is unambiguous here -- the two dispatches retire the same
    # number of array passes, so pairing by the node order in the graph is what the
    # manager decides, not us. Pair by the kernel-span containment instead.
    mgr = _spans(ev, GEMM_TID, "MGR_RUN_KERNEL")
    qk_cc, pv_cc = [], []
    for m in mgr:
        t0, t1 = m.get("ts", 0), m.get("ts", 0) + m.get("dur", 0)
        inner = [r for r in gemm_runs if t0 <= r.get("ts", 0) < t1]
        if not inner:
            continue
        name = str(m.get("args", {}).get("node_name", "")) or str(m.get("name", ""))
        (qk_cc if "QK" in name else pv_cc if "PV" in name else qk_cc).append(
            sum(_cc(r) for r in inner))

    # Fall back to strict alternation when the trace carries no node names.
    if not pv_cc and len(gemm_runs) == 2 * nkv:
        qk_cc = [_cc(g) for g in gemm_runs[0::2]]
        pv_cc = [_cc(g) for g in gemm_runs[1::2]]

    s1_passes = cfg["M"] * cfg["N"] * cfg["K"]
    s2_passes = cfg["S2_M"] * cfg["S2_N"] * cfg["S2_K"]
    gemm_busy = sum(qk_cc) + sum(pv_cc)
    simd_busy = sum(_cc(s) for s in simd_runs)

    all_runs = gemm_runs + simd_runs
    t0 = min(e.get("ts", 0) for e in all_runs)
    t1 = max(e.get("ts", 0) + e.get("dur", 0) for e in all_runs)
    # ts/dur are in the trace's own time unit; convert through the ratio the cycle counts
    # give us, so `pipeline` is in cycles like everything else.
    span_units = t1 - t0
    unit_cc = (sum(_cc(e) for e in all_runs)
               / max(1e-9, sum(e.get("dur", 0) for e in all_runs)))
    pipeline = int(span_units * unit_cc)

    def per_pass(cc, passes, n):
        if not n or not passes:
            return 0.0
        return cc / float(passes * n)

    print("\n=== FlashAttention on the four-engine cluster (BINGO port) ===")
    print(f"  tile             Br={cfg['Br']} Bc={cfg['Bc']} d={cfg['d']}, "
          f"{nkv} KV tiles, mesh {'x'.join(str(x) for x in cfg['mesh'])}")
    print(f"  S^T=K.Q^T        {sum(qk_cc):6d} cycles for {s1_passes * nkv:6d} array "
          f"passes ({per_pass(sum(qk_cc), s1_passes, nkv):.2f} cyc/pass, "
          f"{nkv * cfg['M'] * cfg['N']} blocks x {cfg['K']} accum)")
    print(f"  O^T=V^T.P^T      {sum(pv_cc):6d} cycles for {s2_passes * nkv:6d} array "
          f"passes ({per_pass(sum(pv_cc), s2_passes, nkv):.2f} cyc/pass, "
          f"{nkv * cfg['S2_M'] * cfg['S2_N']} blocks x {cfg['S2_K']} accum)")
    print(f"  softmax          {simd_busy:6d} cycles, {simd_busy // max(1, nkv)} per KV tile")
    print(f"  pipeline         {pipeline:6d} cycles, {pipeline // max(1, nkv)} per KV tile")
    print(f"  GEMM core        busy {gemm_busy:6d} ({100 * gemm_busy // max(1, pipeline):3d}%)")
    print(f"  SIMD core        busy {simd_busy:6d} ({100 * simd_busy // max(1, pipeline):3d}%)")
    print(f"  occupancy        {100 * (gemm_busy + simd_busy) // max(1, pipeline)}% "
          f"of the pipeline (200% = both engines saturated)")
    cfg_gemm = sum(_cc(e) for e in gemm_cfgs)
    cfg_simd = sum(_cc(e) for e in simd_cfgs)
    print(f"  config overhead  GEMM {cfg_gemm:5d} ({len(gemm_cfgs)} dispatches), "
          f"SIMD {cfg_simd:5d} ({len(simd_cfgs)} kernels)")
    print("\n  per-tile detail (cycles)")
    for j in range(max(len(qk_cc), len(pv_cc), len(simd_runs))):
        q = qk_cc[j] if j < len(qk_cc) else 0
        v = pv_cc[j] if j < len(pv_cc) else 0
        s = _cc(simd_runs[j]) if j < len(simd_runs) else 0
        print(f"    tile {j}:  QK {q:6d}   softmax {s:6d}   PV {v:6d}")
    return {"qk": qk_cc, "pv": pv_cc, "simd": [_cc(s) for s in simd_runs],
            "gemm_busy": gemm_busy, "simd_busy": simd_busy, "pipeline": pipeline}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-yaml", default=os.path.join(_THIS, "task_fa.yaml"))
    ap.add_argument("--ci-dir", default=_THIS)
    ap.add_argument("--out", default=os.path.join(_THIS, "fa_cycles.json"))
    args = ap.parse_args()

    order = parse_task_order(args.task_yaml)
    results = {}
    for idx, workload in enumerate(order):
        cfg_path = os.path.join(_WORKLOADS, workload, "configs.json")
        if not os.path.exists(cfg_path):
            sys.exit(f"no configs.json for {workload}; run the workload's datagen first")
        with open(cfg_path) as f:
            cfg = json.load(f)["configs"][0]
        results[workload] = report(idx, args.ci_dir, cfg)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
