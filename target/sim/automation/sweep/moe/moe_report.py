#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Where an MoE layer's time goes, per expert and per engine.
#
# This workload's whole point is that some lanes DO NOT RUN, so a report that only totals
# engine time answers the wrong question. What matters is:
#   - which experts ran at all (a skipped lane has no events, not zero-length events),
#   - what the winners cost, stage by stage,
#   - and how much of the makespan is nobody computing -- the bubble the router's decision
#     and the cross-cluster push open up.
#
# Each expert owns a cluster, so `tid` ("Cluster N Core M") separates the lanes for free.
# FA needed guesswork here because four shards shared a cluster; this one does not.
#
#   python3 moe_report.py [task_dir]        # default: newest task_*/ next to this file

import argparse
import collections
import json
import os
import glob
import sys

# ns per cycle. The cluster and the SoC fabric run at different rates, so a single divisor
# silently rescales every cluster figure by 2.33x. Overridable because a cfg can change them.
NS_PER_CC_CLUSTER = 12.0
NS_PER_CC_HOST = 28.0

# core index -> engine, from snax_core_roles_defs.h on snax_split_cluster. Read, not assumed:
# a cfg change moves these and a wrong map silently attributes GEMM time to the SIMD.
DEFAULT_ROLES = {0: "gemm", 1: "simd", 2: "xdma", 3: "dm"}


def load_roles():
    """The generated role map, so the engine names follow the cluster cfg."""
    here = os.path.dirname(os.path.abspath(__file__))
    hdr = os.path.normpath(os.path.join(
        here, "../../../../sw/device/runtime/snax/snax_core_roles_defs.h"))
    roles = {}
    try:
        with open(hdr) as f:
            for line in f:
                for key, eng in (("SNAX_CORE_GEMM", "gemm"), ("SNAX_CORE_SIMD", "simd"),
                                 ("SNAX_CORE_XDMA", "xdma"), ("SNAX_CORE_IDMA", "dm")):
                    if f"#define {key} " in line:
                        roles[int(line.split()[-1])] = eng
    except OSError:
        pass
    return roles or DEFAULT_ROLES


def find_trace(task_dir=None):
    here = os.path.dirname(os.path.abspath(__file__))
    if task_dir:
        cands = [os.path.join(task_dir, "bin", "logs", "bingo_trace.json"),
                 os.path.join(task_dir, "bingo_trace.json"), task_dir]
    else:
        cands = sorted(glob.glob(os.path.join(here, "task_*", "bin", "logs",
                                              "bingo_trace.json")),
                       key=os.path.getmtime, reverse=True)
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


# MGR_GET_READY is the manager POLLING for a dispatchable task -- pure waiting. On a
# dispatch-bound graph it covers nearly the whole run, so counting it as busy reports every
# engine at ~100% and hides where the time actually goes.
#
# MGR_PREP and MGR_WRITE_DONE are deliberately NOT here. They are real per-task dispatch
# cost paid on the core, and at small tile sizes that overhead IS the bottleneck -- hiding
# it would answer the wrong question.
WAIT_MARKERS = ("MGR_GET_READY",)


# Markers only a REAL accelerator kernel emits. The compiler puts an exit node on every
# core, and those still raise MGR_RUN_KERNEL, so "this cluster has events" is not the same
# as "this expert ran" -- a skipped lane still shows ~700 ns of exit at the very end.
# GEMM_FULL_*/SIMD_*/XDMA_*/IDMA_* come only from kernels that actually computed.
ENGINE_MARKERS = ("GEMM_", "SIMD_", "XDMA_", "IDMA_")


def is_work(ev):
    name = str(ev.get("name", "")).replace("BINGO_TRACE_", "")
    return not name.startswith(WAIT_MARKERS)


def is_engine(ev):
    name = str(ev.get("name", "")).replace("BINGO_TRACE_", "")
    return name.startswith(ENGINE_MARKERS)


def spans(events):
    """Total time covered by these events, MERGING overlaps.

    Summing durations double-counts whenever two markers nest (a cfg span inside a run
    span), which inflates a busy figure past the makespan and makes utilisation exceed
    100%. Merging is what makes 'busy' mean occupied-at-all.
    """
    iv = sorted((int(e["ts"]), int(e["ts"]) + int(e.get("dur", 0))) for e in events)
    total, cur_s, cur_e = 0, None, None
    for s, e in iv:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        total += cur_e - cur_s
    return total


def main():
    p = argparse.ArgumentParser(description="MoE per-expert / per-engine time report.")
    p.add_argument("task_dir", nargs="?", default=None)
    p.add_argument("--cluster-ns", type=float, default=NS_PER_CC_CLUSTER)
    p.add_argument("--host-ns", type=float, default=NS_PER_CC_HOST)
    a = p.parse_args()

    path = find_trace(a.task_dir)
    if not path:
        print("no bingo_trace.json found; run the sweep first, then "
              "`cd target/sim && make traces`")
        return 1
    print(f"trace: {path}")
    with open(path) as f:
        raw = json.load(f)
    events = raw["traceEvents"] if isinstance(raw, dict) else raw
    X = [e for e in events if e.get("ph") == "X" and "ts" in e]
    if not X:
        print("trace has no complete (X) events")
        return 1

    roles = load_roles()
    t0 = min(int(e["ts"]) for e in X)
    t1 = max(int(e["ts"]) + int(e.get("dur", 0)) for e in X)
    makespan = t1 - t0

    by_lane = collections.defaultdict(list)
    for e in X:
        by_lane[(str(e.get("pid", "?")), str(e.get("tid", "?")))].append(e)

    # ---- which experts ran -------------------------------------------------------------
    clusters = collections.defaultdict(list)
    host = []
    for (pid, tid), evs in by_lane.items():
        if tid == "Host Core":
            host += evs
        elif tid.startswith("Cluster "):
            clusters[int(tid.split()[1])] += evs

    # The LAYER is the window in which accelerators actually computed. The total makespan
    # also contains the host's verification pass, which on this graph compares thousands of
    # fp16 elements in a scalar CVA6 loop and dwarfs the layer -- reporting that as the
    # layer's cost would be measuring the test, not the workload.
    eng = [e for e in X if is_engine(e)]
    if eng:
        l0 = min(int(e["ts"]) for e in eng)
        l1 = max(int(e["ts"]) + int(e.get("dur", 0)) for e in eng)
        layer = l1 - l0
    else:
        l0, l1, layer = t0, t1, makespan

    print(f"\ntotal makespan {makespan:,} ns ({makespan / a.cluster_ns:,.0f} cluster cc)"
          f"   <- includes the host verification pass")
    print(f"LAYER window   {layer:,} ns ({layer / a.cluster_ns:,.0f} cluster cc)"
          f"   <- first..last accelerator kernel")

    # An expert RAN iff its cluster drove a GEMM. Every core gets an exit node, so mere
    # presence of events proves nothing about the routing decision.
    gemm_by_cluster = collections.defaultdict(int)
    for e in eng:
        if str(e.get("name", "")).replace("BINGO_TRACE_", "").startswith("GEMM_"):
            tid = str(e.get("tid", ""))
            if tid.startswith("Cluster "):
                gemm_by_cluster[int(tid.split()[1])] += 1
    allc = sorted(clusters)
    ran = sorted(c for c in allc if gemm_by_cluster.get(c))
    idle = [c for c in allc if c not in ran]
    print(f"experts that RAN (drove a GEMM): {ran}"
          + (f"   SKIPPED: {idle}" if idle else ""))
    if not idle:
        print("  NOTE: every cluster drove a GEMM. For top-k with k < E the skip did not "
              "happen -- check the router and the CERF write mask.")

    # ---- per-cluster, per-engine ------------------------------------------------------
    print(f"\n{'cluster':>8} {'engine':>6} {'busy ns':>12} {'busy cc':>10} "
          f"{'% layer':>11}  first..last ns")
    for c in sorted(clusters):
        per_core = collections.defaultdict(list)
        for e in clusters[c]:
            tid = str(e.get("tid"))
            per_core[int(tid.split()[-1])].append(e)
        for core in sorted(per_core):
            evs = [e for e in per_core[core] if is_work(e)]
            if not evs:
                continue
            busy = spans(evs)
            s = min(int(x["ts"]) for x in evs) - t0
            en = max(int(x["ts"]) + int(x.get("dur", 0)) for x in evs) - t0
            print(f"{c:>8} {roles.get(core, f'core{core}'):>6} {busy:>12,} "
                  f"{busy / a.cluster_ns:>10,.0f} {100.0 * busy / layer:>10.1f}% "
                  f"  {s:,}..{en:,}")

    host = [e for e in host if is_work(e)]
    if host:
        busy = spans(host)
        print(f"{'host':>8} {'cva6':>6} {busy:>12,} {busy / a.host_ns:>10,.0f} "
              f"{100.0 * busy / makespan:>10.1f}%")

    # ---- where the time actually goes --------------------------------------------------
    # Union across every device lane: makespan minus this is time when NO engine on any
    # cluster was occupied, which is the bubble to attack first.
    all_dev = [e for c in clusters.values() for e in c if is_work(e)]
    union = spans(all_dev)
    print(f"\nany device engine busy: {union:,} ns "
          f"({100.0 * union / layer:.1f}% of the LAYER window)")
    print(f"no device engine busy : {layer - union:,} ns "
          f"({100.0 * (layer - union) / layer:.1f}%)   <- router decision, "
          f"cross-cluster push, combine serialisation")

    # ---- by marker kind, across all device lanes ---------------------------------------
    per_kind = collections.defaultdict(list)
    for c in clusters.values():
        for e in c:
            per_kind[str(e.get("name", "?")).replace("BINGO_TRACE_", "")].append(e)
    print(f"\n{'marker':<34} {'count':>6} {'span ns':>12} {'% layer':>11}  kind")
    for k, evs in sorted(per_kind.items(), key=lambda kv: -spans(kv[1]))[:14]:
        b = spans(evs)
        kind = "WAIT" if k.startswith(WAIT_MARKERS) else "work"
        print(f"{k:<34} {len(evs):>6} {b:>12,} {100.0 * b / layer:>10.1f}%  {kind}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
