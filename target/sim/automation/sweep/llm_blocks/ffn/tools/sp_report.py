#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""sp_report.py <run dir> <workload under workloads/ffn>: where the time of a measured FFN prefill went.

The workload is generated in-process into a scratch directory (never in place) for each GEMM's array
steps; the run's bingo_progress.log gives every task's release, start and done. Prints:
- the window (start anchor done to the last store done), the MAC utilisation over all 32 clusters
  (array steps x 2 ns at the (16,4,16) array's peak, over clusters x window) and the busiest cluster's;
- per cluster: GEMM busy, head (start anchor to first GEMM start), the gate|up span, the gap to the first
  down GEMM, the down span with its idle time, the tail (last down GEMM done to the window end);
- a fit of GEMM time = fixed + steps x rate, separately for gemm_full and gemm_minimal;
- single pass: per down slice, the latest down GEMM done over every cluster, the latest fold done, the
  gather's start and done and the store's done;
- the longest idle stretches of the latest-finishing cluster's GEMM core, with what the next GEMM waited
  for (released late = a predecessor; started after its release = the core's queue)."""
import argparse
import collections
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from replay_pf import W, generate, size_of  # noqa: E402

GEMM = re.compile(r"^(gu|dn)_c\d+")


def events(run, wl_dfg):
    import csv
    names = {int(r["ID"]): r["Name"] for r in csv.DictReader(open(wl_dfg))}
    log = next(p for p in (os.path.join(run, d, "bin", "bingo_progress.log") for d in os.listdir(run)
                           if d.startswith("task_")) if os.path.exists(p))
    ev = collections.defaultdict(dict)
    for ln in open(log, errors="replace"):
        if not ln.startswith("[BINGO EV]"):
            continue
        kv = dict(re.findall(r"(\w+)=(\S+)", ln))
        n = names.get(int(kv["id"]))
        if n:
            ev[n].setdefault(kv["ev"], int(kv["t"]) / 1e6)
    return ev


def fit(pts):
    """Least squares t = f + r x s over [(s, t)]: (f us, r ns per step, n)."""
    n = len(pts)
    if n < 2:
        return None
    ms = sum(s for s, _ in pts) / n
    mt = sum(t for _, t in pts) / n
    vs = sum((s - ms) ** 2 for s, _ in pts)
    if vs == 0:
        return (mt, 0.0, n)
    r = sum((s - ms) * (t - mt) for s, t in pts) / vs
    return (mt - r * ms, r * 1e3, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("workload")
    ap.add_argument("--gaps", type=int, default=8, help="idle stretches to list")
    a = ap.parse_args()
    ev = events(a.run, f"{W}/{a.workload}/final_dfg.csv")
    with tempfile.TemporaryDirectory(prefix="sp_report_") as out:
        S = generate(a.workload, out)
    nodes = sorted(S.dfg.node_list, key=lambda t: t.node_id)
    steps, kern, last = {}, {}, {}
    for nd in nodes:
        k = nd.kernel_name or ""
        if not GEMM.match(nd.node_name or ""):
            continue
        key = (nd.assigned_chiplet_id, nd.assigned_cluster_id, nd.assigned_core_id)
        if k.endswith("gemm_full"):
            last[key] = size_of(nd)
            steps[nd.node_name] = last[key]
        elif k.endswith("gemm_minimal"):
            steps[nd.node_name] = last.get(key)
        kern[nd.node_name] = "minimal" if k.endswith("gemm_minimal") else "full"
    t0 = ev["A0_start_k00_cl0"]["D"]
    stores = [n for n in ev if n.startswith("y_store_") and "D" in ev[n]]
    t1 = max(ev[n]["D"] for n in stores)
    win = t1 - t0
    done = {n: e for n, e in ev.items() if "S" in e and "D" in e}
    missing = [n for n in steps if n not in done]
    tot = sum(steps[n] for n in steps if steps[n])
    ncl = len({n.split("_k")[-1] for n in steps})
    print(f"{a.workload} in {os.path.basename(a.run)}: window {win:.1f} us ({t0:.1f} -> {t1:.1f}); "
          f"{len(steps)} GEMMs, {tot:,} array steps, {len(missing)} GEMMs without start/done")
    ideal = tot * 2e-3 / ncl
    print(f"  MAC utilisation {100 * ideal / win:.1f} % ({ideal:.1f} us of array time a cluster at peak, {ncl} clusters)")
    per = collections.defaultdict(list)
    for n in steps:
        if n in done:
            per[n.split("_k")[-1]].append((done[n]["S"] - t0, done[n]["D"] - t0, n))
    rows = []
    for cl, v in per.items():
        v.sort()
        gu = [x for x in v if x[2].startswith("gu")]
        dn = [x for x in v if x[2].startswith("dn")]
        busy = sum(d - s for s, d, _ in v)
        st = sum(steps[x[2]] or 0 for x in v)
        idle_gu = (gu[-1][1] - gu[0][0]) - sum(d - s for s, d, _ in gu) if gu else 0
        idle_dn = (dn[-1][1] - dn[0][0]) - sum(d - s for s, d, _ in dn) if dn else 0
        rows.append((cl, busy, st, v[0][0], gu, dn, idle_gu, idle_dn, v[-1][1]))
    rows.sort(key=lambda r: -r[2])
    big = rows[0]
    print(f"  busiest cluster {big[0]}: {big[2]:,} steps, MAC utilisation {100 * big[2] * 2e-3 / win:.1f} % of the window, "
          f"GEMM busy {big[1]:.1f} us ({100 * big[1] / win:.1f} %)")
    print(f"  {'cluster':10s} {'steps':>8s} {'busy':>7s} {'head':>6s} {'gate|up':>15s} {'idle':>6s} {'gap':>6s} "
          f"{'down':>15s} {'idle':>6s} {'tail':>6s}")
    for cl, busy, st, h, gu, dn, ig, idn, end in sorted(rows, key=lambda r: -r[8])[:6] + \
            ([big] if big not in sorted(rows, key=lambda r: -r[8])[:6] else []):
        gap = dn[0][0] - gu[-1][1] if gu and dn else 0
        print(f"  {cl:10s} {st:8,d} {busy:7.1f} {h:6.1f} {gu[0][0]:6.1f}-{gu[-1][1]:6.1f}  {ig:6.1f} {gap:6.1f} "
              f"{dn[0][0]:6.1f}-{dn[-1][1]:6.1f}  {idn:6.1f} {win - end:6.1f}")
    for kind in ("full", "minimal"):
        for ph in ("gu", "dn"):
            pts = [(steps[n], done[n]["D"] - done[n]["S"]) for n in steps
                   if n in done and steps[n] and kern[n] == kind and n.startswith(ph)]
            f = fit(pts)
            if f:
                print(f"  GEMM {kind:7s} {ph}: {f[2]:4d} tasks, fixed {f[0]:5.2f} us + {f[1]:.3f} ns/step "
                      f"(mean {sum(t for _, t in pts) / len(pts):.2f} us)")
    if getattr(S, "single", False):
        print(f"  {'slice':6s} {'down done':>9s} {'push':>7s} {'fold':>7s} {'gather':>15s} {'store':>7s}")
        for j in range(S.NDN):
            def mx(rx, ph="D"):
                v = [e[ph] - t0 for n, e in done.items() if re.match(rx, n)]
                return max(v) if v else float("nan")
            sfx = r"(p\d+|b\d+)?"              # a tail_split part or a split_first_dn block
            print(f"  {j:<6d} {mx(rf'dn_c{j}_i0{sfx}_g'):9.1f} {mx(rf'part_push_j{j}{sfx}_k'):7.1f} "
                  f"{mx(rf'loc_add_j{j}{sfx}_k'):7.1f} {mx(rf'y_chain_j{j}{sfx}_k', 'S'):7.1f}-{mx(rf'y_chain_j{j}{sfx}_k'):7.1f} "
                  f"{mx(rf'y_store_j{j}{sfx}_k'):7.1f}")
    late = max(rows, key=lambda r: r[8])
    v = sorted(per[late[0]])
    gaps = sorted(((b[0] - p[1], p[2], b[2]) for p, b in zip(v, v[1:])), reverse=True)[:a.gaps]
    print(f"  longest GEMM idle on {late[0]} (the last to finish its GEMMs):")
    for g, p, b in gaps:
        e = done[b]
        why = "released late (a predecessor)" if e.get("R", e["S"]) - t0 > done[p]["D"] - t0 + 0.05 else "queue"
        print(f"    {g:6.1f} us  {p} -> {b}: released {e.get('R', float('nan')) - t0:.1f}, started {e['S'] - t0:.1f} ({why})")


if __name__ == "__main__":
    main()
