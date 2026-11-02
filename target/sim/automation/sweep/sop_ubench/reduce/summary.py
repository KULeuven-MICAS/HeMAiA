#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# summary.py: results/reduce.csv -> results/summary.md: the points of one payload (default
# 128 KiB) for every arm, the fastest arm per (P, payload) (lat_prog_cc), and the fastest
# in-fabric arm (chain / two / binomial) per (P, payload).
import csv
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(D, "results", "reduce.csv"))))
arms = ["chain", "two", "binomial", "pull", "swchain", "swtree"]
Ps = [2, 4, 8, 16]
sizes = sorted({int(r["bytes"]) for r in rows})
pick = int(sys.argv[1]) if len(sys.argv) > 1 else 131072
get = {(r["arm"], int(r["P"]), int(r["bytes"])): r for r in rows}
out = []
out.append(f"## {pick // 1024} KiB, every arm\n")
out.append("| arm | P | lat_prog_cc | lat_cc | SRAM total | SRAM busiest (chip) | check | sync_cc | levels (l:start:end:sync) |")
out.append("|---|---|---|---|---|---|---|---|---|")
for a in arms:
    for P in Ps:
        r = get.get((a, P, pick))
        if r is None:
            out.append(f"| {a} | {P} | - | - | - | - | missing | | |")
            continue
        lv = ";".join(
            f"{l}:{r[f'lvl{l}_start_cc']}:{r[f'lvl{l}_end_cc']}:{r[f'lvl{l}_sync_cc']}"
            for l in range(4) if r[f"lvl{l}_start_cc"] != "")
        same = f" (= {r['same_as']})" if r["same_as"] not in ("-", "") else ""
        out.append(f"| {a}{same} | {P} | {int(r['lat_prog_cc']):,} | {int(r['lat_cc']):,} | "
                   f"{int(r['sram_total']):,} | {int(r['sram_max']):,} ({r['busiest']}) | "
                   f"{r['check']} | {r['sync_cc']} | {lv} |")


def fastest(cands):
    out.append("| P \\ KiB | " + " | ".join(str(s // 1024) for s in sizes) + " |")
    out.append("|---|" + "---|" * len(sizes))
    for P in Ps:
        cells = []
        for s in sizes:
            # an arm that is the plain chain at this P (two at P <= 4, binomial at P = 2) is
            # the chain point again: left out
            pts = [(int(get[(a, P, s)]["lat_prog_cc"]), a) for a in cands
                   if (a, P, s) in get and get[(a, P, s)]["check"] == "PASS"
                   and get[(a, P, s)]["same_as"] in ("-", "")]
            if not pts:
                cells.append("-")
                continue
            pts.sort()
            best = pts[0]
            name = best[1]
            second = f" ({pts[1][1]} +{100 * (pts[1][0] - best[0]) / best[0]:.0f}%)" if len(pts) > 1 else ""
            cells.append(f"{name} {best[0]:,}{second}")
        out.append(f"| {P} | " + " | ".join(cells) + " |")


out.append("\n## Fastest arm per (P, payload), lat_prog_cc (runner-up and its margin)\n")
fastest(arms)
out.append("\n## Fastest in-fabric arm (chain, two, binomial)\n")
fastest(["chain", "two", "binomial"])
fails = [r for r in rows if r["check"] != "PASS"]
out.append(f"\n{len(rows)} points, {len(fails)} not PASS" +
           ("".join(f"\n- {r['arm']} P={r['P']} {r['bytes']}" for r in fails)))
path = os.path.join(D, "results", "summary.md")
open(path, "w").write("\n".join(out) + "\n")
print("\n".join(out))
