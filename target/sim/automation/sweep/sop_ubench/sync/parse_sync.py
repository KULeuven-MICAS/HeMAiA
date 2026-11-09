#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""E3 results: the UARTs of the three arm runs -> results/four_arm_sync_latency.csv.

    parse_sync.py --ab <run dir> --c <run dir> --d <run dir> [--out results/four_arm_sync_latency.csv]

ab: every participating chip prints, per mechanism and P, the min/avg/max of 32 back-to-back
barriers; cc is the mean over the participants of their avg, min/max the extremes over them.
c: chip 0x00 prints one round trip per phase (SYNC[sw] lines). d: chip 0x00's host reporter
prints one round trip per phase index (SYNC[hw] lines); the generator's sync_phases.csv,
snapshot into the run's workload_tables/, says what each index is. All cycles are cluster cycles.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

AB_RE = re.compile(r"Chip\((\w+),\s*(\w+)\) chip-barrier\[(sw|hw)\] (PASS|FAIL), err=(-?\d+) \| "
                   r"P=\s*(\d+) \| barrier cc min/avg/max (\d+)/(\d+)/(\d+)")
C_RE = re.compile(r"SYNC\[sw\] (1to1|1toall)\s+P=\s*(\d+) hops=(\d+) edges=\s*(\d+) \| rt=(\d+) cc")
C_LOCAL_RE = re.compile(r"SYNC\[sw\] LOCAL\s+P= 1 hops=0 edges= 0 \| rt=(\d+) cc")
C_HEAP_RE = re.compile(r"SYNC\[sw\] tasks=(\d+) \| L2 heap capacity=(\d+) peak=(\d+) oom=(\d+)")
D_RE = re.compile(r"SYNC\[hw\] phase=\s*(\d+) \| rt=(\d+) cc")


def task_bin(run: Path) -> Path:
    tasks = sorted(run.glob("task_*"))
    if len(tasks) != 1:
        sys.exit(f"[parse_sync] {run}: expected one task dir, found {len(tasks)}")
    return tasks[0]


def uarts(run: Path) -> str:
    return "".join(f.read_text(errors="replace") for f in sorted((task_bin(run) / "bin").glob("uart_chip_*.log")))


def chip00(run: Path) -> str:
    return (task_bin(run) / "bin" / "uart_chip_0_0.log").read_text(errors="replace")


def parse_ab(run: Path):
    per = {}
    for m in AB_RE.finditer(uarts(run)):
        x, y, mech, verdict, err, p, lmin, lavg, lmax = m.groups()
        if verdict != "PASS" or int(err):
            sys.exit(f"[parse_sync] ab: chip ({x},{y}) [{mech}] P={p} {verdict} err={err}")
        per.setdefault((mech, int(p)), []).append((int(lmin), int(lavg), int(lmax)))
    rows = []
    for (mech, p), v in sorted(per.items()):
        if len(v) != p:
            sys.exit(f"[parse_sync] ab: [{mech}] P={p} has {len(v)} chip reports, expected {p}")
        rows.append({"P": p, "arm": "A" if mech == "sw" else "B", "mode": "barrier",
                     "cc": round(sum(a for _, a, _ in v) / p), "min": min(a for a, _, _ in v),
                     "max": max(b for _, _, b in v), "hops": "", "edges": "", "run": run.name})
    return rows


def parse_c(run: Path):
    txt = chip00(run)
    rows = [{"P": int(p), "arm": "C", "mode": mode, "cc": int(rt), "min": "", "max": "",
             "hops": int(h), "edges": int(e), "run": run.name}
            for mode, p, h, e, rt in C_RE.findall(txt)]
    local = C_LOCAL_RE.search(txt)
    heap = C_HEAP_RE.search(txt)
    if local:
        print(f"[parse_sync] c: local control {local.group(1)} cc")
    if heap:
        print(f"[parse_sync] c: tasks {heap.group(1)}, L2 heap peak {heap.group(3)} of {heap.group(2)}, "
              f"oom {heap.group(4)}")
    return rows


def parse_d(run: Path):
    tables = task_bin(run) / "workload_tables" / "sync_phases.csv"
    phases = {int(r["phase"]): r for r in csv.DictReader(tables.open())}
    rows = []
    for ph, rt in D_RE.findall(chip00(run)):
        meta = phases[int(ph)]
        if meta["kind"] == "local":
            print(f"[parse_sync] d: local control {rt} cc")
        if meta["kind"] not in ("1to1", "1toall"):
            continue
        rows.append({"P": int(meta["P"]), "arm": "D", "mode": meta["kind"], "cc": int(rt), "min": "",
                     "max": "", "hops": int(meta["hops"]), "edges": int(meta["edges"]), "run": run.name})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", type=Path, required=True)
    ap.add_argument("--c", type=Path, required=True)
    ap.add_argument("--d", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "four_arm_sync_latency.csv")
    a = ap.parse_args()
    rows = parse_ab(a.ab) + parse_c(a.c) + parse_d(a.d)
    for arm in "ABCD":
        if not any(r["arm"] == arm for r in rows):
            sys.exit(f"[parse_sync] arm {arm}: no result lines")
    order = {("A", "barrier"): 0, ("B", "barrier"): 1, ("C", "1to1"): 2, ("D", "1to1"): 3,
             ("C", "1toall"): 4, ("D", "1toall"): 5}
    rows.sort(key=lambda r: (r["P"], order[(r["arm"], r["mode"])]))
    a.out.parent.mkdir(exist_ok=True)
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["P", "arm", "mode", "cc", "min", "max", "hops", "edges", "run"])
        w.writeheader()
        w.writerows(rows)
    print(f"[parse_sync] {len(rows)} rows -> {a.out}")
    for r in rows:
        print(f"  P={r['P']:>2} {r['arm']} {r['mode']:<7} {r['cc']:>6} cc")


if __name__ == "__main__":
    main()
