#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""timeline.py <run dir> [chip ...]: the task spans of a moe prefill stage run, in us of simulated time.

The spans come from <task dir>/tasks.csv (written by the after-stop step from the core traces) or, when that holds
no rows, from bin/bingo_progress.log's scheduler events (S = start, D = done, by node id). The workload's
final_dfg.csv names the nodes. Prints every task of the chips named (default: the root, 00), with its instruction stall time
when tasks.csv has it (trace_ffn: gaps of over 100 ns between retired instructions), and the milestones of
every chip: the route table's put, the row lists, the last shared and expert gate|up GEMMs, quantise, down GEMM,
ChainGather and store, and the last task.
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(str(HERE).split("/target/sim/")[0])
WLS = REPO / "target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/moe"
EV = re.compile(r"\[BINGO EV\] t=(\d+) chip=(\w+) cl=(\d+) co=(\d+) ev=(\w) id=(\d+)")
MILESTONES = (("route put", "rt_bput"), ("lists", "rt_dispatch"), ("sh gate|up", "sg_c"), ("ex gate|up", "gu_s"),
              ("quantise", ("q_sh_g", "q_s")), ("down", "dn_j"), ("chain", "y_chain_c"), ("store", "y_store_c"),
              ("end", ""))


def is_(name, key):
    """A task of a milestone: its name starts with the key (or one of the keys); every task for ""."""
    return name.startswith(key if isinstance(key, tuple) else (key,))


def workload(task: Path) -> Path:
    variant = re.sub(r"_snax-[\w-]+$", "", task.name.split("_moe_", 1)[-1])
    for d in WLS.glob("*/*"):
        if (d / "final_dfg.csv").exists() and str(d.relative_to(WLS)).replace("/", "_") == variant:
            return d
    raise SystemExit(f"no workload for {task.name}")


def spans(task: Path):
    rows = []
    tc = task / "tasks.csv"
    if tc.exists():
        rows = [dict(name=r["name"], chip=r["chip"], cl=int(r["cluster"]), co=int(r["core"]),
                     s=float(r["start_ns"]) / 1e3, e=float(r["end_ns"]) / 1e3,
                     st=float(r["stall_ns"]) / 1e3 if r.get("stall_ns") else float("nan"))
                for r in csv.DictReader(tc.open())]
    if rows:
        return rows, "tasks.csv"
    names = {r["ID"]: r["Name"] for r in csv.DictReader((workload(task) / "final_dfg.csv").open())}
    st, dn = {}, {}
    with (task / "bin" / "bingo_progress.log").open(errors="replace") as f:
        for line in f:
            m = EV.match(line)
            if not m:
                continue
            t, chip, cl, co, ev, i = m.groups()
            if ev == "S":
                st[(chip, i)] = (int(t) / 1e6, int(cl), int(co))
            elif ev == "D":
                dn.setdefault((chip, i), int(t) / 1e6)
    for (chip, i), (s, cl, co) in st.items():
        rows.append(dict(name=names.get(i, i), chip=chip, cl=cl, co=co, s=s, e=dn.get((chip, i), float("nan")),
                         st=float("nan")))
    return rows, "bingo_progress.log"


def main():
    run = Path(sys.argv[1])
    show = sys.argv[2:] or ["00"]
    for task in sorted(p for p in run.iterdir() if p.is_dir() and p.name.startswith("task_")):
        rows, src = spans(task)
        if not rows:
            print(f"{task.name}: no spans")
            continue
        t0 = min(r["s"] for r in rows)
        print(f"== {task.name} ({len(rows)} tasks, from {src}; times in us after the first task, {t0:.1f})")
        for chip in show:
            for r in sorted((r for r in rows if r["chip"] == chip), key=lambda r: r["s"]):
                stall = f"{r['st']:7.1f}" if r["st"] == r["st"] else f"{'':7s}"
                print(f"  {r['s'] - t0:8.1f} {r['e'] - t0:8.1f} {r['e'] - r['s']:7.1f} {stall}  {chip} "
                      f"c{r['cl']}/{r['co']}  {r['name'][:60]}")
        by_chip = defaultdict(list)
        for r in rows:
            by_chip[r["chip"]].append(r)
        print("  milestones (latest end, us):  chip  " + "  ".join(f"{m:>10s}" for m, _ in MILESTONES))
        for chip in sorted(by_chip):
            cells = []
            for _, key in MILESTONES:
                ends = [r["e"] for r in by_chip[chip] if is_(r["name"], key) and r["e"] == r["e"]]
                cells.append(f"{max(ends) - t0:10.1f}" if ends else f"{'-':>10s}")
            print(f"                                {chip}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
