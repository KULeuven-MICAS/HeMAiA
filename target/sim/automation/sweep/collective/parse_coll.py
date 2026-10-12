#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""parse_coll.py <run dir> [...]: the collective runs into results/collective.csv.

Latency. The root (chip 0x00) prints one line per point on its UART, after the point:
    [COLL] <primitive> <impl> P=<P> r=<repeat> k=<point> rt=<cycles> s=<first mark slot>
rt is the root's t1 - t0 in cluster cycles. A run that dies partway keeps the points before it.

SRAM accesses. The root writes an SRAM mark at t0 (slot s) and at t1 (slot s + 1); the
testbench's probe answers each with every bank array's read and write counters
(bin/snax_probes.log, "[SRAM mark]" lines, one per chip and memory: L1 = a cluster's TCDM,
L3 = the chip's memory). A point's accesses are the difference of its two snapshots, in 64-bit
bank accesses (reads + writes), summed over the chips and at the busiest chip. Memory chiplets
are not counted (they only forward).

Every NET metric subtracts the median EMPTY point of the same primitive and P (GO and READY
alone), which every implementation pays alike.

Checks. "Check [<primitive>_<impl>_P<P>_c<chip>]: PASS|FAIL" lines on any chip's UART; a point
gets FAIL if any chip failed its (impl, P), PASS if one passed and none failed.
"""

import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

COLL_RE = re.compile(r"\[COLL\] (\S+) (\S+) P=(\d+) r=(\d+) k=(\d+) rt=(\d+) s=(\d+)")
MARK_RE = re.compile(r"\[SRAM mark\] slot=(\d+) t=(\d+) (\S+) rd=(\d+) wr=(\d+)")
CHIP_RE = re.compile(r"\.i_hemaia_(\d+)_(\d+)\.")
CHECK_RE = re.compile(r"Check \[([a-z_]+)_([a-z0-9]+)_P(\d+)_c[0-9a-f]+\]: (PASS|FAIL)")
HERE = Path(__file__).resolve().parent
NS_PER_CC = 2.0          # cluster clock: 3.5 GHz / 7 = 500 MHz
METRICS = ["rt_cc", "sram_l1", "sram_l3", "sram", "sram_max_chip"]


def marks(probes: Path):
    """{slot: ({(chip, level): (rd, wr)}, time)} from the probe log."""
    snap = defaultdict(dict)
    when = {}
    if not probes.exists():
        return snap, when
    for m in MARK_RE.finditer(probes.read_text(errors="replace")):
        c = CHIP_RE.search(m.group(3))
        if not c:
            continue
        chip = f"{c.group(1)}_{c.group(2)}"
        level = "L3" if "i_hemaia_mem_system" in m.group(3) else "L1"
        key = (chip, level, m.group(3))
        slot = int(m.group(1))
        snap[slot][key] = (int(m.group(4)), int(m.group(5)))
        when[slot] = int(m.group(2))
    return snap, when


def sram(snap, s0):
    """The accesses between slots s0 and s0 + 1: (L1, L3, total, busiest chip's total, its name)."""
    a, b = snap.get(s0), snap.get(s0 + 1)
    if not a or not b:
        return None
    per_chip = defaultdict(int)
    lv = {"L1": 0, "L3": 0}
    for key, (rd1, wr1) in b.items():
        rd0, wr0 = a.get(key, (0, 0))
        d = (rd1 - rd0) + (wr1 - wr0)
        lv[key[1]] += d
        per_chip[key[0]] += d
    busy = max(per_chip, key=per_chip.get)
    return lv["L1"], lv["L3"], lv["L1"] + lv["L3"], per_chip[busy], busy


def checks(b: Path):
    res = {}
    for f in sorted(b.glob("uart_chip_*.log")):
        for m in CHECK_RE.finditer(f.read_text(errors="replace")):
            key = (m.group(1), m.group(2), int(m.group(3)))
            if m.group(4) == "FAIL" or key not in res:
                res[key] = m.group(4)
    return res


def parse_task(task: Path, run: str, verdict: str):
    b = task / "bin"
    uart = b / "uart_chip_0_0.log"
    if not uart.exists():
        return []
    snap, when = marks(b / "snax_probes.log")
    ck = checks(b)
    rows = []
    for m in COLL_RE.finditer(uart.read_text(errors="replace")):
        prim, impl, P, rep, k, rt, s0 = m.groups()
        r = {"run": run, "task": task.name, "verdict": verdict, "prim": prim, "impl": impl,
             "P": int(P), "rep": int(rep), "k": int(k), "rt_cc": int(rt), "slot": int(s0)}
        s = sram(snap, int(s0))
        if s:
            r["sram_l1"], r["sram_l3"], r["sram"], r["sram_max_chip"], r["busiest_chip"] = s
        if int(s0) in when and int(s0) + 1 in when:
            r["rt_ns_tb"] = (when[int(s0) + 1] - when[int(s0)]) / 1000.0
        r["check"] = ck.get((prim, impl, int(P)), "")
        rows.append(r)
    empty = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["impl"] == "empty":
            for k in METRICS:
                if r.get(k, "") != "":
                    empty[(r["prim"], r["P"])][k].append(int(r[k]))
    out = []
    for r in rows:
        if r["impl"] == "empty":
            continue
        base = empty.get((r["prim"], r["P"]), {})
        for k in METRICS:
            if r.get(k, "") == "":
                continue
            e = statistics.median(base[k]) if base.get(k) else 0
            r[f"net_{k}"] = int(r[k]) - e
        r["empty_cc"] = statistics.median(base["rt_cc"]) if base.get("rt_cc") else 0
        r["net_cc"] = r["net_rt_cc"]
        r["net_us"] = round(r["net_cc"] * NS_PER_CC / 1000, 3)
        out.append(r)
    return out


def main():
    runs = [Path(a) for a in sys.argv[1:]]
    allrows = []
    for run in runs:
        v = {}
        vp = run / "verdict.json"
        if vp.exists():
            # a run stopped by its wall-clock timeout with no failed check and no watchdog keeps
            # the points it finished: TIMEOUT, not FAIL
            v = {x["task"]: ("TIMEOUT" if x["verdict"] == "FAIL" and x.get("timeout")
                             and not x.get("checks_fail") and not x.get("live_watchdog")
                             else x["verdict"])
                 for x in json.loads(vp.read_text()).values()}
        for task in sorted(run.glob("task_*")):
            allrows += parse_task(task, run.name, v.get(task.name, "?"))
    if not allrows:
        print("[parse_coll] nothing parsed")
        return
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    keys = ["run", "task", "verdict", "check", "prim", "impl", "P", "rep", "k", "slot", "rt_cc",
            "rt_ns_tb", "empty_cc", "net_cc", "net_us", "sram_l1", "sram_l3", "sram",
            "sram_max_chip", "busiest_chip", "net_sram_l1", "net_sram_l3", "net_sram",
            "net_sram_max_chip"]
    path = res / "collective.csv"
    old = []
    if path.exists():
        done = {r.name for r in runs}
        old = [r for r in csv.DictReader(path.open()) if r.get("run") not in done and "prim" in r]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in old + allrows:
            w.writerow(r)
    print(f"[parse_coll] {len(allrows)} points from {len(runs)} run(s) -> {path}")
    for r in allrows:
        print(f"  {r['prim']:10s} {r['impl']:6s} P={r['P']:>2} r={r['rep']} "
              f"net={r['net_cc']:>8} cc  SRAM net={r.get('net_sram', '-'):>8} "
              f"(L1 {r.get('net_sram_l1', '-')}, L3 {r.get('net_sram_l3', '-')}; busiest "
              f"{r.get('busiest_chip', '-')} {r.get('net_sram_max_chip', '-')})  {r['check']}")


if __name__ == "__main__":
    main()
