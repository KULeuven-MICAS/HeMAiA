#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""parse_ffn.py <run dir> [...]: the FFN runs into results/ffn.csv.

Latency. The root's UART carries one SYNC line per window between consecutive anchors
(__host_bingo_kernel_sync_report after the end anchor): phase 0 is start -> end, or, with
params anchor_reduce, start -> reduce and reduce -> end. A window is counted in DM-core
cycles of chip 0x00.

SRAM accesses. Every anchor also stores a mark that the testbench's SRAM probe answers with a
snapshot of every bank array's counters (bin/snax_probes.log, "[SRAM mark]" lines), so a
window's accesses are the difference of its two anchors' snapshots: summed over the chips, per
level (L1 = the clusters' TCDM, L3 = the chips' memory), and at the busiest chip.

Checks. The root's host compares y16, y_raw and one chip's loc against the golden byte for
byte; quiet checks print only failures, and each chip one "Checks: N PASS, M FAIL" line.

Phases (trace_ffn.py, from the core traces, so before the run is pruned): each phase's latest
task end, in ns after the start anchor -- x on every cluster, the first weight chunk, the last
gate|up and down GEMVs, the last chip's loc, the reduce, y16, the end anchor.
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import hjson

import trace_ffn

HERE = Path(__file__).resolve().parent
REPO = Path(str(HERE).split("/target/sim/")[0])
SYNC_RE = re.compile(r"SYNC\[hw\] phase=\s*(\d+) \| rt=(\d+) cc")
MARK_RE = re.compile(r"\[SRAM mark\] slot=(\d+) t=\S+ (\S+) rd=(\d+) wr=(\d+)")
CHIP_RE = re.compile(r"i_hemaia_(?:mem_chip_)?(\d+)_(\d+)\.")
CHECKS_RE = re.compile(r"Checks:\s*(\d+) PASS,\s*(\d+) FAIL")
NS_PER_CC = 2.0          # cluster clock 500 MHz on the hemaia_16c4m platforms
FAMILIES = ("ffn", "moe", "fa")          # workloads/<family>/<model>/<variant>


def marks(probes: Path):
    snap = defaultdict(dict)
    if not probes.exists():
        return snap
    for m in MARK_RE.finditer(probes.read_text(errors="replace")):
        c = CHIP_RE.search(m.group(2))
        if not c:
            continue
        chip = f"{c.group(1)}_{c.group(2)}"
        level = "L3" if "i_hemaia_mem_system" in m.group(2) else "L1"
        snap[int(m.group(1))][(chip, level)] = (int(m.group(3)), int(m.group(4)))
    return snap


def window(snap, i0, i1):
    a, b = snap.get(i0), snap.get(i1)
    if not a or not b:
        return {}
    per_chip, lv = defaultdict(int), {"L1": 0, "L3": 0}
    for key, (rd1, wr1) in b.items():
        rd0, wr0 = a.get(key, (0, 0))
        d = (rd1 - rd0) + (wr1 - wr0)
        lv[key[1]] += d
        per_chip[key[0]] += d
    busy = max(per_chip, key=per_chip.get)
    return dict(sram_l1=lv["L1"], sram_l3=lv["L3"], sram=lv["L1"] + lv["L3"],
                sram_max_chip=per_chip[busy], busiest_chip=busy)


def parse_task(task: Path, run: str, verdict: str):
    b = task / "bin"
    uart = b / "uart_chip_0_0.log"
    if not uart.exists():
        return []
    rts = [int(m.group(2)) for m in SYNC_RE.finditer(uart.read_text(errors="replace"))]
    checks = [0, 0]
    for u in b.glob("uart_chip_*.log"):
        for m in CHECKS_RE.finditer(u.read_text(errors="replace")):
            checks[0] += int(m.group(1))
            checks[1] += int(m.group(2))
    snap = marks(b / "snax_probes.log")
    # task_<idx>_<host app>_<chip type>_<family>_<model>_<variant>_<dev app>, family ffn or moe
    fam = next((f for f in FAMILIES if f"_{f}_" in task.name), "ffn")
    variant = re.sub(r"_snax-[\w-]+$", "", task.name.split(f"_{fam}_", 1)[-1])
    # the workload directory whose path under workloads/ffn, "/" read as "_", is the variant
    wls = REPO / "target/sw/host/apps/offload_bingo_hw/multi_chip/workloads" / fam
    wl_dir = next((d for d in wls.glob("*/*") if (d / "params.hjson").exists() and
                   str(d.relative_to(wls)).replace("/", "_") == variant), wls / variant)
    params = dict(hjson.loads((wl_dir / "params.hjson").read_text())) \
        if (wl_dir / "params.hjson").exists() else {}
    ph = {}
    trows = []
    # .tasks_extracted: the spans were taken from the traces once the window was over and the
    # traces trimmed after it (they then hold only the idle tail); keep that tasks.csv
    if (task / ".tasks_extracted").exists() and (task / "tasks.csv").exists():
        trows = [dict(r, start_ns=float(r["start_ns"]), end_ns=float(r["end_ns"]))
                 for r in csv.DictReader((task / "tasks.csv").open())]
    elif (wl_dir / "offload_bingo_hw.h").exists() and list((b / "logs").glob("*.dasm")):
        trows = trace_ffn.tasks(task, wl_dir)
        if trows:
            with (task / "tasks.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(trows[0]))
                w.writeheader()
                w.writerows(sorted(trows, key=lambda r: r["start_ns"]))
    elif (task / "tasks.csv").exists():          # pruned: the spans extracted before
        trows = [dict(r, start_ns=float(r["start_ns"]), end_ns=float(r["end_ns"]))
                 for r in csv.DictReader((task / "tasks.csv").open())]
    if trows:
        ph = {f"t_{k}_ns": v for k, v in trace_ffn.phases(trows).items()}
    rows = []
    names = ["ffn"] if len(rts) == 1 else [f"phase{i}" for i in range(len(rts))]
    for i, (nm, rt) in enumerate(zip(names, rts)):
        r = dict(run=run, task=task.name, family=fam, variant=variant, verdict=verdict, window=nm,
                 rt_cc=rt, us=round(rt * NS_PER_CC / 1000, 3),
                 crest=int(bool(params.get("weight_crest", False))),
                 bcast=params.get("bcast", ""), reduce=params.get("reduce", ""),
                 sync=params.get("sync", "edge"),
                 barrier_mode=(params.get("barrier_mode", "bcast")
                               if params.get("sync", "edge") == "barrier" else ""),
                 tokens=int(params.get("tokens", 1)),
                 route=params.get("route", ""),
                 checks_pass=checks[0], checks_fail=checks[1])
        r.update(window(snap, i, i + 1))
        r.update(ph)
        rows.append(r)
    return rows


OUT_CSV = "ffn.csv"
RESULTS = HERE / "results"           # a wrapper (moe/parse_moe.py) points both elsewhere


def main():
    runs = [Path(a) for a in sys.argv[1:]]
    allrows = []
    for run in runs:
        v = {}
        vp = run / "verdict.json"
        if vp.exists():
            v = {x["task"]: x["verdict"] for x in json.loads(vp.read_text()).values()}
        for task in sorted(run.glob("task_*")):
            allrows += parse_task(task, run.name, v.get(task.name, "?"))
    if not allrows:
        print("[parse_ffn] nothing parsed")
        return
    res = RESULTS
    res.mkdir(exist_ok=True)
    keys = ["run", "task", "family", "variant", "verdict", "window", "crest", "bcast", "reduce",
            "route", "sync", "barrier_mode", "tokens", "rt_cc",
            "us", "sram_l1", "sram_l3", "sram", "sram_max_chip", "busiest_chip", "checks_pass",
            "checks_fail", "t_x_ns", "t_route_ns", "t_first_chunk_ns", "t_gu_ns", "t_dn_ns",
            "t_pass1_ns", "t_mstar_ns", "t_marr_ns", "t_pass2_ns", "t_loc_ns", "t_reduce_ns",
            "t_y16_ns", "t_out_ns", "t_end_ns"]
    path = res / OUT_CSV
    old = []
    if path.exists():
        done = {r.name for r in runs}
        old = [r for r in csv.DictReader(path.open()) if r["run"] not in done]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in old + allrows:
            w.writerow(r)
    print(f"[parse_ffn] {len(allrows)} windows from {len(runs)} run(s) -> {path}")
    for r in allrows:
        print(f"  {r['variant']:18s} {r['window']:7s} {r['rt_cc']:>8} cc = {r['us']:>8} us  "
              f"SRAM {r.get('sram', '-')} (L1 {r.get('sram_l1', '-')}, L3 {r.get('sram_l3', '-')}); "
              f"checks {r['checks_pass']} pass {r['checks_fail']} fail; {r['verdict']}")


if __name__ == "__main__":
    main()
