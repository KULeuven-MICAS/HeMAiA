#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""parse_pf.py <run dir> [...]: the FFN prefill runs (workloads/ffn/dsv2_l0_prefill) into
results/pf.csv and results/pf_summary.json.

Latency: the root's SYNC line (start -> end anchor) in DM-core cycles of chip 0x00, 2 ns each.
MAC utilisation: the prefill's multiply-accumulates (T x (2 H I + I H)) over 32 clusters x
1,024 MAC/cycle x the window. GEMM busy: each cluster's gemm_full spans inside the window, from
the core traces (trace_ffn.tasks). Milestones per iteration, in us after the start anchor's end:
the latest X fetch, gate|up GEMM, down GEMM, chip fold, reduce, and the root's store of Y_raw.
SRAM accesses and checks as parse_ffn.py.
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import hjson

import parse_ffn
import trace_ffn

HERE = Path(__file__).resolve().parent
REPO = Path(str(HERE).split("/target/sim/")[0])
WLS = REPO / "target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/ffn"
NS_PER_CC = parse_ffn.NS_PER_CC
MACS_PER_TOKEN = 2 * 2048 * 10944 + 10944 * 2048
ARRAY = 32 * 1024
ITER_RE = re.compile(r"_i(\d+)_")


def wl_of(task: Path):
    variant = re.sub(r"_snax-[\w-]+$", "", task.name.split("_ffn_", 1)[-1])
    return next((d for d in WLS.glob("*/*") if (d / "params.hjson").exists() and
                 str(d.relative_to(WLS)).replace("/", "_") == variant), WLS / variant), variant


def spans(task: Path, wl_dir: Path):
    f = task / "tasks.csv"
    if f.exists():
        return [dict(r, start_ns=float(r["start_ns"]), end_ns=float(r["end_ns"]))
                for r in csv.DictReader(f.open())]
    rows = trace_ffn.tasks(task, wl_dir)
    if rows:
        with f.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r["start_ns"]))
    return rows


def parse_task(task: Path, run: str, verdict: str):
    b = task / "bin"
    uart = b / "uart_chip_0_0.log"
    if not uart.exists():
        return None
    rts = [int(m.group(2)) for m in parse_ffn.SYNC_RE.finditer(uart.read_text(errors="replace"))]
    checks = [0, 0]
    for u in b.glob("uart_chip_*.log"):
        for m in parse_ffn.CHECKS_RE.finditer(u.read_text(errors="replace")):
            checks[0] += int(m.group(1))
            checks[1] += int(m.group(2))
    wl_dir, variant = wl_of(task)
    p = dict(hjson.loads((wl_dir / "params.hjson").read_text()))
    T = int(p.get("tokens", 128))
    r = dict(run=run, variant=variant, verdict=verdict, bcast=p.get("bcast"), reduce=p.get("reduce"),
             sync=p.get("sync"), tokens=T, rows=p.get("rows"), checks_pass=checks[0],
             checks_fail=checks[1])
    if rts:
        rt = rts[0]
        r.update(rt_cc=rt, us=round(rt * NS_PER_CC / 1000, 3),
                 mac_util=round(T * MACS_PER_TOKEN / (ARRAY * rt), 4))
    r.update(parse_ffn.window(parse_ffn.marks(b / "snax_probes.log"), 0, 1))
    rows = spans(task, wl_dir)
    st = next((x for x in rows if x["name"].startswith("A0_start")), None)
    en = next((x for x in rows if x["name"].startswith("A1_end")), None)
    if st and en:
        t0, t1 = st["end_ns"], en["start_ns"]
        busy = defaultdict(float)
        for x in rows:
            if x["name"].startswith(("gu_c", "dn_c")):
                a, e = max(x["start_ns"], t0), min(x["end_ns"], t1)
                if e > a:
                    busy[(x["chip"], x["cluster"])] += e - a
        w = t1 - t0
        if busy:
            v = list(busy.values())
            r.update(gemm_busy_mean=round(sum(v) / len(v) / w, 4), gemm_busy_min=round(min(v) / w, 4),
                     gemm_busy_max=round(max(v) / w, 4))
        ms = defaultdict(float)
        for x in rows:
            m = ITER_RE.search(x["name"] + "_")
            if not m:
                continue
            i = int(m.group(1))
            for key, pre in (("x", "x_"), ("gu", "gu_c"), ("dn", "dn_c"), ("loc", "loc_add"),
                             ("reduce", ("y_chain", "y_add")), ("store", "y_store")):
                if x["name"].startswith(pre):
                    ms[f"i{i}_{key}_us"] = max(ms[f"i{i}_{key}_us"], (x["end_ns"] - t0) / 1000)
        r.update({k: round(v, 3) for k, v in sorted(ms.items())})
    return r


def main():
    out = []
    for a in sys.argv[1:]:
        run = Path(a).resolve()
        vj = run / "verdict.json"
        verdicts = json.loads(vj.read_text()) if vj.exists() else {}
        for task in sorted(run.glob("task_*")):
            v = next((x["verdict"] for k, x in verdicts.items() if k in task.name or
                      task.name.endswith(k)), "?")
            r = parse_task(task, run.name, v)
            if r:
                out.append(r)
                print(json.dumps(r))
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    f = res / "pf.csv"
    old = list(csv.DictReader(f.open())) if f.exists() else []
    keep = [o for o in old if o["run"] not in {r["run"] for r in out}]
    rows = keep + out
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("run", "variant", "verdict", "us"), k))
    with f.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (res / "pf_summary.json").write_text(json.dumps({r["run"]: r for r in rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
