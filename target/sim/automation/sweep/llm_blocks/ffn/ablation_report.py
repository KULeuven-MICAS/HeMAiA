#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""ablation_report.py [rung=tag ...]: the paper's FFN ablation from its runs: DeepSeek-V2-Lite's dense FFN, 64 tokens in
one pass on 16 chips (workloads/ffn/dsv2_l0_prefill_ablation, runs/stg_abl64_<rung>[_n]): the newest run of each
rung unless one is named; base and all with no run of their own take the scaling sweep's 16-chip runs (DEFAULT_TAGS:
the same params, workloads/ffn/scaling/base_n16 and all_n16). Per rung:
the verdict and the root's checks, the window from the root's SYNC line (start -> end anchor, in cycles of the 500 MHz
cluster clock), the MAC-bound time over it (the 32 arrays' peak), the step over the rung below and over base, and the
chip barriers in the window. Writes results/ablation_t64.{csv,md}; a rung with no finished run is TBD."""
import csv
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
RUNGS = [
    ("base", "R0", "X unicast, root fold, a barrier after every kernel"),
    ("bcast", "R1", "+ broadcast: X by one multicast"),
    ("bcast_reduce", "R2", "+ in-transit reduce: ChainGathers"),
    ("all", "R3", "+ per-edge sync: BINGO hardware edges, no barrier"),
]
PREFIX = "stg_abl64_"
# the gate, up and down MACs of 64 tokens over the 32 arrays' 1,024 MACs a cycle
MAC_BOUND_CC = 3 * 2048 * 10944 * 64 // (32 * 1024)
OUT = "ablation_t64"
# base and all with no stg_abl64_ run: the scaling sweep's 16-chip runs (workloads/ffn/scaling/base_n16, all_n16, the same params)
DEFAULT_TAGS = {"base": "stg_sweep64_base_n16", "all": "stg_poc64_g8"}
SYNC = re.compile(r"SYNC\[hw\] phase=\s*0 \| rt=(\d+) cc")
CHECKS = re.compile(r"Checks: (\d+) PASS, (\d+) FAIL")


def newest(rung):
    tags = [t for t in glob.glob(f"{RUNS}/{PREFIX}{rung}") + glob.glob(f"{RUNS}/{PREFIX}{rung}_[0-9]*")
            if os.path.isdir(t)]
    return os.path.basename(max(tags, key=os.path.getmtime)) if tags else None


def one(rung, tag):
    r = dict(rung=rung, tag=tag or "", verdict="not run", checks="", window_cc="", window_us="", mac_pct="",
             barriers="")
    if not tag:
        return r
    task = (glob.glob(f"{RUNS}/{tag}/task_0*") or [None])[0]
    try:
        v = next(iter(json.load(open(f"{RUNS}/{tag}/verdict.json")).values()))
        r["verdict"] = v.get("verdict", "?")
    except (OSError, ValueError, StopIteration):
        r["verdict"] = "running"
    if task is None:
        return r
    uart = os.path.join(task, "bin", "uart_chip_0_0.log")
    txt = open(uart, errors="replace").read() if os.path.exists(uart) else ""
    m, ck = SYNC.search(txt), CHECKS.search(txt)
    if ck:
        r["checks"] = f"{ck.group(1)} PASS, {ck.group(2)} FAIL"
    if m:
        cc = int(m.group(1))
        r.update(window_cc=cc, window_us=round(cc * 0.002, 1), mac_pct=round(100 * MAC_BOUND_CC / cc, 1))
    dfg = os.path.join(task, "workload_tables", "final_dfg.csv")
    if os.path.exists(dfg):
        r["barriers"] = sum(1 for x in csv.DictReader(open(dfg))
                            if re.match(r"barrier\d+_k00_cl0$", x["Name"]))
    return r


def main():
    named = dict(a.split("=", 1) for a in sys.argv[1:])
    rows = [one(rung, named.get(rung) or newest(rung) or DEFAULT_TAGS.get(rung)) for rung, _, _ in RUNGS]
    prev, base = None, rows[0]["window_cc"] or None
    for r, (_, lab, what) in zip(rows, RUNGS):
        r["label"], r["what"] = lab, what
        cc = r["window_cc"] or None
        r["step"] = f"{prev / cc:.2f}x" if prev and cc else ""
        r["vs_base"] = f"{base / cc:.2f}x" if base and cc else ""
        prev = cc
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    keys = ["label", "rung", "what", "tag", "verdict", "checks", "window_cc", "window_us", "mac_pct", "step", "vs_base",
            "barriers"]
    with open(os.path.join(HERE, "results", f"{OUT}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows({k: r[k] for k in keys} for r in rows)
    md = ["| rung | what changes | run | verdict | window | MAC % | step | vs base | barriers |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        win = f"{r['window_us']:,} us ({r['window_cc']:,} cc)" if r["window_cc"] else "TBD"
        md.append(f"| {r['label']} {r['rung']} | {r['what']} | {r['tag'] or '-'} | {r['verdict']} {r['checks']} | "
                  f"{win} | {r['mac_pct'] or 'TBD'} | {r['step'] or '-'} | {r['vs_base'] or '-'} | {r['barriers']} |")
    open(os.path.join(HERE, "results", f"{OUT}.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
