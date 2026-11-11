#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""runtime_mac.py <run tag> [...]: a running run's liveness and MAC rate (runs/<tag> of any experiment: RUNS_DIR or EXP
pick its runs/, else the one sweep/*/*/runs holding the tag), from its scheduler monitor
log, its task graph and its workload's header (each GEMM's M x K x N array steps, by task id: the workload must not
have been regenerated since the launch). Per run, one line: simulated time, the window so far, tasks done and since
the last check, the GEMMs' MACs (a GEMM still running counts the steps it has had time for, one a cycle) as a share of the 32 arrays' peak over
the window and over the last interval, UART lines reporting a failed check or a kernel error, and a verdict: ALIVE
when tasks were done since the last check (before the window: when simulated time advanced) and no UART reports an
error (a stretch without GEMMs, such as a root fold or a barrier, is not a stall). The last check per run is kept in runs/<tag>.mon.json."""
import collections
import csv
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.dirname(HERE)
WLS = os.path.join(HERE.split("/target/sim/")[0], "target/sw/host/apps/offload_bingo_hw/multi_chip/workloads")
PEAK_MAC_US = 32 * 1024 * 500                      # MAC a microsecond of the whole package; check() takes the run's own chips
EV = re.compile(rb"\[BINGO EV\] t=(\d+) chip=\w+ cl=\d co=\d ev=([SD]) id=(\d+) ty=0")
ARG = re.compile(r"\s*\.a(\d+) = \{(.*)\},?\s*$")


def runs_of(tag):
    """The runs/ directory of run <tag>: RUNS_DIR, or sweep/$EXP/runs when EXP is set, else the one sweep/*/*/runs
    holding the tag (two holding it is an error: set EXP)."""
    if os.environ.get("RUNS_DIR"):
        return os.environ["RUNS_DIR"]
    if os.environ.get("EXP"):
        return os.path.join(SWEEP, os.environ["EXP"], "runs")
    hits = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(SWEEP, "*", "*", "runs", tag))
                   + glob.glob(os.path.join(SWEEP, "*", "*", "runs", tag + ".*"))})
    if len(hits) != 1:
        sys.exit(f"run {tag}: {', '.join(hits) or 'not found in sweep/*/*/runs'}; set EXP=<experiment>")
    return hits[0]


def gemms(task, wl):
    """{task id: array steps} of every GEMM of the graph (cached in runs/, per workload)."""
    cache = os.path.join(os.path.dirname(os.path.dirname(task)), f".mon_gemms_{wl.replace('/', '_')}.json")
    hdr = os.path.join(WLS, wl, "offload_bingo_hw.h")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(hdr):
        return {int(k): v for k, v in json.load(open(cache)).items()}
    rows = list(csv.DictReader(open(f"{task}/workload_tables/final_dfg.csv")))
    names = {int(r["ID"]): r["Name"] for r in rows}
    shape = {}
    for ln in open(hdr, errors="replace"):
        m = ARG.match(ln)
        if m and int(m.group(1)) in names:
            f = dict(re.findall(r"\.(\w+) = \(?(?:uint32_t\))?\s*(-?\d+)", m.group(2)))
            if "M" in f and "K" in f and "N" in f:
                shape[int(m.group(1))] = int(f["M"]) * int(f["K"]) * int(f["N"])
    # a gemm_minimal reruns its core's previous GEMM's configuration (the compiler's CSR reuse), so it has that GEMM's
    # shape; a core runs its tasks in id order. A split GEMM (params async_exec: gemm_start, a configure task
    # <name>_cfg, gemm_wait <name>_w) has its steps done when its wait is: the wait takes the start's shape, the others
    # none
    out, prev = {}, {}
    by_name = {r["Name"]: int(r["ID"]) for r in rows}
    for r in sorted(rows, key=lambda r: int(r["ID"])):
        i, core, k = int(r["ID"]), (r["Chiplet"], r["Cluster"], r["Core"]), r["Kernel"]
        if k.endswith("gemm_start"):
            continue
        if k.endswith("gemm_wait"):
            m = re.match(r"^(.*)_w(_k[0-9a-f]{2}_cl\d)$", r["Name"])
            st = by_name.get(m.group(1) + m.group(2)) if m else None
            if st in shape:
                out[i] = shape[st]
        elif i in shape:
            out[i] = prev[core] = shape[i]
        elif k.endswith("gemm_minimal") and core in prev:
            out[i] = prev[core]
    json.dump(out, open(cache, "w"))
    return out


def check(tag):
    R = runs_of(tag)
    tasks = glob.glob(f"{R}/{tag}/task_0*")
    if not tasks:
        print(f"WAITING {time.strftime('%H:%M')} {tag}: no task directory yet (building)")
        return True
    task = tasks[0]
    wl = re.search(r"workload: (\S+),", open(f"{R}/{tag}.yaml").read()).group(1)
    log = f"{task}/bin/bingo_progress.log"
    if not os.path.exists(f"{task}/workload_tables/final_dfg.csv") or not os.path.exists(log):
        print(f"WAITING {time.strftime('%H:%M')} {tag}: simulation not started yet")
        return True
    names = {int(r["ID"]): r["Name"] for r in csv.DictReader(open(f"{task}/workload_tables/final_dfg.csv"))}
    a0 = next(i for i, n in names.items() if n.startswith(("S0_start", "A0_start")))     # A<n>_: older runs
    a1 = next((i for i, n in names.items() if n.startswith(("S1_end", "A1_end"))), None)
    g = gemms(task, wl)
    # the peak of the arrays the run has: two a chip of its graph (params chip_rows leaves rows out)
    chips = {r["Chiplet"] for r in csv.DictReader(open(f"{task}/workload_tables/final_dfg.csv"))}
    peak = len(chips) * 2 * 1024 * 500
    t0, t1, last, done, mac = None, None, 0.0, 0, 0
    seen, started = set(), {}
    with open(log, "rb") as f:
        for ln in f:
            m = EV.match(ln)
            if not m:
                continue
            i, t = int(m.group(3)), int(m.group(1)) / 1e6
            last = max(last, t)
            if m.group(2) == b"S":
                started.setdefault(i, t)
                continue
            if i in seen:
                continue
            seen.add(i)
            done += 1
            t0 = t if i == a0 else t0
            t1 = t if i == a1 else t1
            mac += g.get(i, 0) * 1024
    # a GEMM still running counts with the steps it has had time for, one a cycle (500 a microsecond)
    mac += sum(min(g[i], (last - t) * 500) * 1024 for i, t in started.items() if i in g and i not in seen)
    wd = {}
    try:
        wd = json.load(open(f"{R}/{tag}.watchdog.json"))
    except (OSError, ValueError):
        pass
    sim = float(wd.get("sim_us") or last)
    errs = collections.Counter()
    for u in glob.glob(f"{task}/bin/uart_chip_*.log"):
        for ln in open(u, errors="replace"):
            ck = re.search(r"Checks: \d+ PASS, (\d+) FAIL", ln)
            if (ck and int(ck.group(1))) or (not ck and ("Error" in ln or "FAIL" in ln or "failed" in ln)):
                errs[re.sub(r"0x[0-9a-f]+|\d+", "#", ln.strip())[:90]] += 1
    state_f = os.path.join(R, f"{tag}.mon.json")
    prev = json.load(open(state_f)) if os.path.exists(state_f) else None
    win = (last - t0) if t0 is not None else 0.0
    util = 100 * mac / (win * peak) if win > 0 else 0.0
    line = (f"{time.strftime('%H:%M')} {tag}: sim {sim:,.1f} us, last event {last:,.1f} us; "
            f"window {win:,.1f} us ({win * 500:,.0f} cc); tasks done {done:,}/{len(names):,}")
    if prev:
        dd, dm, dw = done - prev["done"], mac - prev["mac"], win - prev["win"]
        iu = 100 * dm / (dw * peak) if dw > 0 else 0.0
        line += (f" (+{dd:,}); MAC {mac / 1e9:,.2f} G, {util:.1f} % of peak over the window, {iu:.1f} % in the last "
                 f"{dw:,.1f} us")
        alive = dd > 0 or t1 is not None or (t0 is None and sim > prev["sim"])
    else:
        line += f"; MAC {mac / 1e9:,.2f} G, {util:.1f} % of peak over the window (first check)"
        alive = True
    if t1 is not None:
        line += f"; WINDOW DONE {t1 - t0:,.1f} us ({(t1 - t0) * 500:,.0f} cc), checks running"
    if errs:
        alive = False
        line += "; UART ERRORS: " + "; ".join(f"{n} x {e}" for e, n in errs.most_common(3))
    if wd.get("state") not in (None, "running"):
        line += f"; watchdog state {wd.get('state')} verdict {wd.get('verdict')}"
    print(("ALIVE   " if alive else "STALLED ") + line)
    json.dump(dict(done=done, mac=mac, win=win, last=last, sim=sim), open(state_f, "w"))
    return alive


if __name__ == "__main__":
    ok = [check(t) for t in sys.argv[1:]]
    sys.exit(0 if all(ok) else 1)
