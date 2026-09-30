#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""One number sheet per dsv2 layer-1 run: timing, phases, engines, bytes, verdict.

Reads what bingo_gantt.py wrote for each compute chip (tasks.csv), the simulator's log
(the HBM model's statistics, the end time) and the UART (the byte-exact checks), and
writes summary.json + summary.md next to them.

    python3 target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/dsv2/dsv2_summary.py \\
        --task-bin <task>/bin \\
        --gantt <dir with chip_XX/ from bingo_gantt.py> --out <dir> [--clk-mhz 500]

THE LAYER WINDOW is the first cluster task's start to the end of the last task that is not
a check copy (stash_*/xchg_* copies and the host's comparisons come after the layer). Every
time is simulation time; cycles are at --clk-mhz.

ARRAY ACTIVITY comes from the testbench's SpatialArray probe (snax_probes.log, `[ARRAY final]`):
the cycles each GEMM array fired. Divided by the layer's cycles it is the share of the layer the
array computes -- a hardware count, unlike the task spans in busy_pct, which include the array
waiting on its streamers.
"""

import argparse
import csv
import glob
import json
import os
import re

# Critical-path milestones, by exact stage name (regex, fullmatch): (key, stages, first|last)
MILESTONES = [
    ("attn_start", r"attn", "first"),
    ("attn_end", r"attn", "last"),
    ("h_ready", r"h_c\d", "last"),
    ("route_end", r"route", "last"),
    ("experts_end", r"e16_s\d|ys", "last"),
    ("out_end", r"out_s\d", "last"),
]
# Consecutive segments between milestones (None = the layer's start)
PHASES = [
    ("MLA: norm, W_DKV, W_Q, W_UK", None, "attn_start"),
    ("MLA: attention, 8 KV tiles", "attn_start", "attn_end"),
    ("MLA: W_UV, W_O, residual", "attn_end", "h_ready"),
    ("MoE: norm, router, top-6", "h_ready", "route_end"),
    ("MoE: shared + routed experts", "route_end", "experts_end"),
    ("MoE: weighted combine", "experts_end", "out_end"),
]


def load_tasks(gantt_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(gantt_dir, "chip_*", "tasks.csv"))):
        chip = os.path.basename(os.path.dirname(path))[5:]
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    r["run_s"], r["run_e"] = float(r["run_s"]), float(r["run_e"])
                except (KeyError, ValueError):
                    continue
                r["chip"] = chip
                rows.append(r)
    return rows


def is_check_copy(stage):
    """Work that is not the layer: the check copies, and BINGO's per-core exit nodes
    (stage 'Node', run once the host has finished comparing)."""
    return (stage.startswith("stash") or stage.startswith("xchg") or stage == "checks"
            or stage == "Node")


def parse_sim_log(path):
    out = {"hbm": {}}
    if not os.path.exists(path):
        return out
    txt = open(path, errors="replace").read()
    m = re.search(r"\$finish at simulation time\s+(\d+)", txt)
    if m:
        out["end_ps"] = int(m.group(1))
    m = re.search(r"reads : (\d+) bursts, (\d+) beats, latency avg (\d+) ps max (\d+) ps", txt)
    if m:
        out["hbm"].update(read_bursts=int(m.group(1)), read_beats=int(m.group(2)),
                          read_lat_avg_ns=int(m.group(3)) / 1e3,
                          read_lat_max_ns=int(m.group(4)) / 1e3)
    m = re.search(r"rows\s*: (\d+) hit, (\d+) closed, (\d+) conflict \((\d+)% hit\)", txt)
    if m:
        out["hbm"].update(row_hit_pct=int(m.group(4)))
    m = re.search(r"(\d+) bytes in (\d+) ns = (\d+) MB/s", txt)
    if m:
        out["hbm"].update(bytes=int(m.group(1)), active_ns=int(m.group(2)),
                          MBps=int(m.group(3)))
    out["live_watchdog"] = "LIVE WATCHDOG" in txt
    return out


def parse_array(path):
    """{"<chip>/c<cluster>": computeFire cycles} from the SpatialArray probe's final report."""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        if line.startswith("[ARRAY final]"):
            m = re.search(r"i_hemaia_(\d+)_(\d+)\..*i_occamy_cluster_(\d+)\..*computeFire=(\d+)", line)
            if m:
                out[f"{m.group(1)}{m.group(2)}/c{m.group(3)}"] = int(m.group(4))
    return out


def parse_uart(task_bin):
    checks, extra = [], []
    for path in sorted(glob.glob(os.path.join(task_bin, "uart_chip_*.log"))):
        for line in open(path, errors="replace"):
            m = re.search(r"Check \[(\S+)\]: (PASS|FAIL)", line)
            if m:
                checks.append((m.group(1), m.group(2)))
            elif re.search(r"Start at|exit code|DDR|FAIL|Error", line):
                extra.append(line.strip())
    return checks, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-bin", required=True)
    ap.add_argument("--gantt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clk-mhz", type=float, default=500.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rows = load_tasks(args.gantt)
    work = [r for r in rows if not is_check_copy(r["stage"])]
    s = {"label": args.label}
    if work:
        t0 = min(r["run_s"] for r in work)
        t1 = max(r["run_e"] for r in work)
        s["layer_us"] = (t1 - t0) / 1e3
        s["layer_cycles"] = (t1 - t0) * args.clk_mhz / 1e3
        s["t0_us"], s["t1_us"] = t0 / 1e3, t1 / 1e3
        ms = {"start": t0}
        for key, rx, which in MILESTONES:
            sel = [r for r in work if re.fullmatch(rx, r["stage"])]
            if sel:
                ms[key] = (min(r["run_s"] for r in sel) if which == "first"
                           else max(r["run_e"] for r in sel))
        s["milestones_us"] = {k: (v - t0) / 1e3 for k, v in ms.items()}
        phases = []
        for name, a, b in PHASES:
            ps, pe = ms.get(a or "start"), ms.get(b)
            if ps is not None and pe is not None:
                phases.append({"phase": name, "start_us": (ps - t0) / 1e3,
                               "end_us": (pe - t0) / 1e3, "span_us": (pe - ps) / 1e3})
        s["phases"] = phases
        busy = {}
        for r in work:
            key = f"{r['chip']}/c{r['cluster']}/{r['role']}"
            busy[key] = busy.get(key, 0.0) + (r["run_e"] - r["run_s"])
        s["busy_pct"] = {k: round(100 * v / (t1 - t0), 1) for k, v in sorted(busy.items())}
    sim = parse_sim_log(os.path.join(args.task_bin, "sim_run.log"))
    s.update(sim)
    checks, extra = parse_uart(args.task_bin)
    s["checks_pass"] = sum(1 for _, v in checks if v == "PASS")
    s["checks_fail"] = [n for n, v in checks if v == "FAIL"]
    s["uart"] = extra
    if "layer_us" in s and sim["hbm"].get("bytes"):
        s["hbm_GBps_over_layer"] = sim["hbm"]["bytes"] / (s["layer_us"] * 1e3)
    fires = parse_array(os.path.join(args.task_bin, "snax_probes.log"))
    if fires:
        s["array_fires"] = fires
        if "layer_cycles" in s:
            s["array_fire_pct"] = {k: round(100 * v / s["layer_cycles"], 2)
                                   for k, v in sorted(fires.items())}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    md = [f"# {args.label}", ""]
    if "layer_us" in s:
        md.append(f"layer {s['layer_us']:.1f} us = {s['layer_cycles']:,.0f} cycles "
                  f"@ {args.clk_mhz:g} MHz")
        for p in s["phases"]:
            md.append(f"  {p['phase']:<40} {p['start_us']:8.1f} .. {p['end_us']:8.1f} us "
                      f"({p['span_us']:.1f})")
    h = sim["hbm"]
    if h:
        md.append(f"HBM: {h.get('bytes', 0) / 2**20:.2f} MiB read, latency avg "
                  f"{h.get('read_lat_avg_ns', 0):.0f} ns, rows {h.get('row_hit_pct', '?')}% hit")
    if s.get("array_fire_pct"):
        md.append("GEMM array computing: " + ", ".join(
            f"{k} {v:.1f}%" for k, v in s["array_fire_pct"].items()) + " of the layer's cycles")
    md.append(f"checks: {s['checks_pass']} PASS, {len(s['checks_fail'])} FAIL "
              f"{s['checks_fail'][:5]}")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
