#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""fa_trace.py <run dir> <workload dir> [--out DIR] [--bin US]: where an FA prefill run's window goes.

Reads the scheduler monitor's task events (bin/bingo_progress.log: every task's start and end), the
workload's final_dfg.csv (task names, chips, clusters, cores) and offload_bingo_hw.h (each task's static
arguments: GEMM shapes, load sizes). Writes, under --out (default results/<run tag>/):

  summary.txt   window, MAC utilisation, per-engine busy share, per-chunk phase spans, the slowest
                cluster of each phase
  series.json   per time bin: the arrays' pass share (all clusters), the ring bytes each memory chip has
                pushed, the share of clusters in each phase
  window.png    two panels over the window: the ring stream per memory chip against its two links' peak,
                and the arrays' pass share
  chips.png     the root chip's and the chip in the middle of the fold order's engines, task by task

Array share: a GEMM's passes (M x K x N of its static arguments) are spread over its start-to-wait span
(the start task's start to the matching wait's end); one pass a cycle is 100 %. Ring stream: both clusters
of a chip load each ring slot the memory chip pushed once, so a slot (chip, seq) counts once, at its first
load to finish: its bytes are spread over that load's span (it waits for the push, then copies), per memory
chip (the generator's pairing: memory chip 0x2y serves the compute chips of row y)."""

import argparse
import collections
import csv
import glob
import json
import math
import os
import re
from pathlib import Path

import hjson

CLK_NS = 2.002
MAC_PASS = 1024
CLUSTERS = 32
LINK_GBPS = 28.9                 # one D2D link, sustained (MoE and FFN pushes)


def load(run_dir, wl_dir):
    log = glob.glob(f"{run_dir}/task_0*/bin/bingo_progress.log")[0]
    names = {}
    for r in csv.DictReader(open(f"{wl_dir}/final_dfg.csv")):
        names[int(r["ID"])] = dict(chip=r["Chiplet"], cl=int(r["Cluster"]), core=int(r["Core"]),
                                   kernel=r["Kernel"], name=r["Name"])
    ev = collections.defaultdict(dict)
    for ln in open(log, errors="replace"):
        m = re.match(r"\[BINGO EV\] t=(\d+) chip=(\w+) cl=(\d) co=(\d) ev=(\w) id=(\d+) ty=0", ln)
        if m:
            ev[int(m.group(6))][m.group(5)] = int(m.group(1)) / 1e6
    args = {}
    for ln in open(f"{wl_dir}/offload_bingo_hw.h", errors="replace"):
        m = re.match(r"\s*\.a(\d+) = \{(.*)\},?\s*$", ln)
        if m:
            f = dict(re.findall(r"\.(\w+) = \(?(?:uint32_t\))?\s*(-?\d+)", m.group(2)))
            args[int(m.group(1))] = {k: int(v) for k, v in f.items()}
    return names, ev, args


def kind(name):
    """Phase of a task from its generator name."""
    n = re.sub(r"_k[0-9a-f]{2}_cl\d$", "", name)
    if n.startswith(("x_bput", "x_wait", "x_copy", "x_poll", "x_zero")):
        return "x"
    if re.match(r"s\d+_", n) or n.startswith(("Ld_K", "Ld_V")):
        return "sweep"
    # the post first: its dqu and d2au share a prefix with the projection's dq and d2a
    if n.startswith(("o2a", "ot8", "Ld_wuv", "guv", "dqu", "d2au", "o8_", "Ld_wo", "gwo")):
        return "post"
    if n.startswith(("Ld_wq", "Ld_wuk", "gq", "dq", "d2a", "qn_", "gab", "dqt", "rope", "swap", "q8")):
        return "proj"
    if n.startswith(("y_chain", "y_pull", "y_add", "y_store")):
        return "reduce"
    return "other"


def chunk_of(name):
    m = re.search(r"_i(\d+)", name) or re.match(r"s(\d+)_", name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("wl_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bin", type=float, default=10.0)
    a = ap.parse_args()
    run_dir, wl_dir = Path(a.run_dir), Path(a.wl_dir)
    out = Path(a.out) if a.out else Path(__file__).resolve().parent / "results" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    names, ev, args = load(run_dir, wl_dir)
    p = hjson.load(open(wl_dir / "params.hjson"))
    keys, T = int(p["keys"]), int(p["tokens"])
    mac = (2048 * 192 + 128 * 512 + keys * 576 + keys * 512 + 512 * 128 + 128 * 2048) * T * 16
    a0 = next(i for i, r in names.items() if r["name"].startswith("A0_start"))
    a1 = next(i for i, r in names.items() if r["name"].startswith("A1_end"))
    # a run still going (or stopped early): the window ends at the last event so far
    partial = "S" not in ev.get(a1, {})
    t0 = ev[a0]["D"]
    t1 = max(t for e in ev.values() for t in e.values()) if partial else ev[a1]["S"]
    win = t1 - t0
    peak_us = CLUSTERS * MAC_PASS / (CLK_NS * 1e-3)
    lines = [f"run {run_dir.name}: " + (f"PARTIAL, last event {win:.1f} us after the start anchor ({t0:.1f} to {t1:.1f})"
             if partial else f"window {win:.1f} us ({t0:.1f} to {t1:.1f})") + f"; useful {mac / 1e9:.3f} GMAC; "
             f"ideal {mac / peak_us:.1f} us" + ("" if partial else f"; MAC utilisation {100 * mac / peak_us / win:.1f} %")]

    # GEMM spans: start task -> its wait (same name + "_w")
    by_name = {r["name"]: i for i, r in names.items()}
    gemms = []                       # (cluster key, s, e, passes, phase)
    for i, r in names.items():
        k = r["kernel"]
        if not (k.endswith("_start") and "gemm" in k):
            continue
        w = by_name.get(re.sub(r"(_k[0-9a-f]{2}_cl\d)$", r"_w\1", r["name"]))
        if w is None or "S" not in ev.get(i, {}) or "D" not in ev.get(w, {}):
            continue
        g = args.get(i, {})
        passes = g.get("M", 0) * g.get("K", 0) * g.get("N", 0)
        gemms.append(((r["chip"], r["cl"]), ev[i]["S"], ev[w]["D"], passes, kind(r["name"])))
    total_passes = sum(x[3] for x in gemms)
    lines.append(f"GEMM tasks {len(gemms)}, passes {total_passes:,} = {total_passes * MAC_PASS / 1e9:.3f} GMAC "
                 f"executed ({100 * mac / max(1, total_passes * MAC_PASS):.1f} % useful)")

    # per-engine busy share in the window (task spans, all clusters)
    eng = {0: "GEMM core", 1: "SIMD core", 2: "xDMA core", 3: "DM core", 4: "host"}
    busy = collections.Counter()
    for i, r in names.items():
        e = ev.get(i, {})
        if "S" in e and "D" in e and r["core"] in eng:
            s, d = max(e["S"], t0), min(e["D"], t1)
            if d > s:
                busy[r["core"]] += d - s
    for c in (0, 1, 2, 3):
        lines.append(f"  {eng[c]:10s} busy {100 * busy[c] / (win * CLUSTERS):5.1f} % of the window (task spans, mean of 32 clusters)")
    arr = collections.Counter()
    for key, s, e, n, _ in gemms:
        arr[key] += n * CLK_NS * 1e-3
    lines.append(f"  arrays at one pass a cycle: {100 * sum(arr.values()) / (win * CLUSTERS):.1f} % of the window "
                 f"(busiest cluster {100 * max(arr.values()) / win:.1f} %)")

    # phase spans per chunk (all clusters): earliest start, latest end
    spans = collections.defaultdict(lambda: [1e18, -1e18])
    for i, r in names.items():
        e = ev.get(i, {})
        if "S" not in e or "D" not in e:
            continue
        c = chunk_of(r["name"])
        if c is None:
            continue
        sp = spans[(c, kind(r["name"]))]
        sp[0] = min(sp[0], e["S"])
        sp[1] = max(sp[1], e["D"])
    lines.append("phase spans, us after the start anchor (all clusters):")
    for c in sorted({k[0] for k in spans}):
        row = []
        for ph in ("x", "proj", "sweep", "post", "reduce"):
            if (c, ph) in spans:
                s, e = spans[(c, ph)]
                row.append(f"{ph} {s - t0:7.1f}-{e - t0:7.1f}")
        lines.append(f"  chunk {c}: " + " | ".join(row))

    # time series
    nb = int(math.ceil(win / a.bin))
    pass_bins = [0.0] * nb
    for key, s, e, n, _ in gemms:
        s, e = max(s, t0), min(e, t1)
        if e <= s:
            continue
        rate = n / (e - s)
        b = int((s - t0) // a.bin)
        while b < nb and t0 + b * a.bin < e:
            lo, hi = max(s, t0 + b * a.bin), min(e, t0 + (b + 1) * a.bin)
            if hi > lo:
                pass_bins[b] += rate * (hi - lo)
            b += 1
    cyc_per_bin = a.bin / (CLK_NS * 1e-3)
    arr_share = [100 * x / (cyc_per_bin * CLUSTERS) for x in pass_bins]
    # ring slots pushed, by memory chip (compute chip 0xXY -> memory chip 0x2Y): each (chip, seq) once, at
    # the first of its loads to finish
    first, l1_bytes = {}, 0
    for i, r in names.items():
        if r["kernel"] != "__snax_bingo_kernel_idma_ring_load":
            continue
        e = ev.get(i, {})
        if "S" not in e or "D" not in e:
            continue
        g = args.get(i, {})
        l1_bytes += g.get("size", 0)
        key = (r["chip"], g.get("seq", i))
        if key not in first or e["D"] < first[key][1]:
            first[key] = (e["S"], e["D"], g.get("size", 0))
    ring = collections.defaultdict(lambda: [0.0] * nb)
    for (chip, _), (s, d, size) in first.items():
        mem = "2" + chip[1]
        s, d = max(s, t0), min(d, t1)
        if d <= s:
            continue
        rate = size / (d - s)
        b = int((s - t0) // a.bin)
        while b < nb and t0 + b * a.bin < d:
            lo, hi = max(s, t0 + b * a.bin), min(d, t0 + (b + 1) * a.bin)
            if hi > lo:
                ring[mem][b] += rate * (hi - lo)
            b += 1
    ring_gbps = {m: [x / (a.bin * 1e3) for x in v] for m, v in ring.items()}   # bytes/us -> GB/s
    tot_ring = sum(sum(v) for v in ring.values())
    lines.append(f"ring: {tot_ring / 2**20:.1f} MiB pushed in the window ({l1_bytes / 2**20:.1f} MiB loaded into L1); "
                 f"per memory chip {tot_ring / len(ring) / 2**20:.1f} MiB, mean {tot_ring / len(ring) / win / 1e3:.1f} GB/s "
                 f"over the window (two links, {2 * LINK_GBPS:.1f} GB/s peak)")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    json.dump(dict(bin_us=a.bin, window_us=win, t0=t0, partial=partial,
                   mac_util=None if partial else 100 * mac / peak_us / win,
                   busy={eng[c]: 100 * busy[c] / (win * CLUSTERS) for c in (0, 1, 2, 3)},
                   arrays=100 * sum(arr.values()) / (win * CLUSTERS),
                   pushed_mib_per_mem=tot_ring / len(ring) / 2**20 if ring else 0,
                   push_gbps_mean=tot_ring / len(ring) / win / 1e3 if ring else 0,
                   array_share=arr_share, ring_gbps=ring_gbps,
                   spans={f"{c}:{ph}": [s - t0, e - t0] for (c, ph), (s, e) in spans.items()}),
              open(out / "series.json", "w"))
    print("\n".join(lines))

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [a.bin * (b + 0.5) for b in range(nb)]
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for m in sorted(ring_gbps):
        ax[0].plot(x, ring_gbps[m], lw=1, label=f"memory chip 0x{m}")
    ax[0].axhline(2 * LINK_GBPS, ls="--", c="grey", lw=0.8)
    ax[0].set_ylabel("ring pushed, GB/s")
    ax[0].legend(fontsize=7, ncol=4)
    ax[1].fill_between(x, arr_share, step="mid", alpha=0.6)
    ax[1].set_ylabel("arrays, % passes")
    ax[1].set_xlabel("us after the start anchor")
    ax[1].set_ylim(0, 100)
    for c in sorted({k[0] for k in spans}):
        if (c, "sweep") in spans:
            ax[1].axvspan(spans[(c, "sweep")][0] - t0, spans[(c, "sweep")][1] - t0, color="C1", alpha=0.08)
    fig.suptitle(f"{run_dir.name}: window {win:.0f} us, MAC {100 * mac / peak_us / win:.1f} %")
    fig.tight_layout()
    fig.savefig(out / "window.png", dpi=120)

    # chips: root and the middle of the fold order (cluster 0 and 1, cores 0..3)
    colors = {"x": "C7", "proj": "C0", "sweep": "C1", "post": "C2", "reduce": "C3", "other": "C9"}
    chips = ["00", sorted({r["chip"] for r in names.values()})[8]]
    fig, ax = plt.subplots(figsize=(13, 5))
    lanes = []
    for ch in chips:
        for cl in (0, 1):
            for core in (0, 1, 2, 3):
                lanes.append((ch, cl, core))
    for li, (ch, cl, core) in enumerate(lanes):
        for i, r in names.items():
            if r["chip"] != ch or r["cl"] != cl or r["core"] != core:
                continue
            e = ev.get(i, {})
            if "S" in e and "D" in e and e["D"] > t0 and e["S"] < t1:
                ax.barh(li, e["D"] - e["S"], left=e["S"] - t0, height=0.8, color=colors[kind(r["name"])], lw=0)
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([f"{ch} c{cl} {eng[co]}" for ch, cl, co in lanes], fontsize=7)
    ax.set_xlabel("us after the start anchor")
    ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in colors.values()]
    ax.legend(handles, colors.keys(), fontsize=7, ncol=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "chips.png", dpi=120)


if __name__ == "__main__":
    main()
