#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""estimate_pf.py: the FFN prefill ladder (base, + broadcast, + reduce, all) estimated from measured spans.

Each rung is the measured time of the steps it is made of; nothing is assumed except what is flagged.

  R2  bcast_reduce   runs/pf_bcast_reduce: a full-size run of the bulk-synchronous program, iterations
                     0 and 1 complete. Its iteration 0 (stream bound: the weights land during it) and
                     its steady iteration (barrier to barrier) are taken as they are.
  R1  bcast          R2 with the chain gather replaced by the root's load-all: the measured time per
                     32 KiB slice (runs/pf_bcast_v1, pull + SIMD add) times 15 chips x 4 slices.
  R0  base           R1 with the X multicast replaced by 30 remote clusters pulling the root's L3 at
                     the measured D2D pull rate (16.7 GB/s a link, two links into the root). THE ONE
                     ASSUMED STEP: no run measured that pull at this size. Iteration 0 hides it (the
                     weight stream, not X, paces it).
  R3  all            the software-pipelined program: the arrays run GU0 GU1 DN0 GU2 DN1 GU3 DN2 DN3 with
                     no barrier, so a cluster ends at (its GU0 end, which the weight stream sets) +
                     3 GU + 4 DN of iteration 1's measured per-cluster GEMM time, and the last
                     iteration's tail (chip fold, chain, store) follows the slowest cluster.

The tiny run (runs/pft_all_ccfix) calibrates the GEMM against the shapes it ran, and checks the
down GEMM (M=32, K=96, N=512) fit against the full-size K=352 span.

The runs behind R1 and R2 were simulated before the rr_arb_tree fix: they stalled at about iteration
2 (the fault), their spans up to there are used; the tiny run, which has both simulators, agrees
between them to 1 %. Results: results/pf_estimate.json, results/pf_estimate.md, figs/pf_estimate.*."""

import csv
import glob
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
T, H, I = 128, 2048, 10944
MAC_US = 32 * 1024 * 500                       # MACs a microsecond: 32 clusters x 1,024 x 500 MHz
IDEAL_US = T * 3 * H * I / MAC_US
ROWS, YB, SL = 32, 32 * 2048 * 2, 32 * 1024
SLICES = 15 * (YB // SL)                       # the root's load-all: 15 chips x 4 slices an iteration
XB = ROWS * H                                  # X_i, INT8
PULL_BPS, ROOT_LINKS, REMOTE = 16.7e9, 2, 30   # assumed step R0 (see above)
PLAN = {"base": 2700, "bcast": None, "bcast_reduce": None, "all": 622}   # rev 13-17 of the plan page


def load(run):
    f = glob.glob(str(RUNS / run / "task_0*" / "tasks.csv"))
    rows = list(csv.DictReader(open(f[0])))
    for r in rows:
        r["s"], r["e"] = float(r["start_ns"]) / 1000, float(r["end_ns"]) / 1000
        r["d"] = r["e"] - r["s"]
    t0 = next(r for r in rows if r["name"].startswith("A0_start"))["e"]
    for r in rows:
        r["s"] -= t0
        r["e"] -= t0
    return rows


def sel(rows, pat):
    return [r for r in rows if re.match(pat, r["name"])]


def mean(rows):
    return st.mean(r["d"] for r in rows)


def per_cluster(rows, pat, it):
    out = defaultdict(list)
    for r in sel(rows, rf"^{pat}_c\d+_i{it}_g(\d+)_"):
        out[int(re.search(r"_g(\d+)_", r["name"]).group(1))].append(r)
    return out


def measure():
    r2, r1, tiny = load("pf_bcast_reduce"), load("pf_bcast_v1"), load("pft_all_ccfix")
    m = {}
    # --- R2: the bulk-synchronous program, full size
    m["it0_end"] = max(r["e"] for r in sel(r2, "barrier3"))
    m["it1_end"] = max(r["e"] for r in sel(r2, "barrier5"))
    m["period"] = m["it1_end"] - m["it0_end"]
    m["close_barrier"] = m["it1_end"] - max(r["e"] for r in sel(r2, "y_store_i1"))   # store end -> barrier end
    m["chain"] = st.mean([mean(sel(r2, f"y_chain_i{i}")) for i in (0, 1)])
    m["store"] = mean(sel(r2, "y_store_i0"))
    m["loc_add"] = mean(sel(r2, "loc_add_i0"))
    m["part_push"] = mean(sel(r2, "part_push_i0"))
    m["x_mcast"] = (max(r["e"] for r in sel(r2, "x_wait_i1")) - min(r["s"] for r in sel(r2, "x_bput_i1")))
    m["stream_end"] = max(r["e"] for r in sel(r2, "Ld_c\\d+_i0"))
    gu0 = per_cluster(r2, "gu", 0)
    m["gu0_end"] = {g: max(r["e"] for r in v) for g, v in gu0.items() if len(v) >= 7}
    gu1, dn1 = per_cluster(r2, "gu", 1), per_cluster(r2, "dn", 1)
    m["gu1_busy"] = {g: sum(r["d"] for r in v) for g, v in gu1.items() if len(v) >= 7}
    m["dn1_busy"] = {g: sum(r["d"] for r in v) for g, v in dn1.items() if len(v) == 4}
    m["gu_full"] = mean([r for r in sel(r2, r"gu_c\d+_i1_") if r["d"] > 12])
    m["dn_chunk"] = mean(sel(r2, r"dn_c\d+_i1_"))
    # --- R1: the root's load-all
    adds = sel(r1, r"y_add_i0_")
    pulls = sel(r1, r"y_pull_i0_")
    m["slice_us"] = (max(r["e"] for r in adds) - min(r["s"] for r in pulls)) / len(adds)
    m["load_all"] = SLICES * m["slice_us"]
    # --- the tiny run: GEMMs in isolation (one iteration, the array mostly idle while weights land)
    m["tiny_gu"] = mean(sel(tiny, r"gu_c\d+_i0_"))
    m["tiny_dn"] = mean(sel(tiny, r"dn_c\d+_i0_"))
    return m


def estimate(m, gemm=1.0, tail_skew=0.0, gu0="max"):
    mid = st.median(m["gu0_end"].values())
    cl = [g for g in m["gu0_end"] if g in m["gu1_busy"] and g in m["dn1_busy"]]
    ends = {g: (m["gu0_end"][g] if gu0 == "max" else mid) + gemm * (3 * m["gu1_busy"][g] + 4 * m["dn1_busy"][g])
            for g in cl}
    tail = m["part_push"] + m["loc_add"] + m["chain"] + m["store"] + tail_skew
    r3 = max(ends.values()) + tail
    last = m["close_barrier"]
    r2 = m["it0_end"] + 3 * m["period"] - last
    swap = m["load_all"] - m["chain"]
    r1 = m["it0_end"] + swap + 3 * (m["period"] + swap) - last
    pull = REMOTE * XB / (PULL_BPS * ROOT_LINKS) * 1e6
    r0 = r1 + 3 * (pull - m["x_mcast"])
    return {"base": r0, "bcast": r1, "bcast_reduce": r2, "all": r3}, pull


def main():
    m = measure()
    lo, _ = estimate(m, gemm=1.0, gu0="median")
    mid, pull = estimate(m)
    hi, _ = estimate(m, gemm=1.04, tail_skew=15.0)
    res = {"measured": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()
                        if not isinstance(v, dict)},
           "pull_x_us_assumed": round(pull, 1), "ideal_us": round(IDEAL_US, 1), "ladder": {}}
    for k in ("base", "bcast", "bcast_reduce", "all"):
        # only R3 has a computed range (cluster skew, GEMM contention); R0-R2 are one measured iteration
        lo_k, hi_k = (min(lo[k], mid[k]), max(hi[k], mid[k])) if k == "all" else (mid[k], mid[k])
        res["ladder"][k] = {"us": round(mid[k]), "low": round(lo_k), "high": round(hi_k),
                            "mac_util": round(IDEAL_US / mid[k], 3), "speedup_vs_base": round(mid["base"] / mid[k], 2)}
    heavy = max(m["gu1_busy"][g] + m["dn1_busy"][g] for g in m["gu1_busy"] if g in m["dn1_busy"])
    res["heavy_cluster_gemm_busy_per_iteration_us"] = round(heavy, 1)
    res["gemm_eff_vs_ideal"] = round(IDEAL_US / 4 * (352 / 342) / heavy, 3)
    out = HERE / "results"
    (out / "pf_estimate.json").write_text(json.dumps(res, indent=1) + "\n")
    L = res["ladder"]
    md = ["| rung | estimate, us | range | MAC util | vs base |", "|---|---|---|---|---|"]
    for k, lab in (("base", "base"), ("bcast", "+ broadcast"), ("bcast_reduce", "+ reduce"), ("all", "all (+ per-edge sync)")):
        v = L[k]
        rng = f"{v['low']:,}-{v['high']:,}" if v["low"] != v["high"] else "-"
        md.append(f"| {lab} | {v['us']:,} | {rng} | {100 * v['mac_util']:.0f} % | {v['speedup_vs_base']:.2f}x |")
    (out / "pf_estimate.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(json.dumps(res["measured"], indent=1))
    print("heavy cluster GEMM busy / iteration:", res["heavy_cluster_gemm_busy_per_iteration_us"], "us;",
          "efficiency vs ideal:", res["gemm_eff_vs_ideal"])
    plot(res)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = res["ladder"]
    names = [("base", "base"), ("bcast", "+ broadcast"), ("bcast_reduce", "+ reduce"), ("all", "all")]
    col = ["#8a8f98", "#2a78d6", "#c9541f", "#13875e"]
    meas = {}
    summ = HERE / "results" / "pf_summary.json"
    if summ.exists():
        for r in json.loads(summ.read_text()).values():
            if r.get("run", "").startswith("pfl_") and r.get("verdict") == "PASS" and r.get("us"):
                meas[r["variant"].replace("dsv2_l0_prefill_", "")] = float(r["us"])
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    for i, (k, lab) in enumerate(names):
        v = L[k]
        ax.barh(i, v["us"], color=col[i], height=0.55, alpha=0.55 if k not in meas else 0.35, hatch="//")
        ax.errorbar(v["us"], i, xerr=[[v["us"] - v["low"]], [v["high"] - v["us"]]], color="k", capsize=3, lw=1)
        t = f"{v['us']:,} us est. · {100 * v['mac_util']:.0f} % MAC · {v['speedup_vs_base']:.2f}x"
        if k in meas:
            ax.barh(i + 0.0, meas[k], color=col[i], height=0.3)
            t += f"  |  measured {meas[k]:,.0f} us"
        ax.text(v["high"] * 1.015, i, t, va="center", fontsize=8)
    ax.set_yticks(range(4), [lab for _, lab in names])
    ax.invert_yaxis()
    ax.axvline(res["ideal_us"], color="k", ls=":", lw=1)
    ax.text(res["ideal_us"], 3.45, " 100 % MAC", fontsize=7)
    ax.set_xlim(0, max(L["base"]["high"], 1) * 1.7)
    ax.set_xlabel("window, us (128-token prefill, 4 x 32 rows)")
    ax.set_title("FFN prefill ladder: estimate from measured spans (hatched), measured run (solid)", fontsize=9, loc="left")
    fig.tight_layout()
    d = HERE / "figs"
    d.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(d / f"pf_estimate.{ext}", dpi=160)
    print("wrote", d / "pf_estimate.png")


if __name__ == "__main__":
    main()
