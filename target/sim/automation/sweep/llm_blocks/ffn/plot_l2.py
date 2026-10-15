#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""plot_l2.py: the level-2 ablation figures, one per workload (FFN, MoE, FA), from the CSVs
parse_ffn.py / parse_moe.py / parse_fa.py write; plus results/l2_summary.json (every variant's
latency, SRAM accesses and phases, for the plan page).

Each figure has three panels:
  (a) the ladder: R0 base (iDMA pulls, the root's gather, chip barriers) -> R1 + the
      multicast broadcast (bcast) -> R2 + the in-transit reduce (bcast_reduce) -> R3 + hardware
      sync (all), each bar the window split at the phase milestones the traces give (each the
      latest end over the clusters), the speedup over R0 at its end;
  (b) leave-one-out from R3: the latency without each mechanism, against R3;
  (c) SRAM accesses in the window, L1 and L3, every variant.
Only variants whose run passed every check are drawn.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parent
FAMILIES = {
    "ffn": dict(csv=SWEEP / "ffn/results/ffn.csv", out=SWEEP / "ffn/results", model="dsv2_l0",
                title="FFN: DeepSeek-V2-Lite layer-0 dense FFN, one token, INT4"),
    "moe": dict(csv=SWEEP / "moe/results/moe.csv", out=SWEEP / "moe/results",
                model="dsv2_l1",
                title="MoE: DeepSeek-V2-Lite layer-1 MoE (top-6 + 2 shared), one token, INT4"),
    "fa": dict(csv=SWEEP / "fa/results/fa.csv", out=SWEEP / "fa/results", model="dsv2_l1",
               title="FA: DeepSeek-V2-Lite layer-1 MLA decode over a real 8K-token cache"),
}
LADDER = [("base", "R0 baseline"), ("bcast", "R1 + broadcast"),
          ("bcast_reduce", "R2 + reduce"), ("all", "R3 + HW sync")]
LOO = [("all", "R3 (all)"), ("reduce_sync", "no broadcast"), ("bcast_sync", "no reduce"),
       ("bcast_reduce", "no HW sync")]
ORDER = ["base", "bcast", "reduce", "sync", "bcast_reduce", "bcast_sync", "reduce_sync", "all"]
# the phases of each family's critical path: (label, milestone column); a segment runs from the
# previous milestone to its own
PHASES = {
    "ffn": [("broadcast x", "t_x_ns"), ("weight stream + GEMVs", "t_dn_ns"),
            ("chip fold", "t_loc_ns"), ("reduce", "t_reduce_ns"), ("tail", "t_end_ns")],
    "moe": [("broadcast x", "t_x_ns"), ("router + route", "t_route_ns"),
            ("expert stream + GEMVs", "t_dn_ns"), ("chip fold", "t_loc_ns"),
            ("reduce", "t_reduce_ns"), ("tail", "t_end_ns")],
    "fa": [("broadcast q", "t_x_ns"), ("pass 1: K stream, QK", "t_pass1_ns"),
           ("m* all-reduce", "t_marr_ns"), ("pass 2: V stream, PV", "t_pass2_ns"),
           ("chip fold", "t_loc_ns"), ("reduce", "t_reduce_ns"),
           ("normalise", "t_out_ns"), ("tail", "t_end_ns")],
}
# categorical, fixed order (validated: adjacent pairs distinct under CVD)
COLORS = ["#2a78d6", "#13875e", "#c9541f", "#4a3aa7", "#b23a6d", "#8a6d00", "#0f7f8c", "#767a83"]


def load(fam):
    """{variant: row} of the passing runs, the newest per variant."""
    p = FAMILIES[fam]["csv"]
    if not p.exists():
        return {}
    model = FAMILIES[fam]["model"]
    out = {}
    for r in csv.DictReader(p.open()):
        if r["verdict"] != "PASS" or r.get("checks_fail", "0") not in ("0", ""):
            continue
        v = r["variant"]
        if v.startswith(model + "_"):
            v = v[len(model) + 1:]
        out[v] = r
    return out


def segments(fam, r):
    """[(label, start us, length us)] of the window, the end anchor last."""
    out, prev = [], 0.0
    us = float(r["us"])
    for lab, col in PHASES[fam]:
        t = r.get(col, "")
        if t in ("", None):
            continue
        t = min(max(float(t) / 1000.0, prev), us)
        out.append((lab, prev, t - prev))
        prev = t
    if us > prev:
        out.append(("tail", prev, us - prev)) if out and out[-1][0] != "tail" else None
    return out


def figure(fam, rows):
    cfg = FAMILIES[fam]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9), gridspec_kw=dict(width_ratios=[1.6, 1, 1.1]))
    fig.suptitle(cfg["title"], fontsize=11, x=0.01, ha="left")
    # (a) the ladder
    a = ax[0]
    labs = [lab for lab, _ in PHASES[fam]]
    col = {lab: COLORS[i % len(COLORS)] for i, lab in enumerate(labs)}
    r0 = rows.get("base")
    y = 0
    yt, yl = [], []
    for v, lab in LADDER:
        if v not in rows:
            continue
        r = rows[v]
        for nm, s, w in segments(fam, r):
            a.barh(y, w, left=s, color=col.get(nm, COLORS[-1]), height=0.62,
                   edgecolor="white", linewidth=1)
        us = float(r["us"])
        txt = f"{us:.1f} us"
        if r0 is not None and v != "base":
            txt += f"  ({float(r0['us']) / us:.2f}x)"
        a.text(us + 2, y, txt, va="center", fontsize=8.5)
        yt.append(y)
        yl.append(lab)
        y -= 1
    a.set_yticks(yt, yl, fontsize=9)
    a.set_xlabel("microseconds after the start anchor (cluster clock 500 MHz)", fontsize=8.5)
    a.set_title("(a) the ladder, split at the phase milestones", fontsize=9.5, loc="left")
    xmax = max((float(r["us"]) for r in rows.values()), default=1)
    a.set_xlim(0, xmax * 1.28)
    hs = [plt.Rectangle((0, 0), 1, 1, color=col[l]) for l in labs]
    a.legend(hs, labs, fontsize=7.2, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2),
             frameon=False)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    # (b) leave-one-out
    b = ax[1]
    ref = rows.get("all")
    xs, hs_, cs = [], [], []
    for i, (v, lab) in enumerate(LOO):
        if v not in rows:
            continue
        us = float(rows[v]["us"])
        xs.append(lab)
        hs_.append(us)
        cs.append("#4c5058" if v == "all" else "#2a78d6")
    bars = b.bar(range(len(xs)), hs_, color=cs, width=0.6)
    for i, (bar, us) in enumerate(zip(bars, hs_)):
        t = f"{us:.1f}"
        if ref is not None and xs[i] != "R3 (all)":
            d = (us / float(ref['us']) - 1) * 100
            t += f"\n{'+' if d >= 0 else '−'}{abs(d):.0f}%"
        b.text(bar.get_x() + bar.get_width() / 2, us, t, ha="center", va="bottom", fontsize=8)
    b.set_xticks(range(len(xs)), xs, fontsize=8, rotation=12)
    b.set_ylabel("latency, us", fontsize=8.5)
    b.set_ylim(0, max(hs_, default=1) * 1.25)
    b.set_title("(b) leave one mechanism out of R3", fontsize=9.5, loc="left")
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    # (c) SRAM accesses
    c = ax[2]
    vs = [v for v in ORDER if v in rows]
    l1 = [float(rows[v].get("sram_l1") or 0) / 1e6 for v in vs]
    l3 = [float(rows[v].get("sram_l3") or 0) / 1e6 for v in vs]
    c.bar(range(len(vs)), l1, color="#2a78d6", width=0.6, label="L1 (clusters)")
    c.bar(range(len(vs)), l3, bottom=l1, color="#eb6834", width=0.6, label="L3 (chips)")
    for i in range(len(vs)):
        c.text(i, l1[i] + l3[i], f"{l1[i] + l3[i]:.1f}", ha="center", va="bottom", fontsize=7.5)
    c.set_xticks(range(len(vs)), vs, fontsize=7.5, rotation=30, ha="right")
    c.set_ylabel("SRAM accesses in the window, millions", fontsize=8.5)
    c.set_title("(c) SRAM accesses", fontsize=9.5, loc="left")
    c.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    for s in ("top", "right"):
        c.spines[s].set_visible(False)
    fig.tight_layout()
    cfg["out"].mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(cfg["out"] / f"{fam}_ablation.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return cfg["out"] / f"{fam}_ablation.png"


def main():
    summary = {}
    for fam in FAMILIES:
        rows = load(fam)
        if not rows:
            print(f"[plot_l2] {fam}: no passing runs")
            continue
        summary[fam] = {v: dict(us=float(r["us"]), rt_cc=int(r["rt_cc"]),
                                sram_l1=int(float(r.get("sram_l1") or 0)),
                                sram_l3=int(float(r.get("sram_l3") or 0)),
                                sram_max_chip=int(float(r.get("sram_max_chip") or 0)),
                                crest=int(r["crest"]), bcast=r["bcast"], reduce=r["reduce"],
                                route=r.get("route", ""), sync=r["sync"],
                                barrier_mode=r.get("barrier_mode", ""),
                                phases=[(lab, round(s, 2), round(w, 2))
                                        for lab, s, w in segments(fam, r)])
                         for v, r in rows.items()}
        png = figure(fam, rows)
        print(f"[plot_l2] {fam}: {sorted(rows)} -> {png}")
    out = SWEEP / "ffn/results/l2_summary.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"[plot_l2] -> {out}")


if __name__ == "__main__":
    sys.exit(main())
