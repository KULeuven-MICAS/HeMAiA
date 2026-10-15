#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""E2 figure and summary table, from results/collective.csv only.

    plot_coll.py [--results results] [--figs figs] [--all-verdicts] [--log]

  fig_e2_collectives  latency (cycles) on top, SRAM accesses below:
                      1.1 broadcast, 1.2 reduce, 1.3 all-reduce against P, every
                      implementation; 1.4 the three side by side at the largest P
  results/collective_summary.csv
                      net latency and SRAM accesses per (primitive, implementation, P), and each
                      implementation over the reference one (the in-fabric one) for both

A point's value is its fastest repeat in the latest run that measured it (the first repeat of an implementation pays a cold
instruction cache), latency and SRAM accesses both from that repeat. SRAM accesses are 64-bit
bank accesses (reads + writes) summed over every compute chip's L1 and L3. Only runs whose
verdict is PASS (or TIMEOUT: stopped by the wall clock with no failed check) and points whose check
did not fail are used unless --all-verdicts. Each figure
is written as PDF and PNG.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent

# the reference categorical palette, light mode, in its fixed order
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# one colour and marker per implementation family, the same in every panel
FAMILY = {"d2d": (SLOT[0], "o"), "xdma": (SLOT[0], "o"), "pull": (SLOT[1], "s"),
          "chain": (SLOT[2], "^"), "ring": (SLOT[2], "^"), "tree": (SLOT[3], "D"),
          "rd": (SLOT[3], "D"), "a2a": (SLOT[1], "s"), "xtree": (SLOT[4], "v"),
          "d2dn": (SLOT[0], "P", "--")}
# primitive -> (title, [(impl, legend label)] in drawing order, reference impl)
PRIM = {
    "broadcast": ("broadcast", [("d2d", "D2D router broadcast"), ("pull", "iDMA")], "d2d"),
    "reduce": ("reduce", [("xdma", "xDMA in-transit chain"), ("xtree", "xDMA tree + SW sync"),
                          ("pull", "iDMA load all + local fold"),
                          ("chain", "iDMA chain + SW sync"), ("tree", "iDMA tree + SW sync")],
               "xdma"),
    "all_reduce": ("all-reduce", [("xdma", "xDMA chain + D2D broadcast"),
                                  ("xtree", "xDMA tree + SW sync + D2D broadcast"),
                                  ("a2a", "iDMA all-to-all push + local fold"),
                                  ("pull", "iDMA load all + local fold"),
                                  ("ring", "iDMA ring + SW sync"),
                                  ("rd", "iDMA recursive doubling + SW sync")], "xdma"),
}
SHORT = {"d2d": "D2D", "xdma": "xDMA", "pull": "load all", "a2a": "all-to-all", "chain": "chain",
         "ring": "ring", "tree": "tree", "rd": "rec. dbl.", "xtree": "xDMA tree", "d2dn": "D2D at once"}


def kfmt(v, _pos=None):
    """Tick labels: 0, 950, 12K, 1.5M."""
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.1f}M".replace(".0M", "M")
    if a >= 1e4:
        return f"{v / 1e3:.0f}K"
    if a >= 1e3:
        return f"{v / 1e3:.1f}K".replace(".0K", "K")
    return f"{v:.0f}"


def style_axes(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def load(path, all_verdicts, tag):
    """{(prim, impl, P): {"cc", "sram", "sram_max"}} from the fastest repeat of each point, over the
    runs of one round (run names start with `tag`). A point measured by several runs takes the
    run parsed last (rows are in parse order): a rerun supersedes."""
    best, run_of = {}, {}
    for r in csv.DictReader(path.open()):
        if r.get("prim") not in PRIM or not r.get("run", "").startswith(tag + "_"):
            continue
        if not all_verdicts and (r["verdict"] not in ("PASS", "TIMEOUT") or r.get("check") == "FAIL"):
            continue
        if r.get("net_cc", "") in ("", None):
            continue
        key = (r["prim"], r["impl"], int(r["P"]))
        cc = float(r["net_cc"])
        if run_of.get(key) == r["run"] and best[key]["cc"] <= cc:
            continue
        run_of[key] = r["run"]
        m = {"cc": cc}
        for col, name in (("net_sram", "sram"), ("net_sram_max_chip", "sram_max")):
            if r.get(col, "") not in ("", None):
                m[name] = float(r[col])
        best[key] = m
    return best


def summary(med, out):
    rows = []
    for (prim, impl, P), m in sorted(med.items()):
        ref = PRIM.get(prim, (None, None, None))[2]
        b = med.get((prim, ref, P), {})
        rows.append({"prim": prim, "impl": impl, "P": P, "net_cc": round(m["cc"]),
                     "net_sram": round(m["sram"]) if "sram" in m else "",
                     "net_sram_busiest_chip": round(m["sram_max"]) if "sram_max" in m else "",
                     "latency_over_ref": round(m["cc"] / b["cc"], 2) if b.get("cc") else "",
                     "sram_over_ref": round(m["sram"] / b["sram"], 2)
                     if b.get("sram") and "sram" in m else ""})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def spread(vals, lo, hi, gap):
    """Label positions (axes fraction) for end values `vals`, at least `gap` apart."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    pos = [0.0] * len(vals)
    last = -1.0
    for i in order:
        f = (vals[i] - lo) / (hi - lo) if hi > lo else 0.5
        f = max(f, last + gap)
        pos[i] = f
        last = f
    return pos


def vs_p(ax_t, ax_b, med, prim, log):
    title, arms, _ref = PRIM[prim]
    for ax, m in ((ax_t, "cc"), (ax_b, "sram")):
        ends = []
        for impl, lab in arms:
            xs = sorted(k[2] for k in med if k[0] == prim and k[1] == impl)
            pts = [(P, med[(prim, impl, P)][m]) for P in xs if m in med[(prim, impl, P)]]
            if not pts:
                continue
            col, mk, *ls = FAMILY[impl]
            ax.plot([p for p, _ in pts], [y for _, y in pts], color=col, marker=mk, ms=4.5,
                    lw=1.8, ls=ls[0] if ls else "-", label=lab)
            ends.append((pts[-1], col))
        style_axes(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16])
        ax.set_xticklabels(["2", "4", "8", "16"])
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(kfmt))
        # end labels, pushed apart so none overlap
        if ends:
            lo, hi = ax.get_ylim()
            if log:
                import math
                f = lambda v: math.log10(max(v, 1e-9))  # noqa: E731
                vals, lo, hi = [f(y) for (_, y), _ in ends], f(lo), f(hi)
            else:
                vals = [y for (_, y), _ in ends]
            pos = spread(vals, lo, hi, 0.07)
            for ((x, y), _col), p in zip(ends, pos):
                ax.annotate(f"{y:,.0f}", xy=(x, y), xycoords="data",
                            xytext=(1.02, p), textcoords=("axes fraction"),
                            fontsize=6.5, color=INK2, va="center")
    ax_t.set_title(title, fontsize=9, color=INK)
    ax_b.set_xlabel("participating chiplets P", fontsize=8, color=INK2)
    h, lab = ax_t.get_legend_handles_labels()
    ax_t.legend(h, lab, fontsize=6, frameon=False, loc="upper left")


def side_by_side(ax_t, ax_b, med):
    groups = []
    for prim, (title, arms, ref) in PRIM.items():
        Ps = [k[2] for k in med if k[0] == prim]
        if not Ps:
            continue
        P = max(Ps)
        groups.append((title, P, ref, [(impl, med.get((prim, impl, P), {})) for impl, _ in arms]))
    for ax, m in ((ax_t, "cc"), (ax_b, "sram")):
        style_axes(ax)
        for gi, (title, P, ref, bars) in enumerate(groups):
            have = [(impl, v) for impl, v in bars if m in v]
            n = len(have)
            w = 0.8 / max(n, 1)
            refv = dict(have).get(ref, {}).get(m)
            for bi, (impl, v) in enumerate(have):
                x = gi - 0.4 + w * (bi + 0.5)
                ax.bar(x, v[m], w * 0.92, color=FAMILY[impl][0], edgecolor="white", lw=0.8,
                       hatch="///" if len(FAMILY[impl]) > 2 else None)
                if refv and impl != ref:
                    ax.annotate(f"{v[m] / refv:.1f}x", (x, v[m]), textcoords="offset points",
                                xytext=(0, 2), ha="center", va="bottom", rotation=90,
                                fontsize=5.5, color=INK)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[0] for g in groups], fontsize=7, rotation=20)
        ax.set_yscale("log")
        ax.margins(y=0.25)
    P = groups[0][1] if groups else 16
    ax_t.set_title(f"side by side, P = {P}", fontsize=9, color=INK)
    # one legend for the families
    from matplotlib.patches import Patch
    fam = [("D2D / xDMA (in fabric)", SLOT[0]), ("iDMA load all / all-to-all push", SLOT[1]),
           ("iDMA chain / ring + SW sync", SLOT[2]), ("iDMA tree / rec. dbl. + SW sync", SLOT[3]),
           ("xDMA tree + SW sync", SLOT[4])]
    ax_b.figure.legend([Patch(color=c) for _, c in fam], [l for l, _ in fam], fontsize=7,
                       frameon=False, loc="lower center", ncol=len(fam),
                       bbox_to_anchor=(0.5, -0.04))


def fig_collectives(med, figs, log, suffix=""):
    prims = [p for p in PRIM if any(k[0] == p for k in med)]
    ncol = len(prims) + 1
    fig, axes = plt.subplots(2, ncol, figsize=(3.0 * ncol + 1.6, 5.1), sharex="col",
                             squeeze=False, gridspec_kw={"width_ratios": [1] * (ncol - 1) + [1.7]})
    for col, prim in enumerate(prims):
        vs_p(axes[0][col], axes[1][col], med, prim, log)
    side_by_side(axes[0][ncol - 1], axes[1][ncol - 1], med)
    axes[0][0].set_ylabel("latency (cycles)", fontsize=8, color=INK2)
    axes[1][0].set_ylabel("SRAM accesses (64-bit)", fontsize=8, color=INK2)
    for i in range(ncol):
        axes[0][i].text(-0.12, 1.08, f"1.{i + 1}", transform=axes[0][i].transAxes, fontsize=9,
                        weight="bold", color=INK)
    fig.tight_layout(w_pad=2.2)
    for ext in ("pdf", "png"):
        fig.savefig(figs / f"fig_e2_collectives{suffix}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[plot_coll] fig_e2_collectives{suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--figs", type=Path, default=HERE / "figs")
    ap.add_argument("--all-verdicts", action="store_true")
    ap.add_argument("--log", action="store_true", help="log y axes in the panels against P")
    ap.add_argument("--tag", default=None,
                    help="the round to plot, by run-name prefix: e2d (D2D links in DDR) or e2l "
                         "(SDR); default e2d if it has points, else e2l")
    a = ap.parse_args()
    a.figs.mkdir(exist_ok=True)
    a.results.mkdir(exist_ok=True)
    if not (a.results / "collective.csv").exists():
        print("[plot_coll] no results/collective.csv yet")
        return
    tag = a.tag
    if tag is None:
        tag = "e2d" if load(a.results / "collective.csv", True, "e2d") else "e2l"
    med = load(a.results / "collective.csv", a.all_verdicts, tag)
    print(f"[plot_coll] round {tag}")
    if not med:
        print("[plot_coll] no points")
        return
    sfx = "" if tag == "e2d" else f"_{tag}"
    rows = summary(med, a.results / f"collective_summary{sfx}.csv")
    print(f"[plot_coll] {len(rows)} summary rows -> {a.results / f'collective_summary{sfx}.csv'}")
    fig_collectives(med, a.figs, a.log, sfx)


if __name__ == "__main__":
    main()
