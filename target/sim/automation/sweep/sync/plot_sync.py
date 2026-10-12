#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""E3 figure: the cost of one cross-chip synchronisation for four mechanisms, from
results/four_arm_sync_latency.csv only.

    plot_sync.py [--results results] [--figs figs]  ->  figs/fig_e3_four_arm.{pdf,png}

A and B are one barrier over P chiplets; C and D are a round trip (two edges) through the BINGO
software and hardware task managers, for a 1-to-1 and a 1-to-all edge.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent

# categorical slots in fixed order (the reference palette, light mode)
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ARMS = {("A", "barrier"): (SLOT[0], "s", (4, 2), "A: SW barrier"),
        ("B", "barrier"): (SLOT[1], "D", (4, 2), "B: 0xFF barrier"),
        ("C", "1to1"): (SLOT[4], "o", (1, 1.5), "C: SW manager, 1-to-1"),
        ("C", "1toall"): (SLOT[7], "v", (1, 1.5), "C: SW manager, 1-to-all"),
        ("D", "1to1"): (SLOT[2], "o", None, "D: HW manager, 1-to-1"),
        ("D", "1toall"): (SLOT[6], "v", None, "D: HW manager, 1-to-all")}


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--figs", type=Path, default=HERE / "figs")
    a = ap.parse_args()
    a.figs.mkdir(exist_ok=True)
    rows = list(csv.DictReader((a.results / "four_arm_sync_latency.csv").open()))
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for (arm, mode), (col, mk, dash, lab) in ARMS.items():
        pts = sorted((int(r["P"]), int(r["cc"])) for r in rows if r["arm"] == arm and r["mode"] == mode)
        if not pts:
            continue
        ln, = ax.plot([p for p, _ in pts], [c for _, c in pts], color=col, marker=mk, markersize=5,
                      linewidth=1.6, label=lab, markeredgecolor="white", markeredgewidth=0.6)
        if dash:
            ln.set_dashes(dash)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([2, 4, 8, 16])
    ax.set_xticklabels(["2", "4", "8", "16"])
    ax.set_xlabel("participating chiplets P", color=INK)
    ax.set_ylabel("one synchronisation (cycles, log)", color=INK)
    ax.set_title("A, B: one barrier; C, D: a round trip (two edges)", fontsize=8.5, color=INK, loc="left")
    style_axes(ax)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(a.figs / f"fig_e3_four_arm.{ext}", dpi=200, bbox_inches="tight")
    print(f"[plot_sync] {a.figs / 'fig_e3_four_arm'}.pdf/.png")


if __name__ == "__main__":
    main()
