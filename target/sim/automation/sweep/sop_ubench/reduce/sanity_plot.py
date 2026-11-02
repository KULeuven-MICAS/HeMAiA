#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# sanity_plot.py: a quick look at results/reduce.csv (not a paper figure): latency (lat_prog_cc)
# against payload, one panel per P, one line per arm; and SRAM accesses at 128 KiB against P.
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(D, "results", "reduce.csv"))))
arms = ["chain", "two", "binomial", "pull", "swchain", "swtree"]
Ps = [2, 4, 8, 16]
fig, ax = plt.subplots(1, 5, figsize=(22, 4))
for i, P in enumerate(Ps):
    for a in arms:
        pts = sorted((int(r["bytes"]), int(r["lat_prog_cc"])) for r in rows
                     if r["arm"] == a and int(r["P"]) == P and r["check"] == "PASS")
        if pts:
            ax[i].plot([b / 1024 for b, _ in pts], [l for _, l in pts], marker="o", label=a)
    ax[i].set_xscale("log", base=2)
    ax[i].set_yscale("log")
    ax[i].set_title(f"P = {P}")
    ax[i].set_xlabel("KiB per chip")
    ax[i].set_ylabel("lat_prog_cc")
    ax[i].grid(True, which="both", alpha=0.3)
ax[0].legend(fontsize=8)
for a in arms:
    pts = sorted((int(r["P"]), int(r["sram_total"])) for r in rows
                 if r["arm"] == a and int(r["bytes"]) == 131072 and r["check"] == "PASS")
    if pts:
        ax[4].plot([p for p, _ in pts], [s for _, s in pts], marker="o", label=a)
ax[4].set_title("SRAM accesses, 128 KiB")
ax[4].set_xlabel("P")
ax[4].set_yscale("log")
ax[4].grid(True, which="both", alpha=0.3)
fig.tight_layout()
out = os.path.join(D, "results", "sanity.png")
fig.savefig(out, dpi=110)
print(out)
