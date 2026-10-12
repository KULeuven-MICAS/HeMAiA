#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""plot_pf.py: the FFN prefill ladder from results/pf_summary.json into figs/pf_ladder.{png,pdf}.

(a) the window of each rung, base -> + broadcast -> + reduce -> + per-edge sync, each bar
    labelled with its MAC utilisation and its speedup over base;
(b) each rung's per-iteration milestones (the last down GEMM and the root's store of Y_raw of
    every iteration), in us after the start anchor.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
LADDER = [("dsv2_l0_prefill_base", "R0 base\npulls, root load-all,\nchip barriers"),
          ("dsv2_l0_prefill_bcast", "R1 + broadcast"),
          ("dsv2_l0_prefill_bcast_reduce", "R2 + reduce"),
          ("dsv2_l0_prefill_all", "R3 + per-edge sync")]
COL = ["#8a8f98", "#2a78d6", "#c9541f", "#13875e"]


def main():
    s = json.loads((HERE / "results" / "pf_summary.json").read_text())
    by_var = {r["variant"]: r for r in s.values() if r.get("verdict") == "PASS" and r.get("us")}
    rungs = [(v, lab) for v, lab in LADDER if v in by_var]
    if not rungs:
        raise SystemExit("no passing prefill run in results/pf_summary.json")
    base = by_var.get(LADDER[0][0])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    a = ax[0]
    for i, (v, lab) in enumerate(rungs):
        r = by_var[v]
        a.barh(i, r["us"], color=COL[LADDER.index((v, lab))], height=0.6)
        txt = f"{r['us']:.0f} us · {100 * float(r['mac_util']):.0f} % MAC"
        if base:
            txt += f" · {base['us'] / r['us']:.2f}x"
        a.text(r["us"] * 1.01, i, txt, va="center", fontsize=9)
    a.set_yticks(range(len(rungs)), [lab for _, lab in rungs], fontsize=9)
    a.invert_yaxis()
    a.set_xlabel("window, us (128-token prefill, 4 x 32 rows)")
    a.set_xlim(0, max(by_var[v]["us"] for v, _ in rungs) * 1.45)
    a.set_title("(a) the ladder", loc="left", fontsize=10)
    b = ax[1]
    for i, (v, lab) in enumerate(rungs):
        r = by_var[v]
        it = sorted({int(k[1]) for k in r if k.startswith("i") and k.endswith("_store_us")})
        dn = [r.get(f"i{j}_dn_us") for j in it]
        stv = [r.get(f"i{j}_store_us") for j in it]
        c = COL[LADDER.index((v, lab))]
        b.plot(it, dn, "o--", color=c, alpha=0.6, label=f"{lab.splitlines()[0]}: last down GEMM")
        b.plot(it, stv, "s-", color=c, label=f"{lab.splitlines()[0]}: Y stored")
    b.set_xlabel("iteration (32 rows each)")
    b.set_ylabel("us after the start anchor")
    b.set_xticks(range(4))
    b.legend(fontsize=7, ncol=1)
    b.set_title("(b) per iteration", loc="left", fontsize=10)
    fig.tight_layout()
    out = HERE / "figs"
    out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"pf_ladder.{ext}", dpi=160)
    print("wrote", out / "pf_ladder.png")


if __name__ == "__main__":
    main()
