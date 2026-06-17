#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Shared markdown cycle-report for the sweep CSVs (ara_cycles.csv / xdma_cycles.csv).
# Groups the measured points by op and prints a per-op table of the measured
# cycles -- documentation only, no curve fitting (that lives in the bingo
# framework, which consumes the CSVs).  The per-sweep report scripts are thin
# wrappers that just supply their op list + default CSV path and call run_cli().

import argparse
import csv

_META = {"op_name", "op_node", "cycles"}


def _load_csv(path):
    """op_name -> {"feats": [param...], "samples": [(xtuple, cycles)]}."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        param_cols = [c for c in reader.fieldnames if c not in _META]
        groups = {}
        for r in reader:
            op = r["op_name"]
            g = groups.get(op)
            if g is None:
                feats = [c for c in param_cols if r.get(c, "") != ""]
                g = groups[op] = {"feats": feats, "samples": []}
            x = tuple(int(float(r[c])) for c in g["feats"])
            g["samples"].append((x, int(float(r["cycles"]))))
    return groups


def _fmt_points(feats, samples):
    return "; ".join(
        ",".join(f"{f}={v}" for f, v in zip(feats, x)) + f"->{y}"
        for x, y in sorted(samples)
    )


def report(csv_path, ops, out=None):
    """Print (and optionally write) the markdown cycle table for *ops*."""
    groups = _load_csv(csv_path)
    rows = ["| op | params | n pts | cycles (min..max) | points |",
            "|----|--------|-------|-------------------|--------|"]
    for op in ops:
        g = groups.get(op)
        if not g or not g["samples"]:
            rows.append(f"| {op} | — | 0 | MISSING | — |")
            continue
        feats = g["feats"]
        cyc = [y for _, y in g["samples"]]
        rows.append(f"| {op} | {','.join(feats)} | {len(g['samples'])} | "
                    f"{min(cyc)}..{max(cyc)} | {_fmt_points(feats, g['samples'])} |")
    table = "\n".join(rows)
    print(table)
    if out:
        with open(out, "w") as f:
            f.write(table + "\n")
        print(f"\nwrote {out}")


def run_cli(default_csv, ops):
    """Argparse entry point shared by the per-sweep report wrappers."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=default_csv)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report(args.csv, ops, args.out)
