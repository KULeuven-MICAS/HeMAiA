#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Documents the gathered Ara op cycle measurements (ara_cycles.csv): groups the
# points by kernel and prints a markdown table of the measured cycles per size.
# This is measurement documentation only — curve fitting lives in the bingo
# framework, so this script has no bingo dependency.
#
#   python3 ara_lut_report.py [--csv FILE] [--out FILE]

import argparse
import csv
import os

_THIS = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.join(_THIS, "ara_cycles.csv")

# The kernels this sweep produces (so the report flags any that are missing).
OPS = [
    "add", "sub", "mul", "div", "max", "min", "silu_mul",
    "exp", "sigmoid", "sqrt", "relu", "neg", "abs", "tanh", "reciprocal",
    "silu", "gelu", "reduce_sum", "reduce_max", "reduce_mean",
    "quantize", "dequantize", "softmax", "rmsnorm",
]

_META = {"op_name", "op_node", "cycles"}


def _load_csv(path):
    """op_name -> {"feats": [param...], "samples": [(xtuple, cycles)]}."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        param_cols = [c for c in reader.fieldnames if c not in _META]
        groups = {}
        for r in reader:
            op = r["op_name"]
            feats = [c for c in param_cols if r.get(c, "") != ""]
            g = groups.setdefault(op, {"feats": feats, "samples": []})
            x = tuple(int(float(r[c])) for c in g["feats"])
            g["samples"].append((x, int(float(r["cycles"]))))
    return groups


def _fmt_points(feats, samples):
    return "; ".join(
        ",".join(f"{f}={v}" for f, v in zip(feats, x)) + f"->{y}"
        for x, y in sorted(samples)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=_DEFAULT_CSV)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    groups = _load_csv(args.csv)
    rows = ["| op | params | n pts | cycles (min..max) | points |",
            "|----|--------|-------|-------------------|--------|"]
    for op in OPS:
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
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
