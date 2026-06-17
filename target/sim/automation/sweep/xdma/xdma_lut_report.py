#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Documents the gathered xDMA op cycle measurements (xdma_cycles.csv): groups the
# points by op and prints a markdown table of the measured cycles per shape.
# This is measurement documentation only — curve fitting lives in the bingo
# framework, so this script has no bingo dependency.
#
#   python3 xdma_lut_report.py [--csv FILE] [--out FILE]

import argparse
import csv
import os

_THIS = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.join(_THIS, "xdma_cycles.csv")

# The 14 ops this sweep produces (so the report flags any that are missing).
OPS = [
    "xdma_6d", "xdma_transpose_2d", "xdma_submatrix_2d", "xdma_expand_2d",
    "xdma_concat_2d", "xdma_pad_2d", "xdma_gather_2d", "xdma_elementwise_add",
    "xdma_row_major_to_a", "xdma_a_to_row_major", "xdma_row_major_to_b",
    "xdma_b_to_row_major", "xdma_row_major_to_d", "xdma_d_to_row_major",
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
