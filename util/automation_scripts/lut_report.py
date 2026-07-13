#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Shared markdown cycle-report for the sweep CSVs (ara_cycles.csv / xdma_cycles.csv).
# Groups the measured points by op and prints a per-op table of the measured
# cycles -- documentation only.  The per-sweep report scripts are thin
# wrappers that just supply their op list + default CSV path and call run_cli().

import argparse
import csv

# Label columns -- identify the curve, are NOT numeric features to fit/tabulate.
# op_id/prec exist only in the precision-swept ara CSV (op_id = "<op>_<prec>", the
# bingo LUT key); the xdma CSV has neither. Leaving them out of _META made the
# loader try int(float("abs_fp16")) and blow up.
_META = {"op_id", "op_name", "op_node", "prec", "cycles"}


def _load_csv(path):
    """key -> {"op_name", "prec", "feats": [param...], "samples": [(xtuple, cycles)]}.

    One group per measured CURVE. When the CSV carries a precision (ara), that is
    (op_name, prec) -- keyed by op_id -- so the fp32/fp16/int8/int16/int32 curves of
    one op stay separate instead of collapsing into one op with duplicate n. When it
    does not (xdma), the curve is just op_name.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        param_cols = [c for c in reader.fieldnames if c not in _META]
        groups = {}
        for r in reader:
            prec = r.get("prec") or None
            key = r.get("op_id") or r["op_name"]
            g = groups.get(key)
            if g is None:
                feats = [c for c in param_cols if r.get(c, "") != ""]
                g = groups[key] = {"op_name": r["op_name"], "prec": prec,
                                   "feats": feats, "samples": []}
            x = tuple(int(float(r[c])) for c in g["feats"])
            g["samples"].append((x, int(float(r["cycles"]))))
    return groups


def _fmt_points(feats, samples):
    return "; ".join(
        ",".join(f"{f}={v}" for f, v in zip(feats, x)) + f"->{y}"
        for x, y in sorted(samples)
    )


def report(csv_path, ops, out=None):
    """Print (and optionally write) the markdown cycle table for *ops*.

    *ops* are BASE op names. An op measured at several precisions contributes one
    row per precision (its op_id), so `add` yields add_fp32/add_fp16/add_int8/
    add_int16/add_int32. Selection is by the op_name COLUMN, not by name prefix --
    prefix matching would let `silu` swallow every `silu_mul_*` curve.
    """
    groups = _load_csv(csv_path)
    by_op = {}
    for key, g in groups.items():
        by_op.setdefault(g["op_name"], []).append((key, g))
    rows = ["| op | params | n pts | cycles (min..max) | points |",
            "|----|--------|-------|-------------------|--------|"]
    for op in ops:
        entries = sorted(by_op.get(op, []), key=lambda kv: kv[1]["prec"] or "")
        if not entries:
            rows.append(f"| {op} | — | 0 | MISSING | — |")
            continue
        for key, g in entries:
            if not g["samples"]:
                rows.append(f"| {key} | — | 0 | MISSING | — |")
                continue
            feats = g["feats"]
            cyc = [y for _, y in g["samples"]]
            rows.append(f"| {key} | {','.join(feats)} | {len(g['samples'])} | "
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
