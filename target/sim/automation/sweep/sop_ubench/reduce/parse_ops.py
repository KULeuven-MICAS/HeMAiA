#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# parse_ops.py: the +OP runs of tb_reduce -> results/reduce_ops.csv, one row per (op, arm, P, bytes).
#
#   ./parse_ops.py log ...
#
# Columns: op (smx, topk), arm, P, bytes, lat_prog_cc, lat_cc, sram_total, sram_max, check, log.
# chain / two: the fold socket's junction; pull / swchain / swtree: the sum's payload with every
# fold lasting the merge's measured cycles on one cluster (see tb_reduce +OP). Sum (add) runs are
# skipped: parse.py writes them to results/reduce.csv. A point in several logs keeps the last.

import csv
import os
import re
import sys

D = os.path.dirname(os.path.abspath(__file__))
LINE = re.compile(r"^\[REDUCE\] (.*)$")
COLS = ["op", "arm", "P", "bytes", "lat_prog_cc", "lat_cc", "sram_total", "sram_max", "check", "log"]


def main(argv):
    rows = {}
    for path in argv[1:]:
        for line in open(path, errors="replace"):
            m = LINE.match(line)
            if not m:
                continue
            kv = dict(t.split("=", 1) for t in m.group(1).split() if "=" in t)
            if kv.get("op", "add") == "add":
                continue
            key = (kv["op"], kv["arm"], int(kv["P"]), int(kv["bytes"]))
            rows[key] = {"op": kv["op"], "arm": kv["arm"], "P": int(kv["P"]),
                         "bytes": int(kv["bytes"]), "lat_prog_cc": int(kv["lat_prog_cc"]),
                         "lat_cc": int(kv["lat_cc"]), "sram_total": int(kv["sram_total"]),
                         "sram_max": int(kv["sram_max"]), "check": kv["check"],
                         "log": os.path.basename(path)}
    out = os.path.join(D, "results", "reduce_ops.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])
    print(f"{len(rows)} points -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
