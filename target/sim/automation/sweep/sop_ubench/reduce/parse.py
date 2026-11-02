#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# parse.py: tb_reduce logs -> results/reduce.csv, one row per (arm, P, bytes).
#
#   ./parse.py [log ...]        default: results/logs/*.log
#
# Columns: arm, P, bytes, lat_prog_cc (from the first programming action), lat_cc (from the
# first START / DMA copy), sram_total, sram_max, busiest, check, levels, the per-level times of
# the trees (lvl<l>_start_cc, lvl<l>_end_cc: the level's first programming and last completion,
# from the first programming action; lvl<l>_sync_cc: the level's longest edge, programming after
# its last dependency completed), sync_cc (the sum of lvl<l>_sync_cc over l >= 1), edges (BINGO
# dep-set writes), handoffs (software flags waited for), same_as, log. Every *_cc is in 500 MHz
# cluster cycles. A point that appears in several logs keeps the last one parsed.
import csv
import glob
import os
import re
import sys

D = os.path.dirname(os.path.abspath(__file__))
MAXLVL = 4
LINE = re.compile(r"^\[REDUCE\] (.*)$")


def parse_line(body):
    kv = dict(tok.split("=", 1) for tok in body.split() if "=" in tok)
    row = {
        "arm": kv["arm"], "P": int(kv["P"]), "bytes": int(kv["bytes"]),
        "lat_prog_cc": int(kv["lat_prog_cc"]), "lat_cc": int(kv["lat_cc"]),
        "sram_total": int(kv["sram_total"]), "sram_max": int(kv["sram_max"]),
        "busiest": "0x" + kv["busiest"], "check": kv["check"], "levels": int(kv["levels"]),
        "edges": int(kv["edges"]), "handoffs": int(kv["handoffs"]), "same_as": kv["same_as"],
    }
    sync = 0
    for l in range(MAXLVL):
        for f in ("start", "end", "sync"):
            row[f"lvl{l}_{f}_cc"] = ""
    if kv.get("lvl"):
        for part in kv["lvl"].split(";"):
            l, start, end, s = (int(x) for x in part.split(":"))
            if l < MAXLVL:
                row[f"lvl{l}_start_cc"], row[f"lvl{l}_end_cc"], row[f"lvl{l}_sync_cc"] = start, end, s
            if l >= 1:
                sync += s
    row["sync_cc"] = sync if row["levels"] > 1 else ""
    return row


def main(argv):
    logs = argv[1:] or sorted(glob.glob(os.path.join(D, "results", "logs", "*.log")))
    rows = {}
    bad = []
    for log in logs:
        with open(log, errors="replace") as f:
            text = f.read()
        for line in text.splitlines():
            m = LINE.match(line)
            if m:
                r = parse_line(m.group(1))
                r["log"] = os.path.relpath(log, D)
                rows[(r["arm"], r["P"], r["bytes"])] = r
        if "RESULT: PASS" not in text:
            bad.append(log)
    cols = (["arm", "P", "bytes", "lat_prog_cc", "lat_cc", "sram_total", "sram_max", "busiest",
             "check", "levels"]
            + [f"lvl{l}_{f}_cc" for l in range(MAXLVL) for f in ("start", "end", "sync")]
            + ["sync_cc", "edges", "handoffs", "same_as", "log"])
    order = ["chain", "two", "binomial", "pull", "swchain", "swtree"]
    out = os.path.join(D, "results", "reduce.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k in sorted(rows, key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[2], k[1])):
            w.writerow(rows[k])
    print(f"{len(rows)} points -> {os.path.relpath(out, D)}")
    for log in bad:
        print(f"no RESULT: PASS in {os.path.relpath(log, D)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
