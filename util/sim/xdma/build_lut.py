#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Correlate XDMA_RUN trace events with a sweep's configs.json to build a
# shape -> cycles look-up table (CSV).
#
# The per-op sweep runs configs serially, so the Nth XDMA_RUN event (sorted by
# timestamp) corresponds to CONFIGS[N]. We pair them in order and print a CSV.

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="bin/logs/bingo_trace.json")
    ap.add_argument("--configs", required=True, help="workload configs.json")
    ap.add_argument("--out", default=None, help="optional CSV output path")
    args = ap.parse_args()

    with open(args.trace) as f:
        trace = json.load(f)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    # bingo_trace names the paired event "BINGO_TRACE_XDMA_RUN" (the perf-tracing
    # marker name); match the XDMA_RUN suffix so renames stay compatible.
    runs = [e for e in events
            if e.get("ph") == "X" and "XDMA_RUN" in str(e.get("name", ""))]
    runs.sort(key=lambda e: e.get("ts", 0))

    with open(args.configs) as f:
        cfg = json.load(f)
    op = cfg.get("op", "xdma")
    configs = cfg["configs"]

    if len(runs) != len(configs):
        print(f"# WARNING: {len(runs)} XDMA_RUN events vs {len(configs)} configs "
              f"— pairing the first {min(len(runs), len(configs))} in order")

    rows = [["op", "config", "cycles"]]
    for cdict, ev in zip(configs, runs):
        dur_cc = ev.get("args", {}).get("dur_cc", "")
        rows.append([op, json.dumps(cdict, separators=(",", ":")), str(dur_cc)])

    csv = "\n".join(",".join('"' + c + '"' if "," in c else c for c in r)
                    for r in rows)
    print(csv)
    if args.out:
        with open(args.out, "w") as f:
            f.write(csv + "\n")
        print(f"\n# wrote {args.out}")


if __name__ == "__main__":
    main()
