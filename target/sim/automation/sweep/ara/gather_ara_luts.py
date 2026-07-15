#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Gather the Ara (CVA6+Ara) cycle sweep into one CSV, keyed by the BINGO OP ID.
#
# Each ara_<kernel> app sweeps precision AND size and prints, over UART:
#     CYCLES,<kernel>,<prec>,<N>,<rep>,<cycles>
# Precision is a RUNTIME ARG to the `__host_bingo_kernel_<kernel>` DISPATCHER -- there is ONE C symbol
# per kernel, not one per precision (see target/sw/host/runtime/ara_sweep.h).
#
# --------------------------------------------------------------------------------------------------
# THE CSV KEY
#
# Rows are keyed on the id the bingo framework uses, where PRECISION IS A SUFFIX:
#
#     abs        + fp16   -> abs_fp16              regular op:  <base>_<prec>
#     reduce_sum + int8   -> reduce_sum_int8
#     quantize   + f32i8  -> quantize_f32i8        conversion:  the prec token IS the in->out pair,
#     dequantize + i32f32 -> dequantize_i32f32                  exactly as the C kernel names it
#
# so `calibrate/ingest_ara_csv.py` (bingo) maps a row straight onto inputs/op_lib/simd/<op_id>.hjson
# with no translation table. Keep the two in step: the op ids here ARE the framework's op ids.
#
# `op_node` is the C symbol the app actually calls, which for a regular op is the bare dispatcher --
# `__host_bingo_kernel_<op>` -- because precision is a runtime arg, not part of the symbol name.
# --------------------------------------------------------------------------------------------------
#
# This stays at the MEASUREMENT level: no fitting, no scaling, no derived rows. The bingo LUT holds RTL
# measurements and nothing else -- an op it has no points for REFUSES to price rather than being
# guessed at -- so a row that was not measured here must simply be absent.

import argparse
import csv
import glob
import os
import re
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, "../../../../../"))
sys.path.insert(0, os.path.join(_ROOT, "util/automation_scripts"))
from bingo_trace_gather import parse_task_order  # noqa: E402

_ARA_DIR = _THIS
_DEFAULT_OUT = os.path.join(_ARA_DIR, "ara_cycles.csv")

# Every regular kernel is reached through the BARE dispatcher; precision is a runtime arg. The two
# conversions are their own C symbols (their "precision" is an in->out pair, not a mode).
_CONVERSION_NODE = {
    "quantize": {"f32i8": "__host_bingo_kernel_quantize_f32i8",
                 "f16i8": "__host_bingo_kernel_quantize_f16i8"},
    "dequantize": {"i32f32": "__host_bingo_kernel_dequantize_i32f32"},
}


def op_node(kernel: str, prec: str) -> str:
    """The C symbol that ran: the bare dispatcher, or a conversion's own per-pair symbol."""
    if kernel in _CONVERSION_NODE:
        return _CONVERSION_NODE[kernel].get(prec, f"__host_bingo_kernel_{kernel}")
    return f"__host_bingo_kernel_{kernel}"


def op_id(kernel: str, prec: str) -> str:
    """The BINGO op id: precision is a SUFFIX. Uniform for regular ops and conversions alike."""
    return f"{kernel}_{prec}"


# CYCLES,<kernel>,<prec>,<N>,<rep>,<cycles>   -- the precision-swept format.
_CYCLES_RE = re.compile(r"CYCLES,([A-Za-z0-9_]+),([A-Za-z0-9]+),(\d+),(\d+),(\d+)")
# CYCLES,<kernel>,<N>,<rep>,<cycles>          -- 5-field form, carrying NO precision token. The two
# conversion apps print this; each runs a single in->out pair, so its precision is filled in from
# _LEGACY_PREC. A kernel printing this form without an entry there cannot be typed and its lines are
# dropped, so an app must emit a prec token to be swept at more than one precision.
_LEGACY_RE = re.compile(r"CYCLES,([A-Za-z0-9_]+),(\d+),(\d+),(\d+)\s*$")
_LEGACY_PREC = {"quantize": "f32i8", "dequantize": "i32f32"}


def _parse_cycles(uart_log):
    """(kernel, prec) -> {N: [cycles, ...]} from one UART log; plus the kernels seen in legacy form."""
    out, legacy = {}, set()
    with open(uart_log, errors="replace") as f:
        for line in f:
            m = _CYCLES_RE.search(line)
            if m:
                kernel, prec = m.group(1), m.group(2)
                n, cyc = int(m.group(3)), int(m.group(5))
            else:
                m = _LEGACY_RE.search(line)
                if not m:
                    continue
                kernel = m.group(1)
                prec = _LEGACY_PREC.get(kernel)
                if prec is None:
                    continue                 # a 5-field line for a kernel we cannot type: skip it
                n, cyc = int(m.group(2)), int(m.group(4))
                legacy.add(kernel)
            out.setdefault((kernel, prec), {}).setdefault(n, []).append(cyc)
    return out, legacy


def gather_all(ci_dir, out_csv, only=None, verbose=True):
    agg, legacy = {}, set()
    logs = sorted(glob.glob(os.path.join(ci_dir, "task_*", "bin", "uart_chip_0_0.log")))
    if not logs:
        print(f"No UART logs found under {ci_dir}/task_*/bin/uart_chip_0_0.log")
        return 0
    for log in logs:
        got, leg = _parse_cycles(log)
        legacy |= leg
        for key, by_n in got.items():
            dst = agg.setdefault(key, {})
            for n, cycs in by_n.items():
                dst.setdefault(n, []).extend(cycs)

    rows = []
    for kernel, prec in sorted(agg):
        if only and kernel not in only:
            continue
        points = [(n, round(sum(c) / len(c))) for n, c in sorted(agg[(kernel, prec)].items())]
        for n, cyc in points:
            rows.append({"op_id": op_id(kernel, prec), "op_name": kernel, "prec": prec,
                         "op_node": op_node(kernel, prec), "n": n, "cycles": cyc})
        if verbose:
            pts = ", ".join(f"n={n}:{cyc}" for n, cyc in points)
            print(f"{op_id(kernel, prec):22} {len(points)} points  [{pts}]")

    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["op_id", "op_name", "prec", "op_node", "n", "cycles"])
            w.writeheader()
            w.writerows(rows)

    if legacy:
        print(f"\n[legacy format] {', '.join(sorted(legacy))} still print CYCLES with NO precision "
              f"field; typed here by assumption ({_LEGACY_PREC}). Update those apps to emit a prec "
              f"token so the format is uniform (and so quantize_f16i8 can be swept at all).")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-yaml", default=os.path.join(_ARA_DIR, "task_ara.yaml"))
    ap.add_argument("--ci-dir", default=_ARA_DIR)
    ap.add_argument("--out", default=_DEFAULT_OUT)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    if os.path.exists(args.task_yaml):
        order = parse_task_order(args.task_yaml)
        print(f"Expected {len(order)} ara tasks from {os.path.basename(args.task_yaml)}\n")

    n = gather_all(args.ci_dir, args.out, only=args.only)
    print(f"\nWrote {n} row(s) to {args.out}")
    print("Ingest into the framework's LUT (from the bingo repo):\n"
          "    ./.venv/bin/python -m calibrate.ingest_ara_csv <this csv>")


if __name__ == "__main__":
    main()
