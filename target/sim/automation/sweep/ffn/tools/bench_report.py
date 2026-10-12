#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""bench_report.py <run dir> [bench variant, default gemm_bench_paired]: the GEMM shapes of a GEMM bench run
(common/gemm_bench_mc.py), from its scheduler log.

For each shape, both passes (r0 also fetches its instructions): microseconds, array steps (row tiles x K steps x
column tiles), nanoseconds a step (2.0 at the (16,4,16) array's peak at 500 MHz) and the fraction of peak; then the
gate|up shape next to a DMA copy and next to a SIMD add, against its quiet second pass."""
import collections
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE.split("/target/sim/")[0]
sys.path.insert(0, f"{REPO}/target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/ffn/common")
from gemm_bench_mc import SHAPES  # noqa: E402

run = sys.argv[1]
variant = sys.argv[2] if len(sys.argv) > 2 else "gemm_bench_paired"
task = next(os.path.join(run, d) for d in os.listdir(run) if d.startswith("task_"))
dfg = f"{REPO}/target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/ffn/dsv2_l0_prefill_tiny/{variant}/final_dfg.csv"
names = {int(r["ID"]): r["Name"] for r in csv.DictReader(open(dfg))}
ev = collections.defaultdict(dict)
for ln in open(os.path.join(task, "bin", "bingo_progress.log"), errors="replace"):
    if ln.startswith("[BINGO EV]"):
        kv = dict(re.findall(r"(\w+)=(\S+)", ln))
        n = names.get(int(kv["id"]))
        if n:
            ev[n].setdefault(kv["ev"], int(kv["t"]) / 1e6)
dur = {n[len("bench_"):-len("_k00_cl0")]: e["D"] - e["S"] for n, e in ev.items()
       if n.startswith("bench_") and "S" in e and "D" in e}
print(f"{'shape':10s} {'M':>2s} {'K':>4s} {'N':>3s} {'steps':>7s} {'pass 0 us':>9s} {'pass 1 us':>9s} {'ns/step':>8s} {'% peak':>7s}")
for name, m, k, n in SHAPES:
    st = m * k * n
    r0, r1 = dur.get(f"{name}_r0"), dur.get(f"{name}_r1")
    if r1 is None:
        print(f"{name:10s} {m:2d} {k:4d} {n:3d} {st:7d}   (not done)")
        continue
    print(f"{name:10s} {m:2d} {k:4d} {n:3d} {st:7d} {r0:9.2f} {r1:9.2f} {1e3 * r1 / st:8.3f} {100 * 2.0 * st / (1e3 * r1):6.1f}%")
q = dur.get("gu32_r1")
for probe in ("gu32_dma", "gu32_simd"):
    if probe in dur and q:
        print(f"{probe:10s} {dur[probe]:7.2f} us against {q:.2f} quiet ({100 * (dur[probe] / q - 1):+.1f} %)")
