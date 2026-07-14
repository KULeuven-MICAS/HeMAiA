#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Shared trace-gathering helpers for the per-domain LUT sweep gatherers.

``gather_{ara,gemm,xdma}_luts.py`` all need the same building blocks: the task
order from a task YAML, the dasm -> txt -> bingo_trace.json conversion pipeline,
and the per-event ``dur_cc`` extraction. Those used to be copy-pasted (byte for
byte) into each gatherer; they live here once now. Only the op-specific event
name (``GEMM_FULL_RUN`` / ``XDMA_RUN``) differs and is passed in.
"""

import json
import os
import subprocess
from pathlib import Path

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
_GEN_TRACE = os.path.join(_ROOT, "util/trace/gen_trace.py")
_BINGO_TRACE = os.path.join(_ROOT, "util/bingo_trace/bingo_trace.py")
_PERF_HEADER = os.path.join(_ROOT, "target/sw/shared/runtime/perf_tracing.h")


def parse_task_order(task_yaml):
    """Return the ordered workload names from the task YAML (index = task idx)."""
    from hemaia_sim_runner import parse_tasks  # noqa: E402  (no PyYAML needed)
    return [t["workload"] for t in parse_tasks(Path(task_yaml))]


def task_dir(ci_dir, idx):
    """Path to the run directory for task *idx*, or None if it is absent.

    Run directories are named ``task_<idx>_<ci_name>``; look them up by index
    rather than assembling the name, since only the index is known here.
    """
    from hemaia_sim_runner import find_task_dir  # noqa: E402
    found = find_task_dir(ci_dir, idx)
    return str(found) if found is not None else None


def convert_traces(logs_dir, verbose=True):
    """spike-dasm | gen_trace.py --permissive for every .dasm lacking a fresh .txt."""
    import glob
    made = 0
    for dasm in sorted(glob.glob(os.path.join(logs_dir, "trace_chip_*_hart_*.dasm"))):
        txt = dasm[:-len(".dasm")] + ".txt"
        if os.path.exists(txt) and os.path.getmtime(txt) >= os.path.getmtime(dasm):
            continue
        cmd = f"spike-dasm < {dasm!r} | python3 {_GEN_TRACE!r} --permissive > {txt!r}"
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ! gen_trace failed on {os.path.basename(dasm)}: "
                  f"{r.stderr.strip()[:200]}")
            continue
        made += 1
    if verbose:
        print(f"  converted {made} .dasm -> .txt in {logs_dir}")


def run_bingo_trace(logs_dir):
    """Run bingo_trace.py over *logs_dir*; return the bingo_trace.json path or None."""
    out = os.path.join(logs_dir, "bingo_trace.json")
    r = subprocess.run(
        ["python3", _BINGO_TRACE, "--trace-header", _PERF_HEADER,
         "--log-dir", logs_dir, "--output", out],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ! bingo_trace failed: {r.stderr.strip()[:300]}")
        return None
    return out


def extract_run_cycles(bingo_json, event_name):
    """Ordered list of dur_cc values for X-events whose name contains *event_name*."""
    with open(bingo_json) as f:
        trace = json.load(f)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    runs = [e for e in events
            if e.get("ph") == "X" and event_name in str(e.get("name", ""))]
    runs.sort(key=lambda e: e.get("ts", 0))
    return [int(e.get("args", {}).get("dur_cc", 0)) for e in runs]
