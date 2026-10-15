#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""trim_live.py [--once] <run dir> [...]: keep a running simulation inside the disk quota.

Every hart writes a trace line a cycle, so a 16-chip run fills the quota with idle cores: the host
spends milliseconds initialising tables while every device core polls. Two stages, once a minute:

  window not over   each device trace (.dasm) loses the head it wrote before its first kernel
                    bracket (the magic NOP pair of trace_ffn.py), as a hole punched into the file
                    (fallocate --punch-hole, KEEP_SIZE): the blocks are freed, the writer's offsets
                    stay valid, and a head with no bracket in it holds no task. Host traces are
                    left alone. trace_ffn.spans_dasm skips the hole.
  window over       the root's UART has its SYNC line: trim_traces.py takes every task's span into
                    tasks.csv and truncates the traces, as often as the simulation keeps writing.

Runs until every run dir's runner has printed RUN EXIT (<run dir>.log), or once with --once."""

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trace_ffn  # noqa: E402
import trim_traces  # noqa: E402

NEEDLES = tuple(f"DASM({trace_ffn.NOP(x):08x})".encode() for x in (trace_ffn.RUN_START, trace_ffn.RUN_END))
CHUNK = 64 << 20
BLOCK = 4096
MARGIN = 4096        # a trace line is under 1 KiB: the line holding a bracket stays whole
state = {}           # trace path -> [bytes scanned for a bracket, bytes punched, bracket seen]


def punch(path: Path, end: int, start: int) -> int:
    end -= end % BLOCK
    if end - start < 16 * BLOCK:
        return start
    r = subprocess.run(["fallocate", "--punch-hole", "--keep-size", "-o", str(start), "-l", str(end - start),
                        str(path)], capture_output=True)
    return end if r.returncode == 0 else start


def trim_device_trace(path: Path) -> None:
    st = state.setdefault(str(path), [0, 0, False])
    if st[2]:
        return
    size = path.stat().st_size
    with path.open("rb") as f:
        while st[0] < size:
            f.seek(max(st[0] - 64, st[1]))
            buf = f.read(min(CHUNK, size - max(st[0] - 64, st[1])))
            base = max(st[0] - 64, st[1])
            hit = min((i for i in (buf.find(n) for n in NEEDLES) if i >= 0), default=-1)
            if hit >= 0:
                st[2] = True
                st[1] = punch(path, base + hit - MARGIN, st[1])
                return
            st[0] = base + len(buf)
    st[1] = punch(path, st[0] - MARGIN, st[1])


def window_over(task: Path) -> bool:
    uart = task / "bin" / "uart_chip_0_0.log"
    return uart.exists() and "SYNC[hw]" in uart.read_text(errors="replace")


def step(run: Path) -> None:
    for task in sorted(run.glob("task_*")):
        logs = task / "bin" / "logs"
        if not logs.is_dir():
            continue
        if window_over(task):
            print(f"[trim_live] {task.name[:40]}: {trim_traces.trim(task)}", flush=True)
        else:
            for p in logs.glob("trace_chip_*_hart_*.dasm"):
                trim_device_trace(p)


def finished(run: Path) -> bool:
    log = run.with_suffix(".log")
    return log.exists() and "RUN EXIT" in log.read_text(errors="replace")


def main():
    once = "--once" in sys.argv
    runs = [Path(a).resolve() for a in sys.argv[1:] if not a.startswith("--")]
    while True:
        for r in runs:
            step(r)
        if once or all(finished(r) for r in runs):
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
