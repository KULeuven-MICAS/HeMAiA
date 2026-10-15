#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""trim_traces.py <task dir> [...]: once a run's window is over (its SYNC line is in the root's
UART), take every task's span from the core traces into tasks.csv (trace_ffn.tasks, as
parse_ffn.py does), mark it (.tasks_extracted, which parse_ffn.py then keeps), and truncate the
trace files. What a simulator writes after the window -- the checks, every idle core's polling --
can outgrow the disk quota; the measurement needs none of it. UART, the SRAM probe log and the
run's other files are left alone. A task dir already marked is only truncated again."""

import csv
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trace_ffn  # noqa: E402

REPO = Path(str(HERE).split("/target/sim/")[0])
FAMILIES = ("ffn", "moe", "fa")


def wl_dir_of(task: Path) -> Path:
    fam = next((f for f in FAMILIES if f"_{f}_" in task.name), "ffn")
    variant = re.sub(r"_snax-[\w-]+$", "", task.name.split(f"_{fam}_", 1)[-1])
    wls = REPO / "target/sw/host/apps/offload_bingo_hw/multi_chip/workloads" / fam
    return next((d for d in wls.glob("*/*") if (d / "params.hjson").exists() and
                 str(d.relative_to(wls)).replace("/", "_") == variant), wls / variant)


def trim(task: Path, force: bool = False) -> str:
    """force: a run that was stopped (a hang) has no SYNC line; take what its traces hold anyway and
    keep the root chip's compute and xDMA traces (under 400 MiB each) for the analysis."""
    uart = task / "bin" / "uart_chip_0_0.log"
    if not force and (not uart.exists() or "SYNC[hw]" not in uart.read_text(errors="replace")):
        return "window not over"
    marker = task / ".tasks_extracted"
    if not marker.exists():
        rows = trace_ffn.tasks(task, wl_dir_of(task))
        if not rows:
            return "no spans found; left untouched"
        with (task / "tasks.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r["start_ns"]))
        marker.write_text(f"{len(rows)} tasks\n")
    freed = 0
    for t in (task / "bin" / "logs").glob("trace_chip_*"):
        try:
            if force and re.match(r"trace_chip_00_hart_0000[123567]\.dasm$", t.name) and t.stat().st_blocks * 512 < 400 * 2**20:
                continue
            freed += t.stat().st_blocks * 512
            os.truncate(t, 0)
        except OSError:
            pass
    return f"tasks.csv kept; traces truncated ({freed / 2**30:.1f} GiB)"


if __name__ == "__main__":
    force = "--force" in sys.argv
    for a in [x for x in sys.argv[1:] if not x.startswith("--")]:
        for task in (sorted(Path(a).glob("task_*")) if not Path(a).name.startswith("task_") else [Path(a)]):
            print(f"[trim_traces] {task.parent.name}: {trim(task, force)}", flush=True)
