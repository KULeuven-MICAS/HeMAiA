#!/usr/bin/env python3
"""
HeMAiA xDMA LUT sweep runner
============================

Builds and runs ``task_xdma.yaml`` (one task per xDMA op) through the
shared :class:`HeMAiASimRunner`, using the single-chip config (16 MiB host
WIDE_SPM, no D2D/macro).  Defaults to the VCS engine with no waveform recording.

After this finishes, post-process the per-task traces into a cycle CSV:

    python3 gather_xdma_luts.py    # -> xdma_cycles.csv
    python3 xdma_lut_report.py     # human-readable summary of the measured cycles

    python3 run_xdma_sweep.py [-j JOBS] [-f task.yaml] [--engine vcs|vsim] [--waveform 0|1]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "util" / "automation_scripts"))
from sweep_runner import run_sweep_cli  # noqa: E402

if __name__ == "__main__":
    run_sweep_cli(
        __file__,
        description="Build and run the HeMAiA xDMA LUT sweep.",
        default_task_name="task_xdma.yaml",
    )
