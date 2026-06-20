#!/usr/bin/env python3
"""
HeMAiA Ara FP32 op-cost sweep runner
====================================

Builds and runs ``task_ara.yaml`` (one task per decomposed ara_<kernel> app)
through the shared :class:`HeMAiASimRunner`, on the single-chip config (the
kernels run on the CVA6 host + Ara vector unit).  Defaults to the VCS engine
with no waveform recording.

After this finishes, post-process the per-task UART logs into a cycle CSV:

    python3 gather_ara_luts.py     # -> ara_cycles.csv
    python3 ara_lut_report.py      # human-readable summary of the measured cycles

    python3 run_ara_sweep.py [-j JOBS] [-f task.yaml] [--engine vcs|vsim] [--waveform 0|1]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "util" / "automation_scripts"))
from sweep_runner import run_sweep_cli  # noqa: E402

if __name__ == "__main__":
    run_sweep_cli(
        __file__,
        description="Build and run the HeMAiA Ara FP32 op-cost sweep.",
        default_task_name="task_ara.yaml",
    )
