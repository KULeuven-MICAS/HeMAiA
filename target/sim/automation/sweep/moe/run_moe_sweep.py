#!/usr/bin/env python3
"""
HeMAiA MoE sweep runner
=======================

Builds and runs ``task_moe.yaml`` through the shared :class:`HeMAiASimRunner`.

This is the conditional-execution workload, so the interesting number is not only
whether it passes but WHICH experts ran. `moe_report.py` reads bingo_trace.json and
reports the per-expert lane spans, so a lane that was skipped shows up as absent
rather than as zero cycles.

    source ~/no_backup/src_hemaia_eda.sh
    python3 run_moe_sweep.py --engine vcs --waveform 0 -j 1 \
        --cfg ../../../rtl/cfg/hemaia_singlechiplet_16MB_4cluster.hjson

Add --sw-only to reuse an existing RTL build and rebuild only the app binaries.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "util" / "automation_scripts"))
from sweep_runner import run_sweep_cli  # noqa: E402

if __name__ == "__main__":
    run_sweep_cli(
        __file__,
        description="Build and run the HeMAiA MoE (conditional execution) sweep.",
        default_task_name="task_moe.yaml",
    )
