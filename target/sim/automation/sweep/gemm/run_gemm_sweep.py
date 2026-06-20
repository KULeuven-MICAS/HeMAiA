#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""
HeMAiA VersaCore GEMM precision LUT sweep runner.

Builds and runs ``task_gemm.yaml`` (one task per precision) through the shared
:class:`HeMAiASimRunner`, using the single-chip config — same pattern as the
xDMA/ara sweeps. Each task is a ``gemm_psweep_<prec>_1cluster`` workload that
sweeps the shared (M,K,N,array_shape) grid in ONE sim (precision can't be looped
in one binary), so the 5 precisions run in parallel.

After this finishes, post-process the per-task traces into a cycle CSV:

    python3 gather_gemm_luts.py    # -> gemm_cycles.csv  (per precision x shape)

    python3 run_gemm_sweep.py [-j JOBS] [-f task.yaml] [--engine vcs|vsim] [--waveform 0|1]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "util" / "automation_scripts"))
from sweep_runner import run_sweep_cli  # noqa: E402

if __name__ == "__main__":
    run_sweep_cli(
        __file__,
        description="Build and run the HeMAiA GEMM precision LUT sweep.",
        default_task_name="task_gemm.yaml",
    )
