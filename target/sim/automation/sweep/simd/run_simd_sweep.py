#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""
HeMAiA SIMD LUT sweep runner
============================

Builds and runs ``task_simd.yaml`` (one task per SIMD workload) through the shared
:class:`HeMAiASimRunner` on the single-chip 16 MiB config, whose cluster is
``snax_split_cluster``: four engines on four harts, with the SIMD operator chain
(EW0 -> Map -> Reduce -> EW1 -> Fp16ToInt8) on hart 1.

That split is why this sweep is separate from the xDMA one. The operator chain used
to hang off the xDMA's reader-extension socket, so one sweep measured both; now each
engine has its own hart, its own CSR bank and its own trace markers
(``BINGO_TRACE_SIMD_*`` vs ``BINGO_TRACE_XDMA_*``), and a run that mixed them would
be attributing one engine's cycles to the other's cost model.

Defaults to the VCS engine with no waveform recording.

After this finishes, post-process the per-task traces into a cycle CSV:

    python3 gather_simd_luts.py    # -> simd_cycles.csv
    python3 simd_lut_report.py     # human-readable summary of the measured cycles

    python3 run_simd_sweep.py [-j JOBS] [-f task.yaml] [--engine vcs|vsim]
                              [--waveform 0|1] [--cfg CFG] [--sw-only]

``--sw-only`` reuses the already-compiled simulation and rebuilds only the per-task
app binaries; use it when iterating on the kernels rather than the hardware.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "util" / "automation_scripts"))
from sweep_runner import run_sweep_cli  # noqa: E402

if __name__ == "__main__":
    run_sweep_cli(
        __file__,
        description="Build and run the HeMAiA SIMD LUT sweep.",
        default_task_name="task_simd.yaml",
    )
