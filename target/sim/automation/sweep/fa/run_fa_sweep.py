#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""
HeMAiA FlashAttention runner
============================

Builds and runs ``task_fa.yaml`` through the shared :class:`HeMAiASimRunner` on the
single-chip 16 MiB config with ``snax_split_cluster``.

Its own directory, not a task inside the SIMD sweep, because the task_<idx> run dirs
live next to the runner and a second task list in the same place would delete the
first one's results.

    python3 run_fa_sweep.py [-j JOBS] [--engine vcs|vsim] [--waveform 0|1]
                            [--cfg CFG] [--sw-only] [-f TASK_YAML]

``--sw-only`` reuses the RTL and the compiled simulation from a previous run at the
same ``--cfg`` and rebuilds only the app binaries -- which is what you want while
iterating on the kernels.

Afterwards:

    python3 gather_fa_cycles.py    # per-engine cycles, against the reference's numbers
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "util" / "automation_scripts"))
from sweep_runner import run_sweep_cli  # noqa: E402

if __name__ == "__main__":
    run_sweep_cli(
        __file__,
        description="Build and run HeMAiA FlashAttention.",
        default_task_name="task_fa_both.yaml",
    )
