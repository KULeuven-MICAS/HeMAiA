#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""
HeMAiA DeepSeek-V2-Lite layer-1 runner
======================================

Builds and runs the dsv2 workloads on ``hemaia_twochiplet_16MBL3_4cluster.hjson``: one
compute chiplet with four snax_split_clusters and, east of it, the memory chiplet whose
simulated HBM holds the weights. It is the only cfg with an HBM, and the flow is the one the
HBM was validated with: the macro and D2D-link private modules pulled, no vendor PLL.

Only the task list's apps are built (``build_sw_fleet=False``): the ~96-app fleet does not
target this cfg, and building it costs most of the setup.

    source ~/no_backup/src_hemaia_eda.sh
    python3 run_dsv2_sweep.py [-f task_dsv2.yaml] [--sw-only | --reuse-build]

``--sw-only`` reuses the RTL and the compiled simulation of a previous run at the same
``--cfg`` and rebuilds only the app binaries.

THE RUNNER'S EXIT CODE IS NOT THE VERDICT. Read the UART; the checks run upstream first and
the first FAIL ends the run:

    grep -E "Check|PASS|FAIL" task_*/bin/uart_chip_0_0.log
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "util" / "automation_scripts"))
from sweep_runner import run_sweep_cli  # noqa: E402

if __name__ == "__main__":
    run_sweep_cli(
        __file__,
        description="Build and run the DeepSeek-V2-Lite layer-1 workloads.",
        default_task_name="task_dsv2.yaml",
        cfg="target/rtl/cfg/hemaia_twochiplet_16MBL3_4cluster.hjson",
        with_macro=True,
        with_d2d=True,
        build_sw_fleet=False,
    )
