#!/usr/bin/env python3
r"""
HeMAiA tapeout RTL sim (with vendor PLL)
========================================
Uses ``target/rtl/cfg/hemaia_tapeout.hjson`` and
``target/sim/cfg/sim_rtl_with_pll.hjson``. Pulls the full set of private
vendor repos: TSMC16 macros, D2D link, and the vendor PLL controller.

This is the full tapeout configuration: 2x2 compute + 1 memory chip on the
virtual interposer with ``use_vendor_pll: true``. Use this flow for the
final pre-silicon sign-off pass.

Defaults to the Questasim (vsim) engine with a recorded waveform/log so the run
is easy to debug; pass ``--engine vcs --waveform 0`` for a fast batch run.
The run lands in ``test/task_0/bin/`` (transcript + vsim.wlf).

Run it
------
From this directory, with ``vsim`` on your PATH and SSH access to the private
vendor repos (this flow clones the TSMC16 macros, D2D link, and vendor PLL):

    # Use the defaults set below (offload_bingo_hw / multi_chip / gemm_stacked_1cluster):
    python3 2_start_tapeout_sim.py

    # Override any SW workload knob from the CLI:
    python3 2_start_tapeout_sim.py \
        --host-app-type offload_bingo_hw \
        --workload gemm_stacked_1cluster \
        --dev-app snax-bingo-offload

    # List every flag and its current default:
    python3 2_start_tapeout_sim.py --help

Simulation time: >2h for modest workloads
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT.parents[4]  # target/sim/automation/test -> repo root
sys.path.insert(0, str(_REPO_ROOT / "util" / "automation_scripts"))

from hemaia_sim_runner import (  # noqa: E402
    HeMAiASimRunner, make_task, parse_workload_args,
)

# Default SW workload selection (each value is CLI-overridable; see --help).
HOST_APP_TYPE = "offload_bingo_hw"  # host_only | offload_legacy | offload_bingo_hw | offload_bingo_sw
CHIP_TYPE = "multi_chip"
WORKLOAD = "gemm_stacked_1cluster"
DEV_APP = "snax-bingo-offload"
ENGINE = "vsim"
SIM_WITH_WAVEFORM = 1


def main() -> None:
    args = parse_workload_args(
        default_host_app_type=HOST_APP_TYPE,
        default_chip_type=CHIP_TYPE,
        default_workload=WORKLOAD,
        default_dev_app=DEV_APP,
        default_engine=ENGINE,
        default_waveform=SIM_WITH_WAVEFORM,
        description=__doc__,
    )
    runner = HeMAiASimRunner(
        repo_root=_REPO_ROOT,
        output_dir=_SCRIPT.parent,
        engine=args.engine,
        with_waveform=bool(args.waveform),
        cfg="target/rtl/cfg/hemaia_tapeout.hjson",
        sim_cfg="target/sim/cfg/sim_rtl_with_pll.hjson",
        # Tapeout sim needs all three vendor modules.
        with_macro=True,
        with_d2d=True,
        with_pll=True,
        max_jobs=1,
    )
    runner.run([make_task(
        host_app_type=args.host_app_type,
        chip_type=args.chip_type,
        workload=args.workload,
        dev_app=args.dev_app,
    )])


if __name__ == "__main__":
    main()
