#!/usr/bin/env python3
r"""
HeMAiA single-chiplet RTL sim
=============================
Uses ``target/rtl/cfg/hemaia_singlechiplet_1cluster.hjson`` and
``target/sim/cfg/sim_rtl.hjson``. No private vendor modules are required:
no TSMC16 macros, no D2D link, no vendor PLL.
It is used as a quick sanity check that the RTL sim flow can run end-to-end on a
single chiplet configuration, without the overhead of setting up the full
multi-chiplet environment.

Defaults to the Questasim (vsim) engine with a recorded waveform/log so the run
is easy to debug; pass ``--engine vcs --waveform 0`` for a fast batch run.
The run lands in ``test/task_0/bin/`` (transcript + vsim.wlf).

Run it
------
From this directory, with ``vsim`` on your PATH:

    # Use the defaults set below (offload_legacy / single_chip / no workload):
    python3 0_start_single_chiplet_sim.py

    # Override any SW workload knob from the CLI:
    python3 0_start_single_chiplet_sim.py \
        --host-app-type offload_bingo_hw \
        --workload gemm_stacked_1cluster \
        --dev-app snax-bingo-offload

    # List every flag and its current default:
    python3 0_start_single_chiplet_sim.py --help

Simulation time: ~5 min
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
HOST_APP_TYPE = "offload_legacy"  # host_only | offload_legacy | offload_bingo_hw | offload_bingo_sw
CHIP_TYPE = "single_chip"
WORKLOAD = "None"
DEV_APP = "versacore-matmul-profile-1cluster-4chip"
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
        cfg="target/rtl/cfg/hemaia_singlechiplet_16MB_1cluster.hjson",
        sim_cfg="target/sim/cfg/sim_rtl.hjson",
        # Single chiplet needs none of the private vendor modules.
        with_macro=False,
        with_d2d=False,
        with_pll=False,
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
