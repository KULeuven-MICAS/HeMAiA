#!/usr/bin/env python3
r"""
HeMAiA single-chiplet RTL sim
=============================
Uses ``target/rtl/cfg/hemaia_singlechip.hjson`` and
``target/sim/cfg/sim_rtl.hjson``. No private vendor modules are required:
no TSMC16 macros, no D2D link, no vendor PLL.
It is used as a quick sanity check that the RTL sim flow can run end-to-end on a single chiplet configuration, without the overhead of setting up the full multi-chiplet environment.

This can be used to test the current single-cluster and the upcoming multi-cluster RTL sim flow.

Run it
------
From this directory, with ``vsim`` on your PATH:

    # Use the defaults set below (offload_legacy / single_chip / no workload):
    python3 0_start_single_chiplet_sim.py

    # Override any SW workload knob from the CLI. The constants below are only
    # the defaults; the fixed single-chip cfg / vendor-module flags are not
    # exposed as arguments.
    python3 0_start_single_chiplet_sim.py \
        --host-app-type offload_bingo_hw \
        --workload gemm_stacked_1cluster \
        --dev-app snax-bingo-offload

    # List every flag and its current default:
    python3 0_start_single_chiplet_sim.py --help

Simulation time: ~5 min
"""

from __future__ import annotations

from pathlib import Path

from test_util import parse_workload_args, run_sim_flow

# Default SW workload selection. Each value is overridable on the CLI via
# --host-app-type / --chip-type / --workload / --dev-app (see --help); the
# constant is used whenever the matching flag is omitted.
HOST_APP_TYPE = "offload_legacy" # host_only | offload_legacy | offload_bingo_hw | offload_bingo_sw
CHIP_TYPE = "single_chip"
WORKLOAD = "None"
DEV_APP = "snax-versacore-matmul-profile"

# HOST_APP_TYPE = "offload_bingo_hw" 
# CHIP_TYPE = "single_chip"
# WORKLOAD = "gemm_stacked_1cluster"
# DEV_APP = "snax-bingo-offload"



def main() -> None:
    args = parse_workload_args(
        default_host_app_type=HOST_APP_TYPE,
        default_chip_type=CHIP_TYPE,
        default_workload=WORKLOAD,
        default_dev_app=DEV_APP,
        description=__doc__,
    )
    run_sim_flow(
        cfg_name="hemaia_singlechip.hjson",
        sim_cfg_name="sim_rtl.hjson",
        # Single chiplet needs none of the private vendor modules.
        with_macro=False,  # tech_cells_tsmc16
        with_d2d=False,    # hemaia_d2d_link
        with_pll=False,    # hemaia_clk_rst_controller
        # Workload selection (CLI-overridable; constants above are the defaults).
        host_app_type=args.host_app_type,
        chip_type=args.chip_type,
        workload=args.workload,
        dev_app=args.dev_app,
        script_path=Path(__file__),
    )


if __name__ == "__main__":
    main()
