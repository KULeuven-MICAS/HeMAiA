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

Run it
------
From this directory, with ``vsim`` on your PATH and SSH access to the private
vendor repos (this flow clones the TSMC16 macros, D2D link, and vendor PLL):

    # Use the defaults set below (offload_bingo_hw / multi_chip / gemm_stacked_1cluster):
    python3 2_start_tapeout_sim.py

    # Override any SW workload knob from the CLI. The constants below are only
    # the defaults; the fixed tapeout cfg / vendor-module flags are not
    # exposed as arguments.
    python3 2_start_tapeout_sim.py \
        --host-app-type offload_bingo_hw \
        --workload gemm_stacked_1cluster \
        --dev-app snax-bingo-offload

    # List every flag and its current default:
    python3 2_start_tapeout_sim.py --help

Simulation time: >2h for modest workloads
"""

from __future__ import annotations

from pathlib import Path

from test_util import parse_workload_args, run_sim_flow


# Default SW workload selection. Each value is overridable on the CLI via
# --host-app-type / --chip-type / --workload / --dev-app (see --help); the
# constant is used whenever the matching flag is omitted.
HOST_APP_TYPE = "offload_bingo_hw"  # host_only | offload_legacy | offload_bingo_hw | offload_bingo_sw
CHIP_TYPE = "multi_chip"
WORKLOAD = "gemm_stacked_1cluster"
DEV_APP = "snax-bingo-offload"

def main() -> None:
    args = parse_workload_args(
        default_host_app_type=HOST_APP_TYPE,
        default_chip_type=CHIP_TYPE,
        default_workload=WORKLOAD,
        default_dev_app=DEV_APP,
        description=__doc__,
    )
    run_sim_flow(
        cfg_name="hemaia_tapeout.hjson",
        sim_cfg_name="sim_rtl_with_pll.hjson",
        # Tapeout sim needs all three vendor modules.
        with_macro=True,   # tech_cells_tsmc16
        with_d2d=True,     # hemaia_d2d_link
        with_pll=True,     # hemaia_clk_rst_controller
        # Workload selection (CLI-overridable; constants above are the defaults).
        host_app_type=args.host_app_type,
        chip_type=args.chip_type,
        workload=args.workload,
        dev_app=args.dev_app,
        script_path=Path(__file__),
    )


if __name__ == "__main__":
    main()
