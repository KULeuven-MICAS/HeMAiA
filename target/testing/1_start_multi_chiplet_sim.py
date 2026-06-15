#!/usr/bin/env python3
r"""
HeMAiA multi-chiplet RTL sim
============================
Uses ``target/rtl/cfg/hemaia_multichip_ci.hjson`` and
``target/sim/cfg/sim_rtl.hjson``. Pulls the TSMC16 macros and the D2D-link
private repos but does NOT pull the vendor PLL controller.

Multi-chiplet configuration (2x2 compute + 1 memory chip on the virtual
interposer). Use this flow to validate cross-chiplet workloads with the
D2D link and TSMC16 macros enabled, without the overhead of the vendor PLL.

The only difference between this multi-chiplet config and the full tapeout config is the PLL,
so this is a good intermediate step to validate the multi-chiplet sim flow before enabling the PLL.

Run it
------
From this directory, with ``vsim`` on your PATH and SSH access to the private
vendor repos (flows 1 and 2 clone them over SSH):

    # Use the defaults set below (offload_bingo_hw / multi_chip / gemm_stacked_1cluster):
    python3 1_start_multi_chiplet_sim.py

    # Override any SW workload knob from the CLI. The constants below are only
    # the defaults; the fixed multi-chip cfg / vendor-module flags are not
    # exposed as arguments.
    python3 1_start_multi_chiplet_sim.py \
        --host-app-type offload_bingo_hw \
        --workload gemm_stacked_1cluster \
        --dev-app snax-bingo-offload

    # List every flag and its current default:
    python3 1_start_multi_chiplet_sim.py --help

Simulation time: ~15-40min for modest workloads
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
        cfg_name="hemaia_multichip_ci.hjson",
        sim_cfg_name="sim_rtl.hjson",
        # Multi-chiplet uses macros + D2D, no vendor PLL.
        with_macro=True,   # tech_cells_tsmc16
        with_d2d=True,     # hemaia_d2d_link
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
