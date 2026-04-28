#!/usr/bin/env python3
"""
HeMAiA single-chiplet RTL sim
=============================
Uses ``target/rtl/cfg/hemaia_singlechip.hjson`` and
``target/sim/cfg/sim_rtl.hjson``. No private vendor modules are required:
no TSMC16 macros, no D2D link, no vendor PLL.
It is used as a quick sanity check that the RTL sim flow can run end-to-end on a single chiplet configuration, without the overhead of setting up the full multi-chiplet environment.

This can be used to test the current single-cluster and the upcoming multi-cluster RTL sim flow.

Simulation time: ~5 min
"""

from __future__ import annotations

from pathlib import Path

from test_util import run_sim_flow

HOST_APP_TYPE = "offload_legacy" # host_only | offload_legacy | offload_bingo_hw | offload_bingo_sw
CHIP_TYPE = "single_chip"
WORKLOAD = "None"
DEV_APP = "snax-versacore-matmul-profile"

# HOST_APP_TYPE = "offload_bingo_hw" 
# CHIP_TYPE = "single_chip"
# WORKLOAD = "gemm_stacked_1cluster"
# DEV_APP = "snax-bingo-offload"



def main() -> None:
    run_sim_flow(
        cfg_name="hemaia_singlechip.hjson",
        sim_cfg_name="sim_rtl.hjson",
        # Single chiplet needs none of the private vendor modules.
        with_macro=False,  # tech_cells_tsmc16
        with_d2d=False,    # hemaia_d2d_link
        with_pll=False,    # hemaia_clk_rst_controller
        # Workload selection.
        host_app_type=HOST_APP_TYPE,
        chip_type=CHIP_TYPE,
        workload=WORKLOAD,
        dev_app=DEV_APP,
        script_path=Path(__file__),
    )


if __name__ == "__main__":
    main()
