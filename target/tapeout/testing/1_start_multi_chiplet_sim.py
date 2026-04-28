#!/usr/bin/env python3
"""
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

Simulation time: ~15-40min for modest workloads
"""

from __future__ import annotations

from pathlib import Path

from test_util import run_sim_flow

HOST_APP_TYPE = "offload_bingo_hw"  # host_only | offload_legacy | offload_bingo_hw | offload_bingo_sw
CHIP_TYPE = "multi_chip"
WORKLOAD = "gemm_stacked_1cluster"
DEV_APP = "snax-bingo-offload"

def main() -> None:
    run_sim_flow(
        cfg_name="hemaia_multichip_ci.hjson",
        sim_cfg_name="sim_rtl.hjson",
        # Multi-chiplet uses macros + D2D, no vendor PLL.
        with_macro=True,   # tech_cells_tsmc16
        with_d2d=True,     # hemaia_d2d_link
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
