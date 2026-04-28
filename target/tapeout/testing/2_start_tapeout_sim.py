#!/usr/bin/env python3
"""
HeMAiA tapeout RTL sim (with vendor PLL)
========================================
Uses ``target/rtl/cfg/hemaia_tapeout.hjson`` and
``target/sim/cfg/sim_rtl_with_pll.hjson``. Pulls the full set of private
vendor repos: TSMC16 macros, D2D link, and the vendor PLL controller.

This is the full tapeout configuration: 2x2 compute + 1 memory chip on the
virtual interposer with ``use_vendor_pll: true``. Use this flow for the
final pre-silicon sign-off pass.

Simulation time: >2h for modest workloads
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
        cfg_name="hemaia_tapeout.hjson",
        sim_cfg_name="sim_rtl_with_pll.hjson",
        # Tapeout sim needs all three vendor modules.
        with_macro=True,   # tech_cells_tsmc16
        with_d2d=True,     # hemaia_d2d_link
        with_pll=True,     # hemaia_clk_rst_controller
        # Workload selection.
        host_app_type=HOST_APP_TYPE,
        chip_type=CHIP_TYPE,
        workload=WORKLOAD,
        dev_app=DEV_APP,
        script_path=Path(__file__),
    )


if __name__ == "__main__":
    main()
