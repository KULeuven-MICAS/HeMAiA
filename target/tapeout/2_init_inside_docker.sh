#!/bin/bash
script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cfg_name="hemaia_tapeout.hjson"
HOST_APP_TYPE="offload_legacy"
CHIP_TYPE="single_chip"
WORKLOAD=""
DEV_APP="snax-test-integration"

make -C $script_dir/../.. clean
make -C $script_dir/../.. sw CFG_OVERRIDE=target/rtl/cfg/$cfg_name
make -C $script_dir/../.. apps HOST_APP_TYPE=$HOST_APP_TYPE CHIP_TYPE=$CHIP_TYPE WORKLOAD=$WORKLOAD DEV_APP=$DEV_APP
make -C $script_dir/../.. bootrom CFG_OVERRIDE=target/rtl/cfg/$cfg_name
make -C $script_dir/../.. rtl CFG_OVERRIDE=target/rtl/cfg/$cfg_name
make -C "$script_dir/../.." hemaia_system_vsim_preparation SIM_WITH_MACRO=1
mv -f "$script_dir/../sim/testharness/testharness.sv" "$script_dir/../sim/testharness/testharness_macro.sv"
mv -f "$script_dir/../sim/work-vsim/compile.vsim.tcl" "$script_dir/../sim/work-vsim/compile_macro.vsim.tcl"
make -C "$script_dir/../.." hemaia_system_vsim_preparation SIM_WITH_MACRO=1 SIM_WITH_NETLIST=1
mv -f "$script_dir/../sim/testharness/testharness.sv" "$script_dir/../sim/testharness/testharness_netlist.sv"
mv -f "$script_dir/../sim/work-vsim/compile.vsim.tcl" "$script_dir/../sim/work-vsim/compile_netlist.vsim.tcl"
bender script synopsys -t rtl -t synthesis -t occamy -t hemaia -t tsmc16 -t exclude_d2d_link_phy $(cat ../rtl/src/bender_targets.tmp) > $script_dir/dc_work_hemaia/scripts/flist.tcl
echo "Initialization complete. You can copy the entire folder to the synthesis machine."

