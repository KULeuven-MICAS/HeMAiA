#!/bin/bash
script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cfg_name="hemaia_tapeout.hjson"

cp -f "$script_dir/../sim/testharness/testharness_macro.sv" "$script_dir/../sim/testharness/testharness.sv"
cp -f "$script_dir/../sim/work-vsim/compile_macro.vsim.tcl" "$script_dir/../sim/work-vsim/compile.vsim.tcl"

make -C "$script_dir/../.." hemaia_system_vsim
