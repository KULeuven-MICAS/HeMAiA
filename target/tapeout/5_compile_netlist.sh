#!/bin/bash
script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cfg_name="hemaia_tapeout.hjson"

cd "$script_dir/dc_work_hemaia/scripts"
lc_shell -f conv_mem_macro.tcl
dc_shell -f syn.tcl
