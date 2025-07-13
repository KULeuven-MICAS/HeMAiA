#!/bin/bash
script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cfg_name="hemaia_tapeout.hjson"

make -C "$script_dir/../.." hemaia_system_vsim_preparation SIM_WITH_TSMC16_MACRO=1
make -C "$script_dir/../.." hemaia_system_vsim
