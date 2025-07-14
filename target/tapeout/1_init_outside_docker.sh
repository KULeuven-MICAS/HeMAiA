#!/bin/bash
script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cfg_name="hemaia_tapeout.hjson"
if [ ! -d "$script_dir/../../hw/hemaia/tech_cells_tsmc16" ]; then
    git clone https://github.com/IveanEx/tech_cells_tsmc16.git "$script_dir/../../hw/hemaia/tech_cells_tsmc16"
fi
if [ ! -d "$script_dir/../../hw/hemaia/hemaia_d2d_link" ]; then
    git clone https://github.com/KULeuven-MICAS/hemaia_d2d_link.git "$script_dir/../../hw/hemaia/hemaia_d2d_link"
fi
if [ ! -d "$script_dir/dc_work_hemaia" ]; then
    git clone https://github.com/KULeuven-MICAS/dc_work_hemaia.git "$script_dir/dc_work_hemaia"
fi
