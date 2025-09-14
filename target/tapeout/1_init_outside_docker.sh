#!/bin/bash
script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cfg_name="hemaia_tapeout.hjson"
if [ ! -d "$script_dir/../../hw/hemaia/tech_cells_tsmc16" ]; then
    git clone https://github.com/IveanEx/tech_cells_tsmc16.git "$script_dir/../../hw/hemaia/tech_cells_tsmc16"
fi
if [ ! -d "$script_dir/../../hw/hemaia/hemaia_d2d_link" ]; then
    git clone https://github.com/IveanEx/hemaia_d2d_link.git "$script_dir/../../hw/hemaia/hemaia_d2d_link"
fi
if [ ! -d "$script_dir/dc_work_hemaia" ]; then
    git clone https://github.com/IveanEx/dc_work_hemaia.git "$script_dir/dc_work_hemaia"
fi

bender_local_file="$script_dir/../../Bender.local"

if ! grep -q "hemaia_d2d_link" "$bender_local_file"; then
    echo '  hemaia_d2d_link:    { path: hw/hemaia/hemaia_d2d_link }' >> "$bender_local_file"
else
    # Replace the existing hemaia_d2d_link line
    sed -i '/hemaia_d2d_link:/c\  hemaia_d2d_link:    { path: hw/hemaia/hemaia_d2d_link }' "$bender_local_file"
fi

if ! grep -q "tech_cells_generic" "$bender_local_file"; then
    echo '  tech_cells_generic:  { path: hw/hemaia/tech_cells_tsmc16 }' >> "$bender_local_file"
else
    # Replace the existing tech_cells_generic line
    sed -i '/tech_cells_generic:/c\  tech_cells_generic:  { path: hw/hemaia/tech_cells_tsmc16 }' "$bender_local_file"
fi
