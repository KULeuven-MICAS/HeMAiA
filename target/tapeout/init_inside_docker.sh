#!/bin/bash
script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cfg_name="hemaia_tapeout.hjson"
bender_file="$script_dir/../../Bender.local"
if ! grep -q "hemaia_d2d_link" "$bender_file"; then
    echo '  hemaia_d2d_link:    { path: hw/hemaia/hemaia_d2d_link }' >> "$bender_file"
else
    # Replace the existing hemaia_d2d_link line
    sed -i '/hemaia_d2d_link:/c\  hemaia_d2d_link:    { path: hw/hemaia/hemaia_d2d_link }' "$bender_file"
fi

if ! grep -q "tech_cells_generic" "$bender_file"; then
    echo '  tech_cells_generic:  { path: hw/hemaia/tech_cells_tsmc16 }' >> "$bender_file"
else
    # Replace the existing tech_cells_generic line
    sed -i '/tech_cells_generic:/c\  tech_cells_generic:  { path: hw/hemaia/tech_cells_tsmc16 }' "$bender_file"
fi
make -C $script_dir/../.. sw CFG_OVERRIDE=target/rtl/cfg/$cfg_name
make -C $script_dir/../.. bootrom CFG_OVERRIDE=target/rtl/cfg/$cfg_name
make -C $script_dir/../.. rtl CFG_OVERRIDE=target/rtl/cfg/$cfg_name
bender script synopsys -t rtl -t synthesis -t occamy -t hemaia -t tsmc16 $(cat ../rtl/src/bender_targets.tmp) > $script_dir/flist.tcl
echo "Initialization complete. You can copy the entire folder to the synthesis machine."
