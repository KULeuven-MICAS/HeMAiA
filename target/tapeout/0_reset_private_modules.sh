#!/bin/bash
# Author: Yunhao Deng <yunhao.deng@kuleuven.be>
#         Fanchen Kong <fanchen.kong@kuleuven.be>

# This script is meant to be run outside of the Docker container, on the host machine. It removes all the vendor-specific modules and DC scripts that were initialized by 1_init_outside_docker.sh. This is useful for cleaning up the environment before starting a new tapeout iteration, or for resetting the environment if something goes wrong.

script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Repository root (two levels up from this script)
repo_root="$(cd "$script_dir/../../" && pwd)"

# Remove DC scripts
rm -rf "$script_dir/dc_work_hemaia"

# Remove vendor-specific modules
rm -rf "$repo_root/hw/hemaia/tech_cells_tsmc16"
rm -rf "$repo_root/hw/hemaia/hemaia_d2d_link"
rm -rf "$repo_root/hw/hemaia/hemaia_clk_rst_controller"

# Reset Bender.local entries to empty paths
# Path to Bender.local in the repo root
bender_local_file="$repo_root/Bender.local"
# Use safe substitutions to replace matching lines in-place without adding blank lines
sed -i '/^[[:space:]]*hemaia_d2d_link:/d' "$bender_local_file"
sed -i 's|^[[:space:]]*tech_cells_generic:.*$|  tech_cells_generic: { git: https://github.com/KULeuven-MICAS/tech_cells_generic.git, rev    : master }|' "$bender_local_file"
sed -i '/^[[:space:]]*hemaia_clk_rst_controller:/d'  "$bender_local_file"

# Remove the Bender.lock
rm -f "$repo_root/Bender.lock"
