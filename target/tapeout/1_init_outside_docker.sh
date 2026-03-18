#!/bin/bash

# Author: Yunhao Deng <yunhao.deng@kuleuven.be>
#         Fanchen Kong <fanchen.kong@kuleuven.be>
# The vendor-specific modules are stored in private repositories.
script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Allow overriding these from the environment or via CLI flags below.
# Examples: SIM_WITH_PLL=0 ./1_init_outside_docker.sh
# ./1_init_outside_docker.sh --pll=0 --d2d=0
# Defaults: MARCO=1, D2D=1, PLL=0
SIM_WITH_MARCO="${SIM_WITH_MARCO:-1}"
SIM_WITH_D2D="${SIM_WITH_D2D:-1}"
SIM_WITH_PLL="${SIM_WITH_PLL:-0}"

# Simple CLI parsing to override the same variables when calling the script.
while [ $# -gt 0 ]; do
    case "$1" in
        --marco=*) SIM_WITH_MARCO="${1#*=}"; shift ;;
        --d2d=*) SIM_WITH_D2D="${1#*=}"; shift ;;
        --pll=*) SIM_WITH_PLL="${1#*=}"; shift ;;
        -h|--help) echo "Usage: $0 [--d2d=0|1] [--marco=0|1] [--pll=0|1]"; echo "Defaults: d2d=1, marco=1, pll=0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate that options are either 0 or 1
for _var in SIM_WITH_D2D SIM_WITH_MARCO SIM_WITH_PLL; do
    val="${!_var}"
    if ! [[ "$val" =~ ^[01]$ ]]; then
        echo "Invalid value for ${_var}: '${val}' (must be 0 or 1)" >&2
        exit 1
    fi
done

# DC Scripts
if [ ! -d "$script_dir/dc_work_hemaia" ]; then
    git clone git@github.com:IveanEx/dc_work_hemaia.git "$script_dir/dc_work_hemaia"
else
    cd $script_dir/dc_work_hemaia || exit
    git pull
fi

# Initialize submodules
if [ "$SIM_WITH_MARCO" -eq 1 ]; then
    if [ ! -d "$script_dir/../../hw/hemaia/tech_cells_tsmc16" ]; then
        git clone git@github.com:IveanEx/tech_cells_tsmc16.git "$script_dir/../../hw/hemaia/tech_cells_tsmc16"
    else
        cd $script_dir/../../hw/hemaia/tech_cells_tsmc16 || exit
        git pull
    fi
else
    echo "SIM_WITH_MARCO=0: skipping tech_cells_tsmc16 checkout/update"
fi

if [ "$SIM_WITH_D2D" -eq 1 ]; then
    if [ ! -d "$script_dir/../../hw/hemaia/hemaia_d2d_link" ]; then
        git clone git@github.com:IveanEx/hemaia_d2d_link.git "$script_dir/../../hw/hemaia/hemaia_d2d_link"
    else
        cd $script_dir/../../hw/hemaia/hemaia_d2d_link || exit
        git pull
    fi
else
    echo "SIM_WITH_D2D=0: skipping hemaia_d2d_link checkout/update"
fi
if [ "$SIM_WITH_PLL" -eq 1 ]; then
    if [ ! -d "$script_dir/../../hw/hemaia/hemaia_clk_rst_controller" ]; then
        git clone git@github.com:IveanEx/hemaia_clk_rst_controller.git "$script_dir/../../hw/hemaia/hemaia_clk_rst_controller"
    else
        cd $script_dir/../../hw/hemaia/hemaia_clk_rst_controller || exit
        git pull
    fi
else
    echo "SIM_WITH_PLL=0: skipping hemaia_clk_rst_controller checkout/update"
fi

# Update Bender.local with the correct paths for the modules, but only if they are enabled. This way, users can disable certain modules without having to manually edit Bender.local.
bender_local_file="$script_dir/../../Bender.local"

if [ "$SIM_WITH_D2D" -eq 1 ]; then
    if ! grep -q "hemaia_d2d_link" "$bender_local_file"; then
        echo '  hemaia_d2d_link:    { path: hw/hemaia/hemaia_d2d_link }' >> "$bender_local_file"
    else
        # Replace the existing hemaia_d2d_link line
        sed -i '/hemaia_d2d_link:/c\  hemaia_d2d_link:    { path: hw/hemaia/hemaia_d2d_link }' "$bender_local_file"
    fi
else
    echo "SIM_WITH_D2D=0: leaving Bender.local entry for hemaia_d2d_link untouched"
fi

if [ "$SIM_WITH_MARCO" -eq 1 ]; then
    if ! grep -q "tech_cells_generic" "$bender_local_file"; then
        echo '  tech_cells_generic:  { path: hw/hemaia/tech_cells_tsmc16 }' >> "$bender_local_file"
    else
        # Replace the existing tech_cells_generic line
        sed -i '/tech_cells_generic:/c\  tech_cells_generic:  { path: hw/hemaia/tech_cells_tsmc16 }' "$bender_local_file"
    fi
else
    echo "SIM_WITH_MARCO=0: leaving Bender.local entry for tech_cells_generic untouched"
fi

if [ "$SIM_WITH_PLL" -eq 1 ]; then
    if ! grep -q "hemaia_clk_rst_controller" "$bender_local_file"; then
        echo '  hemaia_clk_rst_controller:  { path: hw/hemaia/hemaia_clk_rst_controller }' >> "$bender_local_file"
    else
        # Replace the existing hemaia_clk_rst_controller line
        sed -i '/hemaia_clk_rst_controller:/c\  hemaia_clk_rst_controller:  { path: hw/hemaia/hemaia_clk_rst_controller }' "$bender_local_file"
    fi
else
    echo "SIM_WITH_PLL=0: leaving Bender.local entry for hemaia_clk_rst_controller untouched"
fi

