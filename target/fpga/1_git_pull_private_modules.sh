#!/bin/bash

# Author: Yunhao Deng <yunhao.deng@kuleuven.be>
#         Fanchen Kong <fanchen.kong@kuleuven.be>
#
# FPGA-chip private-module setup.
# This does not clone tech_cells_tsmc16. Use the public tech_cells_generic dependency
# unless an existing local tech_cells_tsmc16 override provides FPGA mappings.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

FPGA_WITH_D2D="${FPGA_WITH_D2D:-1}"
FPGA_WITH_PLL="${FPGA_WITH_PLL:-1}"

while [ $# -gt 0 ]; do
    case "$1" in
        --d2d=*) FPGA_WITH_D2D="${1#*=}"; shift ;;
        --pll=*) FPGA_WITH_PLL="${1#*=}"; shift ;;
        --macro=*) echo "Ignoring $1: FPGA setup does not clone tech_cells_tsmc16"; shift ;;
        -h|--help)
            echo "Usage: $0 [--d2d=0|1] [--pll=0|1]"
            echo "Defaults: d2d=1, pll=1"
            echo "Keeps an existing tech_cells_tsmc16 override if it provides FPGA mappings."
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

for _var in FPGA_WITH_D2D FPGA_WITH_PLL; do
    val="${!_var}"
    if ! [[ "$val" =~ ^[01]$ ]]; then
        echo "Invalid value for ${_var}: '${val}' (must be 0 or 1)" >&2
        exit 1
    fi
done

clone_or_pull() {
    local repo_url="$1"
    local checkout_dir="$2"

    if [ ! -d "$checkout_dir" ]; then
        git clone "$repo_url" "$checkout_dir"
    else
        git -C "$checkout_dir" pull
    fi
}

bender_local_file="$repo_root/Bender.local"

if [ ! -f "$bender_local_file" ]; then
    cat > "$bender_local_file" <<'EOF'
# Local dependency overrides.

overrides:
EOF
fi

if ! grep -q '^[[:space:]]*overrides:' "$bender_local_file"; then
    printf '\noverrides:\n' >> "$bender_local_file"
fi

upsert_override() {
    local key="$1"
    local line="$2"

    if grep -q "^[[:space:]]*$key:" "$bender_local_file"; then
        sed -i "/^[[:space:]]*$key:/c\\$line" "$bender_local_file"
    else
        echo "$line" >> "$bender_local_file"
    fi
}

tech_cells_generic_fpga='  tech_cells_generic: { git: https://github.com/KULeuven-MICAS/tech_cells_generic.git, rev    : master }'

# Keep the local checkout when it supports FPGA SRAMs. Otherwise restore the public
# dependency if the tapeout setup selected an ASIC-only checkout.
if grep -q '^[[:space:]]*tech_cells_generic:.*tech_cells_tsmc16' "$bender_local_file"; then
    if [ -f "$repo_root/hw/hemaia/tech_cells_tsmc16/src/fpga/tc_sram_xilinx.sv" ] &&
       grep -q 'src/fpga/tc_sram_xilinx.sv' "$repo_root/hw/hemaia/tech_cells_tsmc16/Bender.yml"; then
        echo "Keeping Bender.local tech_cells_tsmc16 override with FPGA SRAM support"
    else
        upsert_override "tech_cells_generic" "$tech_cells_generic_fpga"
        echo "Restored Bender.local tech_cells_generic override to KULeuven-MICAS"
    fi
elif grep -q '^[[:space:]]*tech_cells_generic:' "$bender_local_file"; then
    echo "Keeping existing Bender.local tech_cells_generic override"
else
    echo "$tech_cells_generic_fpga" >> "$bender_local_file"
    echo "Added Bender.local tech_cells_generic override to KULeuven-MICAS"
fi

if [ "$FPGA_WITH_D2D" -eq 1 ]; then
    clone_or_pull git@github.com:IveanEx/hemaia_d2d_link.git "$repo_root/hw/hemaia/hemaia_d2d_link"
    upsert_override "hemaia_d2d_link" "  hemaia_d2d_link:    { path: hw/hemaia/hemaia_d2d_link }"
else
    echo "FPGA_WITH_D2D=0: leaving Bender.local entry for hemaia_d2d_link untouched"
fi

if [ "$FPGA_WITH_PLL" -eq 1 ]; then
    clone_or_pull git@github.com:IveanEx/hemaia_clk_rst_controller.git "$repo_root/hw/hemaia/hemaia_clk_rst_controller"
    upsert_override "hemaia_clk_rst_controller" "  hemaia_clk_rst_controller:  { path: hw/hemaia/hemaia_clk_rst_controller }"
else
    echo "FPGA_WITH_PLL=0: leaving Bender.local entry for hemaia_clk_rst_controller untouched"
fi
