#!/bin/bash

# Author: Yunhao Deng <yunhao.deng@kuleuven.be>
#         Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Simulation private-module setup for the D2D link only.
# This intentionally does not clone or override tech_cells_tsmc16 or the
# clk/rst controller; use the tapeout/fpga setup scripts for those flows.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

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

clone_or_pull git@github.com:IveanEx/hemaia_d2d_link.git \
    "$repo_root/hw/hemaia/hemaia_d2d_link"
upsert_override "hemaia_d2d_link" \
    "  hemaia_d2d_link:    { path: hw/hemaia/hemaia_d2d_link }"
