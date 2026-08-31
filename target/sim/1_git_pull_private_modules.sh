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
HEMAIA_D2D_LINK_BRANCH="${HEMAIA_D2D_LINK_BRANCH:-full_duplex}"

clone_or_update_branch() {
    local repo_url="$1"
    local checkout_dir="$2"
    local branch="$3"

    if [ ! -d "$checkout_dir/.git" ]; then
        if [ -e "$checkout_dir" ]; then
            echo "Cannot clone $repo_url: $checkout_dir exists but is not a Git checkout" >&2
            return 1
        fi
        git clone --branch "$branch" --single-branch "$repo_url" "$checkout_dir"
    else
        git -C "$checkout_dir" fetch origin \
            "refs/heads/$branch:refs/remotes/origin/$branch"
        if git -C "$checkout_dir" show-ref --verify --quiet "refs/heads/$branch"; then
            git -C "$checkout_dir" switch "$branch"
        else
            git -C "$checkout_dir" switch --track -c "$branch" "origin/$branch"
        fi
        git -C "$checkout_dir" pull --ff-only origin "$branch"
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

d2d_repo="$repo_root/hw/hemaia/hemaia_d2d_link"
clone_or_update_branch git@github.com:IveanEx/hemaia_d2d_link.git \
    "$d2d_repo" "$HEMAIA_D2D_LINK_BRANCH"
echo "hemaia_d2d_link: branch $(git -C "$d2d_repo" branch --show-current), commit $(git -C "$d2d_repo" rev-parse --short HEAD)"
upsert_override "hemaia_d2d_link" \
    "  hemaia_d2d_link:    { path: hw/hemaia/hemaia_d2d_link }"
