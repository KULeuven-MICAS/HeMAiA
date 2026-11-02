#!/bin/bash
# Measure the BINGO manager's dependency-edge latency (tb_sched_latency) with VCS.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
B=${BINGO:-$HOME/no_backup/bingo_hw_manager}
VCS=${VCS:-/esat/micas-data/software/Synopsys/vcs/W-2024.09-SP2/bin/vcs}
export VCS_TARGET_ARCH=${VCS_TARGET_ARCH:-linux64}
W=$D/work; mkdir -p "$W"
(cd "$B" && bender --local script flist-plus -t rtl -t test -t simulation) | awk '!/\/test\/[a-z_0-9]*\.sv$/ || /axi_test\.sv$/' > "$W/files.f"
{ echo "+incdir+$D"; echo "$D/tb_sched_latency.sv"; } >> "$W/files.f"
(cd "$W" && "$VCS" -full64 -sverilog -override_timescale=1ns/1ps -assert svaext \
    -ignore initializer_driver_checks -f "$W/files.f" -top tb_sched_latency -o "$W/simv" \
    -Mdir="$W/csrc" -l "$W/compile.log") > /dev/null || { echo "build failed: $W/compile.log"; exit 1; }
(cd "$W" && ./simv -l "$D/sched_latency.log") > /dev/null
grep "^\[SCHED\]\|PASSED\|FAILED" "$D/sched_latency.log"
