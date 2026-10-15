#!/usr/bin/env bash
# Copyright 2026 KU Leuven.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Standalone push testbench (tb_hemaia_push.sv) with VCS: compiles once, then runs every
# configuration given, or the default sweep, and prints one PUSH_TB RESULT line each.
#     source src_hemaia_eda.sh
#     hw/hemaia/hemaia_mem_system/push_tb/run.sh                       # default sweep
#     hw/hemaia/hemaia_mem_system/push_tb/run.sh "+mode=1 +engines=2"  # one configuration
# BUILD=<dir> puts the build elsewhere; NO_COMPILE=1 reuses it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
BUILD="${BUILD:-$HERE/build}"
mkdir -p "$BUILD"

if [ "${NO_COMPILE:-0}" != 1 ]; then
  (cd "$ROOT" && bender script flist-plus -p idma -p hemaia_d2d_link -t rtl -t simulation -t simulations) \
    > "$BUILD/deps.f"
  (cd "$BUILD" && vcs -full64 -sverilog -timescale=1ns/1ps -assert disable_cover \
    +define+HEMAIA_D2D_PHY_CORE_CLOCK \
    -f deps.f \
    "$ROOT/hw/hemaia/hemaia_mem_system/hemaia_sys_idma_queue.sv" \
    "$ROOT/hw/hemaia/hemaia_mem_system/hemaia_sys_idma_engine.sv" \
    "$HERE/tb_hemaia_push.sv" \
    -top tb_hemaia_push -o simv > compile.log 2>&1) || { tail -40 "$BUILD/compile.log"; exit 1; }
fi

if [ $# -gt 0 ]; then
  configs=("$@")
else
  configs=()
  for chunks in 1 4 8; do
    for eng in 1 2; do
      for mode in 0 1; do
        configs+=("+mode=$mode +engines=$eng +run_chunks=$chunks +tag=m${mode}e${eng}c${chunks}")
      done
    done
  done
fi

fail=0
for cfg in "${configs[@]}"; do
  name="$(echo "$cfg" | tr -c 'A-Za-z0-9=_' '_' | tr -s '_')"
  # shellcheck disable=SC2086
  (cd "$BUILD" && ./simv $cfg > "sim_${name}.log" 2>&1) || true
  if grep -q "PUSH_TB PASS" "$BUILD/sim_${name}.log"; then
    grep -h "PUSH_TB RESULT\|PUSH_TB host\|round trip" "$BUILD/sim_${name}.log"
  else
    echo "FAIL: $cfg (see $BUILD/sim_${name}.log)"
    grep -h -m5 "Error\|PUSH_TB" "$BUILD/sim_${name}.log" || true
    fail=1
  fi
done
exit $fail
