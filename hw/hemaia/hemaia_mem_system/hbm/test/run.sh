#!/usr/bin/env bash
# Copyright 2026 KU Leuven.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Unit test of hemaia_hbm_model with VCS. Needs the EDA env and a bender checkout
# (for common_cells / common_verification):
#     source src_hemaia_eda.sh
#     hw/hemaia/hemaia_mem_system/hbm/test/run.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
BUILD="${BUILD:-$HERE/build}"
AXI="$ROOT/hw/vendor/pulp_platform_axi"
CC_DIR="$(ls -d "$ROOT"/.bender/git/checkouts/common_cells-* | head -1)"
CV_DIR="$(ls -d "$ROOT"/.bender/git/checkouts/common_verification-* | head -1)"

rm -rf "$BUILD"
mkdir -p "$BUILD"
cd "$BUILD"
python3 "$HERE/gen_data.py" "$BUILD"
md5_before="$(md5sum hbm/a.bin | cut -d' ' -f1)"

vcs -full64 -sverilog -timescale=1ns/1ps -assert disable_cover +lint=PCWM-L \
  -cpp "${CXX:-g++}" -CFLAGS "-std=c++17 -O2" \
  +incdir+"$AXI/include" +incdir+"$CC_DIR/include" \
  "$CV_DIR/src/clk_rst_gen.sv" "$CV_DIR/src/rand_id_queue.sv" \
  "$AXI/src/axi_pkg.sv" "$AXI/src/axi_intf.sv" "$AXI/src/axi_test.sv" \
  "$HERE/../hemaia_hbm_pkg.sv" "$HERE/../hemaia_hbm_model.sv" \
  "$HERE/tb_hemaia_hbm_model.sv" "$HERE/../hemaia_hbm_dpi.cc" \
  -top tb_hemaia_hbm_model -o simv > compile.log 2>&1 || { tail -40 compile.log; exit 1; }

./simv > sim.log 2>&1 || true

fail=0
grep -q "HBM TEST PASSED" sim.log || { echo "FAIL: testbench did not pass"; fail=1; }
if grep -E "^(Error|Fatal)" sim.log; then fail=1; fi
[ "$(md5sum hbm/a.bin | cut -d' ' -f1)" = "$md5_before" ] \
  || { echo "FAIL: a write reached a.bin on disk (copy-on-write broken)"; fail=1; }
# dump of the first 256 B: a.bin with word 9 overwritten by the COW test.
python3 - <<'PY' || fail=1
import struct, sys
a = bytearray(open("hbm/a.bin", "rb").read(256))
a[72:80] = struct.pack("<Q", 0xC0FFEE00DEADBEEF)
if open("dump_a.bin", "rb").read() != bytes(a):
    sys.exit("FAIL: dump_a.bin differs from the expected HBM content")
PY

grep -E "^== |^   |\[HBM tb_hemaia_hbm_model.i_dut\]" sim.log || true
if [ $fail -eq 0 ]; then echo "HBM UNIT TEST PASSED"; else echo "HBM UNIT TEST FAILED (see $BUILD/sim.log)"; exit 1; fi
