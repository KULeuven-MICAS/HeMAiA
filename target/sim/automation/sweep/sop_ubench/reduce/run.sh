#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# run.sh: build (VCS) and run tb_reduce, the standalone endpoint chain-reduce testbench.
#
#   ./run.sh build                 compile tb/ and only the RTL it needs -> work/simv
#   ./run.sh smoke                 chain, P = 2, 4 KiB                     -> logs/smoke.log
#   ./run.sh validate              chain, P = 2,4,8,16, 128 KiB            -> logs/validate.log
#   ./run.sh run <tag> <plusargs>  any point, e.g. ./run.sh run p4_32k +BYTES=32768 +PLIST=4
#   ./run.sh table [log...]        the [REDUCE] lines of logs as a table
#   ./run.sh sweep                 every arm x payload, all P (M7), JOBS at a time -> results/logs/
#   ./run.sh parse                 results/logs -> results/reduce.csv
#
# Environment
#   VCS          the vcs binary (default: the W-2024.09-SP2 install)
#   TCDM_BANKED  1 (default): 32-bank TCDM arbitration; 0: every TCDM access granted at once
#   LINK_TXWBUF  FramerTxWBufBeats of every link (default 64, the RTL's; 0 = diagnostic only)
#   WORK, LOGS   build and log directories (default work/, logs/), for side-by-side builds
#   PHY_BEHAVIORAL=1  simulate the PHY's RX clock delay line behaviourally
#                     (+define+HEMAIA_D2D_PHY_BEHAVIORAL, as target/sim's SIM_D2D_PHY_BEHAVIORAL);
#                     default 0, the netlist delay line the full-package simv uses
#
# Nothing outside this directory is written. The RTL is read from the HeMAiA checkout this
# directory lives in: the generated xDMA (deps/snitch_cluster/target/snitch_cluster/generated/
# snax_split_cluster_xdma) must be current for hemaia_16c4m_8MBL3_2cluster; this script does not
# regenerate it.
set -euo pipefail

D=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$D/../../../../../.." && pwd)
W=${WORK:-$D/work}
L=${LOGS:-$D/logs}
VCS=${VCS:-/esat/micas-data/software/Synopsys/vcs/W-2024.09-SP2/bin/vcs}
BENDER=${BENDER:-bender}
TCDM_BANKED=${TCDM_BANKED:-1}
PHY_BEHAVIORAL=${PHY_BEHAVIORAL:-0}
LINK_TXWBUF=${LINK_TXWBUF:-64}
export VCS_TARGET_ARCH=${VCS_TARGET_ARCH:-linux64}

SC=$ROOT/deps/snitch_cluster
GEN=$SC/target/snitch_cluster/generated/snax_split_cluster_xdma

flist() {
  mkdir -p "$W"
  # The RTL packages the testbench instantiates (the cluster iDMA included), with their dependencies, for the targets the
  # full-package VCS build uses (bender script vcs adds `vcs` and `simulation`).
  (cd "$ROOT" && "$BENDER" --local script flist-plus \
      -t rtl -t vcs -t simulation -t hemaia -t occamy -t hemaia_xdma \
      -t snax_split_cluster_xdma -t snitch_cluster -t snax_split_cluster \
      -p common_cells -p axi -p register_interface -p tech_cells_generic \
      -p hemaia_d2d_link -p axi_riscv_atomics -p xdma_axi_adapter -p idma) > "$W/rtl.f"
  {
    cat "$W/rtl.f"
    # From snitch_cluster, only what the xDMA wrapper needs: the TCDM/reqrsp typedefs and
    # reqrsp_pkg (AMO enum), and the generated xDMA of snax_split_cluster.
    echo "+incdir+$SC/hw/tcdm_interface/include"
    echo "+incdir+$SC/hw/reqrsp_interface/include"
    echo "$SC/hw/reqrsp_interface/src/reqrsp_pkg.sv"
    echo "$GEN/snax_split_cluster_blocks.sv"
    echo "$GEN/snax_split_cluster_xdma_wrapper.sv"
    echo "$D/tb/reduce_pkg.sv"
    echo "$D/tb/reduce_compute_node.sv"
    echo "$D/tb/reduce_simd_model.sv"
    echo "$D/tb/tb_reduce.sv"
  } > "$W/files.f"
}

build() {
  flist
  local defs=""
  [ "$PHY_BEHAVIORAL" = 1 ] && defs="+define+HEMAIA_D2D_PHY_BEHAVIORAL"
  echo "TCDM_BANKED=$TCDM_BANKED PHY_BEHAVIORAL=$PHY_BEHAVIORAL LINK_TXWBUF=$LINK_TXWBUF" > "$W/build.cfg"
  # The analysis and elaboration flags of target/sim's VCS flow (sim.mk / Makefile).
  (cd "$W" && "$VCS" -full64 -sverilog \
      -timescale=1ns/1ps -override_timescale=1ns/1ps \
      -assert svaext -assert disable_cover \
      +lint=PCWM-L,TFIPC-L -error=PCWM-L,TFIPC-L \
      -O4 -j16 +nospecify -ignore initializer_driver_checks \
      $defs -pvalue+tb_reduce.TcdmBanked=$TCDM_BANKED \
      -pvalue+tb_reduce.LinkTxWBufBeats=$LINK_TXWBUF \
      -f "$W/files.f" -top tb_reduce -o "$W/simv" -Mdir="$W/csrc" \
      -l "$W/compile.log") > /dev/null
  if grep -qE "^Error-|^Error:" "$W/compile.log"; then
    echo "build FAILED: see $W/compile.log"; exit 1
  fi
  [ -x "$W/simv" ] || { echo "build FAILED: no simv (see $W/compile.log)"; exit 1; }
  echo "built $W/simv ($(cat "$W/build.cfg"))"
}

run() {
  local tag=$1; shift
  mkdir -p "$L"
  [ -x "$W/simv" ] || build
  (cd "$W" && ./simv +vcs+lic+wait +ARM=chain "$@" -l "$L/$tag.log") > /dev/null
  table "$L/$tag.log"
  grep -q "RESULT: PASS" "$L/$tag.log"
}

table() {
  printf "%-6s %-4s %-8s %8s %8s %8s %8s %8s %11s %9s %5s %s\n" \
    arm P bytes lat_cc fin_cc first_cc last_cc task_cc sram_total sram_max busy check
  grep -h "^\[REDUCE\]" "$@" | sed 's/^\[REDUCE\] //' | awk '{
      for (i = 1; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] }
      printf "%-6s %-4s %-8s %8s %8s %8s %8s %8s %11s %9s %5s %s\n", v["arm"], v["P"], v["bytes"],
             v["lat_cc"], v["fin_cc"], v["first_cc"], v["last_cc"], v["task_cc"],
             v["sram_total"], v["sram_max"], v["busiest"], v["check"] }'
  grep -h "RESULT" "$@" || true
}

# The sweep: one simulation per arm x payload (all P in it), JOBS at a time, into results/logs/.
# VCS runtime licences are scarce on this site: every simulation queues for one
# (+vcs+lic+wait) instead of failing. Points whose log already holds RESULT: PASS are skipped
# (rerun with FORCE=1).
SWEEP_ARMS=${SWEEP_ARMS:-"chain two binomial pull swchain swtree"}
SWEEP_BYTES=${SWEEP_BYTES:-"4096 8192 16384 32768 65536 131072 262144"}
SWEEP_P=${SWEEP_P:-"2,4,8,16"}
JOBS=${JOBS:-4}

sweep() {
  [ -x "$W/simv" ] || build
  local R=$D/results/logs
  mkdir -p "$R"
  local list=$W/sweep.jobs
  : > "$list"
  for a in $SWEEP_ARMS; do for b in $SWEEP_BYTES; do
    local log=$R/${a}_${b}.log
    if [ "${FORCE:-0}" != 1 ] && grep -q "RESULT: PASS" "$log" 2>/dev/null; then continue; fi
    echo "$a $b $SWEEP_P $log" >> "$list"
  done; done
  echo "sweep: $(wc -l < "$list") simulations, $JOBS at a time -> $R"
  # the longest first: software arms, then large payloads
  awk '{w = ($1 ~ /^(pull|swchain|swtree)$/) ? 1 : 0; print w, $0}' "$list" \
    | sort -k1,1nr -k3,3nr | cut -d' ' -f2- \
    | xargs -P "$JOBS" -L 1 bash -c \
    'cd "$0" && ./simv +vcs+lic+wait +ARM=$1 +BYTES=$2 +PLIST=$3 +VERBOSE=1 +TIMEOUT_US=200000 -l "$4" > /dev/null 2>&1; echo "done $1 $2: $(grep -h "^\[REDUCE\]" "$4" | grep -o " P=[0-9]*\|lat_prog_cc=[0-9]*\|check=[A-Z]*" | tr "\n" " ") $(grep -h "RESULT" "$4")"' "$W"
  python3 "$D/parse.py"
}

case "${1:-}" in
  build)    build ;;
  smoke)    run smoke +BYTES=4096 +PLIST=2 +VERBOSE=1 ;;
  validate) run validate +BYTES=131072 +PLIST=2,4,8,16 +VERBOSE=1 ;;
  run)      shift; run "$@" ;;
  table)    shift; table "${@:-$L/*.log}" ;;
  sweep)    sweep ;;
  parse)    python3 "$D/parse.py" ;;
  *) sed -n '6,30p' "$0"; exit 1 ;;
esac
