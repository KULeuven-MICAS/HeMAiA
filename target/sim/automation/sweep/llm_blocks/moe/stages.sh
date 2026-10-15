#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# stages.sh <stage> [bound us]: one MoE prefill variant on the standing FGP simulator, ONE simulation at a
# time, with its watchdog and hang guard.
#   stage   poc = workloads/moe/dsv2_l1_prefill_tiny/all, every expert's first 384 intermediate columns;
#           poc_route / poc_a8 = the same, stopped after the route / after every A8 (bring-up)
#           poc3 = dsv2_l1_prefill_tiny/v3 (all + the route and GEMM set-up levers); poc4 = v4 (v3 with the
#           root's X copies local and the route dispatched by row block); full = dsv2_l1_prefill/all;
#           full5 = dsv2_l1_prefill/v5 (all with the X puts first on the root's cluster 1); poc5 = its tiny twin
#           full6 / poc6 = dsv2_l1_prefill{,_tiny}/v6: v5 at 128 tokens (4 row blocks of X)
# What it does: refuses to start while another simulator of this user runs; launches the run through
# common/launch.sh (the runner's event log); arms common/sim_watchdog.py with the moe_prefill profile (it
# reads the scheduler monitor's bingo_progress.log and stops the simulator on a hang, a stalled AXI link, an
# assertion, low quota, too few FGP threads or the bound) and ffn/trim_live.py (frees the idle head of the
# traces). The simulator is also bounded in simulated time (+vcs+finish). Tag: stg_<stage>[_<n>].
# WD_SET="key=value ..." passes extra parameters to the watchdog; FGP_THREADS (default 8) sets the
# simulator's worker threads and the watchdog's thread floor; PARALLEL=1 lets it share the standing simv with
# running simulations (SIM_SLOTS of them, default 3). EXTRA_PLUSARGS go to the simulator (e.g. +push_rate_ns=1000).
S=$(cd "$(dirname "$0")" && pwd); H=${S%/target/sim/automation/sweep/*}; EXP=${S#"$H"/target/sim/automation/sweep/}
stage=$1
[ -z "$stage" ] && { sed -n 6,12p "$0"; exit 1; }
case $stage in
  poc)       wl=moe/dsv2_l1_prefill_tiny/all; bound=${2:-3000};;
  poc_route) wl=moe/dsv2_l1_prefill_tiny/stop_route; bound=${2:-2000};;
  poc_a8)    wl=moe/dsv2_l1_prefill_tiny/stop_a8; bound=${2:-2500};;
  poc3)      wl=moe/dsv2_l1_prefill_tiny/v3; bound=${2:-3000};;
  poc4)      wl=moe/dsv2_l1_prefill_tiny/v4; bound=${2:-3000};;
  poc5)      wl=moe/dsv2_l1_prefill_tiny/v5; bound=${2:-3000};;
  full)      wl=moe/dsv2_l1_prefill/all; bound=${2:-4000}; TIMEOUT_H=${TIMEOUT_H:-12};;
  full5)     wl=moe/dsv2_l1_prefill/v5; bound=${2:-4000}; TIMEOUT_H=${TIMEOUT_H:-12};;
  poc6)      wl=moe/dsv2_l1_prefill_tiny/v6; bound=${2:-3000};;
  full6)     wl=moe/dsv2_l1_prefill/v6; bound=${2:-4000}; TIMEOUT_H=${TIMEOUT_H:-12};;
  poc7a)     wl=moe/dsv2_l1_prefill_tiny/v7a; bound=${2:-3000};;
  poc7b)     wl=moe/dsv2_l1_prefill_tiny/v7b; bound=${2:-3000};;
  full7)     wl=moe/dsv2_l1_prefill/v7; bound=${2:-4000}; TIMEOUT_H=${TIMEOUT_H:-12};;
  full8)     wl=moe/dsv2_l1_prefill/v8; bound=${2:-4000}; TIMEOUT_H=${TIMEOUT_H:-12};;
  full9)     wl=moe/dsv2_l1_prefill/v9; bound=${2:-4000}; TIMEOUT_H=${TIMEOUT_H:-12};;
  full10)    wl=moe/dsv2_l1_prefill/v10; bound=${2:-4000}; TIMEOUT_H=${TIMEOUT_H:-12};;
  poc11)     wl=moe/dsv2_l1_prefill_tiny/v11; bound=${2:-3000};;
  full12)    wl=moe/dsv2_l1_prefill/v12; bound=${2:-4000}; TIMEOUT_H=${TIMEOUT_H:-12};;
  full13)    wl=moe/dsv2_l1_prefill/v13; bound=${2:-4000}; TIMEOUT_H=${TIMEOUT_H:-12};;
  full14)    wl=moe/dsv2_l1_prefill/v14; bound=${2:-4000}; TIMEOUT_H=${TIMEOUT_H:-12};;
  full15)    wl=moe/dsv2_l1_prefill/v15; bound=${2:-4000}; TIMEOUT_H=${TIMEOUT_H:-12};;
  full16)    wl=moe/dsv2_l1_prefill/v16; bound=${2:-4000}; TIMEOUT_H=${TIMEOUT_H:-12};;
  full17)    wl=moe/dsv2_l1_prefill/v17; bound=${2:-4000}; TIMEOUT_H=${TIMEOUT_H:-12};;
  *) echo "unknown stage $stage"; exit 1;;
esac
FGP_THREADS=${FGP_THREADS:-8}
busy=$(ps -u "$USER" -o pid=,comm= | awk '$2=="occamy_chip.vcs" {print $1}' | tr '\n' ' ')
if [ "${PARALLEL:-0}" != 1 ]; then
  [ -n "$busy" ] && { echo "a simulator is running (pid $busy): one at a time (PARALLEL=1 to share the simv)"; exit 1; }
  export SIM_SLOTS=1
else
  export SIM_SLOTS=${SIM_SLOTS:-3}
fi
n=0; tag=stg_$stage; while [ -e "$S/runs/$tag.events" ]; do n=$((n + 1)); tag=stg_${stage}_$n; done
cat > "$S/runs/$tag.yaml" <<EOT
cfg: target/rtl/cfg/hemaia_16c4m_8MBL3_2cluster.hjson
memchip: true
fgp_threads: $FGP_THREADS
timeout_h: ${TIMEOUT_H:-6}
points:
  - {name: moe_$stage, host_app_type: offload_bingo_hw, chip_type: multi_chip,
     workload: $wl, dev_app: snax-bingo-offload}
EOT
export SIM_PLUSARGS="-l sim_live.log +vcs+finish+$((bound * 1000000)) $EXTRA_PLUSARGS"
"$H/target/sim/automation/sweep/common/launch.sh" run "$EXP" "$tag" "$S/runs/$tag.yaml" --jobs 1 | cut -c1-120
"$H/target/sim/automation/sweep/common/watchdog.sh" "$S/runs/$tag" moe_prefill bound_us=$bound min_threads=$((FGP_THREADS + 1)) "dfg=$H/target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/$wl/final_dfg.csv" $WD_SET | tail -1
setsid nohup python3 -u "$S/../ffn/trim_live.py" "$S/runs/$tag" > "$S/runs/$tag.trim_live.log" 2>&1 < /dev/null &
echo "$tag: bound $bound us; events $S/runs/$tag.events"
