#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# stages.sh <stage> [bound us]: one MoE prefill workload on the standing FGP simulator, with its watchdog.
#   poc<v>          workloads/moe/dsv2_l1_prefill_tiny/v<v>  (proof of concept: every expert's first 384
#                   intermediate columns, the shared MLP's first 1024; bound 4000 us)
#   full<v>         workloads/moe/dsv2_l1_prefill/v<v>       (the real size; bound 4000 us, 12 h)
#   gtb_<variant>   workloads/moe/gather_tb/<variant>        (the reduce on its own; bound 2500 us)
# Each variant's params.hjson says what it is; the variants and their results are in README.md.
#
# What it does: refuses to start while another simulator of this user runs (PARALLEL=1 shares the standing
# simv with up to SIM_SLOTS running simulations, default 3); launches the run through common/launch.sh (the
# runner's event log); arms common/sim_watchdog.py with the moe_prefill profile (it reads the scheduler
# monitor's bingo_progress.log and stops the simulator on a hang, a stalled AXI link, an assertion, low quota,
# too few FGP threads or the bound) and ffn/trim_live.py (frees the idle head of the traces). The simulator
# is also bounded in simulated time (+vcs+finish). Tag: stg_<stage>[_<n>].
# WD_SET="key=value ..." passes extra parameters to the watchdog; FGP_THREADS (default 8) sets the
# simulator's worker threads and the watchdog's thread floor; EXTRA_PLUSARGS go to the simulator.
S=$(cd "$(dirname "$0")" && pwd); H=${S%/target/sim/automation/sweep/*}; EXP=${S#"$H"/target/sim/automation/sweep/}
WL=$H/target/sw/host/apps/offload_bingo_hw/multi_chip/workloads
stage=$1
[ -z "$stage" ] && { sed -n 6,11p "$0"; exit 1; }
case $stage in
  poc*)  wl=moe/dsv2_l1_prefill_tiny/v${stage#poc}; bound=${2:-4000};;
  full*) wl=moe/dsv2_l1_prefill/v${stage#full}; bound=${2:-4000}; TIMEOUT_H=${TIMEOUT_H:-12};;
  gtb_*) wl=moe/gather_tb/${stage#gtb_}; bound=${2:-2500};;
  *)     echo "unknown stage $stage"; exit 1;;
esac
[ -f "$WL/$wl/params.hjson" ] || { echo "stage $stage: no workload $wl"; exit 1; }
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
"$H/target/sim/automation/sweep/common/watchdog.sh" "$S/runs/$tag" moe_prefill bound_us=$bound min_threads=$((FGP_THREADS + 1)) "dfg=$WL/$wl/final_dfg.csv" $WD_SET | tail -1
setsid nohup python3 -u "$S/../ffn/trim_live.py" "$S/runs/$tag" > "$S/runs/$tag.trim_live.log" 2>&1 < /dev/null &
echo "$tag: bound $bound us; events $S/runs/$tag.events"
