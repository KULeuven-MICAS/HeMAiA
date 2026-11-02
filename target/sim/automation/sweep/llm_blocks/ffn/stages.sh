#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# stages.sh <stage> [bound us]: one FFN prefill variant on the standing FGP simulator, ONE simulation at a
# time, with its watchdog and hang guard.
#   stage   dsv2_<app> = workloads/ffn/dsv2_l0_prefill/<app> (DeepSeek-V2-Lite's FFN), llama3_<app> =
#           workloads/ffn/llama3_prefill/<app> (Llama-3-8B's FFN at the model's size); app = stage0, stage123, stage45
#           (one stage alone, ffn_stages_mc.py) or full (the 256-token block). TIMEOUT_H: the runner's wall-clock limit
#           abl_<rung> = workloads/ffn/dsv2_l0_prefill_ablation/<rung>, the ablation ladder (base, bcast, bcast_reduce,
#           fused, all); with PARALLEL=1 the five run side by side
# What it does: refuses to start while another simulator of this user runs; launches the run through
# common/launch.sh (the runner's event log); arms common/sim_watchdog.py with the ffn_prefill profile (it
# reads the scheduler monitor's bingo_progress.log and stops the simulator on a hang, a stalled AXI link, an
# assertion, low quota, too few FGP threads or the bound) and trim_live.py (frees the idle head of the
# traces). The simulator is also bounded in simulated time (+vcs+finish). Tag: stg_<stage>[_<n>].
# WD_SET="key=value ..." passes extra parameters to the watchdog (sim_watchdog.py params); FGP_THREADS (default 0: one core, the fastest per core) sets
# the simulator's worker threads and the watchdog's thread floor; PARALLEL=1 lets it share the standing simv with
# running simulations (SIM_SLOTS of them, default 8).
S=$(cd "$(dirname "$0")" && pwd); H=${S%/target/sim/automation/sweep/*}; EXP=${S#"$H"/target/sim/automation/sweep/}
stage=$1
[ -z "$stage" ] && { sed -n 6,10p "$0"; exit 1; }
case $stage in
  abl_base|abl_bcast) wl=ffn/dsv2_l0_prefill_ablation/${stage#abl_}; bound=${2:-6500}; TIMEOUT_H=${TIMEOUT_H:-48};;
  abl_*)    wl=ffn/dsv2_l0_prefill_ablation/${stage#abl_}; bound=${2:-4000}; TIMEOUT_H=${TIMEOUT_H:-30};;
  dsv2_full) wl=ffn/dsv2_l0_prefill/full; bound=${2:-3300}; TIMEOUT_H=${TIMEOUT_H:-12};;
  dsv2_*)   wl=ffn/dsv2_l0_prefill/${stage#dsv2_}; bound=${2:-2000}; TIMEOUT_H=${TIMEOUT_H:-8};;
  llama3_full) wl=ffn/llama3_prefill/full; bound=${2:-6000}; TIMEOUT_H=${TIMEOUT_H:-24};;
  llama3_*) wl=ffn/llama3_prefill/${stage#llama3_}; bound=${2:-2000}; TIMEOUT_H=${TIMEOUT_H:-8};;
  *) echo "unknown stage $stage"; exit 1;;
esac
# one simulation at a time unless PARALLEL=1 (then at most SIM_SLOTS, default 8, share the standing simv)
FGP_THREADS=${FGP_THREADS:-0}
busy=$(ps -u "$USER" -o pid=,comm= | awk '$2=="occamy_chip.vcs" {print $1}' | tr '\n' ' ')
if [ "${PARALLEL:-0}" != 1 ]; then
  [ -n "$busy" ] && { echo "a simulator is running (pid $busy): one at a time (PARALLEL=1 to share the simv)"; exit 1; }
  export SIM_SLOTS=1
else
  export SIM_SLOTS=${SIM_SLOTS:-8}
fi
n=0; tag=stg_$stage; while [ -e "$S/runs/$tag.events" ]; do n=$((n + 1)); tag=stg_${stage}_$n; done
cat > "$S/runs/$tag.yaml" <<EOT
cfg: target/rtl/cfg/hemaia_16c4m_8MBL3_2cluster.hjson
memchip: true
fgp_threads: $FGP_THREADS
timeout_h: $TIMEOUT_H
points:
  - {name: ffn_$stage, host_app_type: offload_bingo_hw, chip_type: multi_chip,
     workload: $wl, dev_app: snax-bingo-offload}
EOT
# transcript to a file (an assertion survives a dead runner), bounded in simulated time (ps)
export SIM_PLUSARGS="-l sim_live.log +vcs+finish+$((bound * 1000000))"
"$H/target/sim/automation/sweep/common/launch.sh" run "$EXP" "$tag" "$S/runs/$tag.yaml" --jobs 1 | cut -c1-120
"$H/target/sim/automation/sweep/common/watchdog.sh" "$S/runs/$tag" ffn_prefill bound_us=$bound min_threads=$((FGP_THREADS + 1)) "dfg=$H/target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/$wl/final_dfg.csv" $WD_SET | tail -1
setsid nohup python3 -u "$S/trim_live.py" "$S/runs/$tag" > "$S/runs/$tag.trim_live.log" 2>&1 < /dev/null &
echo "$tag: bound $bound us; events $S/runs/$tag.events"
