#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# stages.sh <stage> [bound us]: one FA prefill workload on the standing FGP simulator, with its watchdog.
#   fresh_<v>    workloads/fa/dsv2_l1_prefill/<v> (DSv2-Lite's prefill over a fresh prompt, no cache; bound 6000 us,
#                16 h)
#   llama3_<v>   workloads/fa/llama3_prefill/<v>        (Llama-3-8B's GQA attention over a fresh 256-token prompt:
#                full, or a stage alone, stage1234 / stage5 / stage678; bound 9000 us, 16 h)
#   llama3abl_<v> workloads/fa/llama3_prefill_ablation/<v> (the Llama prefill's ablation rungs at 1,024 tokens, base,
#                bcast and all (all = llama3_prefill/t1024's switches); bound 16000 us, 120 h)
#   llama3dec_<v> workloads/fa/llama3_decode_full/<v> (the Llama decode's ablation rungs on the whole attention block,
#                base, bcast, bcast_reduce, all;
#                one token over an 8,192-position cache; the decode watchdog profile with its quiet cap off (a check
#                that reads HBM directly is quiet for about a millisecond); bound 8000 us, 24 h)
# Each variant's params.hjson says what it is (its `stop` says how far the program runs, for bisecting a
# failure: x, q8, sweep, post, full); every run and its finding are in results/prefill_stages.md.
#
# What it does: refuses to start while another simulator of this user runs (PARALLEL=1 shares the standing
# simv with up to SIM_SLOTS running simulations, default 8); launches the run through common/launch.sh (the
# runner's event log); arms common/sim_watchdog.py with the fa_prefill profile (it reads the scheduler
# monitor's bingo_progress.log and stops the simulator on a hang, a stalled AXI link, an assertion, low quota,
# too few FGP threads or the bound) and common/trim_live.py (frees the idle head of the traces). The simulator
# is also bounded in simulated time (+vcs+finish). Tag: stg_<stage>[_<n>].
# WD_SET="key=value ..." passes extra parameters to the watchdog (a full run's host init outlasts the profile's
# stall floor, so full runs default to stall_floor_us=1500); FGP_THREADS (default 0: one core, the fastest per core) sets the
# simulator's worker threads and the watchdog's thread floor; EXTRA_PLUSARGS go to the simulator.
S=$(cd "$(dirname "$0")" && pwd); H=${S%/target/sim/automation/sweep/*}; EXP=${S#"$H"/target/sim/automation/sweep/}
WL=$H/target/sw/host/apps/offload_bingo_hw/multi_chip/workloads
stage=$1
[ -z "$stage" ] && { sed -n 6,15p "$0"; exit 1; }
case $stage in
  fresh_*) wl=fa/dsv2_l1_prefill/${stage#fresh_}; bound=${2:-6000}; TIMEOUT_H=${TIMEOUT_H:-16}; wd_def=stall_floor_us=1500;;
  llama3_*) wl=fa/llama3_prefill/${stage#llama3_}; bound=${2:-9000}; TIMEOUT_H=${TIMEOUT_H:-16}; wd_def=stall_floor_us=1500;;
  llama3abl_*) wl=fa/llama3_prefill_ablation/${stage#llama3abl_}; bound=${2:-16000}; TIMEOUT_H=${TIMEOUT_H:-120}; wd_def=stall_floor_us=1500;;
  llama3dec_*) wl=fa/llama3_decode_full/${stage#llama3dec_}; bound=${2:-8000}; TIMEOUT_H=${TIMEOUT_H:-24}; prof=decode; wd_def=stall_cap_us=0;;
  *)      echo "unknown stage $stage"; exit 1;;
esac
[ -f "$WL/$wl/params.hjson" ] || { echo "stage $stage: no workload $wl"; exit 1; }
FGP_THREADS=${FGP_THREADS:-0}
busy=$(ps -u "$USER" -o pid=,comm= | awk '$2=="occamy_chip.vcs" {print $1}' | tr '\n' ' ')
if [ "${PARALLEL:-0}" != 1 ]; then
  [ -n "$busy" ] && { echo "a simulator is running (pid $busy): one at a time (PARALLEL=1 to share the simv)"; exit 1; }
  export SIM_SLOTS=1
else
  export SIM_SLOTS=${SIM_SLOTS:-8}
fi
n=0; tag=stg_$stage; while [ -e "$S/runs/$tag.events" ]; do n=$((n + 1)); tag=stg_${stage}_$n; done
mkdir -p "$S/runs"
cat > "$S/runs/$tag.yaml" <<EOT
cfg: target/rtl/cfg/hemaia_16c4m_8MBL3_2cluster.hjson
memchip: true
fgp_threads: $FGP_THREADS
timeout_h: ${TIMEOUT_H:-6}
points:
  - {name: fa_$stage, host_app_type: offload_bingo_hw, chip_type: multi_chip,
     workload: $wl, dev_app: snax-bingo-offload}
EOT
export SIM_PLUSARGS="-l sim_live.log +vcs+finish+$((bound * 1000000)) $EXTRA_PLUSARGS"
"$H/target/sim/automation/sweep/common/launch.sh" run "$EXP" "$tag" "$S/runs/$tag.yaml" --jobs 1 | cut -c1-120
"$H/target/sim/automation/sweep/common/watchdog.sh" "$S/runs/$tag" ${prof:-fa_prefill} bound_us=$bound min_threads=$((FGP_THREADS + 1)) "dfg=$WL/$wl/final_dfg.csv" $wd_def $WD_SET | tail -1
setsid nohup python3 -u "$H/target/sim/automation/sweep/common/trim_live.py" "$S/runs/$tag" > "$S/runs/$tag.trim_live.log" 2>&1 < /dev/null &
echo "$tag: bound $bound us; events $S/runs/$tag.events"
