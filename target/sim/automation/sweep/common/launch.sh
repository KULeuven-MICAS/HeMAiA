#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# launch.sh build <cfg> [tree]                         build a cfg's simv, detached
# launch.sh run <exp> <tag> <points.yaml> [hsweep args] run points, detached
# Each launch is detached (setsid nohup) and gets its own event log (watch.sh); progress lands
# in <run dir>.events and in common/all.events. With SIM_WATCHDOG=<profile> a run also gets the hang guard
# (sim_watchdog.py, parameters in sim_watchdog.profiles.json): it reads the scheduler monitor's log and stops
# the simulator it watches when the workload stops moving.
C=$(cd "$(dirname "$0")" && pwd)
SW=$(dirname "$C")
T=$(git -C "$C" rev-parse --show-toplevel)     # the checkout this script sits in
source /users/micas/fkong/no_backup/src_hemaia_eda.sh >/dev/null 2>&1
[ "${VCS_HOME##*/}" = "W-2024.09-SP2" ] || { echo "BAD EDA ENV: $VCS_HOME"; exit 1; }
mode=$1; shift
# MEMORY CAPS (the machine is shared): each launch, the runner with the simulator it starts, runs in a systemd user
# scope of its own (RUN_MEM for a run, BUILD_MEM for a build; 0 launches without a scope) inside one slice for all of
# this user's simulation work (HEMAIA_SLICE; `systemctl --user set-property hemaia.slice MemoryMax=...` sets the
# total). The runner's build containers take HEMAIA_CONTAINER_MEM in the same slice (hemaia_sim_runner.py). A job
# over its cap is killed by the kernel on its own; the machine keeps its memory.
export HEMAIA_SLICE=${HEMAIA_SLICE:-hemaia.slice} HEMAIA_CONTAINER_MEM=${HEMAIA_CONTAINER_MEM:-96g}
mem=$([ "$mode" = build ] && echo "${BUILD_MEM:-96G}" || echo "${RUN_MEM:-48G}")
CAP=()
if [ "$mem" != 0 ] && systemd-run --user --scope -q --slice="$HEMAIA_SLICE" true 2>/dev/null; then
  CAP=(systemd-run --user --scope -q --slice="$HEMAIA_SLICE" -p MemoryMax="$mem" -p MemorySwapMax=0)
fi
case $mode in
  build)
    cfg=$1; tree=${2:-$T}
    R=$C/builds/$(basename "$cfg" .hjson)_$(date +%m%d_%H%M); mkdir -p "$(dirname "$R")"
    L=$R.log
    setsid nohup "${CAP[@]}" bash -c "cd '$tree' && python3 -u '$C/hsweep.py' build --cfg '$cfg' --tree '$tree' > '$L' 2>&1" > /dev/null 2>&1 &
    ;;
  run)
    exp=$1; tag=$2; pts=$3; shift 3
    R=$SW/$exp/runs/$tag; mkdir -p "$(dirname "$R")"
    L=$R.log
    # simv writes its transcript live (-l), so an assertion is visible while the runner still
    # holds the captured output
    export SIM_PLUSARGS="${SIM_PLUSARGS:--l sim_live.log}"
    setsid nohup "${CAP[@]}" bash -c "cd '$T' && python3 -u '$C/hsweep.py' run --exp '$exp' --tag '$tag' --points '$pts' $* > '$L' 2>&1" > /dev/null 2>&1 &
    ;;
  *) echo "usage: launch.sh build <cfg> [tree] | run <exp> <tag> <points.yaml> [args]"; exit 1;;
esac
sleep 3
# the runner's own pid: the python process whose arguments name this build's cfg or this run's tag
case $mode in
  build) key="hsweep.py build --cfg $cfg --tree $tree";;
  run)   key="hsweep.py run --exp $exp --tag $tag ";;
esac
PID=$(ps -u "$USER" -o pid=,args= | awk -v k="$key" 'index($0, k) && $2 ~ /python3?$/ {print $1; exit}')
setsid nohup "$C/watch.sh" "$R" "$L" "${PID:-0}" > /dev/null 2>&1 &
[ "$mode" = run ] && [ -n "$SIM_WATCHDOG" ] && "$C/watchdog.sh" "$R" "$SIM_WATCHDOG" | tail -1
echo "launched $mode: $R (runner pid ${PID:-?}; watchdog armed: $R.events; memory cap ${CAP:+$mem in $HEMAIA_SLICE}${CAP:-none})"
