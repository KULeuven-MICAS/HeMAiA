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
source /users/micas/fkong/no_backup/src_hemaia_eda.sh >/dev/null 2>&1
[ "${VCS_HOME##*/}" = "W-2024.09-SP2" ] || { echo "BAD EDA ENV: $VCS_HOME"; exit 1; }
mode=$1; shift
case $mode in
  build)
    cfg=$1; tree=${2:-/users/micas/fkong/no_backup/HeMAiA}
    R=$C/builds/$(basename "$cfg" .hjson)_$(date +%m%d_%H%M); mkdir -p "$(dirname "$R")"
    L=$R.log
    setsid nohup bash -c "cd '$tree' && python3 -u '$C/hsweep.py' build --cfg '$cfg' --tree '$tree' > '$L' 2>&1" > /dev/null 2>&1 &
    ;;
  run)
    exp=$1; tag=$2; pts=$3; shift 3
    R=$SW/$exp/runs/$tag; mkdir -p "$(dirname "$R")"
    L=$R.log
    # simv writes its transcript live (-l), so an assertion is visible while the runner still
    # holds the captured output
    export SIM_PLUSARGS="${SIM_PLUSARGS:--l sim_live.log}"
    setsid nohup bash -c "cd /users/micas/fkong/no_backup/HeMAiA && python3 -u '$C/hsweep.py' run --exp '$exp' --tag '$tag' --points '$pts' $* > '$L' 2>&1" > /dev/null 2>&1 &
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
echo "launched $mode: $R (runner pid ${PID:-?}; watchdog armed: $R.events)"
