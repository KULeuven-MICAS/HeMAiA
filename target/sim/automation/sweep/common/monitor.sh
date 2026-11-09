#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# monitor.sh <minutes> [run tag ...]: every <minutes>, runtime_mac.py on the runs whose watchdog still says running (or
# that are still building), one line each appended to runs/$MON_NAME.log (default monitor.log), then a SAMPLE line.
# The runs are those of EXP (an experiment under sweep/, or RUNS_DIR), or, with tags and no EXP, of the first tag's
# experiment (every tag must be in it). Without tags it watches every runs/stg_* run whose watchdog says running, picked
# again each round, and ends when none is left and no queue (queue.sh, runs/*.pending) waits. Report only: stopping a
# hung simulator is the run's sim_watchdog's job. Start it detached (setsid nohup ... &), so it outlives the shell and
# the session that started it.
. "$(dirname "$0")/_runs.sh"
M=${1:?minutes}; shift
auto=0; [ $# -eq 0 ] && auto=1
if [ $auto = 1 ] || [ -n "${EXP:-}${RUNS_DIR:-}" ]; then R=$(exp_runs) || exit 1; else R=$(runs_of "$1") || exit 1; fi
export RUNS_DIR=$R
LOG=$R/${MON_NAME:-monitor}.log
echo "$(date '+%d %H:%M') monitor started (pid $$, every $M min) for: ${*:-every running stg_* run}" >> "$LOG"
while :; do
    sleep $((M * 60))
    if [ $auto = 1 ]; then
        set --
        for j in "$R"/stg_*.watchdog.json; do
            [ "$(python3 -c "import json; print(json.load(open('$j')).get('state'))" 2>/dev/null)" = running ] \
                && set -- "$@" "$(basename "$j" .watchdog.json)"
        done
        if [ $# -eq 0 ]; then
            compgen -G "$R/*.pending" > /dev/null && { echo "$(date '+%d %H:%M') SAMPLE no run running, queue waiting" >> "$LOG"; continue; }
            echo "$(date '+%d %H:%M') monitor ends: no run left" >> "$LOG"; exit 0
        fi
    fi
    live=()
    for t in "$@"; do
        st=$(python3 -c "import json; print(json.load(open('$R/$t.watchdog.json')).get('state'))" 2>/dev/null)
        if [ "$st" = running ] || [ ! -e "$R/$t.watchdog.json" ]; then live+=("$t")
        else echo "$(date '+%d %H:%M') $t: watchdog state $st, no longer watched" >> "$LOG"; fi
    done
    [ ${#live[@]} -eq 0 ] && { echo "$(date '+%d %H:%M') monitor ends: no run left" >> "$LOG"; exit 0; }
    set -- "${live[@]}"
    for t in "$@"; do
        out=$(timeout 900 python3 "$SW/common/runtime_mac.py" "$t" 2>&1 | tail -1)
        echo "$(date '+%d') $out" >> "$LOG"
    done
    echo "$(date '+%d %H:%M') SAMPLE $(quota -s 2>/dev/null | tail -1 | awk '{print "quota " $1 " of " $3}')" >> "$LOG"
done
