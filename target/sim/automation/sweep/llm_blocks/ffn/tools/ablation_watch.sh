#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# ablation_watch.sh <minutes> <run tag> ...: every <minutes>, runtime_mac.py on the runs whose watchdog still says
# running (or that are still building), one line each appended to runs/ablation_monitor.log, then a SAMPLE line.
# Report only: stopping a hung simulator is the run's sim_watchdog's job. Ends when no run is left. Start it
# detached (setsid nohup ... &), so it outlives the shell and the session that started it.
D=$(cd "$(dirname "$0")" && pwd); R=$(dirname "$D")/runs; LOG=$R/ablation_monitor.log
M=${1:?minutes}; shift
echo "$(date '+%d %H:%M') monitor started (pid $$, every $M min) for: $*" >> "$LOG"
while :; do
    sleep $((M * 60))
    live=()
    for t in "$@"; do
        st=$(python3 -c "import json; print(json.load(open('$R/$t.watchdog.json')).get('state'))" 2>/dev/null)
        if [ "$st" = running ] || [ ! -e "$R/$t.watchdog.json" ]; then live+=("$t")
        else echo "$(date '+%d %H:%M') $t: watchdog state $st, no longer watched" >> "$LOG"; fi
    done
    [ ${#live[@]} -eq 0 ] && { echo "$(date '+%d %H:%M') monitor ends: no run left" >> "$LOG"; exit 0; }
    set -- "${live[@]}"
    for t in "$@"; do
        out=$(timeout 900 python3 "$D/runtime_mac.py" "$t" 2>&1 | tail -1)
        echo "$(date '+%d') $out" >> "$LOG"
    done
    echo "$(date '+%d %H:%M') SAMPLE $(quota -s 2>/dev/null | tail -1 | awk '{print "quota " $1 " of " $3}')" >> "$LOG"
done
