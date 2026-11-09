#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# queue.sh <stage> ...: launch the stages of EXP (an experiment under sweep/, e.g. llm_blocks/ffn: its stages.sh,
# PARALLEL=1) one after another as simulator slots free: every 5 minutes, while fewer than SIM_SLOTS (default 8)
# simulators of this user run, the next stage starts; after a launch it waits 10 minutes for the simulator to appear.
# The stages still waiting are in runs/$QUEUE_NAME.pending (QUEUE_NAME default queue: a second queue with its own name
# can carry its own environment, e.g. BOUND_US and WD_SET for stages.sh); each launch is logged to runs/$MON_NAME.log
# (default monitor.log). Start it detached (setsid nohup ... &).
. "$(dirname "$0")/_runs.sh"
[ -n "${EXP:-}" ] || { echo "set EXP=<experiment under sweep/, e.g. llm_blocks/ffn>" >&2; exit 1; }
S=$SW/$EXP; R=$S/runs; LOG=$R/${MON_NAME:-monitor}.log; NAME=${QUEUE_NAME:-queue}; Q=$R/$NAME.pending
[ -x "$S/stages.sh" ] || { echo "no $S/stages.sh" >&2; exit 1; }
SLOTS=${SIM_SLOTS:-8}
mkdir -p "$R"; printf '%s\n' "$@" > "$Q"
echo "$(date '+%d %H:%M') queue $NAME started (pid $$, $SLOTS slots): $*" >> "$LOG"
while [ -s "$Q" ]; do
    n=$(ps -u "$USER" -o comm= | grep -c '^occamy_chip.vcs$')
    if [ "$n" -lt "$SLOTS" ]; then
        st=$(head -1 "$Q"); sed -i 1d "$Q"
        out=$(cd "$S" && PARALLEL=1 TIMEOUT_H=${TIMEOUT_H:-24} ./stages.sh "$st" 2>&1 | tail -1)
        echo "$(date '+%d %H:%M') queue $NAME: $st launched ($n simulators were running): $out" >> "$LOG"
        sleep 600
    else
        sleep 300
    fi
done
rm -f "$Q"
echo "$(date '+%d %H:%M') queue $NAME ends" >> "$LOG"
