#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# wait_sample.sh [max minutes]: wait until runs/ablation_monitor.log gets its next SAMPLE line (or a run's watchdog
# writes a STOP / EXITED / DONE event), then print what the log gained. The waiter a session arms over the monitor.
D=$(cd "$(dirname "$0")" && pwd); R=$(dirname "$D")/runs; LOG=$R/ablation_monitor.log
end=$(( $(date +%s) + ${1:-40} * 60 ))
n0=$(cat "$LOG" 2>/dev/null | wc -l); s0=$(cat "$LOG" 2>/dev/null | grep -c SAMPLE)
e0=$(cat "$R"/stg_abl_*.events 2>/dev/null | grep -cE "WATCHDOG STOP|EXITED|DONE")
while [ "$(date +%s)" -lt "$end" ]; do
    sleep 60
    s=$(cat "$LOG" 2>/dev/null | grep -c SAMPLE)
    e=$(cat "$R"/stg_abl_*.events 2>/dev/null | grep -cE "WATCHDOG STOP|EXITED|DONE")
    [ "$s" -gt "$s0" ] || [ "$e" -gt "$e0" ] && break
done
tail -n +$((n0 + 1)) "$LOG"
grep -hE "WATCHDOG STOP|EXITED|DONE" "$R"/stg_abl_*.events 2>/dev/null | tail -5
