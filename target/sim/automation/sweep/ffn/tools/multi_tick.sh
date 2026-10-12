#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# multi_tick.sh <seconds> <tag> [tag ...]: ONE waiter over several runs (runs/<tag>). Returns at once on the first
# EVENT of any of them (a watchdog stop or warning, a failed check, an assertion, an OOM, the run ending) or on STALE
# (a run's events file silent for 25 min while its simulator runs: its watchdog died), else after <seconds>, and
# prints one status line per run. Events already reported by an earlier call are not reported again.
S=$(cd "$(dirname "$0")/.." && pwd); R=$S/runs
secs=$1; shift
end=$(( $(date +%s) + secs ))
W=${FFN_WORK:-$S/runs/.tick}; mkdir -p "$W"
PAT="WATCHDOG STOP|WATCHDOG WARN|WATCHDOG SLOW|WATCHDOG CHECK-ERROR|WATCHDOG no simulator|ASSERTION|CHECK FAIL|L1 OOM|BUILD ERROR|EXITED|DONE|GONE|LIVE WATCHDOG|LOWSPACE"
show() {
  echo "--- $(date +%H:%M)"
  for t in "$@"; do python3 "$S/../common/sim_watchdog.py" status "$R/$t" 2>/dev/null | cut -c1-230; done
  q=$(quota -u "$USER" 2>/dev/null | tail -1 | awk '{print $1}' | tr -d '*'); echo "quota headroom $(( (157286400 - ${q:-0}) / 1048576 )) GB"
}
while [ "$(date +%s)" -lt "$end" ]; do
  hit=""
  for t in "$@"; do
    seen=$W/seen_$t; touch "$seen"
    new=$(grep -hE "$PAT" "$R/$t.events" 2>/dev/null | sort -u | comm -13 <(sort -u "$seen") -)
    if [ -n "$new" ]; then echo "$new" | cut -c1-230; echo "$new" >> "$seen"; hit=1; fi
    st=$(python3 "$S/../common/sim_watchdog.py" status "$R/$t" 2>/dev/null)
    case "$st" in *STALE*) echo "STALE: $st" | cut -c1-230; hit=1;; esac
  done
  [ -n "$hit" ] && { show "$@"; exit 0; }
  sleep 30
done
show "$@"
