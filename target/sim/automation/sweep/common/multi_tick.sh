#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# multi_tick.sh <seconds> <tag> [tag ...]: ONE waiter over several runs (runs/<tag> of any experiment: _runs.sh finds
# each; EXP picks one when two have a tag). Returns at once on the first
# EVENT of any of them (a watchdog stop or warning, a failed check, an assertion, an OOM, the run ending) or on STALE
# (a run's events file silent for 25 min while its simulator runs: its watchdog died), else after <seconds>, and
# prints one status line per run. Events already reported by an earlier call (kept in the run's runs/.tick) are not
# reported again.
. "$(dirname "$0")/_runs.sh"
secs=$1; shift
end=$(( $(date +%s) + secs ))
declare -A RD
for t in "$@"; do RD[$t]=$(runs_of "$t") || exit 1; done
PAT="WATCHDOG STOP|WATCHDOG WARN|WATCHDOG SLOW|WATCHDOG CHECK-ERROR|WATCHDOG no simulator|ASSERTION|CHECK FAIL|L1 OOM|BUILD ERROR|EXITED|DONE|GONE|LIVE WATCHDOG|LOWSPACE"
show() {
  echo "--- $(date +%H:%M)"
  for t in "$@"; do python3 "$SW/common/sim_watchdog.py" status "${RD[$t]}/$t" 2>/dev/null | cut -c1-230; done
  q=$(quota -u "$USER" 2>/dev/null | tail -1 | awk '{print $1}' | tr -d '*'); echo "quota headroom $(( (157286400 - ${q:-0}) / 1048576 )) GB"
}
while [ "$(date +%s)" -lt "$end" ]; do
  hit=""
  for t in "$@"; do
    W=${FFN_WORK:-${RD[$t]}/.tick}; mkdir -p "$W"; seen=$W/seen_$t; touch "$seen"
    new=$(grep -hE "$PAT" "${RD[$t]}/$t.events" 2>/dev/null | sort -u | comm -13 <(sort -u "$seen") -)
    if [ -n "$new" ]; then echo "$new" | cut -c1-230; echo "$new" >> "$seen"; hit=1; fi
    st=$(python3 "$SW/common/sim_watchdog.py" status "${RD[$t]}/$t" 2>/dev/null)
    case "$st" in *STALE*) echo "STALE: $st" | cut -c1-230; hit=1;; esac
  done
  [ -n "$hit" ] && { show "$@"; exit 0; }
  sleep 30
done
show "$@"
