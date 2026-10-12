#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# stage_tick.sh <tag> [seconds]: ONE waiter on a stage run. Returns after <seconds> (default 3600) with a short
# status, or at once on an EVENT: a watchdog stop or warning, a failed check, an assertion, an OOM, the run ending,
# or STALE. STALE = the run's events file has not changed for 25 min while a simulator runs: the watchdog writes a
# HEARTBEAT every ~10 min, so silence means the WATCHDOG itself died (the failure that went unseen for hours;
# bring it back with common/watchdog.sh).
# Arm one waiter at a time (check `ps -u $USER -o pid=,args= | grep stage_tick` first), re-arm it only after it
# returned, and do not report "still healthy" to the user between events.
S=$(cd "$(dirname "$0")/.." && pwd); R=$S/runs
T=$1; end=$(( $(date +%s) + ${2:-3600} ))
W=${FFN_WORK:-/tmp/opencode/w}; mkdir -p "$W"; SEEN=$W/seen_$T; touch "$SEEN"
PAT="WATCHDOG STOP|WATCHDOG WARN|WATCHDOG SLOW|WATCHDOG CHECK-ERROR|WATCHDOG no simulator|ASSERTION|CHECK FAIL|L1 OOM|BUILD ERROR|EXITED|DONE|GONE|LIVE WATCHDOG|LOWSPACE"
show() { echo "--- $(date +%H:%M)"; python3 "$S/../common/sim_watchdog.py" status "$R/$T"; "$S/pf_status.sh" stg_ | grep "^quota"; tail -n 3 "$R/$T.events" | cut -c1-200; }
while [ "$(date +%s)" -lt "$end" ]; do
  new=$(grep -hE "$PAT" "$R/$T.events" 2>/dev/null | sort -u | comm -13 <(sort -u "$SEEN") -)
  if [ -n "$new" ]; then echo "$new" | cut -c1-230; echo "$new" >> "$SEEN"; show; exit 0; fi
  sims=$(ps -u "$USER" -o pid=,comm= | awk '$2=="occamy_chip.vcs"' | wc -l)
  age=$(( $(date +%s) - $(stat -c %Y "$R/$T.events" 2>/dev/null || echo 0) ))
  if [ "$sims" -gt 0 ] && [ "$age" -gt 1500 ]; then echo "STALE: events of $T unchanged for $((age / 60)) min with a simulator running: the watchdog is dead or stuck"; show; exit 0; fi
  sleep 30
done
show
