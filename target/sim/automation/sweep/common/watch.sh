#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# watch.sh <run dir> <runner log> <runner pid>: a run's watchdog, detached for the run's whole
# life. It appends one line per event to <run dir>.events and to common/all.events, and it never
# kills anything.
#   ARMED / BUILD ERROR / STAGED / SIM START / HEARTBEAT (sim time, every 20 min) / CHECK FAIL /
#   L1 OOM / LIVE WATCHDOG / ASSERTION / FROZEN (host sim time unchanged 20 min) / EXITED (pass, fail) /
#   LOWSPACE / DONE (the runner's RUN EXIT) / GONE (runner gone without RUN EXIT)
# Liveness is simulation time (the host trace's last "<n>ns"), not trace growth: a waiting
# scheduler keeps writing trace while making no progress. The host core writes nothing once a workload
# window opens, so FROZEN is only judged for a simulator without the scheduler monitor (no
# bingo_progress.log); with it, sim_watchdog.py owns liveness.
R=$1; L=$2; PID=$3
ALL=$(dirname "$0")/all.events
E=$R.events
run=$(basename "$R")
emit() { local l="$(date +%H:%M) $run $*"; echo "$l" >> "$E"; echo "$l" >> "$ALL"; }
# A run tag can be launched again while the old run's watchdog is still alive. Only the watchdog
# armed last for a run dir speaks for it: a superseded one stops without writing anything.
current() { [ "$(grep -o 'ARMED (runner pid [0-9]*)' "$E" 2>/dev/null | tail -1)" = "ARMED (runner pid $PID)" ]; }
declare -A seen tlast tsince hb
emit "ARMED (runner pid $PID)"
lowspace=0
while true; do
  sleep 60
  current || exit 0
  if [ -z "${seen[builderr]}" ] && grep -qE "Traceback|make(\[[0-9]+\])?: \*\*\*" "$L" 2>/dev/null; then
    seen[builderr]=1; emit "BUILD ERROR: $(grep -m1 -E 'Error|make(\[[0-9]+\])?: \*\*\*' "$L" | cut -c1-160)"
  fi
  for d in "$R"/task_*; do
    [ -d "$d/bin" ] || continue
    t=$(basename "$d" | cut -c1-60); b=$d/bin
    [ -z "${seen[st$t]}" ] && { seen[st$t]=1; emit "STAGED $t"; }
    if [ -z "${seen[ss$t]}" ] && ls "$b"/logs/trace_chip_00_hart_00000.log >/dev/null 2>&1; then
      seen[ss$t]=1; tsince[$t]=$(date +%s); hb[$t]=$(date +%s); emit "SIM START $t"
    fi
    [ -z "${seen[ss$t]}" ] && continue
    if [ -z "${seen[oom$t]}" ] && grep -q "L1 malloc failed" "$b"/uart_chip_*.log 2>/dev/null; then
      seen[oom$t]=1; emit "L1 OOM $t"
    fi
    if [ -z "${seen[lw$t]}" ] && grep -q "LIVE WATCHDOG" "$b"/snax_probes.log 2>/dev/null; then
      seen[lw$t]=1; emit "LIVE WATCHDOG $t"
    fi
    # the first SystemVerilog assertion failure; simv stops on it (the runner's -assert
    # global_finish_maxfail), so the line names what went wrong and where
    if [ -z "${seen[as$t]}" ] && grep -qE "ASSERT FAILED|^Error: \"|Too many assertion failures" "$b"/sim_live.log 2>/dev/null; then
      seen[as$t]=1; emit "ASSERTION $t: $(grep -m1 -E '^Error: "' "$b"/sim_live.log | sed 's/.*testharness\.i_dut\.//' | cut -c1-200)"
    fi
    grep -h "Check \[.*FAIL" "$b"/uart_chip_*.log 2>/dev/null | head -5 | while read -r line; do
      k=$(echo "$t$line" | md5sum | cut -c1-8)
      grep -q "$k" "$E.fails" 2>/dev/null || { echo "$k" >> "$E.fails"; emit "CHECK FAIL $t: $(echo "$line" | cut -c1-120)"; }
    done
    if [ -z "${seen[ex$t]}" ] && grep -qE "All chips finished|finished with errors|TIMEOUT" "$b"/sim_run.log 2>/dev/null; then
      seen[ex$t]=1
      p=$(cat "$b"/uart_chip_*.log 2>/dev/null | grep -c "Check \[.*PASS"); fl=$(cat "$b"/uart_chip_*.log 2>/dev/null | grep -c "Check \[.*FAIL")
      emit "EXITED $t: $(grep -m1 -oE 'All chips finished successfully|finished with errors|TIMEOUT' "$b"/sim_run.log), checks pass $p fail $fl"
      continue
    fi
    [ -n "${seen[ex$t]}" ] && continue
    st=$(tail -c 4000 "$b"/logs/trace_chip_00_hart_00000.log 2>/dev/null | grep -oE "^ *[0-9]+ns" | tail -1 | tr -dc 0-9)
    [ -z "$st" ] && continue
    if [ "$st" != "${tlast[$t]}" ]; then tlast[$t]=$st; tsince[$t]=$(date +%s)
    elif [ $(( $(date +%s) - ${tsince[$t]} )) -gt 1200 ] && [ -z "${seen[fz$t]}" ] && [ ! -e "$b/bingo_progress.log" ]; then
      seen[fz$t]=1; emit "FROZEN $t: host sim time stuck at $((st/1000)) us for 20 min"
    fi
    if [ $(( $(date +%s) - ${hb[$t]} )) -ge 1200 ]; then
      hb[$t]=$(date +%s); emit "HEARTBEAT $t: sim time $((st/1000)) us"
    fi
  done
  fr=$(df --output=avail -BG /tmp | tail -1 | tr -dc 0-9)
  if [ "$fr" -lt 10 ] && [ $lowspace = 0 ]; then lowspace=1; emit "LOWSPACE: ${fr}G free in /tmp"; fi
  if grep -q "RUN EXIT" "$L" 2>/dev/null; then emit "DONE ($(grep -m1 -E 'verdicts|DONE in' "$L" | cut -c1-200))"; exit 0; fi
  if ! kill -0 "$PID" 2>/dev/null; then
    sleep 20; grep -q "RUN EXIT" "$L" 2>/dev/null && { emit "DONE"; exit 0; }
    emit "GONE: runner exited without RUN EXIT ($(tail -n 2 "$L" | tr '\n' ' ' | cut -c1-160))"; exit 0
  fi
done
