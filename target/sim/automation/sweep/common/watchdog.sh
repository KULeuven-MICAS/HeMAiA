#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# watchdog.sh <run dir> [profile] [key=value ...]: start sim_watchdog.py for one run, detached from this shell
# (it needs no one to keep it alive), and refuse a second one for the same run. Run it again to bring a dead
# watchdog back: it finds the simulator by name and directory, so it can start before or after the simulator.
#   watchdog.sh target/sim/automation/sweep/ffn/runs/stg_full ffn_prefill bound_us=2500
# Status:  sim_watchdog.py status <run dir>     Parameters:  sim_watchdog.py params --profile <profile>
C=$(cd "$(dirname "$0")" && pwd)
R=${1:?run dir}; P=${2:-default}; shift 2 2>/dev/null || shift $#
sets=(); for kv in "$@"; do sets+=(--set "$kv"); done
setsid nohup python3 -u "$C/sim_watchdog.py" watch --run-dir "$R" --profile "$P" "${sets[@]}" > "$R.watchdog.out" 2>&1 < /dev/null &
sleep 2
cat "$R.watchdog.out"
echo "watchdog for $R: profile $P, events $R.events, status $R.watchdog.json"
