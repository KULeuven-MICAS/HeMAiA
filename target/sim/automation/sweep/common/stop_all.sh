#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# stop_all.sh: stop every sweep run of mine, whatever its experiment: helpers first (so nothing relaunches or
# truncates), then the runners, then each simulator running in a runs/stg_* directory by PID with SIGINT.
me=$$
for p in $(ps -u $USER -o pid=,args= | awk '$3 ~ /sim_watchdog[.]py$/ || $4 ~ /sim_watchdog[.]py$/ || $4 ~ /stage_tick[.]sh$/ || $3 ~ /stage_tick[.]sh$/ || $3 ~ /trim_live[.]py$/ || $4 ~ /trim_live[.]py$/ || $3 ~ /progress_logger[.]sh$/ {print $1}'); do
  [ "$p" != "$me" ] && kill $p 2>/dev/null && echo "stopped helper $p"
done
for p in $(ps -u $USER -o pid=,args= | awk '/hsweep[.]py run/ && /--tag stg_/ {print $1}'); do kill $p 2>/dev/null && echo "stopped runner $p"; done
sleep 2
for p in $(ps -u $USER -o pid=,comm= | awk '$2=="occamy_chip.vcs" {print $1}'); do
  if readlink /proc/$p/cwd | grep -q "/runs/stg_"; then echo "SIGINT simv $p"; kill -INT $p; fi
done
sleep 40
echo "--- remaining:"; ps -u $USER -o pid=,comm= | awk '$2=="occamy_chip.vcs"' | wc -l
