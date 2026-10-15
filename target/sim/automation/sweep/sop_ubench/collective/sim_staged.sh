#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# sim_staged.sh <primitive> [timeout]: simulate the already-staged task of runs/$COLL_TAG_<primitive>
# (built and staged by a run.sh runner), then verdict + parse + prune, as run.sh would. For when
# a runner has staged its task but cannot get a simulation slot. Appends "RUN EXIT <code>" to
# runs/e2l_<primitive>.log.
set -u
D=$(cd "$(dirname "$0")" && pwd)
SW=${D%/sweep/*}/sweep
p=$1; tmo=${2:-5h}
tag=${COLL_TAG:-e2d}_$p
B=$(ls -d "$D/runs/$tag"/task_*/bin | head -1)
[ -x "$B/occamy_chip.vcs" ] || { echo "no staged simv in $B"; exit 1; }
source /users/micas/fkong/no_backup/src_hemaia_eda.sh >/dev/null 2>&1
rm -f "$B/sim_run.log"
echo "[sim_staged] SIM START $(date +%H:%M) $B" >> "$D/runs/$tag.log"
# The Snitch instruction traces are compiled into this simv (only +define+SNITCH_NO_TRACE removes
# them), and a polling DM core writes ~200 B per cycle. The parse needs only the UARTs and the
# SRAM marks, so every Snitch trace file is pre-created as a link to /dev/null (the tracer
# opens a fixed name under logs/ with "w"). KEEP_ROOT_TRACE=1 keeps the root's DM core (hart 4
# of chip 00) for debugging; KEEP_TRACES=1 keeps them all.
if [ "${KEEP_TRACES:-0}" != 1 ]; then
  mkdir -p "$B/logs"
  for a in "$B"/app_chip_*; do
    xy=$(basename "$a" | sed 's/app_chip_\([0-9a-f]*\)_\([0-9a-f]*\)/\1\2/')
    for h in 1 2 3 4 5 6 7 8; do
      [ "${KEEP_ROOT_TRACE:-0}" = 1 ] && [ "$xy" = 00 ] && [ $h = 4 ] && continue
      ln -sf /dev/null "$B/logs/trace_chip_${xy}_hart_0000${h}.dasm"
    done
  done
fi
(cd "$B" && timeout "$tmo" ./occamy_chip.vcs +vcs+lic+wait -no_save ${SIM_PLUSARGS:-} > sim_run.log 2>&1)
rc=$?
[ $rc -eq 124 ] && echo "TIMEOUT: killed after $tmo" >> "$B/sim_run.log"
echo "[sim_staged] sim exit $rc $(date +%H:%M)" >> "$D/runs/$tag.log"
echo "RUN EXIT $rc" >> "$D/runs/$tag.log"
python3 "$SW/common/hsweep.py" verdict "$D/runs/$tag" >> "$D/runs/$tag.log" 2>&1
mkdir -p "$D/results"
flock "$D/results/.parse.lock" python3 "$D/parse_coll.py" "$D/runs/$tag" >> "$D/runs/$tag.log" 2>&1
python3 "$SW/common/hsweep.py" prune "$D/runs/$tag" >> "$D/runs/$tag.log" 2>&1
echo "[sim_staged] DONE $(date +%H:%M)" >> "$D/runs/$tag.log"
