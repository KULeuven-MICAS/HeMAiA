#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# run_staged.sh [primitive ...]: build and stage each primitive through a common/launch.sh runner
# (app builds serialised by the tree lock), then simulate it with sim_staged.sh without waiting
# for a slot in the shared queue: once the runner has staged its task, the runner is stopped and
# sim_staged.sh runs the simulation, verdict, parse and prune. The tag is $COLL_TAG_<primitive>
# (default e2d). Runs detached; progress in runs/$COLL_TAG_staged.log.
set -u
D=$(cd "$(dirname "$0")" && pwd)
SW=${D%/sweep/*}/sweep; EXP=${D#"$SW"/}   # the sweep dir, and this experiment: its path under it
TAG=${COLL_TAG:-e2d}
CFG=${COLL_CFG:-target/rtl/cfg/hemaia_16c4m_8MBL3_2cluster.hjson}
ps_=${*:-broadcast reduce allreduce}
LOG=$D/runs/${TAG}_staged.log
mkdir -p "$D/runs"
log() { echo "$(date +%H:%M) $*" >> "$LOG"; }

for p in $ps_; do
  tag=${TAG}_$p
  cat > "$D/runs/$tag.yaml" <<EOF
cfg: $CFG
memchip: true
timeout_h: 5
points:
  - {name: $p, host_app_type: offload_legacy, chip_type: multi_chip,
     workload: None, dev_app: snax-coll-$p}
EOF
  "$SW/common/launch.sh" run "$EXP" "$tag" "$D/runs/$tag.yaml" --jobs 1 >> "$LOG" 2>&1
done

pending=" $ps_ "
while [ -n "$(echo $pending)" ]; do
  for p in $pending; do
    tag=${TAG}_$p
    L=$D/runs/$tag.log
    if grep -q "Traceback\|make.*\*\*\*" "$L" 2>/dev/null; then
      log "$tag build failed"; pending=${pending/ $p / }; continue
    fi
    grep -q "^\[Step 5\] Running simulations" "$L" 2>/dev/null || continue
    rp=$(ps -u "$USER" -o pid=,args= | awk -v k="hsweep.py run --exp $EXP --tag $tag " 'index($0, k) && $2 ~ /python3/ {print $1}')
    wp=$(ps -u "$USER" -o pid=,args= | awk -v k="collective/runs/$tag " 'index($0, k) && /watch.sh/ {print $1}')
    [ -n "$rp" ] && kill $rp
    [ -n "$wp" ] && kill $wp
    COLL_TAG=$TAG setsid nohup "$D/sim_staged.sh" "$p" "${COLL_TIMEOUT:-5h}" "$tag" > /dev/null 2>&1 &
    sleep 2
    sp=$(ps -u "$USER" -o pid=,args= | awk -v k="sim_staged.sh $p ${COLL_TIMEOUT:-5h} $tag" 'index($0, k) {print $1; exit}')
    setsid nohup "$SW/common/watch.sh" "$D/runs/$tag" "$L" "${sp:-0}" > /dev/null 2>&1 &
    log "$tag staged: runner $rp stopped, sim_staged $sp started"
    pending=${pending/ $p / }
  done
  sleep 30
done
log "all staged"
