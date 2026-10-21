#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# E2 reduce: the one button.
#   ./run.sh run                     simulate the reduce through common/launch.sh (watchdog, sim
#                                    slots); then verdict + parse + prune
#   ./run.sh parse [tag ...]         verdict, parse, prune
#   ./run.sh plot                    the figures from results/collective.csv
#   ./run.sh all                     run + plot
# The device app is snax-coll-reduce under offload_legacy/multi_chip (software sync only). Its
# points (P, implementations, repeats, payload) are compiled in from
# target/sw/device/apps/snax/snax-coll-reduce/data/params.hjson. The run lands in
# runs/$COLL_TAG_reduce.
set -u
D=$(cd "$(dirname "$0")" && pwd)
SW=${D%/sweep/*}/sweep; EXP=${D#"$SW"/}   # the sweep dir, and this experiment: its path under it
ALL="reduce"   # the primitives: the device app is snax-coll-<primitive>
COLL_PAR=${COLL_PAR:-5}
# the platform of the layer experiments: 16 compute chiplets (2 split clusters each) and 4
# memory chiplets in one column, 5 x 4
CFG=${COLL_CFG:-target/rtl/cfg/hemaia_16c4m_8MBL3_2cluster.hjson}
cmd=${1:-all}; shift || true

# COLL_TAG: the run tag prefix (e2l: the first, SDR-link round; e2d: links in DDR)
TAG=${COLL_TAG:-e2d}
tag_of() { echo "${TAG}_$1"; }

parse_one() {
  local r=$D/runs/$1
  [ -d "$r" ] || { echo "no run $r"; return 1; }
  python3 "$SW/common/hsweep.py" verdict "$r"
  python3 "$D/parse_coll.py" "$r"
  python3 "$SW/common/hsweep.py" prune "$r"
}

launch_one() {
  local p=$1 tag; tag=$(tag_of "$p")
  mkdir -p "$D/runs"
  cat > "$D/runs/$tag.yaml" <<EOF
cfg: $CFG
memchip: true
timeout_h: 5
points:
  - {name: $p, host_app_type: offload_legacy, chip_type: multi_chip,
     workload: None, dev_app: snax-coll-$p}
EOF
  "$SW/common/launch.sh" run "$EXP" "$tag" "$D/runs/$tag.yaml" --jobs 1
}

wait_one() {
  local tag=$1
  while ! grep -q "RUN EXIT" "$D/runs/$tag.log" 2>/dev/null; do
    if grep -q "GONE" "$D/runs/$tag.events" 2>/dev/null; then echo "$tag: runner gone"; return 1; fi
    sleep 60
  done
  parse_one "$tag"
}

case $cmd in
  run)
    ps=${*:-$ALL}
    running=()
    for p in $ps; do
      launch_one "$p"
      running+=("$(tag_of "$p")")
      if [ ${#running[@]} -ge "$COLL_PAR" ]; then
        wait_one "${running[0]}" || true
        running=("${running[@]:1}")
      fi
    done
    for t in "${running[@]}"; do wait_one "$t" || true; done ;;
  parse)
    tags=${*:-$(for p in $ALL; do tag_of "$p"; done)}
    for t in $tags; do parse_one "$t"; done ;;
  plot)
    python3 "$D/plot_coll.py" ;;
  all)
    "$0" run && "$0" plot ;;
  *) echo "usage: run.sh run | parse [tag ...] | plot | all"; exit 1 ;;
esac
