#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# L2 FA: the one button.
#   ./run.sh run   [variant ...]     simulate the variants (default: all, in the order below),
#                                    one runner each through common/launch.sh (watchdog, sim
#                                    slots), at most $FA_PAR runners in flight; then verdict +
#                                    parse + prune for each
#   ./run.sh parse [tag ...]         verdict, parse, prune
#   ./run.sh all                     run
# A variant is its path under workloads/fa/ (llama3_decode_full/all, ...); its switches are its
# params.hjson. Runs land in runs/l2f_<variant with / as _>. stages.sh launches the same workloads one at a
# time with their watchdog profile (stg_<stage> tags); the paper's runs went through it.
set -u
D=$(cd "$(dirname "$0")" && pwd)
SW=${D%/sweep/*}/sweep; EXP=${D#"$SW"/}   # the sweep dir, and this experiment: its path under it
# the ladder R0 -> R3 of the three mechanisms (multicast broadcast, in-transit reduce, hardware
# sync), each rung adding one: Llama-3-8B's whole decode attention block over an 8,192-position cache
ALL="llama3_decode_full/base llama3_decode_full/bcast llama3_decode_full/bcast_reduce llama3_decode_full/all"
FA_PAR=${FA_PAR:-3}
CFG=target/rtl/cfg/hemaia_16c4m_8MBL3_2cluster.hjson
cmd=${1:-all}; shift || true

tag_of() { echo "l2f_${1//\//_}"; }

parse_one() {
  local r=$D/runs/$1
  [ -d "$r" ] || { echo "no run $r"; return 1; }
  python3 "$SW/common/hsweep.py" verdict "$r"
  python3 "$D/parse_fa.py" "$r"
  python3 "$SW/common/hsweep.py" prune "$r"
}

launch_one() {
  local w=$1 tag; tag=$(tag_of "$w")
  mkdir -p "$D/runs"
  cat > "$D/runs/$tag.yaml" <<EOT
cfg: $CFG
memchip: true
timeout_h: 6
points:
  - {name: fa_${w//\//_}, host_app_type: offload_bingo_hw, chip_type: multi_chip,
     workload: fa/$w, dev_app: snax-bingo-offload}
EOT
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
    ws=${*:-$ALL}
    running=()
    for w in $ws; do
      launch_one "$w"
      running+=("$(tag_of "$w")")
      if [ ${#running[@]} -ge "$FA_PAR" ]; then
        wait_one "${running[0]}" || true
        running=("${running[@]:1}")
      fi
    done
    for t in "${running[@]}"; do wait_one "$t" || true; done ;;
  parse)
    tags=${*:-$(for w in $ALL; do tag_of "$w"; done)}
    for t in $tags; do parse_one "$t"; done ;;
  all)
    "$0" run ;;
  *) echo "usage: run.sh run [variant ...] | parse [tag ...] | all"; exit 1 ;;
esac
