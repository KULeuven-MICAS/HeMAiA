#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# E3 synchronisation: the cost of one cross-chip synchronisation, four mechanisms, on the
# standing hemaia_16c4m_8MBL3_2cluster simulator.
#   ./run.sh launch [ab] [c] [d]     one run per arm (default all three), each with its watchdog
#                                    -> runs/stg_sync_<arm>[_<n>]
#   ./run.sh parse <ab> <c> <d>      three run dirs -> results/four_arm_sync_latency.csv
#   ./run.sh [plot]                  results/four_arm_sync_latency.csv -> figs/fig_e3_four_arm.{pdf,png}
#
# The arms: ab = snax-test-chip-barrier under offload_legacy (A software barrier, B 0xFF
# broadcast barrier, one binary); c = offload_bingo_sw sync_latency_sweep_1cluster (BINGO
# software manager); d = offload_bingo_hw sync_latency_sweep_1cluster (BINGO hardware manager).
# PARALLEL=1 shares the standing simv with up to SIM_SLOTS (default 8) running simulations;
# otherwise launch refuses while another simulator of this user runs. TIMEOUT_H bounds a runner
# (defaults below, per arm).
set -eu
D=$(cd "$(dirname "$0")" && pwd)
H=${D%/target/sim/automation/sweep/*}
EXP=${D#"$H"/target/sim/automation/sweep/}
C=$H/target/sim/automation/sweep/common
CFG=target/rtl/cfg/hemaia_16c4m_8MBL3_2cluster.hjson
HW_WL=$H/target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/sync_latency_sweep_1cluster

launch_one() {
  local arm=$1 app wl dev bound th wd
  case $arm in
    ab) app=offload_legacy;   wl=None;                        dev=snax-test-chip-barrier; bound=8000;  th=10;;
    c)  app=offload_bingo_sw; wl=sync_latency_sweep_1cluster; dev=snax-bingo-offload;     bound=20000; th=24;;
    d)  app=offload_bingo_hw; wl=sync_latency_sweep_1cluster; dev=snax-bingo-offload;     bound=4000;  th=8;;
    *)  echo "unknown arm $arm (ab, c, d)"; exit 1;;
  esac
  th=${TIMEOUT_H:-$th}
  local n=0 tag=stg_sync_$arm
  while [ -e "$D/runs/$tag.events" ]; do n=$((n + 1)); tag=stg_sync_${arm}_$n; done
  mkdir -p "$D/runs"
  cat > "$D/runs/$tag.yaml" <<EOT
cfg: $CFG
memchip: true
fgp_threads: 0
timeout_h: $th
points:
  - {name: sync_$arm, host_app_type: $app, chip_type: multi_chip, workload: $wl, dev_app: $dev}
EOT
  SIM_PLUSARGS="-l sim_live.log +vcs+finish+$((bound * 1000000))" \
    "$C/launch.sh" run "$EXP" "$tag" "$D/runs/$tag.yaml" --jobs 1 | cut -c1-120
  # Arm d has a BINGO hardware task list, which the scheduler monitor follows. Arms ab and c have
  # none, so the guard there watches the clock, the logs and the bound only (arm_deadline_us=0).
  if [ "$arm" = d ]; then
    wd=(cerf_node "dfg=$HW_WL/final_dfg.csv")
  else
    wd=(default arm_deadline_us=0)
  fi
  "$C/watchdog.sh" "$D/runs/$tag" "${wd[@]}" bound_us=$bound min_threads=1 find_timeout_s=21600 | tail -1
  echo "$tag: bound $bound us, timeout $th h; events $D/runs/$tag.events"
}

case ${1:-plot} in
  launch)
    shift
    arms=("$@"); [ ${#arms[@]} -eq 0 ] && arms=(ab c d)
    busy=$(ps -u "$USER" -o pid=,comm= | awk '$2=="occamy_chip.vcs" {print $1}' | tr '\n' ' ')
    if [ "${PARALLEL:-0}" != 1 ]; then
      [ -n "$busy" ] && { echo "a simulator is running (pid $busy): PARALLEL=1 to share the simv"; exit 1; }
    fi
    export SIM_SLOTS=${SIM_SLOTS:-8}
    for arm in "${arms[@]}"; do launch_one "$arm"; done
    ;;
  parse)
    shift
    python3 "$D/parse_sync.py" --ab "$1" --c "$2" --d "$3"
    ;;
  plot)
    python3 "$D/plot_sync.py"
    ;;
  *) sed -n 6,18p "$0"; exit 1;;
esac
