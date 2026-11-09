#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# gen_check.sh <workload under workloads/, e.g. ffn/dsv2_l0_prefill/stage0> <out dir> <log>: the generator alone, its
# static checks included, with every output (headers, final_dfg.csv, the HBM image) written to <out dir>, never into the
# workload directory: a running simulation maps that directory's build/hbm image, and rewriting it under the simulation
# crashes it (hemaia_hbm_read). It runs in the build container the apps' make uses (HEMAIA_IMAGE; memory cap
# HEMAIA_CONTAINER_MEM, default 96g), with <out dir> as its working directory; the data header is named as the
# workload's block .mk names it (DATA_H).
H=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
W=$H/target/sw/host/apps/offload_bingo_hw/multi_chip/workloads
SN=$H/deps/snitch_cluster
inc=$(sed -n 's/^include \$(MK_DIR)//p' "$W/$1/Makefile" 2>/dev/null | head -1)
dh=$(sed -n 's/^DATA_H = \$(MK_DIR)//p' "$W/$1/$inc" 2>/dev/null | head -1)
[ -n "$dh" ] || { echo "gen_check: no workload $1, or no DATA_H in its block .mk" >&2; exit 1; }
mkdir -p "$2" && O=$(cd "$2" && pwd) || exit 1
podman run --rm --memory "${HEMAIA_CONTAINER_MEM:-96g}" -v "$H:$H:z" -v "$O:$O:z" -w "$O" \
  "${HEMAIA_IMAGE:-ghcr.io/kuleuven-micas/hemaia:main}" \
  python3 "$W/$1/main_bingo.py" --output_dir "$O" --data_h "$O/$dh" -c "$W/$1/params.hjson" \
  --hwcfg $SN/target/snitch_cluster/cfg/snax_split_cluster.hjson --platformcfg $H/target/sw/shared/platform/generated/occamy.h \
  > "$3" 2>&1; echo "EXIT $?" >> "$3"
