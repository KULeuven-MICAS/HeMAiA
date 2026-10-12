#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# gen_check.sh <workload dir under workloads/ffn> <out dir> <log>: the generator alone, its static checks included, with
# every output (headers, final_dfg.csv, the HBM image) written to <out dir>. Never into the workload directory: a running
# simulation maps that directory's build/hbm image, and rewriting it under the simulation crashes it (hemaia_hbm_read).
H=/users/micas/fkong/no_backup/HeMAiA
W=$H/target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/ffn
SN=$H/deps/snitch_cluster
export PATH=$H/target/sim/automation/sweep/ffn/tools/bin:$PATH
mkdir -p "$2"
cd "$W/$1" && python3 main_bingo.py --output_dir "$2" --data_h "$2/ffn_data.h" -c "$W/$1/params.hjson" \
  --hwcfg $SN/target/snitch_cluster/cfg/snax_split_cluster.hjson --platformcfg $H/target/sw/shared/platform/generated/occamy.h \
  --dsv2_dir $SN/target/snitch_cluster/sw/apps/dsv2 > "$3" 2>&1; echo "EXIT $?" >> "$3"
