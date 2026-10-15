#!/bin/bash
# gen_one.sh <workload dir under workloads/ffn> <log>: the generator alone
H=/users/micas/fkong/no_backup/HeMAiA
W=$H/target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/ffn
SN=$H/deps/snitch_cluster
export PATH=$H/target/sim/automation/sweep/llm_blocks/ffn/tools/bin:$PATH
cd $W/$1 && python3 main_bingo.py --output_dir $W/$1 --data_h $W/$1/ffn_data.h -c $W/$1/params.hjson \
  --hwcfg $SN/target/snitch_cluster/cfg/snax_split_cluster.hjson --platformcfg $H/target/sw/shared/platform/generated/occamy.h \
  --dsv2_dir $SN/target/snitch_cluster/sw/apps/dsv2 > $2 2>&1; echo "EXIT $?" >> $2
