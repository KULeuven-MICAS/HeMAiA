# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Per-precision GEMM cycle-sweep workload (i8i4_i32). Runs every (M,K,N,shape) in
# gemm_psweep_lib.CONFIGS via the gemm_i8i4_i32 wrapper kernel; cycles are read
# from the BINGO_TRACE_GEMM_FULL_RUN_* markers by sweep/gemm/gather_gemm_luts.py.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = single_chip
WORKLOAD      = gemm_psweep_i8i4_i32_1cluster
DEV_APP       = snax-bingo-offload
SRCS = ../../src/offload_bingo_hw.c
BINGO_HOST ?= 1
INCL_DEVICE_BINARY ?= 1

L1_SIZE_KB ?= 500

DATA_CFG ?= $(MK_DIR)/params.hjson
DATA_H = $(MK_DIR)/gemm_data.h
OFFLOAD_H = $(MK_DIR)/offload_bingo_hw.h
CONFIGS_JSON = $(MK_DIR)/configs.json
BENDER   = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
PLATFORM_H = $(MK_DIR)/../../../../../../../../target/sw/shared/platform/generated/occamy.h

$(DATA_H) $(OFFLOAD_H) $(CONFIGS_JSON): $(MK_DIR)/main_bingo.py $(DATA_CFG) $(MK_DIR)/../../../../../../../../util/sim/gemm/gemm_psweep_lib.py $(MK_DIR)/../../../../../../../../util/sim/gemm/gemm_sim_utils.py $(PLATFORM_H)
	python3 $(MK_DIR)/main_bingo.py \
		--output_dir $(MK_DIR) \
		--data_h $(DATA_H) \
		-c $(DATA_CFG) \
		--hwcfg $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_versacore_to_cluster.hjson \
		--configs_out $(CONFIGS_JSON) \
		--l1_size_kb $(L1_SIZE_KB) \
		--platformcfg $(PLATFORM_H)

INCDIRS += $(MK_DIR)
PARTIAL_OUTPUTS += $(DATA_H)
PARTIAL_OUTPUTS += $(OFFLOAD_H)

.PHONY: clean-data clean-offload clean
clean-data:
	rm -f $(DATA_H) $(CONFIGS_JSON)
clean-offload:
	rm -rf $(OFFLOAD_H) *.png *.csv
clean: clean-data clean-offload

gen-data: $(DATA_H)
gen-offload: $(OFFLOAD_H)
gen-configs: $(CONFIGS_JSON)

include ../../../../common.mk
