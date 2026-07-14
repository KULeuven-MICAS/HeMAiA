# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Placeholder 2-cluster workload for the hemaia_tapeout_2c_simd CI suite.

# Usage of absolute paths is required to externally include this Makefile
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = single_chip
WORKLOAD      = dummy_simd_2cluster
DEV_APP       = snax-bingo-offload
SRCS = ../../src/offload_bingo_hw.c
BINGO_HOST ?= 1 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 1 # True to include the device binary

# DATA
DATA_CFG ?= $(MK_DIR)/params.hjson
DATA_H = $(MK_DIR)/dummy_data.h
OFFLOAD_H = $(MK_DIR)/offload_bingo_hw.h
BENDER   = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
PLATFORM_H = $(MK_DIR)/../../../../../../../../target/sw/shared/platform/generated/occamy.h

# main_bingo.py reads the cluster count out of PLATFORM_H, so it must re-run when
# the active RTL cfg changes -- otherwise the first build after switching cfg
# generates the DFG against a stale occamy.h and guard_cluster_count() rejects it.
$(DATA_H) $(OFFLOAD_H): $(MK_DIR)/main_bingo.py $(DATA_CFG) $(PLATFORM_H)
	python3 $(MK_DIR)/main_bingo.py \
		--output_dir $(MK_DIR) \
		--data_h $(DATA_H) \
		-c $(DATA_CFG) \
		--hwcfg $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_versacore_to_256KB_simd_cluster.hjson \
		--platformcfg $(PLATFORM_H)

###############
# Build settings
###############
INCDIRS += $(MK_DIR)
PARTIAL_OUTPUTS += $(DATA_H)
PARTIAL_OUTPUTS += $(OFFLOAD_H)

.PHONY: clean-data clean-offload clean

clean-data:
	rm -f $(DATA_H)

clean-offload:
	rm -rf $(OFFLOAD_H)
	rm -rf *.png
	rm -rf *.csv

clean: clean-data clean-offload

gen-data: $(DATA_H)

gen-offload: $(OFFLOAD_H)

include ../../../../common.mk
