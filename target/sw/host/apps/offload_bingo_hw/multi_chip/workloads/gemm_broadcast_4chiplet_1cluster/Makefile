# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

# Usage of absolute paths is required to externally include this Makefile
MK_DIR   := $(realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST)))))
HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = multi_chip
WORKLOAD      = gemm_broadcast_4chiplet_1cluster
DEV_APP       = snax-bingo-offload
SRCS = ../../src/offload_bingo_hw.c
BINGO_HOST ?= 1 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 1 # True to include the device binary


# DATA
DATA_CFG ?= $(MK_DIR)/params.hjson

DATA_H = $(MK_DIR)/gemm_data.h
OFFLOAD_H = $(MK_DIR)/offload_bingo_hw.h
BENDER   = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
PLATFORM_H = $(MK_DIR)/../../../../../../../../target/sw/shared/platform/generated/occamy.h

$(DATA_H) $(OFFLOAD_H): $(MK_DIR)/main_bingo.py $(MK_DIR)/gemm_multi_chiplet_datagen.py $(MK_DIR)/../../../../../../../../util/sim/snax_utils.py $(MK_DIR)/../../../../../../../../util/sim/sim_golden_models.py $(DATA_CFG)
	python3 $(MK_DIR)/main_bingo.py \
		--output_dir $(MK_DIR) \
		--data_h $(DATA_H) \
		-c $(DATA_CFG) \
		--hwcfg $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_versacore_to_cluster.hjson \
		--platformcfg $(PLATFORM_H)

###############
# Build settings
############### 
INCDIRS += $(MK_DIR)
# Add the DATA_H to partial outputs
PARTIAL_OUTPUTS += $(DATA_H)
# Add the OFFLOAD_H to partial outputs
PARTIAL_OUTPUTS += $(OFFLOAD_H)

.PHONY: clean-data clean

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
