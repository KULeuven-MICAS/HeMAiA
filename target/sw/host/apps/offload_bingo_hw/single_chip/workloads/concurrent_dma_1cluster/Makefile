# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>
# Fanchen Kong <fanchen.kong@kuleuven.be>

# Usage of absolute paths is required to externally include this Makefile
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = single_chip
WORKLOAD      = concurrent_dma_1cluster
DEV_APP       = snax-bingo-offload
SRCS = ../../src/offload_bingo_hw.c
BINGO_HOST ?= 1 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 1 # True to include the device binary

# DATA
DATA_CFG ?= $(MK_DIR)/params.hjson
DATA_H = $(MK_DIR)/dma_data.h
OFFLOAD_H = $(MK_DIR)/offload_bingo_hw.h
BENDER   = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
PLATFORM_H = $(MK_DIR)/../../../../../../../../target/sw/shared/platform/generated/occamy.h

# The generated header is EMITTED BY the mini_compiler and must match the kernel ABI in
# device_kernel_args.h. Neither used to be a prerequisite here, so an ABI change silently
# left every workload whose main_bingo.py had not changed building against a STALE header:
# the struct gained a field, the generated init code did not set it, and the run wedged
# mid-pipeline with no build error anywhere. Listing them is what makes that impossible.
BINGO_GEN_DEPS = $(wildcard $(MK_DIR)/../../../../../../../../target/sw/host/runtime/libbingo/mini_compiler/*.py) \
                 $(MK_DIR)/../../../../../../../../target/sw/host/runtime/libbingo/include/libbingo/device_kernel_args.h

$(DATA_H) $(OFFLOAD_H): $(MK_DIR)/main_bingo.py $(DATA_CFG) $(PLATFORM_H) $(BINGO_GEN_DEPS)
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
