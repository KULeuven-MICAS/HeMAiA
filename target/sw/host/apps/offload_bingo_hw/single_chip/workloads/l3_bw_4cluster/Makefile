# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Main-memory bandwidth in three phases: read-only, write-only, and both at once.
# Data is embedded as a C array (no mem chip), so this needs no --systemcfg.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = single_chip
WORKLOAD      = l3_bw_4cluster
DEV_APP       = snax-bingo-offload
SRCS = ../../src/offload_bingo_hw.c
BINGO_HOST ?= 1
INCL_DEVICE_BINARY ?= 1

DATA_CFG ?= $(MK_DIR)/params.hjson
DATA_H = $(MK_DIR)/l3bw_data.h
OFFLOAD_H = $(MK_DIR)/offload_bingo_hw.h
BENDER   = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
PLATFORM_H = $(MK_DIR)/../../../../../../../../target/sw/shared/platform/generated/occamy.h

# The generated header is EMITTED BY the mini_compiler and must match the kernel ABI in
# device_kernel_args.h, so both are prerequisites: without them an ABI change leaves a
# workload whose main_bingo.py has not moved building against a stale header, and the run
# wedges mid-pipeline with no build error anywhere.
BINGO_GEN_DEPS = $(wildcard $(MK_DIR)/../../../../../../../../target/sw/host/runtime/libbingo/mini_compiler/*.py) \
                 $(MK_DIR)/../../../../../../../../target/sw/host/runtime/libbingo/include/libbingo/device_kernel_args.h

BINGO_STATIC_L1 ?= 0
ifeq ($(BINGO_STATIC_L1),1)
BINGO_STATIC_L1_FLAG = --static-l1
endif

$(DATA_H) $(OFFLOAD_H): $(MK_DIR)/main_bingo.py $(DATA_CFG) $(PLATFORM_H) $(BINGO_GEN_DEPS)
	python3 $(MK_DIR)/main_bingo.py \
		--output_dir $(MK_DIR) \
		--data_h $(DATA_H) \
		-c $(DATA_CFG) \
		--hwcfg $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_split_cluster.hjson \
		--platformcfg $(PLATFORM_H) $(BINGO_STATIC_L1_FLAG)

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

# offload_bingo_hw.c includes the generated workload header. Parent builds can target the
# final ELF directly, so ensure dependency generation creates it first.
$(DEP): $(OFFLOAD_H) $(DATA_H)
