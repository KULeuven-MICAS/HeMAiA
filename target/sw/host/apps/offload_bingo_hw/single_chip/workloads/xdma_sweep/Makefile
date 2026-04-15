# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# xDMA RTL cycle characterization workload — runs every config in the
# CONFIGS list (see main_bingo.py) in a single DFG.  After the sim, run
# `make bingo-vis-traces` and parse the JSON to map XDMA_RUN events back
# to configs (in order).
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = single_chip
WORKLOAD      = xdma_sweep
DEV_APP       = snax-bingo-offload
SRCS = ../../src/offload_bingo_hw.c
BINGO_HOST ?= 1
INCL_DEVICE_BINARY ?= 1

DATA_H = $(MK_DIR)/xdma_sweep_data.h
OFFLOAD_H = $(MK_DIR)/offload_bingo_hw.h
CONFIGS_JSON = $(MK_DIR)/configs.json
PLATFORM_H = $(MK_DIR)/../../../../../../../../target/sw/shared/platform/generated/occamy.h
RTL_CFG    = $(MK_DIR)/../../../../../../../../target/rtl/cfg/lru.hjson

$(DATA_H) $(OFFLOAD_H) $(CONFIGS_JSON): $(MK_DIR)/main_bingo.py
	python3 $(MK_DIR)/main_bingo.py \
		--output_dir $(MK_DIR) \
		--data_h $(DATA_H) \
		--configs_out $(CONFIGS_JSON) \
		--platformcfg $(PLATFORM_H) \
		--rtlcfg $(RTL_CFG)

###############
# Build settings
###############
INCDIRS += $(MK_DIR)
PARTIAL_OUTPUTS += $(DATA_H)
PARTIAL_OUTPUTS += $(OFFLOAD_H)

.PHONY: clean-data clean-offload clean

clean-data:
	rm -f $(DATA_H) $(CONFIGS_JSON)

clean-offload:
	rm -rf $(OFFLOAD_H)
	rm -rf *.png
	rm -rf *.csv
clean: clean-data clean-offload

gen-data: $(DATA_H)
gen-offload: $(OFFLOAD_H)
gen-configs: $(CONFIGS_JSON)

include ../../../../common.mk
