# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

MK_DIR   := $(realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST)))))
HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = multi_chip
WORKLOAD      = gemm_msplit_4chiplet_2cluster_basic
DEV_APP       = snax-bingo-offload
SRCS = ../../src/offload_bingo_hw.c
BINGO_HOST ?= 1
INCL_DEVICE_BINARY ?= 1

DATA_CFG ?= $(MK_DIR)/params.hjson

DATA_H = $(MK_DIR)/gemm_data.h
OFFLOAD_H = $(MK_DIR)/offload_bingo_hw.h
BENDER   = bender
PYTHON ?= python3
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
PLATFORM_H = $(MK_DIR)/../../../../../../../../target/sw/shared/platform/generated/occamy.h

# Resolve hardware geometry from the active SoC configuration.
CLUSTER_CFG_CONSUMER := $(abspath $(lastword $(MAKEFILE_LIST)))
include $(abspath $(dir $(CLUSTER_CFG_CONSUMER))/../../../../../../../../util/cluster_cfg.mk)
.SECONDEXPANSION:

$(DATA_H) $(OFFLOAD_H) &: $(MK_DIR)/main_bingo.py $(MK_DIR)/gemm_multi_chiplet_datagen.py $(MK_DIR)/../../../../../../../../util/sim/gemm/gemm_sim_utils.py $(MK_DIR)/../../../../../../../../util/sim/common/sim_golden_models.py $(DATA_CFG) $(PLATFORM_H) $(CLUSTER_CFG_CONSUMER) $$(CLUSTER_CFG_DEPS)
	$(PYTHON) $(MK_DIR)/main_bingo.py \
		--output_dir $(MK_DIR) \
		--data_h $(DATA_H) \
		-c $(DATA_CFG) \
		--hwcfg $(GEMM_HWCFG) \
		--platformcfg $(PLATFORM_H)

INCDIRS += $(MK_DIR)
PARTIAL_OUTPUTS += $(DATA_H)
PARTIAL_OUTPUTS += $(OFFLOAD_H)

.PHONY: clean-data clean-offload clean gen-data gen-offload

clean-data:
	rm -f $(DATA_H) $(CLUSTER_CFG_STAMP)

clean-offload:
	rm -rf $(OFFLOAD_H)
	rm -rf *.png
	rm -rf *.csv

clean: clean-data clean-offload

gen-data: $(DATA_H)

gen-offload: $(OFFLOAD_H)

include ../../../../common.mk
