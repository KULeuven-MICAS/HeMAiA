# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Parameterised M/K/N-split GEMM, multi-chip.
#
# CHIP_TYPE must agree with both this directory and the data config's num_chips;
# a mismatch surfaces as "workload not found" or as wrong-hardware errors later.
#
# The platform is not fixed here: SYSTEM_CFG follows CFG when the flow sets one,
# and the data config selects a matching worker grid, so one workload covers
# every rung from 4 chiplets to 64 workers. params/README.md lists the validated
# pairings.

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = multi_chip
WORKLOAD      = gemm_parametric
DEV_APP       = snax-bingo-offload
SRCS = ../../src/offload_bingo_hw.c
BINGO_HOST ?= 1
INCL_DEVICE_BINARY ?= 1

# HOST_DATA_CFG lets a CI task pick one of params/, so the same workload can
# appear several times in a task list at different shapes. Deliberately not
# called DATA_CFG: that name is a repo-wide convention the device applications'
# data Makefiles use too, and a command-line DATA_CFG= would propagate into them
# through recursive make.
DATA_CFG ?= $(if $(HOST_DATA_CFG),$(HOST_DATA_CFG),$(MK_DIR)/params.hjson)
DATA_H = $(MK_DIR)/gemm_data.h
OFFLOAD_H = $(MK_DIR)/offload_bingo_hw.h
# The memory-chiplet coordinate and the real spm_wide size are read from here.
SYSTEM_CFG ?= $(if $(CFG),$(CFG),$(MK_DIR)/../../../../../../../../target/rtl/cfg/hemaia_tapeout_1c.hjson)
BENDER   = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
PLATFORM_H = $(MK_DIR)/../../../../../../../../target/sw/shared/platform/generated/occamy.h

$(DATA_H) $(OFFLOAD_H): $(MK_DIR)/gemm_parametric.py $(MK_DIR)/gemm_datagen.py $(DATA_CFG) $(SYSTEM_CFG) $(PLATFORM_H)
	python3 $(MK_DIR)/gemm_parametric.py \
		--output_dir $(MK_DIR) \
		--data_h $(DATA_H) \
		-c $(DATA_CFG) \
		--hwcfg $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_versacore_to_cluster.hjson \
		--systemcfg $(SYSTEM_CFG) \
		--platformcfg $(PLATFORM_H)

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
