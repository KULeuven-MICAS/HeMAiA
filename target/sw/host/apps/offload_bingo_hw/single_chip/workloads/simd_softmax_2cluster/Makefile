# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# 2-cluster SIMD variant of the per-op xDMA sweep workload for "softmax".
# See util/sim/xdma/xdma_ops_lib.md and ../simd_softmax_1cluster.

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = single_chip
WORKLOAD      = simd_softmax_2cluster
DEV_APP       = snax-bingo-offload
SRCS = ../../src/offload_bingo_hw.c
BINGO_HOST ?= 1
INCL_DEVICE_BINARY ?= 1

DATA_CFG ?= $(MK_DIR)/params.hjson
DATA_H = $(MK_DIR)/softmax_data.h
OFFLOAD_H = $(MK_DIR)/offload_bingo_hw.h
CONFIGS_JSON = $(MK_DIR)/configs.json
BENDER       = bender
SNITCH_ROOT  = $(shell $(BENDER) path snitch_cluster)
PLATFORM_H = $(MK_DIR)/../../../../../../../../target/sw/shared/platform/generated/occamy.h

# main_bingo.py reads the cluster count out of PLATFORM_H, so it must re-run when
# the active RTL cfg changes -- otherwise the first build after switching cfg
# generates the DFG against a stale occamy.h and guard_cluster_count() rejects it.
# The generated header is EMITTED BY the mini_compiler and must match the kernel ABI in
# device_kernel_args.h. Neither used to be a prerequisite here, so an ABI change silently
# left every workload whose main_bingo.py had not changed building against a STALE header:
# the struct gained a field, the generated init code did not set it, and the run wedged
# mid-pipeline with no build error anywhere. Listing them is what makes that impossible.
# RECURSIVELY, which it was not: the mini_compiler is a package tree (graph/ kernels/ libs/
# passes/ platform/) and a flat $(wildcard .../*.py) matches only the one module left at the
# top level. This tracked 1 file out of 52 -- it had stopped being a guard at all.
BINGO_GEN_DEPS = $(shell find $(MK_DIR)/../../../../../../../../target/sw/host/runtime/libbingo/mini_compiler -name '*.py') \
                 $(MK_DIR)/../../../../../../../../target/sw/host/runtime/libbingo/include/libbingo/device_kernel_args.h

$(DATA_H) $(OFFLOAD_H) $(CONFIGS_JSON): $(MK_DIR)/main_bingo.py $(DATA_CFG) $(PLATFORM_H) $(BINGO_GEN_DEPS)
	python3 $(MK_DIR)/main_bingo.py \
		--output_dir $(MK_DIR) \
		--data_h $(DATA_H) \
		-c $(DATA_CFG) \
		--hwcfg $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_versacore_to_256KB_simd_cluster.hjson \
		--platformcfg $(PLATFORM_H) \
		--configs_out $(CONFIGS_JSON)

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
	rm -rf *.png *.csv

clean: clean-data clean-offload

gen-data: $(DATA_H)
gen-offload: $(OFFLOAD_H)
gen-configs: $(CONFIGS_JSON)

include ../../../../common.mk
