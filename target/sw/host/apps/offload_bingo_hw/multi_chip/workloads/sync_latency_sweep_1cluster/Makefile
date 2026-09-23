# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# ARM D -- cross-chip synchronization latency of the BINGO hardware manager.
# No datagen: this workload touches no data.

MK_DIR   := $(realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST)))))
HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = multi_chip
WORKLOAD      = sync_latency_sweep_1cluster
DEV_APP       = snax-bingo-offload
SRCS = ../../src/offload_bingo_hw.c
BINGO_HOST ?= 1
INCL_DEVICE_BINARY ?= 1

DATA_CFG ?= $(MK_DIR)/params.hjson
OFFLOAD_H = $(MK_DIR)/offload_bingo_hw.h
# Keeping the $(PLATFORM_H) prerequisite is not optional: without it the first build
# after a cfg switch generates the DFG against a stale occamy.h.
PLATFORM_H = $(MK_DIR)/../../../../../../../../target/sw/shared/platform/generated/occamy.h

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

$(OFFLOAD_H): $(MK_DIR)/main_bingo.py $(DATA_CFG) $(PLATFORM_H) $(BINGO_GEN_DEPS)
	python3 $(MK_DIR)/main_bingo.py \
		--output_dir $(MK_DIR) \
		-c $(DATA_CFG) \
		--platformcfg $(PLATFORM_H)

INCDIRS += $(MK_DIR)
PARTIAL_OUTPUTS += $(OFFLOAD_H)

.PHONY: clean-offload clean gen-offload
clean-offload:
	rm -rf $(OFFLOAD_H) $(MK_DIR)/sync_phases.csv

clean: clean-offload

gen-offload: $(OFFLOAD_H)

include ../../../../common.mk
