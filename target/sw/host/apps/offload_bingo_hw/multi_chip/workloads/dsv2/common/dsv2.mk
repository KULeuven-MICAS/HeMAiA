# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# The build of a staged DeepSeek-V2-Lite workload (workloads/dsv2/<platform>/stage*/). A stage's
# Makefile sets MK_DIR and WORKLOAD, then includes this file:
#
#     MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
#     WORKLOAD  = dsv2/two_chiplet/stage4_latent_rope_wuk
#     include $(MK_DIR)../../common/dsv2.mk
#
# WORKLOAD is the directory's path under workloads/ -- what `make apps WORKLOAD=...` is given --
# so the app root, the repo root and everything else are found from it, never by counting ../
# levels (which is right for one depth only).
#
# Build and run on hemaia_twochiplet_16MBL3_4cluster.hjson, the only cfg whose memory chiplet
# has an HBM:
#     python3 target/sim/automation/sweep/dsv2/run_dsv2_sweep.py

HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = multi_chip
DEV_APP       = snax-bingo-offload

APP_ROOT  := $(patsubst %/workloads/$(WORKLOAD)/,%,$(MK_DIR))
ifeq ($(APP_ROOT),$(MK_DIR))
    $(error WORKLOAD=$(WORKLOAD) is not the path of $(MK_DIR) under workloads/)
endif
REPO_ROOT := $(abspath $(APP_ROOT)/../../../../../..)
DSV2_ROOT := $(APP_ROOT)/workloads/dsv2

SRCS = $(APP_ROOT)/src/offload_bingo_hw.c
BINGO_HOST ?= 1
INCL_DEVICE_BINARY ?= 1

# params.hjson is the stage's one-token run; `make ... DATA_CFG=<dir>/params_t4.hjson` builds
# its 4-token pass instead
DATA_CFG ?= $(MK_DIR)params.hjson
DATA_H = $(MK_DIR)dsv2_data.h
OFFLOAD_H = $(MK_DIR)offload_bingo_hw.h
# The HBM image the testharness maps at time zero: target/sim/apps links build/hbm/ into
# the simulation (hw/hemaia/hemaia_mem_system/hbm/README.md).
HBM_MANIFEST = $(MK_DIR)build/hbm/manifest.txt
BENDER   = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
# The golden: snax_cluster's DeepSeek-V2-Lite package, the one its snax-dsv2-* apps use.
DSV2_DIR = $(SNITCH_ROOT)/target/snitch_cluster/sw/apps/dsv2
PLATFORM_H = $(REPO_ROOT)/target/sw/shared/platform/generated/occamy.h

# The generated header must match the kernel ABI, so the whole mini_compiler tree (the staging
# helper included), both args headers and the golden package are prerequisites -- and so is every
# stage file, since stage N builds on stages 1..N-1, and the shared driver and datagen.
BINGO_GEN_DEPS = $(shell find $(REPO_ROOT)/target/sw/host/runtime/libbingo/mini_compiler -name '*.py') \
                 $(REPO_ROOT)/target/sw/host/runtime/libbingo/include/libbingo/device_kernel_args.h \
                 $(REPO_ROOT)/target/sw/host/runtime/libbingo/include/libbingo/host_kernel_args.h \
                 $(wildcard $(DSV2_DIR)/util/*.py) \
                 $(wildcard $(DSV2_ROOT)/common/*.py) \
                 $(wildcard $(dir $(MK_DIR:/=))*/main_bingo.py)

$(DATA_H) $(OFFLOAD_H) $(HBM_MANIFEST): $(MK_DIR)main_bingo.py $(DATA_CFG) $(PLATFORM_H) $(BINGO_GEN_DEPS)
	python3 $(MK_DIR)main_bingo.py \
		--output_dir $(MK_DIR) \
		--data_h $(DATA_H) \
		-c $(DATA_CFG) \
		--hwcfg $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_split_cluster.hjson \
		--platformcfg $(PLATFORM_H) \
		--dsv2_dir $(DSV2_DIR)

INCDIRS += $(MK_DIR)
PARTIAL_OUTPUTS += $(DATA_H)
PARTIAL_OUTPUTS += $(OFFLOAD_H)
PARTIAL_OUTPUTS += $(HBM_MANIFEST)

.PHONY: clean-data clean-offload clean
clean-data:
	rm -f $(DATA_H)
	rm -rf $(MK_DIR)build/hbm
clean-offload:
	rm -rf $(OFFLOAD_H) $(MK_DIR)*.png $(MK_DIR)*.csv $(MK_DIR)block_dfg
clean: clean-data clean-offload

include $(APP_ROOT)/../../common.mk
