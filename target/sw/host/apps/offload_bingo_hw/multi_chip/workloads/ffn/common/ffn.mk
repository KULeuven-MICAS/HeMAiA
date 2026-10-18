# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# The build of an ffn workload (workloads/ffn/<model>/<variant>/). Its Makefile sets MK_DIR and
# WORKLOAD, then includes this file:
#
#     MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
#     WORKLOAD  = ffn/dsv2_l0/all
#     include $(MK_DIR)../../common/ffn.mk
#
# WORKLOAD is the directory's path under workloads/, so the roots are found from it rather than
# by counting ../ levels. Built and run on hemaia_16c4m_8MBL3_2cluster.hjson with
# target/sim/automation/sweep/llm_blocks/ffn/run.sh; the weights come from that sweep's cache
# (sweep/llm_blocks/ffn/fetch_weights.py).

HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = multi_chip
DEV_APP       = snax-bingo-offload

APP_ROOT  := $(patsubst %/workloads/$(WORKLOAD)/,%,$(MK_DIR))
ifeq ($(APP_ROOT),$(MK_DIR))
    $(error WORKLOAD=$(WORKLOAD) is not the path of $(MK_DIR) under workloads/)
endif
REPO_ROOT := $(abspath $(APP_ROOT)/../../../../../..)
FFN_ROOT  := $(APP_ROOT)/workloads/ffn

SRCS = $(APP_ROOT)/src/offload_bingo_hw.c
BINGO_HOST ?= 1
INCL_DEVICE_BINARY ?= 1

DATA_CFG ?= $(MK_DIR)params.hjson
DATA_H = $(MK_DIR)ffn_data.h
OFFLOAD_H = $(MK_DIR)offload_bingo_hw.h
# The HBM image the testharness maps at time zero: target/sim/apps links build/hbm/ into the
# simulation (hw/hemaia/hemaia_mem_system/hbm/README.md).
HBM_MANIFEST = $(MK_DIR)build/hbm/manifest.txt
BENDER   = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
# snax_cluster's DeepSeek-V2-Lite package: the device-arithmetic models the golden uses
DSV2_DIR = $(SNITCH_ROOT)/target/snitch_cluster/sw/apps/dsv2
PLATFORM_H = $(REPO_ROOT)/target/sw/shared/platform/generated/occamy.h
WEIGHTS_MANIFEST = $(REPO_ROOT)/target/sim/automation/sweep/llm_blocks/ffn/cache/manifest.json

BINGO_GEN_DEPS = $(shell find $(REPO_ROOT)/target/sw/host/runtime/libbingo/mini_compiler -name '*.py') \
                 $(REPO_ROOT)/target/sw/host/runtime/libbingo/include/libbingo/device_kernel_args.h \
                 $(REPO_ROOT)/target/sw/host/runtime/libbingo/include/libbingo/host_kernel_args.h \
                 $(wildcard $(DSV2_DIR)/util/*.py) \
                 $(wildcard $(FFN_ROOT)/common/*.py) \
                 $(wildcard $(APP_ROOT)/workloads/common/*.py) \
                 $(WEIGHTS_MANIFEST)

# ONE run makes all three: a grouped target (&:, GNU make >= 4.3)
$(DATA_H) $(OFFLOAD_H) $(HBM_MANIFEST) &: $(MK_DIR)main_bingo.py $(DATA_CFG) $(PLATFORM_H) $(BINGO_GEN_DEPS)
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

# Checks print only failures, and each chip one "Checks: N PASS, M FAIL" line at exit.
CHECK_QUIET ?= 1
RISCV_CFLAGS += -DBINGO_CHECK_QUIET=$(CHECK_QUIET)

include $(APP_ROOT)/../../common.mk
