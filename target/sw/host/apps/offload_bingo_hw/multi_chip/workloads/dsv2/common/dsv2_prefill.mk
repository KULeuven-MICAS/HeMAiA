# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# The build of a whole-layer workload (workloads/dsv2/layer{0,1}/{prefill,decode}/<app>/). Its Makefile sets
# MK_DIR and WORKLOAD, then includes this file:
#
#     MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
#     WORKLOAD  = dsv2/layer1/prefill/full
#     include $(MK_DIR)../../../common/dsv2_prefill.mk
#
# WORKLOAD is the directory's path under workloads/, so the roots are found from it rather than by counting
# ../ levels. Built and run on hemaia_16c4m_8MBL3_2cluster.hjson; the drivers are dsv2_layer_prefill_mc.py (prefill) and
# dsv2_decode_mc.py (decode), which build the attention (workloads/fa) and the MoE (workloads/moe) or FFN
# (workloads/ffn) blocks into one graph. The cache is
# sweep/llm_blocks/fa/real_kv_cache.py's, the weights sweep/common/weights' cache (the checkpoint's layer 1).

HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = multi_chip
DEV_APP       = snax-bingo-offload

APP_ROOT  := $(patsubst %/workloads/$(WORKLOAD)/,%,$(MK_DIR))
ifeq ($(APP_ROOT),$(MK_DIR))
    $(error WORKLOAD=$(WORKLOAD) is not the path of $(MK_DIR) under workloads/)
endif
REPO_ROOT := $(abspath $(APP_ROOT)/../../../../../..)
WL_ROOT   := $(APP_ROOT)/workloads

SRCS = $(APP_ROOT)/src/offload_bingo_hw.c
BINGO_HOST ?= 1
INCL_DEVICE_BINARY ?= 1

DATA_CFG ?= $(MK_DIR)params.hjson
DATA_H = $(MK_DIR)dsv2_data.h
OFFLOAD_H = $(MK_DIR)offload_bingo_hw.h
# The HBM image the testharness maps at time zero: target/sim/apps links build/hbm/ into the
# simulation (hw/hemaia/hemaia_mem_system/hbm/README.md).
HBM_MANIFEST = $(MK_DIR)build/hbm/manifest.txt
BENDER   = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
PLATFORM_H = $(REPO_ROOT)/target/sw/shared/platform/generated/occamy.h
WEIGHTS_MANIFEST = $(REPO_ROOT)/target/sim/automation/sweep/common/weights/cache/manifest.json
KV_MANIFEST = $(REPO_ROOT)/target/sim/automation/sweep/llm_blocks/fa/cache/manifest.json

BINGO_GEN_DEPS = $(shell find $(REPO_ROOT)/target/sw/host/runtime/libbingo/mini_compiler -name '*.py') \
                 $(REPO_ROOT)/target/sw/host/runtime/libbingo/include/libbingo/device_kernel_args.h \
                 $(REPO_ROOT)/target/sw/host/runtime/libbingo/include/libbingo/host_kernel_args.h \
                 $(wildcard $(WL_ROOT)/ffn/common/*.py) $(wildcard $(WL_ROOT)/fa/common/*.py) \
                 $(wildcard $(WL_ROOT)/moe/common/*.py) $(wildcard $(WL_ROOT)/dsv2/common/*.py) \
                 $(wildcard $(REPO_ROOT)/util/sim/multichip/*.py $(REPO_ROOT)/util/sim/llm/*.py) \
                 $(addprefix $(REPO_ROOT)/util/sim/common/,driver.py fp.py hwmodel.py layout.py pack.py) \
                 $(REPO_ROOT)/util/sim/simd/simd.py $(REPO_ROOT)/util/sim/llm/config.json $(REPO_ROOT)/util/sim/gemm/gemv_utils.py \
                 $(WEIGHTS_MANIFEST) $(KV_MANIFEST)

# ONE run makes all three: a grouped target (&:, GNU make >= 4.3)
$(DATA_H) $(OFFLOAD_H) $(HBM_MANIFEST) &: $(MK_DIR)main_bingo.py $(DATA_CFG) $(PLATFORM_H) $(BINGO_GEN_DEPS)
	python3 $(MK_DIR)main_bingo.py \
		--output_dir $(MK_DIR) \
		--data_h $(DATA_H) \
		-c $(DATA_CFG) \
		--hwcfg $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_split_cluster.hjson \
		--platformcfg $(PLATFORM_H)

INCDIRS += $(MK_DIR)
PARTIAL_OUTPUTS += $(DATA_H)
PARTIAL_OUTPUTS += $(OFFLOAD_H)
PARTIAL_OUTPUTS += $(HBM_MANIFEST)

.PHONY: clean-data clean-offload clean
clean-data:
	rm -f $(DATA_H)
	rm -rf $(MK_DIR)build/hbm
clean-offload:
	rm -rf $(OFFLOAD_H) $(MK_DIR)per_chip_ids $(MK_DIR)*.png $(MK_DIR)*.csv $(MK_DIR)block_dfg
clean: clean-data clean-offload

# Checks print only failures, and each chip one "Checks: N PASS, M FAIL" line at exit.
CHECK_QUIET ?= 1
RISCV_CFLAGS += -DBINGO_CHECK_QUIET=$(CHECK_QUIET)
# A failing check prints its FAIL line only, so every verdict reaches the UART before the run ends.
CHECK_ERR_PRINT ?= 0
RISCV_CFLAGS += -DBINGO_CHECK_ERR_PRINT_MAX=$(CHECK_ERR_PRINT)u
# A failing run waits this many host cycles at exit, so the UART drains every verdict.
CHECK_FAIL_DRAIN ?= 600000
RISCV_CFLAGS += -DBINGO_CHECK_FAIL_DRAIN_CYCLES=$(CHECK_FAIL_DRAIN)
# 1: the host also re-derives every args field the generator initialised statically and aborts on a
# difference (bingo_api.h BINGO_CHECK_STATIC_ARGS)
CHECK_STATIC_ARGS ?= 0
RISCV_CFLAGS += -DBINGO_CHECK_STATIC_ARGS=$(CHECK_STATIC_ARGS)
# 1: the host weight prefetchers write only the engine registers that change and keep each ring's next run
# once found (host_kernel_lib.h WEIGHT_PREFETCH_LEAN), as the MoE block's best runs
PREFETCH_LEAN ?= 1
RISCV_CFLAGS += -DWEIGHT_PREFETCH_LEAN=$(PREFETCH_LEAN)

include $(APP_ROOT)/../../common.mk
