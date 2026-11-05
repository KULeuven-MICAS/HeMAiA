# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP = gemm_fpga
SRCS = src/gemm_fpga.c
INCDIRS += data
DATA_CFG ?= $(abspath data/params.hjson)
DATA_H = $(abspath data/data.h)

include ../../common.mk

# Use the same cluster configuration as the platform being built. An explicit
# FPGA_HWCFG is useful when generating data without rebuilding the platform.
FPGA_PLATFORM_CFG = $(if $(CFG),$(CFG),$(ROOT)/target/rtl/cfg/lru.hjson)
FPGA_CLUSTER = $(shell python3 -c 'import hjson,sys; print(hjson.load(open(sys.argv[1]))["clusters"][0])' $(FPGA_PLATFORM_CFG))
FPGA_HWCFG ?= $(SNITCH_ROOT)/target/snitch_cluster/cfg/$(FPGA_CLUSTER).hjson

$(DATA_H): data/datagen.py $(DATA_CFG) $(FPGA_HWCFG) $(FPGA_PLATFORM_CFG)
	python3 data/datagen.py --swcfg $(DATA_CFG) --hwcfg $(FPGA_HWCFG) --output $@

# Dependencies are per host app in common.mk. Also protect direct ELF builds.
$(DEP_FILES) $(foreach host,$(HOST_APPS_FOR_DEV),$(ELF_$(host))): $(DATA_H)

.PHONY: gen-data clean-data
gen-data: $(DATA_H)
clean-data:
	rm -f $(DATA_H)
clean: clean-data
