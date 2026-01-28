# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>
# Fanchen Kong <fanchen.kong@kuleuven.be>
# Yunhao Deng <yunhao.deng@kuleuven.be>

# Root directory (absolute path to this Makefile's directory)
export HOST_DIR = $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# Application categories (now organized by chip count)
HOST_ONLY_SUBDIRS = apps/host_only/multi_chip apps/host_only/single_chip
OFFLOAD_BINGO_HW_SUBDIRS = apps/offload_bingo_hw/multi_chip apps/offload_bingo_hw/single_chip
OFFLOAD_BINGO_SW_SUBDIRS = apps/offload_bingo_sw/multi_chip apps/offload_bingo_sw/single_chip
OFFLOAD_LEGACY_SUBDIRS = apps/offload_legacy/multi_chip apps/offload_legacy/single_chip

SUBDIRS = $(HOST_ONLY_SUBDIRS) $(OFFLOAD_BINGO_HW_SUBDIRS) $(OFFLOAD_BINGO_SW_SUBDIRS) $(OFFLOAD_LEGACY_SUBDIRS)

.DEFAULT_GOAL := all
TARGET ?= all

.PHONY: all libs $(SUBDIRS) host_only offload_bingo_hw offload_bingo_sw offload_legacy clean single_sw

all:
ifeq ($(TARGET),all)
	$(MAKE) libs
	$(MAKE) partial-build
	$(MAKE) finalize-build
else
	$(MAKE) $(TARGET)
endif

LIB_BINGO_DIR = runtime/libbingo
libs:
	$(MAKE) -C $(LIB_BINGO_DIR) gen-lib

# Category-specific targets
host_only: $(HOST_ONLY_SUBDIRS)
offload_bingo_hw: $(OFFLOAD_BINGO_HW_SUBDIRS)
offload_bingo_sw: $(OFFLOAD_BINGO_SW_SUBDIRS)
offload_legacy: $(OFFLOAD_LEGACY_SUBDIRS)

# Global clean
clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done

# Single sw compilation (by category and chip type for precise control)
single_sw:
	@if [ "$(HOST_APP_TYPE)" = "host_only" ]; then \
		DIR=apps/host_only/$(CHIP_TYPE)/$(WORKLOAD); \
	elif [ "$(HOST_APP_TYPE)" = "offload_legacy" ]; then \
		DIR=apps/offload_legacy/$(CHIP_TYPE); \
	else \
		DIR=apps/$(HOST_APP_TYPE)/$(CHIP_TYPE)/workloads/$(WORKLOAD); \
	fi; \
	if [ ! -d "$$DIR" ]; then echo "App Directory $$DIR not found"; exit 1; fi; \
	$(MAKE) -C $$DIR $(TARGET) DEV_APP=$(DEV_APP)

partial-build:
	@echo "Building partial host applications..."
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir TARGET=partial-build; \
	done

device-build:
	@echo "Building device applications..."
	$(MAKE) -C ../device all

finalize-build:
	@echo "Finalizing host applications with device binaries..."
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir TARGET=finalize-build; \
	done
