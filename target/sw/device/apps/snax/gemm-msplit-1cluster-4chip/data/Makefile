# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# Usage of absolute paths is required to externally include this Makefile
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR))
IS_CLEAN_GOAL := $(filter clean clean-%,$(MAKECMDGOALS))

DATA_CFG ?= $(DATA_DIR)/params.hjson

DATA_H = $(DATA_DIR)/data.h
# The datagen writes the memory-chip image (mempool.bin) as a side effect of
# producing data.h, so mempool.bin only needs an ordering dependency on data.h.
MEMPOOL_BIN = $(DATA_DIR)/mempool.bin

ifeq ($(IS_CLEAN_GOAL),)
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
else
SNITCH_ROOT =
endif

$(DATA_H): $(DATA_DIR)/datagen.py $(DATA_CFG)
	python3 $< --swcfg $(DATA_CFG) --hwcfg $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_versacore_to_cluster.hjson --out-dir $(DATA_DIR) > $@

$(MEMPOOL_BIN): $(DATA_H)

.PHONY: mempool-bin
mempool-bin: $(MEMPOOL_BIN)

.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H) $(MEMPOOL_BIN)

clean: clean-data
