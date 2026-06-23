# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

# Usage of absolute paths is required to externally include this Makefile
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR))

DATA_CFG ?= $(DATA_DIR)/params.hjson

DATA_H = $(DATA_DIR)/data.h
MEMPOOL_BIN = $(DATA_DIR)/mempool.bin

$(DATA_H): $(DATA_DIR)/datagen.py $(DATA_CFG)
	python3 $< -c $(DATA_CFG) --out-dir $(DATA_DIR) > $@

$(MEMPOOL_BIN): $(DATA_H)

.PHONY: mempool-bin
mempool-bin: $(MEMPOOL_BIN)

.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H) $(MEMPOOL_BIN)

clean: clean-data
