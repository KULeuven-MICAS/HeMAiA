# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

# Usage of absolute paths is required to externally include this Makefile
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR))

DATA_CFG ?= $(DATA_DIR)/params.hjson

DATA_H = $(DATA_DIR)/gemm_data.h
OFFLOAD_H = $(DATA_DIR)/offload_hw_bingo.h	
BINGO_META_H = $(DATA_DIR)/bingo_workload.h

BENDER   = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)

$(DATA_H): $(DATA_DIR)/gemm_datagen.py $(DATA_CFG) 
	$< -c $(DATA_CFG) --hwcfg $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_versacore_to_cluster.hjson> $@

$(OFFLOAD_H): $(DATA_DIR)/main_bingo.py
	python3 $< --output_dir $(DATA_DIR)


.PHONY: clean-data clean

clean-data:
	rm -rf $(DATA_H)
clean-offload:
	rm -rf $(OFFLOAD_H)
	rm -rf $(BINGO_META_H)
	rm -rf *.png
	rm -rf *.csv

clean: clean-data clean-offload

gen-data: $(DATA_H)

gen-offload: $(OFFLOAD_H)