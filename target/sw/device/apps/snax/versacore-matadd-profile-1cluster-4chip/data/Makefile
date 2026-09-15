# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

# Usage of absolute paths is required to externally include this Makefile
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR))
IS_CLEAN_GOAL := $(filter clean clean-%,$(MAKECMDGOALS))

DATA_CFG ?= $(DATA_DIR)/params.hjson

DATA_H = $(DATA_DIR)/data.h
BENDER ?= bender
ifeq ($(IS_CLEAN_GOAL),)
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
else
SNITCH_ROOT =
endif

# Resolve hardware geometry from the active SoC configuration.
CLUSTER_CFG_CONSUMER := $(abspath $(lastword $(MAKEFILE_LIST)))
include $(abspath $(dir $(CLUSTER_CFG_CONSUMER))/../../../../../../../util/cluster_cfg.mk)
.SECONDEXPANSION:

$(DATA_H): $(DATA_DIR)/datagen.py $(DATA_CFG) $(CLUSTER_CFG_CONSUMER) $$(CLUSTER_CFG_DEPS)
	$< --swcfg $(DATA_CFG) --hwcfg $(GEMM_HWCFG) > $@

.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H) $(CLUSTER_CFG_STAMP)

clean: clean-data
