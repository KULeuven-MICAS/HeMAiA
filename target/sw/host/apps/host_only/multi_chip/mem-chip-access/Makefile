# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

MK_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

HOST_APP_TYPE = host_only
CHIP_TYPE     = multi_chip
WORKLOAD      = mem-chip-access
SRCS          = src/mem-chip-access.c
BINGO_HOST ?= 1
INCL_DEVICE_BINARY ?= 0

BUILDDIR ?= $(MK_DIR)build
DATA_H = $(MK_DIR)/data/data.h
MEMPOOL_BIN = $(BUILDDIR)/mempool.bin
PARTIAL_OUTPUTS += $(DATA_H) $(MEMPOOL_BIN)

$(DATA_H): $(MK_DIR)/data/datagen.py $(MK_DIR)/data/params.hjson
	@mkdir -p $(BUILDDIR)
	python3 $< -c $(MK_DIR)/data/params.hjson --out-dir $(BUILDDIR) > $(DATA_H)

$(MEMPOOL_BIN): $(MK_DIR)/data/datagen.py $(MK_DIR)/data/params.hjson
	@mkdir -p $(BUILDDIR)
	python3 $< -c $(MK_DIR)/data/params.hjson --out-dir $(BUILDDIR) > $(DATA_H)

INCDIRS += $(MK_DIR)/data
include ../../../common.mk
