# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
# Application specific settings
MK_DIR   := $(realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST)))))
HOST_APP_TYPE = host_only
CHIP_TYPE     = multi_chip
WORKLOAD      = system_xdma_copy
SRCS = src/system_xdma_copy.c
BINGO_HOST ?= 1 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 0 # True to include the device binary

DATA_H = $(MK_DIR)/data/data.h
PARTIAL_OUTPUTS += $(DATA_H)

$(DATA_H): $(MK_DIR)/data/datagen.py $(MK_DIR)/data/params.hjson
	python3 $< -c data/params.hjson > $@
INCDIRS += $(MK_DIR)/data
include ../../../common.mk
