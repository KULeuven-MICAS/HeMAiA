# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
# Application specific settings
MK_DIR   := $(realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST)))))
HOST_APP_TYPE = host_only
CHIP_TYPE     = multi_chip
WORKLOAD      = d2d_push_pull_bw
SRCS = src/d2d_push_pull_bw.c
BINGO_HOST ?= 1 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 0 # True to include the device binary

# The header, and the HBM image the testharness loads (target/sim/apps stages
# build/hbm/ into the simulation).
DATA_H       = $(MK_DIR)/data/data.h
HBM_MANIFEST = $(MK_DIR)/build/hbm/manifest.txt
PARTIAL_OUTPUTS += $(DATA_H) $(HBM_MANIFEST)

$(DATA_H): $(MK_DIR)/data/datagen.py
	python3 $< --header > $@
$(HBM_MANIFEST): $(MK_DIR)/data/datagen.py
	python3 $< --hbm-dir $(dir $@)
INCDIRS += $(MK_DIR)/data
include ../../../common.mk
