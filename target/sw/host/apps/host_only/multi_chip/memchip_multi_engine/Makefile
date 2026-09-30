# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
# Application specific settings
MK_DIR   := $(realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST)))))
HOST_APP_TYPE = host_only
CHIP_TYPE     = multi_chip
WORKLOAD      = memchip_multi_engine
SRCS = src/memchip_multi_engine.c
BINGO_HOST ?= 0 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 0 # True to include the device binary

# The header, and each memory chip's images, which target/sim/apps stages into the
# simulation: build/mempool.bin (every memory chip without its own), build/mempool_chip_1_1.bin
# (memory chip (1,1)'s own) and build/hbm/ (one HBM image per chip).
DATA_H       = $(MK_DIR)/data/data.h
HBM_MANIFEST = $(MK_DIR)/build/hbm/manifest.txt
PARTIAL_OUTPUTS += $(DATA_H) $(HBM_MANIFEST)

$(DATA_H): $(MK_DIR)/data/datagen.py
	python3 $< --header > $@
$(HBM_MANIFEST): $(MK_DIR)/data/datagen.py
	python3 $< --build-dir $(MK_DIR)/build
INCDIRS += $(MK_DIR)/data
include ../../../common.mk
