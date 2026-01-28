# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
# Application specific settings
HOST_APP_TYPE = host_only
CHIP_TYPE     = multi_chip
WORKLOAD      = system_xdma_copy
SRCS = src/system_xdma_copy.c
BINGO_HOST ?= 1 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 0 # True to include the device binary

PARTIAL_OUTPUTS += data/data.h

data/data.h: data/datagen.py data/params.hjson
	python3 $< -c data/params.hjson > $@

include ../../../common.mk
