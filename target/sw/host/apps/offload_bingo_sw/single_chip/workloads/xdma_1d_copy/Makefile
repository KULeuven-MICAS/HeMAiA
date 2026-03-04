# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
MK_DIR   := $(abspath $(dir $(realpath $(lastword $(MAKEFILE_LIST)))))

HOST_APP_TYPE = offload_bingo_sw
CHIP_TYPE     = single_chip
WORKLOAD      = xdma_1d_copy
DEV_APP       = snax-bingo-offload

SRCS = ../../src/offload_bingo_sw.c

BINGO_HOST ?= 1 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 1 # True to include the device binary
###############
# Build settings
############### 
INCDIRS += $(MK_DIR)

include ../../../../common.mk
