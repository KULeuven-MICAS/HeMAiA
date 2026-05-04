# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# Test all FP32 RVV operator kernels that bingo dispatches to CVA6+Ara.

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HOST_APP_TYPE = host_only
CHIP_TYPE     = single_chip
WORKLOAD      = ara_test
SRCS = src/main.c
BINGO_HOST ?= 0
INCL_DEVICE_BINARY ?= 0

# Generated test data
OP_DATA_H = $(MK_DIR)/include/op_test_data.h
$(OP_DATA_H): $(MK_DIR)/include/gen_op_data.py
	python3 $< > $@
PARTIAL_OUTPUTS += $(OP_DATA_H)

# Include paths
INCDIRS += $(MK_DIR)/include
INCDIRS += $(MK_DIR)/../../../../../runtime

include ../../../common.mk
