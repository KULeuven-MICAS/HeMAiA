# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Single-kernel ara cycle sweep: max.

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HOST_APP_TYPE = host_only
CHIP_TYPE     = single_chip
WORKLOAD      = ara_max
SRCS = src/main.c
BINGO_HOST ?= 0
INCL_DEVICE_BINARY ?= 0

# Ara config
NR_LANES ?= 2
RISCV_CFLAGS += -DNR_LANES=$(NR_LANES)

# Generated inputs + golden (shared generator lives in util/sim/ara_lib.py).
GEN_OP_DATA = $(abspath $(MK_DIR)../../../../../../../util/sim/ara_lib.py)
OP_DATA_H = $(MK_DIR)/include/op_test_data.h
$(OP_DATA_H): $(GEN_OP_DATA)
	@mkdir -p $(@D)
	python3 $< $(WORKLOAD:ara_%=%) > $@
PARTIAL_OUTPUTS += $(OP_DATA_H)

INCDIRS += $(MK_DIR)/include
INCDIRS += $(MK_DIR)/../../../../../runtime

include ../../../common.mk
