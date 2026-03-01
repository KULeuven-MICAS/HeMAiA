# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# Application specific settings
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HOST_APP_TYPE = host_only
CHIP_TYPE     = single_chip
WORKLOAD      = ara_test
SRCS = src/main.c
BINGO_HOST ?= 0 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 0 # True to include the device binary

# Defines
NR_LANES ?= 2
ARA_DEFINES += -DNR_LANES=$(NR_LANES)
RISCV_CFLAGS += $(ARA_DEFINES)


# Data generation
DATA_H = $(MK_DIR)/include/data.h
$(MK_DIR)/include/data.h: $(MK_DIR)/include/gen_data.py
	python3 $< > $@

# Clean
clean:
	rm -f $(DATA_H)
###############
# Build settings
############### 
INCDIRS += $(MK_DIR)/include
# Add the DATA_H to partial outputs
PARTIAL_OUTPUTS += $(DATA_H)
include ../../../common.mk
