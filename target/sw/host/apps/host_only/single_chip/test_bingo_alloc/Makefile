# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# Application specific settings
HOST_APP_TYPE = host_only
CHIP_TYPE     = single_chip
WORKLOAD      = test_bingo_alloc
SRCS = src/test_bingo_alloc.c
BINGO_HOST ?= 0
INCL_DEVICE_BINARY ?= 0

include ../../../common.mk

# Compile bingo_alloc.c directly (SWDIR is defined by common.mk)
SRCS += $(SWDIR)/shared/vendor/bingo_alloc/bingo_alloc/bingo_alloc.c
