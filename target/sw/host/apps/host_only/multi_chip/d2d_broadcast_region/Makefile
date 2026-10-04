# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
# Application specific settings
MK_DIR   := $(realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST)))))
HOST_APP_TYPE = host_only
CHIP_TYPE     = multi_chip
WORKLOAD      = d2d_broadcast_region
SRCS = src/d2d_broadcast_region.c
BINGO_HOST ?= 0 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 0 # True to include the device binary
include ../../../common.mk
