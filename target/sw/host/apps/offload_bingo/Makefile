# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
APP  = offload_bingo
SRCS = src/offload_bingo.c
BINGO_HOST ?= 1 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 1 # True to include the device binary
DEVICE_APPS = snax-bingo-offload
####################################
# New bingo workloads defined here #
####################################
INCDIRS += include
INCDIRS += include/workloads/dummy
INCDIRS += include/workloads/double_buffer
INCDIRS += include/workloads/xdma_1d_copy
INCDIRS += include/workloads/versacore

include ../common.mk
