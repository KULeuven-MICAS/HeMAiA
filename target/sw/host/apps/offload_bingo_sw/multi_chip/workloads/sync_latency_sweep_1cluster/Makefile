# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

MK_DIR   := $(realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST)))))
HOST_APP_TYPE = offload_bingo_sw
CHIP_TYPE     = multi_chip
WORKLOAD      = sync_latency_sweep_1cluster
DEV_APP       = snax-bingo-offload

SRCS = ../../src/offload_bingo_sw.c

BINGO_HOST ?= 1
INCL_DEVICE_BINARY ?= 1

# No datagen: this workload touches no data.
INCDIRS += $(MK_DIR) $(MK_DIR)/../sync_latency_common

include ../../../../common.mk
