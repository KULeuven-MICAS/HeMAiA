# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
HOST_APP_TYPE = offload_legacy
CHIP_TYPE     = multi_chip
DEV_APP = snax-xdma-multicast-chiplet

SRCS = src/offload_legacy_multi_chip.c

BINGO_HOST ?= 0 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 1 # True to include the device binary

# Include more dev apps here
DEVICE_APPS = snax-xdma-multicast-chiplet
include ../../common.mk