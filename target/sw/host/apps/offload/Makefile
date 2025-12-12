# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

APP  = offload
SRCS = src/offload.c

BINGO_HOST ?= 0 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 1 # True to include the device binary

# those are the legacy apps that we want to keep supporting and for ci
DEVICE_APPS = snax-test-integration
DEVICE_APPS += snax-xdma-copy
DEVICE_APPS += snax-xdma-hazard
DEVICE_APPS += snax-versacore-matmul-profile

include ../common.mk
