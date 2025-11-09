# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

APP  = offload_bingo_multichip
SRCS = src/offload_bingo_multichip.c

BINGO_HOST ?= 1 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 1 # True to include the device binary
DEVICE_APPS = snax-bingo-offload
INCDIRS += include
####################################
# New bingo workloads defined here #
####################################
INCDIRS += include/workloads/dummy

include ../common.mk
