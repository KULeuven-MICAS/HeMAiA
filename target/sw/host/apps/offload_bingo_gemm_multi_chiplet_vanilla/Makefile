# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
APP  = offload_bingo_gemm_multi_chiplet_vanilla
SRCS = src/offload_bingo_gemm_multi_chiplet.c
BINGO_HOST ?= 1 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 1 # True to include the device binary
DEVICE_APPS = snax-bingo-offload
####################################
# New bingo workloads defined here #
####################################
INCDIRS += include


include ../common.mk
include ./include/Makefile

$(DEP): $(DATA_H)
