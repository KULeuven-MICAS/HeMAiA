# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

APP     = gemm-msplit-1cluster-4chip

INCDIRS += data ../../runtime/snax/versacore/include

SRCS    = src/gemm-msplit-1cluster-4chip.c

include ./data/Makefile
include ../../common.mk

$(DEP): $(DATA_H)
