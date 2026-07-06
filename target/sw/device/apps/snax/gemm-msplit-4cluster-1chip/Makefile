# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

APP     = gemm-msplit-4cluster-1chip

INCDIRS += data ../../runtime/snax/versacore/include

SRCS    = src/gemm-msplit-4cluster-1chip.c

include ./data/Makefile
include ../../common.mk

$(DEP): $(DATA_H)
