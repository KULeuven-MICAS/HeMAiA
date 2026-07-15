# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

APP     = versacore-matmul-profile-2cluster-4chip

INCDIRS += data ../../runtime/snax/versacore/include

SRCS    = src/versacore-matmul-profile-2cluster-4chip.c

include ./data/Makefile
include ../../common.mk

$(DEP): $(DATA_H)
