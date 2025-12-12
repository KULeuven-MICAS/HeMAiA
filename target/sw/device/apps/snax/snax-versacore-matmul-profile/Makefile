# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yunhao Deng <yunhao.deng@kuleuven.be>

APP     = snax-versacore-matmul-profile

INCDIRS += data ../../runtime/snax/versacore/include

SRCS    = src/snax-versacore-matmul-profile.c

include ./data/Makefile
include ../../common.mk

$(DEP): $(DATA_H)
