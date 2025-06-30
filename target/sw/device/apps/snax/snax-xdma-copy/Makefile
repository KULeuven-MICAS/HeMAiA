# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yunhao Deng <yunhao.deng@kuleuven.be>

APP     = snax-xdma-copy

INCDIRS += data
INCDIRS += INCDIRS += ../../../snax/xdma/include/

# Add binary to final build
RISCV_LDFLAGS += ../../../snax/xdma/build/snax-xdma-lib.o

SRCS    = src/snax-xdma-copy.c

include ./data/Makefile
include ../../common.mk

$(DEP): $(DATA_H)
