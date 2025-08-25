# Copyright 2023 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

APP     = snax-gemmx-resnet8

INCDIRS = data

INCDIRS += ../../../snax/gemmx/include

# Include this binary in the final build
RISCV_LDFLAGS += ../../../snax/gemmx/build/snax-gemmx-lib.o

INCDIRS += INCDIRS += ../../../snax/xdma/include/

# Add binary to final build
RISCV_LDFLAGS += ../../../snax/xdma/build/snax-xdma-lib.o

SRCS    = src/snax-gemmx-resnet8.c

include ./data/Makefile
include ../../common.mk

$(DEP): $(DATA_H)
