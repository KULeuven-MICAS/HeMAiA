# Copyright 2023 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

APP     = snax-mlperf-tiny-ad01

INCDIRS = ../../../snax-mlir/runtime/include
INCDIRS += data

# Include the snax-mlir prebuilt binary in the final build
RISCV_LDFLAGS += bin/snax-mlperf-tiny-ad01.o

SRCS    = src/snax-mlperf-tiny-ad01.c

include ./data/Makefile
include ../../common.mk


$(DEP): $(DATA_H)

build/snax-mlperf-tiny-ad01.part.elf: $(DEP) $(LD_SRCS) | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) $(RISCV_LDFLAGS) $(SRCS) -o $@


