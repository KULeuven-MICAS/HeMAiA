# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# JYIN

APP     = snax-cgra-cmatmul
SRCS    = src/$(APP).c
INCDIRS = ./data
INCDIRS += ../../../snax/snax-cgra/include

RISCV_LDFLAGS += ../../../snax/snax-cgra/build/snax-cgra-lib.o
# RISCV_LDFLAGS += -Wl,-Map=gc.map
# RISCV_LDFLAGS += -Wl,--gc-sections

include ../../common.mk
#include ./data/Makefile
$(DEP): $(DATA_H)
