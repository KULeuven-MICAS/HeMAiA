# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Jun Yin <jun.yin@esat.kuleuven.be>

APP     = snax-cgra-run

INCDIRS = data

INCDIRS += ../../../snax/snax-cgra/include

# Include this binary in the final build
RISCV_LDFLAGS += ../../../snax/snax-cgra/build/snax-cgra-lib.o

SRCS    = src/snax-cgra-run.c

include ../../common.mk
#include ./data/Makefile
#$(DEP): $(DATA_H)
