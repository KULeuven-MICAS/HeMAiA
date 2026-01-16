# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Ryan Antonio <ryan.antonio@esat.kuleuven.be>

APP     = snax-nvsa

# Get the data
INCDIRS += data

# Grab the include directory from the snax-nvsa
INCDIRS += ../../../snax/hypercorex/include/
INCDIRS += ../../../snax/gemmx/include
INCDIRS += ../../../snax/xdma/include/
INCDIRS += ../../../snax/snax-cgra/include

# Add binary to final build
RISCV_LDFLAGS += ../../../snax/gemmx/build/snax-gemmx-lib.o
RISCV_LDFLAGS += ../../../snax/xdma/build/snax-xdma-lib.o
RISCV_LDFLAGS += ../../../snax/hypercorex/build/snax-hypercorex-lib.o
RISCV_LDFLAGS += ../../../snax/snax-cgra/build/snax-cgra-lib.o

SRCS    = src/snax-nvsa.c

# include ./data/Makefile
include ../../common.mk

# $(DEP): $(DATA_H)
