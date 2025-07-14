# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Ryan Antonio <ryan.antonio@esat.kuleuven.be>

APP     = snax-hypercorex-dna-recog

# Get the data
INCDIRS += data

# Grab the include directory from the snax-hypercorex
INCDIRS += ../../../snax/hypercorex/include/

# Add binary to final build
RISCV_LDFLAGS += ../../../snax/hypercorex/build/snax-hypercorex-lib.o

SRCS    = src/dna-recog.c

include ./data/Makefile
include ../../common.mk

$(DEP): $(DATA_H)
