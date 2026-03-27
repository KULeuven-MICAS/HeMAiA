# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>


######################
# Invocation options #
######################

DEBUG ?= OFF # ON to turn on debugging symbols

APP  = bin_search
SRCS = src/$(APP).c
INCDIRS = data

include ./data/Makefile
include ../common.mk

$(DEP): $(DATA_H)

# Make CVA6 binary
RISCV_OBJCOPY = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-objcopy

.PHONY: finalize-build
finalize-build: $(BUILDDIR)/$(APP).bin

$(BUILDDIR)/$(APP).bin: $(BUILDDIR)/$(APP).elf | $(BUILDDIR)
	$(RISCV_OBJCOPY) -O binary $< $@
