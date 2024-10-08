# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

include ../toolchain.mk

###################
# Build variables #
###################

# Directories
SNITCH_ROOT = $(shell bender path snitch_cluster)
BUILDDIR    = $(abspath build/)
SRC_DIR     = $(abspath src/)
SW_DIR      = $(abspath ../../)
SNRT_DIR    = $(SNITCH_ROOT)/sw/snRuntime

# Dependencies
INCDIRS += $(SNRT_DIR)/src
INCDIRS += $(SNRT_DIR)/api
INCDIRS += $(SNRT_DIR)/vendor/riscv-opcodes
INCDIRS += $(SW_DIR)/shared/platform/generated
INCDIRS += $(SW_DIR)/shared/runtime
INCDIRS += $(SNRT_DIR)/../math/arch/riscv64/
INCDIRS += $(SNRT_DIR)/../math/arch/generic
INCDIRS += $(SNRT_DIR)/../math/src/include
INCDIRS += $(SNRT_DIR)/../math/src/internal
INCDIRS += $(SNRT_DIR)/../math/include/bits
INCDIRS += $(SNRT_DIR)/../math/include
# Sources
SRCS += $(SRC_DIR)/occamy_start.S
SRCS += $(SRC_DIR)/snrt.c

###########
# Outputs #
###########

OBJS        = $(addprefix $(BUILDDIR)/,$(addsuffix .o,$(basename $(notdir $(SRCS)))))
DEPS        = $(addprefix $(BUILDDIR)/,$(addsuffix .d,$(basename $(notdir $(SRCS)))))
LIB         = $(BUILDDIR)/libsnRuntime.a
DUMP        = $(BUILDDIR)/libsnRuntime.dump
ALL_OUTPUTS = $(DEPS) $(LIB) $(DUMP)

#########
# Rules #
#########

.PHONY: all
all: $(ALL_OUTPUTS)

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)

$(BUILDDIR):
	mkdir -p $@

$(BUILDDIR)/%.o: $(SRC_DIR)/%.S | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o: $(SRC_DIR)/%.c | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -c $< -o $@

$(BUILDDIR)/%.d: $(SRC_DIR)/%.c | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -MM -MT '$(@:.d=.o)' $< > $@

$(LIB): $(OBJS) | $(BUILDDIR)
	$(RISCV_AR) $(RISCV_ARFLAGS) $@ $^

$(DUMP): $(LIB) | $(BUILDDIR)
	$(RISCV_OBJDUMP) -D $< > $@

ifneq ($(MAKECMDGOALS),clean)
-include $(DEPS)
endif
