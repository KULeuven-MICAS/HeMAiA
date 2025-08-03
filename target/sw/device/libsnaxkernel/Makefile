# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Fanchen Kong <fanchen.kong@kuleuven.be>
MK_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
include $(MK_DIR)/../toolchain.mk
###############
# Directories #
###############

# Fixed paths in repository tree
ROOT        = $(abspath $(MK_DIR)/../../../../)
SNITCH_ROOT = $(shell bender path snitch_cluster)
RUNTIME_DIR = $(ROOT)/target/sw/device/runtime
SNRT_DIR    = $(SNITCH_ROOT)/sw/snRuntime
SW_DIR      = $(ROOT)/target/sw/
MATH_DIR    = $(ROOT)/target/sw/device/math

###################
# Build variables #
###################


# Dependencies

INCDIRS += $(RUNTIME_DIR)/src
INCDIRS += $(RUNTIME_DIR)/api
# snitch runtime
INCDIRS += $(SNRT_DIR)/api
INCDIRS += $(SNRT_DIR)/src
# riscv-opcodes
INCDIRS += $(SNRT_DIR)/vendor/riscv-opcodes
# math lib
INCDIRS += $(SNRT_DIR)/../math/arch/riscv64/
INCDIRS += $(SNRT_DIR)/../math/arch/generic
INCDIRS += $(SNRT_DIR)/../math/src/include
INCDIRS += $(SNRT_DIR)/../math/src/internal
INCDIRS += $(SNRT_DIR)/../math/include/bits
INCDIRS += $(SNRT_DIR)/../math/include
# occamy platform shared
INCDIRS += $(SW_DIR)/shared/platform/generated
# INCDIRS += $(SW_DIR)/shared/platform
INCDIRS += $(SW_DIR)/shared/runtime
# local include file
INCDIRS += ./include
# vendor
INCDIRS += $(SW_DIR)/shared/vendor/o1heap/o1heap
BUILDDIR    = build
$(BUILDDIR):
	mkdir -p $@

SRCS = src/snax_kernel_lib.c
OBJS = $(BUILDDIR)/snax_kernel_lib.o
SNAX_LIB = $(BUILDDIR)/libsnaxkernel.a
SNAX_LIB_DUMP = $(BUILDDIR)/libsnaxkernel.dump

ALL_OUTPUTS = $(OBJS) $(SNAX_LIB) $(SNAX_LIB_DUMP)

all: $(ALL_OUTPUTS)

$(BUILDDIR)/snax_kernel_lib.o : $(SRCS) | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) $(RISCV_LDFLAGS) -c $< -o $@

$(SNAX_LIB): $(OBJS) | $(BUILDDIR)
	$(RISCV_AR) $(RISCV_ARFLAGS) $@ $^

$(SNAX_LIB_DUMP): $(SNAX_LIB) | $(BUILDDIR)
	$(RISCV_OBJDUMP) -D $< > $@

clean: 
	rm -rf $(BUILDDIR)
	