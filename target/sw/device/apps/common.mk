# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

# Usage of absolute paths is required to externally include
# this Makefile from multiple different locations
MK_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
include $(MK_DIR)/../toolchain.mk

###############
# Directories #
###############

# Fixed paths in repository tree
ROOT         = $(abspath $(MK_DIR)/../../../../)
SNITCH_ROOT  = $(shell bender path snitch_cluster)
APPSDIR      = $(abspath $(MK_DIR))
SW_DIR       = $(ROOT)/target/sw
RUNTIME_DIR  = $(SW_DIR)/device/runtime
MATH_DIR     = $(SW_DIR)/device/math
SNRT_DIR     = $(SNITCH_ROOT)/sw/snRuntime

# Paths relative to the app including this Makefile
BUILDDIR    = $(abspath build)

###################
# Build variables #
###################

# Dependencies
INCDIRS += $(RUNTIME_DIR)/src
INCDIRS += $(RUNTIME_DIR)/api
INCDIRS += $(RUNTIME_DIR)/snax/xdma
INCDIRS += $(RUNTIME_DIR)/snax/gemmx
INCDIRS += $(RUNTIME_DIR)/libsnaxkernel
INCDIRS += $(SNRT_DIR)/api
INCDIRS += $(SNRT_DIR)/src
INCDIRS += $(SNRT_DIR)/vendor/riscv-opcodes
INCDIRS += $(SW_DIR)/shared/platform/generated
INCDIRS += $(SW_DIR)/shared/runtime
INCDIRS += $(SNITCH_ROOT)/sw/blas
INCDIRS += $(SNRT_DIR)/../math/arch/riscv64/
INCDIRS += $(SNRT_DIR)/../math/arch/generic
INCDIRS += $(SNRT_DIR)/../math/src/include
INCDIRS += $(SNRT_DIR)/../math/src/internal
INCDIRS += $(SNRT_DIR)/../math/include/bits
INCDIRS += $(SNRT_DIR)/../math/include
INCDIRS += $(SW_DIR)/shared/vendor/o1heap/o1heap

# Linking sources
# BASE_LD       = $(abspath $(SNRT_DIR)/base.ld)
BASE_LD       = $(abspath $(APPSDIR)/base.ld)
MEMORY_LD     = $(abspath $(APPSDIR)/memory.ld)
ORIGIN_LD     = $(abspath $(BUILDDIR)/origin.ld)
# SNRT LIB
SNRT_LIB_DIR  = $(abspath $(RUNTIME_DIR)/build/)
SNRT_LIB_NAME = snRuntime
SNRT_LIB      = $(realpath $(SNRT_LIB_DIR)/lib$(SNRT_LIB_NAME).a)
# LD SRCS
LD_SRCS       = $(BASE_LD) $(MEMORY_LD) $(ORIGIN_LD) $(SNRT_LIB)

# Linker script
RISCV_LDFLAGS += -L$(APPSDIR)
RISCV_LDFLAGS += -L$(BUILDDIR)
RISCV_LDFLAGS += -T$(BASE_LD)
# RISCV_LDFLAGS += -T$(MEMORY_LD)
# Link snRuntime library
# Since the snax kernel is not used in the dev.c
# We need to keep the snax kernel symbol table by using the --whole-archive
RISCV_LDFLAGS += -Wl,--whole-archive
RISCV_LDFLAGS += -L$(SNRT_LIB_DIR)
RISCV_LDFLAGS += -l$(SNRT_LIB_NAME)
RISCV_LDFLAGS += -Wl,--no-whole-archive
# Link math library
RISCV_LDFLAGS += -L$(MATH_DIR)/build
RISCV_LDFLAGS += -lmath

# Objcopy flags
OBJCOPY_FLAGS  = -O binary
OBJCOPY_FLAGS += --remove-section=.comment
OBJCOPY_FLAGS += --remove-section=.riscv.attributes
OBJCOPY_FLAGS += --remove-section=.debug_info
OBJCOPY_FLAGS += --remove-section=.debug_abbrev
OBJCOPY_FLAGS += --remove-section=.debug_line
OBJCOPY_FLAGS += --remove-section=.debug_str
OBJCOPY_FLAGS += --remove-section=.debug_aranges

###########
# Outputs #
###########

ELF         = $(abspath $(BUILDDIR)/$(APP).elf)
DEP         = $(abspath $(BUILDDIR)/$(APP).d)
BIN         = $(abspath $(BUILDDIR)/$(APP).bin)
DUMP        = $(abspath $(BUILDDIR)/$(APP).dump)
DWARF       = $(abspath $(BUILDDIR)/$(APP).dwarf)
ALL_OUTPUTS = $(BIN) $(DUMP) $(DWARF)

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

$(DEP): $(SRCS) | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -MM -MT '$(ELF)' $< > $@

$(ELF): $(DEP) $(LD_SRCS) | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) $(RISCV_LDFLAGS) $(SRCS) -o $@

$(BIN): $(ELF) | $(BUILDDIR)
	$(RISCV_OBJCOPY) $(OBJCOPY_FLAGS) $< $@

$(DUMP): $(ELF) | $(BUILDDIR)
	$(RISCV_OBJDUMP) -D $< > $@

$(DWARF): $(ELF) | $(BUILDDIR)
	$(RISCV_DWARFDUMP) $< > $@

ifneq ($(MAKECMDGOALS),clean)
-include $(DEP)
endif
