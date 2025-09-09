# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# This file will be included to all the device application Makefiles.
# This device application Makefile needs to be run several times for different host apps.
# The host app name will be created on target/sw/device/apps/host_app_list.tmp
# Each host app will have different origin.ld

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
SNRT_DIR     = $(SNITCH_ROOT)/sw/snRuntime
SW_DIR       = $(ROOT)/target/sw
DEVICE_DIR   = $(SW_DIR)/device
RUNTIME_DIR  = $(DEVICE_DIR)/runtime
MATH_DIR     = $(DEVICE_DIR)/math

# Paths relative to the app including this Makefile
BUILDDIR    = $(abspath build)

# Read all host apps and origins
HOST_APP_NAMES := $(shell cat $(DEVICE_DIR)/host_app_list.tmp)
HOST_APP_ORIGINS := $(shell cat $(DEVICE_DIR)/host_app_origin.tmp)
DEV_APP_LISTS := $(shell cat $(DEVICE_DIR)/dev_app_list.tmp)

# Find which host apps use the current device app
HOST_APP_INDICES := $(shell \
    idx=1; \
    while read -r line; do \
        for dev in $$line; do \
            if [ "$$dev" = "$(APP)" ]; then \
                echo "$$idx"; \
            fi; \
        done; \
        idx=$$((idx+1)); \
    done < $(DEVICE_DIR)/dev_app_list.tmp \
)

# Get the host app names and origins that use this device app
HOST_APPS_FOR_DEV := $(foreach idx,$(HOST_APP_INDICES),$(word $(idx),$(HOST_APP_NAMES)))
HOST_ORIGINS_FOR_DEV := $(foreach idx,$(HOST_APP_INDICES),$(word $(idx),$(HOST_APP_ORIGINS)))

# Dependencies
INCDIRS += $(RUNTIME_DIR)/src
INCDIRS += $(RUNTIME_DIR)/api
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
# snax libs
INCDIRS += $(RUNTIME_DIR)/snax/xdma
INCDIRS += $(RUNTIME_DIR)/snax/gemmx

# Linking sources
BASE_TEMPLATE_LD = $(abspath $(APPSDIR)/base.template.ld)

# SNRT LIB
SNRT_LIB_DIR  	 = $(abspath $(RUNTIME_DIR)/build/)
SNRT_LIB_NAME    = snRuntime
SNRT_LIB         = $(realpath $(SNRT_LIB_DIR)/lib$(SNRT_LIB_NAME).a)

# Linker script
RISCV_LDFLAGS += -L$(APPSDIR)
RISCV_LDFLAGS += -L$(BUILDDIR)
# Link math library
RISCV_LDFLAGS += -L$(MATH_DIR)/build
RISCV_LDFLAGS += -lmath
# Link snRuntime
RISCV_LDFLAGS += -L$(SNRT_LIB_DIR)
RISCV_LDFLAGS += -l$(SNRT_LIB_NAME)

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

# Define outputs for each host app that uses this device app
define HOST_APP_RULES
ELF_$(1)         = $(abspath $(BUILDDIR)/$(1)_$(APP).elf)
DEP_$(1)         = $(abspath $(BUILDDIR)/$(1)_$(APP).d)
BIN_$(1)         = $(abspath $(BUILDDIR)/$(1)_$(APP).bin)
DUMP_$(1)        = $(abspath $(BUILDDIR)/$(1)_$(APP).dump)
DWARF_$(1)       = $(abspath $(BUILDDIR)/$(1)_$(APP).dwarf)
ORIGIN_LD_$(1)   = $(abspath $(BUILDDIR)/origin_$(1).ld)
BASE_LD_$(1)     = $(abspath $(BUILDDIR)/base_$(1).ld)
ALL_OUTPUTS += $$(ELF_$(1)) $$(BIN_$(1)) $$(DUMP_$(1)) $$(DWARF_$(1)) 

RISCV_LDFLAGS_$(1) = $(RISCV_LDFLAGS) -T$$(BASE_LD_$(1))

# Origin LD generation
$$(ORIGIN_LD_$(1)): | $(BUILDDIR)
	@echo "Generating origin LD for $(1) with origin $(2)"
	echo "L3_ORIGIN = $(2);" > $$@
	echo "L3_LENGTH = 0x100000000 - L3_ORIGIN;" >> $$@
	echo "MEMORY" >> $$@
	echo "{" >> $$@
	echo "    L3 (rwxa) : ORIGIN = L3_ORIGIN, LENGTH = L3_LENGTH" >> $$@
	echo "}" >> $$@

# Base LD generation
$$(BASE_LD_$(1)): $(BASE_TEMPLATE_LD) | $(BUILDDIR)
	@echo "Generating base LD for $(1)"
	sed 's|/\* INCLUDE directive will be replaced by Makefile with correct origin file \*/|INCLUDE origin_$(1).ld|' $$< > $$@

# Dependencies
$$(DEP_$(1)): $(SRCS) $(DATA_H) | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -MM -MT '$$(ELF_$(1))' $$< > $$@

# ELF generation
$$(ELF_$(1)): $$(DEP_$(1)) $$(BASE_LD_$(1)) $$(ORIGIN_LD_$(1)) $(SNRT_LIB) | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) $$(RISCV_LDFLAGS_$(1)) $(SRCS) -o $$@

# Other outputs
$$(BIN_$(1)): $$(ELF_$(1)) | $(BUILDDIR)
	$(RISCV_OBJCOPY) $(OBJCOPY_FLAGS) $$< $$@

$$(DUMP_$(1)): $$(ELF_$(1)) | $(BUILDDIR)
	$(RISCV_OBJDUMP) -D $$< > $$@

$$(DWARF_$(1)): $$(ELF_$(1)) | $(BUILDDIR)
	$(RISCV_DWARFDUMP) $$< > $$@
endef

# Create rules for each host app that uses this device app
$(foreach host,$(HOST_APPS_FOR_DEV),\
    $(eval $(call HOST_APP_RULES,$(host),$(word $(shell \
        idx=1; \
        for h in $(HOST_APP_NAMES); do \
            if [ "$$h" = "$(host)" ]; then \
                echo "$$idx"; \
                break; \
            fi; \
            idx=$$((idx+1)); \
        done \
    ),$(HOST_APP_ORIGINS))))\
)

#########
# Rules #
#########

.PHONY: all
all: $(ALL_OUTPUTS)

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)
	rm -rf $(DATA_H)
$(BUILDDIR):
	mkdir -p $@

ifneq ($(MAKECMDGOALS),clean)
DEP_FILES := $(foreach host,$(HOST_APPS_FOR_DEV),$(DEP_$(host)))
-include $(DEP_FILES)
endif