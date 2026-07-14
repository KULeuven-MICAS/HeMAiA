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
# A recipe that dies after the shell has created/truncated `$@` leaves a partial
# file with a fresh mtime, which make then treats as up to date.
.DELETE_ON_ERROR:

MK_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
include $(MK_DIR)/../toolchain.mk
IS_CLEAN_GOAL := $(filter clean clean-%,$(MAKECMDGOALS))

###############
# Directories #
###############

# Fixed paths in repository tree
ROOT         = $(abspath $(MK_DIR)/../../../../)
ifeq ($(IS_CLEAN_GOAL),)
SNITCH_ROOT  := $(shell $(BENDER) path snitch_cluster)
else
SNITCH_ROOT  =
endif
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

# Names and origins are paired POSITIONALLY below ($(word $(idx),...)). If the two
# lists ever differ in length, every device ELF from the offending index onward is
# linked at another host app's L3_ORIGIN -- silently, with no error. Refuse instead.
# (DEV_APP_LISTS is not checked: it holds several device apps per line, so its word
# count legitimately differs.)
ifneq ($(words $(HOST_APP_NAMES)),$(words $(HOST_APP_ORIGINS)))
    $(error host_app_list.tmp ($(words $(HOST_APP_NAMES)) apps) and host_app_origin.tmp \
      ($(words $(HOST_APP_ORIGINS)) origins) are out of step -- \
      rm $(DEVICE_DIR)/*.tmp and rebuild)
endif

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

# Optional single-host-app filter. host_app_list.tmp accumulates EVERY host app that has
# registered (55 today, 53 of which offload to snax-bingo-offload), so an unfiltered build
# of one workload emits that device binary -- plus its dump and dwarf -- once per host app,
# 53 times over. `single-sw` passes the one host app it is building as HOST_APPS; `make sw`
# leaves it empty and builds the full matrix. Filtering on the INDICES keeps the names and
# origins below in sync.
ifneq ($(strip $(HOST_APPS)),)
HOST_APP_INDICES := $(foreach idx,$(HOST_APP_INDICES),\
    $(if $(filter $(HOST_APPS),$(word $(idx),$(HOST_APP_NAMES))),$(idx)))
endif

# Get the host app names and origins that use this device app
HOST_APPS_FOR_DEV := $(foreach idx,$(HOST_APP_INDICES),$(word $(idx),$(HOST_APP_NAMES)))
HOST_ORIGINS_FOR_DEV := $(foreach idx,$(HOST_APP_INDICES),$(word $(idx),$(HOST_APP_ORIGINS)))

# Dependencies
INCDIRS += $(RUNTIME_DIR)/src
INCDIRS += $(RUNTIME_DIR)/api
INCDIRS += $(RUNTIME_DIR)/generated
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
INCDIRS += $(SW_DIR)/shared/vendor/bingo_alloc/bingo_alloc
INCDIRS += $(SW_DIR)/shared/vendor/xdma
# snax libs
INCDIRS += $(RUNTIME_DIR)/snax/xdma
# Pull in every snax-generated SW-header dir (one per accelerator library,
# keyed on snax_library_name in the cluster hjson). Using a wildcard avoids
# hard-coding a single accelerator path, so the SW build transparently picks
# up whichever libraries `snax-sw-gen` produced for the active $(CFG)
# (versacore-to, gemmx, bingo, …). Listed before the local fallback so the
# generated streamer_csr_addr_map.h always wins over any stale checked-in copy.
INCDIRS += $(wildcard $(SNITCH_ROOT)/target/snitch_cluster/sw/snax/*/include)
INCDIRS += $(RUNTIME_DIR)/snax/versacore

# libbingo headers — host/device share <libbingo/device_kernel_args.h> so
# the device-side BINGO_GET_SP / BINGO_SW_GUARD_CHECK macros can derive the
# trailer offset via sizeof(args_t).
INCDIRS += $(SW_DIR)/host/runtime/libbingo/include

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

# Origin LD generation.
#
# Written ATOMICALLY: one redirection into a temp, then mv, so the file on disk is either
# the old one or the complete new one. An interrupted or re-entered build must not be able
# to leave a PARTIAL .ld behind -- the linker would then die on a malformed script
# ("{ expected" / "MEMORY") in a file nobody touched.
#
# The origin value is baked into the file, so the ORIGIN LIST is a prerequisite: if the host
# app relocates, the script is regenerated.
$$(ORIGIN_LD_$(1)): $(DEVICE_DIR)/host_app_origin.tmp | $(BUILDDIR)
	@echo "Generating origin LD for $(1) with origin $(2)"
	@{ echo "L3_ORIGIN = $(2);"; \
	   echo "L3_LENGTH = 0x100000000 - L3_ORIGIN;"; \
	   echo "MEMORY"; \
	   echo "{"; \
	   echo "    L3 (rwxa) : ORIGIN = L3_ORIGIN, LENGTH = L3_LENGTH"; \
	   echo "}"; } > $$@.tmp && mv -f $$@.tmp $$@

# Base LD generation
$$(BASE_LD_$(1)): $(BASE_TEMPLATE_LD) | $(BUILDDIR)
	@echo "Generating base LD for $(1)"
	sed 's|/\* INCLUDE directive will be replaced by Makefile with correct origin file \*/|INCLUDE origin_$(1).ld|' $$< > $$@

# Dependencies
# -MT names the .d ITSELF as well as the ELF, so the .d is regenerated whenever any header
# it lists changes. That is what keeps its header set current when an #include is added to
# a HEADER rather than to $(SRCS) or $(DATA_H); otherwise a later edit to that header alone
# would rebuild nothing and the device binary would go stale.
$$(DEP_$(1)): $(SRCS) $(DATA_H) | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -MM -MT '$$(ELF_$(1)) $$(DEP_$(1))' $$< > $$@

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
