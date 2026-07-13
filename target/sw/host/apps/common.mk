# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# Root directory (absolute path to the host directory)
ifndef HOST_DIR
HOST_DIR = $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
endif

# There are three main stages of building a host application:
# 1) host-partial-build: compile and link the host application without the device binary. This is mainly to determine the relocation
#    address of the device binary, which is needed to create the origin.ld file.
# 2) device-build: based on the generated origin.ld file, build the device application to get the device binary.
# 3) host-finalize-build: compile and link the host application with the device binary

# Each host application will include this common makefile to define the build rules.

# APP name composition
ifndef APP
    ifeq ($(HOST_APP_TYPE), host_only)
        APP = $(WORKLOAD)
    else ifeq ($(HOST_APP_TYPE), offload_legacy)
        APP = offload_legacy_$(CHIP_TYPE)
    else ifneq ($(HOST_APP_TYPE),)
        APP = $(HOST_APP_TYPE)_$(CHIP_TYPE)_$(WORKLOAD)
    endif
endif

# Validation of input variables
ALLOWED_HOST_APP_TYPES = host_only offload_legacy offload_bingo_sw offload_bingo_hw
ALLOWED_CHIP_TYPES     = single_chip multi_chip

# Skip validation for clean target
ifneq ($(MAKECMDGOALS),clean)
    ifeq ($(filter $(HOST_APP_TYPE),$(ALLOWED_HOST_APP_TYPES)),)
        $(error Invalid or missing HOST_APP_TYPE: "$(HOST_APP_TYPE)". Allowed values: $(ALLOWED_HOST_APP_TYPES))
    endif

    ifeq ($(filter $(CHIP_TYPE),$(ALLOWED_CHIP_TYPES)),)
        $(error Invalid or missing CHIP_TYPE: "$(CHIP_TYPE)". Allowed values: $(ALLOWED_CHIP_TYPES))
    endif

    # Check for WORKLOAD if not offload_legacy
    ifneq ($(HOST_APP_TYPE), offload_legacy)
        ifeq ($(WORKLOAD),)
            $(error WORKLOAD must be specified for HOST_APP_TYPE=$(HOST_APP_TYPE))
        endif
    endif

    # Check for DEV_APP if it's an offload type
    ifneq ($(filter $(HOST_APP_TYPE),offload_legacy offload_bingo_sw offload_bingo_hw),)
        ifeq ($(DEV_APP),)
            $(error DEV_APP must be specified for offload applications)
        endif
    endif
endif

# Use DEV_APP if provided to restrict or set DEVICE_APPS
ifneq ($(DEV_APP),)
    DEVICE_APPS := $(DEV_APP)
endif

######################
# Invocation options #
######################

DEBUG ?= OFF # ON to turn on debugging symbols

###################
# Build variables #
###################

# Compiler toolchain
# CVA6_GCC_ROOT = /opt/riscv
CVA6_GCC_ROOT ?= 
RISCV_CC      = $(CVA6_GCC_ROOT)riscv64-unknown-elf-gcc
RISCV_OBJCOPY = $(CVA6_GCC_ROOT)riscv64-unknown-elf-objcopy
RISCV_OBJDUMP = $(CVA6_GCC_ROOT)riscv64-unknown-elf-objdump
RISCV_READELF = $(CVA6_GCC_ROOT)riscv64-unknown-elf-readelf
# Directories
# Notice the build dir is the dir under each app
BUILDDIR    ?= $(abspath build)
# Global DIRs
SWDIR       = $(abspath $(HOST_DIR)/..)
RUNTIME_DIR = $(abspath $(HOST_DIR)/runtime)
DEVICE_DIR  = $(abspath $(HOST_DIR)/../device)

# Dependencies
INCDIRS += $(RUNTIME_DIR)
INCDIRS += $(abspath $(SWDIR)/shared/platform/generated)
INCDIRS += $(abspath $(SWDIR)/shared/runtime)
SRCS    += $(RUNTIME_DIR)/start.S

# Include XDMA
INCDIRS += $(SWDIR)/shared/vendor/xdma

# libbingo headers are always needed because host_kernel_lib.h (included via
# host.h in every host app) references libbingo symbols (BINGO_CHECK_TYPE_*,
# BINGO_GATING_MODE_*, BINGO_PRINTF, bingo_cerf_update, bingo_utils.h helpers).
# Linking against libbingo remains conditional on BINGO_HOST below.
INCDIRS += $(abspath $(HOST_DIR)/runtime/libbingo/include)
INCDIRS += $(SWDIR)/shared/vendor/bingo_alloc/bingo_alloc

# VersaCore shared headers — notably gemm_shapes.h, used by host workloads
# that size GEMM buffers from bingo_gemm_shape_params[array_shape]. The
# device-only snax_versacore_lib.h also lives here; host code should only
# include <gemm_shapes.h>.
INCDIRS += $(abspath $(DEVICE_DIR)/runtime/snax/versacore)

# Compiler flags
RISCV_CFLAGS += $(addprefix -I,$(INCDIRS))
RISCV_CFLAGS += -march=rv64gcv_zvfh # v extension for ara + zvfh for fp16 vectors
RISCV_CFLAGS += -mabi=lp64d
RISCV_CFLAGS += -mcmodel=medany
RISCV_CFLAGS += -ffast-math
RISCV_CFLAGS += -fno-builtin-printf
RISCV_CFLAGS += -fno-common
RISCV_CFLAGS += -O2
RISCV_CFLAGS += -fno-tree-vectorize -fno-tree-loop-vectorize -fno-tree-slp-vectorize
RISCV_CFLAGS += -ffunction-sections
RISCV_CFLAGS += -fdata-sections
RISCV_CFLAGS += -Wextra
RISCV_CFLAGS += -Werror
RISCV_CFLAGS += -std=gnu99
ifeq ($(DEBUG),ON)
RISCV_CFLAGS += -g
endif
RISCV_CFLAGS += $(USER_FLAGS)
# Linking sources
LINKER_SCRIPT = $(abspath $(HOST_DIR)/runtime/host.ld)
LD_SRCS       = $(LINKER_SCRIPT)
# Linker flags
RISCV_LDFLAGS += -nostartfiles
RISCV_LDFLAGS += -lm
RISCV_LDFLAGS += -lgcc
RISCV_LDFLAGS += -T$(LINKER_SCRIPT)
# The host runs bare-metal from a unified SPM (host.ld maps code+data+bss into
# one WIDE_SPM region), so the single PT_LOAD segment is intentionally RWX.
# Silence binutils >= 2.39's W^X advisory warning (the flag was introduced in
# the same release that added the warning, so any ld that warns accepts it).
RISCV_LDFLAGS += -Wl,--no-warn-rwx-segments
# GC unreferenced functions/data (paired with -ffunction-sections/-fdata-sections): the multichip
# 1-cluster chips have a small 128 KiB wide SPM, and linking all of libbingo's host kernels leaves
# too little for the L3 task-metadata heap -> host L3 OOM. Stripping unused kernels shrinks .text and
# frees the heap. The offload_bingo_hw.h references its live kernels by &symbol, so they are kept.
RISCV_LDFLAGS += -Wl,--gc-sections


# if the host application uses the bingo runtime
ifneq (,$(filter $(BINGO_HOST),True true TRUE 1))
LIBBINGO_DIR = $(abspath $(RUNTIME_DIR)/libbingo)


# libbingo (includes already added above unconditionally)
BINGO_LIB_DIR = $(LIBBINGO_DIR)/build
BINGO_LIB_NAME = bingo
BINGO_LIB = $(BINGO_LIB_DIR)/lib$(BINGO_LIB_NAME).a

LD_SRCS += $(BINGO_LIB) 

# link bingo lib
RISCV_LDFLAGS += -L$(BINGO_LIB_DIR)
RISCV_LDFLAGS += -l$(BINGO_LIB_NAME)

endif


###########
# Outputs #
###########

# Whatever the APP declared as generated (data headers, offload headers, ...) -- captured
# HERE, before common.mk appends its own outputs to PARTIAL_OUTPUTS below, so this is the
# app's generated files and nothing else. $(SRCS) #include some of them (gemm_data.h,
# xdma_data.h, offload_bingo_hw.h, op_test_data.h), so the -MM dependency scan must not
# run until they exist -- see the $(DEP) rule.
APP_GEN_FILES := $(PARTIAL_OUTPUTS)

PARTIAL_ELF     = $(abspath $(BUILDDIR)/$(APP).part.elf)
PARTIAL_DUMP    = $(abspath $(BUILDDIR)/$(APP).part.dump)
DEP             = $(abspath $(BUILDDIR)/$(APP).d)

ifneq (,$(filter $(INCL_DEVICE_BINARY),True true TRUE 1))
PARTIAL_HOST_APP_LIST     = $(abspath $(DEVICE_DIR)/host_app_list.$(APP).tmp)
PARTIAL_HOST_APP_ORIGIN   = $(abspath $(DEVICE_DIR)/host_app_origin.$(APP).tmp)
PARTIAL_DEV_APP_LIST      = $(abspath $(DEVICE_DIR)/dev_app_list.$(APP).tmp)



PARTIAL_TMPS = $(PARTIAL_HOST_APP_LIST) $(PARTIAL_HOST_APP_ORIGIN) $(PARTIAL_DEV_APP_LIST)
PARTIAL_OUTPUTS += $(PARTIAL_ELF) $(PARTIAL_DUMP) $(PARTIAL_TMPS)
ELFS             = $(addprefix $(BUILDDIR)/$(APP)_, $(addsuffix .elf, $(DEVICE_APPS)))
DUMPS            = $(ELFS:.elf=.dump)
DWARFS           = $(ELFS:.elf=.dwarf)
BINS             = $(ELFS:.elf=.bin)
FINAL_OUTPUTS    = $(ELFS) $(DUMPS) $(DWARFS) $(BINS)
else
PARTIAL_OUTPUTS += $(PARTIAL_ELF) $(PARTIAL_DUMP)
ELF              = $(abspath $(BUILDDIR)/$(APP).elf)
DUMP             = $(abspath $(BUILDDIR)/$(APP).dump)
DWARF            = $(abspath $(BUILDDIR)/$(APP).dwarf)
BIN              = $(abspath $(BUILDDIR)/$(APP).bin)
FINAL_OUTPUTS    = $(ELF) $(DUMP) $(DWARF) $(BIN)
endif


#########
# Rules #
#########

.DEFAULT_GOAL := all
TARGET ?= all

.PHONY: all partial-build finalize-build clean FORCE
all:
ifeq ($(TARGET),all)
	$(MAKE) partial-build
	$(MAKE) finalize-build
else
	$(MAKE) $(TARGET)
endif

FORCE:

ifneq (,$(filter $(INCL_DEVICE_BINARY),True true TRUE 1))
# Host app list
$(PARTIAL_HOST_APP_LIST): FORCE | $(DEVICE_BUILDDIR)
	echo "$(APP)" > $@

# Host app origin (from partial ELF)
$(PARTIAL_HOST_APP_ORIGIN): $(PARTIAL_ELF) FORCE | $(DEVICE_BUILDDIR)
	@RELOC_ADDR=$$($(RISCV_OBJDUMP) -t $< | grep snitch_main | cut -c9-16); \
	echo "0x$$RELOC_ADDR" > $@

# Dev app list (for each host app -> which device apps it needs)
$(PARTIAL_DEV_APP_LIST): FORCE | $(DEVICE_BUILDDIR)
	echo "$(DEVICE_APPS)" > $@
endif



# ---- Platform-applicability guard (bingo_hw workloads only) -----------------
# A bingo workload declares the platform it targets (num_chiplets / num_clusters in
# its params.hjson). The mini_compiler enforces that: guard_chiplet_count /
# guard_cluster_count REFUSE to emit $(OFFLOAD_H) when the HW actually built does not
# match (e.g. a 4-chiplet workload under a 1-chiplet cfg), print a WARNING, and exit
# cleanly -- "this workload does not apply to this platform".
#
# make used to ignore that and compile the .c anyway, dying on the missing include
# ("fatal error: offload_bingo_hw.h: No such file or directory"). Because the app
# trees build their workloads as ordinary prerequisites, make stops at the FIRST
# inapplicable one -- so a single non-matching workload (gemm_msplit_4chiplet_1cluster,
# first in APPS) took down every other workload behind it, and the only reason the
# tree ever built was a stale offload header left on disk from a matching cfg.
#
# So honour the guard here: generate the header first, and if the guard declined to
# produce it, build nothing for this workload and say so. Applicable workloads take
# the identical path as before (the header is already made when the outputs are built).
ifneq ($(strip $(OFFLOAD_H)),)

.PHONY: partial-build finalize-build __bingo-partial __bingo-finalize

partial-build finalize-build:
	@$(MAKE) --no-print-directory $(OFFLOAD_H)
	@if [ -f "$(OFFLOAD_H)" ]; then \
	    $(MAKE) --no-print-directory __bingo-$(patsubst %-build,%,$@); \
	else \
	    echo "[skip] $(APP): platform guard did not emit $(notdir $(OFFLOAD_H)) — workload does not target this HW cfg; nothing to build"; \
	fi

__bingo-partial: $(PARTIAL_OUTPUTS)
__bingo-finalize: $(FINAL_OUTPUTS)

else

.PHONY: partial-build
partial-build: $(PARTIAL_OUTPUTS)

.PHONY: finalize-build
finalize-build: $(FINAL_OUTPUTS)

endif

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)
	rm -rf $(PARTIAL_TMPS)

$(BUILDDIR):
	mkdir -p $@

$(DEVICE_BUILDDIR):
	mkdir -p $@

# Every target whose recipe COMPILES $(SRCS). They must all carry the header
# prerequisites the compiler discovers. ($(ELFS) and $(ELF) are set in mutually
# exclusive branches above, so exactly one of them is non-empty here.)
DEP_TARGETS = $(PARTIAL_ELF) $(ELFS) $(ELF)

# $(APP_GEN_FILES) are prerequisites, not just $(SRCS): the -MM scan COMPILES the sources,
# which #include the app's generated headers, so those must be generated first. Without
# them here, `-include $(DEP)` (below) made make remake $(DEP) during its parse phase --
# ahead of any normal target ordering -- and the scan died on the not-yet-generated
# gemm_data.h / xdma_data.h.
$(DEP): $(SRCS) $(APP_GEN_FILES) | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -MM -MT '$(strip $(DEP_TARGETS))' $< > $@

# Read the discovered prerequisites. WITHOUT this include, the .d file was generated and
# then never consulted: $(DEP) itself only depends on $(SRCS), so editing a HEADER
# (host_kernel_lib.h, ara_sweep.h, ...) regenerated nothing, left $(DEP) untouched, and
# make declared the ELF up to date -- "Nothing to be done for 'partial-build'" -- while
# silently re-staging a binary built from the OLD header. Only a full `make clean`, which
# the sweep runner happens to do, ever picked up a kernel edit.
# (make re-makes an included file if it is out of date, so this stays self-updating.)
#
# But a bingo_hw workload's $(SRCS) #includes a GENERATED header ($(OFFLOAD_H)), and
# including $(DEP) makes `make` REMAKE it -- which runs the compiler's -MM pass over those
# sources. So only pull $(DEP) in once that header exists. Otherwise the -MM pass dies on
# the missing include, and for a workload the platform guard above deliberately skips
# (wrong chiplet/cluster count) the header NEVER appears, so it would die every time --
# turning a clean "[skip]" into a hard error. On the very first build the header is not
# there yet either, but that build compiles everything from scratch anyway, and the next
# one picks the deps up.
ifeq ($(strip $(OFFLOAD_H)),)
-include $(DEP)
else ifneq ($(wildcard $(OFFLOAD_H)),)
-include $(DEP)
endif

# Partially linked object
$(PARTIAL_ELF): $(DEP) $(LD_SRCS) | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) $(SRCS) $(RISCV_LDFLAGS)  -o $@

$(PARTIAL_DUMP): $(PARTIAL_ELF) | $(BUILDDIR)
	$(RISCV_OBJDUMP) -D $< > $@

ifneq (,$(filter $(INCL_DEVICE_BINARY),True true TRUE 1))
# only the incldevicebinary option is enabled, we build the final host app with device binaries

define elf_rule_template =
    $$(BUILDDIR)/$$(APP)_$(1).elf: $$(DEVICE_DIR)/apps/snax/$(1)/build/$$(APP)_$(1).bin $$(DEP) $$(LD_SRCS) | $$(BUILDDIR)
		$$(RISCV_CC) $$(RISCV_CFLAGS) $$(SRCS) -DDEVICEBIN=\"$$<\" $$(RISCV_LDFLAGS)  -o $$@
endef
$(foreach f,$(DEVICE_APPS),$(eval $(call elf_rule_template,$(f))))

$(BUILDDIR)/$(APP)_%.dump: $(BUILDDIR)/$(APP)_%.elf | $(BUILDDIR)
	$(RISCV_OBJDUMP) -D $< > $@

$(BUILDDIR)/$(APP)_%.dwarf: $(BUILDDIR)/$(APP)_%.elf | $(BUILDDIR)
	$(RISCV_READELF) --debug-dump $< > $@

$(BUILDDIR)/$(APP)_%.bin: $(BUILDDIR)/$(APP)_%.elf | $(BUILDDIR)
	$(RISCV_OBJCOPY) -O binary $< $@
else
# else we only build the host app without device binaries
$(ELF): $(DEP) $(LD_SRCS) | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) $(SRCS) $(RISCV_LDFLAGS) -o $@
$(DUMP): $(ELF) | $(BUILDDIR)
	$(RISCV_OBJDUMP) -D $< > $@
$(DWARF): $(ELF) | $(BUILDDIR)
	$(RISCV_READELF) --debug-dump $< > $@
$(BIN): $(ELF) | $(BUILDDIR)
	$(RISCV_OBJCOPY) -O binary $< $@
endif