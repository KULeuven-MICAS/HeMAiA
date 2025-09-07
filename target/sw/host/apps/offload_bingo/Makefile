# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

######################
# Invocation options #
######################

DEBUG ?= OFF # ON to turn on debugging symbols

###################
# Build variables #
###################

APP  = offload_bingo
SRCS = src/offload_bingo.c

# Compiler toolchain
CVA6_GCC_ROOT = /tools/riscv/bin
RISCV_CC      = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-gcc
RISCV_OBJCOPY = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-objcopy
RISCV_OBJDUMP = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-objdump
RISCV_READELF = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-readelf

# Directories
BUILDDIR     = $(abspath build)
SWDIR        = $(abspath ../../../)
HOST_DIR     = $(abspath ../../)
RUNTIME_DIR  = $(abspath $(HOST_DIR)/runtime)
DEVICE_DIR   = $(abspath $(HOST_DIR)/../device)
LIBBINGO_DIR = $(abspath $(RUNTIME_DIR)/libbingo)
LIBHERO_DIR  = $(abspath $(RUNTIME_DIR)/libhero)
# now we only include the snax app
DEVICE_APPS += apps/snax/snax-bingo-offload

# Dependencies
INCDIRS += include
INCDIRS += $(RUNTIME_DIR)
INCDIRS += $(HOST_DIR)/../shared/platform/generated
INCDIRS += $(HOST_DIR)/../shared/runtime
# libhero
INCDIRS += $(LIBHERO_DIR)/include
INCDIRS += $(LIBHERO_DIR)/src
INCDIRS += $(SWDIR)/shared/vendor/o1heap/o1heap
# libbingo
INCDIRS += $(LIBBINGO_DIR)/include

##############################
# New workloads defined here #
##############################
# INCDIRS += include/workloads/<workload_name>
INCDIRS += include/workloads/dummy
INCDIRS += include/workloads/xdma_1d


SRCS    += $(RUNTIME_DIR)/start.S

# Compiler flags
RISCV_CFLAGS += $(addprefix -I,$(INCDIRS))
RISCV_CFLAGS += -march=rv64imafdc
RISCV_CFLAGS += -mabi=lp64d
RISCV_CFLAGS += -mcmodel=medany
RISCV_CFLAGS += -ffast-math
RISCV_CFLAGS += -fno-builtin-printf
RISCV_CFLAGS += -fno-common
RISCV_CFLAGS += -O3
RISCV_CFLAGS += -ffunction-sections
RISCV_CFLAGS += -Wextra
RISCV_CFLAGS += -Werror
ifeq ($(DEBUG), ON)
RISCV_CFLAGS += -g
endif

# Linking sources
LINKER_SCRIPT = $(abspath $(HOST_DIR)/runtime/host.ld)
# HERO LIB
HERO_LIB_DIR = $(LIBHERO_DIR)/build
HERO_LIB_NAME = hero
HERO_LIB = $(HERO_LIB_DIR)/lib$(HERO_LIB_NAME).a
# BINGO LIB
BINGO_LIB_DIR = $(LIBBINGO_DIR)/build
BINGO_LIB_NAME = bingo
BINGO_LIB = $(BINGO_LIB_DIR)/lib$(BINGO_LIB_NAME).a

LD_SRCS       = $(LINKER_SCRIPT) $(BINGO_LIB) $(HERO_LIB)

# Linker flags
RISCV_LDFLAGS += -nostartfiles
RISCV_LDFLAGS += -lm
RISCV_LDFLAGS += -lgcc
RISCV_LDFLAGS += -T$(LINKER_SCRIPT)
# link bingo lib
RISCV_LDFLAGS += -Wl,--start-group
RISCV_LDFLAGS += -L$(BINGO_LIB_DIR)
RISCV_LDFLAGS += -l$(BINGO_LIB_NAME)
# link hero lib
RISCV_LDFLAGS += -L$(HERO_LIB_DIR)
RISCV_LDFLAGS += -l$(HERO_LIB_NAME)
RISCV_LDFLAGS += -Wl,--end-group

# Device binaries
DEVICE_BUILDDIRS = $(addsuffix /build, $(addprefix $(DEVICE_DIR)/, $(DEVICE_APPS)))

###########
# Outputs #
###########

PARTIAL_ELF     = $(abspath $(BUILDDIR)/$(APP).part.elf)
DEP             = $(abspath $(BUILDDIR)/$(APP).d)
PARTIAL_DUMP    = $(abspath $(BUILDDIR)/$(APP).part.dump)
LD_ORIGINS      = $(abspath $(addsuffix /origin.ld, $(DEVICE_BUILDDIRS)))
ELFS            = $(abspath $(addsuffix .elf, $(addprefix $(BUILDDIR)/$(APP)-, $(notdir $(DEVICE_APPS)))))
BINS            = $(abspath $(addsuffix .bin, $(addprefix $(BUILDDIR)/$(APP)-, $(notdir $(DEVICE_APPS)))))
DUMPS           = $(abspath $(addsuffix .dump, $(addprefix $(BUILDDIR)/$(APP)-, $(notdir $(DEVICE_APPS)))))
DWARFS          = $(abspath $(addsuffix .dwarf, $(addprefix $(BUILDDIR)/$(APP)-, $(notdir $(DEVICE_APPS)))))
PARTIAL_OUTPUTS = $(PARTIAL_ELF) $(PARTIAL_DUMP) $(LD_ORIGINS)
FINAL_OUTPUTS   = $(ELFS) $(DUMPS) $(DWARFS) $(BINS)

#########
# Rules #
#########
.PHONY: $(DEVICE_DIR)/app_list.tmp
$(DEVICE_DIR)/app_list.tmp:
	echo "$(DEVICE_APPS)" > $(DEVICE_DIR)/app_list.tmp

.PHONY: partial-build
partial-build: $(PARTIAL_OUTPUTS)

.PHONY: finalize-build
finalize-build: $(FINAL_OUTPUTS)

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)
	rm -f $(ORIGIN_LD)

$(BUILDDIR):
	mkdir -p $@

$(DEVICE_BUILDDIRS):
	mkdir -p $@

$(DEP): $(SRCS) | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -MM -MT '$(PARTIAL_ELF)' $< > $@
	for elf in $(ELFS); do \
		$(RISCV_CC) $(RISCV_CFLAGS) -MM -MT '$$elf' $< >> $@; \
	done

# Partially linked object
$(PARTIAL_ELF): $(DEP) $(LD_SRCS) | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) $(SRCS) $(RISCV_LDFLAGS)  -o $@

$(PARTIAL_DUMP): $(PARTIAL_ELF) | $(BUILDDIR)
	$(RISCV_OBJDUMP) -D $< > $@

# Device object relocation address
.PHONY: $(DEVICE_DIR)/%/build/origin.ld
$(DEVICE_DIR)/%/build/origin.ld: $(PARTIAL_ELF) | $(DEVICE_DIR)/%/build
	@RELOC_ADDR=$$($(RISCV_OBJDUMP) -t $< | grep snitch_main | cut -c9-16); \
	echo "Writing device object relocation address 0x$$RELOC_ADDR to $@"; \
	echo "L3_ORIGIN = 0x$$RELOC_ADDR;" > $@

# Generates a rule which looks somewhat like:
#
# $(BUILDDIR)/$(APP)-%.elf: $(DEVICE_DIR)/apps/%/build/%.bin $(DEP) $(LD_SRCS) | $(BUILDDIR)
# 	$(RISCV_CC) $(RISCV_CFLAGS) -DDEVICEBIN=\"$<\" $(RISCV_LDFLAGS) $(SRCS) -o $@
#
# This approach is required cause you can't use multiple %-signs in a prerequisite
define elf_rule_template =
    $$(BUILDDIR)/$$(APP)-$(notdir $(1)).elf: $$(DEVICE_DIR)/$(1)/build/$(notdir $(1)).bin $$(DEP) $$(LD_SRCS) | $$(BUILDDIR)
		$$(RISCV_CC) $$(RISCV_CFLAGS) $$(SRCS) -DDEVICEBIN=\"$$<\" $$(RISCV_LDFLAGS)  -o $$@
endef
$(foreach f,$(DEVICE_APPS),$(eval $(call elf_rule_template,$(f))))

$(BUILDDIR)/$(APP)-%.dump: $(BUILDDIR)/$(APP)-%.elf | $(BUILDDIR)
	$(RISCV_OBJDUMP) -D $< > $@

$(BUILDDIR)/$(APP)-%.dwarf: $(BUILDDIR)/$(APP)-%.elf | $(BUILDDIR)
	$(RISCV_READELF) --debug-dump $< > $@

$(BUILDDIR)/$(APP)-%.bin: $(BUILDDIR)/$(APP)-%.elf | $(BUILDDIR)
	$(RISCV_OBJCOPY) -O binary $< $@
ifneq ($(MAKECMDGOALS),clean)
-include $(DEP)
endif