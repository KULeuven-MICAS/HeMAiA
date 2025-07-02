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

APP  = offload
SRCS = src/offload.c
# DEVICE_APPS += $(shell cd ../../../device/apps/;find -mindepth 2 -maxdepth 2 -type d | sed 's|^\./||')

# Compiler toolchain
CVA6_GCC_ROOT = /tools/riscv/bin
RISCV_CC      = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-gcc
RISCV_OBJCOPY = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-objcopy
RISCV_OBJDUMP = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-objdump
RISCV_READELF = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-readelf

# Directories
BUILDDIR    = $(abspath build)
HOST_DIR    = $(abspath ../../)
RUNTIME_DIR = $(abspath $(HOST_DIR)/runtime)
DEVICE_DIR  = $(abspath $(HOST_DIR)/../device)

# now we only include the snax app
DEVICE_APPS += snax/snax-gemmx-matmul
DEVICE_APPS += snax/snax-gemmx-matmul-tiling
DEVICE_APPS += snax/snax-gemmx-conv
DEVICE_APPS += snax/snax-test-integration
DEVICE_APPS += snax/snax-hypercorex-test-csr
DEVICE_APPS += snax/snax-hypercorex-char-recog
DEVICE_APPS += snax/snax-xdma-maxpool
DEVICE_APPS += snax/snax-xdma-memset
DEVICE_APPS += snax/snax-cgra-run
DEVICE_APPS += snax/snax-dimc-mha
DEVICE_APPS += snax/snax-dimc-conv
DEVICE_APPS += snax/snax-gemmx-dimc-cosim

# Dependencies
INCDIRS += $(RUNTIME_DIR)
INCDIRS += $(HOST_DIR)/../shared/platform/generated
INCDIRS += $(HOST_DIR)/../shared/runtime
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
LD_SRCS       = $(LINKER_SCRIPT)

# Linker flags
RISCV_LDFLAGS += -nostartfiles
RISCV_LDFLAGS += -lm
RISCV_LDFLAGS += -lgcc
RISCV_LDFLAGS += -T$(LINKER_SCRIPT)

# Device binaries
DEVICE_BUILDDIRS = $(addsuffix /build, $(addprefix $(DEVICE_DIR)/apps/, $(DEVICE_APPS)))

###########
# Outputs #
###########

PARTIAL_ELF     = $(abspath $(BUILDDIR)/$(APP).part.elf)
DEP             = $(abspath $(BUILDDIR)/$(APP).d)
PARTIAL_DUMP    = $(abspath $(BUILDDIR)/$(APP).part.dump)
LD_ORIGINS      = $(abspath $(addsuffix /origin.ld, $(DEVICE_BUILDDIRS)))
ELFS            = $(abspath $(addsuffix .elf, $(addprefix $(BUILDDIR)/$(APP)-, $(notdir $(DEVICE_APPS)))))
DUMPS           = $(abspath $(addsuffix .dump, $(addprefix $(BUILDDIR)/$(APP)-, $(notdir $(DEVICE_APPS)))))
DWARFS          = $(abspath $(addsuffix .dwarf, $(addprefix $(BUILDDIR)/$(APP)-, $(notdir $(DEVICE_APPS)))))
BINS           = $(abspath $(addsuffix .bin, $(addprefix $(BUILDDIR)/$(APP)-, $(notdir $(DEVICE_APPS)))))
PARTIAL_OUTPUTS = $(PARTIAL_ELF) $(PARTIAL_DUMP) $(LD_ORIGINS)
FINAL_OUTPUTS   = $(ELFS) $(DUMPS) $(DWARFS) $(BINS)

#########
# Rules #
#########

.PHONY: partial-build
partial-build: $(PARTIAL_OUTPUTS)

.PHONY: finalize-build
finalize-build: $(FINAL_OUTPUTS)

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)
	rm -f $(OFFSET_LD)

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
	$(RISCV_CC) $(RISCV_CFLAGS) $(RISCV_LDFLAGS) $(SRCS) -o $@

$(PARTIAL_DUMP): $(PARTIAL_ELF) | $(BUILDDIR)
	$(RISCV_OBJDUMP) -D $< > $@

# Device object relocation address
$(DEVICE_DIR)/apps/%/build/origin.ld: $(PARTIAL_ELF) | $(DEVICE_DIR)/apps/%/build
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
    $$(BUILDDIR)/$$(APP)-$(notdir $(1)).elf: $$(DEVICE_DIR)/apps/$(1)/build/$(notdir $(1)).bin $$(DEP) $$(LD_SRCS) | $$(BUILDDIR)
		$$(RISCV_CC) $$(RISCV_CFLAGS) -DDEVICEBIN=\"$$<\" $$(RISCV_LDFLAGS) $$(SRCS) -o $$@
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
