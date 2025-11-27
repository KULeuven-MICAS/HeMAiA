# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# There are three main stages of building a host application:
# 1) host-partial-build: compile and link the host application without the device binary. This is mainly to determine the relocation
#    address of the device binary, which is needed to create the origin.ld file.
# 2) device-build: based on the generated origin.ld file, build the device application to get the device binary.
# 3) host-finalize-build: compile and link the host application with the device binary

######################
# Invocation options #
######################

DEBUG ?= OFF # ON to turn on debugging symbols

###################
# Build variables #
###################

# Compiler toolchain
CVA6_GCC_ROOT = /tools/riscv/bin
RISCV_CC      = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-gcc
RISCV_OBJCOPY = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-objcopy
RISCV_OBJDUMP = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-objdump
RISCV_READELF = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-readelf
# Directories
BUILDDIR    = $(abspath build)
HOST_DIR    = $(abspath ../../)
SWDIR       = $(abspath ../../../)
RUNTIME_DIR = $(abspath $(HOST_DIR)/runtime)
DEVICE_DIR  = $(abspath $(HOST_DIR)/../device)

# Dependencies
INCDIRS += $(RUNTIME_DIR)
INCDIRS += $(HOST_DIR)/../shared/platform/generated
INCDIRS += $(HOST_DIR)/../shared/runtime
SRCS    += $(RUNTIME_DIR)/start.S

# Include XDMA
INCDIRS += $(SWDIR)/shared/vendor/xdma

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
ifeq ($(DEBUG),ON)
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


# if the host application uses the bingo runtime
ifneq (,$(filter $(BINGO_HOST),True true TRUE 1))
LIBBINGO_DIR = $(abspath $(RUNTIME_DIR)/libbingo)


# libbingo
INCDIRS += $(LIBBINGO_DIR)/include
INCDIRS += $(SWDIR)/shared/vendor/o1heap/o1heap64
# INCDIRS += $(SWDIR)/host/runtime/o1heap32
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



.PHONY: partial-build
partial-build: $(PARTIAL_OUTPUTS)

.PHONY: finalize-build
finalize-build: $(FINAL_OUTPUTS)

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)
	rm -rf $(PARTIAL_TMPS)

$(BUILDDIR):
	mkdir -p $@

$(DEVICE_BUILDDIR):
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