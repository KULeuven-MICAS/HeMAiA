MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
TARGET 		:= $(MKFILE_DIR)../..
ROOT        := ${MKFILE_DIR}../../..

CVA6_GCC_ROOT = /tools/riscv/bin
RISCV_CC      = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-gcc
RISCV_OBJCOPY = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-objcopy
RISCV_OBJDUMP = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-objdump
RISCV_READELF = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-readelf

# Find the ELF files with full paths
ELFS = $(shell find $(TARGET)/sim/sw/host/apps -type f -name "*.elf")
# Generate BIN filenames from the ELF filenames
BINS = $(ELFS:.elf=.bin)

.PHONY: apps copy_apps clean

copy_apps:
	@cp $(BINS) $(MKFILE_DIR)

apps: $(BINS) copy_apps

# Use full paths for the dependencies
%.bin: %.elf
	$(RISCV_OBJCOPY) -O binary $< $@

clean:
	@rm -f $(MKFILE_DIR)/*.bin
	@find $(MKFILE_DIR) -mindepth 1 -type d -exec rm -rf {} +
