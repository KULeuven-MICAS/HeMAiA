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

.PHONY: apps hex clean

apps: $(BINS)
	python3 bin2hex.py

# Use full paths for the dependencies
%.bin: %.elf
	$(RISCV_OBJCOPY) -O binary $< $@
	mv $@ $(MKFILE_DIR)

clean:
	@rm -f $(MKFILE_DIR)/*.bin $(MKFILE_DIR)/*.hex
	@find $(MKFILE_DIR) -mindepth 1 -type d -exec rm -rf {} +
