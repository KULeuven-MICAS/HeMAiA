MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
TARGET 		:= $(MKFILE_DIR)../../..
ROOT        := ${MKFILE_DIR}../../../..

APP			?= $(TARGET)/sim/sw/host/apps/hello_world/build/hello_world.elf

CVA6_GCC_ROOT = /tools/riscv/bin
RISCV_CC      = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-gcc
RISCV_OBJCOPY = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-objcopy
RISCV_OBJDUMP = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-objdump
RISCV_READELF = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-readelf

.PHONY: download_sw

app.bin: 
	$(RISCV_OBJCOPY) -O binary $(APP) $@

download_sw: 
	${VIVADO} -mode batch -source download_sw.tcl

clean:
	@rm -f app.bin
