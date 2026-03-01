# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
###################
# Build variables #
###################
# For the runtime compilation, we only need a minimal set of flags
# Compiler toolchain
# CVA6_GCC_ROOT = /opt/riscv
CVA6_GCC_ROOT ?= 
RISCV_CC      = $(CVA6_GCC_ROOT)riscv64-unknown-elf-gcc
RISCV_AR      = $(CVA6_GCC_ROOT)riscv64-unknown-elf-ar
RISCV_OBJCOPY = $(CVA6_GCC_ROOT)riscv64-unknown-elf-objcopy
RISCV_OBJDUMP = $(CVA6_GCC_ROOT)riscv64-unknown-elf-objdump
RISCV_READELF = $(CVA6_GCC_ROOT)riscv64-unknown-elf-readelf

RISCV_CFLAGS += -march=rv64gcv # include the v extention for ara
RISCV_CFLAGS += -mabi=lp64d
RISCV_CFLAGS += -mcmodel=medany
RISCV_CFLAGS += -ffast-math
RISCV_CFLAGS += -fno-builtin-printf
RISCV_CFLAGS += -fno-tree-vectorize -fno-tree-loop-vectorize -fno-tree-slp-vectorize
RISCV_CFLAGS += -fno-common
RISCV_CFLAGS += -O2
RISCV_CFLAGS += -fPIC
RISCV_CFLAGS += -ffunction-sections
RISCV_CFLAGS += -Wextra
RISCV_CFLAGS += -Werror
RISCV_CFLAGS += -std=gnu99
RISCV_CFLAGS += $(USER_FLAGS)
BUILD_DIR     = build

$(BUILD_DIR):
	mkdir -p $@

.PHONY: clean 

clean:
	rm -rf $(BUILD_DIR)