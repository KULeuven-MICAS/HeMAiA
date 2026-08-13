# Copyright 2024 KU Leuven
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yunhao Deng <yunhao.deng@kuleuven.be>
# Fanchen Kong <fanchen.kong@kuleuven.be>

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
ROOT        := ${MKFILE_DIR}../../../..
util		:= $(ROOT)/util
# CVA6_GCC_ROOT = /opt/riscv/
CVA6_GCC_ROOT = 
CC = $(CVA6_GCC_ROOT)riscv64-unknown-elf-gcc
OBJDUMP = $(CVA6_GCC_ROOT)riscv64-unknown-elf-objdump
OBJCOPY = $(CVA6_GCC_ROOT)riscv64-unknown-elf-objcopy
BIN2SV  = $(util)/bin2sv.py

# This file is included before the per-directory rules, so the first rule below
# would otherwise become make's default goal -- the parent Makefile invokes each
# boot ROM directory without naming a goal, and would then build only the linker
# script and stop. Pin the default explicitly.
.DEFAULT_GOAL := bootrom

# The linker script is generated: the SoC control scratch addresses the boot ROM
# jumps through are derived from the same generated headers the host software
# uses, so a register-map change cannot leave the two disagreeing. Both boot ROM
# variants link against the one copy in this shared directory.
PLATFORM_GEN   := $(ROOT)/target/sw/shared/platform/generated
BOOTROM_LD     := $(MKFILE_DIR)bootrom.ld
BOOTROM_LD_TPL := $(MKFILE_DIR)bootrom.ld.tpl

$(BOOTROM_LD): $(BOOTROM_LD_TPL) $(PLATFORM_GEN)/occamy_soc_ctrl.h $(PLATFORM_GEN)/occamy_base_addr.h $(PLATFORM_GEN)/occamy_memory_map.h
	@echo "GEN   <= $(notdir $<)"
	@python3 $(MKFILE_DIR)gen_bootrom_ld.py --tpl $< --headers $(PLATFORM_GEN) --out $@

CFLAGS 	= -Os -g -Werror -ffreestanding -fno-strict-aliasing
CFLAGS += -static -nostartfiles
CFLAGS += -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-stack-protector -fno-tree-vectorize
CFLAGS += -mno-save-restore -mstrict-align
CFLAGS += -mabi=lp64d -march=rv64imafd
CFLAGS += -mcmodel=medany
