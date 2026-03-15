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

CFLAGS 	= -Os -g -Werror -ffreestanding -fno-strict-aliasing
CFLAGS += -static -nostartfiles
CFLAGS += -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-stack-protector -fno-tree-vectorize
CFLAGS += -mno-save-restore -mstrict-align
CFLAGS += -mabi=lp64d -march=rv64imafd
CFLAGS += -mcmodel=medany
