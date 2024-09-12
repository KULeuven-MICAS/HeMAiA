# Copyright 2024 KU Leuven
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yunhao Deng <yunhao.deng@kuleuven.be>

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
ROOT        := ${MKFILE_DIR}../../..
util				:= $(ROOT)/util

include $(ROOT)/util/Makefrag

INCLUDES = -I./ -I./src
SRCS_C = src/main.c 
OBJS_C = $(SRCS_C:.c=.o)

CVA6_GCC_ROOT = /tools/riscv/bin
CC = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-gcc
OBJDUMP = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-objdump
OBJCOPY = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-objcopy
BIN2SV  = $(util)/bin2sv.py

CFLAGS 	= -Os -g -Werror -ffreestanding -fno-strict-aliasing
CFLAGS += -static -nostartfiles -pie
CFLAGS += -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-stack-protector -fPIE
CFLAGS += -mno-save-restore -mstrict-align
CFLAGS += -mabi=lp64d -march=rv64imafd
CFLAGS += -mcmodel=medany

.PHONY: bootrom clean
bootrom: bootrom.sv

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
	@echo "CC    <= $<"

bootrom.elf bootrom.dump bootrom.bin: bootrom.S src/bootrom.o bootrom.ld
	$(CC) $(CFLAGS) $(INCLUDES) -Tbootrom.ld $< src/bootrom.o -o bootrom.elf
	$(OBJDUMP) -d bootrom.elf > bootrom.dump
	$(OBJCOPY) -j .text -j .rodata -j .srodata -O binary bootrom.elf bootrom.bin

bootrom.sv: bootrom.bin
	$(BIN2SV) "$(MKFILE_DIR)$<" -m bootrom > bootrom.sv

clean:
	rm -rf *.bin *.dump *.elf src/*.o
