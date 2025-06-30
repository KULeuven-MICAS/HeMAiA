# Copyright 2024 KU Leuven
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yunhao Deng <yunhao.deng@kuleuven.be>
# Fanchen Kong <fanchen.kong@kuleuven.be>

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
COMMON_DIR  := $(MKFILE_DIR)../shared
include $(COMMON_DIR)/common.mk

.PHONY: bootrom clean

bootrom: bootrom.elf bootrom.dump bootrom.bin bootrom.sv

bootrom.elf: bootrom.S
	$(CC) $(CFLAGS) $(INCLUDES) -T$(COMMON_DIR)/bootrom.ld $< -o bootrom.elf

bootrom.dump: bootrom.elf
	$(OBJDUMP) -d bootrom.elf > bootrom.dump

bootrom.bin: bootrom.elf
	$(OBJCOPY) -j .text -j .rodata -j .srodata -O binary bootrom.elf bootrom.bin

bootrom.sv: bootrom.bin
	$(BIN2SV) $< -m bootrom > $@

clean:
	rm -rf *.bin *.dump *.elf *.o