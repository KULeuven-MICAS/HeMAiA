# Copyright 2024 KU Leuven
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yunhao Deng <yunhao.deng@kuleuven.be>
# Fanchen Kong <fanchen.kong@kuleuven.be>
MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
SW_HOST_APPS_DIR 		:= $(MKFILE_DIR)../../sw/host/apps/
SW_DEVICE_APPS_DIR 	:= $(MKFILE_DIR)../../sw/device/apps/snax/
BIN_DIR := $(MKFILE_DIR)../bin

CVA6_GCC_ROOT = /tools/riscv/bin
RISCV_CC      = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-gcc
RISCV_OBJCOPY = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-objcopy
RISCV_OBJDUMP = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-objdump
RISCV_READELF = $(CVA6_GCC_ROOT)/riscv64-unknown-elf-readelf

HOST_APP ?=
DEVICE_APP ?= 
# If DEVICE_APP is not set, default to 'offload'
# If DEVICE_APP is set, the app name will be HOST_APP + "_" + DEVICE_APP
APP_NAME = $(if $(DEVICE_APP),$(HOST_APP)_$(DEVICE_APP),$(HOST_APP))
APP_ELF = $(SW_HOST_APPS_DIR)$(HOST_APP)/build/$(APP_NAME).elf
APP_BIN = $(SW_HOST_APPS_DIR)$(HOST_APP)/build/$(APP_NAME).bin
APP_DUMP = $(SW_HOST_APPS_DIR)$(HOST_APP)/build/$(APP_NAME).dump
APP_MEM = $(SW_HOST_APPS_DIR)$(HOST_APP)/build/mempool.bin

.PHONY: apps hex clean-apps

apps: $(APP_ELF)
	cp $(APP_ELF) $(MKFILE_DIR)
	echo "$(APP_ELF)" > $(MKFILE_DIR).rtlbinary
	cp $(APP_BIN) $(MKFILE_DIR)
	cp $(APP_DUMP) $(MKFILE_DIR)
	python3 bin2hex.py -i $(APP_NAME).bin
	python3 copy_n_times.py -i $(APP_NAME) -o $(BIN_DIR) -c $(CFG)
	if [ -f "$(APP_MEM)" ]; then \
		cp -f "$(APP_MEM)" $(MKFILE_DIR); \
		python3 bin2hex.py -i mempool.bin; \
		cp -r -f $(MKFILE_DIR)/mempool $(BIN_DIR); \
	fi
clean-apps:
	@rm -rf .rtlbinary
	@rm -f $(MKFILE_DIR)/*.bin $(MKFILE_DIR)/*.hex $(MKFILE_DIR)/*.elf $(MKFILE_DIR)/*.dump
	@find $(MKFILE_DIR) -mindepth 1 -type d -exec rm -rf {} +
