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

HOST_APP_TYPE ?=
CHIP_TYPE     ?=
WORKLOAD      ?=
DEV_APP       ?=

# Mapping logic for different application categories to determine binary names and paths
ifeq ($(HOST_APP_TYPE), host_only)
    # E.g., target/sw/host/apps/host_only/single_chip/hello_world/build/hello_world.elf
    APP_NAME = $(WORKLOAD)
    APP_REL_PATH = host_only/$(CHIP_TYPE)/$(WORKLOAD)
else ifeq ($(HOST_APP_TYPE), offload_legacy)
    # E.g., target/sw/host/apps/offload_legacy/single_chip/build/offload_legacy_single_chip_snax-test-integration.elf
    APP_NAME = offload_legacy_$(CHIP_TYPE)_$(DEV_APP)
    APP_REL_PATH = offload_legacy/$(CHIP_TYPE)
else
    # E.g., target/sw/host/apps/offload_bingo_sw/single_chip/workloads/gemm_tiled/build/offload_bingo_sw_single_chip_gemm_tiled_snax-bingo-offload.elf
    APP_NAME = $(HOST_APP_TYPE)_$(CHIP_TYPE)_$(WORKLOAD)_$(DEV_APP)
    APP_REL_PATH = $(HOST_APP_TYPE)/$(CHIP_TYPE)/workloads/$(WORKLOAD)
endif

APP_ELF  = $(SW_HOST_APPS_DIR)$(APP_REL_PATH)/build/$(APP_NAME).elf
APP_BIN  = $(SW_HOST_APPS_DIR)$(APP_REL_PATH)/build/$(APP_NAME).bin
APP_DUMP = $(SW_HOST_APPS_DIR)$(APP_REL_PATH)/build/$(APP_NAME).dump
APP_MEM  = $(SW_HOST_APPS_DIR)$(APP_REL_PATH)/build/mempool.bin

.PHONY: apps hex clean-apps

$(APP_ELF):
	$(MAKE) -C $(MKFILE_DIR)../../sw single-sw CFG=$(CFG) \
		HOST_APP_TYPE=$(HOST_APP_TYPE) \
		CHIP_TYPE=$(CHIP_TYPE) \
		WORKLOAD=$(WORKLOAD) \
		DEV_APP=$(DEV_APP)

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
