# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# The Libhero will be compiled to a .so



###################
# PATH            #
###################
MK_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
SW_DIR  = $(abspath $(MK_DIR)/../../../)
include ../common.mk
###################
# Build variables #
###################

INCDIRS       = src/
INCDIRS      += include/
INCDIRS      += $(SW_DIR)/shared/vendor/o1heap/o1heap # for o1heap
INCDIRS      += $(SW_DIR)/shared/runtime # for uart print
INCDIRS      += $(SW_DIR)/shared/platform/generated # for occamy_memory_map
RISCV_CFLAGS += $(addprefix -I,$(INCDIRS))

SRCS          = src/snitch_cluster.c
SRCS         += src/hero_api.c
SRCS         += $(SW_DIR)/shared/vendor/o1heap/o1heap/o1heap.c
OBJS          = $(BUILD_DIR)/o1heap.o
OBJS         += $(BUILD_DIR)/hero_api.o
OBJS         += $(BUILD_DIR)/snitch_cluster.o
HERO_LIB      = $(BUILD_DIR)/libhero.a
HERO_LIB_DUMP = $(BUILD_DIR)/libhero.dump
ALL_OUTPUTS   = $(OBJS) $(HERO_LIB) $(HERO_LIB_DUMP)

.PHONY: gen-lib  
gen-lib: $(ALL_OUTPUTS)

$(BUILD_DIR)/o1heap.o: $(SW_DIR)/shared/vendor/o1heap/o1heap/o1heap.c | $(BUILD_DIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/hero_api.o: src/hero_api.c | $(BUILD_DIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/snitch_cluster.o: src/snitch_cluster.c | $(BUILD_DIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -c $< -o $@

$(HERO_LIB): $(OBJS) | $(BUILD_DIR)
	$(RISCV_AR) rvs -o $@ $^

$(HERO_LIB_DUMP): $(HERO_LIB)
	$(RISCV_OBJDUMP) -D $< > $@

