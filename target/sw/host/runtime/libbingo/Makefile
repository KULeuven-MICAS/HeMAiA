# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
###################
# PATH            #
###################
MK_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
SW_DIR  = $(abspath $(MK_DIR)/../../../)
include ../common.mk
###################
# Build variables #
###################
INCDIRS      += include/
INCDIRS      += $(SW_DIR)/host/runtime/
INCDIRS      += $(SW_DIR)/shared/vendor/o1heap/o1heap64
# INCDIRS      += $(SW_DIR)/host/runtime/o1heap32
INCDIRS      += $(SW_DIR)/shared/runtime # for print/mailbox/chipid
INCDIRS      += $(SW_DIR)/shared/platform/generated # for occamy_memory_map
RISCV_CFLAGS += $(addprefix -I,$(INCDIRS))

SRCS           = src/bingo_api.c
OBJS           = $(BUILD_DIR)/o1heap64.o
# OBJS		  += $(BUILD_DIR)/o1heap32.o
OBJS          += $(BUILD_DIR)/bingo_api.o
BINGO_LIB      = $(BUILD_DIR)/libbingo.a
BINGO_LIB_DUMP = $(BUILD_DIR)/libbingo.dump
ALL_OUTPUTS   = $(OBJS) $(BINGO_LIB) $(BINGO_LIB_DUMP)
.PHONY: gen-lib 
gen-lib: $(ALL_OUTPUTS)

$(BUILD_DIR)/o1heap64.o: $(SW_DIR)/shared/vendor/o1heap/o1heap64/o1heap64.c | $(BUILD_DIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -c $< -o $@

# $(BUILD_DIR)/o1heap32.o: $(SW_DIR)/host/runtime/o1heap32/o1heap32.c | $(BUILD_DIR)
# 	$(RISCV_CC) $(RISCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/bingo_api.o: src/bingo_api.c | $(BUILD_DIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -c $< -o $@

$(BINGO_LIB): $(OBJS) | $(BUILD_DIR)
	$(RISCV_AR) rvs -o $@ $^

$(BINGO_LIB_DUMP): $(BINGO_LIB)
	$(RISCV_OBJDUMP) -D $< > $@
