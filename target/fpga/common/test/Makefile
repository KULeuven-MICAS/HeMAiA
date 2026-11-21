# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

VERILATOR ?= verilator
BUILD_DIR ?= build

.PHONY: test
test:
	$(VERILATOR) --binary --timing --timescale 1ns/1ps --assert --Mdir $(BUILD_DIR) \
		--top-module tb_hemaia_dram_address_mapper ../hemaia_dram_address_mapper.sv \
		tb_hemaia_dram_address_mapper.sv
	$(BUILD_DIR)/Vtb_hemaia_dram_address_mapper
