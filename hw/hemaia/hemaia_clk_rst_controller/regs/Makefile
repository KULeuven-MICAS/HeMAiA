# Copyright 2025 KU Leuven.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Yunhao Deng <yunhao.deng@kuleuven.be>
MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))


BENDER   = bender
REGTOOL := $(shell $(BENDER) path register_interface)/vendor/lowrisc_opentitan/util/regtool.py

hemaia_clk_rst_ctrl_reg_top.sv: $(REGTOOL) hemaia_clk_rst_controller_reg.hjson
	$(REGTOOL) -r -t $(MKFILE_DIR) $(shell realpath hemaia_clk_rst_controller_reg.hjson)

.PHONY: all
all: hemaia_clk_rst_ctrl_reg_top.sv

.DEFAULT_GOAL := all
