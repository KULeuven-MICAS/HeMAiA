# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

.PHONY: sw clean-sw

############
# Makefrag #
############

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
SIM_MKFILE_DIR  := $(dir $(MKFILE_PATH))
ROOT        := $(SIM_MKFILE_DIR)../..
SW_DIR      := $(abspath $(ROOT)/target/sw)
include $(ROOT)/target/rtl/Makefile
############
# Programs #
############
REGGEN     ?= $(shell $(BENDER) path register_interface)/vendor/lowrisc_opentitan/util/regtool.py
OCCAMYGEN  ?= $(ROOT)/util/occamygen/occamygen.py

#########################
# Files and directories #
#########################
PLATFORM_HEADERS_DIR ?= shared/platform/generated
PLATFORM_HEADERS_TEMPLATE_DIR ?= shared/platform/template
SW_DEVICE_RT_DIR     ?= device/runtime

###################
# Software Headers#
###################

# Common rule to generate C header with REGGEN
# $1: target name, $2: prerequisite (hjson description file)
define reggen_generate_header
	@echo "[REGGEN] Generating $1"
	@$(REGGEN) -D -o $1 $2
endef

# REGGEN headers
PLATFORM_HEADERS  = $(PLATFORM_HEADERS_DIR)/clint.h
PLATFORM_HEADERS += $(PLATFORM_HEADERS_DIR)/occamy_soc_ctrl.h
PLATFORM_HEADERS += $(PLATFORM_HEADERS_DIR)/snitch_cluster_peripheral.h
PLATFORM_HEADERS += $(PLATFORM_HEADERS_DIR)/snitch_quad_peripheral.h
PLATFORM_HEADERS += $(PLATFORM_HEADERS_DIR)/hemaia_d2d_link_peripheral.h
PLATFORM_HEADERS += $(PLATFORM_HEADERS_DIR)/hemaia_clk_rst_controller_peripheral.h
# OCCAMYGEN headers
PLATFORM_HEADERS += $(PLATFORM_HEADERS_DIR)/occamy_cfg.h
PLATFORM_HEADERS += $(PLATFORM_HEADERS_DIR)/occamy_memory_map.h
PLATFORM_HEADERS += $(PLATFORM_HEADERS_DIR)/occamy.h
PLATFORM_HEADERS += $(PLATFORM_HEADERS_DIR)/occamy_base_addr.h
PLATFORM_HEADERS += $(SW_DEVICE_RT_DIR)/src/occamy_defs.h

# REGGEN headers
$(PLATFORM_HEADERS_DIR)/clint.h: $(TARGET_SRC_DIR)/clint/clint.hjson
	$(call reggen_generate_header,$@,$<)
	
$(PLATFORM_HEADERS_DIR)/occamy_soc_ctrl.h: $(TARGET_SRC_DIR)/soc_ctrl/occamy_soc_reg.hjson
	$(call reggen_generate_header,$@,$<)

$(PLATFORM_HEADERS_DIR)/snitch_cluster_peripheral.h: $(SNITCH_ROOT)/hw/snitch_cluster/src/snitch_cluster_peripheral/snitch_cluster_peripheral_reg.hjson
	$(call reggen_generate_header,$@,$<)

$(PLATFORM_HEADERS_DIR)/snitch_quad_peripheral.h: $(TARGET_SRC_DIR)/quadrant_s1_ctrl/occamy_quadrant_s1_reg.hjson
	$(call reggen_generate_header,$@,$<)

$(PLATFORM_HEADERS_DIR)/hemaia_d2d_link_peripheral.h: $(shell bender path hemaia_d2d_link)/src/ctrl/hemaia_d2d_link_reg.hjson
	$(call reggen_generate_header,$@,$<)

$(PLATFORM_HEADERS_DIR)/hemaia_clk_rst_controller_peripheral.h: $(ROOT)/hw/hemaia/hemaia_clk_rst_controller/regs/hemaia_clk_rst_controller_reg.hjson
	$(call reggen_generate_header,$@,$<)


# OCCAMYGEN headers
$(PLATFORM_HEADERS_DIR)/occamy_cfg.h: $(PLATFORM_HEADERS_TEMPLATE_DIR)/occamy_cfg.h.tpl $(CFG)
	@echo "[OCCAMYGEN] Generating $@"
	@$(OCCAMYGEN) -c $(CFG) --outdir $(PLATFORM_HEADERS_DIR) --cheader $<

$(PLATFORM_HEADERS_DIR)/occamy_memory_map.h: $(PLATFORM_HEADERS_TEMPLATE_DIR)/occamy_memory_map.h.tpl $(CFG)
	@echo "[OCCAMYGEN] Generating $@"
	@$(OCCAMYGEN) -c $(CFG) --outdir $(PLATFORM_HEADERS_DIR) --cheader $<

$(PLATFORM_HEADERS_DIR)/occamy.h: $(PLATFORM_HEADERS_TEMPLATE_DIR)/occamy.h.tpl $(CFG)
	@echo "[OCCAMYGEN] Generating $@"
	@$(OCCAMYGEN) -c $(CFG) --outdir $(PLATFORM_HEADERS_DIR) --cheader $<

$(PLATFORM_HEADERS_DIR)/occamy_base_addr.h: $(CFG)
	@echo "[OCCAMYGEN] Generating $@"
	@$(OCCAMYGEN) -c $(CFG) --outdir $(PLATFORM_HEADERS_DIR) -D $@

$(SW_DEVICE_RT_DIR)/src/occamy_defs.h: $(SW_DEVICE_RT_DIR)/template/occamy_defs.h.tpl $(CFG)
	@echo "[OCCAMYGEN] Generating $@"
	@$(OCCAMYGEN) -c $(CFG) --outdir $(SW_DEVICE_RT_DIR)/src --cheader $<

sw: $(PLATFORM_HEADERS)
	$(MAKE) -C host libs
	$(MAKE) -C host TARGET=partial-build
	$(MAKE) -C device all
	$(MAKE) -C host TARGET=finalize-build

clean-sw:
# clean generated tmp files
	rm -rf $(SW_DIR)/device/*.tmp
# clean device build
	$(MAKE) -C device TARGET=clean
# clean host build
	$(MAKE) -C host TARGET=clean
