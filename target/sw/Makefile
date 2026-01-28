# Copyright 2025 KU Leuven
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yunhao Deng <yunhao.deng@kuleuven.be>
# Fanchen Kong <fanchen.kong@kuleuven.be>
# <Try not to blindly include other makefiles>
# This makefile generates the SW sources for HeMAiA SoC
.PHONY: sw clean-sw single-sw
############
# Makefrag #
############
BENDER                ?= bender
MKFILE_PATH           := $(abspath $(lastword $(MAKEFILE_LIST)))
SIM_MKFILE_DIR        := $(dir $(MKFILE_PATH))

#########################
# Files and directories #
#########################
ROOT                  = $(abspath $(SIM_MKFILE_DIR)../..)
SNITCH_ROOT           = $(shell $(BENDER) path snitch_cluster)
SW_DIR                = $(ROOT)/target/sw
TARGET_SRC_DIR        = $(SW_DIR)/shared/platform/template
TARGET_CLINT_DIR      = $(TARGET_SRC_DIR)/clint
TARGET_PLIC_DIR       = $(TARGET_SRC_DIR)/rv_plic
TARGET_SOCCTRL_DIR	  = $(TARGET_SRC_DIR)/soc_ctrl
TARGET_QUADPERIPH_DIR = $(TARGET_SRC_DIR)/quad_periph
PLATFORM_HEADERS_DIR  = shared/platform/generated
PLATFORM_HEADERS_TEMPLATE_DIR = shared/platform/template
SW_DEVICE_RT_DIR      = device/runtime
HOST_RUNTIME_DIR      = $(SW_DIR)/host/runtime
LIB_BINGO_DIR         = $(HOST_RUNTIME_DIR)/libbingo

############
# Programs #
############
REGGEN     			  = $(shell $(BENDER) path register_interface)/vendor/lowrisc_opentitan/util/regtool.py
OCCAMYGEN  			  = $(ROOT)/util/occamygen/occamygen.py

#######################
# Config prerequisite #
#######################
CFG				  	  ?= $(ROOT)/target/rtl/cfg/lru.hjson
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
PLATFORM_HEADERS += $(PLATFORM_HEADERS_DIR)/rv_plic.h
PLATFORM_HEADERS += $(PLATFORM_HEADERS_DIR)/occamy_soc_ctrl.h
PLATFORM_HEADERS += $(PLATFORM_HEADERS_DIR)/occamy_quad_periph.h
PLATFORM_HEADERS += $(PLATFORM_HEADERS_DIR)/snitch_cluster_peripheral.h
PLATFORM_HEADERS += $(PLATFORM_HEADERS_DIR)/hemaia_d2d_link_peripheral.h
PLATFORM_HEADERS += $(PLATFORM_HEADERS_DIR)/hemaia_clk_rst_controller_peripheral.h
# OCCAMYGEN headers
PLATFORM_HEADERS += $(PLATFORM_HEADERS_DIR)/occamy_memory_map.h
PLATFORM_HEADERS += $(PLATFORM_HEADERS_DIR)/occamy.h
PLATFORM_HEADERS += $(PLATFORM_HEADERS_DIR)/occamy_base_addr.h
PLATFORM_HEADERS += $(SW_DEVICE_RT_DIR)/generated/occamy_defs.h

###########
# 1.CLINT #
###########
$(TARGET_CLINT_DIR):
	mkdir -p $@
SOURCE_CLINT_DIR  = $(shell $(BENDER) path clint)
# CLINT needs the hjson tmp first to generate the hjson
$(TARGET_CLINT_DIR)/clint.hjson: $(SOURCE_CLINT_DIR)/data/clint.hjson.tpl $(CFG)
	@echo "[OCCAMYGEN] Generate $@"
	@$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_CLINT_DIR) --chip $<
# Then the hjson can be used to generate the header file
$(PLATFORM_HEADERS_DIR)/clint.h: $(TARGET_CLINT_DIR)/clint.hjson
	$(call reggen_generate_header,$@,$<)

###########
# 2.PLIC  #
###########
$(TARGET_PLIC_DIR):
	mkdir -p $@
SOURCE_PLIC_DIR       = $(ROOT)/hw/vendor/pulp_platform_opentitan_peripherals/src/rv_plic
PLICGEN               = $(SOURCE_PLIC_DIR)/util/reg_rv_plic.py
$(TARGET_PLIC_DIR)/rv_plic.hjson: $(SOURCE_PLIC_DIR)/data/rv_plic.hjson.tpl $(CFG) | $(TARGET_PLIC_DIR)
	@echo "[PLICGEN] Generate $@"
	@$(PLICGEN) -s 73 -t 2 -p 6 $< > $@

# Currently there are no sw related to the the PLIC
$(PLATFORM_HEADERS_DIR)/rv_plic.h: $(TARGET_PLIC_DIR)/rv_plic.hjson
	$(call reggen_generate_header,$@,$<)

##############
# 3.SoC Ctrl #
##############
$(TARGET_SOCCTRL_DIR):
	mkdir -p $@
# SoC Ctrl needs the hjson tmp first to generate the hjson
SOURCE_SOCCTRL_DIR = $(ROOT)/hw/occamy/soc_ctrl
$(TARGET_SOCCTRL_DIR)/occamy_soc_reg.hjson: $(SOURCE_SOCCTRL_DIR)/occamy_soc_reg.hjson.tpl $(CFG)
	@echo "[OCCAMYGEN] Generate $@"
	@$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_SOCCTRL_DIR) --ctrl $<
# Then the hjson can be used to generate the header file
$(PLATFORM_HEADERS_DIR)/occamy_soc_ctrl.h: $(TARGET_SOCCTRL_DIR)/occamy_soc_reg.hjson
	$(call reggen_generate_header,$@,$<)

###############
# 4.Quad Ctrl #
###############
$(TARGET_QUADPERIPH_DIR):
	mkdir -p $@
# SoC Ctrl needs the hjson tmp first to generate the hjson
SOURCE_QUADPERIPH_DIR = $(ROOT)/hw/occamy/quad_periph
$(TARGET_QUADPERIPH_DIR)/occamy_quad_periph_reg.hjson: $(SOURCE_QUADPERIPH_DIR)/occamy_quad_periph_reg.hjson.tpl $(CFG)
	@echo "[OCCAMYGEN] Generate $@"
	@$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_QUADPERIPH_DIR) --ctrl $<
# Then the hjson can be used to generate the header file
$(PLATFORM_HEADERS_DIR)/occamy_quad_periph.h: $(TARGET_QUADPERIPH_DIR)/occamy_quad_periph_reg.hjson
	$(call reggen_generate_header,$@,$<)
######################
# Cluster Peripheral #
######################

$(PLATFORM_HEADERS_DIR)/snitch_cluster_peripheral.h: $(SNITCH_ROOT)/hw/snitch_cluster/src/snitch_cluster_peripheral/snitch_cluster_peripheral_reg.hjson
	$(call reggen_generate_header,$@,$<)

########################
# HeMAiA D2D Link Ctrl #
########################

$(PLATFORM_HEADERS_DIR)/hemaia_d2d_link_peripheral.h: $(shell bender path hemaia_d2d_link)/src/ctrl/hemaia_d2d_link_reg.hjson
	$(call reggen_generate_header,$@,$<)

#######################
# HeMAiA Clk Rst Ctrl #
#######################

$(PLATFORM_HEADERS_DIR)/hemaia_clk_rst_controller_peripheral.h: $(ROOT)/hw/hemaia/hemaia_clk_rst_controller/regs/hemaia_clk_rst_controller_reg.hjson
	$(call reggen_generate_header,$@,$<)

# OCCAMYGEN headers
$(PLATFORM_HEADERS_DIR)/occamy_memory_map.h: $(PLATFORM_HEADERS_TEMPLATE_DIR)/occamy_memory_map.h.tpl $(CFG)
	@echo "[OCCAMYGEN] Generating $@"
	@$(OCCAMYGEN) -c $(CFG) --outdir $(PLATFORM_HEADERS_DIR) --cheader $<

$(PLATFORM_HEADERS_DIR)/occamy.h: $(PLATFORM_HEADERS_TEMPLATE_DIR)/occamy.h.tpl $(CFG)
	@echo "[OCCAMYGEN] Generating $@"
	@$(OCCAMYGEN) -c $(CFG) --outdir $(PLATFORM_HEADERS_DIR) --cheader $<

$(PLATFORM_HEADERS_DIR)/occamy_base_addr.h: $(CFG)
	@echo "[OCCAMYGEN] Generating $@"
	@$(OCCAMYGEN) -c $(CFG) --outdir $(PLATFORM_HEADERS_DIR) -D $@
	
# OCCAMYGEN Device runtime
$(SW_DEVICE_RT_DIR)/generated/occamy_defs.h: $(SW_DEVICE_RT_DIR)/template/occamy_defs.h.tpl $(CFG)
	@echo "[OCCAMYGEN] Generating $@"
	@$(OCCAMYGEN) -c $(CFG) --outdir $(SW_DEVICE_RT_DIR)/generated --cheader $<

host/runtime/host.ld:
	@echo "[OCCAMYGEN] Generating host.ld"
	@$(OCCAMYGEN) -c $(CFG) --gen-host-ld --outdir host/runtime

debug-platform-headers: $(PLATFORM_HEADERS)
	@echo "Generated Platform Headers:"
	@ls -lh $(PLATFORM_HEADERS)


sw: $(PLATFORM_HEADERS) host/runtime/host.ld
	$(MAKE) -C host libs
	$(MAKE) -C host TARGET=partial-build
	$(MAKE) -C device all
	$(MAKE) -C host TARGET=finalize-build

single-sw: $(PLATFORM_HEADERS) host/runtime/host.ld
	$(MAKE) -C host libs
	$(MAKE) -C host single_sw TARGET=partial-build \
		HOST_APP_TYPE=$(HOST_APP_TYPE) \
		CHIP_TYPE=$(CHIP_TYPE) \
		WORKLOAD=$(WORKLOAD) \
		DEV_APP=$(DEV_APP)
	$(MAKE) -C device all DEVICE_APPS=$(DEV_APP)
	$(MAKE) -C host single_sw TARGET=finalize-build \
		HOST_APP_TYPE=$(HOST_APP_TYPE) \
		CHIP_TYPE=$(CHIP_TYPE) \
		WORKLOAD=$(WORKLOAD) \
		DEV_APP=$(DEV_APP)

clean:
# clean device build
	$(MAKE) -C device all TARGET=clean
# clean host build
	$(MAKE) -C host TARGET=clean
# clean host runtimes
	$(MAKE) -C $(LIB_BINGO_DIR) clean
# clean generated tmp files
	rm -rf $(SW_DIR)/device/*.tmp
# clean generated headers
	rm -f $(PLATFORM_HEADERS)
# clean generated hjson files
	rm -rf $(TARGET_CLINT_DIR) $(TARGET_SOCCTRL_DIR) $(TARGET_QUADPERIPH_DIR) $(TARGET_PLIC_DIR)
	rm -rf host/runtime/host.ld
