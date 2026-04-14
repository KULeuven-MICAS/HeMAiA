# Copyright 2025 KU Leuven
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yunhao Deng <yunhao.deng@kuleuven.be>
# Fanchen Kong <fanchen.kong@kuleuven.be>
# <Try not to blindly include other makefiles>
# This makefile generates the SW sources for HeMAiA SoC
.PHONY: sw clean-sw single-sw snax-sw-gen clean-snax-sw-gen
############
# Makefrag #
############
BENDER                ?= bender
MKFILE_PATH           := $(abspath $(lastword $(MAKEFILE_LIST)))
SIM_MKFILE_DIR        := $(dir $(MKFILE_PATH))
IS_CLEAN_GOAL         := $(filter clean clean-%,$(MAKECMDGOALS))

#########################
# Files and directories #
#########################
ROOT                  = $(abspath $(SIM_MKFILE_DIR)../..)
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
OCCAMYGEN  			  = $(ROOT)/util/occamygen/occamygen.py

#######################
# Config prerequisite #
#######################
CFG				  	  ?= $(ROOT)/target/rtl/cfg/lru.hjson

ifeq ($(IS_CLEAN_GOAL),)
SNITCH_ROOT                 = $(shell $(BENDER) path snitch_cluster)
REGGEN     			       = $(shell $(BENDER) path register_interface)/vendor/lowrisc_opentitan/util/regtool.py
SOURCE_CLINT_DIR            = $(shell $(BENDER) path clint)
HEMAIA_D2D_LINK_REG_HJSON   = $(shell $(BENDER) path hemaia_d2d_link)/src/ctrl/hemaia_d2d_link_reg.hjson
HEMAIA_CLK_RST_REG_HJSON    = $(shell $(BENDER) path hemaia_clk_rst_controller)/regs/hemaia_clk_rst_controller_reg.hjson
else
SNITCH_ROOT                 =
REGGEN     			       =
SOURCE_CLINT_DIR            =
HEMAIA_D2D_LINK_REG_HJSON   =
HEMAIA_CLK_RST_REG_HJSON    =
endif
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

$(PLATFORM_HEADERS_DIR):
	mkdir -p $@

###########
# 1.CLINT #
###########
$(TARGET_CLINT_DIR):
	mkdir -p $@
# CLINT needs the hjson tmp first to generate the hjson
$(TARGET_CLINT_DIR)/clint.hjson: $(SOURCE_CLINT_DIR)/data/clint.hjson.tpl $(CFG)
	@echo "[OCCAMYGEN] Generate $@"
	@$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_CLINT_DIR) --chip $<
# Then the hjson can be used to generate the header file
$(PLATFORM_HEADERS_DIR)/clint.h: $(TARGET_CLINT_DIR)/clint.hjson | $(PLATFORM_HEADERS_DIR)
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
$(PLATFORM_HEADERS_DIR)/rv_plic.h: $(TARGET_PLIC_DIR)/rv_plic.hjson | $(PLATFORM_HEADERS_DIR)
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
$(PLATFORM_HEADERS_DIR)/occamy_soc_ctrl.h: $(TARGET_SOCCTRL_DIR)/occamy_soc_reg.hjson | $(PLATFORM_HEADERS_DIR)
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
$(PLATFORM_HEADERS_DIR)/occamy_quad_periph.h: $(TARGET_QUADPERIPH_DIR)/occamy_quad_periph_reg.hjson | $(PLATFORM_HEADERS_DIR)
	$(call reggen_generate_header,$@,$<)
######################
# Cluster Peripheral #
######################

$(PLATFORM_HEADERS_DIR)/snitch_cluster_peripheral.h: $(SNITCH_ROOT)/hw/snitch_cluster/src/snitch_cluster_peripheral/snitch_cluster_peripheral_reg.hjson | $(PLATFORM_HEADERS_DIR)
	$(call reggen_generate_header,$@,$<)

########################
# HeMAiA D2D Link Ctrl #
########################

$(PLATFORM_HEADERS_DIR)/hemaia_d2d_link_peripheral.h: $(HEMAIA_D2D_LINK_REG_HJSON) | $(PLATFORM_HEADERS_DIR)
	$(call reggen_generate_header,$@,$<)

#######################
# HeMAiA Clk Rst Ctrl #
#######################

$(PLATFORM_HEADERS_DIR)/hemaia_clk_rst_controller_peripheral.h: $(HEMAIA_CLK_RST_REG_HJSON) | $(PLATFORM_HEADERS_DIR)
	$(call reggen_generate_header,$@,$<)

# OCCAMYGEN headers
$(PLATFORM_HEADERS_DIR)/occamy_memory_map.h: $(PLATFORM_HEADERS_TEMPLATE_DIR)/occamy_memory_map.h.tpl $(CFG) | $(PLATFORM_HEADERS_DIR)
	@echo "[OCCAMYGEN] Generating $@"
	@$(OCCAMYGEN) -c $(CFG) --outdir $(PLATFORM_HEADERS_DIR) --cheader $<

$(PLATFORM_HEADERS_DIR)/occamy.h: $(PLATFORM_HEADERS_TEMPLATE_DIR)/occamy.h.tpl $(CFG) | $(PLATFORM_HEADERS_DIR)
	@echo "[OCCAMYGEN] Generating $@"
	@$(OCCAMYGEN) -c $(CFG) --outdir $(PLATFORM_HEADERS_DIR) --cheader $<

$(PLATFORM_HEADERS_DIR)/occamy_base_addr.h: $(CFG) | $(PLATFORM_HEADERS_DIR)
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


# SW-only snax codegen: emits streamer_csr_addr_map.h (and any future SW-side
# headers) without running Chisel RTL elaboration. 
SNAX_SW_GEN_MARKER = $(SW_DIR)/.snax_sw_gen_targets_group

snax-sw-gen: $(SNAX_SW_GEN_MARKER)

$(SNAX_SW_GEN_MARKER): $(CFG)
	@echo "[SNAX-SW-GEN] Generate SNAX SW headers"
	@# Wipe every stale streamer_csr_addr_map.h from the snax SW tree first.
	@# The device build's INCDIRS globs sw/snax/*/include alphabetically, and
	@# a header from a previously-active library (e.g. gemmx) will silently
	@# shadow the current one (e.g. versacore-to) because both file names
	@# match. Deleting all of them before regeneration guarantees only the
	@# maps occamygen writes for $(CFG) survive.
	@find $(SNITCH_ROOT)/target/snitch_cluster/sw/snax -name streamer_csr_addr_map.h -delete
	$(OCCAMYGEN) --cfg $(CFG) \
		--snitch $(SNITCH_ROOT) \
		--snitch-sw-only
	@touch $@

clean-snax-sw-gen:
	rm -rf $(SNAX_SW_GEN_MARKER)

# Structural sync check that every function defined in the upstream
# snax-versacore-to-lib.c also exists in the local snax_versacore_lib.h and
# programs the same ordered sequence of CSRs (modulo the GEMMX_*→VERSACORE_*
# rename). See util/hemaia/check_snax_versacore_lib_sync.py for the rules.
# Runs after snax-sw-gen so the bender-managed snax checkout is present.
.PHONY: check-snax-versacore-sync
SNAX_VERSACORE_LOCAL_H   = $(SW_DIR)/device/runtime/snax/versacore/snax_versacore_lib.h
SNAX_VERSACORE_UPSTREAM_C = $(SNITCH_ROOT)/target/snitch_cluster/sw/snax/versacore-to/src/snax-versacore-to-lib.c
check-snax-versacore-sync: $(SNAX_SW_GEN_MARKER)
	@python3 $(ROOT)/util/hemaia/check_snax_versacore_lib_sync.py \
		$(SNAX_VERSACORE_LOCAL_H) $(SNAX_VERSACORE_UPSTREAM_C)

sw: $(PLATFORM_HEADERS) host/runtime/host.ld $(SNAX_SW_GEN_MARKER) check-snax-versacore-sync
	$(MAKE) -C host libs
	$(MAKE) -C host TARGET=partial-build
	$(MAKE) -C device all HOST_APP_TYPE=$(HOST_APP_TYPE)
	$(MAKE) -C host TARGET=finalize-build

single-sw: $(PLATFORM_HEADERS) host/runtime/host.ld $(SNAX_SW_GEN_MARKER) check-snax-versacore-sync
	$(MAKE) -C host libs
	$(MAKE) -C host single_sw TARGET=partial-build \
		HOST_APP_TYPE=$(HOST_APP_TYPE) \
		CHIP_TYPE=$(CHIP_TYPE) \
		WORKLOAD=$(WORKLOAD) \
		DEV_APP=$(DEV_APP)
	$(MAKE) -C device all DEVICE_APPS=$(DEV_APP) HOST_APP_TYPE=$(HOST_APP_TYPE)
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
# clean snax-sw-gen marker so the next build regenerates
	rm -rf $(SNAX_SW_GEN_MARKER)
