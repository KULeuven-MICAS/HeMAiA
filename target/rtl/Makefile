# Copyright 2025 KU Leuven
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yunhao Deng <yunhao.deng@kuleuven.be>
# Fanchen Kong <fanchen.kong@kuleuven.be>
# <Try not to blindly include other makefiles>
# This makefile generates the RTL sources for HeMAiA SoC

############
# Makefrag #
############
BENDER      ?= bender
MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))


#########################
# Files and directories #
#########################
ROOT                  = $(abspath $(MKFILE_DIR)../..)
SW_DIR                = $(ROOT)/target/sw
SW_HJSON_DIR          = $(SW_DIR)/shared/platform/template

TARGET_SRC_DIR        = $(ROOT)/target/rtl/src
TARGET_CLINT_DIR      = $(TARGET_SRC_DIR)/clint
TARGET_PLIC_DIR       = $(TARGET_SRC_DIR)/rv_plic
TARGET_SOCCTRL_DIR	  = $(TARGET_SRC_DIR)/soc_ctrl
TARGET_QUADPERIPH_DIR = $(TARGET_SRC_DIR)/quad_periph
SNITCH_ROOT           = $(shell $(BENDER) path snitch_cluster)
HEMAIA_UTIL           = $(ROOT)/util/hemaia/util.py
SOURCE_OCCAMY_DIR     = $(ROOT)/hw/occamy
SOURCE_QUADPERIPH_DIR   = $(ROOT)/hw/occamy/quad_periph
SOURCE_SOCCTRL_DIR    = $(ROOT)/hw/occamy/soc_ctrl
SOURCE_PLIC_DIR       = $(ROOT)/hw/vendor/pulp_platform_opentitan_peripherals/src/rv_plic
SOURCE_CLINT_DIR      = $(shell $(BENDER) path clint)
TARGET_RTL			  = $(ROOT)/target/rtl
HEMAIA_MEM_SYS_DIR    = $(ROOT)/hw/hemaia/hemaia_mem_system
############
# Programs #
############


OCCAMYGEN  	 = $(ROOT)/util/occamygen/occamygen.py
REGGEN     	 = $(shell $(BENDER) path register_interface)/vendor/lowrisc_opentitan/util/regtool.py


#######################
# Config prerequisite #
#######################
CFG		     ?= $(ROOT)/target/rtl/cfg/lru.hjson

.PHONY: all clean
all: rtl

.PHONY: rtl clint plic socctrl quadctrl soc
.PHONY: clean-rtl-src clean-rtl clean-clint clean-plic clean-socctrl clean-quadctrl clean-snax-cluster clean-soc

clean: clean-rtl-src
###############
# RTL sources #
###############



rtl: clint plic socctrl quadctrl soc snax-cluster src/bender_targets.tmp
clean-rtl: clean-clint clean-plic clean-socctrl clean-quadctrl clean-soc clean-snax-cluster
clean-rtl-src:
	rm -rf $(TARGET_SRC_DIR)
	rm -rf .snax_cluster_occamygen_targets_group
	rm -rf .misc_occamygen_targets_group
	rm -rf .quadctrl_reggen_targets_group
	rm -rf .plic_reggen_targets_group
	rm -rf .clint_reggen_targets_group
	rm -rf .socctrl_reggen_targets_group
########
# SNAX #
########

# Make the snax gen as a seperate target
snax-cluster: .snax_cluster_occamygen_targets_group

.snax_cluster_occamygen_targets_group: $(CFG)
	@echo "[OCCAMYGEN] START Generate SNAX CLUSTERS"

	$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_SRC_DIR)\
		--snitch           $(SNITCH_ROOT)

	@echo "[OCCAMYGEN] FINISH Generate SNAX CLUSTERS"
	@touch $@
clean-snax-cluster:
	rm -rf .snax_cluster_occamygen_targets_group

#######
# SOC #
#######

MISC_OCCAMYGEN_SV_TEMPLATES  = $(SOURCE_OCCAMY_DIR)/occamy_*.sv.tpl
MISC_OCCAMYGEN_SV_TARGETS  = $(TARGET_SRC_DIR)/occamy_*.sv
MISC_OCCAMYGEN_SV_TARGETS  += $(TARGET_SRC_DIR)/snax_*.sv

MISC_OCCAMYGEN_TARGETS  = $(MISC_OCCAMYGEN_SV_TARGETS)
MISC_OCCAMYGEN_TARGETS += $(TARGET_SRC_DIR)/occamy_memories.json


soc: $(MISC_OCCAMYGEN_TARGETS)
clean-soc:
	rm -rf $(MISC_OCCAMYGEN_TARGETS) .misc_occamygen_targets_group
	rm -rf $(TARGET_SRC_DIR)/*noc*
	rm -rf $(TARGET_SRC_DIR)/axi_flat.sv
	rm -rf $(TARGET_SRC_DIR)/apb_flat.sv
	rm -rf $(TARGET_SRC_DIR)/bender_targets.tmp
	rm -rf $(HEMAIA_MEM_SYS_DIR)/hemaia_xdma_wrapper.sv
	rm -rf $(HEMAIA_MEM_SYS_DIR)/hemaia_xdma.sv
	rm -rf $(HEMAIA_MEM_SYS_DIR)/hemaia_mem_chip_xdma_wrapper.sv
	rm -rf $(HEMAIA_MEM_SYS_DIR)/hemaia_mem_chip_xdma.sv

$(MISC_OCCAMYGEN_TARGETS): .misc_occamygen_targets_group
.misc_occamygen_targets_group: $(CFG) $(MISC_OCCAMYGEN_SV_TEMPLATES)
	@echo "[OCCAMYGEN] Generate $(MISC_OCCAMYGEN_TARGETS)"

	$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_SRC_DIR)\
		--top-sv           $(SOURCE_OCCAMY_DIR)/occamy_top.sv.tpl \
		--soc-sv           $(SOURCE_OCCAMY_DIR)/occamy_soc.sv.tpl \
		--xdma             \
		--chip             $(SOURCE_OCCAMY_DIR)/occamy_chip.sv.tpl \
		--io               $(SOURCE_OCCAMY_DIR)/hemaia.sv.tpl \
		--pkg-sv           $(SOURCE_OCCAMY_DIR)/occamy_pkg.sv.tpl \
		--quad             $(SOURCE_OCCAMY_DIR)/occamy_quad.sv.tpl \
		--quad-ctrl        $(SOURCE_OCCAMY_DIR)/occamy_quad_ctrl.sv.tpl \
		--quad-noc         $(SOURCE_OCCAMY_DIR)/quad_noc/quad_noc.yml.tpl \
		--xilinx-sv        $(SOURCE_OCCAMY_DIR)/occamy_xilinx.sv.tpl \
		--cva6-sv          $(SOURCE_OCCAMY_DIR)/occamy_cva6.sv.tpl

# File included by occamy_xilinx needs to be put in the same folder
	cp -f $(SOURCE_OCCAMY_DIR)/axi_flat.sv $(TARGET_SRC_DIR)
	cp -f $(SOURCE_OCCAMY_DIR)/apb_flat.sv $(TARGET_SRC_DIR)
	@touch $@

.PHONY: debug-soc-gen debug-quad-ctrl-gen debug-quad-gen debug-occamy-gen

debug-occamy-gen:
	$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_SRC_DIR)\
		--top-sv           $(SOURCE_OCCAMY_DIR)/occamy_top.sv.tpl \
		--soc-sv           $(SOURCE_OCCAMY_DIR)/occamy_soc.sv.tpl \
		--chip             $(SOURCE_OCCAMY_DIR)/occamy_chip.sv.tpl \
		--pkg-sv           $(SOURCE_OCCAMY_DIR)/occamy_pkg.sv.tpl \
		--quad             $(SOURCE_OCCAMY_DIR)/occamy_quad.sv.tpl \
		--quad-ctrl        $(SOURCE_OCCAMY_DIR)/occamy_quad_ctrl.sv.tpl \
		--quad-noc         $(SOURCE_OCCAMY_DIR)/quad_noc/quad_noc.yml.tpl \
		--xilinx-sv        $(SOURCE_OCCAMY_DIR)/occamy_xilinx.sv.tpl \
		--cva6-sv          $(SOURCE_OCCAMY_DIR)/occamy_cva6.sv.tpl	

debug-soc-gen:
	$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_SRC_DIR) \
	--soc-sv $(SOURCE_OCCAMY_DIR)/occamy_soc.sv.tpl \
	--top-sv $(SOURCE_OCCAMY_DIR)/occamy_top.sv.tpl

debug-quad-ctrl-gen:
	$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_SRC_DIR) --quad-ctrl $(SOURCE_OCCAMY_DIR)/occamy_quad_ctrl.sv.tpl

debug-quad-gen:
	$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_SRC_DIR) \
	--quad              $(SOURCE_OCCAMY_DIR)/occamy_quad.sv.tpl \
	--quad-noc          $(SOURCE_OCCAMY_DIR)/quad_noc/quad_noc.yml.tpl

###########
# SOCCTRL #
###########
$(TARGET_SOCCTRL_DIR):
	@mkdir -p $@
SOCCTRL_HJSON_DIR	    = $(SW_HJSON_DIR)/soc_ctrl
SOCCTRL_RTL_TARGET      = $(TARGET_SOCCTRL_DIR)/occamy_soc_ctrl.sv
SOCCTRL_REGGEN_TARGETS  = $(TARGET_SOCCTRL_DIR)/occamy_soc_reg_pkg.sv
SOCCTRL_REGGEN_TARGETS += $(TARGET_SOCCTRL_DIR)/occamy_soc_reg_top.sv

$(SOCCTRL_REGGEN_TARGETS): .socctrl_reggen_targets_group

# SOCCTRL RTL needs the hjson file
# Normally if you generate the SW first, the hjson file is already generated
# But if you generate the RTL first, we need to generate the hjson from the template
$(SOCCTRL_HJSON_DIR)/occamy_soc_reg.hjson: $(SOURCE_SOCCTRL_DIR)/occamy_soc_reg.hjson.tpl $(CFG)
	@echo "[OCCAMYGEN] Generate $@"
	@$(OCCAMYGEN) --cfg $(CFG) --outdir $(SOCCTRL_HJSON_DIR) --ctrl $<
# Generate the RTL file from the template
$(SOCCTRL_RTL_TARGET): $(SOURCE_SOCCTRL_DIR)/occamy_soc_ctrl.sv.tpl | $(TARGET_SOCCTRL_DIR)
	@echo "[OCCAMYGEN] Generate $@"
	@$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_SOCCTRL_DIR) --ctrl $<

socctrl: $(SOCCTRL_RTL_TARGET) $(SOCCTRL_REGGEN_TARGETS)
clean-socctrl:
	rm -rf $(SOCCTRL_REGGEN_TARGETS) .socctrl_reggen_targets_group
# Generate the reggen targets from the hjson file
.socctrl_reggen_targets_group: $(SOCCTRL_HJSON_DIR)/occamy_soc_reg.hjson | $(TARGET_SOCCTRL_DIR)
	@$(REGGEN) -r -t $(TARGET_SOCCTRL_DIR) $<
	@touch $@

###############
# QUAD PERIPH #
###############

$(TARGET_QUADPERIPH_DIR):
	@mkdir -p $@

QUADPERIPH_REGGEN_TARGETS = $(TARGET_QUADPERIPH_DIR)/occamy_quad_periph_reg_pkg.sv
QUADPERIPH_REGGEN_TARGETS += $(TARGET_QUADPERIPH_DIR)/occamy_quad_periph_reg_top.sv

$(QUADPERIPH_REGGEN_TARGETS): .quadperiph_reggen_targets_group

quadperiph: $(QUADPERIPH_REGGEN_TARGETS)
clean-quadperiph:
	rm -rf $(QUADPERIPH_REGGEN_TARGETS) .quadperiph_reggen_targets_group
# Generate the reggen targets from the hjson file
.quadperiph_reggen_targets_group: $(SOURCE_QUADPERIPH_DIR)/occamy_quad_periph_reg.hjson | $(TARGET_QUADPERIPH_DIR)
	@$(REGGEN) -r -t $(TARGET_QUADPERIPH_DIR) $<
	@cp $(SOURCE_QUADPERIPH_DIR)/occamy_quad_periph.sv $(TARGET_QUADPERIPH_DIR)/occamy_quad_periph.sv
	@touch $@
	
########
# PLIC #
########

PLICGEN      = $(SOURCE_PLIC_DIR)/util/reg_rv_plic.py
# PLIC do not need the occamygen
$(TARGET_PLIC_DIR):
	@mkdir -p $@
$(PLIC_HJSON_DIR):
	@mkdir -p $@
PLIC_HJSON_DIR       = $(SW_HJSON_DIR)/rv_plic
PLIC_RTL_TARGET     = $(TARGET_PLIC_DIR)/rv_plic.sv
PLIC_REGGEN_TARGETS = $(TARGET_PLIC_DIR)/rv_plic_reg_pkg.sv
PLIC_REGGEN_TARGETS += $(TARGET_PLIC_DIR)/rv_plic_reg_top.sv

$(PLIC_REGGEN_TARGETS): .plic_reggen_targets_group
# PLIC RTL needs the hjson file
# Normally if you generate the SW first, the hjson file is already generated
# But if you generate the RTL first, we need to generate the hjson from the template
$(PLIC_HJSON_DIR)/rv_plic.hjson: $(SOURCE_PLIC_DIR)/data/rv_plic.hjson.tpl $(CFG) | $(PLIC_HJSON_DIR)
	@echo "[PLICGEN] Generate $@"
	@$(PLICGEN) -s 73 -t 2 -p 6 $< > $@

# Generate the RTL file from the template
$(PLIC_RTL_TARGET): $(SOURCE_PLIC_DIR)/data/rv_plic.sv.tpl | $(TARGET_PLIC_DIR)
	@echo "[PLICGEN] Generate $@"
	@$(PLICGEN) -s 73 -t 2 -p 6 $< > $@

plic: $(PLIC_RTL_TARGET) $(PLIC_REGGEN_TARGETS)
clean-plic:
	rm -rf $(PLIC_RTL_TARGET) $(PLIC_REGGEN_TARGETS) .plic_reggen_targets_group
# Generate the reggen targets from the hjson file
.plic_reggen_targets_group: $(PLIC_HJSON_DIR)/rv_plic.hjson | $(TARGET_PLIC_DIR)
	@$(REGGEN) -r -t $(TARGET_PLIC_DIR) $<
	@touch $@

#########
# CLINT #
#########
$(TARGET_CLINT_DIR):
	@mkdir -p $@
CLINT_HJSON_DIR	   	  = $(SW_HJSON_DIR)/clint
CLINT_RTL_TARGET     = $(TARGET_CLINT_DIR)/clint.sv
CLINT_REGGEN_TARGETS = $(TARGET_CLINT_DIR)/clint_reg_pkg.sv
CLINT_REGGEN_TARGETS += $(TARGET_CLINT_DIR)/clint_reg_top.sv

$(CLINT_REGGEN_TARGETS): .clint_reggen_targets_group

# CLINT RTL needs the hjson file
# Normally if you generate the SW first, the hjson file is already generated
# But if you generate the RTL first, we need to generate the hjson from the template
$(CLINT_HJSON_DIR)/clint.hjson: $(SOURCE_CLINT_DIR)/data/clint.hjson.tpl $(CFG)
	@echo "[OCCAMYGEN] Generate $@"
	@$(OCCAMYGEN) --cfg $(CFG) --outdir $(CLINT_HJSON_DIR) --chip $<
# Generate the RTL file from the template
$(CLINT_RTL_TARGET): $(SOURCE_CLINT_DIR)/data/clint.sv.tpl | $(TARGET_CLINT_DIR)
	@echo "[OCCAMYGEN] Generate $@"
	@$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_CLINT_DIR) --chip $<

clint: $(CLINT_RTL_TARGET) $(CLINT_REGGEN_TARGETS)
clean-clint:
	rm -rf $(CLINT_RTL_TARGET)$(CLINT_REGGEN_TARGETS) .clint_reggen_targets_group
# Generate the reggen targets from the hjson file
.clint_reggen_targets_group: $(CLINT_HJSON_DIR)/clint.hjson | $(TARGET_CLINT_DIR)
	@$(REGGEN) -r -t $(TARGET_CLINT_DIR) $<
	@touch $@

###########################################
# SNAX Accelerator  + Wrapper Generations #
###########################################

#######################
# Step 1: Wrapper Gen #
#######################

# This tag is integrated into SNAX Cluster

###########################
# Step 2: Accelerator Gen #
###########################

# This tag is integrated into SNAX Cluster

#############################
# Step 3: Bender Target Gen #
#############################
src/bender_targets.tmp: $(CFG) # Generate bender_targets.tmp file
	echo $(shell $(HEMAIA_UTIL) --cfg_path $(CFG) --get_bender_targets) > src/bender_targets.tmp
