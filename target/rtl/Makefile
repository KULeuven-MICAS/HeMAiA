# The dependencies in this file are below the targets, in reverse order. (for you to read them easier)


# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Fabian Schuiki <fschuiki@iis.ee.ethz.ch>
# Florian Zaruba <zarubaf@iis.ee.ethz.ch>

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
ROOT        := $(MKFILE_DIR)../..
TARGET_SRC_DIR       ?= src
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
HEMAIA_UTIL = $(ROOT)/util/hemaia/util.py

SOURCE_OCCAMY_DIR    ?= $(ROOT)/hw/occamy

SOURCE_QUADCTRL_DIR  ?= $(ROOT)/hw/occamy/quadrant_s1_ctrl
TARGET_QUADCTRL_DIR  ?= $(TARGET_SRC_DIR)/quadrant_s1_ctrl


SOURCE_SOCCTRL_DIR   ?= $(ROOT)/hw/occamy/soc_ctrl
TARGET_SOCCTRL_DIR   ?= $(TARGET_SRC_DIR)/soc_ctrl

SOURCE_PLIC_DIR       = $(ROOT)/hw/vendor/pulp_platform_opentitan_peripherals/src/rv_plic
TARGET_PLIC_DIR      ?= $(TARGET_SRC_DIR)/rv_plic

SOURCE_CLINT_DIR      = $(shell $(BENDER) path clint)
TARGET_CLINT_DIR     ?= $(TARGET_SRC_DIR)/clint

TARGET_RTL			 ?= $(ROOT)/target/rtl

HEMAIA_MEM_SYS_DIR   ?= $(ROOT)/hw/hemaia/hemaia_mem_system
############
# Programs #
############

BENDER       ?= bender
#VERIBLE_FMT  ?= verible-verilog-format
OCCAMYGEN  ?= $(ROOT)/util/occamygen/occamygen.py
REGGEN     ?= $(shell $(BENDER) path register_interface)/vendor/lowrisc_opentitan/util/regtool.py
PLICGEN     = $(SOURCE_PLIC_DIR)/util/reg_rv_plic.py

#######################
# Config prerequisite #
#######################
CFG = $(TARGET_RTL)/cfg/lru.hjson

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
	rm -rf .socctrl_reggen_targets_grouip


$(TARGET_PLIC_DIR) $(TARGET_CLINT_DIR) $(TARGET_SOCCTRL_DIR) $(TARGET_QUADCTRL_DIR):
	@mkdir -p $@


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
	rm -rf $(HEMAIA_MEM_SYS_DIR)/hemaia_xdma_wrapper.sv $(HEMAIA_MEM_SYS_DIR)/hemaia_xdma.sv

$(MISC_OCCAMYGEN_TARGETS): .misc_occamygen_targets_group
.misc_occamygen_targets_group: $(CFG) $(MISC_OCCAMYGEN_SV_TEMPLATES)
	@echo "[OCCAMYGEN] Generate $(MISC_OCCAMYGEN_TARGETS)"

	$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_SRC_DIR)\
		--top-sv           $(SOURCE_OCCAMY_DIR)/occamy_top.sv.tpl \
		--soc-sv           $(SOURCE_OCCAMY_DIR)/occamy_soc.sv.tpl \
		--xdma             \
		--chip             $(SOURCE_OCCAMY_DIR)/occamy_chip.sv.tpl \
		--pkg-sv           $(SOURCE_OCCAMY_DIR)/occamy_pkg.sv.tpl \
		--quadrant-s1      $(SOURCE_OCCAMY_DIR)/occamy_quadrant_s1.sv.tpl \
		--quadrant-s1-ctrl $(SOURCE_OCCAMY_DIR)/occamy_quadrant_s1_ctrl.sv.tpl \
		--quadrant-s1-noc  $(SOURCE_OCCAMY_DIR)/quad_noc/quad_noc.yml.tpl \
		--xilinx-sv        $(SOURCE_OCCAMY_DIR)/occamy_xilinx.sv.tpl \
		--cva6-sv          $(SOURCE_OCCAMY_DIR)/occamy_cva6.sv.tpl
# --dts 			   $(TARGET_DTS_DIR)/occamy.dts
#why?	@$(VERIBLE_FMT) --inplace $(MISC_OCCAMYGEN_SV_TARGETS)

# File included by occamy_xilinx needs to be put in the same folder
	cp -f $(SOURCE_OCCAMY_DIR)/axi_flat.sv $(TARGET_SRC_DIR)
	cp -f $(SOURCE_OCCAMY_DIR)/apb_flat.sv $(TARGET_SRC_DIR)
	@touch $@

.PHONY: debug-soc-gen debug-quadrant-ctrl-gen debug-quadrant-gen debug-occamy-gen

debug-occamy-gen:
	$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_SRC_DIR)\
		--top-sv           $(SOURCE_OCCAMY_DIR)/occamy_top.sv.tpl \
		--soc-sv           $(SOURCE_OCCAMY_DIR)/occamy_soc.sv.tpl \
		--chip             $(SOURCE_OCCAMY_DIR)/occamy_chip.sv.tpl \
		--pkg-sv           $(SOURCE_OCCAMY_DIR)/occamy_pkg.sv.tpl \
		--quadrant-s1      $(SOURCE_OCCAMY_DIR)/occamy_quadrant_s1.sv.tpl \
		--quadrant-s1-ctrl $(SOURCE_OCCAMY_DIR)/occamy_quadrant_s1_ctrl.sv.tpl \
		--quadrant-s1-noc  $(SOURCE_OCCAMY_DIR)/quad_noc/quad_noc.yml.tpl \
		--xilinx-sv        $(SOURCE_OCCAMY_DIR)/occamy_xilinx.sv.tpl \
		--cva6-sv          $(SOURCE_OCCAMY_DIR)/occamy_cva6.sv.tpl	

debug-soc-gen:
	$(ROOT)/util/occamygen/occamy_soc_gen.py --cfg $(CFG) --outdir $(TARGET_SRC_DIR) --soc-sv $(SOURCE_OCCAMY_DIR)/occamy_soc.sv.tpl --top-sv           $(SOURCE_OCCAMY_DIR)/occamy_top.sv.tpl 

debug-quadrant-ctrl-gen:
	$(ROOT)/util/occamygen/occamy_soc_gen.py --cfg $(CFG) --outdir $(TARGET_SRC_DIR) --quadrant-s1-ctrl $(SOURCE_OCCAMY_DIR)/occamy_quadrant_s1_ctrl.sv.tpl

debug-quadrant-gen:
	$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_SRC_DIR) \
	--quadrant-s1      $(SOURCE_OCCAMY_DIR)/occamy_quadrant_s1.sv.tpl \
	--quadrant-s1-noc  $(SOURCE_OCCAMY_DIR)/quad_noc/quad_noc.yml.tpl
############
# QUADCTRL #
############

QUADCTRL_OCCAMYGEN_TARGETS = $(TARGET_QUADCTRL_DIR)/occamy_quadrant_s1_reg.hjson
QUADCTRL_REGGEN_TARGETS = $(addprefix $(TARGET_QUADCTRL_DIR)/,occamy_quadrant_s1_reg_pkg.sv occamy_quadrant_s1_reg_top.sv)

quadctrl: $(QUADCTRL_OCCAMYGEN_TARGETS) $(QUADCTRL_REGGEN_TARGETS)
clean-quadctrl:
	rm -rf $(QUADCTRL_OCCAMYGEN_TARGETS) $(QUADCTRL_REGGEN_TARGETS) .quadctrl_reggen_targets_group

$(QUADCTRL_REGGEN_TARGETS): .quadctrl_reggen_targets_group
.quadctrl_reggen_targets_group: $(TARGET_QUADCTRL_DIR)/occamy_quadrant_s1_reg.hjson | $(TARGET_QUADCTRL_DIR)
	@echo "[REGGEN] Generate $(QUADCTRL_REGGEN_TARGETS)"
	@$(REGGEN) -r -t $(TARGET_QUADCTRL_DIR) $<
#why?	@$(VERIBLE_FMT) --inplace $(QUADCTRL_REGGEN_TARGETS)
	@touch $@

$(TARGET_QUADCTRL_DIR)/occamy_quadrant_s1_reg.hjson: $(SOURCE_QUADCTRL_DIR)/occamy_quadrant_s1_reg.hjson.tpl | $(TARGET_QUADCTRL_DIR)
	@echo "[OCCAMYGEN] Generate $@"
	@$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_QUADCTRL_DIR) --quadrant-s1-ctrl $<

###########
# SOCCTRL #
###########

SOCCTRL_OCCAMYGEN_TARGETS = $(TARGET_SOCCTRL_DIR)/occamy_soc_reg.hjson
SOCCTRL_OCCAMYGEN_TARGETS += $(TARGET_SOCCTRL_DIR)/occamy_soc_ctrl.sv
SOCCTRL_REGGEN_TARGETS = $(addprefix $(TARGET_SOCCTRL_DIR)/,occamy_soc_reg_pkg.sv occamy_soc_reg_top.sv)

socctrl: $(SOCCTRL_OCCAMYGEN_TARGETS) $(SOCCTRL_REGGEN_TARGETS)
clean-socctrl:
	rm -rf $(SOCCTRL_OCCAMYGEN_TARGETS) $(SOCCTRL_REGGEN_TARGETS) .socctrl_reggen_targets_group

$(SOCCTRL_REGGEN_TARGETS): .socctrl_reggen_targets_group
.socctrl_reggen_targets_group: $(TARGET_SOCCTRL_DIR)/occamy_soc_reg.hjson | $(TARGET_SOCCTRL_DIR)
	@echo "[REGGEN] Generate $(SOCCTRL_REGGEN_TARGETS)"
	@$(REGGEN) -r -t $(TARGET_SOCCTRL_DIR) $<
#why?	@$(VERIBLE_FMT) --inplace $(SOCCTRL_REGGEN_TARGETS)
	@touch $@

$(TARGET_SOCCTRL_DIR)/occamy_soc_reg.hjson: $(SOURCE_SOCCTRL_DIR)/occamy_soc_reg.hjson.tpl | $(TARGET_SOCCTRL_DIR)
	@echo "[OCCAMYGEN] Generate $@"
	@$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_SOCCTRL_DIR) --ctrl $<

$(TARGET_SOCCTRL_DIR)/occamy_soc_ctrl.sv: $(SOURCE_SOCCTRL_DIR)/occamy_soc_ctrl.sv.tpl | $(TARGET_SOCCTRL_DIR)
	@echo "[OCCAMYGEN] Generate $@"
	@$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_SOCCTRL_DIR) --ctrl $<

########
# PLIC #
########

PLIC_PLICGEN_TARGETS = $(addprefix $(TARGET_PLIC_DIR)/,rv_plic.sv rv_plic.hjson)
PLIC_REGGEN_TARGETS = $(addprefix $(TARGET_PLIC_DIR)/,rv_plic_reg_pkg.sv rv_plic_reg_top.sv)

plic: $(PLIC_PLICGEN_TARGETS) $(PLIC_REGGEN_TARGETS)
clean-plic:
	rm -rf $(PLIC_PLICGEN_TARGETS) $(PLIC_REGGEN_TARGETS) .plic_reggen_targets_group

$(PLIC_REGGEN_TARGETS): .plic_reggen_targets_group
.plic_reggen_targets_group: $(TARGET_PLIC_DIR)/rv_plic.hjson | $(TARGET_PLIC_DIR)
	@echo "[REGGEN] Generate $(PLIC_REGGEN_TARGETS)"
	@$(REGGEN) -r -t $(TARGET_PLIC_DIR) $<
#why?	@$(VERIBLE_FMT) --inplace $(PLIC_REGGEN_TARGETS)
	@touch $@

$(TARGET_PLIC_DIR)/rv_plic.%: $(SOURCE_PLIC_DIR)/data/rv_plic.%.tpl | $(TARGET_PLIC_DIR)
	@echo "[PLICGEN] Generate $@"
	@$(PLICGEN) -s 73 -t 2 -p 6 $< > $@

#########
# CLINT #
#########

CLINT_OCCAMYGEN_TARGETS = $(addprefix $(TARGET_CLINT_DIR)/,clint.sv clint.hjson)
CLINT_REGGEN_TARGETS = $(addprefix $(TARGET_CLINT_DIR)/,clint_reg_pkg.sv clint_reg_top.sv)

clint: $(CLINT_OCCAMYGEN_TARGETS) $(CLINT_REGGEN_TARGETS)
clean-clint:
	rm -rf $(CLINT_OCCAMYGEN_TARGETS) $(CLINT_REGGEN_TARGETS) .clint_reggen_targets_group

$(CLINT_REGGEN_TARGETS): .clint_reggen_targets_group
.clint_reggen_targets_group: $(TARGET_CLINT_DIR)/clint.hjson | $(TARGET_CLINT_DIR)
	@echo "[REGGEN] Generate $(CLINT_REGGEN_TARGETS)"
	@$(REGGEN) -r -t $(TARGET_CLINT_DIR) $<
#why?	@$(VERIBLE_FMT) --inplace $(CLINT_REGGEN_TARGETS)
	@touch $@

$(TARGET_CLINT_DIR)/clint.%: $(SOURCE_CLINT_DIR)/data/clint.%.tpl $(CFG) | $(TARGET_CLINT_DIR)
	@echo "[OCCAMYGEN] Generate $@"
	@$(OCCAMYGEN) --cfg $(CFG) --outdir $(TARGET_CLINT_DIR) --chip $<

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
