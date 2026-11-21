# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Nils Wistoff <nwistoff@iis.ee.ethz.ch>

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
ROOT        := ${MKFILE_DIR}../../..
DEBUG       ?= 1
EXT_JTAG    ?= 1

VIVADO ?= vivado
BENDER ?= bender
TARGET_PLATFORM ?= vpk180

BENDER_TARGETS_FILE := $(MKFILE_DIR)../../rtl/src/bender_targets.tmp
BENDER_TARGETS = -t rtl
BENDER_TARGETS += -t exclude_first_pass_decoder
BENDER_TARGETS += $(shell cat $(BENDER_TARGETS_FILE))
BENDER_TARGETS += -t hemaia
BENDER_TARGETS += -t $(TARGET_PLATFORM)
ifeq ($(TARGET_PLATFORM),vpk180)
	BENDER_TARGETS += -t nod2dphy -t exclude_d2d_link_phy
endif

ifeq ($(EXT_JTAG), 0)
	BENDER_TARGETS += -t bscane
endif

include $(ROOT)/target/fpga/Makefrag

PACKAGE_STAMP := hemaia_chip/.package-$(TARGET_PLATFORM)-jtag$(EXT_JTAG)-complete
SOURCE_CONFIG := .bender-$(TARGET_PLATFORM)-jtag$(EXT_JTAG).tmp
GENERATED_RTL := $(MKFILE_DIR)../../rtl/src
PACKAGE_DEPS := $(GENERATED_RTL)/occamy_chip.sv $(GENERATED_RTL)/occamy_pkg.sv \
                hemaia_chip_$(TARGET_PLATFORM).tcl synth_constraints.xdc
ifeq ($(TARGET_PLATFORM),vpk180)
PACKAGE_DEPS += ../common/hemaia_package_vpk180_dram.tcl ../common/hemaia_fpga_ip_ports.tcl \
                ../common/hemaia_vpk180_dram_bridge.sv ../common/hemaia_dram_address_mapper.sv
endif

.PHONY: hemaia_chip clean
hemaia_chip: $(PACKAGE_STAMP)

$(PACKAGE_STAMP): define-sources.tcl $(PACKAGE_DEPS)
	rm -f hemaia_chip/.package-*-complete
	${VIVADO} -mode batch -source hemaia_chip_${TARGET_PLATFORM}.tcl -tclargs $(DEBUG) $(EXT_JTAG)
	touch $@

$(SOURCE_CONFIG):
	rm -f .bender-*-jtag*.tmp
	touch $@

define-sources.tcl: $(SOURCE_CONFIG) $(BENDER_TARGETS_FILE) $(ROOT)/Bender.yml $(ROOT)/Bender.lock $(MKFILE_PATH)
	${BENDER} script vivado $(BENDER_TARGETS) > $@

clean:
	rm -rf .Xil hemaia_chip dram_bridge vivado* define-sources.tcl .bender-*-jtag*.tmp component.xml xgui *.jou *.log
