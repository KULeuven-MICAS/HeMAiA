# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Nils Wistoff <nwistoff@iis.ee.ethz.ch>

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
ROOT        := ${MKFILE_DIR}../../..
CVA6_SDK    ?= ${ROOT}/../cva6-sdk
DEBUG       ?= 1
EXT_JTAG    ?= 1

VIVADO ?= vivado

BENDER_TARGETS_FILE := $(MKFILE_DIR)../../rtl/src/bender_targets.tmp
BENDER_TARGETS = $(shell cat $(BENDER_TARGETS_FILE))
BENDER_TARGETS += -t hemaia
TARGET_PLATFORM ?= vpk180
BENDER_TARGETS += -t $(TARGET_PLATFORM)
ifeq ($(TARGET_PLATFORM),vpk180)
	BENDER_TARGETS += -t nod2dphy -t exclude_d2d_link_phy
endif

default: all
all: hemaia_system

include $(ROOT)/target/fpga/Makefrag

CHIP_RECIPE := ../hemaia_chip
.PHONY: chip_package hemaia_system
chip_package:
	$(MAKE) -C $(CHIP_RECIPE) hemaia_chip TARGET_PLATFORM=$(TARGET_PLATFORM) EXT_JTAG=$(EXT_JTAG)

hemaia_system: chip_package define_defines_includes_no_simset.tcl \
                    hemaia_system_$(TARGET_PLATFORM).tcl hemaia_system_$(TARGET_PLATFORM)_impl.xdc
	${VIVADO} -mode batch -source hemaia_system_${TARGET_PLATFORM}.tcl -tclargs $(DEBUG) $(EXT_JTAG)

define_defines_includes_no_simset.tcl: $(BENDER_TARGETS_FILE) $(ROOT)/Bender.yml $(ROOT)/Bender.lock $(MKFILE_PATH)
	${BENDER} script vivado $(BENDER_TARGETS) --only-defines --only-includes --no-simset > $@

clean:
	rm -rf .Xil hemaia_system *.jou *.log *.str define_defines_includes_no_simset.tcl

.PHONY: program flash clean
