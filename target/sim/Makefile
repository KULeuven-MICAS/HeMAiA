# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Fabian Schuiki <fschuiki@iis.ee.ethz.ch>
# Florian Zaruba <zarubaf@iis.ee.ethz.ch>

#######################
# Makefile invocation #
#######################

DEBUG        ?= OFF # ON to turn on debugging symbols
CFG          ?= $(realpath $(MKFILE_DIR)../rtl/cfg/lru.hjson)

.DEFAULT_GOAL := help
.PHONY: all apps clean clean-testharness clean-vsim clean-vlt clean-vcs clean-apps clean-logs clean-bin
all: apps addrmap
clean: clean-testharness clean-vsim clean-vlt clean-vcs clean-apps clean-logs clean-bin

############
# Makefrag #
############

# Common definitions used by included Makefiles

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
SIM_MKFILE_DIR  := $(dir $(MKFILE_PATH))
ROOT        := $(SIM_MKFILE_DIR)../..
SNITCH_ROOT := $(shell bender path snitch_cluster)

TARGET = occamy_chip

include $(SIM_MKFILE_DIR)/sim.mk

#########################
# Files and directories #
#########################

BIN_DIR              ?= bin
UART_DPI_DIR         ?= $(SIM_MKFILE_DIR)testharness/uartdpi

#####################
# Simulator options #
#####################
SIM_WITH_MACRO ?= 0
SIM_WITH_NETLIST ?= 0
SIM_WITH_WAVEFORM ?= 1
SIM_WITH_INTERPOSER ?= 0

VSIM_BENDER += $(shell cat ../rtl/src/bender_targets.tmp)
VLT_BENDER  += $(shell cat ../rtl/src/bender_targets.tmp)
VCS_BENDER  += $(shell cat ../rtl/src/bender_targets.tmp)

ifeq ($(SIM_WITH_NETLIST), 1)
	VSIM_BENDER += -t hemaia_netlist
	VLT_BENDER  += -t hemaia_netlist
	VCS_BENDER  += -t hemaia_netlist
else
	VSIM_BENDER += -t hemaia
	VLT_BENDER  += -t hemaia
	VCS_BENDER  += -t hemaia
endif

ifeq ($(SIM_WITH_MACRO), 1)
	VSIM_BENDER += -t tsmc16_sim -t tsmc16
	VLT_BENDER  += -t tsmc16_sim -t tsmc16
	VCS_BENDER  += -t tsmc16_sim -t tsmc16
endif

ifeq ($(SIM_WITH_INTERPOSER), 1)
	VSIM_BENDER += -t simulation_hemaia_interposer
	VLT_BENDER  += -t simulation_hemaia_interposer
	VCS_BENDER  += -t simulation_hemaia_interposer
else
	VSIM_BENDER += -t simulation_hemaia
	VLT_BENDER  += -t simulation_hemaia
	VCS_BENDER  += -t simulation_hemaia
endif

VSIM_FLAGS += -t 1ps

ifeq ($(SIM_WITH_WAVEFORM), 1)
	VSIM_FLAGS += -voptargs=+acc
endif

VSIM_FLAGS += -do "log -r /*; run -a"
# Downgraded from error as VSIM *wrongly* reports illegal enum assignments in riscv_dbg 4-phase CDCs
# TODO: Workaround; revise
VSIM_FLAGS += -warning 8386

VLOG_FLAGS += -svinputport=compat
VLOG_FLAGS += -override_timescale 1ns/1ps
VLOG_FLAGS += -suppress 2583
VLOG_FLAGS += -suppress 13314
VLOG_FLAGS += ${QUESTA_64BIT}
VLOG_FLAGS += -timescale 1ns/1ps

VCS_FLAGS += +nospecify
VCS_FLAGS += -Mdir=work-vcs
VCS_FLAGS += -Mlib=work-vcs
VCS_FLAGS += -full64
VCS_FLAGS += +lint=TFIPC-L
# Additional Optimizations
VCS_FLAGS += -O4
# Multi-threading compilation
VCS_FLAGS += -j16
# Multi-core
VCS_FLAGS += -fgp
# Debug Options
VCS_FLAGS += -assert disable_cover
# Waveform Visibility
VCS_FLAGS += -kdb
VCS_FLAGS += +vcs+fsdbon
VCS_FLAGS += -debug_access+all

# Verilated and compiled Occamy system
VLT_AR     = ${VLT_BUILDDIR}/Vtestharness__ALL.a

# VLT_FLAGS += --trace       # Enable VCD dumping
VLT_FLAGS += --prof-cfuncs # Enable code profiling

###############
# C testbench #
###############

TB_CC_SOURCES += $(UART_DPI_DIR)/uartdpi.c

TB_CC_FLAGS  = -std=c++17
TB_CC_FLAGS +=-I$(UART_DPI_DIR)

# Required C sources for the verilator TB that are linked against the verilated model
VLT_COBJ += $(VLT_BUILDDIR)/test/uartdpi/uartdpi.o
# Sources from verilator root
VLT_COBJ += $(VLT_BUILDDIR)/vlt/verilated.o
VLT_COBJ += $(VLT_BUILDDIR)/vlt/verilated_dpi.o
VLT_COBJ += $(VLT_BUILDDIR)/vlt/verilated_vcd_c.o
ifeq ($(VERILATOR_VERSION), 5)
	VLT_COBJ += $(VLT_BUILDDIR)/vlt/verilated_timing.o
	VLT_COBJ += $(VLT_BUILDDIR)/vlt/verilated_threads.o
endif
# Testharness
VLT_COBJ += $(VLT_BUILDDIR)/testharness/testharness.o

############
### Apps ###
############
apps: 
	$(MAKE) -C apps apps

clean-apps:
	$(MAKE) -C apps clean-apps

###############
# Testharness #
###############

OCCAMYGEN ?= $(ROOT)/util/occamygen/occamygen.py

testharness/testharness.sv: testharness/testharness.sv.tpl $(CFG)
	@echo "[OCCAMYGEN] Generate $@"

ifneq ($(SIM_WITH_NETLIST), 0)
	@$(OCCAMYGEN) --cfg $(CFG) --outdir testharness --netlist-testharness-sv $<
else ifneq ($(SIM_WITH_MACRO), 0)
	@$(OCCAMYGEN) --cfg $(CFG) --outdir testharness --mem-macro-testharness-sv $<
else
	@$(OCCAMYGEN) --cfg $(CFG) --outdir testharness --testharness-sv $<
endif

clean-testharness:
	rm -rf testharness/testharness.sv
#############
# Verilator #
#############

${VLT_AR}: testharness/testharness.sv
	$(call VERILATE,testharness)

# Quick sanity check, not really meant for simulation.
verilate: ${VLT_AR}

# Build targets for verilator TB
$(VLT_BUILDDIR)/vlt/%.o: $(VLT_ROOT)/include/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(VLT_CFLAGS) -c $< -o $@
$(VLT_BUILDDIR)/test/%.o: test/%.cc
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(VLT_CFLAGS) -c $< -o $@

$(VLT_BUILDDIR)/testharness/%.o: testharness/%.cc $(VLT_AR)
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(VLT_CFLAGS) -c $< -o $@

$(VLT_BUILDDIR)/test/uartdpi/uartdpi.o: $(UART_DPI_DIR)/uartdpi.c
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(VLT_CFLAGS) -c $< -o $@

# Link verilated archive wich $(VLT_COBJ)
$(BIN_DIR)/$(TARGET).vlt: $(VLT_AR) $(VLT_COBJ)
	mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -std=c++17 -L ${VLT_BUILDDIR}/lib -o $@ $(VLT_COBJ) $(VLT_AR) -lpthread -lutil

# Clean all build directories and temporary files for Verilator simulation
.PHONY: clean-vlt
clean-vlt: clean-work
	rm -rf work-vlt
	rm -f $(BIN_DIR)/$(TARGET).vlt

#############
# Questasim #
#############

$(VSIM_BUILDDIR):
	mkdir -p $@

$(VSIM_BUILDDIR)/compile.vsim.tcl: testharness/testharness.sv  | $(VSIM_BUILDDIR)
	$(BENDER) script vsim $(VSIM_BENDER) --vlog-arg="$(VLOG_FLAGS) -work $(dir $@) " > $@
	echo '$(VLOG) -work $(dir $@) $(TB_CC_SOURCES) -vv -ccflags "$(TB_CC_FLAGS)"' >> $@
	echo 'return 0' >> $@

$(BIN_DIR):
	mkdir -p $@

.PHONY: $(BIN_DIR)/occamy_chip.vsim clean-vsim

$(BIN_DIR)/occamy_chip.vsim: $(VSIM_BUILDDIR)/compile.vsim.tcl |$(BIN_DIR)
	touch $@
	vsim -c -do "source $<; quit" | tee $(dir $<)vlog.log
	@! grep -P "Errors: [1-9]*," $(dir $<)vlog.log
	vopt $(VOPT_FLAGS) -work $(VSIM_BUILDDIR) testharness -o testharness_opt | tee $(dir $<)vopt.log
	@! grep -P "Errors: [1-9]*," $(dir $<)vopt.log
	@echo "#!/bin/bash" > $@
	@echo 'script_dir="$$( cd -- "$$( dirname -- "$${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"' >> $@
	@echo 'cd $$script_dir' >> $@
	@echo 'vsim +permissive $(VSIM_FLAGS) $$2 -work ../$(VSIM_BUILDDIR) -c \
				-ldflags "-Wl,-rpath,-L -lutil" \
				testharness_opt +permissive-off ++$$1' >> $@
	@chmod +x $@
	@echo "#!/bin/bash" > $@.gui
	@echo 'script_dir="$$( cd -- "$$( dirname -- "$${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"' >> $@.gui
	@echo 'cd $$script_dir' >> $@.gui
	@echo 'vsim +permissive $(VSIM_FLAGS) -work ../$(VSIM_BUILDDIR) \
				-ldflags "-Wl,-rpath,-L -lutil" \
				testharness_opt +permissive-off ++$$1' >> $@.gui
	@chmod +x $@.gui

clean-vsim: clean-work
	rm -rf $(BIN_DIR)/$(TARGET).vsim $(BIN_DIR)/$(TARGET).vsim.gui $(VSIM_BUILDDIR) $(BIN_DIR)/vsim.wlf $(BIN_DIR)/transcript transcript


#######
# VCS #
#######

$(VCS_BUILDDIR):
	mkdir -p $@

$(VCS_BUILDDIR)/compile.sh: testharness/testharness.sv | $(VCS_BUILDDIR)
	mkdir -p $(VCS_BUILDDIR)
	${BENDER} script vcs ${VCS_BENDER} --vlog-arg="${VLOGAN_FLAGS}" --vcom-arg="${VHDLAN_FLAGS}" > $@
	chmod +x $@

$(BIN_DIR)/occamy_chip.vcs: $(VCS_BUILDDIR)/compile.sh
	bash work-vcs/compile.sh | tee $(dir $<)vcs_compile.log
	mkdir -p $(BIN_DIR)
	$(VCS) $(VCS_FLAGS) -o $(BIN_DIR)/occamy_chip.vcs -cc $(CC) -cpp $(CXX) \
		-override_timescale=1ns/1ps -full64 -ignore initializer_driver_checks testharness $(TB_CC_SOURCES) $(RTL_CC_SOURCES) \
		-CFLAGS "$(TB_CC_FLAGS)" -lutil +error+100 | tee $(dir $<)vcs.log
.PHONY: clean-vcs
clean-vcs: clean-work
	rm -rf $(VCS_BUILDDIR) AN.DB work.lib++ vc_hdrs.h $(BIN_DIR)/$(TARGET).vcs $(BIN_DIR)/inter.fsdb $(BIN_DIR)/$(TARGET).vcs.daidir $(BIN_DIR)/verdi_config_file $(BIN_DIR)/verdiLog $(BIN_DIR)/novas* $(BIN_DIR)/ucli.key

########
# Util #
########

.PHONY: clean-work clean-logs help

clean-work:
	rm -rf work

clean-logs:
	rm -rf $(LOGS_DIR)/

clean-bin:
	rm -rf $(BIN_DIR)
