# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Fabian Schuiki <fschuiki@iis.ee.ethz.ch>
# Florian Zaruba <zarubaf@iis.ee.ethz.ch>

#######################
# Makefile invocation #
#######################

# Every EDA compile below pipes its output through `| tee <log>`, and a shell
# pipeline reports the LAST command's status -- tee's, which is always 0. pipefail
# is what lets a dead simulator (vlog/vopt/vcs failing to start, an elaboration
# error) fail the build instead of leaving make green with no simulation binary.
SHELL        := /bin/bash
.SHELLFLAGS  := -o pipefail -c

DEBUG        ?= OFF # ON to turn on debugging symbols
CFG          ?= $(realpath $(MKFILE_DIR)../rtl/cfg/lru.hjson)
SIM_CFG      ?= $(realpath $(MKFILE_DIR)cfg/sim_rtl.hjson)
# Backend-only mode: consume the immutable outputs of the prepare phase.  A
# command-line assignment is inherited through the root Makefile's recursive
# $(MAKE), so `make hemaia_system_vcs PREPARED_SIM_INPUTS=1` reaches this file.
PREPARED_SIM_INPUTS ?= 0

# A recipe that dies after the shell has already created/truncated `$@` leaves a
# partial file with a fresh mtime, which make then treats as up to date -- the next
# build happily feeds a half-written compile.sh / .d / .hjson to the next tool.
.DELETE_ON_ERROR:

.DEFAULT_GOAL := help
.PHONY: help all apps clean clean-testharness clean-vsim clean-vlt clean-vcs clean-apps clean-logs clean-bin
all: apps
clean: clean-testharness clean-vsim clean-vlt clean-vcs clean-apps clean-logs clean-bin

help:
	@echo "HeMAiA simulation targets (run from target/sim):"
	@echo "  apps                          build the host+device app binaries"
	@echo "                                needs HOST_APP_TYPE=, CHIP_TYPE=, WORKLOAD=, DEV_APP="
	@echo "  bin/occamy_chip.vcs           compile the VCS simulation binary"
	@echo "  bin/occamy_chip.vsim          compile the Questasim simulation binary"
	@echo "  bin/occamy_chip.vlt           compile the Verilator simulation binary"
	@echo "  testharness/testharness.sv    generate the testharness"
	@echo "  clean                         remove every build/ and bin/ artefact"
	@echo
	@echo "Most users go through target/sim/automation instead:"
	@echo "  make -C ../sim/automation local-ci SUITE=tapeout_1c"

############
# Makefrag #
############

# Common definitions used by included Makefiles

MKFILE_PATH     := $(abspath $(lastword $(MAKEFILE_LIST)))
SIM_MKFILE_DIR  := $(dir $(MKFILE_PATH))
ROOT            := $(SIM_MKFILE_DIR)../..

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
SIM_WITH_WAVEFORM ?= 1
# Extract a boolean flag from a hjson file: returns 1 if key exists and isn't false, 0 otherwise
cfg_flag = $(shell if [ -f "$(1)" ]; then \
	if grep -E -q '^[[:space:]]*$(2)[[:space:]]*[:=][[:space:]]*false' "$(1)"; then echo 0; \
	elif grep -E -q '^[[:space:]]*$(2)' "$(1)"; then echo 1; \
	else echo 0; fi; else echo 0; fi)

SIM_WITH_MACRO      ?= $(call cfg_flag,$(SIM_CFG),sim_with_mem_macro)
SIM_WITH_NETLIST    ?= $(call cfg_flag,$(SIM_CFG),sim_with_netlist)
SIM_WITH_INTERPOSER ?= $(call cfg_flag,$(SIM_CFG),sim_with_interposer)
SIM_WITH_PLL        ?= $(call cfg_flag,$(SIM_CFG),sim_with_pll)
SIM_WITH_JTAG_CHECK ?= $(call cfg_flag,$(SIM_CFG),sim_with_jtag_check)

# The cfg-dependent bender target set, written by `make -C target/rtl`. Read once
# (:=) so the shell is forked a single time, and named by absolute path so it
# resolves from any working directory.
BENDER_TARGETS_FILE := $(ROOT)/target/rtl/src/bender_targets.tmp
BENDER_TARGETS      := $(shell cat $(BENDER_TARGETS_FILE) 2>/dev/null)

VSIM_BENDER += $(BENDER_TARGETS)
VLT_BENDER  += $(BENDER_TARGETS)
VCS_BENDER  += $(BENDER_TARGETS)

VSIM_BENDER += -t simulation_hemaia -t simulation_hemaia_interposer_model
VLT_BENDER  += -t simulation_hemaia
VCS_BENDER  += -t simulation_hemaia -t simulation_hemaia_interposer_model

VSIM_BENDER += -t rtl
VLT_BENDER  += -t rtl
VCS_BENDER  += -t rtl

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

ifeq ($(SIM_WITH_PLL), 1)
	VSIM_BENDER += -t tsmc16_pll_sim 
	VLT_BENDER  += -t tsmc16_pll_sim 
	VCS_BENDER  += -t tsmc16_pll_sim
endif

VSIM_DO_CMD = run -a
ifeq ($(SIM_WITH_WAVEFORM), 1)
	# Preserve broad signal visibility only in the explicitly requested debug
	# build.  Keeping +acc out of batch builds lets vopt optimize the design.
	VOPT_FLAGS += +acc
	VSIM_DO_CMD = log -r /*; run -a
endif

# This flow does not annotate SDF: the gate-level netlist is a functional,
# zero-delay simulation.  These switches let vopt discard the unused timing
# structures.  Dead-cell elimination is reserved for non-waveform batches so
# a debug build keeps every otherwise-unused object available for inspection.
ifeq ($(SIM_WITH_NETLIST), 1)
	VOPT_FLAGS += +nospecify
	VOPT_FLAGS += -nosdf
ifeq ($(SIM_WITH_WAVEFORM), 0)
	VOPT_FLAGS += -dce
endif
endif

VSIM_FLAGS += -t 1ps
VSIM_FLAGS += -do "$(VSIM_DO_CMD)"
# Downgraded from error as VSIM *wrongly* reports illegal enum assignments in riscv_dbg 4-phase CDCs
# TODO: Workaround; revise
VSIM_FLAGS += -warning 8386

VLOG_FLAGS += -svinputport=compat
VLOG_FLAGS += -override_timescale 1ns/1ps
VLOG_FLAGS += -timescale 1ns/1ps
VLOG_FLAGS += -suppress 2583
VLOG_FLAGS += -suppress 13314
VLOG_FLAGS += ${QUESTA_64BIT}


VCS_FLAGS += +nospecify
VCS_FLAGS += -Mdir=work-vcs
VCS_FLAGS += -Mlib=work-vcs
VCS_FLAGS += -full64
VCS_FLAGS += +lint=TFIPC-L
# Additional Optimizations
VCS_FLAGS += -O4
# Multi-threading compilation
VCS_COMPILE_JOBS ?= 16
VCS_FLAGS += -j$(VCS_COMPILE_JOBS)
# Debug Options
VCS_FLAGS += -assert disable_cover

# VCS documents these as compile-time runtime optimizers; -hsopt=gates is
# specifically intended for functional and SDF gate-level designs.  None of
# them enables Fine-Grained Parallelism at simulation time.
ifeq ($(SIM_WITH_NETLIST), 1)
	VCS_FLAGS += -hsopt=gates
	VCS_FLAGS += -Xkeyopt=rtopt
	VCS_FLAGS += +rad
endif

# Independent CI simulations provide the process-level parallelism.  Do not
# add VCS -fgp or Questa -mc2 here: each batch simulation should remain
# single-core-friendly when many jobs are launched concurrently.
# Waveform Visibility
# for looking at the cells
# for looking at the waveform
# huge performace decline when enabled, use only for debugging
ifeq ($(SIM_WITH_WAVEFORM), 1)
	VCS_FLAGS += -kdb
	VLOGAN_FLAGS += -kdb
	VHDLAN_FLAGS += -kdb
	VCS_FLAGS += +vcs+fsdbon
	VCS_FLAGS += -debug_access+all
endif

# Verilated and compiled Occamy system
VLT_AR     = ${VLT_BUILDDIR}/Vtestharness__ALL.a

# VLT_FLAGS += --trace       # Enable VCD dumping
VLT_FLAGS += --prof-cfuncs # Enable code profiling
VLT_FLAGS += -Itestharness  # Include path for testharness utility files

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

TESTHARNESS_TPL_DIR  = testharness/template
TESTHARNESS_TPLS     = $(wildcard $(TESTHARNESS_TPL_DIR)/*.sv.tpl)
TESTHARNESS_OUTPUTS  = testharness/testharness.sv \
                       testharness/io_wrapper.sv \
                       testharness/dut.sv \
                       testharness/util/load_binary_rtl.sv \
                       testharness/util/load_binary_mem_macro.sv \
                       testharness/util/load_binary_netlist.sv \
                       testharness/util/check_finish_rtl.sv \
                       testharness/util/check_finish_mem_macro.sv \
                       testharness/util/check_finish_netlist.sv

ifeq ($(PREPARED_SIM_INPUTS), 1)
.PHONY: check-prepared-vsim-inputs check-prepared-vcs-inputs

PREPARED_VSIM_INPUTS = $(VSIM_BUILDDIR)/compile.vsim.tcl $(TESTHARNESS_OUTPUTS)
PREPARED_VCS_INPUTS  = $(VCS_BUILDDIR)/compile.sh $(TESTHARNESS_OUTPUTS)

check-prepared-vsim-inputs:
	@missing=0; \
	for input in $(PREPARED_VSIM_INPUTS); do \
		if [ ! -s "$$input" ]; then \
			echo "ERROR: PREPARED_SIM_INPUTS=1 requires transferred Questa input: $$input" >&2; \
			missing=1; \
		fi; \
	done; \
	if [ "$$missing" -ne 0 ]; then \
		echo "Run the simulation prepare phase and transfer its generated inputs before backend compilation." >&2; \
		exit 1; \
	fi

check-prepared-vcs-inputs:
	@missing=0; \
	for input in $(PREPARED_VCS_INPUTS); do \
		if [ ! -s "$$input" ]; then \
			echo "ERROR: PREPARED_SIM_INPUTS=1 requires transferred VCS input: $$input" >&2; \
			missing=1; \
		fi; \
	done; \
	if [ "$$missing" -ne 0 ]; then \
		echo "Run the simulation prepare phase and transfer its generated inputs before backend compilation." >&2; \
		exit 1; \
	fi
endif

OCCAMYGEN_TH_CMD = $(OCCAMYGEN) --cfg $(CFG) --outdir testharness \
		--testharness-sv $(TESTHARNESS_TPL_DIR)/testharness.sv.tpl \
		--sim_with_mem_macro $(SIM_WITH_MACRO) \
		--sim_with_netlist $(SIM_WITH_NETLIST) \
		--sim_with_interposer $(SIM_WITH_INTERPOSER) \
		--sim_with_pll $(SIM_WITH_PLL) \
		--sim_with_jtag_check $(SIM_WITH_JTAG_CHECK) \
		--sim_with_verilator 0

OCCAMYGEN_TH_VLT_CMD = $(OCCAMYGEN) --cfg $(CFG) --outdir testharness \
		--testharness-sv $(TESTHARNESS_TPL_DIR)/testharness.sv.tpl \
		--sim_with_mem_macro 0 \
		--sim_with_netlist 0 \
		--sim_with_interposer 0 \
		--sim_with_pll 0 \
		--sim_with_jtag_check 0 \
		--sim_with_verilator 1

# Primary target: generates all testharness outputs in one invocation
testharness/testharness.sv: $(TESTHARNESS_TPLS) $(CFG) $(SIM_CFG)
	@echo "[OCCAMYGEN] Generate testharness files"
	@$(OCCAMYGEN_TH_CMD)

# Secondary outputs are produced by the same recipe
testharness/io_wrapper.sv testharness/dut.sv testharness/util/load_binary_rtl.sv testharness/util/load_binary_mem_macro.sv testharness/util/load_binary_netlist.sv testharness/util/check_finish_rtl.sv testharness/util/check_finish_mem_macro.sv testharness/util/check_finish_netlist.sv: testharness/testharness.sv

clean-testharness:
	rm -rf $(TESTHARNESS_OUTPUTS)
	rm -rf testharness/testharness_*.sv

# Debug target: run generation verbosely (no @-silencing)
debug-testharness-gen:
	$(OCCAMYGEN_TH_CMD)
#############
# Verilator #
#############

# Regenerate testharness for Verilator (no D2D, no interposer)
vlt-testharness: $(TESTHARNESS_TPLS) $(CFG) $(SIM_CFG)
	@echo "[OCCAMYGEN] Generate testharness files (Verilator)"
	@$(OCCAMYGEN_TH_VLT_CMD)

${VLT_AR}: vlt-testharness
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

ifeq ($(PREPARED_SIM_INPUTS), 1)
$(VSIM_BUILDDIR)/compile.vsim.tcl:
	@echo "ERROR: PREPARED_SIM_INPUTS=1 requires transferred Questa input: $@" >&2
	@exit 1
else
$(VSIM_BUILDDIR)/compile.vsim.tcl: | $(VSIM_BUILDDIR)
	$(BENDER) script vsim $(VSIM_BENDER) --vlog-arg="$(VLOG_FLAGS) -work $(dir $@) " > $@
	echo '$(VLOG) -work $(dir $@) $(TB_CC_SOURCES) -vv -ccflags "$(TB_CC_FLAGS)"' >> $@
	echo 'return 0' >> $@
endif

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

ifeq ($(PREPARED_SIM_INPUTS), 1)
$(BIN_DIR)/occamy_chip.vsim: | check-prepared-vsim-inputs
endif

clean-vsim: clean-work
	rm -rf $(BIN_DIR)/$(TARGET).vsim $(BIN_DIR)/$(TARGET).vsim.gui $(VSIM_BUILDDIR) $(BIN_DIR)/vsim.wlf $(BIN_DIR)/transcript transcript


#######
# VCS #
#######

$(VCS_BUILDDIR):
	mkdir -p $@

# The script's contents are a function of $(VCS_BENDER) -- which is read from
# bender_targets.tmp (cfg-dependent) -- and of the bender manifests, so both are real
# prerequisites: switching CFG has to re-emit the script, or VCS analyses the source
# set of a configuration that is no longer the active one.
ifeq ($(PREPARED_SIM_INPUTS), 1)
$(VCS_BUILDDIR)/compile.sh:
	@echo "ERROR: PREPARED_SIM_INPUTS=1 requires transferred VCS input: $@" >&2
	@exit 1
VCS_TESTHARNESS_PREREQUISITES :=
else
$(VCS_BUILDDIR)/compile.sh: $(BENDER_TARGETS_FILE) $(ROOT)/Bender.yml $(ROOT)/Bender.lock | $(VCS_BUILDDIR)
	mkdir -p $(VCS_BUILDDIR)
	${BENDER} script vcs ${VCS_BENDER} --vlog-arg="${VLOGAN_FLAGS}" --vcom-arg="${VHDLAN_FLAGS}" > $@
	chmod +x $@
VCS_TESTHARNESS_PREREQUISITES := $(TESTHARNESS_OUTPUTS)
endif

$(VCS_RUN_SCRIPT): | $(VCS_BUILDDIR)
	echo 'run' > $@

# $(VCS_SOURCES) is the RTL file list bender resolves for these targets. It is a
# prerequisite so that editing any .sv -- or regenerating target/rtl/src -- rebuilds
# the simulation binary.
$(BIN_DIR)/occamy_chip.vcs: $(VCS_BUILDDIR)/compile.sh $(VCS_RUN_SCRIPT) $(VCS_SOURCES) $(VCS_TESTHARNESS_PREREQUISITES)
	bash work-vcs/compile.sh | tee $(dir $<)vcs_compile.log
	mkdir -p $(BIN_DIR)
	$(VCS) $(VCS_FLAGS) -o $(BIN_DIR)/occamy_chip.vcs -cc $(CC) -cpp $(CXX) \
		-override_timescale=1ns/1ps -full64 -ignore initializer_driver_checks testharness $(TB_CC_SOURCES) $(RTL_CC_SOURCES) \
		-CFLAGS "$(TB_CC_FLAGS)" -lutil +error+100 | tee $(dir $<)vcs.log
	@echo "#!/bin/bash" > $@.gui
	@echo 'script_dir="$$( cd -- "$$( dirname -- "$${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"' >> $@.gui
	@echo 'cd $$script_dir' >> $@.gui
	@echo './occamy_chip.vcs -gui -ucli -i ../$(VCS_RUN_SCRIPT) "$$@"' >> $@.gui
	@chmod +x $@.gui

ifeq ($(PREPARED_SIM_INPUTS), 1)
$(BIN_DIR)/occamy_chip.vcs: | check-prepared-vcs-inputs
endif

.PHONY: clean-vcs
clean-vcs: clean-work
	rm -rf $(VCS_BUILDDIR) AN.DB work.lib++ vc_hdrs.h $(BIN_DIR)/$(TARGET).vcs $(BIN_DIR)/$(TARGET).vcs.gui $(BIN_DIR)/inter.fsdb $(BIN_DIR)/$(TARGET).vcs.daidir $(BIN_DIR)/verdi_config_file $(BIN_DIR)/verdiLog $(BIN_DIR)/novas* $(BIN_DIR)/ucli.key vish_stacktrace.vstf

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
