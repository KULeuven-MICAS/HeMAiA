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
CFG_OVERRIDE ?=     # Override default config file

.DEFAULT_GOAL := help
.PHONY: all clean
all: rtl sw addrmap
clean: clean-vsim clean-vlt clean-apps

############
# Makefrag #
############

# Common definitions used by included Makefiles

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
SIM_MKFILE_DIR  := $(dir $(MKFILE_PATH))
ROOT        := $(SIM_MKFILE_DIR)../..
SNITCH_ROOT := $(shell bender path snitch_cluster)

TARGET = occamy_chip

CVA6_TXT_TRACE       = $(LOGS_DIR)/trace_hart_00000.txt
CVA6_PERF_DUMP       = $(LOGS_DIR)/hart_00000_perf.json
CVA6_ANNOTATED_TRACE = $(LOGS_DIR)/trace_hart_00000.s
TXT_TRACES          += $(CVA6_TXT_TRACE)
PERF_DUMPS          += $(CVA6_PERF_DUMP)
ANNOTATED_TRACES    += $(CVA6_ANNOTATED_TRACE)

include $(SIM_MKFILE_DIR)/sim.mk

#########################
# Files and directories #
#########################

BIN_DIR              ?= bin
TARGET_TEST_DIR      ?= $(ROOT)/target/rtl/test
CVA6_TRACE            = trace_hart_0.log

#####################
# Simulator options #
#####################

VSIM_BENDER += $(shell cat ../rtl/src/bender_targets.tmp)
VLT_BENDER += $(shell cat ../rtl/src/bender_targets.tmp)

VLT_BENDER  += -t SYNTHESIS_SNAX_DEV_ONLY

VSIM_BENDER += -t simulation_hemaia
VLT_BENDER += -t simulation_hemaia

VSIM_FLAGS += -t 1ps
VSIM_FLAGS += -voptargs=+acc
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

# Verilated and compiled Occamy system
VLT_AR     = ${VLT_BUILDDIR}/Vtestharness__ALL.a

VLT_FLAGS += -Wno-SYMRSVDWORD -Wno-BLKLOOPINIT
VLT_FLAGS += --trace       # Enable VCD dumping
VLT_FLAGS += --prof-cfuncs # Enable code profiling

###############
# C testbench #
###############

TB_CC_SOURCES += $(TARGET_TEST_DIR)/uartdpi/uartdpi.c

TB_CC_FLAGS  = -std=c++17
TB_CC_FLAGS +=-I$(TARGET_TEST_DIR)/uartdpi

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


##########
# Traces #
##########

CVA6_BINARY ?= $(shell cat $(LOGS_DIR)/.rtlbinary)

CVA6_TXT_TRACE       = $(LOGS_DIR)/trace_hart_00000.txt
CVA6_PERF_TRACE      = $(LOGS_DIR)/hart_00000_perf.json
CVA6_ANNOTATED_TRACE = $(LOGS_DIR)/trace_hart_00000.s
CVA6_DIFF_TRACE      = $(LOGS_DIR)/trace_hart_00000.diff

TXT_TRACES       += $(CVA6_TXT_TRACE)
PERF_TRACES      += $(CVA6_PERF_TRACE)
ANNOTATED_TRACES += $(CVA6_ANNOTATED_TRACE)
DIFF_TRACES      += $(CVA6_DIFF_TRACE)

traces: $(CVA6_TXT_TRACE) $(CVA6_PERF_TRACE)
annotate: $(CVA6_ANNOTATED_TRACE)

# CVA6 traces require different handling
$(CVA6_PERF_TRACE): $(CVA6_TXT_TRACE) $(EVENTS_PY)
	$(PYTHON) $(EVENTS_PY) -f cva6 $< -o $@
$(CVA6_ANNOTATED_TRACE): $(CVA6_TXT_TRACE) $(ANNOTATE_PY)
	$(PYTHON) $(ANNOTATE_PY) $(ANNOTATE_FLAGS) -o $@ $(CVA6_BINARY) $<
$(CVA6_DIFF_TRACE): $(CVA6_TXT_TRACE) $(ANNOTATE_PY)
	$(PYTHON) $(ANNOTATE_PY) $(ANNOTATE_FLAGS) -o $@ $(CVA6_BINARY) $< -d

############
### Apps ###
############
.PHONY: apps clean-apps
apps: 
	$(MAKE) -C apps apps

clean-apps:
	$(MAKE) -C apps clean

#############
# Verilator #
#############

${VLT_AR}: ${VLT_SOURCES} ${TB_SRCS}
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

$(VLT_BUILDDIR)/test/uartdpi/uartdpi.o: $(TARGET_TEST_DIR)/uartdpi/uartdpi.c
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

$(VSIM_BUILDDIR)/compile.vsim.tcl: $(BENDER_LOCK) | $(VSIM_BUILDDIR)
	$(BENDER) script vsim $(VSIM_BENDER) --vlog-arg="$(VLOG_FLAGS) -work $(dir $@) " > $@
	echo '$(VLOG) -work $(dir $@) $(TB_CC_SOURCES) -vv -ccflags "$(TB_CC_FLAGS)"' >> $@
	echo 'return 0' >> $@

$(BIN_DIR):
	mkdir -p $@

.PHONY: $(BIN_DIR)/occamy_chip.vsim

$(BIN_DIR)/occamy_chip.vsim: $(VSIM_BUILDDIR)/compile.vsim.tcl |$(BIN_DIR)
	touch $@
	vsim -c -do "source $<; quit" | tee $(dir $<)vlog.log
	@! grep -P "Errors: [1-9]*," $(dir $<)vlog.log
	vopt $(VOPT_FLAGS) -work $(VSIM_BUILDDIR) testharness -o testharness_opt | tee $(dir $<)vopt.log
	@! grep -P "Errors: [1-9]*," $(dir $<)vopt.log
	@echo "#!/bin/bash" > $@
	@echo 'vsim +permissive $(VSIM_FLAGS) $$2 -work $(SIM_MKFILE_DIR)/$(VSIM_BUILDDIR) -c \
				-ldflags "-Wl,-rpath,-L -lutil" \
				testharness_opt +permissive-off ++$$1' >> $@
	@# Rename CVA6 trace to align with Snitch trace names
	@echo "mv ${CVA6_TRACE} $(LOGS_DIR)/trace_hart_00000.txt" >> $@
	@# Move all other traces to logs folder as well
	@echo "mv *trace*.log $(LOGS_DIR)/" >> $@
	@chmod +x $@
	@echo "#!/bin/bash" > $@.gui
	@echo 'vsim +permissive $(VSIM_FLAGS) -work $(SIM_MKFILE_DIR)/$(VSIM_BUILDDIR) \
				-ldflags "-Wl,-rpath,-L -lutil" \
				testharness_opt +permissive-off ++$$1' >> $@.gui
	@echo "mv ${CVA6_TRACE} $(LOGS_DIR)/trace_hart_00000.txt" >> $@.gui
	@echo "mv *trace*.log $(LOGS_DIR)/" >> $@.gui
	@chmod +x $@.gui

clean-vsim: clean-work
	rm -rf $(BIN_DIR)/$(TARGET).vsim $(BIN_DIR)/$(TARGET).vsim.gui $(VSIM_BUILDDIR) vsim.wlf transcript

########
# Util #
########

.PHONY: clean-work clean-logs help

clean-work:
	rm -rf work
	rm -rf work-vsim

clean-logs:
	rm -rf $(LOGS_DIR)/

# Help command
Blue=\033[1;34m
Black=\033[0m
help:
	@echo -e "${Blue}Makefile Targets${Black} for the ${Blue}Snitch Cluster${Black}"
	@echo -e "Use 'make <target>' where <target> is one of:"
	@echo -e ""
	@echo -e "${Blue}help           ${Black}Show an overview of all Makefile targets."
	@echo -e ""
	@echo -e "${Blue}bin/$(TARGET).vcs  ${Black}Build compilation script and compile all sources for VCS simulation. @IIS: vcs-2020.12 make bin/$(TARGET).vcs"
	@echo -e "${Blue}bin/$(TARGET).vlt  ${Black}Build compilation script and compile all sources for Verilator simulation."
	@echo -e "${Blue}bin/$(TARGET).vsim ${Black}Build compilation script and compile all sources for Questasim simulation. @IIS: QUESTA_HOME=/usr/pack/modelsim-10.7b-kgf/questasim/ CC=$(QUESTA_HOME)/gcc-5.3.0-linux_x86_64/bin/gcc CXX=$(QUESTA_HOME)/gcc-5.3.0-linux_x86_64/bin/g++ LD=$(QUESTA_HOME)/gcc-5.3.0-linux_x86_64/bin/ld make bin/$(TARGET).vsim"
	@echo -e ""
	@echo -e "${Blue}all            ${Black}Update all SW and HW related sources (by, e.g., re-generating the RegGen registers and their c-header files)."
	@echo -e "${Blue}fpga           ${Black}Build a small Occamy version (CVA6 + 1xcluster) for the VCU128 FPGA board."
	@echo -e ""
	@echo -e "${Blue}addrmap        ${Black}Generate the addressmap in the following formats: MD, PNG, TEX, PDF."
	@echo -e ""
	@echo -e "${Blue}clean          ${Black}Clean everything except traces in logs directory."
	@echo -e "${Blue}clean-bender   ${Black}Clean all Bender generated files."
	@echo -e "${Blue}clean-sw       ${Black}Clean all software."
	@echo -e "${Blue}clean-addrmap  ${Black}Delete all addrmap outputs."
	@echo -e "${Blue}clean-logs     ${Black}Delete all traces in logs directory."
	@echo -e "${Blue}clean-vcs      ${Black}Clean all build directories and temporary files for VCS simulation."
	@echo -e "${Blue}clean-vlt      ${Black}Clean all build directories and temporary files for Verilator simulation."
	@echo -e "${Blue}clean-vsim     ${Black}Clean all build directories and temporary files for Questasim simulation."
	@echo -e ""
	@echo -e "${Blue}sw             ${Black}Build all software."
	@echo -e "${Blue}rtl            ${Black}Build all auto-generated RTL sources."
	@echo -e ""
	@echo -e "Additional useful targets from the included Makefrag:"
	@echo -e "${Blue}traces         ${Black}Generate the better readable traces in .logs/trace_hart_<hart_id>.txt with spike-dasm."
	@echo -e "${Blue}annotate       ${Black}Annotate the better readable traces in .logs/trace_hart_<hart_id>.s with the source code related with the retired instructions."