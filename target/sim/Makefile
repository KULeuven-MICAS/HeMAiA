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
HBM_DPI_DIR          ?= $(ROOT)/hw/hemaia/hemaia_mem_system/hbm

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
# zero-delay simulation.  Keep zero-delay specify paths so Questa can evaluate
# optimized standard cells in the same delta with -cellimmeval.  Timing checks
# remain disabled separately by +notimingchecks in sim.mk.  Dead-cell
# elimination is reserved for non-waveform batches so a debug build keeps every
# otherwise-unused object available for inspection.
ifeq ($(SIM_WITH_NETLIST), 1)
	# Match VCS +vcs+initreg+random for register initialization.  SRAM contents
	# are initialized through the TSMC model's own tasks below, not +initmem.
	# A minimal delay on undelayed sequential UDPs prevents zero-delay GLS races.
	VOPT_FLAGS += +initreg
	VOPT_FLAGS += -add_seq_delay 1ps
	# Questa 2022.4 can optimize away packed-struct variables that remain
	# referenced by the RTL memory-chip model.  Preserve variables only; full
	# +acc is still reserved for waveform/debug builds.
	VOPT_FLAGS += +acc=r
	VSIM_FLAGS += +initreg+1
	VSIM_FLAGS += -cellimmeval
ifeq ($(SIM_WITH_WAVEFORM), 0)
	VOPT_FLAGS += -dce
endif
endif

VSIM_FLAGS += -t 1ps
VSIM_FLAGS += -do "$(VSIM_DO_CMD)"
# Downgraded from error as VSIM *wrongly* reports illegal enum assignments in riscv_dbg 4-phase CDCs
# TODO: Workaround; revise
VSIM_FLAGS += -warning 8386

# Treat every RTL module-port width mismatch found during elaboration as an
# error.  Synthesized gate netlists routinely leave unused cell outputs open,
# so netlist CI must retain Questa's default non-fatal severity.
ifneq ($(SIM_WITH_NETLIST), 1)
VOPT_FLAGS += -error 2241
endif

VLOG_FLAGS += -svinputport=compat
VLOG_FLAGS += -override_timescale 1ns/1ps
VLOG_FLAGS += -timescale 1ns/1ps
VLOG_FLAGS += -suppress 2583
VLOG_FLAGS += -suppress 13314
VLOG_FLAGS += ${QUESTA_64BIT}

# Initialize SRAMs through the supported TSMC functional-model hooks.  These
# defines must be present while the macro model is compiled, for both Questa's
# vlog flow and VCS's vlogan flow.
ifeq ($(SIM_WITH_NETLIST), 1)
ifeq ($(SIM_WITH_MACRO), 1)
	TSMC_MEMORY_INIT_DEFINES := +define+TSMC_INITIALIZE_MEM_USING_DEFAULT_TASKS +define+TSMC_MEM_LOAD_0
	VLOG_FLAGS += $(TSMC_MEMORY_INIT_DEFINES)
	VLOGAN_FLAGS += $(TSMC_MEMORY_INIT_DEFINES)
endif
endif


VCS_FLAGS += -Mdir=work-vcs
VCS_FLAGS += -Mlib=work-vcs
VCS_FLAGS += -full64

# Synthesized gate netlists legitimately omit unused standard-cell outputs.
# TFIPC-L is useful for catching incomplete RTL instantiations, but promoting it
# in GLS turns every intentionally open output into a fatal elaboration error.
ifeq ($(SIM_WITH_NETLIST), 1)
VCS_FLAGS += +lint=PCWM-L
VCS_FLAGS += -error=PCWM-L
else
VCS_FLAGS += +lint=PCWM-L,TFIPC-L
VCS_FLAGS += -error=PCWM-L,TFIPC-L
endif
# Additional Optimizations
VCS_FLAGS += -O4
# Multi-threading compilation
VCS_COMPILE_JOBS ?= 16
VCS_FLAGS += -j$(VCS_COMPILE_JOBS)
# Debug Options
VCS_FLAGS += -assert disable_cover

# Fine-Grained Parallelism (FGP), RTL and netlist alike. FGP splits the events
# of a time step -- always blocks, assignments and gates -- over worker threads,
# so it pays when much of the design is active each cycle (all clusters
# computing) and can lose on a mostly idle one. The thread count is a runtime
# choice: simv -fgp=num_threads:N takes N + 1 cores (the sim runner's --fgp),
# and without it the same binary runs on one core, so one build serves every
# per-job core count. VCS_FGP=1 builds with it. An FGP build simulates the split cluster's narrow
# TCDM crossbar wrongly (its bank arbiter returns an X index for a defined one-hot request, which a
# build with debug access or without -fgp does not); the reproducer is snax
# hw/snitch_cluster/test/snitch_tcdm_xbar_x_tb.sv. RTL builds therefore default to off.
VCS_FGP ?= 0
ifeq ($(VCS_FGP), 1)
	VCS_FLAGS += -fgp
endif

# SIM_D2D_PHY_BEHAVIORAL=1 simulates the D2D PHY's RX clock delay line as a behavioral model (every
# tap the RX clock 20 ps later: hemaia_d2d_link src/phy/phy_analog/sim/clk_delay_lines_behavioral.sv)
# instead of its netlist, whose 32 taps of 5 ps and 100 ps cells switch at dozens of distinct times
# on every link clock edge, each a simulation time step. The tap the link registers select at reset
# samples the same.
SIM_D2D_PHY_BEHAVIORAL ?= 0
ifeq ($(SIM_D2D_PHY_BEHAVIORAL), 1)
	VLOGAN_FLAGS += +define+HEMAIA_D2D_PHY_BEHAVIORAL
endif

# SIM_D2D_PHY_CORE_CLOCK=1 simulates every D2D PHY as its core-clock model (hemaia_d2d_link
# src/phy/sim/hemaia_d2d_link_phy_model.sv): the PHY and forwarded-clock cycles stepped in batches
# on the digital clock, the same digital-side behaviour cycle for cycle as the PHY with
# SIM_D2D_PHY_BEHAVIORAL=1, without a simulation time step per link clock edge. It assumes the
# digital clock is 1/7 of the PHY clock, as sim_clock.core_clk_div = 7 sets.
SIM_D2D_PHY_CORE_CLOCK ?= 0
ifeq ($(SIM_D2D_PHY_CORE_CLOCK), 1)
	VLOGAN_FLAGS += +define+HEMAIA_D2D_PHY_CORE_CLOCK
endif
# The models of a link pass their records as a signal the testharness wires (HEMAIA_D2D_PHY_MODEL_XMR):
# each model can then run on its own FGP thread. SIM_D2D_PHY_MODEL_XMR=0: the shared record table.
SIM_D2D_PHY_MODEL_XMR ?= 1
ifeq ($(SIM_D2D_PHY_CORE_CLOCK)$(SIM_D2D_PHY_MODEL_XMR), 11)
	VLOGAN_FLAGS += +define+HEMAIA_D2D_PHY_MODEL_XMR
endif

# SIM_PROFILE=1 compiles the simv for VCS's unified simulation profiler (-simprofile): run it with
# `-simprofile time` (optionally -simprofile_start/_stop <ns>), then `profrpt -view time_all
# simprofile_dir`. Profile a build without -fgp: the profiler attributes a single thread's time.
SIM_PROFILE ?= 0
ifeq ($(SIM_PROFILE), 1)
	VCS_FLAGS += -simprofile
endif

# SIM_NO_TRACE=1 compiles the Snitch instruction and DMA traces out (+define+SNITCH_NO_TRACE,
# SNITCH_NO_DMA_TRACE in snitch_cc.sv): no logs/trace_chip_*_hart_*.dasm, no dma_trace_*.log. The
# design behaves the same; tools that read the .dasm traces (sweep/llm_blocks/ffn/trace_ffn.py) have nothing
# to read, the [BINGO EV] log stays.
SIM_NO_TRACE ?= 0
ifeq ($(SIM_NO_TRACE), 1)
	VLOGAN_FLAGS += +define+SNITCH_NO_TRACE +define+SNITCH_NO_DMA_TRACE
endif

# SIM_LEAN_PROBES=1 binds only the simulation probes the sweep tools read
# (target/sim/testharness/soc_probes.sv); the others are pure monitors.
SIM_LEAN_PROBES ?= 0
ifeq ($(SIM_LEAN_PROBES), 1)
	VLOGAN_FLAGS += +define+HEMAIA_SIM_LEAN_PROBES
endif

# SIM_FAST_ARB=1 simulates every rr_arb_tree as common_cells' fast_sim model (bender target fast_sim:
# src/fast_sim/rr_arb_tree.sv, the same module described procedurally instead of as a tree of nodes).
SIM_FAST_ARB ?= 0
ifeq ($(SIM_FAST_ARB), 1)
	VSIM_BENDER += -t fast_sim
	VLT_BENDER  += -t fast_sim
	VCS_BENDER  += -t fast_sim
endif

# SIM_NO_ASSERT=1 compiles the common_cells assertion macros out (ASSERTS_OFF, COMMON_CELLS_ASSERTS_OFF):
# the design behaves the same, its SystemVerilog assertions are no longer checked.
SIM_NO_ASSERT ?= 0
ifeq ($(SIM_NO_ASSERT), 1)
	VLOGAN_FLAGS += +define+ASSERTS_OFF +define+COMMON_CELLS_ASSERTS_OFF
endif

# VCS_EXTRA_FLAGS: further vcs (elaboration) flags, e.g. +rad or -Xkeyopt=rtopt
VCS_EXTRA_FLAGS ?=
VCS_FLAGS += $(VCS_EXTRA_FLAGS)

# VCS documents these as compile-time runtime optimizers; -hsopt=gates is
# specifically intended for functional and SDF gate-level designs.
ifeq ($(SIM_WITH_NETLIST), 1)
	VCS_FLAGS += +notimingcheck
	# Enable time-zero random 0/1 initialization.  VCS also requires the same
	# option when simv is launched; the CI runner supplies that runtime option.
	VCS_FLAGS += +vcs+initreg+random
	VCS_RUNTIME_FLAGS += +vcs+initreg+random
	# Mirror Questa's per-run transcript for interactive VCS launches.
	VCS_RUNTIME_FLAGS += -l transcript
	VCS_RUNTIME_FLAGS += +vcs+flush+log
	# Preserve synchronous ordering in zero-delay functional GLS by adding a
	# minimal clock-to-Q delay to delay-less sequential UDPs.  This is not SDF.
	VCS_FLAGS += -add_seq_delay 1ps
	VCS_FLAGS += -hsopt=gates
	VCS_FLAGS += -Xkeyopt=rtopt
	VCS_FLAGS += +rad
else
	# Preserve the historical non-netlist behavior.
	VCS_FLAGS += +nospecify
endif

# Waveform Visibility
# for looking at the cells
# for looking at the waveform
# huge performace decline when enabled, use only for debugging
# +vcs+fsdbon dumps the whole design from time 0. SIM_FSDB_AUTO=0 builds the same debug access
# without it, for a dump the run starts itself (UCLI fsdbDumpvars on a scope and time window).
# SIM_DEBUG_ACCESS is appended to -debug_access: +all (write, force, line debug) by default;
# empty gives read access, which is enough for dumping and costs much less simulation speed.
SIM_FSDB_AUTO ?= 1
SIM_DEBUG_ACCESS ?= +all
ifeq ($(SIM_WITH_WAVEFORM), 1)
	VCS_FLAGS += -kdb
	VLOGAN_FLAGS += -kdb
	VHDLAN_FLAGS += -kdb
ifeq ($(SIM_FSDB_AUTO), 1)
	VCS_FLAGS += +vcs+fsdbon
endif
	VCS_FLAGS += -debug_access$(SIM_DEBUG_ACCESS)
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
# Storage of the memchip's simulated HBM (hemaia_hbm_model.sv)
TB_CC_SOURCES += $(HBM_DPI_DIR)/hemaia_hbm_dpi.cc

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

# Backend-only split runs may use a derived copy of the immutable preparation
# script to add technology-model compile defines without invalidating its handoff
# manifest.  Normal builds keep using Bender's generated file directly.
VSIM_COMPILE_INPUT ?= $(VSIM_BUILDDIR)/compile.vsim.tcl

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

$(BIN_DIR)/occamy_chip.vsim: $(VSIM_COMPILE_INPUT) |$(BIN_DIR)
	touch $@
	vsim -c -do "source $(VSIM_COMPILE_INPUT); quit" | tee $(VSIM_BUILDDIR)/vlog.log
	@! grep -P "Errors: [1-9]*," $(VSIM_BUILDDIR)/vlog.log
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

# The analysis flags and the bender targets are baked into compile.sh too (VLOGAN_FLAGS:
# SIM_D2D_PHY_BEHAVIORAL's define, ...; VCS_BENDER: SIM_FAST_ARB's fast_sim, ...). A stamp holds
# them, rewritten only when they change, so turning such a switch on or off re-emits the script
# instead of reusing one analysed without (or with) it.
VCS_VLOGAN_STAMP := $(VCS_BUILDDIR)/vlogan_flags.stamp
$(VCS_VLOGAN_STAMP): vcs_vlogan_flags_check | $(VCS_BUILDDIR)
	@echo '$(VLOGAN_FLAGS) $(VCS_BENDER)' | cmp -s - $@ || echo '$(VLOGAN_FLAGS) $(VCS_BENDER)' > $@
.PHONY: vcs_vlogan_flags_check
vcs_vlogan_flags_check: ;

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
$(VCS_BUILDDIR)/compile.sh: $(BENDER_TARGETS_FILE) $(ROOT)/Bender.yml $(ROOT)/Bender.lock $(VCS_VLOGAN_STAMP) | $(VCS_BUILDDIR)
	mkdir -p $(VCS_BUILDDIR)
	${BENDER} script vcs ${VCS_BENDER} --vlog-arg="${VLOGAN_FLAGS}" --vcom-arg="${VHDLAN_FLAGS}" > $@
	chmod +x $@
VCS_TESTHARNESS_PREREQUISITES := $(TESTHARNESS_OUTPUTS)
endif

# See VSIM_COMPILE_INPUT above.  The runner sets this only for a prepared
# backend handoff; ordinary builds execute the generated script unchanged.
VCS_COMPILE_INPUT ?= $(VCS_BUILDDIR)/compile.sh

$(VCS_RUN_SCRIPT): | $(VCS_BUILDDIR)
	echo 'run' > $@

# $(VCS_SOURCES) is the RTL file list bender resolves for these targets. It is a
# prerequisite so that editing any .sv -- or regenerating target/rtl/src -- rebuilds
# the simulation binary.
$(BIN_DIR)/occamy_chip.vcs: $(VCS_COMPILE_INPUT) $(VCS_RUN_SCRIPT) $(VCS_SOURCES) $(VCS_TESTHARNESS_PREREQUISITES)
	bash $(VCS_COMPILE_INPUT) | tee $(VCS_BUILDDIR)/vcs_compile.log
	mkdir -p $(BIN_DIR)
	$(VCS) $(VCS_FLAGS) -o $(BIN_DIR)/occamy_chip.vcs -cc $(CC) -cpp $(CXX) \
		-override_timescale=1ns/1ps -full64 -ignore initializer_driver_checks testharness $(TB_CC_SOURCES) $(RTL_CC_SOURCES) \
		-CFLAGS "$(TB_CC_FLAGS)" -lutil +error+100 | tee $(dir $<)vcs.log
	@echo "#!/bin/bash" > $@.gui
	@echo 'script_dir="$$( cd -- "$$( dirname -- "$${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"' >> $@.gui
	@echo 'cd $$script_dir' >> $@.gui
	@echo './occamy_chip.vcs $(VCS_RUNTIME_FLAGS) -gui -ucli -i ../$(VCS_RUN_SCRIPT) "$$@"' >> $@.gui
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
