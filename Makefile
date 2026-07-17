.PHONY: clean-repo clean-bender clean-automation FORCE bootrom sw single-sw rtl open_terminal hemaia_system_vivado_preparation \
		hemaia_chip_vivado hemaia_chip_east_vivado hemaia_chip_west_vivado hemaia_chip_vivado_gui \
		hemaia_system_vivado hemaia_system_east_vivado hemaia_system_west_vivado hemaia_system_vivado_gui \
		hemaia_system_east_vivado_gui hemaia_system_vlt occamy_system_vsim_preparation occamy_system_vsim \
		hemaia_system_vsim_preparation hemaia_system_vsim bingo-vis-traces

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(dir $(MKFILE_PATH))

CFG_OVERRIDE ?=

DEFAULT_CFG = $(MKFILE_DIR)target/rtl/cfg/hemaia_ci.hjson
CFG = $(MKFILE_DIR)target/rtl/cfg/lru.hjson
DEFAULT_SIM_CFG = $(MKFILE_DIR)target/sim/cfg/sim_rtl.hjson
SIM_CFG ?= $(DEFAULT_SIM_CFG)
# If the configuration file is overriden on the command-line (through
# CFG_OVERRIDE) and this file differs from the least recently used
# (LRU) config, all targets depending on the configuration file have
# to be rebuilt. This file is used to express this condition as a
# prerequisite for other rules.

$(CFG): FORCE
	@# If the LRU config file doesn't exist, we use the default config.
	@if [ ! -e $@ ] ; then \
		DEFAULT_CFG="$(DEFAULT_CFG)"; \
		echo "Using default config file: $$DEFAULT_CFG"; \
		cp $$DEFAULT_CFG $@; \
	fi
	@# If a config file is provided on the command-line then we override the
	@# LRU file with it -- but only when it actually differs. cp always bumps
	@# the mtime, so an unconditional copy makes repeated invocations with the
	@# same CFG_OVERRIDE (e.g. one per CI task) needlessly retrigger every
	@# $(CFG)-dependent rebuild (SW headers, host.ld, RTL gen).
	@if [ $(CFG_OVERRIDE) ] && ! cmp -s $(CFG_OVERRIDE) $@ ; then \
		echo "Overriding config file with: $(CFG_OVERRIDE)"; \
		cp $(CFG_OVERRIDE) $@; \
	fi
FORCE:

clean-sw:
	$(MAKE) -C ./target/sw/ clean

# Drop the task_<idx>/ run dirs, summaries and __pycache__ that the CI / sweep /
# test flows leave under target/sim/automation.  Kept out of clean-repo on
# purpose: every simulation run starts by calling `make clean`, so folding this in
# would make each run wipe the results of every other flow.  Sweep LUTs/CSVs are
# preserved either way.
clean-automation:
	$(MAKE) -C ./target/sim/automation clean

clean-repo:
	$(MAKE) -C ./target/fpga/hemaia_chip/ clean
	$(MAKE) -C ./target/fpga/hemaia_chip_east_io/ clean
	$(MAKE) -C ./target/fpga/hemaia_chip_west_io/ clean
	$(MAKE) -C ./target/fpga/hemaia_system/ clean
	$(MAKE) -C ./target/fpga/hemaia_system_east/ clean
	$(MAKE) -C ./target/fpga/hemaia_system_west/ clean
	$(MAKE) -C ./target/sw/  clean
	$(MAKE) -C ./target/rtl/bootrom/  clean
	$(MAKE) -C ./target/sim/ clean
	$(MAKE) -C ./target/rtl/ clean
	$(MAKE) -C ./target/tapeout clean
	rm -rf ./target/rtl/src/bender_targets.tmp
	rm -rf ./target/rtl/cfg/lru.hjson
	rm -rf ./target/rtl/cfg/generated
	cd ./target/tapeout && ./0_reset_private_modules.sh

clean-bender:
	rm -f Bender.lock
	rm -rf deps
	rm -rf .bender

clean: clean-repo clean-bender
#######################
# Software Generation #
#######################
# The software from simulation and FPGA prototyping comes from one source.
# Execute in SNAX Docker
DEBUG_LEVEL ?= 0
PERF_TRACING ?= 1

# User flags for SW compilation
# Useful for enabling debug prints or performance tracing
# e.g. make sw DEBUG_LEVEL=1 PERF_TRACING=0
USER_FLAGS = -DBINGO_DEBUG_LEVEL=$(DEBUG_LEVEL)
ifeq ($(PERF_TRACING), 1)
    USER_FLAGS += -DBINGO_PERF_TRACING
endif

# `sw` is the one build that is safe to run in parallel (the RTL gen and the sim
# compile are not), and it is the slow one, so it builds with -j by default. SW_JOBS
# sets the job count -- defaults to the host's core count; SW_JOBS=1 serialises. The -j
# is applied only to this sub-make, so it never leaks to a full `make` that also builds
# RTL or the simulator.
SW_JOBS ?= $(shell nproc 2>/dev/null || echo 8)

sw: $(CFG)
	$(MAKE) -j$(SW_JOBS) -C ./target/sw sw CFG=$(CFG) USER_FLAGS="$(USER_FLAGS)"

# Single SW compilation (by category and chip type for precise control)
# - HOST_APP_TYPE: The category of the host application.
#   Options: host_only, offload_legacy, offload_bingo_sw, offload_bingo_hw
# - CHIP_TYPE: The target platform configuration.
#   Options: single_chip, multi_chip
# - WORKLOAD: The specific workload or application name.
#   Note: Set to 'None' for offload_legacy as it does not have workloads, it only has device applications.
# - DEV_APP: The target device-side accelerator/application.
#   Note: Set to 'None' for host_only as it does not target a device-side accelerator.
#         For bingo workloads, set to 'snax-bingo-offload'.
HOST_APP_TYPE ?= offload_legacy
CHIP_TYPE     ?= single_chip
WORKLOAD      ?= None
DEV_APP       ?= versacore-matmul-profile-1cluster-4chip
single-sw: $(CFG)
	$(MAKE) -C ./target/sw single-sw \
		USER_FLAGS="$(USER_FLAGS)" \
		HOST_APP_TYPE=$(HOST_APP_TYPE) \
		CHIP_TYPE=$(CHIP_TYPE) \
		WORKLOAD=$(WORKLOAD) \
		DEV_APP=$(DEV_APP)



#########################
# App Binary Generation #
#########################
# This target prepares the software binaries for simulation and/or FPGA execution.
apps: $(CFG)
	$(MAKE) -C ./target/sim apps CFG=$(CFG) USER_FLAGS="$(USER_FLAGS)" \
		HOST_APP_TYPE=$(HOST_APP_TYPE) \
		CHIP_TYPE=$(CHIP_TYPE) \
		WORKLOAD=$(WORKLOAD) \
		DEV_APP=$(DEV_APP)

# Generate the Bingo Perfetto trace from simulation logs via target/sim.
bingo-vis-traces:
	$(MAKE) -C ./target/sim bingo-vis-traces CFG=$(CFG) SIM_CFG=$(SIM_CFG)

######################
# Bootrom Generation #
######################

bootrom: # In SNAX Docker
# Here will generate two bootroms:
# 1. The bootrom used for simulation (light-weight bootrom)
# 2. The bootrom used for tapeout / FPGA prototyping (embedded real rom, full-functional bootrom with different frequency settings)
	$(MAKE) -C ./target/rtl/bootrom all

# Hardware Generation
# In SNAX Docker
rtl: $(CFG)
	$(MAKE) -C ./target/rtl/ rtl CFG=$(CFG)

####################
# Tapeout Workflow #
####################

.PHONY: tapeout_syn_flist tapeout_preparation

tapeout_preparation: rtl tapeout_syn_flist

# Generating filelist per cluster
# Needed for a per-cluster synthesis
tapeout_syn_flist:
	$(MAKE) -C ./target/tapeout/ syn_gen_list CFG=$(CFG)

#################
# FPGA Workflow #
#################

# FPGA Workflow
# Please be attention that in this configuration, injecting any binary files by Xilinx Vivado are not possible anymore; please use JTAG or embedded bootrom to load the binary
hemaia_system_vivado_preparation: # In SNAX Docker
	$(MAKE) -C ./target/fpga/hemaia_system/ define_defines_includes_no_simset.tcl
	$(MAKE) -C ./target/fpga/hemaia_chip/ define-sources.tcl

hemaia_chip_vivado:	# In ESAT Server
	$(MAKE) -C ./target/fpga/hemaia_chip hemaia_chip

hemaia_chip_east_vivado:	# In ESAT Server
	$(MAKE) -C ./target/fpga/hemaia_chip_east_io hemaia_chip_east

hemaia_chip_west_vivado:	# In ESAT Server
	$(MAKE) -C ./target/fpga/hemaia_chip_west_io hemaia_chip_west

hemaia_chip_vivado_gui: # In ESAT Server
	sh -c "cd ./target/fpga/hemaia_chip/hemaia_chip/;vivado hemaia_chip.xpr"

hemaia_system_vivado: hemaia_chip_vivado # In ESAT Server
	$(MAKE) -C ./target/fpga/hemaia_system hemaia_system

hemaia_system_east_vivado: hemaia_chip_east_vivado # In ESAT Server
	$(MAKE) -C ./target/fpga/hemaia_system_east hemaia_system_east

hemaia_system_west_vivado: hemaia_chip_west_vivado # In ESAT Server
	$(MAKE) -C ./target/fpga/hemaia_system_west hemaia_system_west

hemaia_system_vivado_gui: # In ESAT Server
	sh -c "cd ./target/fpga/hemaia_system/hemaia_system/;vivado hemaia_system.xpr"

hemaia_system_east_vivado_gui: # In ESAT Server
	sh -c "cd ./target/fpga/hemaia_system_east/hemaia_system_east/;vivado hemaia_system_east.xpr"

######################
# Verilator Workflow #
######################

hemaia_system_vlt: # In SNAX Docker
	$(MAKE) -C ./target/sim bin/occamy_chip.vlt CFG=$(CFG) SIM_CFG=$(SIM_CFG)

#####################
# Questasim Workflow #
######################

hemaia_system_vsim_preparation: $(CFG) # In SNAX Docker
	$(MAKE) -C ./target/sim testharness/testharness.sv CFG=$(CFG) SIM_CFG=$(SIM_CFG)
	$(MAKE) -C ./target/sim work-vsim/compile.vsim.tcl CFG=$(CFG) SIM_CFG=$(SIM_CFG)

hemaia_system_vsim: # In ESAT Server
	$(MAKE) -C ./target/sim bin/occamy_chip.vsim SIM_CFG=$(SIM_CFG)

################
# VCS Workflow #
################

hemaia_system_vcs_preparation: $(CFG) # In SNAX Docker
	$(MAKE) -C ./target/sim testharness/testharness.sv CFG=$(CFG) SIM_CFG=$(SIM_CFG)
	$(MAKE) -C ./target/sim work-vcs/compile.sh CFG=$(CFG) SIM_CFG=$(SIM_CFG)

hemaia_system_vcs: # In ESAT Server
	$(MAKE) -C ./target/sim bin/occamy_chip.vcs SIM_CFG=$(SIM_CFG)
# Single-core-friendly batch run: cd ./target/sim/bin; ./occamy_chip.vcs | tee run.log
