.PHONY: clean FORCE bootrom sw rtl occamy_ip_vcu128 occamy_ip_vcu128_gui occamy_system_vcu128 \
		occamy_system_vcu128_gui occamy_system_download_sw open_terminal hemaia_system_vivado_preparation \
		hemaia_chip_vcu128 hemaia_chip_vcu128_gui hemaia_system_vcu128 hemaia_system_vcu128_gui \
		occamy_system_vlt occamy_system_vsim_preparation occamy_system_vsim hemaia_system_vsim_preparation \
		hemaia_system_vsim

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(dir $(MKFILE_PATH))

CFG_OVERRIDE ?= 
DEFAULT_CFG = $(MKFILE_DIR)target/rtl/cfg/hemaia_ci.hjson
CFG = $(MKFILE_DIR)target/rtl/cfg/lru.hjson
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
	@# If a config file is provided on the command-line 
	@# then we override the LRU file with it
	@if [ $(CFG_OVERRIDE) ] ; then \
		echo "Overriding config file with: $(CFG_OVERRIDE)"; \
		cp $(CFG_OVERRIDE) $@; \
	fi
FORCE:

clean-sw: 
	$(MAKE) -C ./target/sw/ clean

clean:
	$(MAKE) -C ./target/fpga/ clean
	$(MAKE) -C ./target/fpga/vivado_ips/ clean
	$(MAKE) -C ./target/fpga_chip/hemaia_chip/ clean
	$(MAKE) -C ./target/fpga_chip/hemaia_chip_east_io/ clean
	$(MAKE) -C ./target/fpga_chip/hemaia_system/ clean
	$(MAKE) -C ./target/fpga_chip/hemaia_system_east/ clean
	$(MAKE) -C ./target/sw/  clean
	$(MAKE) -C ./target/rtl/bootrom/  clean
	$(MAKE) -C ./target/sim/ clean
	$(MAKE) -C ./target/rtl/ clean
	$(MAKE) -C ./target/fpga/sw clean
	$(MAKE) -C ./target/tapeout clean
	rm -rf Bender.lock .bender deps
	rm -rf ./target/rtl/src/bender_targets.tmp
	rm -rf ./target/rtl/cfg/lru.hjson

#######################
# Software Generation #
#######################
# The software from simulation and FPGA prototyping comes from one source.
# Execute in SNAX Docker
sw: $(CFG)
	$(MAKE) -C ./target/sw sw CFG=$(CFG)


#########################
# App Binary Generation #
#########################
HOST_APP ?= offload
DEVICE_APP ?= snax-test-integration
apps:
	$(MAKE) -C ./target/sim apps CFG=$(CFG) HOST_APP=$(HOST_APP) DEVICE_APP=$(DEVICE_APP)
 
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

occamy_system_vivado_preparation: # In SNAX Docker
	$(MAKE) -C ./target/fpga/ define_defines_includes_no_simset.tcl
	$(MAKE) -C ./target/fpga/vivado_ips/ define-sources.tcl

occamy_system_vivado: occamy_ip_vcu128 # In ESAT Server
	$(MAKE) -C ./target/fpga occamy_vcu128

occamy_system_download_sw: # In ESAT Server; this procedure will only inject the bootrom at present; however, it can also inject the software.
	$(MAKE) -C ./target/fpga/sw download_sw

open_terminal:	# It opens ttyUSB1 without locking it, and set baudrate at 1Mbps
	$(info "shell minicom -D /dev/ttyUSB1 -b 1000000 -o")

# FPGA Workflow (with no Xilinx IP - tapeout configuration)
# Please be attention that in this configuration, injecting any binary files by Xilinx Vivado are not possible anymore; please use JTAG or embedded bootrom to load the binary
hemaia_system_vivado_preparation: # In SNAX Docker
	$(MAKE) -C ./target/fpga_chip/hemaia_system/ define_defines_includes_no_simset.tcl
	$(MAKE) -C ./target/fpga_chip/hemaia_chip/ define-sources.tcl

hemaia_chip_vivado:	# In ESAT Server
	$(MAKE) -C ./target/fpga_chip/hemaia_chip hemaia_chip

hemaia_chip_east_vivado:	# In ESAT Server
	$(MAKE) -C ./target/fpga_chip/hemaia_chip_east_io hemaia_chip_east

hemaia_chip_vivado_gui: # In ESAT Server
	sh -c "cd ./target/fpga/fpga_chip/hemaia_chip/hemaia_chip/;vivado hemaia_chip.xpr"

hemaia_system_vivado: hemaia_chip_vivado # In ESAT Server
	$(MAKE) -C ./target/fpga_chip/hemaia_system hemaia_system

hemaia_system_east_vivado: hemaia_chip_east_vivado # In ESAT Server
	$(MAKE) -C ./target/fpga_chip/hemaia_system_east hemaia_system_east

hemaia_system_vivado_gui: # In ESAT Server
	sh -c "cd ./target/fpga_chip/hemaia_system/hemaia_system/;vivado hemaia_system.xpr"

hemaia_system_east_vivado_gui: # In ESAT Server
	sh -c "cd ./target/fpga_chip/hemaia_system_east/hemaia_system_east/;vivado hemaia_system_east.xpr"

######################
# Verilator Workflow #
######################

hemaia_system_vlt: # In SNAX Docker
	$(MAKE) -C ./target/sim bin/occamy_chip.vlt CFG=$(CFG)

#####################
# Questasim Workflow #
######################

hemaia_system_vsim_preparation: # In SNAX Docker
	$(MAKE) -C ./target/sim work-vsim/compile.vsim.tcl CFG=$(CFG)

hemaia_system_vsim: # In ESAT Server
	$(MAKE) -C ./target/sim bin/occamy_chip.vsim

################
# VCS Workflow #
################

hemaia_system_vcs_preparation: # In SNAX Docker
	$(MAKE) -C ./target/sim work-vcs/compile.sh

hemaia_system_vcs: # In ESAT Server
	$(MAKE) -C ./target/sim bin/occamy_chip.vcs
# How to start the execution of the simulation: cd ./target/sim/bin; ./occamy_chip.vcs -gui -R -fgp=num_threads:8
