.PHONY: clean bootrom sw rtl occamy_ip_vcu128 occamy_ip_vcu128_gui occamy_system_vcu128 \
		occamy_system_vcu128_gui occamy_system_download_sw open_terminal hemaia_system_vivado_preparation \
		hemaia_chip_vcu128 hemaia_chip_vcu128_gui hemaia_system_vcu128 hemaia_system_vcu128_gui \
		occamy_system_vlt occamy_system_vsim_preparation occamy_system_vsim hemaia_system_vsim_preparation \
		hemaia_system_vsim

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(dir $(MKFILE_PATH))

CFG_OVERRIDE ?= target/rtl/cfg/hemaia.hjson
CFG = $(realpath $(CFG_OVERRIDE))

clean:
	$(MAKE) -C ./target/fpga/ clean
	$(MAKE) -C ./target/fpga/vivado_ips/ clean
	$(MAKE) -C ./target/fpga_chip/hemaia_chip/ clean
	$(MAKE) -C ./target/fpga_chip/hemaia_system/ clean
	$(MAKE) -C ./target/sim/ clean
	$(MAKE) -C ./target/sim_chip/ clean
	$(MAKE) -C ./target/rtl/ clean
	$(MAKE) -C ./target/fpga/sw clean
	$(MAKE) -C ./target/tapeout clean
	rm -rf Bender.lock .bender deps
	rm -rf ./target/rtl/src/bender_targets.tmp

# Software Generation
bootrom: # In SNAX Docker
# The bootrom used for simulation (light-weight bootrom)
	$(MAKE) -C ./target/sim bootrom CFG_OVERRIDE=$(CFG)

# The bootrom used for tapeout / FPGA prototyping (embedded real rom, full-functional bootrom with different frequency settings)
	$(MAKE) -C ./target/rtl/bootrom bootrom CFG_OVERRIDE=$(CFG)

sw: # In SNAX Docker
	+$(MAKE) -C ./target/sim sw CFG_OVERRIDE=$(CFG)

# The software from simulation and FPGA prototyping comes from one source. 

# Hardware Generation
rtl: # In SNAX Docker
	$(MAKE) -C ./target/rtl/ rtl CFG_OVERRIDE=$(CFG)

####################
# Tapeout Workflow #
####################

.PHONY: tapeout_syn_flist tapeout_preparation

tapeout_preparation: rtl tapeout_syn_flist

# Generating filelist per cluster
# Needed for a per-cluster synthesis
tapeout_syn_flist:
	$(MAKE) -C ./target/tapeout/ syn_gen_list CFG_OVERRIDE=$(CFG)


#################
# FPGA Workflow #
#################

occamy_system_vivado_preparation: # In SNAX Docker
	$(MAKE) -C ./target/fpga/ define_defines_includes_no_simset.tcl
	$(MAKE) -C ./target/fpga/vivado_ips/ define-sources.tcl

occamy_ip_vcu128:	# In ESAT Server
	#                                                                                          debug  jtag  (put 1 or 0)
	sh -c "cd ./target/fpga/vivado_ips/;vivado -mode batch -source occamy_xilinx.tcl -tclargs      1     1"

occamy_ip_vcu128_gui: # In ESAT Server
	sh -c "cd ./target/fpga/vivado_ips/occamy_xilinx/;vivado occamy_xilinx.xpr"

occamy_system_vcu128: occamy_ip_vcu128 # In ESAT Server
	#                                                                                          debug  jtag  (put 1 or 0)
	sh -c "cd ./target/fpga;vivado -mode batch -source occamy_vcu128_2023.tcl -tclargs             1     1"

occamy_system_vcu128_gui: # In ESAT Server
	sh -c "cd ./target/fpga/occamy_vcu128_2023/;vivado occamy_vcu128_2023.xpr"

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

hemaia_chip_vivado_gui: # In ESAT Server
	sh -c "cd ./target/fpga/fpga_chip/hemaia_chip/hemaia_chip/;vivado hemaia_chip.xpr"

hemaia_system_vivado: hemaia_chip_vivado # In ESAT Server
	$(MAKE) -C ./target/fpga_chip/hemaia_system hemaia_system

hemaia_system_vivado_gui: # In ESAT Server
	sh -c "cd ./target/fpga_chip/hemaia_system/hemaia_system/;vivado hemaia_system.xpr"

# Verilator Workflow (not working, many errors comes from AXI)
occamy_system_vlt: # In SNAX Docker
	
	+$(MAKE) -C ./target/sim work/lib/libfesvr.a
	+$(MAKE) -C ./target/sim tb
	+$(MAKE) -C ./target/sim bin/occamy_top.vlt

# Questasim Workflow
occamy_system_vsim_preparation: # In SNAX Docker
	$(MAKE) -C ./target/sim work/lib/libfesvr.a
	$(MAKE) -C ./target/sim tb
	$(MAKE) -C ./target/sim work-vsim/compile.vsim.tcl

occamy_system_vsim: # In ESAT Server
	$(MAKE) -C ./target/sim bin/occamy_top.vsim

hemaia_system_vsim_preparation: # In SNAX Docker
	$(MAKE) -C ./target/sim_chip work-vsim/compile.vsim.tcl CFG_OVERRIDE=$(CFG)

hemaia_system_vsim: # In ESAT Server
	$(MAKE) -C ./target/sim_chip bin/occamy_chip.vsim
