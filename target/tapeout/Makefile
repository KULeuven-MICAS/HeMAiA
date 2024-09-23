

##########################
# Default configurations #
##########################

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
ROOT        := $(MKFILE_DIR)../..
BENDER      := bender
SNAX_ROOT   := $(shell $(BENDER) path snitch_cluster)
TAPEOUT_TCL_DIR := $(MKFILE_DIR)/tcl
OCCAMYGEN   = $(ROOT)/util/occamygen/occamygen.py
HEMAIA_UTIL ?= $(ROOT)/util/hemaia/util.py

TARGET_RTL  ?= $(ROOT)/target/rtl
SNITCH_TEST_DIR      ?= $(SNAX_ROOT)/target/common/test
TARGET_TEST_DIR      ?= $(ROOT)/target/rtl/test
include ../../deps/snitch_cluster/target/common/common.mk
BIN_DIR              ?= bin
FESVR                = ../sim/work
#######################
# Config prerequisite #
#######################

# If the configuration file is overriden on the command-line (through
# CFG_OVERRIDE) and this file differs from the least recently used
# (LRU) config, all targets depending on the configuration file have
# to be rebuilt. This file is used to express this condition as a
# prerequisite for other rules.
CFG = $(TARGET_RTL)/cfg/occamy_cfg/lru.hjson

$(CFG): FORCE
	@# If the LRU config file doesn't exist, we use the default config.
	@if [ ! -e $@ ] ; then \
		DEFAULT_CFG="$(TARGET_RTL)/cfg/occamy_cfg/hemaia.hjson"; \
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

#########
# Clean #
#########

.PHONY: clean clean_tcl clean-prep-syn clean-prep-syn-test clean-syn clean-logs

clean: clean_tcl clean-prep-syn clean-prep-syn-test clean-syn clean-logs

clean_tcl:
	rm -f $(TAPEOUT_TCL_DIR)/*

########################
# Generating Filelists #
########################

syn_gen_list: $(CFG)
	@$(HEMAIA_UTIL) --cfg_path $(CFG) --snax-path $(SNAX_ROOT) \
		--outdir ${TAPEOUT_TCL_DIR} --cluster-flist

debug_info:
	@echo "SNAX ROOT: ${SNAX_ROOT}"
	@echo "CFG override: ${CFG_OVERRIDE}"
	@echo "Makefile path: ${MKFILE_PATH}"
	@echo "Makefile dir: ${MKFILE_DIR}"


VSIM_FLAGS += -t 1ps
VSIM_FLAGS += -voptargs=+acc
VSIM_FLAGS += -do "log -r /*; run -a"
VSIM_FLAGS += -warning 8386
############
# prep_syn #
############
# In this phase we want to print the mem info using tc_sram
# and get rid of all the fesvr managed mem (bootrom and dram)
# by using the real rom & sram
# We still need the uart to print
PREP_SYN_BUILDDIR := work-prep-syn
PREP_SYN_BENDER = -t rtl
PREP_SYN_BENDER += -t test
PREP_SYN_BENDER += -t simulation
PREP_SYN_BENDER += -t vsim
PREP_SYN_BENDER += -t print_mem_info
PREP_SYN_BENDER += -t prep_syn
PREP_SYN_BENDER += -t tech_cells_generic_exclude_tc_sram
PREP_SYN_BENDER += $(shell cat ../rtl/src/bender_targets.tmp)

TB_CC_SOURCES += $(SNITCH_TEST_DIR)/ipc.cc
TB_CC_SOURCES += $(SNITCH_TEST_DIR)/rtl_lib.cc
TB_CC_SOURCES += $(SNITCH_TEST_DIR)/common_lib.cc
TB_CC_SOURCES += $(TARGET_TEST_DIR)/uartdpi/uartdpi.c

TB_CC_FLAGS  = -std=c++17
TB_CC_FLAGS +=-I$(FESVR)/include
TB_CC_FLAGS +=-I$(SNITCH_TEST_DIR)
TB_CC_FLAGS +=-I$(TARGET_TEST_DIR)/uartdpi

.PHONY: prep-syn

prep-syn: tb-prep-syn $(PREP_SYN_BUILDDIR)/compile.vsim.tcl
tb-prep-syn: 
	@$(OCCAMYGEN) --cfg $(CFG) --outdir ../sim/test --testharness-sv ../sim/test/testharness.sv.tpl --testharness-sv-flag prep_syn 

$(PREP_SYN_BUILDDIR):
	mkdir -p $@
$(PREP_SYN_BUILDDIR)/compile.vsim.tcl: $(PREP_SYN_BUILDDIR)
	$(BENDER) script vsim $(PREP_SYN_BENDER) --vlog-arg="$(VLOG_FLAGS) -work $(dir $@) " > $@
	echo '$(VLOG) -work $(dir $@) $(TB_CC_SOURCES) -vv -ccflags "$(TB_CC_FLAGS)"' >> $@
	echo 'return 0' >> $@
$(BIN_DIR):
	mkdir -p $@
$(BIN_DIR)/occamy_top_prep_syn.vsim: $(PREP_SYN_BUILDDIR)/compile.vsim.tcl |$(BIN_DIR)
	touch $@
	vsim -c -do "source $<; quit" | tee $(dir $<)vlog.log
	@! grep -P "Errors: [1-9]*," $(dir $<)vlog.log
	vopt $(VOPT_FLAGS) -work $(PREP_SYN_BUILDDIR) tb_bin -o tb_bin_opt | tee $(dir $<)vopt.log
	@! grep -P "Errors: [1-9]*," $(dir $<)vopt.log
	@echo "#!/bin/bash" > $@
	@echo 'binary=$$(realpath $$1)' >> $@
	@echo 'echo $$binary > .rtlbinary' >> $@
	@echo 'vsim +permissive $(VSIM_FLAGS) $$3 -work $(MKFILE_DIR)/$(PREP_SYN_BUILDDIR) -c \
				-ldflags "-Wl,-rpath,$(FESVR)/lib -L$(FESVR)/lib -lfesvr -lutil" \
				tb_bin_opt +permissive-off ++$$binary ++$$2' >> $@
	@# Rename CVA6 trace to align with Snitch trace names
	@echo "mv ${CVA6_TRACE} $(LOGS_DIR)/trace_hart_00000.txt" >> $@
	@# Move all other traces to logs folder as well
	@echo "mv *trace*.log $(LOGS_DIR)/" >> $@
	@echo "mv uart*.log $(LOGS_DIR)/" >> $@
	@echo "mv transcript $(LOGS_DIR)/" >> $@
	@echo "mv uart*.log $(LOGS_DIR)/" >> $@
	@echo "mv .rtlbinary $(LOGS_DIR)/" >> $@
	@chmod +x $@
clean-prep-syn:
	rm -rf $(PREP_SYN_BUILDDIR)
	rm -rf $(BIN_DIR)/occamy_top_prep_syn.vsim
#################
# prep_syn_test #
#################
PREP_SYN_TEST_BUILDDIR := work-prep-syn-test
PREP_SYN_TEST_BENDER = -t rtl
PREP_SYN_TEST_BENDER += -t test
PREP_SYN_TEST_BENDER += -t simulation
PREP_SYN_TEST_BENDER += -t vsim
PREP_SYN_TEST_BENDER += -t prep_syn_test
PREP_SYN_TEST_BENDER += -t tech_cells_generic_exclude_tc_sram
PREP_SYN_TEST_BENDER += $(shell cat ../rtl/src/bender_targets.tmp)

prep-syn-test: $(PREP_SYN_TEST_BUILDDIR)/compile.vsim.tcl

$(PREP_SYN_TEST_BUILDDIR):
	mkdir -p $@
$(PREP_SYN_TEST_BUILDDIR)/compile.vsim.tcl: $(PREP_SYN_TEST_BUILDDIR)
	$(BENDER) script vsim $(PREP_SYN_TEST_BENDER) --vlog-arg="$(VLOG_FLAGS) -work $(dir $@) " > $@
	echo '$(VLOG) -work $(dir $@) $(TB_CC_SOURCES) -vv -ccflags "$(TB_CC_FLAGS)"' >> $@
	echo 'return 0' >> $@
$(BIN_DIR):
	mkdir -p $@
$(BIN_DIR)/occamy_top_prep_syn_test.vsim: $(PREP_SYN_TEST_BUILDDIR)/compile.vsim.tcl |$(BIN_DIR)
	touch $@
	vsim -c -do "source $<; quit" | tee $(dir $<)vlog.log
	@! grep -P "Errors: [1-9]*," $(dir $<)vlog.log
	vopt $(VOPT_FLAGS) -work $(PREP_SYN_TSET_BUILDDIR) tb_bin -o tb_bin_opt | tee $(dir $<)vopt.log
	@! grep -P "Errors: [1-9]*," $(dir $<)vopt.log
	@echo "#!/bin/bash" > $@
	@echo 'binary=$$(realpath $$1)' >> $@
	@echo 'echo $$binary > .rtlbinary' >> $@
	@echo 'vsim +permissive $(VSIM_FLAGS) $$3 -work $(MKFILE_DIR)/$(PREP_SYN_TEST_BUILDDIR) -c \
				-ldflags "-Wl,-rpath,$(FESVR)/lib -L$(FESVR)/lib -lfesvr -lutil" \
				tb_bin_opt +permissive-off ++$$binary ++$$2' >> $@
	@# Rename CVA6 trace to align with Snitch trace names
	@echo "mv ${CVA6_TRACE} $(LOGS_DIR)/trace_hart_00000.txt" >> $@
	@# Move all other traces to logs folder as well
	@echo "mv *trace*.log $(LOGS_DIR)/" >> $@
	@echo "mv uart*.log $(LOGS_DIR)/" >> $@
	@echo "mv transcript $(LOGS_DIR)/" >> $@
	@echo "mv uart*.log $(LOGS_DIR)/" >> $@
	@echo "mv .rtlbinary $(LOGS_DIR)/" >> $@
	@chmod +x $@
clean-prep-syn-test:
	rm -rf $(PREP_SYN_TEST_BUILDDIR)
	rm -rf $(BIN_DIR)/occamy_top_prep_syn_test.vsim
#######
# syn #
#######
.PHONY: gen-syn clean-syn
SYN_BUILDDIR := work-syn
SYN_BENDER = -t rtl
SYN_BENDER += -t synthesis
SYN_BENDER += -t occamy
SYN_BENDER += -t tech_cells_generic_exclude_tc_sram
SYN_BENDER += -t syn
SYN_BENDER += $(shell cat ../rtl/src/bender_targets.tmp)
SYN_ROOT = /users/micas/shares/project_snax/projects/hemaia/HeMAiA


gen-syn:
	mkdir -p $(SYN_BUILDDIR)
	${BENDER} script synopsys ${SYN_BENDER} > $(SYN_BUILDDIR)/$(SYN_FLIST)
	sed -i '2c\set ROOT "$(SYN_ROOT)"' $(SYN_BUILDDIR)/$(SYN_FLIST)
	chmod +x $(SYN_BUILDDIR)/$(SYN_FLIST)
clean-syn:
	rm -rf $(SYN_BUILDDIR)

clean-logs:
	rm -rf $(LOGS_DIR)