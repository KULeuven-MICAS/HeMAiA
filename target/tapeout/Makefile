

##########################
# Default configurations #
##########################

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
ROOT        := $(MKFILE_DIR)../..
BENDER      := bender
SNAX_ROOT   := $(shell $(BENDER) path snitch_cluster)
TAPEOUT_TCL_DIR := $(MKFILE_DIR)/tcl

HEMAIA_UTIL ?= $(ROOT)/util/hemaia/util.py

TARGET_RTL  ?= $(ROOT)/target/rtl

CFG = $(TARGET_RTL)/cfg/lru.hjson

#########
# Clean #
#########

.PHONY: clean clean_tcl

clean: clean_tcl

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

.PHONY: gen-syn clean-syn
SYN_BUILDDIR := work-syn
SYN_FLIST := syn_flist.tcl
SYN_BENDER = -t rtl
SYN_BENDER += -t synthesis
SYN_BENDER += -t occamy
SYN_BENDER += -t tech_cells_generic_exclude_tc_sram
SYN_BENDER += -t hemaia
SYN_BENDER += -t tsmc_pad
SYN_BENDER += $(shell cat ../rtl/src/bender_targets.tmp)
SYN_ROOT = /users/micas/shares/project_snax/projects/hemaia/HeMAiA


gen-syn:
	mkdir -p $(SYN_BUILDDIR)
	${BENDER} script synopsys ${SYN_BENDER} > $(SYN_BUILDDIR)/$(SYN_FLIST)
	sed -i '2c\set ROOT "$(SYN_ROOT)"' $(SYN_BUILDDIR)/$(SYN_FLIST)
	chmod +x $(SYN_BUILDDIR)/$(SYN_FLIST)
clean-syn:
	rm -rf $(SYN_BUILDDIR)

########################
# Extract the Mem Info #
########################
MEMINFO_UTIL ?= $(ROOT)/util/meminfo/meminfo.py
MEMINFO_DIR ?= meminfo
# The transcript file is expected to be passed as an argument to meminfo.py
# One can manually set the tc_sram.sv and override the "PrintSimCfg" parameter to 
# generate the transcript with the mem info
# The expected meminfo are like
#################################################################################
# tc_sram functional instantiated with the configuration:
# Instance: testharness.i_occamy_0_0.i_occamy.i_occamy_soc.i_occamy_quadrant_s1_0.i_occamy_cluster_3.i_cluster.gen_hive[0].i_snitch_hive.i_snitch_icache.i_lookup.i_snitch_icache_tag.g_cache_tag_sets[0].i_tag.i_tc_sram.p_sim_hello
# Number of ports   (dec): 1
# Number of words   (dec): 128
# Address width     (dec): 7
# Data width        (dec): 39
# Byte width        (dec): 8
# Byte enable width (dec): 5
# Latency Cycles    (dec): 1
# Simulation init   (str): none
#################################################################################
TRANSCRIPT ?= transcript

.PHONY: gen-meminfo debug-meminfo clean-meminfo
clean-meminfo:
	rm -rf $(MEMINFO_DIR)
$(MEMINFO_DIR):
	mkdir -p $@
gen-meminfo: ${TRANSCRIPT} | $(MEMINFO_DIR)
	@echo "Generating meminfo from transcript: ${TRANSCRIPT}"
	@python3 ${MEMINFO_UTIL} --sim_mem_info --transcript ${TRANSCRIPT} --outfile $(MEMINFO_DIR)/meminfo.txt
debug-meminfo: ${TRANSCRIPT} | $(MEMINFO_DIR)
	@echo "Generating debug meminfo from transcript: ${TRANSCRIPT}"
	@python3 ${MEMINFO_UTIL} --debug_sim_mem_info --transcript ${TRANSCRIPT} --outfile $(MEMINFO_DIR)/debug_meminfo.txt 