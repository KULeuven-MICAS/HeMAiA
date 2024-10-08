

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

