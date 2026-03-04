# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>
# Fanchen Kong <fanchen.kong@kuleuven.be>
# Usage of absolute paths is required to externally include this Makefile
MK_DIR   := $(realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST)))))
HOST_APP_TYPE = offload_bingo_sw
CHIP_TYPE     = single_chip
WORKLOAD      = gemm_tiled
DEV_APP       = snax-bingo-offload

SRCS = ../../src/offload_bingo_sw.c
BINGO_HOST ?= 1 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 1 # True to include the device binary

# DATA
DATA_CFG ?= $(MK_DIR)/params.hjson

DATA_H = $(MK_DIR)/gemm_data.h

BENDER   = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)

$(DATA_H): $(MK_DIR)/gemm_datagen.py $(DATA_CFG) 
	$< -c $(DATA_CFG) --hwcfg $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_versacore_to_cluster.hjson > $@

# Offload
GEN_WORKLOAD = $(MK_DIR)/offload_bingo_sw.h
$(GEN_WORKLOAD): $(MK_DIR)/tiled_gemm_workload_gen.py $(DATA_CFG) 
	$< -c $(DATA_CFG) --hwcfg $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_versacore_to_cluster.hjson > $@
	
###############
# Build settings
############### 
INCDIRS += $(MK_DIR)
# Add the DATA_H to partial outputs
PARTIAL_OUTPUTS += $(DATA_H)
# Add the GEN_WORKLOAD to partial outputs
PARTIAL_OUTPUTS += $(GEN_WORKLOAD)

.PHONY: clean-data clean-offload clean

clean-data:
	rm -f $(DATA_H)

clean-offload:
	rm -f $(GEN_WORKLOAD)

clean: clean-data clean-offload

gen-offload: $(GEN_WORKLOAD)
include ../../../../common.mk
