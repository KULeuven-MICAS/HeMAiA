# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

# Usage of absolute paths is required to externally include this Makefile
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HW_PARAM_DIR := $(realpath $(MK_DIR))

HW_PARAM_H = $(HW_PARAM_DIR)/versacore_hw_param.h
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)

$(HW_PARAM_H): $(HW_PARAM_DIR)/gen_versacore_hw_param.py $(HW_PARAM_DIR)/snax_kernel_lib.h
# 	generate versacore_hw_param.h
	$< --hwcfg $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_versacore_to_cluster.hjson > $@
# 	cp the latest streamer_csr_addr_map.h
	cp $(SNITCH_ROOT)/target/snitch_cluster/sw/snax/versacore/include/streamer_csr_addr_map.h $(MK_DIR)/../../../runtime/include//snax/versacore/streamer_csr_addr_map.h

.PHONY: clean-data clean

clean-data:
	rm -f $(HW_PARAM_H)

clean: clean-data

# python3 gen_versacore_hw_param.py --hwcfg ../../../../../../../.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/target/snitch_cluster/cfg/snax_versacore_to_cluster.hjson > versacore_hw_param.h
