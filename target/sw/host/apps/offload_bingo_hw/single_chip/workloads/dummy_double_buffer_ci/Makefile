# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>
# Fanchen Kong <fanchen.kong@kuleuven.be>
# Usage of absolute paths is required to externally include this Makefile
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = single_chip
WORKLOAD      = dummy_double_buffer_ci
DEV_APP       = snax-bingo-offload
SRCS = ../../src/offload_bingo_hw.c
BINGO_HOST ?= 1 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 1 # True to include the device binary


# Offload
OFFLOAD_H = offload_bingo_hw.h	
$(OFFLOAD_H): $(MK_DIR)/main_bingo.py
	python3 $< --output_dir $(MK_DIR) --output_offload_file_name $(OFFLOAD_H)


###############
# Build settings
############### 
INCDIRS += $(MK_DIR)

# Add the OFFLOAD_H to partial outputs
PARTIAL_OUTPUTS += $(OFFLOAD_H)

.PHONY: clean-offload clean


clean-offload:
	rm -rf $(OFFLOAD_H)
	rm -rf *.png
	rm -rf *.csv
clean: clean-offload


gen-offload: $(OFFLOAD_H)

include ../../../../common.mk