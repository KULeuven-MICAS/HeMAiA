# Attention test: FP32 softmax kernel in isolation
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = single_chip
WORKLOAD      = attn_test_softmax
DEV_APP       = snax-bingo-offload
SRCS = ../../src/offload_bingo_hw.c
BINGO_HOST ?= 1
INCL_DEVICE_BINARY ?= 1

DATA_CFG ?= $(MK_DIR)/params.hjson
DATA_H = $(MK_DIR)/softmax_data.h
OFFLOAD_H = $(MK_DIR)/offload_bingo_hw.h
BENDER   = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
PLATFORM_H = $(MK_DIR)/../../../../../../../../target/sw/shared/platform/generated/occamy.h
RTL_CFG    = $(MK_DIR)/../../../../../../../../target/rtl/cfg/lru.hjson

$(DATA_H) $(OFFLOAD_H): $(MK_DIR)/main_bingo.py $(DATA_CFG)
	python3 $(MK_DIR)/main_bingo.py \
		--output_dir $(MK_DIR) \
		--data_h $(DATA_H) \
		-c $(DATA_CFG) \
		--hwcfg $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_versacore_to_cluster.hjson \
		--platformcfg $(PLATFORM_H) \
		--rtlcfg $(RTL_CFG)

INCDIRS += $(MK_DIR)
PARTIAL_OUTPUTS += $(DATA_H)
PARTIAL_OUTPUTS += $(OFFLOAD_H)

.PHONY: clean-data clean-offload clean
clean-data:
	rm -f $(DATA_H)
clean-offload:
	rm -rf $(OFFLOAD_H)
	rm -rf *.png
	rm -rf *.csv
clean: clean-data clean-offload
gen-data: $(DATA_H)
gen-offload: $(OFFLOAD_H)

include ../../../../common.mk
