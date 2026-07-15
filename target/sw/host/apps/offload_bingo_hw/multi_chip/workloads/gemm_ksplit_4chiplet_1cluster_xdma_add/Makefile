# K-split GEMM across 4 chiplets + chiplet-0 xDMA reduction + dequantize
MK_DIR   := $(realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST)))))
HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = multi_chip
WORKLOAD      = gemm_ksplit_4chiplet_1cluster_xdma_add
DEV_APP       = snax-bingo-offload
SRCS = ../../src/offload_bingo_hw.c
BINGO_HOST ?= 1
INCL_DEVICE_BINARY ?= 1

DATA_CFG ?= $(MK_DIR)/params.hjson
DATA_H = $(MK_DIR)/ksplit_gemm_data.h
OFFLOAD_H = $(MK_DIR)/offload_bingo_hw.h
COMMON_DATAGEN = $(MK_DIR)/../../../../../../../../util/sim/gemm/gemm_sim_utils.py
BENDER = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
PLATFORM_H = $(MK_DIR)/../../../../../../../../target/sw/shared/platform/generated/occamy.h

$(DATA_H) $(OFFLOAD_H): $(MK_DIR)/main_bingo.py $(MK_DIR)/ksplit_gemm_multi_chiplet_datagen.py $(COMMON_DATAGEN) $(DATA_CFG) $(PLATFORM_H)
	python3 $(MK_DIR)/main_bingo.py \
		--output_dir $(MK_DIR) \
		--data_h $(DATA_H) \
		-c $(DATA_CFG) \
		--hwcfg $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_versacore_to_cluster.hjson \
		--platformcfg $(PLATFORM_H)

INCDIRS += $(MK_DIR)
PARTIAL_OUTPUTS += $(DATA_H)
PARTIAL_OUTPUTS += $(OFFLOAD_H)

.PHONY: clean-data clean-offload clean gen-data gen-offload
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
