# Reference attention workload — hand-crafted modular template for a future
# ONNX → main_bingo.py emitter. Demonstrates the 3 core patterns:
# quantize, K-split GEMM, and softmax.
#
# Build modes (via make MODE=debug|run):
#   MODE=debug : insert check_result after every intermediate output
#   MODE=run   : only the final output is verified

MK_DIR        := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HOST_APP_TYPE  = offload_bingo_hw
CHIP_TYPE      = single_chip
WORKLOAD      = attn_reference_4cluster
DEV_APP        = snax-bingo-offload
SRCS           = ../../src/offload_bingo_hw.c
BINGO_HOST    ?= 1
INCL_DEVICE_BINARY ?= 1

DATA_CFG  ?= $(MK_DIR)/params.hjson
DATA_H     = $(MK_DIR)/attention_datagen_data.h
OFFLOAD_H  = $(MK_DIR)/offload_bingo_hw.h
BENDER     = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
PLATFORM_H = $(MK_DIR)/../../../../../../../../target/sw/shared/platform/generated/occamy.h

MODE ?= debug

# Resolve hardware geometry from the active SoC configuration.
CLUSTER_CFG_CONSUMER := $(abspath $(lastword $(MAKEFILE_LIST)))
include $(abspath $(dir $(CLUSTER_CFG_CONSUMER))/../../../../../../../../util/cluster_cfg.mk)
.SECONDEXPANSION:

$(DATA_H) $(OFFLOAD_H) &: $(MK_DIR)/main_bingo.py $(DATA_CFG) $(PLATFORM_H) $(CLUSTER_CFG_CONSUMER) $$(CLUSTER_CFG_DEPS)
	python3 $(MK_DIR)/main_bingo.py \
		--output_dir $(MK_DIR) \
		--data_h $(DATA_H) \
		-c $(DATA_CFG) \
		--hwcfg $(GEMM_HWCFG) \
		--platformcfg $(PLATFORM_H) \
		--debug-mode $(MODE)

INCDIRS += $(MK_DIR)
PARTIAL_OUTPUTS += $(DATA_H)
PARTIAL_OUTPUTS += $(OFFLOAD_H)

.PHONY: clean-data clean-offload clean

clean-data:
	rm -f $(DATA_H) $(CLUSTER_CFG_STAMP)

clean-offload:
	rm -rf $(OFFLOAD_H)
	rm -rf *.png
	rm -rf *.csv
clean: clean-data clean-offload

gen-data: $(DATA_H)

gen-offload: $(OFFLOAD_H)

include ../../../../common.mk
